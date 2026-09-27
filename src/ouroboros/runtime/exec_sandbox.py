"""One execution sandbox for the commands the controller itself runs.

The controller runs some commands on its own authority, not through an agent:
legacy-verifier replay (``orchestrator/evidence/command_replay.py``) re-runs a
transcript command in a copy of the workspace, and a check package runs
model-written checks on a copy of the checkout. Each copy is a working
directory, not a boundary: a script run there can still write any absolute
path the user can write. This module is the boundary, shared by every such
caller.

``confine`` turns an argv into the argv, environment and working directory
that run it confined, or into ``SandboxUnavailable`` (a typed reason; the
caller records the outcome as indeterminate and runs nothing). The caller
spawns the confined argv with its own process runner and timeout, because
callers differ in how they talk to the process (replay collects its output;
a check package exchanges frames over stdin and stdout). Under confinement:

- **Writes** are allowed only beneath the writable roots the caller names
  (the copy) and the per-run temp directory, plus a few character devices
  (``/dev/null`` and friends). Everything else, including the live
  workspace, the user's home directory and the system temp directory, is
  read-only. Reading and executing are not restricted.
- **Network** (``deny_network=True``): IP traffic is denied. Unix-domain
  sockets stay available for local IPC.
- **Other processes' environments** (``ConfinedCommand.isolates_process_environments``):
  under Landlock a confined process cannot read ``/proc/<pid>/environ``,
  ``mem`` or ``maps`` of any process outside its domain, the controller and
  every other process of the user included, because Landlock denies
  ptrace-mode access across the domain boundary; its own ``/proc/self`` and
  its descendants' stay readable. ``sandbox-exec`` cannot deny the macOS
  equivalent (``sysctl`` ``KERN_PROCARGS2``, which returns a same-user
  process's arguments and environment; neither ``sysctl-read`` nor
  ``process-info*`` rules gate it), so on macOS a confined process can read
  the environment of the controller and of the user's other processes. A
  caller that must keep those secrets from the command requires
  ``isolates_process_environments``.
- **Environment**: built from scratch. Only the variables named in
  ``env_passthrough`` (``DEFAULT_ENV_PASSTHROUGH`` by default: ``PATH``, the
  locale, ``TZ`` and Python I/O settings) are copied from the source
  environment; ``TMPDIR``, ``TMP`` and ``TEMP`` point to the temp directory,
  and so does ``HOME`` unless the caller passes it through; ``env_set``
  values are applied last.

Backends:

- **macOS**: ``sandbox-exec`` with a generated profile: everything allowed,
  then ``file-write*`` denied except beneath the writable roots (passed as
  profile parameters, so no path is ever spliced into the profile text) and
  the devices; network denial allows only loopback IP.
- **Linux**: Landlock, applied by ``_landlock_exec.py`` in the child before it
  execs the command. It is unprivileged and needs no mount or user
  namespace, so it works in containers. Network denial uses an unprivileged
  network namespace (``unshare --user --map-root-user --net``), or nothing
  extra when this process's network namespace already has only loopback (a
  container started with ``--network none``).
- **Anything else** (Windows, a Linux kernel without Landlock, a macOS
  process that is already sandboxed): no backend, and ``confine`` returns
  ``SandboxUnavailable``. A command is never run unconfined as a fallback.

Each backend is probed once per process by running a small Python program
under it that must be able to write inside a writable root and must fail to
write next to it.

Unsafe off switch: ``OUROBOROS_EXEC_SANDBOX=off`` in the environment, or
``execution.exec_sandbox: false`` in ``~/.ouroboros/config.yaml``, makes
``confine`` return the argv unchanged with ``backend=DISABLED`` and
``network_denied=False`` (the environment is still built the same way). The
sandbox is on by default, and a project ``.env`` cannot turn it off
(``config/untrusted_env.py``).

Outside this boundary: effects the confined process asks another, unconfined
process to perform over IPC (a user service manager, a desktop automation
service, a container daemon), and reads of anything the user can read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import functools
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

import structlog

log = structlog.get_logger(__name__)

EXEC_SANDBOX_ENV_VAR = "OUROBOROS_EXEC_SANDBOX"

# Copied from the source environment by default: what any program needs to
# find executables and decode text, and nothing that names a credential,
# a configuration file or an import path.
DEFAULT_ENV_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_COLLATE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "TZ",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
)
TEMP_DIRECTORY_VARIABLES: tuple[str, ...] = ("TMPDIR", "TMP", "TEMP")

_PROBE_TIMEOUT_SECONDS = 10.0
_LANDLOCK_HELPER = Path(__file__).with_name("_landlock_exec.py")

# macOS: profile parameters ``W0``..``Wn`` carry the writable roots.
_DARWIN_DEVICE_RULES = (
    '(allow file-write* (literal "/dev/null") (literal "/dev/zero")'
    ' (literal "/dev/random") (literal "/dev/urandom") (literal "/dev/tty")'
    ' (literal "/dev/dtracehelper") (subpath "/dev/fd"))'
)
_DARWIN_NETWORK_RULES = (
    "(deny network-outbound (remote ip))"
    '(allow network-outbound (remote ip "localhost:*"))'
    "(deny network-inbound (local ip))"
    '(allow network-inbound (local ip "localhost:*"))'
)


class SandboxBackend(StrEnum):
    """The mechanism confining a command."""

    SANDBOX_EXEC = "sandbox_exec"
    LANDLOCK = "landlock"
    DISABLED = "disabled"
    """The unsafe off switch is set: the command runs unconfined."""


class SandboxUnavailableReason(StrEnum):
    """Why a command cannot be run confined (its outcome is indeterminate)."""

    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    """No filesystem confinement backend works on this host."""
    NETWORK_ISOLATION_UNAVAILABLE = "network_isolation_unavailable"
    """Network denial was requested and no mechanism for it works here."""
    INVALID_WRITABLE_ROOT = "invalid_writable_root"
    """A writable root or the temp directory is not an existing directory."""


@dataclass(frozen=True, slots=True)
class SandboxUnavailable:
    """The command was not run; record its outcome as indeterminate."""

    reason: SandboxUnavailableReason
    detail: str = ""

    @property
    def outcome(self) -> str:
        return "indeterminate"


@dataclass(frozen=True, slots=True)
class ConfinedCommand:
    """What to spawn: ``argv`` in ``cwd`` with exactly ``env``."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: str
    backend: SandboxBackend
    writable_roots: tuple[str, ...]
    network_denied: bool
    isolates_process_environments: bool
    """Whether the command cannot read other processes' environments (Landlock only)."""


def sandbox_disabled() -> bool:
    """Whether the unsafe off switch is set (environment first, then config)."""
    raw = os.environ.get(EXEC_SANDBOX_ENV_VAR, "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return True
    if raw in ("1", "true", "on", "yes"):
        return False
    return not config_enabled()


def config_enabled() -> bool:
    """``execution.exec_sandbox`` from ``~/.ouroboros/config.yaml``; True unless false.

    A configuration that is missing or cannot be loaded keeps the sandbox on:
    turning it off is an explicit, unsafe choice.
    """
    from ouroboros.config.loader import load_config
    from ouroboros.core.errors import ConfigError

    try:
        return load_config().execution.exec_sandbox is not False
    except (ConfigError, OSError):
        return True


def _darwin_profile(root_count: int, *, deny_network: bool) -> str:
    roots = " ".join(f'(subpath (param "W{index}"))' for index in range(root_count))
    profile = f"(version 1)(allow default)(deny file-write*)(allow file-write* {roots})"
    profile += _DARWIN_DEVICE_RULES
    return profile + (_DARWIN_NETWORK_RULES if deny_network else "")


def _backend_argv(
    backend: SandboxBackend,
    argv: Sequence[str],
    roots: Sequence[str],
    network_prefix: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """The argv running ``argv`` under ``backend``; ``network_prefix`` None allows network."""
    if backend is SandboxBackend.SANDBOX_EXEC:
        executable = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        profile = _darwin_profile(len(roots), deny_network=network_prefix is not None)
        params = [part for index, root in enumerate(roots) for part in ("-D", f"W{index}={root}")]
        return (executable, "-p", profile, *params, "--", *argv)
    if backend is SandboxBackend.LANDLOCK:
        writes = [part for root in roots for part in ("--write", root)]
        helper = (sys.executable, "-I", "-S", "-B", str(_LANDLOCK_HELPER), *writes, "--")
        return (*(network_prefix or ()), *helper, *argv)
    return tuple(argv)


_PROBE_PROGRAM = (
    "import os, sys\n"
    "inside, outside = sys.argv[1], sys.argv[2]\n"
    "with open(os.path.join(inside, 'probe'), 'w') as handle:\n"
    "    handle.write('ok')\n"
    "try:\n"
    "    open(os.path.join(outside, 'probe'), 'w').close()\n"
    "except OSError:\n"
    "    sys.exit(0)\n"
    "sys.exit(3)\n"
)


@functools.cache
def filesystem_backend() -> SandboxBackend | None:
    """The backend that confines writes on this host, or None; probed once."""
    if sys.platform == "darwin":
        candidate = SandboxBackend.SANDBOX_EXEC
    elif sys.platform.startswith("linux"):
        candidate = SandboxBackend.LANDLOCK
    else:
        return None
    probe_root = Path(tempfile.mkdtemp(prefix="ouroboros-sandbox-probe-")).resolve()
    try:
        inside = probe_root / "inside"
        outside = probe_root / "outside"
        inside.mkdir()
        outside.mkdir()
        argv = _backend_argv(
            candidate,
            (sys.executable, "-I", "-S", "-c", _PROBE_PROGRAM, str(inside), str(outside)),
            (str(inside),),
            None,
        )
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                argv, capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        confined = (
            result.returncode == 0
            and (inside / "probe").is_file()
            and not (outside / "probe").exists()
        )
        if not confined:
            log.info(
                "exec_sandbox.backend_unavailable",
                backend=candidate.value,
                returncode=result.returncode,
                stderr=result.stderr.decode("utf-8", errors="replace")[-500:],
            )
        return candidate if confined else None
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)


def _process_has_only_loopback() -> bool:
    """Return True when this process's network namespace has only loopback.

    The interfaces come from the kernel for the process's own namespace, so a
    container started with ``--network none`` reports only ``lo``.
    """
    try:
        names = [name for _index, name in socket.if_nameindex()]
    except (OSError, AttributeError):
        return False
    return bool(names) and all(name == "lo" for name in names)


@functools.cache
def network_denial_prefix() -> tuple[str, ...] | None:
    """The argv prefix that denies network access, or None; probed once.

    ``()`` when the denial needs no prefix: on macOS it is part of the
    profile, and a Linux process with only loopback is already offline.
    """
    if sys.platform == "darwin":
        return ()
    if not sys.platform.startswith("linux"):
        return None
    if _process_has_only_loopback():
        return ()
    executable = shutil.which("unshare") or ""
    if not executable:
        return None
    prefix = (executable, "--user", "--map-root-user", "--net", "--")
    probe = shutil.which("true") or "/bin/true"
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [*prefix, probe], capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return prefix if result.returncode == 0 else None


def sandbox_unavailable_reason(*, deny_network: bool = True) -> SandboxUnavailableReason | None:
    """Why ``confine`` would refuse on this host, or None when it would confine.

    None as well when the unsafe off switch is set.
    """
    if sandbox_disabled():
        return None
    if filesystem_backend() is None:
        return SandboxUnavailableReason.SANDBOX_UNAVAILABLE
    if deny_network and network_denial_prefix() is None:
        return SandboxUnavailableReason.NETWORK_ISOLATION_UNAVAILABLE
    return None


def build_environment(
    temp_dir: str,
    *,
    source: Mapping[str, str] | None = None,
    passthrough: Sequence[str] = DEFAULT_ENV_PASSTHROUGH,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The complete environment of a confined command (see the module docstring)."""
    values = os.environ if source is None else source
    env = {name: values[name] for name in passthrough if values.get(name)}
    for name in TEMP_DIRECTORY_VARIABLES:
        env[name] = temp_dir
    env.setdefault("HOME", temp_dir)
    env.update(overrides or {})
    return env


def confine(
    argv: Sequence[str],
    *,
    cwd: str,
    writable_roots: Sequence[str],
    temp_dir: str,
    deny_network: bool = True,
    env_source: Mapping[str, str] | None = None,
    env_passthrough: Sequence[str] = DEFAULT_ENV_PASSTHROUGH,
    env_set: Mapping[str, str] | None = None,
) -> ConfinedCommand | SandboxUnavailable:
    """Return how to run ``argv`` confined, or why it cannot be.

    ``writable_roots`` and ``temp_dir`` must be existing directories; the
    caller creates them and removes them afterwards. ``temp_dir`` is writable
    too. ``argv`` is run directly, never through a shell.
    """
    real_temp = os.path.realpath(temp_dir)
    roots: list[str] = []
    for root in (*writable_roots, temp_dir):
        real = os.path.realpath(root)
        if not os.path.isabs(root) or not os.path.isdir(real):
            return SandboxUnavailable(SandboxUnavailableReason.INVALID_WRITABLE_ROOT, root)
        if real not in roots:
            roots.append(real)
    env = build_environment(
        real_temp, source=env_source, passthrough=env_passthrough, overrides=env_set
    )
    if sandbox_disabled():
        _warn_disabled()
        return ConfinedCommand(
            argv=tuple(argv),
            env=env,
            cwd=cwd,
            backend=SandboxBackend.DISABLED,
            writable_roots=tuple(roots),
            network_denied=False,
            isolates_process_environments=False,
        )
    reason = sandbox_unavailable_reason(deny_network=deny_network)
    if reason is not None:
        return SandboxUnavailable(reason)
    backend = filesystem_backend()
    assert backend is not None  # checked by sandbox_unavailable_reason
    network_prefix = network_denial_prefix() if deny_network else None
    return ConfinedCommand(
        argv=_backend_argv(backend, argv, roots, network_prefix),
        env=env,
        cwd=cwd,
        backend=backend,
        writable_roots=tuple(roots),
        network_denied=deny_network,
        isolates_process_environments=backend is SandboxBackend.LANDLOCK,
    )


@functools.cache
def _warn_disabled() -> None:
    log.warning(
        "exec_sandbox.disabled",
        detail=(
            f"{EXEC_SANDBOX_ENV_VAR}=off or execution.exec_sandbox: false; "
            "controller-run commands are not confined"
        ),
    )


__all__ = [
    "DEFAULT_ENV_PASSTHROUGH",
    "EXEC_SANDBOX_ENV_VAR",
    "TEMP_DIRECTORY_VARIABLES",
    "ConfinedCommand",
    "SandboxBackend",
    "SandboxUnavailable",
    "SandboxUnavailableReason",
    "build_environment",
    "confine",
    "filesystem_backend",
    "network_denial_prefix",
    "sandbox_disabled",
    "sandbox_unavailable_reason",
]
