"""The execution sandbox for controller-run commands (``runtime/exec_sandbox.py``).

Tests that need the real backend of this host (``sandbox-exec`` on macOS,
Landlock on Linux) skip with the probe's reason where it is unavailable; the
refusal path is tested on every host by standing in for the probe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from ouroboros.config.untrusted_env import UNTRUSTED_ENV_DENYLIST
from ouroboros.runtime import _landlock_exec, exec_sandbox
from ouroboros.runtime.exec_sandbox import (
    DEFAULT_ENV_PASSTHROUGH,
    EXEC_SANDBOX_ENV_VAR,
    ConfinedCommand,
    SandboxBackend,
    SandboxUnavailable,
    SandboxUnavailableReason,
    build_environment,
    confine,
)


@pytest.fixture(autouse=True)
def _sandbox_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXEC_SANDBOX_ENV_VAR, "on")


def _require_backend(*, deny_network: bool = False) -> None:
    reason = exec_sandbox.sandbox_unavailable_reason(deny_network=deny_network)
    if reason is not None:
        pytest.skip(f"execution sandbox unavailable on this host: {reason.value}")


def _run(command: ConfinedCommand) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - argv built by confine
        list(command.argv),
        cwd=command.cwd,
        env=dict(command.env),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _python(code: str, *args: str) -> tuple[str, ...]:
    return (sys.executable, "-I", "-c", code, *args)


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / name for name in ("copy", "temp", "outside")}
    for path in paths.values():
        path.mkdir()
    return paths


class TestRealBackend:
    def test_writes_inside_succeed_and_writes_outside_are_blocked(
        self, layout: dict[str, Path]
    ) -> None:
        _require_backend()
        copy, outside = layout["copy"].resolve(), layout["outside"].resolve()
        code = (
            "import os, sys, tempfile\n"
            "open('inside.txt', 'w').write('ok')\n"
            "os.mkdir('made'); os.rename('inside.txt', 'made/moved.txt')\n"
            "fd, name = tempfile.mkstemp(); os.write(fd, b'tmp'); os.close(fd)\n"
            "open(os.devnull, 'w').write('discarded')\n"
            "try:\n"
            "    open(os.path.join(sys.argv[1], 'escaped.txt'), 'w').write('no')\n"
            "except OSError:\n"
            "    sys.exit(0)\n"
            "sys.exit(9)\n"
        )
        command = confine(
            _python(code, str(outside)),
            cwd=str(copy),
            writable_roots=(str(copy),),
            temp_dir=str(layout["temp"]),
            deny_network=False,
        )
        assert isinstance(command, ConfinedCommand)
        assert command.backend in (SandboxBackend.SANDBOX_EXEC, SandboxBackend.LANDLOCK)

        result = _run(command)

        assert result.returncode == 0, result.stderr
        assert (copy / "made" / "moved.txt").read_text(encoding="utf-8") == "ok"
        assert len(list(layout["temp"].iterdir())) == 1
        assert not (outside / "escaped.txt").exists()

    def test_a_shell_script_cannot_write_outside_the_copy(
        self, layout: dict[str, Path], tmp_path: Path
    ) -> None:
        _require_backend()
        copy = layout["copy"].resolve()
        target = layout["outside"].resolve() / "from_script.txt"
        script = copy / "run.sh"
        script.write_text(f"#!/bin/sh\necho x > {target}\n", encoding="utf-8")
        script.chmod(0o755)
        command = confine(
            ("./run.sh",),
            cwd=str(copy),
            writable_roots=(str(copy),),
            temp_dir=str(layout["temp"]),
            deny_network=False,
        )
        assert isinstance(command, ConfinedCommand)

        result = _run(command)

        assert result.returncode != 0
        assert not target.exists()

    def test_writable_root_with_quote_and_backslash_in_its_name(self, tmp_path: Path) -> None:
        _require_backend()
        root = tmp_path / 'we"ird\\dir'
        temp = tmp_path / "temp"
        root.mkdir()
        temp.mkdir()
        command = confine(
            _python("open('f', 'w').write('ok')"),
            cwd=str(root),
            writable_roots=(str(root),),
            temp_dir=str(temp),
            deny_network=False,
        )
        assert isinstance(command, ConfinedCommand)

        result = _run(command)

        assert result.returncode == 0, result.stderr
        assert (root / "f").read_text(encoding="utf-8") == "ok"

    def test_network_is_denied_when_requested(self, layout: dict[str, Path]) -> None:
        _require_backend(deny_network=True)
        code = (
            "import socket, sys\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3).close()\n"
            "except OSError:\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=3).close()
        except OSError:
            pytest.skip("this host has no network to deny")
        command = confine(
            _python(code),
            cwd=str(layout["copy"]),
            writable_roots=(str(layout["copy"]),),
            temp_dir=str(layout["temp"]),
            deny_network=True,
        )
        assert isinstance(command, ConfinedCommand) and command.network_denied

        result = _run(command)

        assert result.returncode == 0, result.stderr

    def test_child_environment_is_the_allowlist(
        self, layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_backend()
        monkeypatch.setenv("OUROBOROS_TEST_SECRET", "leak")
        monkeypatch.setenv("LANG", "C.UTF-8")
        command = confine(
            _python("import json, os; print(json.dumps(dict(os.environ)))"),
            cwd=str(layout["copy"]),
            writable_roots=(str(layout["copy"]),),
            temp_dir=str(layout["temp"]),
            deny_network=False,
            env_set={"EXTRA": "1"},
        )
        assert isinstance(command, ConfinedCommand)

        result = _run(command)

        env = json.loads(result.stdout)
        # ``__CF_USER_TEXT_ENCODING`` is added by macOS to every process.
        env.pop("__CF_USER_TEXT_ENCODING", None)
        temp = os.path.realpath(layout["temp"])
        assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == env["HOME"] == temp
        assert env["LANG"] == "C.UTF-8" and env["EXTRA"] == "1"
        assert "OUROBOROS_TEST_SECRET" not in env
        allowed = {*DEFAULT_ENV_PASSTHROUGH, "TMPDIR", "TMP", "TEMP", "HOME", "EXTRA"}
        assert set(env) <= allowed


class TestOtherProcessEnvironments:
    _READER = (
        "import os, sys\n"
        "leaked = []\n"
        "for part in ('environ', 'mem', 'maps'):\n"
        "    try:\n"
        "        with open(f'/proc/{os.getppid()}/{part}', 'rb') as handle:\n"
        "            leaked.append(b'hunter2-sandbox-secret' in handle.read(1 << 20))\n"
        "    except OSError:\n"
        "        pass\n"
        "with open('/proc/self/environ', 'rb') as handle:\n"
        "    handle.read()\n"
        "sys.exit(3 if any(leaked) else 0)\n"
    )

    @staticmethod
    def _through_parent_holding_a_secret(argv: tuple[str, ...], env: dict[str, str]) -> int:
        """Run ``argv`` as the child of a process whose environment holds the secret.

        ``/proc/<pid>/environ`` shows the environment a process started with,
        so the secret must be in the parent's initial environment.
        """
        launcher = (
            "import json, subprocess, sys\n"
            "argv, env = json.loads(sys.argv[1])\n"
            "sys.exit(subprocess.run(argv, env=env).returncode)\n"
        )
        parent_env = {**env, "OUROBOROS_TEST_PARENT_SECRET": "hunter2-sandbox-secret"}
        return subprocess.run(  # noqa: S603 - fixed argv
            [sys.executable, "-I", "-c", launcher, json.dumps([list(argv), env])],
            env=parent_env,
            timeout=60,
            check=False,
        ).returncode

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc is Linux only")
    def test_confined_child_cannot_read_the_parent_environment(
        self, layout: dict[str, Path]
    ) -> None:
        _require_backend()
        unconfined = self._through_parent_holding_a_secret(
            _python(self._READER), {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        )
        assert unconfined == 3, "a same-user /proc read works unconfined"
        command = confine(
            _python(self._READER),
            cwd=str(layout["copy"]),
            writable_roots=(str(layout["copy"]),),
            temp_dir=str(layout["temp"]),
            deny_network=False,
        )
        assert isinstance(command, ConfinedCommand) and command.isolates_process_environments

        confined = self._through_parent_holding_a_secret(command.argv, dict(command.env))

        assert confined == 0

    @pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
    def test_macos_reports_that_it_cannot_hide_other_environments(
        self, layout: dict[str, Path]
    ) -> None:
        _require_backend()
        command = confine(
            ("true",),
            cwd=str(layout["copy"]),
            writable_roots=(str(layout["copy"]),),
            temp_dir=str(layout["temp"]),
        )

        assert isinstance(command, ConfinedCommand)
        assert command.isolates_process_environments is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux only")
def test_linux_ci_runner_has_landlock() -> None:
    """On GitHub Actions the Linux backend must exist, so its tests really run there."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        pytest.skip("only asserted on the GitHub Actions runner")
    abi = _landlock_exec.landlock_abi()
    assert abi >= 1, f"Landlock unavailable on this runner (kernel {os.uname().release})"
    assert exec_sandbox.filesystem_backend() is SandboxBackend.LANDLOCK


class TestUnavailable:
    def test_no_filesystem_backend_is_indeterminate_and_runs_nothing(
        self, layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(exec_sandbox, "filesystem_backend", lambda: None)

        result = confine(
            ("touch", str(layout["outside"] / "ran")),
            cwd=str(layout["copy"]),
            writable_roots=(str(layout["copy"]),),
            temp_dir=str(layout["temp"]),
            deny_network=False,
        )

        assert isinstance(result, SandboxUnavailable)
        assert result.reason is SandboxUnavailableReason.SANDBOX_UNAVAILABLE
        assert result.outcome == "indeterminate"

    def test_network_denial_unavailable_is_indeterminate_only_when_requested(
        self, layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(exec_sandbox, "filesystem_backend", lambda: SandboxBackend.LANDLOCK)
        monkeypatch.setattr(exec_sandbox, "network_denial_prefix", lambda: None)
        kwargs = {
            "cwd": str(layout["copy"]),
            "writable_roots": (str(layout["copy"]),),
            "temp_dir": str(layout["temp"]),
        }

        denied = confine(("true",), deny_network=True, **kwargs)  # type: ignore[arg-type]
        allowed = confine(("true",), deny_network=False, **kwargs)  # type: ignore[arg-type]

        assert isinstance(denied, SandboxUnavailable)
        assert denied.reason is SandboxUnavailableReason.NETWORK_ISOLATION_UNAVAILABLE
        assert isinstance(allowed, ConfinedCommand) and not allowed.network_denied

    def test_unsupported_platform_has_no_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exec_sandbox.sys, "platform", "win32")

        assert exec_sandbox.filesystem_backend.__wrapped__() is None  # type: ignore[attr-defined]
        assert exec_sandbox.network_denial_prefix.__wrapped__() is None  # type: ignore[attr-defined]

    @pytest.mark.parametrize("root", ["relative/dir", "/nonexistent/ouroboros-sandbox-root"])
    def test_invalid_writable_root(self, tmp_path: Path, root: str) -> None:
        result = confine(
            ("true",), cwd=str(tmp_path), writable_roots=(root,), temp_dir=str(tmp_path)
        )

        assert isinstance(result, SandboxUnavailable)
        assert result.reason is SandboxUnavailableReason.INVALID_WRITABLE_ROOT

    @pytest.mark.parametrize(
        ("interfaces", "offline"),
        [([(1, "lo")], True), ([(1, "lo"), (2, "eth0")], False), ([], False)],
    )
    def test_linux_process_with_only_loopback_needs_no_network_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        interfaces: list[tuple[int, str]],
        offline: bool,
    ) -> None:
        monkeypatch.setattr(exec_sandbox.socket, "if_nameindex", lambda: interfaces)
        monkeypatch.setattr(exec_sandbox.sys, "platform", "linux")
        monkeypatch.setattr(exec_sandbox.shutil, "which", lambda _name: None)

        probe = exec_sandbox.network_denial_prefix.__wrapped__  # type: ignore[attr-defined]

        assert exec_sandbox._process_has_only_loopback() is offline
        assert probe() == (() if offline else None)


class TestOffSwitch:
    def test_env_off_runs_unconfined_and_says_so(
        self, layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(EXEC_SANDBOX_ENV_VAR, "off")
        monkeypatch.setattr(exec_sandbox, "filesystem_backend", lambda: None)

        result = confine(
            ("true",),
            cwd=str(layout["copy"]),
            writable_roots=(str(layout["copy"]),),
            temp_dir=str(layout["temp"]),
        )

        assert isinstance(result, ConfinedCommand)
        assert result.backend is SandboxBackend.DISABLED
        assert result.argv == ("true",) and not result.network_denied
        assert exec_sandbox.sandbox_unavailable_reason() is None

    def test_config_off_disables_and_env_on_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import ouroboros.config.loader as loader

        (tmp_path / "config.yaml").write_text("execution:\n  exec_sandbox: false\n")
        monkeypatch.setattr(loader, "get_config_dir", lambda: tmp_path)
        monkeypatch.delenv(EXEC_SANDBOX_ENV_VAR)
        assert exec_sandbox.config_enabled() is False
        assert exec_sandbox.sandbox_disabled() is True
        monkeypatch.setenv(EXEC_SANDBOX_ENV_VAR, "on")
        assert exec_sandbox.sandbox_disabled() is False

    def test_on_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import ouroboros.config.loader as loader

        monkeypatch.delenv(EXEC_SANDBOX_ENV_VAR)
        monkeypatch.setattr(loader, "get_config_dir", lambda: tmp_path)
        assert exec_sandbox.config_enabled() is True
        assert exec_sandbox.sandbox_disabled() is False

    def test_a_project_env_file_cannot_switch_it_off(self) -> None:
        assert EXEC_SANDBOX_ENV_VAR in UNTRUSTED_ENV_DENYLIST


class TestEnvironment:
    def test_built_from_scratch(self) -> None:
        source = {"PATH": "/bin", "LANG": "C", "HOME": "/home/u", "AWS_SECRET_ACCESS_KEY": "x"}

        env = build_environment("/scratch/tmp", source=source, overrides={"X": "1"})

        assert env == {
            "PATH": "/bin",
            "LANG": "C",
            "TMPDIR": "/scratch/tmp",
            "TMP": "/scratch/tmp",
            "TEMP": "/scratch/tmp",
            "HOME": "/scratch/tmp",
            "X": "1",
        }

    def test_home_is_kept_only_when_passed_through(self) -> None:
        source = {"PATH": "/bin", "HOME": "/home/u"}

        env = build_environment(
            "/scratch/tmp", source=source, passthrough=(*DEFAULT_ENV_PASSTHROUGH, "HOME")
        )

        assert env["HOME"] == "/home/u"


class TestLandlockAccessMask:
    def test_rights_follow_the_abi(self) -> None:
        refer, truncate = 1 << 13, 1 << 14

        assert _landlock_exec.handled_write_access(0) == 0
        abi1 = _landlock_exec.handled_write_access(1)
        assert abi1 and not abi1 & (refer | truncate)
        assert _landlock_exec.handled_write_access(2) == abi1 | refer
        assert _landlock_exec.handled_write_access(3) == abi1 | refer | truncate
        assert _landlock_exec.handled_write_access(8) == abi1 | refer | truncate
        # Reading and executing are never handled.
        read_rights = (1 << 0) | (1 << 2) | (1 << 3)
        assert not _landlock_exec.handled_write_access(8) & read_rights

    def test_helper_refuses_without_a_command(self) -> None:
        assert _landlock_exec.main(["--write", "/tmp"]) == _landlock_exec.EXIT_SANDBOX_FAILED
