"""Confine this process with Landlock, then exec the command (Linux).

Run as a standalone script, never imported into the controller:

    python -I -S -B _landlock_exec.py --write DIR [--write DIR ...] -- ARGV...

It is the Linux backend of ``ouroboros.runtime.exec_sandbox``. The ruleset
handles every filesystem right that creates, changes or removes something
(the rights the running kernel's Landlock ABI knows), and grants them only
beneath each ``--write`` directory. Writing to a few character devices
(``/dev/null`` and friends) is also granted. Reading and executing are not
handled, so they stay governed by the ordinary file permissions. The
restriction is applied to this process and inherited by everything it execs
or forks; it cannot be lifted.

It depends on nothing but the standard library, so it starts with ``-S`` (no
``site``) and ``-I`` (no environment-controlled import paths). Every failure
exits before the command runs: status 125 when the sandbox could not be
applied, 126 or 127 when the command could not be executed.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import os
import struct
import sys
from typing import Any

# Generic syscall numbers, the same on x86_64 and aarch64 (and every other
# architecture that uses the unified table).
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
# ``os.O_PATH`` exists only on Linux builds of Python.
_O_PATH: int = getattr(os, "O_PATH", 0o10000000)

_ACCESS_FS_WRITE_FILE = 1 << 1
_ACCESS_FS_REMOVE_DIR = 1 << 4
_ACCESS_FS_REMOVE_FILE = 1 << 5
_ACCESS_FS_MAKE_CHAR = 1 << 6
_ACCESS_FS_MAKE_DIR = 1 << 7
_ACCESS_FS_MAKE_REG = 1 << 8
_ACCESS_FS_MAKE_SOCK = 1 << 9
_ACCESS_FS_MAKE_FIFO = 1 << 10
_ACCESS_FS_MAKE_BLOCK = 1 << 11
_ACCESS_FS_MAKE_SYM = 1 << 12
_ACCESS_FS_REFER = 1 << 13  # ABI 2
_ACCESS_FS_TRUNCATE = 1 << 14  # ABI 3

# ABI 1 rights that change the filesystem.
_WRITE_ACCESS_ABI1 = (
    _ACCESS_FS_WRITE_FILE
    | _ACCESS_FS_REMOVE_DIR
    | _ACCESS_FS_REMOVE_FILE
    | _ACCESS_FS_MAKE_CHAR
    | _ACCESS_FS_MAKE_DIR
    | _ACCESS_FS_MAKE_REG
    | _ACCESS_FS_MAKE_SOCK
    | _ACCESS_FS_MAKE_FIFO
    | _ACCESS_FS_MAKE_BLOCK
    | _ACCESS_FS_MAKE_SYM
)

# Devices a process may write without leaving a trace outside the sandbox.
WRITABLE_DEVICES = (
    "/dev/null",
    "/dev/zero",
    "/dev/full",
    "/dev/random",
    "/dev/urandom",
    "/dev/tty",
)

EXIT_SANDBOX_FAILED = 125
EXIT_NOT_EXECUTABLE = 126
EXIT_NOT_FOUND = 127


class _RulesetAttr(ctypes.Structure):
    # The ABI 1 layout; newer fields (network, scopes) are left unset, which
    # the kernel accepts for a shorter structure.
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class SandboxError(Exception):
    """Landlock could not be applied; the command must not run."""


def handled_write_access(abi: int) -> int:
    """The write rights to handle under Landlock ABI ``abi`` (0 below ABI 1)."""
    if abi < 1:
        return 0
    access = _WRITE_ACCESS_ABI1
    if abi >= 2:
        access |= _ACCESS_FS_REFER
    if abi >= 3:
        access |= _ACCESS_FS_TRUNCATE
    return access


def device_write_access(abi: int) -> int:
    """The rights granted on a writable device (file rights only)."""
    return _ACCESS_FS_WRITE_FILE | (_ACCESS_FS_TRUNCATE if abi >= 3 else 0)


def _syscall() -> Callable[..., Any]:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.syscall
    function.restype = ctypes.c_long
    return function


def _check(result: int, what: str) -> int:
    if result < 0:
        errno = ctypes.get_errno()
        raise SandboxError(f"{what} failed: {os.strerror(errno)} (errno {errno})")
    return result


def landlock_abi() -> int:
    """The running kernel's Landlock ABI version; 0 when Landlock is unavailable."""
    if not sys.platform.startswith("linux"):
        return 0
    result = _syscall()(
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(None),
        ctypes.c_size_t(0),
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    return max(int(result), 0)


def _add_rule(syscall: Callable[..., Any], ruleset: int, path: str, access: int) -> None:
    fd = os.open(path, _O_PATH | os.O_CLOEXEC)
    try:
        # struct landlock_path_beneath_attr is packed: u64 allowed_access, s32 parent_fd.
        attr = ctypes.create_string_buffer(struct.pack("=Qi", access, fd), 12)
        _check(
            syscall(
                _SYS_LANDLOCK_ADD_RULE,
                ctypes.c_int(ruleset),
                ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                ctypes.byref(attr),
                ctypes.c_uint32(0),
            ),
            f"landlock_add_rule({path})",
        )
    finally:
        os.close(fd)


def restrict_writes(writable: list[str]) -> int:
    """Allow writes only beneath ``writable`` for this process; return the ABI."""
    abi = landlock_abi()
    if abi < 1:
        raise SandboxError("Landlock is not supported or not enabled by this kernel")
    syscall = _syscall()
    handled = handled_write_access(abi)
    attr = _RulesetAttr(handled_access_fs=handled)
    ruleset = _check(
        syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(attr),
            ctypes.c_size_t(ctypes.sizeof(attr)),
            ctypes.c_uint32(0),
        ),
        "landlock_create_ruleset",
    )
    try:
        for path in writable:
            if not os.path.isdir(path):
                raise SandboxError(f"writable root is not a directory: {path}")
            _add_rule(syscall, ruleset, path, handled)
        for device in WRITABLE_DEVICES:
            if os.path.exists(device):
                _add_rule(syscall, ruleset, device, device_write_access(abi))
        libc = ctypes.CDLL(None, use_errno=True)
        _check(
            libc.prctl(
                ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
                ctypes.c_ulong(1),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            ),
            "prctl(PR_SET_NO_NEW_PRIVS)",
        )
        _check(
            syscall(_SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset), ctypes.c_uint32(0)),
            "landlock_restrict_self",
        )
    finally:
        os.close(ruleset)
    return abi


def _parse(arguments: list[str]) -> tuple[list[str], list[str]]:
    writable: list[str] = []
    index = 0
    while index < len(arguments) and arguments[index] == "--write":
        if index + 1 >= len(arguments):
            raise SandboxError("--write needs a directory")
        writable.append(arguments[index + 1])
        index += 2
    if index >= len(arguments) or arguments[index] != "--" or index + 1 >= len(arguments):
        raise SandboxError("usage: --write DIR [--write DIR ...] -- ARGV...")
    return writable, arguments[index + 1 :]


def main(arguments: list[str]) -> int:
    try:
        writable, command = _parse(arguments)
        restrict_writes(writable)
    except (SandboxError, OSError) as exc:
        sys.stderr.write(f"ouroboros exec sandbox: {exc}\n")
        return EXIT_SANDBOX_FAILED
    try:
        os.execvp(command[0], command)
    except FileNotFoundError as exc:
        sys.stderr.write(f"ouroboros exec sandbox: {command[0]}: {exc.strerror}\n")
        return EXIT_NOT_FOUND
    except OSError as exc:
        sys.stderr.write(f"ouroboros exec sandbox: {command[0]}: {exc.strerror}\n")
        return EXIT_NOT_EXECUTABLE
    return EXIT_NOT_EXECUTABLE  # pragma: no cover - execvp does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
