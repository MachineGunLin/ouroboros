"""The Windows backend of the execution sandbox: a per-run AppContainer.

Run on the ``windows-latest`` CI job. On GitHub Actions the backend must be
available (a missing backend fails there instead of skipping); elsewhere the
tests skip with the probe's reason.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest

from ouroboros.runtime import _sandbox_probe as probe_module
from ouroboros.runtime import exec_sandbox
from ouroboros.runtime.exec_sandbox import (
    ConfinedCommand,
    SandboxBackend,
    SandboxUnavailable,
    SandboxUnavailableReason,
    confine,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows only")

if sys.platform == "win32":
    import _winapi
    import ctypes

    from ouroboros.runtime import _confine_windows as launcher


def _require_backend() -> None:
    reason = exec_sandbox.sandbox_unavailable_reason(deny_network=False)
    if reason is None:
        return
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.fail(f"the AppContainer backend must work on the CI runner: {reason.value}")
    pytest.skip(f"execution sandbox unavailable on this host: {reason.value}")


def _python(code: str, *args: str) -> tuple[str, ...]:
    return (sys.executable, "-I", "-c", code, *args)


def _run(command: ConfinedCommand, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - argv built by confine
        list(command.argv),
        cwd=command.cwd,
        env=dict(command.env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _confine(layout: dict[str, Path], argv: tuple[str, ...], **kwargs: object) -> ConfinedCommand:
    command = confine(
        argv,
        cwd=str(layout["copy"]),
        writable_roots=(str(layout["copy"]),),
        temp_dir=str(layout["temp"]),
        **{"deny_network": True, **kwargs},  # type: ignore[arg-type]
    )
    assert isinstance(command, ConfinedCommand), command
    assert command.backend is SandboxBackend.APPCONTAINER
    return command


def _container_sid(command: ConfinedCommand) -> str:
    name = command.argv[command.argv.index("--appcontainer") + 1]
    return str(launcher.appcontainer_sid(launcher._api(), name))


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path.resolve()
    paths = {name: base / name for name in ("copy", "temp", "outside")}
    for path in paths.values():
        path.mkdir()
    return paths


def test_windows_ci_runner_has_appcontainer() -> None:
    """On GitHub Actions the probe must pass, so every test below really runs there."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        pytest.skip("only asserted on the GitHub Actions runner")
    assert exec_sandbox.filesystem_backend() is SandboxBackend.APPCONTAINER


class TestWrites:
    def test_every_kind_of_write_outside_is_denied_and_writes_inside_succeed(
        self, layout: dict[str, Path]
    ) -> None:
        _require_backend()
        copy, outside = layout["copy"], layout["outside"]
        # The probe is copied into the root: the container reads only granted paths.
        shutil.copy(probe_module.__file__, copy / "probe.py")
        nested = copy / "pkg" / "deep"
        nested.mkdir(parents=True)
        (nested / "existing.txt").write_text("old", encoding="utf-8")
        inside = copy / "inside"
        inside.mkdir()
        probe_module.prepare(str(outside))
        before = probe_module.snapshot(str(outside))
        code = (
            "import json, runpy, sys\n"
            "probe = runpy.run_path('probe.py')\n"
            "result = probe['run'](sys.argv[1], sys.argv[2])\n"
            "open('pkg/deep/existing.txt', 'a').write('+new')\n"
            "print(json.dumps(result))\n"
        )
        command = _confine(layout, _python(code, str(inside), str(outside)), deny_network=False)

        result = _run(command)

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        denied = {name for name, outcome in report["outside"].items() if outcome == "denied"}
        assert set(probe_module.REQUIRED) <= denied, report
        assert "ok" not in report["outside"].values(), report
        assert set(report["inside"].values()) == {"ok"}, report
        assert (nested / "existing.txt").read_text(encoding="utf-8") == "old+new"
        assert probe_module.snapshot(str(outside)) == before

    def test_a_link_beneath_a_root_does_not_carry_the_write_grant(
        self, layout: dict[str, Path]
    ) -> None:
        """Granting the root must not propagate through a junction or a symlink."""
        _require_backend()
        copy, outside = layout["copy"], layout["outside"]
        (outside / "kept.txt").write_text("keep", encoding="utf-8")
        _winapi.CreateJunction(str(outside), str(copy / "junction"))
        links = ["junction"]
        try:
            os.symlink(outside, copy / "symlink", target_is_directory=True)
            links.append("symlink")
        except OSError:
            pass  # no symlink privilege on this host: the junction still covers it
        code = (
            "import sys\n"
            "escaped = []\n"
            "for link in sys.argv[1:]:\n"
            "    if open(f'{link}/kept.txt').read() != 'keep':\n"
            "        sys.exit(4)\n"
            "    for attempt in (lambda: open(f'{link}/new.txt', 'w').write('x'),\n"
            "                    lambda: open(f'{link}/kept.txt', 'a').write('x')):\n"
            "        try:\n"
            "            attempt()\n"
            "            escaped.append(link)\n"
            "        except OSError:\n"
            "            pass\n"
            "sys.exit(3 if escaped else 0)\n"
        )
        before = launcher.dacl_sddl(str(outside / "kept.txt"))
        command = _confine(layout, _python(code, *links))

        result = _run(command)

        assert result.returncode == 0, (result.returncode, result.stderr)
        assert (outside / "kept.txt").read_text(encoding="utf-8") == "keep"
        assert not (outside / "new.txt").exists()
        assert _container_sid(command) not in launcher.dacl_sddl(str(outside / "kept.txt"))
        assert _container_sid(command) not in before


class TestNetwork:
    _CONNECT = (
        "import socket, sys\n"
        "try:\n"
        "    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=5).close()\n"
        "except OSError:\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    _LOOPBACK = (
        "import socket, sys\n"
        "try:\n"
        "    server = socket.create_server(('127.0.0.1', 0))\n"
        "    client = socket.create_connection(server.getsockname(), timeout=5)\n"
        "    peer, _ = server.accept()\n"
        "    client.sendall(b'ping')\n"
        "    reached = peer.recv(4) == b'ping'\n"
        "except OSError:\n"
        "    reached = False\n"
        "sys.exit(1 if reached else 0)\n"
    )

    def test_network_is_denied(self, layout: dict[str, Path]) -> None:
        _require_backend()
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=5).close()
        except OSError:
            pytest.skip("this host has no network to deny")
        denied = _confine(layout, _python(self._CONNECT, "1.1.1.1", "53"), deny_network=True)
        allowed = _confine(layout, _python(self._CONNECT, "1.1.1.1", "53"), deny_network=False)

        assert denied.network_denied and not allowed.network_denied
        assert _run(denied).returncode == 0
        # With the client capability the same connection works: the denial
        # comes from the missing capability, not from this host.
        assert _run(allowed).returncode == 1

    @pytest.mark.parametrize("deny_network", [True, False])
    def test_loopback_is_unavailable_and_reported(
        self, layout: dict[str, Path], deny_network: bool
    ) -> None:
        _require_backend()
        command = _confine(layout, _python(self._LOOPBACK), deny_network=deny_network)

        assert command.loopback_available is False
        assert _run(command).returncode == 0


def _process_alive(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0) != 0
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


class TestTimeout:
    _TREE = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-I', '-c',\n"
        '    \'import time; time.sleep(4); open("late.txt", "w").write("x")\'])\n'
        "print(child.pid, flush=True)\n"
        "time.sleep(120)\n"
    )

    def test_killing_the_launcher_kills_the_whole_tree(self, layout: dict[str, Path]) -> None:
        _require_backend()
        command = _confine(layout, _python(self._TREE))
        process = subprocess.Popen(  # noqa: S603 - argv built by confine
            list(command.argv),
            cwd=command.cwd,
            env=dict(command.env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        grandchild = int(process.stdout.readline())
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=1)

        process.kill()  # what a caller's timeout does
        process.communicate(timeout=60)

        deadline = time.monotonic() + 10
        while _process_alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _process_alive(grandchild)
        time.sleep(5)
        assert not (layout["copy"] / "late.txt").exists()

    def test_the_callers_timeout_path_leaves_nothing_behind(self, tmp_path: Path) -> None:
        """Through the replay's own runner: timeout, tree killed, roots deletable."""
        import asyncio

        from ouroboros.orchestrator.verify_command_runner import _run_process

        _require_backend()
        scratch = tmp_path.resolve() / "scratch"
        layout = {"copy": scratch / "workspace", "temp": scratch / "tmp"}
        for path in layout.values():
            path.mkdir(parents=True)
        command = _confine(layout, _python(self._TREE))
        sid = _container_sid(command)

        run = asyncio.run(
            _run_process(command.argv, cwd=command.cwd, env=command.env, timeout_seconds=5)
        )

        assert run.timed_out
        time.sleep(5)
        assert not (layout["copy"] / "late.txt").exists()
        # The caller deletes its roots; the per-run grants go with them.
        shutil.rmtree(scratch)
        assert not scratch.exists()
        for path in exec_sandbox._windows_read_paths(()):
            assert sid not in launcher.dacl_sddl(path)


class TestGrants:
    def test_write_grants_are_revoked_after_every_run(self, layout: dict[str, Path]) -> None:
        _require_backend()
        copy, temp = layout["copy"], layout["temp"]
        (copy / "sub").mkdir()
        (copy / "sub" / "file.txt").write_text("x", encoding="utf-8")
        watched = [copy, copy / "sub", copy / "sub" / "file.txt", temp]
        before = {path: launcher.dacl_sddl(str(path)) for path in watched}
        commands = {
            "succeeds": (_python("open('made.txt', 'w').write('x')"), 0),
            "fails": (_python("import sys; sys.exit(3)"), 3),
            "is not found": (("ouroboros-no-such-command",), 127),
        }
        for label, (argv, status) in commands.items():
            command = _confine(layout, argv)
            sid = _container_sid(command)

            result = _run(command)

            assert result.returncode == status, (label, result.stderr)
            after = {path: launcher.dacl_sddl(str(path)) for path in watched}
            assert after == before, label
            if (copy / "made.txt").exists():
                assert sid not in launcher.dacl_sddl(str(copy / "made.txt")), label

    def test_the_persistent_read_grant_is_recorded_and_removable(
        self, layout: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_backend()
        manifest = tmp_path.resolve() / "state" / "read-grants.jsonl"
        monkeypatch.setattr(exec_sandbox, "_read_grant_manifest", lambda: manifest)
        deps = tmp_path.resolve() / "deps"
        deps.mkdir()
        (deps / "module.txt").write_text("dependency", encoding="utf-8")
        original = launcher.dacl_sddl(str(deps))
        _winapi.CreateJunction(str(deps), str(layout["copy"] / "deps"))
        read = _python("import sys; sys.exit(0 if open('deps/module.txt').read() else 1)")

        result = _run(_confine(layout, read))

        assert result.returncode == 0, result.stderr
        recorded = [json.loads(line)["path"] for line in manifest.read_text().splitlines()]
        assert recorded == [str(deps)]
        capability = str(launcher.capability_sid(launcher._api()))
        assert capability in launcher.dacl_sddl(str(deps))
        assert capability in launcher.dacl_sddl(str(deps / "module.txt"))

        removed = exec_sandbox.remove_persistent_read_grants()

        assert removed == (str(deps),)
        assert launcher.dacl_sddl(str(deps)) == original
        assert capability not in launcher.dacl_sddl(str(deps / "module.txt"))
        assert not manifest.exists()
        # Without the grant the container cannot read it; the next run grants it again.
        second = _confine(layout, read)
        assert _run(second).returncode == 0
        assert exec_sandbox.remove_persistent_read_grants() == (str(deps),)


class TestUnavailable:
    def test_no_backend_when_the_appcontainer_cannot_be_created(
        self, layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing AppContainer setup makes the probe fail: nothing runs, ever."""

        def refuse(api: object, name: str) -> object:
            raise launcher.SandboxError(f"DeriveAppContainerSidFromAppContainerName({name!r})")

        marker = layout["copy"] / "ran.txt"
        before = launcher.dacl_sddl(str(layout["copy"]))
        status = os.stat(layout["copy"])
        monkeypatch.setenv(launcher.COMMAND_ENV_VARIABLE, json.dumps({"PATH": os.defpath}))
        monkeypatch.setattr(launcher, "appcontainer_sid", refuse)
        argv = [
            "--appcontainer",
            "ouroboros.sandbox.test",
            "--manifest",
            str(layout["temp"] / "grants.jsonl"),
            "--root",
            str(layout["copy"]),
            str(status.st_dev),
            str(status.st_ino),
            "--",
            *_python(f"open({str(marker)!r}, 'w').close()"),
        ]

        # The root itself verifies: the refusal below is the AppContainer's.
        launcher.Root(str(layout["copy"]), status.st_dev, status.st_ino).close()

        assert launcher.main(argv) == launcher.EXIT_SANDBOX_FAILED
        assert not marker.exists()
        assert launcher.dacl_sddl(str(layout["copy"])) == before

    def test_an_invalid_appcontainer_name_reports_the_sandbox_unavailable(
        self, layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: the real Win32 call fails, the probe fails, confine refuses."""
        monkeypatch.setattr(exec_sandbox, "_appcontainer_name", lambda: "x" * 200)
        probe = exec_sandbox.filesystem_backend.__wrapped__  # type: ignore[attr-defined]
        monkeypatch.setattr(exec_sandbox, "filesystem_backend", probe)

        result = confine(
            _python("pass"),
            cwd=str(layout["copy"]),
            writable_roots=(str(layout["copy"]),),
            temp_dir=str(layout["temp"]),
        )

        assert isinstance(result, SandboxUnavailable)
        assert result.reason is SandboxUnavailableReason.SANDBOX_UNAVAILABLE

    def test_a_root_swapped_after_confine_runs_nothing(self, layout: dict[str, Path]) -> None:
        _require_backend()
        copy, outside = layout["copy"], layout["outside"]
        command = _confine(layout, _python("open('escaped.txt', 'w').write('x')"))
        copy.rename(copy.with_name("copy-moved"))
        _winapi.CreateJunction(str(outside), str(copy))

        result = _run(command)

        assert result.returncode == launcher.EXIT_SANDBOX_FAILED, result.stderr
        assert not (outside / "escaped.txt").exists()


class TestOtherProcesses:
    _READER = (
        "import ctypes, sys\n"
        "kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)\n"
        "kernel32.OpenProcess.restype = ctypes.c_void_p\n"
        "opened = []\n"
        "for pid in map(int, sys.argv[1:]):\n"
        "    # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ: what reading\n"
        "    # another process's environment block needs.\n"
        "    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)\n"
        "    if handle:\n"
        "        opened.append(pid)\n"
        "        kernel32.CloseHandle(ctypes.c_void_p(handle))\n"
        "    elif ctypes.get_last_error() != 5:\n"
        "        sys.exit(4)\n"
        "sys.exit(3 if opened else 0)\n"
    )

    def test_another_process_environment_cannot_be_read(self, layout: dict[str, Path]) -> None:
        _require_backend()
        secret_holder = subprocess.Popen(  # noqa: S603 - fixed argv
            [sys.executable, "-I", "-c", "import time; time.sleep(120)"],
            env={**os.environ, "OUROBOROS_TEST_SECRET": "hunter2-sandbox-secret"},
        )
        try:
            targets = (str(os.getpid()), str(secret_holder.pid))
            unconfined = subprocess.run(  # noqa: S603 - fixed argv
                _python(self._READER, *targets), timeout=60, check=False
            )
            assert unconfined.returncode == 3, "the same read works unconfined"
            command = _confine(layout, _python(self._READER, *targets))

            confined = _run(command)
        finally:
            secret_holder.kill()
            secret_holder.wait(timeout=30)

        assert command.isolates_process_environments
        assert confined.returncode == 0, confined.stderr
