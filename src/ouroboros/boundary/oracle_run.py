"""Controller side of an oracle check: target processes, then the comparator.

One oracle check runs in two kinds of process, never in one:

1. **Target**, one process per case, in the project interpreter
   (``<interpreter> -I -B -c <harness> target <nonce> <call_kind> <symbol>``)
   with the checkout copy as cwd. It imports and resolves the bound symbol and
   writes a ``resolved`` frame; only then does the controller send that
   case's inputs on stdin. It writes the observation as a ``result`` frame.
   Frames are JSON after a per-process random nonce; every other stdout line
   is ignored, and the target code's own output goes to stderr, which is
   discarded. A CLI oracle's target is the bound command itself.
2. **Comparator**, after every target process has exited, in the
   controller's own interpreter (``<python> -I -S -c <harness> compare``)
   with a fresh owner-only directory as cwd (``boundary/controller_dir.py``).
   It receives the frozen oracle data, the binding, and the observations over
   stdin and decides every case. It never imports workspace code.

Expected values therefore never reach a target process or any file: they
exist in the controller's memory and on the comparator's stdin, which is
written only after the last target process has exited.

Outcomes (``OracleRun``):

- a missing ``resolved`` frame (the process died or hung before the target
  was resolved, or the frame is malformed) is a setup failure: exit 3, no
  failure signature, indeterminate;
- ``resolve`` ``missing`` (the bound symbol does not exist) fails every case;
  ``import_error`` is a setup failure;
- after a ``resolved`` frame, a crash or a timeout on a candidate is that
  case failing, with the counterexample ``observed crash`` or ``observed
  timeout``; on the base (admission, and the one base run of a late binding)
  it is indeterminate, as before, and a timeout keeps ``timed_out``;
- a malformed ``result`` frame is indeterminate.

Exit codes of ``OracleRun``: 0 every case passed, 1 a case failed (the
failure signature is "seen"), 3 indeterminate, ``None`` with ``timed_out``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import functools
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import sys
import time
from typing import Any

from ouroboros.boundary.binding import Binding, CallKind
from ouroboros.boundary.controller_dir import (
    controller_mutations,
    create_controller_dir,
    remove_controller_dir,
)
from ouroboros.boundary.oracle import (
    ORACLE_DATA_PATH,
    ORACLE_HARNESS_PATH,
    ORACLE_HARNESS_SOURCE,
    OracleSpec,
)
from ouroboros.boundary.tree import tree_manifest

COMPARATOR_TIMEOUT_SECONDS = 30.0
_FRAME_LIMIT = 8 * 1024 * 1024
_CLI_OUTPUT_LIMIT = 1024 * 1024
_POSIX = sys.platform != "win32"
_DECIDED = frozenset({"ok", "missing"})


@dataclass(frozen=True, slots=True)
class OracleRun:
    """What one oracle check did, in the shape of a completed command."""

    return_code: int | None
    timed_out: bool
    launch_error: str | None
    output: str
    result: dict[str, Any] | None
    duration: float
    ctrl_mutations: tuple[str, ...] = ()

    @property
    def signature_seen(self) -> bool:
        return self.return_code == 1


@functools.cache
def _harness() -> dict[str, Any]:
    """The frozen harness's helper functions, for the call shape the controller sends."""
    namespace: dict[str, Any] = {"__name__": "ouroboros_oracle_harness"}
    exec(compile(ORACLE_HARNESS_SOURCE, ORACLE_HARNESS_PATH, "exec"), namespace)
    return namespace


def comparator_interpreter() -> str:
    """The controller's own interpreter binary (a venv's base interpreter when it has one)."""
    candidate = getattr(sys, "_base_executable", None) or sys.executable
    real = os.path.realpath(candidate)
    return real if os.path.isfile(real) else sys.executable


def target_interpreter(interpreter: str | None, env: Mapping[str, str]) -> str | None:
    """The interpreter a target process runs in: the project's, else ``python3`` on PATH."""
    if interpreter:
        return interpreter
    return shutil.which("python3", path=env.get("PATH")) or shutil.which("python3")


def _within(path: str, roots: tuple[Path, ...]) -> bool:
    resolved = Path(path).resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if _POSIX:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


async def _reap(process: asyncio.subprocess.Process) -> int | None:
    _kill(process)
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        return None
    return process.returncode


async def _next_frame(
    process: asyncio.subprocess.Process, nonce: str, deadline: float
) -> tuple[str, dict[str, Any] | None]:
    """``("frame", payload)``, or ``("eof" | "timeout" | "malformed", None)``."""
    assert process.stdout is not None
    prefix = (nonce + " ").encode("ascii")
    loop = asyncio.get_running_loop()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return "timeout", None
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
        except TimeoutError:
            return "timeout", None
        except (ValueError, asyncio.LimitOverrunError):
            return "malformed", None
        if not line:
            return "eof", None
        if not line.startswith(prefix):
            continue  # not a frame: ignored
        try:
            payload = json.loads(line[len(prefix) :])
        except ValueError:
            return "malformed", None
        return ("frame", payload) if isinstance(payload, dict) else ("malformed", None)


@dataclass(frozen=True, slots=True)
class _Case:
    """One target process: ``kind`` is ``observed``, ``resolve``, or a setup failure."""

    kind: str
    entry: dict[str, Any] | None = None
    resolve: str = "ok"
    detail: str = ""
    timed_out: bool = False


async def _python_case(
    argv: list[str],
    cwd: Path,
    env: Mapping[str, str],
    nonce: str,
    call: dict[str, Any],
    budget: float,
) -> _Case:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(budget, 0.01)
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=_POSIX,
            limit=_FRAME_LIMIT,
        )
    except OSError as exc:
        return _Case("launch_error", resolve="launch_failed", detail=f"{type(exc).__name__}: {exc}")
    try:
        status, frame = await _next_frame(process, nonce, deadline)
        if status == "timeout":
            return _Case("setup", resolve="setup_timeout", timed_out=True)
        if status == "eof":
            return _Case("setup", resolve="setup_failed", detail="no resolved frame")
        if frame is None or frame.get("phase") != "resolved":
            return _Case("setup", resolve="frame_malformed")
        resolve = frame.get("resolve")
        if resolve != "ok":
            if resolve not in ("missing", "import_error"):
                return _Case("setup", resolve="frame_malformed")
            return _Case(
                "resolve", resolve=str(resolve), detail=str(frame.get("detail") or "")[:500]
            )
        assert process.stdin is not None
        try:
            process.stdin.write((json.dumps(call) + "\n").encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        status, frame = await _next_frame(process, nonce, deadline)
        if status == "timeout":
            return _Case("observed", entry={"case_id": call["case_id"], "outcome": "timeout"})
        if status == "eof":
            code = await _reap(process)
            return _Case(
                "observed", entry={"case_id": call["case_id"], "outcome": "crashed", "exit": code}
            )
        entry = (frame or {}).get("entry")
        if (
            status != "frame"
            or (frame or {}).get("phase") != "result"
            or not isinstance(entry, dict)
            or entry.get("case_id") != call["case_id"]
            or entry.get("outcome") not in ("returned", "raised")
        ):
            return _Case("setup", resolve="frame_malformed")
        return _Case("observed", entry=entry)
    finally:
        await _reap(process)


async def _cli_case(
    argv: list[str], cwd: Path, env: Mapping[str, str], stdin: str, budget: float, call_text: str
) -> _Case:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=_POSIX,
        )
    except OSError as exc:
        return _Case("resolve", resolve="missing", detail=f"{call_text}: {exc}")
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(stdin.encode("utf-8")), timeout=max(budget, 0.01)
        )
    except TimeoutError:
        await _reap(process)
        return _Case("observed", entry={"outcome": "timeout", "call": call_text})
    finally:
        await _reap(process)
    return _Case(
        "observed",
        entry={
            "outcome": "exited",
            "call": call_text,
            "exit_code": process.returncode,
            "stdout": stdout[:_CLI_OUTPUT_LIMIT].decode("utf-8", errors="replace"),
        },
    )


async def _observe(
    oracle: OracleSpec,
    binding: Binding,
    harness: str,
    cwd: Path,
    env: Mapping[str, str],
    python: str,
    budget: float,
    *,
    on_base: bool,
) -> tuple[str, str, dict[str, dict[str, Any]], bool]:
    """Run every case; return ``(resolve, detail, observations, timed_out)``."""
    helpers = _harness()
    arg_map = dict(binding.arg_map)
    observations: dict[str, dict[str, Any]] = {}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    cases = list(oracle.cases)
    for position, case in enumerate(cases):
        share = (deadline - loop.time()) / (len(cases) - position)
        if oracle.call_kind is CallKind.CLI:
            argv = helpers["_cli_argv"](
                python, str(cwd), binding.symbol, list(oracle.params), arg_map, case.args
            )
            call_text = " ".join(argv[2:] if argv[0] == python else argv)
            script = binding.symbol
            if not script.startswith("-m ") and not (cwd / script).is_file():
                return "missing", f"{script} not found", {}, False
            outcome = await _cli_case(argv, cwd, env, case.stdin or "", share, call_text)
        else:
            args, kwargs = helpers["_split_args"](list(oracle.params), arg_map, case.args)
            nonce = secrets.token_hex(16)
            argv = [
                python,
                "-I",
                "-B",
                "-c",
                harness,
                "target",
                nonce,
                oracle.call_kind.value,
                binding.symbol,
            ]
            call = {"case_id": case.case_id, "args": args, "kwargs": kwargs, "init": case.init}
            outcome = await _python_case(argv, cwd, env, nonce, call, share)
        if outcome.kind != "observed":
            if position and outcome.kind == "resolve":
                # An earlier case resolved the same symbol: not a stable target.
                return "resolve_unstable", outcome.detail, {}, False
            return outcome.resolve, outcome.detail, {}, outcome.timed_out
        entry = dict(outcome.entry or {})
        if on_base and entry.get("outcome") in ("crashed", "timeout"):
            # On the base a crash or hang is never counted as the intended
            # failure (admission and the base run of a late binding).
            timed_out = entry["outcome"] == "timeout"
            return ("target_timeout" if timed_out else "target_crashed"), "", {}, timed_out
        observations[case.case_id] = entry
    return "ok", "", observations, False


def _comparator_env() -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TMPDIR", "TEMP", "TMP")
    return {key: os.environ[key] for key in keep if key in os.environ}


async def _compare(
    harness: str, request: dict[str, Any], ctrl: Path, budget: float
) -> dict[str, Any] | None:
    argv = [comparator_interpreter(), "-I", "-S", "-c", harness, "compare"]
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=ctrl,
            env=_comparator_env(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=_POSIX,
        )
    except OSError:
        return None
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(json.dumps(request).encode("utf-8")), timeout=budget
        )
    except TimeoutError:
        return None
    finally:
        await _reap(process)
    prefix = request["nonce"] + " "
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix) :])
            except ValueError:
                return None
            if isinstance(value, dict) and isinstance(value.get("cases"), list):
                return value
            return None
    return None


def _undecided_result(
    oracle: OracleSpec, binding: Binding, source: str, resolve: str
) -> dict[str, Any]:
    return {
        "check_id": oracle.check_id,
        "criterion_key": oracle.criterion_key,
        "binding_source": source,
        "symbol": binding.symbol,
        "call_kind": oracle.call_kind.value,
        "resolve": resolve,
        "cases": [
            {"case_id": case.case_id, "held_out": case.held_out, "passed": False, "detail": ""}
            for case in oracle.cases
        ],
    }


def _render(oracle: OracleSpec, result: dict[str, Any], code: int | None, detail: str) -> str:
    """Output for receipts and people: never target output, held-out cases by id only."""
    if code not in (0, 1):
        return f"oracle could not run the target: {result.get('resolve')} {detail}".rstrip()
    if code == 0:
        return "oracle: every case passed"
    lines = [oracle.failure_signature]
    for case in result.get("cases") or ():
        if case.get("passed"):
            continue
        if case.get("held_out"):
            lines.append(f"counterexample (held-out): {case.get('case_id')}")
        else:
            lines.append(f"counterexample: {case.get('detail') or case.get('case_id')}")
    return "\n".join(lines)


async def run_oracle_check(
    package_files: Mapping[str, str],
    oracle: OracleSpec,
    cwd: Path,
    *,
    timeout_seconds: float,
    on_base: bool,
    env: Mapping[str, str],
    interpreter: str | None,
    binding: Binding | None,
    workspace_roots: tuple[Path, ...],
) -> OracleRun:
    """Run one oracle check on the checkout copy at ``cwd`` (see the module docstring)."""
    started = time.monotonic()
    harness = package_files.get(ORACLE_HARNESS_PATH)
    data_text = package_files.get(ORACLE_DATA_PATH)
    source = "declared" if binding is not None else "default"
    bound = binding or oracle.default_binding
    python = target_interpreter(interpreter, env)
    comparator = comparator_interpreter()
    if harness is None or data_text is None:
        launch_error = "oracle files missing from the package"
    elif python is None:
        launch_error = "no interpreter for the target"
    elif _within(comparator, workspace_roots):
        launch_error = f"comparator interpreter inside the workspace: {comparator}"
    else:
        launch_error = None
    if launch_error is not None:
        return OracleRun(None, False, launch_error, launch_error, None, 0.0)
    assert harness is not None and data_text is not None and python is not None
    reserve = min(COMPARATOR_TIMEOUT_SECONDS, max(1.0, timeout_seconds * 0.25))
    budget = max(0.1, timeout_seconds - reserve)
    ctrl = create_controller_dir()
    try:
        before = tree_manifest(ctrl, unprotected_names=())
        resolve, detail, observations, timed_out = await _observe(
            oracle, bound, harness, cwd, env, python, budget, on_base=on_base
        )
        result: dict[str, Any] | None = None
        if resolve in _DECIDED:
            nonce = secrets.token_hex(16)
            result = await _compare(
                harness,
                {
                    "nonce": nonce,
                    "oracle": json.loads(data_text),
                    "check_id": oracle.check_id,
                    "binding": bound.to_dict() if binding is not None else None,
                    "resolve": resolve,
                    "detail": detail,
                    "observations": observations,
                },
                ctrl,
                reserve,
            )
            if result is None:
                resolve = "comparator_failed"
        mutations = controller_mutations(ctrl, before)
    finally:
        remove_controller_dir(ctrl)
    if result is None:
        result = _undecided_result(oracle, bound, source, resolve)
        code: int | None = None if timed_out else 3
    else:
        code = 0 if all(case.get("passed") for case in result["cases"]) else 1
    return OracleRun(
        return_code=code,
        timed_out=timed_out,
        launch_error=None,
        output=_render(oracle, result, code, detail),
        result=result,
        duration=time.monotonic() - started,
        ctrl_mutations=mutations,
    )


__all__ = [
    "COMPARATOR_TIMEOUT_SECONDS",
    "OracleRun",
    "comparator_interpreter",
    "run_oracle_check",
    "target_interpreter",
]
