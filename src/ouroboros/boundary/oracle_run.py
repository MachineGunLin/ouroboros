"""Controller side of an oracle check: target processes, then an in-process comparison.

Threat model: the adversary is the code under test (and the worker that
wrote it), running as the same OS user as the controller, with no OS sandbox.
The design goal is that the verdict cannot be forged and held-out expected
values cannot be read by a target. Everything a target can do is limited to
the bytes it writes on its frame pipe; see "Residual risks" in the boundary
notes for what needs an OS sandbox.

One oracle check runs as follows:

1. **Target**, one process per case, in the project interpreter
   (``<interpreter> -I -B -c <harness> target <nonce> <call_kind> <symbol>``)
   with the checkout copy as cwd, in its own session and process group. It
   imports and resolves the bound symbol and writes a ``resolved`` frame; only
   then does the controller send that case's inputs on stdin. It writes the
   observation as a ``result`` frame. Frames are JSON after a per-process
   random nonce on a pipe; every other line is ignored, and the target code's
   own output goes to stderr, which is discarded. A CLI oracle's target is the
   bound command itself. When the case is over, the controller kills the
   target's whole process group (on Linux also every process still in its
   session) and waits, bounded, for the group to be empty.
2. **Comparison**, in the controller process itself, with the harness's
   comparison functions compiled from ``ORACLE_HARNESS_SOURCE`` when this
   module is imported, before any target runs. The controller parses each
   frame with a bounded parser (size, depth, node count, integer digits,
   finite floats), validates the observation's fields, and compares against
   the frozen expectations it holds in memory. Nothing is imported, executed,
   or read from disk to decide the verdict after the first target starts.

Expected values therefore never reach a target process, a file, or another
process. A target that escapes the kill (a double fork plus ``setsid``)
cannot change the verdict either: the verdict is computed in the controller's
memory from frame bytes that were already read, and no later channel exists.

Outcomes (``OracleRun``):

- a failure before any target code runs (oracle files missing, the harness
  differs from this product version, no interpreter, the first target cannot
  be launched) is indeterminate;
- on the base (admission, and the one base run of a late binding), anything
  but a clean observation of every case (no ``resolved`` frame, an import
  error, a crash, a timeout, a malformed frame) is indeterminate: a base run
  never counts an accident as the intended failure. ``missing`` (the bound
  symbol does not exist) fails every case, which is the expected reproduction
  failure;
- on a candidate, the target's code has run once the first process starts, so
  every anomaly is that case failing with a counterexample: a crash, a
  timeout, a malformed or oversized frame, a missing field, a value out of
  range, or a target that resolves on one call and not on another. ``missing``
  and an import error on the first case fail every case. One bad case never
  makes the other cases undecided.

Exit codes of ``OracleRun``: 0 every case passed, 1 a case failed (the
failure signature is "seen"), 3 indeterminate, ``None`` with ``timed_out``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import sys
import time
from typing import Any

from ouroboros.boundary.binding import Binding, CallKind
from ouroboros.boundary.oracle import (
    ORACLE_DATA_PATH,
    ORACLE_HARNESS_PATH,
    ORACLE_HARNESS_SOURCE,
    OracleSpec,
)

_FRAME_LIMIT = 8 * 1024 * 1024
_CLI_OUTPUT_LIMIT = 1024 * 1024
_READ_CHUNK = 64 * 1024
_MAX_DEPTH = 64
_MAX_NODES = 1_000_000
_MAX_INT_DIGITS = 1000
_GROUP_GRACE_SECONDS = 1.0
_POSIX = sys.platform != "win32"
_LINUX = sys.platform.startswith("linux")
_OBSERVED = frozenset({"returned", "raised"})


@dataclass(frozen=True, slots=True)
class OracleRun:
    """What one oracle check did, in the shape of a completed command."""

    return_code: int | None
    timed_out: bool
    launch_error: str | None
    output: str
    result: dict[str, Any] | None
    duration: float

    @property
    def signature_seen(self) -> bool:
        return self.return_code == 1


def _load_harness() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "ouroboros_oracle_harness"}
    exec(compile(ORACLE_HARNESS_SOURCE, ORACLE_HARNESS_PATH, "exec"), namespace)
    return namespace


# Compiled when the controller imports this module, before any target runs;
# the comparison never compiles, imports, or reads anything later.
_HARNESS = _load_harness()


def _harness() -> dict[str, Any]:
    """The frozen harness's functions (call shapes and the comparison rule)."""
    return _HARNESS


def target_interpreter(interpreter: str | None, env: Mapping[str, str]) -> str | None:
    """The interpreter a target process runs in: the project's, else ``python3`` on PATH."""
    if interpreter:
        return interpreter
    return shutil.which("python3", path=env.get("PATH")) or shutil.which("python3")


# --------------------------------------------------------------------------
# Bounded frame parsing


def _bounded_int(text: str) -> int:
    if len(text.lstrip("-")) > _MAX_INT_DIGITS:
        raise ValueError("integer out of range")
    return int(text)


def _bounded_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("float out of range")
    return value


def _reject_constant(text: str) -> Any:
    raise ValueError(f"{text} is not plain JSON")


def _within_bounds(value: Any) -> bool:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > _MAX_DEPTH or nodes > _MAX_NODES:
            return False
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return True


def parse_frame(raw: bytes) -> dict[str, Any] | None:
    """One frame's JSON object, or ``None`` when it is malformed or out of bounds."""
    if len(raw) > _FRAME_LIMIT:
        return None
    try:
        value = json.loads(
            raw,
            parse_int=_bounded_int,
            parse_float=_bounded_float,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError, OverflowError, MemoryError):
        return None
    if not isinstance(value, dict) or not _within_bounds(value):
        return None
    return value


def valid_entry(entry: Any, case_id: str) -> bool:
    """Whether a ``result`` frame's entry has every field the comparison reads."""
    if not isinstance(entry, dict) or entry.get("case_id") != case_id:
        return False
    if entry.get("outcome") not in _OBSERVED or not isinstance(entry.get("repr"), str):
        return False
    if entry["outcome"] == "returned":
        encodable = entry.get("encodable")
        return isinstance(encodable, bool) and (not encodable or "value" in entry)
    names = entry.get("exception")
    return isinstance(names, list) and bool(names) and all(isinstance(n, str) for n in names)


# --------------------------------------------------------------------------
# Target processes


def _session_members(leader: int) -> list[int]:
    """Linux: processes still in the target's process group or session."""
    if not _LINUX:
        return []
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    own = os.getpid()
    members = []
    for name in names:
        if not name.isdigit() or int(name) == own:
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as handle:
                stat = handle.read()
            # Fields after the command name: state, ppid, pgrp, session.
            fields = stat[stat.rfind(b")") + 2 :].split()
            if int(fields[2]) == leader or int(fields[3]) == leader:
                members.append(int(name))
        except (OSError, IndexError, ValueError):
            continue
    return members


def _kill_group(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the target's process group (and, on Linux, its session)."""
    if not _POSIX:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        return
    # The group outlives its leader while any member is alive, so it is
    # killed whether or not the leader has already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    for pid in _session_members(process.pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue


async def _discard(reader: asyncio.StreamReader | None) -> None:
    while reader is not None and await reader.read(_READ_CHUNK):
        pass


async def _reap(process: asyncio.subprocess.Process) -> int | None:
    """Kill the target's group, reap the leader, and wait (bounded) for the group to empty.

    Unread output left in the pipe is discarded chunk by chunk: asyncio
    reports the exit only once the pipe reaches end of file, so a reader
    stopped at the output cap would otherwise stall the reap.
    """
    _kill_group(process)
    try:
        await asyncio.wait_for(asyncio.gather(_discard(process.stdout), process.wait()), timeout=10)
    except TimeoutError:
        code = None
    else:
        code = process.returncode
    if _POSIX:
        deadline = time.monotonic() + _GROUP_GRACE_SECONDS
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except (ProcessLookupError, PermissionError):
                break
            _kill_group(process)
            await asyncio.sleep(0.02)
    return code


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
        payload = parse_frame(line[len(prefix) :])
        return ("frame", payload) if payload is not None else ("malformed", None)


class CappedOutput:
    """Output read under a hard byte cap while it streams (R3-S2).

    ``fill`` stops reading once more than ``limit`` bytes arrived and sets
    ``overflow``; at most ``limit`` bytes (plus one read chunk in flight) are
    ever held, whatever the process writes. The caller kills the process.
    Partial output survives a cancelled ``fill`` (for a timeout).
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.overflow = False
        self._chunks: list[bytes] = []
        self._size = 0

    async def fill(self, reader: asyncio.StreamReader | None) -> None:
        while reader is not None and not self.overflow:
            chunk = await reader.read(_READ_CHUNK)
            if not chunk:
                return
            room = self.limit - self._size
            if len(chunk) > room:
                self._chunks.append(chunk[:room])
                self._size = self.limit
                self.overflow = True
                return
            self._chunks.append(chunk)
            self._size += len(chunk)

    @property
    def data(self) -> bytes:
        return b"".join(self._chunks)


@dataclass(frozen=True, slots=True)
class _Case:
    """One target process.

    ``kind`` is ``observed`` (``entry`` is the observation, possibly abnormal),
    ``resolve`` (``missing`` or ``import_error``), ``setup`` (the target never
    produced a valid ``resolved`` frame; ``entry`` says how, for a candidate),
    or ``launch_error``.
    """

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
    case_id = call["case_id"]
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
            return _Case(
                "setup",
                entry={"case_id": case_id, "outcome": "timeout"},
                resolve="setup_timeout",
                timed_out=True,
            )
        if status == "eof":
            code = await _reap(process)
            return _Case(
                "setup",
                entry={"case_id": case_id, "outcome": "crashed", "exit": code},
                resolve="setup_failed",
                detail="no resolved frame",
            )
        resolve = (frame or {}).get("resolve")
        if (frame or {}).get("phase") != "resolved" or resolve not in (
            "ok",
            "missing",
            "import_error",
        ):
            return _Case(
                "setup",
                entry={"case_id": case_id, "outcome": "malformed"},
                resolve="frame_malformed",
            )
        detail = (frame or {}).get("detail")
        if resolve != "ok":
            return _Case(
                "resolve",
                resolve=str(resolve),
                detail=(detail if isinstance(detail, str) else "")[:500],
            )
        assert process.stdin is not None
        try:
            # Bounded like everything else in the case (R3-S4): a target that
            # never reads its stdin cannot stall the controller.
            process.stdin.write((json.dumps(call) + "\n").encode("utf-8"))
            await asyncio.wait_for(process.stdin.drain(), timeout=max(deadline - loop.time(), 0.01))
            process.stdin.close()
        except TimeoutError:
            return _Case("observed", entry={"case_id": case_id, "outcome": "timeout"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        status, frame = await _next_frame(process, nonce, deadline)
        if status == "timeout":
            return _Case("observed", entry={"case_id": case_id, "outcome": "timeout"})
        if status == "eof":
            code = await _reap(process)
            return _Case("observed", entry={"case_id": case_id, "outcome": "crashed", "exit": code})
        entry = (frame or {}).get("entry")
        if (frame or {}).get("phase") != "result" or not valid_entry(entry, case_id):
            return _Case("observed", entry={"case_id": case_id, "outcome": "malformed"})
        assert isinstance(entry, dict)
        return _Case("observed", entry=entry)
    finally:
        await _reap(process)


async def _feed(process: asyncio.subprocess.Process, data: bytes) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(data)
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            process.stdin.close()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


async def _cli_case(
    argv: list[str], cwd: Path, env: Mapping[str, str], stdin: str, budget: float, call_text: str
) -> _Case:
    """One CLI call; its stdout is read under ``_CLI_OUTPUT_LIMIT`` while it streams.

    More output than the limit is an oversized observation (this case fails
    on a candidate, the base is undecided): the process group is killed at
    the first byte past the limit, so a target cannot make the controller
    buffer its output (R3-S2). The stdin write shares the case deadline.
    """
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
        )
    except OSError as exc:
        return _Case("resolve", resolve="missing", detail=f"{call_text}: {exc}")
    output = CappedOutput(_CLI_OUTPUT_LIMIT)
    try:
        await asyncio.wait_for(
            asyncio.gather(_feed(process, stdin.encode("utf-8")), output.fill(process.stdout)),
            timeout=max(deadline - loop.time(), 0.01),
        )
        if output.overflow:
            return _Case("observed", entry={"outcome": "malformed", "call": call_text})
        await asyncio.wait_for(process.wait(), timeout=max(deadline - loop.time(), 0.01))
    except TimeoutError:
        return _Case("observed", entry={"outcome": "timeout", "call": call_text})
    finally:
        await _reap(process)
    return _Case(
        "observed",
        entry={
            "outcome": "exited",
            "call": call_text,
            "exit_code": process.returncode,
            "stdout": output.data.decode("utf-8", errors="replace"),
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
    arg_map = dict(binding.arg_map)
    observations: dict[str, dict[str, Any]] = {}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    cases = list(oracle.cases)
    for position, case in enumerate(cases):
        share = (deadline - loop.time()) / (len(cases) - position)
        if oracle.call_kind is CallKind.CLI:
            argv = _HARNESS["_cli_argv"](
                python, str(cwd), binding.symbol, list(oracle.params), arg_map, case.args
            )
            call_text = " ".join(argv[2:] if argv[0] == python else argv)
            script = binding.symbol
            if not script.startswith("-m ") and not (cwd / script).is_file():
                return "missing", f"{script} not found", {}, False
            outcome = await _cli_case(argv, cwd, env, case.stdin or "", share, call_text)
        else:
            args, kwargs = _HARNESS["_split_args"](list(oracle.params), arg_map, case.args)
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
        entry = dict(outcome.entry or {})
        if on_base:
            # A base run never counts an accident as the intended failure
            # (admission and the base run of a late binding).
            if outcome.kind == "observed" and entry.get("outcome") not in (
                "crashed",
                "timeout",
                "malformed",
            ):
                observations[case.case_id] = entry
                continue
            if outcome.kind == "observed":
                timed_out = entry["outcome"] == "timeout"
                reason = {"timeout": "target_timeout", "crashed": "target_crashed"}
                return reason.get(entry["outcome"], "frame_malformed"), "", {}, timed_out
            if position and outcome.kind == "resolve":
                # An earlier case resolved the same symbol: not a stable target.
                return "resolve_unstable", outcome.detail, {}, False
            return outcome.resolve, outcome.detail, {}, outcome.timed_out
        if outcome.kind == "observed":
            observations[case.case_id] = entry
        elif position == 0 and outcome.kind in ("launch_error", "resolve"):
            # launch_error: no target code has run yet (indeterminate).
            # missing / import_error: every case fails (decided).
            return outcome.resolve, outcome.detail, {}, False
        elif outcome.kind == "resolve":
            observations[case.case_id] = {
                "case_id": case.case_id,
                "outcome": "unresolved",
                "detail": f"{outcome.resolve}: {outcome.detail}",
            }
        else:
            # The target's code already ran (earlier case or this import):
            # a setup crash, hang, garbage frame, or launch failure fails
            # this case only.
            observations[case.case_id] = entry or {
                "case_id": case.case_id,
                "outcome": "crashed",
                "exit": None,
            }
    return "ok", "", observations, False


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


def _decided(resolve: str, *, on_base: bool) -> bool:
    if resolve in ("ok", "missing"):
        return True
    # On a candidate an import error is the candidate's code failing.
    return resolve == "import_error" and not on_base


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
) -> OracleRun:
    """Run one oracle check on the checkout copy at ``cwd`` (see the module docstring).

    Never raises: an unexpected controller error is an indeterminate check,
    never a verdict and never a reason to fall back to another verifier.
    """
    started = time.monotonic()
    harness = package_files.get(ORACLE_HARNESS_PATH)
    data_text = package_files.get(ORACLE_DATA_PATH)
    source = "declared" if binding is not None else "default"
    bound = binding or oracle.default_binding
    python = target_interpreter(interpreter, env)
    if harness is None or data_text is None:
        launch_error: str | None = "oracle files missing from the package"
    elif harness != ORACLE_HARNESS_SOURCE:
        launch_error = "oracle harness differs from this product version"
    elif python is None:
        launch_error = "no interpreter for the target"
    else:
        launch_error = None
    if launch_error is not None:
        return OracleRun(None, False, launch_error, launch_error, None, 0.0)
    assert harness is not None and data_text is not None and python is not None
    timed_out = False
    try:
        # Parsed before any target starts; held in memory only.
        oracle_data = json.loads(data_text)
        resolve, detail, observations, timed_out = await _observe(
            oracle,
            bound,
            harness,
            cwd,
            env,
            python,
            max(0.1, timeout_seconds),
            on_base=on_base,
        )
        result: dict[str, Any] | None = None
        if _decided(resolve, on_base=on_base):
            result = _HARNESS["_compare"](
                {
                    "oracle": oracle_data,
                    "check_id": oracle.check_id,
                    "binding": bound.to_dict() if binding is not None else None,
                    "resolve": resolve,
                    "detail": detail,
                    "observations": observations,
                }
            )
    except Exception as exc:  # noqa: BLE001 - a controller fault is indeterminate, never a verdict
        resolve, detail, result = "controller_error", type(exc).__name__, None
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
    )


__all__ = [
    "CappedOutput",
    "OracleRun",
    "parse_frame",
    "run_oracle_check",
    "target_interpreter",
    "valid_entry",
]
