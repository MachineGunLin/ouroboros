"""The constructor's model call keeps no session on disk (held-out values at rest).

The constructor's reply holds every held-out case. The Codex call must carry
``--ephemeral`` and the Claude call ``--no-session-persistence``; any other
runtime is refused before a model call.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from ouroboros.boundary.constructor import CheckConstructor
from ouroboros.boundary.constructor_session import (
    NOT_EPHEMERAL_PREFIX,
    disable_session_persistence,
)
from ouroboros.orchestrator.adapter import ClaudeAgentAdapter
from ouroboros.orchestrator.codex_cli_runtime import CodexCliRuntime
from ouroboros.orchestrator.copilot_cli_runtime import CopilotCliRuntime
from tests.unit.boundary.test_constructor import FakeRuntime, _reply, _seed


def _codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CodexCliRuntime:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    cli_path = tmp_path / "codex"
    cli_path.write_text("#!/bin/sh\necho codex 1.0\n", encoding="utf-8")
    cli_path.chmod(0o755)
    return CodexCliRuntime(cli_path=cli_path, cwd=str(tmp_path), model="gpt-6-luna")


def test_the_codex_constructor_call_runs_codex_exec_ephemeral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _codex(tmp_path, monkeypatch)
    assert "--ephemeral" not in runtime._build_command(str(tmp_path / "last"))
    assert disable_session_persistence(runtime) is None
    command = runtime._build_command(str(tmp_path / "last"))
    assert command[1:3] == ["exec", "--ephemeral"]
    # Other Codex runtimes (workers) are unaffected.
    assert "--ephemeral" not in _codex(tmp_path / "w", monkeypatch)._build_command("/tmp/x")


def _mock_claude_sdk(options_sink: list[dict[str, Any]]) -> dict[str, ModuleType]:
    module = ModuleType("claude_agent_sdk")

    class _Options:
        def __init__(self, **kwargs: Any) -> None:
            options_sink.append(kwargs)

    class _HookMatcher:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    async def query(*, prompt: str, options: Any):
        result = type("ResultMessage", (), {})()
        result.result, result.subtype = "ok", "success"
        yield result

    types_module = ModuleType("claude_agent_sdk.types")
    types_module.HookMatcher = _HookMatcher  # type: ignore[attr-defined]
    module.ClaudeAgentOptions = _Options  # type: ignore[attr-defined]
    module.query = query  # type: ignore[attr-defined]
    module.types = types_module  # type: ignore[attr-defined]
    return {"claude_agent_sdk": module, "claude_agent_sdk.types": types_module}


async def test_the_claude_constructor_call_disables_session_persistence() -> None:
    options: list[dict[str, Any]] = []
    plain = ClaudeAgentAdapter(api_key="test", cwd="/tmp/project")
    switched = ClaudeAgentAdapter(api_key="test", cwd="/tmp/project")
    assert disable_session_persistence(switched) is None
    with patch.dict("sys.modules", _mock_claude_sdk(options)):
        _ = [message async for message in plain.execute_task("hi")]
        _ = [message async for message in switched.execute_task("hi")]
    assert "extra_args" not in options[0]
    assert options[1]["extra_args"] == {"no-session-persistence": None}


def test_a_runtime_without_a_no_persistence_mode_is_refused(tmp_path: Path) -> None:
    copilot = object.__new__(CopilotCliRuntime)  # a Codex-family CLI without --ephemeral
    assert disable_session_persistence(copilot) == f"{NOT_EPHEMERAL_PREFIX}:copilot"
    assert disable_session_persistence(object()) == f"{NOT_EPHEMERAL_PREFIX}:object"


class _PersistingRuntime(FakeRuntime):
    """A runtime with no way to keep its session off disk."""

    _runtime_backend = "gemini"


@pytest.mark.parametrize("per_criterion", [False, True])
async def test_the_constructor_switches_the_runtime_or_makes_no_call(
    tmp_path: Path, per_criterion: bool
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "calc.py").write_text("def add(a, b):\n    return a - b\n")

    def constructor(runtime: FakeRuntime) -> CheckConstructor:
        return CheckConstructor(
            runtime_backend="codex",
            model="gpt-test",
            runtime_factory=lambda **_: runtime,
            system_prompt="SYSTEM",
            per_criterion=per_criterion,
        )

    codex_like = FakeRuntime(_reply())
    outcome = await constructor(codex_like).construct(_seed(), base)
    assert codex_like.calls and codex_like._exec_session_flags == ("--ephemeral",)
    assert outcome.package is not None or per_criterion  # per-criterion replies differ

    persisting = _PersistingRuntime(_reply())
    refused = await constructor(persisting).construct(_seed(), base)
    assert persisting.calls == []  # no model call at all
    assert refused.package is None
    assert f"{NOT_EPHEMERAL_PREFIX}:gemini" in (refused.failure_reason or "")
