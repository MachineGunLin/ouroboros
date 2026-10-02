"""Issue #2441: plugin dispatch drops checkpoint/commit params.

`build_evolve_subagent` must forward the checkpoint contract
(commit_policy, auto_session_id, execution_id, checkpoint_commits,
checkpoint_attempted_ac_ids) so the plugin path matches the in-process
path. `recover_expired_claim=true` is rejected on the start_evolve_step
plugin path (fail-closed, like benchmark_control) instead of being
silently dropped from the child payload.
"""

import pytest

from ouroboros.mcp.tools.subagent import build_evolve_subagent


def _checkpoint_kwargs() -> dict:
    return {
        "commit_policy": "on-pass",
        "auto_session_id": "sess-1",
        "execution_id": "exec-1",
        "checkpoint_commits": ({"ac": "a1"},),
        "checkpoint_attempted_ac_ids": ("a1", "a2"),
    }


class TestBuildEvolveSubagentCheckpointContract:
    def test_forwards_five_params_into_context(self) -> None:
        payload = build_evolve_subagent(lineage_id="lin-1", **_checkpoint_kwargs())
        assert payload.context["commit_policy"] == "on-pass"
        assert payload.context["auto_session_id"] == "sess-1"
        assert payload.context["execution_id"] == "exec-1"
        assert payload.context["checkpoint_commits"] == [{"ac": "a1"}]
        assert payload.context["checkpoint_attempted_ac_ids"] == ["a1", "a2"]

    def test_checkpoint_note_in_prompt(self) -> None:
        payload = build_evolve_subagent(lineage_id="lin-1", **_checkpoint_kwargs())
        assert "Checkpoint" in payload.prompt
        assert "commit_policy" in payload.prompt

    def test_legacy_shape_unchanged_without_checkpoint_params(self) -> None:
        payload = build_evolve_subagent(lineage_id="lin-2")
        assert payload.context == {
            "lineage_id": "lin-2",
            "seed_content": None,
            "execute": True,
            "parallel": True,
            "skip_qa": False,
            "project_dir": None,
        }
        assert "Checkpoint" not in payload.prompt

    def test_lists_only_without_policy_or_session(self) -> None:
        payload = build_evolve_subagent(
            lineage_id="lin-3",
            checkpoint_commits=({"ac": "a1"},),
            checkpoint_attempted_ac_ids=("a1",),
        )
        assert payload.context["checkpoint_commits"] == [{"ac": "a1"}]
        assert payload.context["checkpoint_attempted_ac_ids"] == ["a1"]
        assert "commit_policy" not in payload.context
        assert "auto_session_id" not in payload.context

    def test_execution_id_only(self) -> None:
        payload = build_evolve_subagent(lineage_id="lin-4", execution_id="exec-9")
        assert payload.context["execution_id"] == "exec-9"
        assert "commit_policy" not in payload.context

    def test_auto_session_id_only(self) -> None:
        payload = build_evolve_subagent(lineage_id="lin-5", auto_session_id="sess-9")
        assert payload.context["auto_session_id"] == "sess-9"
        assert "commit_policy" not in payload.context

    def test_commit_policy_only(self) -> None:
        # All five params are required=False in the schema, so a lone
        # commit_policy is legal input and must be preserved.
        payload = build_evolve_subagent(lineage_id="lin-6", commit_policy="on-pass")
        assert payload.context["commit_policy"] == "on-pass"

    def test_commit_policy_none_keeps_legacy_shape(self) -> None:
        payload = build_evolve_subagent(lineage_id="lin-7", commit_policy="none")
        assert "commit_policy" not in payload.context
        assert "Checkpoint" not in payload.prompt


class TestStartEvolveStepPluginCheckpoint:
    @pytest.fixture
    async def event_store(self):
        from ouroboros.persistence.event_store import EventStore

        store = EventStore("sqlite+aiosqlite:///:memory:")
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def handler(self, event_store):
        from unittest.mock import MagicMock

        from ouroboros.mcp.tools.evolution_handlers import StartEvolveStepHandler

        return StartEvolveStepHandler(
            evolve_handler=MagicMock(),
            event_store=event_store,
            job_manager=None,
            agent_runtime_backend="opencode",
            opencode_mode="plugin",
        )

    async def test_plugin_payload_retains_checkpoint_contract(self, handler) -> None:
        result = await handler.handle(
            {
                "lineage_id": "lin-abc",
                "commit_policy": "on-pass",
                "auto_session_id": "sess-1",
                "execution_id": "exec-1",
                "checkpoint_commits": [{"ac": "a1"}, {"ac": "a2"}],
                "checkpoint_attempted_ac_ids": ["a1", "a2", "a3"],
            }
        )
        assert result.is_ok
        context = result.value.meta["_subagent"]["context"]
        assert context["commit_policy"] == "on-pass"
        assert context["auto_session_id"] == "sess-1"
        assert context["execution_id"] == "exec-1"
        assert context["checkpoint_commits"] == [{"ac": "a1"}, {"ac": "a2"}]
        assert context["checkpoint_attempted_ac_ids"] == ["a1", "a2", "a3"]

    async def test_recover_expired_claim_rejected_on_plugin_path(self, handler) -> None:
        result = await handler.handle({"lineage_id": "lin-abc", "recover_expired_claim": True})
        assert result.is_ok is False
        assert "recover_expired_claim" in str(result.error)

    async def test_benchmark_control_rejection_still_holds(self, handler) -> None:
        result = await handler.handle({"lineage_id": "lin-abc", "benchmark_control": True})
        assert result.is_ok is False
        assert "benchmark_control" in str(result.error)

    async def test_lists_only_reach_plugin_context(self, handler) -> None:
        result = await handler.handle(
            {
                "lineage_id": "lin-abc",
                "checkpoint_commits": [{"ac": "a1"}],
                "checkpoint_attempted_ac_ids": ["a1"],
            }
        )
        assert result.is_ok
        context = result.value.meta["_subagent"]["context"]
        assert context["checkpoint_commits"] == [{"ac": "a1"}]
        assert context["checkpoint_attempted_ac_ids"] == ["a1"]


class TestEvolveStepPluginCheckpoint:
    @pytest.fixture
    async def event_store(self):
        from ouroboros.persistence.event_store import EventStore

        store = EventStore("sqlite+aiosqlite:///:memory:")
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def handler(self, event_store):
        from ouroboros.mcp.tools.evolution_handlers import EvolveStepHandler

        return EvolveStepHandler(
            evolutionary_loop=None,
            event_store=event_store,
            agent_runtime_backend="opencode",
            opencode_mode="plugin",
        )

    async def test_plugin_payload_retains_checkpoint_contract(self, handler) -> None:
        result = await handler.handle(
            {
                "lineage_id": "lin-def",
                "commit_policy": "on-pass",
                "auto_session_id": "sess-9",
                "execution_id": "exec-9",
                "checkpoint_commits": [{"ac": "b1"}],
                "checkpoint_attempted_ac_ids": ["b1"],
            }
        )
        assert result.is_ok
        context = result.value.meta["_subagent"]["context"]
        assert context["commit_policy"] == "on-pass"
        assert context["auto_session_id"] == "sess-9"
        assert context["execution_id"] == "exec-9"
        assert context["checkpoint_commits"] == [{"ac": "b1"}]
        assert context["checkpoint_attempted_ac_ids"] == ["b1"]
