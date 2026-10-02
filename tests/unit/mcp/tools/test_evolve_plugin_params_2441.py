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

    def test_prompt_carries_real_values_not_just_names(self) -> None:
        # Bridge-visible contract: the OpenCode bridge drops
        # SubagentPayload.context, so the child can only recover values
        # present verbatim in the prompt.
        payload = build_evolve_subagent(
            lineage_id="lin-v",
            commit_policy="on-pass",
            auto_session_id="sess-v",
            execution_id="exec-v",
            checkpoint_commits=({"ac": "a1"}, {"ac": "a2"}),
            checkpoint_attempted_ac_ids=("a1", "a2", "a3"),
        )
        assert "commit_policy: on-pass" in payload.prompt
        assert "auto_session_id: sess-v" in payload.prompt
        assert "execution_id: exec-v" in payload.prompt
        assert '"ac": "a1"' in payload.prompt
        assert '"ac": "a2"' in payload.prompt
        for ac_id in ("a1", "a2", "a3"):
            assert f'"{ac_id}"' in payload.prompt

    def test_absent_fields_add_no_prompt_noise(self) -> None:
        payload = build_evolve_subagent(lineage_id="lin-8", execution_id="exec-only")
        assert "execution_id: exec-only" in payload.prompt
        assert "commit_policy" not in payload.prompt
        assert "auto_session_id" not in payload.prompt
        assert "checkpoint_commits" not in payload.prompt
        assert "checkpoint_attempted_ac_ids" not in payload.prompt

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


class TestCheckpointStateOversize:
    """P1: checkpoint state must never be silently truncated.

    Full JSON is rendered regardless of size; oversize state fails closed
    at the handler gate (plugin path only) instead of dispatching partial
    idempotency state to the child.
    """

    def test_big_ac_text_keeps_hash_and_later_records(self) -> None:
        from ouroboros.mcp.tools.subagent import render_checkpoint_note

        big_text = "x" * 7000
        note = render_checkpoint_note(
            commit_policy="on-pass",
            auto_session_id="sess-big",
            checkpoint_commits=(
                {"ac_id": "a1", "ac_text": big_text, "commit": "deadbeef"},
                {"ac_id": "a2", "commit": "cafef00d"},
            ),
            checkpoint_attempted_ac_ids=("a1", "a2"),
        )
        assert "deadbeef" in note
        assert "cafef00d" in note
        assert big_text in note
        assert '"a2"' in note

    def test_capacity_gate_passes_normal_state(self) -> None:
        from ouroboros.mcp.tools.subagent import check_checkpoint_state_capacity

        assert (
            check_checkpoint_state_capacity(
                commit_policy="on-pass",
                checkpoint_commits=({"ac": "a1"},),
                checkpoint_attempted_ac_ids=("a1",),
            )
            is None
        )

    def test_capacity_gate_fails_closed_on_big_state(self) -> None:
        from ouroboros.mcp.tools.subagent import check_checkpoint_state_capacity

        error = check_checkpoint_state_capacity(
            checkpoint_attempted_ac_ids=tuple(f"ac-{i:05d}" for i in range(3000)),
        )
        assert error is not None
        assert "plugin transport/prompt capacity" in error


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

    async def test_oversize_state_fails_closed_without_dispatch(self, handler) -> None:
        result = await handler.handle(
            {
                "lineage_id": "lin-abc",
                "checkpoint_attempted_ac_ids": [f"ac-{i:05d}" for i in range(3000)],
            }
        )
        assert result.is_ok is False
        assert "plugin transport/prompt capacity" in str(result.error)

    async def test_multi_record_values_reach_plugin_prompt(self, handler) -> None:
        big_text = "y" * 7000
        result = await handler.handle(
            {
                "lineage_id": "lin-abc",
                "commit_policy": "on-pass",
                "auto_session_id": "sess-m",
                "checkpoint_commits": [
                    {"ac_id": "a1", "ac_text": big_text, "commit": "deadbeef"},
                    {"ac_id": "a2", "commit": "cafef00d"},
                ],
                "checkpoint_attempted_ac_ids": ["a1", "a2"],
            }
        )
        assert result.is_ok
        prompt = result.value.meta["_subagent"]["prompt"]
        assert "deadbeef" in prompt
        assert "cafef00d" in prompt
        assert big_text in prompt


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
