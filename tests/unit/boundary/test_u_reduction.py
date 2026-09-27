"""A29 per-check admission, one replacement call, the behavioral filter, coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ouroboros.boundary.acceptance import (
    PackageCriterionStatus,
)
from ouroboros.boundary.behavioral import (
    NON_BEHAVIORAL,
    non_behavioral_criteria,
)
from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.constructor import (
    CheckConstructor,
    ConstructionOutcome,
    build_constructor_prompt,
    package_from_reply,
)
from ouroboros.boundary.coverage import why_excluded
from ouroboros.boundary.events import (
    ACTOR_STARTED,
    ADMISSION_COMPLETED,
    BOUNDARY_AGGREGATE_TYPE,
    CONSTRUCTION_FAILED,
    PACKAGE_FROZEN,
    SUPERSEDED,
)
from ouroboros.boundary.ledger import verify_boundary_order
from ouroboros.boundary.package import seed_criterion_keys
from ouroboros.boundary.per_check import (
    NO_ADMITTED_REPRODUCTION_CHECK,
    PRESERVATION_FAILS_ON_BASE,
    REPRO_PASSES_ON_BASE,
)
from ouroboros.boundary.rollout import Arm, AssignmentSource, CheckPackageAssignment
from ouroboros.boundary.run_control import CheckPackageRun
from ouroboros.boundary.run_wiring import (
    CheckPackageSettings,
    RegenerationPolicy,
    prepare_check_package,
    render_preparation,
    verify_check_package,
)
from ouroboros.core.seed import OntologySchema, Seed, SeedMetadata
from ouroboros.core.types import Result
from ouroboros.orchestrator.adapter import TaskResult
from ouroboros.persistence.event_store import EventStore

BUGGY = "def clamp(value, low, high):\n    if value > high:\n        return value\n    return max(low, value)\n"
FIXED = "def clamp(value, low, high):\n    return max(low, min(high, value))\n"


def _seed(*criteria: str) -> Seed:
    return Seed(
        goal="clamp helper",
        acceptance_criteria=criteria
        or (
            "clamp(15, 0, 10) returns 10",
            "clamp(5, 0, 10) returns 5",
            "clamp(-5, 0, 10) returns 0",
        ),
        ontology_schema=OntologySchema(name="mathutils", description="math helpers"),
        metadata=SeedMetadata(seed_id="seed_u_reduction", ambiguity_score=0.1),
    )


def _oracle(
    criterion: int, check_id: str, role: str, args: tuple[int, int, int], value: int
) -> dict[str, Any]:
    value_, low, high = args
    return {
        "criterion": criterion,
        "check_id": check_id,
        "role": role,
        "call_kind": "function",
        "params": ["value", "low", "high"],
        "default_binding": {"symbol": "mathutils.clamp"},
        "cases": [
            {
                "case_id": "stated",
                "args": {"value": value_, "low": low, "high": high},
                "expect": {"kind": "returns", "value": value},
            }
        ],
    }


# Base (BUGGY): clamp(15, 0, 10) = 15, clamp(5, 0, 10) = 5, clamp(-5, 0, 10) = 0.
GOOD_REPRO_1 = _oracle(1, "oracle_1", "reproduction", (15, 0, 10), 10)
BAD_REPRO_2 = _oracle(2, "oracle_2", "reproduction", (5, 0, 10), 5)  # passes on base
BAD_PRESERVE_3 = _oracle(3, "oracle_3", "preservation", (-5, 0, 10), 1)  # fails on base
GOOD_PRESERVE_3 = _oracle(3, "oracle_3", "preservation", (-5, 0, 10), 0)


class _Constructor:
    """``construct`` returns ``reply``; ``construct_replacements`` returns ``replacement``."""

    def __init__(
        self,
        seed: Seed,
        base: Path,
        reply: dict[str, Any],
        replacement: dict[str, Any] | None = None,
        *,
        replacement_fails: bool = False,
        replacements: bool = True,
    ) -> None:
        self.seed, self.base = seed, base
        self.reply, self.replacement = reply, replacement
        self.replacement_fails = replacement_fails
        self.construct_calls: list[dict[str, Any]] = []
        self.replacement_calls: list[dict[int, str]] = []
        if not replacements:
            self.construct_replacements = None  # type: ignore[assignment]

    def _outcome(self, reply: dict[str, Any]) -> ConstructionOutcome:
        package = package_from_reply(
            reply, self.seed, input_digest="1" * 64, generator="fake", base_checkout=self.base
        )
        return ConstructionOutcome(package, None, "1" * 64, "fake")

    async def construct(self, seed: Seed, base: Path, *, feedback=(), skip_criteria=()):
        self.construct_calls.append({"feedback": list(feedback), "skip": set(skip_criteria)})
        return self._outcome(self.reply)

    async def construct_replacements(self, seed: Seed, base: Path, *, targets):
        self.replacement_calls.append(dict(targets))
        if self.replacement_fails or self.replacement is None:
            return ConstructionOutcome(None, "constructor_timeout", "2" * 64, "fake")
        return self._outcome(self.replacement)


@pytest.fixture
async def store():
    event_store = EventStore("sqlite+aiosqlite:///:memory:")
    await event_store.initialize()
    yield event_store
    await event_store.close()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mathutils.py").write_text(BUGGY)
    return root


async def _prepare(
    store: EventStore,
    repo: Path,
    tmp_path: Path,
    constructor: Any,
    seed: Seed,
    *,
    policy: RegenerationPolicy = RegenerationPolicy.PRODUCT,
) -> Any:
    return await prepare_check_package(
        seed,
        event_store=store,
        constructor=constructor,
        execution_id="exec_u",
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=CheckPackageSettings(enabled=True, policy=policy),
        store_dir=tmp_path / "store",
    )


async def _types(store: EventStore, boundary_id: str) -> list[str]:
    return [event.type for event in await store.replay(BOUNDARY_AGGREGATE_TYPE, boundary_id)]


# ----------------------------------------------------------------------
# A29: per-check admission


async def test_each_exclusion_reason_excludes_only_its_check(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed()
    reply = {"oracles": [GOOD_REPRO_1, BAD_REPRO_2, BAD_PRESERVE_3]}
    state = await _prepare(
        store,
        repo,
        tmp_path,
        _Constructor(seed, repo, reply),
        seed,
        policy=RegenerationPolicy.STUDY,
    )
    assert state.admitted and state.boundary_id == "exec_u/check_package/v1"
    assert state.admission.excluded_checks == {
        "oracle_2": REPRO_PASSES_ON_BASE,
        "oracle_3": PRESERVATION_FAILS_ON_BASE,
    }
    assert state.admission.check_tiers == {"oracle_1": "A", "oracle_2": "C", "oracle_3": "C"}
    assert state.exclusions == (
        ("exec_u/check_package/v1", "oracle_2", REPRO_PASSES_ON_BASE),
        ("exec_u/check_package/v1", "oracle_3", PRESERVATION_FAILS_ON_BASE),
    )
    # The journal records ids and reasons, never values.
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
    admitted = next(event for event in events if event.type == ADMISSION_COMPLETED)
    assert admitted.data["verdict"] == "admitted"
    assert admitted.data["excluded_checks"] == state.admission.excluded_checks
    assert verify_boundary_order(events) == ()
    assert any("Checks excluded at admission" in line for line in render_preparation(state))
    # After the fix: the admitted check decides criterion 1, the others are uncovered.
    (repo / "mathutils.py").write_text(FIXED)
    verdict = await verify_check_package(
        state,
        event_store=store,
        candidate_checkout=repo,
        settings=CheckPackageSettings(enabled=True),
    )
    keys = seed_criterion_keys(seed)
    assert [(verdict.verdicts[k].status, verdict.verdicts[k].reason) for k in keys] == [
        (PackageCriterionStatus.PASS, "passed"),
        (PackageCriterionStatus.UNCOVERED, f"uncovered:{REPRO_PASSES_ON_BASE}"),
        (PackageCriterionStatus.UNCOVERED, f"uncovered:{PRESERVATION_FAILS_ON_BASE}"),
    ]
    # The excluded checks never ran on the candidate.
    ran = {
        check["check_id"]
        for check in _verified(await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id))
    }
    assert ran == {"oracle_1"}


def _verified(events: list[Any]) -> list[dict[str, Any]]:
    return next(e for e in events if e.type == "boundary.candidate.verified").data["checks"]


async def test_a_criterion_keeps_authority_while_one_admitted_check_covers_it(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed("clamp(15, 0, 10) returns 10")
    extra = _oracle(1, "oracle_1b", "reproduction", (5, 0, 10), 5)  # passes on base
    state = await _prepare(
        store,
        repo,
        tmp_path,
        _Constructor(seed, repo, {"oracles": [GOOD_REPRO_1, extra]}),
        seed,
    )
    assert state.admission.excluded_checks == {"oracle_1b": REPRO_PASSES_ON_BASE}
    assert state.replacement_calls == 0  # nothing left without an admitted check
    (repo / "mathutils.py").write_text(FIXED)
    verdict = await verify_check_package(
        state, event_store=store, candidate_checkout=repo, settings=CheckPackageSettings(True)
    )
    (item,) = verdict.verdicts.values()
    assert item.status is PackageCriterionStatus.PASS and item.tier is CheckTier.A


async def test_losing_the_last_reproduction_check_leaves_the_criterion_uncovered(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed("clamp(5, 0, 10) returns 5")
    repro = _oracle(1, "oracle_r", "reproduction", (5, 0, 10), 5)  # passes on base
    keep = _oracle(1, "oracle_p", "preservation", (-5, 0, 10), 0)  # passes on base
    state = await _prepare(
        store,
        repo,
        tmp_path,
        _Constructor(seed, repo, {"oracles": [repro, keep]}),
        seed,
        policy=RegenerationPolicy.STUDY,
    )
    assert state.admitted
    assert state.admission.excluded_checks == {"oracle_r": REPRO_PASSES_ON_BASE}
    (repo / "mathutils.py").write_text(FIXED)
    verdict = await verify_check_package(
        state, event_store=store, candidate_checkout=repo, settings=CheckPackageSettings(True)
    )
    (item,) = verdict.verdicts.values()
    assert item.status is PackageCriterionStatus.UNCOVERED
    assert item.reason == f"uncovered:{NO_ADMITTED_REPRODUCTION_CHECK}"


# ----------------------------------------------------------------------
# One replacement call (product policy)


async def test_one_replacement_call_supersedes_the_version_before_dispatch(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed()
    constructor = _Constructor(
        seed,
        repo,
        {"oracles": [GOOD_REPRO_1, BAD_REPRO_2, GOOD_PRESERVE_3]},
        {"oracles": [_oracle(2, "r2_repro", "reproduction", (20, 0, 10), 10)]},
    )
    state = await _prepare(store, repo, tmp_path, constructor, seed)
    assert constructor.replacement_calls == [
        {2: "its reproduction check passes on the base code, so it does not reproduce the bug"}
    ]
    assert state.boundary_id == "exec_u/check_package/v2"
    assert state.replacement_calls == 1 and state.replacement_outcome == "admitted"
    assert state.admitted and state.admission.excluded_checks is None
    assert {check.check_id for check in state.package.checks} == {
        "oracle_1",
        "r2_repro",
        "oracle_3",
    }
    v1 = await store.replay(BOUNDARY_AGGREGATE_TYPE, "exec_u/check_package/v1")
    v2 = await store.replay(BOUNDARY_AGGREGATE_TYPE, "exec_u/check_package/v2")
    assert [e.type for e in v1] == [PACKAGE_FROZEN, ADMISSION_COMPLETED, SUPERSEDED]
    assert [e.type for e in v2] == [PACKAGE_FROZEN, ADMISSION_COMPLETED, ACTOR_STARTED]
    superseded = v1[-1]
    assert superseded.data["superseded_by"] == "exec_u/check_package/v2"
    assert superseded.data["reason"] == "replacement_checks"
    # The new commitment is recorded before the worker start.
    assert v2[0].timestamp < v2[-1].timestamp
    assert "package_commitment" in v2[0].data
    assert verify_boundary_order(v1) == () and verify_boundary_order(v2) == ()
    (repo / "mathutils.py").write_text(FIXED)
    verdict = await verify_check_package(
        state, event_store=store, candidate_checkout=repo, settings=CheckPackageSettings(True)
    )
    assert all(v.status is PackageCriterionStatus.PASS for v in verdict.verdicts.values())
    meta = await _meta(state)
    assert meta["replacement_call_count"] == "1" and meta["excluded_check_count"] == "1"


async def test_a_failed_replacement_call_keeps_the_admitted_version(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed()
    constructor = _Constructor(
        seed,
        repo,
        {"oracles": [GOOD_REPRO_1, BAD_REPRO_2, GOOD_PRESERVE_3]},
        replacement_fails=True,
    )
    state = await _prepare(store, repo, tmp_path, constructor, seed)
    assert len(constructor.replacement_calls) == 1
    assert state.boundary_id == "exec_u/check_package/v1" and state.admitted
    assert state.replacement_calls == 1 and state.replacement_outcome == "construction_failed"
    assert await _types(store, "exec_u/check_package/v2") == [CONSTRUCTION_FAILED, SUPERSEDED]
    v2 = await store.replay(BOUNDARY_AGGREGATE_TYPE, "exec_u/check_package/v2")
    assert v2[-1].data["superseded_by"] == "exec_u/check_package/v1"
    assert await _types(store, "exec_u/check_package/v1") == [
        PACKAGE_FROZEN,
        ADMISSION_COMPLETED,
        ACTOR_STARTED,
    ]
    assert any("v2" in line for line in render_preparation(state) if "Superseded" in line)


async def test_the_study_policy_never_regenerates(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed()
    constructor = _Constructor(
        seed,
        repo,
        {"oracles": [GOOD_REPRO_1, BAD_REPRO_2, GOOD_PRESERVE_3]},
        {"oracles": [_oracle(2, "r2_repro", "reproduction", (20, 0, 10), 10)]},
    )
    state = await _prepare(
        store, repo, tmp_path, constructor, seed, policy=RegenerationPolicy.STUDY
    )
    assert constructor.replacement_calls == []
    assert state.replacement_calls == 0 and state.boundary_id == "exec_u/check_package/v1"


async def test_the_replacement_call_is_one_single_call_for_the_targets_only(
    repo: Path,
) -> None:
    from ouroboros.boundary.constructor import CHECK_DIR

    class _Runtime:
        _runtime_backend = "codex"
        _exec_session_flags: tuple[str, ...] = ()

        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def execute_task_to_result(self, prompt: str, tools=None, system_prompt=None):
            self.prompts.append(prompt)
            reply = {
                "oracles": [
                    _oracle(1, "r1_other", "reproduction", (15, 0, 10), 10),
                    _oracle(2, "r2_repro", "reproduction", (20, 0, 10), 10),
                ]
            }
            return Result.ok(TaskResult(success=True, final_message=json.dumps(reply), messages=()))

    runtime = _Runtime()
    constructor = CheckConstructor(
        runtime_backend="codex",
        model="gpt-test",
        runtime_factory=lambda **_kwargs: runtime,
        system_prompt="SYSTEM",
    )  # per_criterion by default: the replacement is still one call
    seed = _seed()
    why = why_excluded(REPRO_PASSES_ON_BASE)
    outcome = await constructor.construct_replacements(seed, repo, targets={2: why})
    assert len(runtime.prompts) == 1
    assert f"- criterion 2: {why}" in runtime.prompts[0]
    assert "5, 0, 10" not in runtime.prompts[0].split("no admitted check")[1]
    assert outcome.package is not None
    assert [check.check_id for check in outcome.package.checks] == ["r2_repro"]
    assert not (repo / CHECK_DIR).exists()


# ----------------------------------------------------------------------
# Behavioral filter


async def test_a_non_behavioral_criterion_gets_no_check_and_no_call(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed(
        "clamp(15, 0, 10) returns 10", "The user is willing to assist with debugging the issue."
    )
    assert non_behavioral_criteria(seed) == (1,)
    # The constructor wrote a check for the prose criterion anyway: it is removed.
    stray = _oracle(2, "oracle_2", "preservation", (-5, 0, 10), 0)
    constructor = _Constructor(seed, repo, {"oracles": [GOOD_REPRO_1, stray]})
    state = await _prepare(store, repo, tmp_path, constructor, seed)
    assert constructor.construct_calls[0]["skip"] == {2}
    keys = seed_criterion_keys(seed)
    assert state.non_behavioral == (keys[1],)
    assert [check.check_id for check in state.package.checks] == ["oracle_1"]
    assert {item.criterion_key: item.reason for item in state.package.uncovered} == {
        keys[1]: NON_BEHAVIORAL
    }
    assert constructor.replacement_calls == []  # never a replacement target
    assert any("Non-behavioral criteria" in line for line in render_preparation(state))
    assert "criterion 2" not in build_constructor_prompt(seed).split("Acceptance")[0]
    assert f'reason "{NON_BEHAVIORAL}"' in build_constructor_prompt(seed, skip={2})
    assert f'reason "{NON_BEHAVIORAL}"' not in build_constructor_prompt(seed)


async def test_a_seed_with_only_non_behavioral_criteria_calls_no_constructor(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed("The user is willing to assist with debugging the issue.")
    constructor = _Constructor(seed, repo, {"oracles": []})
    state = await _prepare(store, repo, tmp_path, constructor, seed)
    assert constructor.construct_calls == []
    assert not state.admitted and state.failure_reason == "all_criteria_non_behavioral"
    assert await _types(store, "exec_u/check_package/v1") == [CONSTRUCTION_FAILED, ACTOR_STARTED]


# ----------------------------------------------------------------------
# Coverage flag and flag-off identity


async def test_the_arm_off_sends_none_of_the_new_properties() -> None:
    run = CheckPackageRun(
        CheckPackageSettings(
            enabled=False,
            assignment=CheckPackageAssignment(Arm.OFF, AssignmentSource.USER_FORCED_OFF),
        )
    )

    class _NoEvents:
        async def query_events(self, **_kwargs: Any) -> list[Any]:
            return []

    meta = await run.outcome_meta(
        _NoEvents(), execution_id="e", session_id="s", terminal_status="completed"
    )  # type: ignore[arg-type]
    assert not {
        "non_behavioral_count",
        "excluded_check_count",
        "replacement_call_count",
        "verification_coverage",
    } & set(meta)
    assert meta["reconciliation"] == "none"
    assert run.render_outcome() == []


async def _meta(state: Any) -> dict[str, str]:
    run = CheckPackageRun(
        CheckPackageSettings(
            enabled=True,
            assignment=CheckPackageAssignment(Arm.ON, AssignmentSource.USER_FORCED_ON),
        ),
        state=state,
        attempted=True,
    )

    class _NoEvents:
        async def query_events(self, **_kwargs: Any) -> list[Any]:
            return []

    return await run.outcome_meta(
        _NoEvents(), execution_id="exec_u", session_id="s", terminal_status="completed"
    )  # type: ignore[arg-type]
