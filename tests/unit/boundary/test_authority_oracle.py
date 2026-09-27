"""Check package on: tiers decide, the legacy verifier annotates, repairs follow the package."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ouroboros.boundary.acceptance import PackageCriterionStatus
from ouroboros.boundary.authority import (
    CheckPackageAuthority,
    apply_legacy_fallback,
    existing_outcomes_from_results,
    legacy_verdict_in_tree,
)
from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.constructor import ConstructionOutcome, package_from_reply
from ouroboros.boundary.package import seed_criterion_keys, verify_commitment
from ouroboros.boundary.rollout import Arm, AssignmentSource, CheckPackageAssignment
from ouroboros.boundary.run_control import CheckPackageRun, tier_summary_value
from ouroboros.boundary.run_wiring import (
    CheckPackageSettings,
    controller_private_dir,
    prepare_check_package,
)
from ouroboros.core.seed import OntologySchema, Seed, SeedMetadata
from ouroboros.orchestrator.evidence_schema import EvidenceRecord
from ouroboros.orchestrator.parallel_executor import ParallelACExecutor
from ouroboros.orchestrator.parallel_executor_models import (
    ACExecutionOutcome,
    ACExecutionResult,
    ParallelExecutionResult,
)
from ouroboros.orchestrator.verifier import VerifierVerdict
from ouroboros.persistence.event_store import EventStore
from ouroboros.telemetry import _check_package_properties

BUGGY = "def clamp(value, low, high):\n    if value > high:\n        return value\n    return max(low, value)\n"
FIXED = "def clamp(value, low, high):\n    return max(low, min(high, value))\n"
GOOD_MIX = "\ndef mix(start, end, weight):\n    return start + (end - start) * weight\n"
BAD_MIX = "\ndef mix(start, end, weight):\n    return start + end - weight\n"
MIX_ENTRY = {"symbol": "mathutils.mix", "arg_map": {"a": "start", "b": "end", "t": "weight"}}


def _seed() -> Seed:
    return Seed(
        goal="math helpers",
        acceptance_criteria=(
            "clamp(15, 0, 10) returns 10",
            "linear interpolation between a and b by t: interpolating 0 and 10 at 0.5 gives 5",
            "the helpers are documented in the README",
        ),
        ontology_schema=OntologySchema(name="mathutils", description="math helpers"),
        metadata=SeedMetadata(seed_id="seed_authority_oracle", ambiguity_score=0.1),
    )


REPLY = {
    "oracles": [
        {
            "criterion": 1,
            "check_id": "oracle_1",
            "role": "reproduction",
            "call_kind": "function",
            "params": ["value", "low", "high"],
            "default_binding": {"symbol": "mathutils.clamp"},
            "cases": [
                {
                    "case_id": "stated",
                    "args": {"value": 15, "low": 0, "high": 10},
                    "expect": {"kind": "returns", "value": 10},
                },
                {
                    "case_id": "held",
                    "args": {"value": -3, "low": -2, "high": 4},
                    "expect": {"kind": "returns", "value": -2},
                },
            ],
        },
        {
            "criterion": 2,
            "check_id": "oracle_2",
            "role": "reproduction",
            "call_kind": "function",
            "params": ["a", "b", "t"],
            "default_binding": {"symbol": "mathutils.interpolate"},
            "cases": [
                {
                    "case_id": "stated",
                    "args": {"a": 0, "b": 10, "t": 0.5},
                    "expect": {"kind": "returns", "value": 5, "approx": 1e-9},
                },
                {
                    "case_id": "held",
                    "args": {"a": 2, "b": 4, "t": 0.25},
                    "expect": {"kind": "returns", "value": 2.5, "approx": 1e-9},
                },
            ],
        },
    ],
    # The reason is descriptive text; it routes nothing (criterion 3 has no
    # admitted check, so the legacy verifier decides it).
    "uncovered": [{"criterion": 3, "reason": "non_behavioral"}],
}


class _Constructor:
    def __init__(self, seed: Seed, base: Path) -> None:
        package = package_from_reply(
            REPLY, seed, input_digest="1" * 64, generator="fake", base_checkout=base
        )
        self.outcome = ConstructionOutcome(package, None, "1" * 64, "fake")

    async def construct(self, seed: Seed, base: Path, *, feedback=()) -> ConstructionOutcome:
        return self.outcome


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


async def _authority(
    store: EventStore, repo: Path, tmp_path: Path
) -> tuple[Seed, CheckPackageAuthority]:
    seed = _seed()
    settings = CheckPackageSettings(enabled=True)
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=_Constructor(seed, repo),
        execution_id="exec_oracle",
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=settings,
        store_dir=tmp_path / "store",
    )
    return seed, CheckPackageAuthority(state, settings, event_store=store, candidate_checkout=repo)


def _legacy_rejected(index: int, *, entry: dict | None = None) -> ACExecutionResult:
    """What the leaf returns with the gate installed: success, legacy rejection kept as annotation."""
    return ACExecutionResult(
        ac_index=index,
        ac_content=f"criterion {index}",
        success=True,
        outcome=ACExecutionOutcome.SUCCEEDED,
        atomic_verifier_verdict=VerifierVerdict(
            passed=False, reasons=("form",), failure_class="EVIDENCE_FORM_MISMATCH"
        ),
        # The executor keeps the rejection it made advisory (gate installed).
        legacy_rejection="legacy verifier: evidence form mismatch",
        typed_evidence=EvidenceRecord(data={"entry_points": [entry]} if entry else {}),
    )


def _transcript_unavailable(index: int) -> ACExecutionResult:
    """What the leaf returns when the transcript could not be collected: no rejection."""
    return ACExecutionResult(
        ac_index=index,
        ac_content=f"criterion {index}",
        success=True,
        outcome=ACExecutionOutcome.SUCCEEDED,
        atomic_verifier_verdict=VerifierVerdict(
            passed=False,
            reasons=("transcript_missing_infrastructure: runtime support messages were empty",),
            failure_class="TRANSCRIPT_MISSING_INFRASTRUCTURE",
        ),
    )


def _executor(repo: Path, retries: int = 2) -> ParallelACExecutor:
    adapter = MagicMock()
    adapter.working_directory = str(repo)
    adapter.runtime_backend = "claude"
    return ParallelACExecutor(
        adapter=adapter,
        event_store=AsyncMock(),
        console=MagicMock(),
        enable_decomposition=False,
        run_verify_commands=False,
        ac_retry_attempts=retries,
    )


async def _batch(executor: ParallelACExecutor, seed: Seed, indices: list[int]) -> list[Any]:
    return await executor._run_batch_with_verify_and_retry(
        seed=seed,
        batch_executable=indices,
        session_id="s",
        execution_id="exec_oracle",
        tools=[],
        tool_catalog=None,
        system_prompt="sys",
        level_contexts=[],
        ac_retry_attempts=dict.fromkeys(indices, 0),
        execution_counters=None,
    )


async def test_no_legacy_triggered_retries_when_the_flag_is_on(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + GOOD_MIX)
    executor = _executor(repo)
    authority.install(executor)
    calls: list[list[int]] = []

    async def fake_batch(**kwargs: Any) -> list[ACExecutionResult]:
        calls.append(list(kwargs["batch_indices"]))
        return [_legacy_rejected(0)]

    executor._execute_ac_batch = fake_batch  # type: ignore[method-assign]
    results = await _batch(executor, seed, [0])
    # The legacy verifier rejected the attempt, the package passed it: one dispatch.
    assert calls == [[0]] and results[0].success is True
    assert authority.gate.log == [{"ac_index": 0, "status": "pass", "tier": "A"}]


async def test_flag_off_legacy_rejection_still_retries(repo: Path) -> None:
    executor = _executor(repo)
    calls: list[list[int]] = []

    async def fake_batch(**kwargs: Any) -> list[ACExecutionResult]:
        calls.append(list(kwargs["batch_indices"]))
        cls = ["EVIDENCE_MISSING", "STALL", "SCOPE_CREEP"][len(calls) - 1]
        return [
            ACExecutionResult(
                ac_index=0,
                ac_content="c",
                success=False,
                error="legacy",
                atomic_verifier_verdict=VerifierVerdict(
                    passed=False, reasons=("legacy",), failure_class=cls
                ),
            )
        ]

    executor._execute_ac_batch = fake_batch  # type: ignore[method-assign]
    await _batch(executor, _seed(), [0])
    assert calls == [[0], [0], [0]]
    assert not hasattr(executor, "check_package_gate")


async def test_package_fail_drives_repair_and_names_the_declared_binding(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + BAD_MIX)
    executor = _executor(repo)
    authority.install(executor)
    prompts: list[dict[int, str]] = []

    async def fake_batch(**kwargs: Any) -> list[ACExecutionResult]:
        prompts.append(dict(kwargs.get("retry_prompts") or {}))
        if len(prompts) == 2:
            (repo / "mathutils.py").write_text(FIXED + GOOD_MIX)  # the repair
        result = replace(_legacy_rejected(1, entry=MIX_ENTRY), retry_attempt=len(prompts) - 1)
        return [result]

    executor._execute_ac_batch = fake_batch  # type: ignore[method-assign]
    results = await _batch(executor, seed, [1])
    assert len(prompts) == 2 and results[0].success is True
    repair = prompts[1][1]
    assert "### Check package counterexample" in repair
    assert "declared entry point: function mathutils.mix" in repair
    assert "mix(start=0, end=10, weight=0.5): expected 5, observed 9.5" in repair
    assert "held-out case(s) also failed" in repair and "2.5" not in repair
    assert "EVIDENCE_FORM_MISMATCH" not in repair  # the legacy class drives nothing
    assert [entry["status"] for entry in authority.gate.log] == ["fail", "pass"]
    assert authority.gate.log[0]["tier"] == "A_prime"


def _legacy_accepted(index: int) -> ACExecutionResult:
    """What the leaf returns when the legacy verifier accepted on evidence."""
    return ACExecutionResult(
        ac_index=index,
        ac_content=f"criterion {index}",
        success=True,
        outcome=ACExecutionOutcome.SUCCEEDED,
        atomic_verifier_verdict=VerifierVerdict(passed=True, reasons=(), failure_class=None),
    )


def _run_for_matrix(authority: CheckPackageAuthority) -> CheckPackageRun:
    return CheckPackageRun(
        CheckPackageSettings(
            enabled=True, assignment=CheckPackageAssignment(Arm.ON, AssignmentSource.USER_FORCED_ON)
        ),
        state=authority.state,
        authority=authority,
        attempted=True,
    )


async def test_authority_matrix_and_exit_semantics(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    """A and A' criteria are decided by the package; the README one by the legacy verifier."""
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + GOOD_MIX)
    authority.install(_executor(repo))
    results = (
        _legacy_rejected(0),  # A, passes; the legacy rejection is advisory
        _legacy_rejected(1, entry=MIX_ENTRY),  # A', passes
        _legacy_rejected(2),  # no admitted check: the legacy rejection decides it
    )
    parallel = ParallelExecutionResult(results=results, success_count=3, failure_count=0)
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    keys = seed_criterion_keys(seed)
    verdicts = authority.outcome.verdict.verdicts
    assert [(verdicts[k].status, verdicts[k].tier, verdicts[k].reason) for k in keys] == [
        (PackageCriterionStatus.PASS, CheckTier.A, "passed"),
        (PackageCriterionStatus.PASS, CheckTier.A_PRIME, "passed"),
        (PackageCriterionStatus.UNCOVERED, CheckTier.U, "uncovered:non_behavioral"),
    ]
    reconciliation = authority.outcome.reconciliation
    # A and A' are accepted over the legacy rejection; the legacy-decided
    # criterion fails, so the run fails (exit 1, durable status failed).
    assert [(d.accepted, d.governed_by.value) for d in reconciliation.decisions] == [
        (True, "check_package"),
        (True, "check_package"),
        (False, "existing_verifier"),
    ]
    assert not reconciliation.run_accepted and not decided.all_succeeded
    assert [r.outcome for r in decided.results] == [
        ACExecutionOutcome.SUCCEEDED,
        ACExecutionOutcome.SUCCEEDED,
        ACExecutionOutcome.FAILED,
    ]
    assert decided.results[2].error == (
        "legacy-decided (uncovered:non_behavioral): legacy verifier: evidence form mismatch"
    )
    assert [d.existing_failure_class for d in reconciliation.decisions] == [
        "EVIDENCE_FORM_MISMATCH"
    ] * 3
    run = _run_for_matrix(authority)
    lines = run.render_outcome()
    assert any(
        line.startswith(
            "AC 3: not accepted by the legacy verifier (legacy-decided, no admitted check: "
            "uncovered:non_behavioral)"
        )
        for line in lines
    )
    assert any(
        line.startswith(
            "Verified by the check package: 2 of 3 passed; legacy-decided: 1; unverified: 0"
        )
        for line in lines
    )
    assert not any(line.startswith("WARNING: insufficient verification") for line in lines)
    meta = await run.outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="failed"
    )  # type: ignore[arg-type]
    assert meta["package_verdict"] == "pass"
    assert meta["unverified_count"] == "0"
    assert meta["check_tier_summary"] == "A:1,A_prime:1,U:1"
    assert meta["legacy_verdict"] == "reject"
    assert meta["reconciliation"] == "legacy_decided_unverified"
    assert meta["verification_coverage"] == "partial"
    assert not {"non_behavioral_count", "label_parse_failure_count"} & set(meta)
    assert meta["legacy_failure_class"] == "evidence_form_mismatch"
    assert meta["legacy_failure_class_count"] == "3+"
    sent = _check_package_properties(meta)
    assert sent["reconciliation"] == "legacy_decided_unverified"
    assert sent["verification_coverage"] == "partial" and "non_behavioral_count" not in sent


async def test_a_legacy_accepted_unverified_criterion_exits_zero(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + GOOD_MIX)
    authority.install(_executor(repo))
    results = (
        _legacy_rejected(0),
        _legacy_rejected(1, entry=MIX_ENTRY),
        _legacy_accepted(2),  # the legacy verifier accepts it on evidence
    )
    parallel = ParallelExecutionResult(results=results, success_count=3, failure_count=0)
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    reconciliation = authority.outcome.reconciliation
    assert reconciliation.run_accepted and decided.all_succeeded
    assert reconciliation.decisions[2].legacy_decided and reconciliation.decisions[2].accepted
    assert reconciliation.accepted_unverified == ()
    meta = await _run_for_matrix(authority).outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="completed"
    )  # type: ignore[arg-type]
    assert meta["reconciliation"] == "package_accepted_over_legacy_reject"
    assert meta["verification_coverage"] == "partial"


async def test_a_package_failure_is_not_masked_by_a_legacy_acceptance(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    """A criteria stay decided by the package: legacy acceptance changes nothing."""
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(BUGGY + GOOD_MIX)
    authority.install(_executor(repo))
    results = (
        _legacy_accepted(0),
        _legacy_rejected(1, entry=MIX_ENTRY),
        _legacy_accepted(2),
    )
    parallel = ParallelExecutionResult(results=results, success_count=3, failure_count=0)
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    first = authority.outcome.reconciliation.decisions[0]
    assert first.governed_by.value == "check_package" and not first.accepted
    assert first.package_status is PackageCriterionStatus.FAIL
    assert not decided.all_succeeded
    # The rejection comes from the package (tier A), not from a legacy-decided criterion.
    meta = await _run_for_matrix(authority).outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="failed"
    )  # type: ignore[arg-type]
    assert meta["reconciliation"] == "agree"


async def test_both_verifiers_without_evidence_leave_the_criterion_unverified(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    """Transcript unavailable on the legacy side, no check on the package side: exit 0, flagged."""
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + GOOD_MIX)
    authority.install(_executor(repo))
    results = (
        _legacy_rejected(0),
        _legacy_rejected(1, entry=MIX_ENTRY),
        _transcript_unavailable(2),
    )
    parallel = ParallelExecutionResult(results=results, success_count=3, failure_count=0)
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    reconciliation = authority.outcome.reconciliation
    assert reconciliation.run_accepted and decided.all_succeeded
    (unverified,) = reconciliation.accepted_unverified
    assert unverified.root_ac_index == 2 and not unverified.legacy_decided
    run = _run_for_matrix(authority)
    lines = run.render_outcome()
    assert "- unverified AC 3: uncovered:non_behavioral" in lines
    assert any(
        line.startswith("WARNING: insufficient verification: the check package decided 2 of 3")
        for line in lines
    )
    meta = await run.outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="completed"
    )  # type: ignore[arg-type]
    assert meta["unverified_count"] == "1"
    assert meta["verification_coverage"] == "low"


async def test_failures_and_unattempted_criteria_are_not_accepted(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(BUGGY + BAD_MIX)
    authority.install(_executor(repo))
    crashed = ACExecutionResult(ac_index=2, ac_content="c", success=False, error="runtime crashed")
    parallel = ParallelExecutionResult(
        results=(_legacy_rejected(0), _legacy_rejected(1, entry=MIX_ENTRY), crashed),
        success_count=2,
        failure_count=1,
    )
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    decisions = authority.outcome.reconciliation.decisions
    assert [(d.package_status.value, d.accepted, d.governed_by.value) for d in decisions] == [
        ("fail", False, "check_package"),
        ("fail", False, "check_package"),
        ("uncovered", False, "execution"),
    ]
    assert authority.outcome.verdict.verdict == "fail"
    assert not decided.all_succeeded
    assert [r.outcome for r in decided.results] == [ACExecutionOutcome.FAILED] * 3


def test_existing_outcomes_with_the_gate_treat_runtime_failures_as_unattempted() -> None:
    parallel = ParallelExecutionResult(
        results=(
            _legacy_rejected(0),
            ACExecutionResult(ac_index=1, ac_content="c", success=False, error="crash"),
            ACExecutionResult(
                ac_index=2,
                ac_content="c",
                success=False,
                error="pkg",
                check_package_failure_class="CHECK_PACKAGE_FAIL:abc",
            ),
        ),
        success_count=1,
        failure_count=2,
    )
    gated = existing_outcomes_from_results(parallel, gated=True)
    assert [gated[i].attempted for i in range(3)] == [True, False, True]
    assert gated[0].failure_class == "EVIDENCE_FORM_MISMATCH" and not gated[0].passed


@pytest.mark.parametrize("gated", [True, False])
def test_an_unavailable_transcript_is_no_legacy_rejection(gated: bool) -> None:
    # R3-A2: the executor keeps such a result successful and sets no
    # rejection (arm off accepts it); a failing verdict alone is no
    # information, on the root and on a sub-AC.
    decomposed = ACExecutionResult(
        ac_index=1,
        ac_content="c1",
        success=True,
        outcome=ACExecutionOutcome.SUCCEEDED,
        is_decomposed=True,
        sub_results=(replace(_transcript_unavailable(1), ac_content="sub"),),
    )
    parallel = ParallelExecutionResult(
        results=(_transcript_unavailable(0), decomposed), success_count=2, failure_count=0
    )
    assert legacy_verdict_in_tree(parallel.results[0]) == (False, None, None)
    assert legacy_verdict_in_tree(decomposed) == (False, None, None)
    outcomes = existing_outcomes_from_results(parallel, gated=gated)
    assert [outcomes[i].passed for i in range(2)] == [True, True]
    kept = apply_legacy_fallback(parallel, outcomes)
    assert (
        kept.all_succeeded
        and [r.outcome for r in kept.results] == [ACExecutionOutcome.SUCCEEDED] * 2
    )


def test_tier_summary_value_is_a_closed_bucketed_enum() -> None:
    assert tier_summary_value({"A": 5, "A_prime": 2, "U": 0, "C": 4}) == "A:3+,A_prime:2,U:0"
    dropped = _check_package_properties(
        {"check_tier_summary": "A:9,A_prime:0,U:0", "unverified_count": "7"}
    )
    assert dropped == {}


async def test_the_gate_decides_each_attempt_once(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ouroboros.boundary import authority as authority_module

    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + BAD_MIX)
    runs: list[int] = []
    real = authority_module.verify_with_bindings

    async def counted(*args: Any, **kwargs: Any) -> Any:
        runs.append(1)
        return await real(*args, **kwargs)

    monkeypatch.setattr(authority_module, "verify_with_bindings", counted)
    attempt = _legacy_rejected(1, entry=MIX_ENTRY)
    first = await authority.gate(seed=seed, ac_index=1, result=attempt)
    again = await authority.gate(seed=seed, ac_index=1, result=attempt)  # a settlement path
    assert runs == [1]
    assert first.success is False and again.success is False
    assert again.check_package_repair == first.check_package_repair


async def test_omitting_entry_points_after_a_counterexample_does_not_withdraw_the_binding(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    # Found by smoke s3c: after a counterexample the worker kept its wrong
    # implementation and simply stopped declaring entry_points, which turned
    # the failing criterion into an unverified one (exit 0).
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + BAD_MIX)
    executor = _executor(repo)
    authority.install(executor)
    calls: list[int] = []

    async def fake_batch(**kwargs: Any) -> list[ACExecutionResult]:
        calls.append(1)
        entry = MIX_ENTRY if len(calls) == 1 else None  # later attempts declare nothing
        return [replace(_legacy_rejected(1, entry=entry), retry_attempt=len(calls) - 1)]

    executor._execute_ac_batch = fake_batch  # type: ignore[method-assign]
    results = await _batch(executor, seed, [1])
    assert [entry["status"] for entry in authority.gate.log] == ["fail"] * len(calls)
    assert results[0].success is False
    parallel = ParallelExecutionResult(
        results=(_legacy_rejected(0), results[0]),
        success_count=1,
        failure_count=1,
    )
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    keys = seed_criterion_keys(seed)
    item = authority.outcome.verdict.verdicts[keys[1]]
    assert (item.status, item.tier) == (PackageCriterionStatus.FAIL, CheckTier.A_PRIME)
    assert not decided.all_succeeded


class _FailingConstructor:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def construct(self, seed: Seed, base: Path, *, feedback=()) -> ConstructionOutcome:
        return ConstructionOutcome(None, "constructor_timeout", "1" * 64, "fake")


class _EmptyStore:
    async def query_events(self, **_kwargs: Any) -> list[Any]:
        return []


@pytest.mark.parametrize(("terminal", "legacy"), [("completed", "accept"), ("failed", "reject")])
async def test_outage_falls_back_to_the_legacy_path(
    store: EventStore,
    repo: Path,
    tmp_path: Path,
    terminal: str,
    legacy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Constructor outage: zero checks admitted. Nothing is installed on the
    # runner, so the executor and the terminal status are the legacy ones;
    # the run exits 0 only when the legacy verifier passed it.
    from types import SimpleNamespace

    monkeypatch.setattr(
        "ouroboros.boundary.run_wiring.default_store_dir",
        lambda execution_id: tmp_path / "store" / execution_id,
    )
    runner = SimpleNamespace(acceptance_authority=None)
    run = CheckPackageRun(
        CheckPackageSettings(
            enabled=True,
            max_construction_attempts=2,
            assignment=CheckPackageAssignment(Arm.ON, AssignmentSource.RANDOMIZED),
        )
    )
    lines = await run.prepare(
        runner,
        _seed(),
        event_store=store,
        execution_id="exec_outage",
        worker_dir=repo,
        runtime_backend="codex",
        model=None,
        resume=False,
        constructor_factory=_FailingConstructor,
    )
    assert runner.acceptance_authority is None and run.authority is None
    assert any("No admitted package (constructor_timeout)" in line for line in lines)
    assert run.render_outcome() == [
        "Check package unavailable (constructor_timeout); legacy verification decided this run.",
        "WARNING: insufficient verification: the check package decided 0 of 3 criteria "
        "(verification_coverage=low).",
    ]
    meta = await run.outcome_meta(
        _EmptyStore(),
        execution_id="exec_outage",
        session_id="s",
        terminal_status=terminal,  # type: ignore[arg-type]
    )
    assert meta["check_package_status"] == "construction_failed"
    assert meta["reconciliation"] == "fallback_to_legacy"
    assert meta["package_verdict"] == "none"
    assert meta["legacy_verdict"] == legacy
    assert "unverified_count" not in meta and "check_tier_summary" not in meta
    assert meta["verification_coverage"] == "low"
    # The executor gets no gate: its prompt and retry loop are the legacy ones.
    executor = _executor(repo)
    assert not hasattr(executor, "check_package_gate")


async def test_held_out_only_failure_reveals_one_case_and_retires_it(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    from ouroboros.boundary.events import BOUNDARY_AGGREGATE_TYPE, CASE_REVEALED

    seed, authority = await _authority(store, repo, tmp_path)
    # mix passes the stated case (0, 10, 0.5 -> 5) and fails the held-out one.
    (repo / "mathutils.py").write_text(
        FIXED + "\ndef mix(start, end, weight):\n    return start + end * weight\n"
    )
    attempt = _legacy_rejected(1, entry=MIX_ENTRY)
    gated = await authority.gate(seed=seed, ac_index=1, result=attempt)
    assert gated.success is False
    repair = gated.check_package_repair
    assert "declared entry point: function mathutils.mix" in repair
    assert (
        "- revealed held-out case: mix(start=2, end=4, weight=0.25): expected 2.5, observed 3.0"
        in repair
    )
    assert authority.revealed == {"oracle_2": {"held"}}
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, authority.state.boundary_id)
    reveal = [e for e in events if e.type == CASE_REVEALED]
    assert [(e.data["check_id"], e.data["case_id"], e.data["root_ac_index"]) for e in reveal] == [
        ("oracle_2", "held", 1)
    ]
    # A second attempt that still fails reveals nothing new (the case is visible now).
    again = await authority.gate(seed=seed, ac_index=1, result=replace(attempt, retry_attempt=1))
    assert "revealed held-out case: mix(start=2, end=4, weight=0.25)" in again.check_package_repair
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, authority.state.boundary_id)
    assert len([e for e in events if e.type == CASE_REVEALED]) == 1
    # Final verdict: the retired case no longer counts as held out.
    parallel = ParallelExecutionResult(
        results=(_legacy_rejected(0), replace(attempt, retry_attempt=1)),
        success_count=2,
        failure_count=0,
    )
    await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    keys = seed_criterion_keys(seed)
    final = authority.outcome.verdict.verdicts[keys[1]]
    assert final.status is PackageCriterionStatus.FAIL and final.failed_heldout_only is False


def _run_for(authority: CheckPackageAuthority) -> CheckPackageRun:
    return CheckPackageRun(
        CheckPackageSettings(
            enabled=True, assignment=CheckPackageAssignment(Arm.ON, AssignmentSource.USER_FORCED_ON)
        ),
        state=authority.state,
        authority=authority,
        attempted=True,
    )


class _NoEvents:
    async def query_events(self, **_kwargs: Any) -> list[Any]:
        return []


async def test_authority_error_with_the_gate_installed_falls_back_to_the_legacy_verdicts(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A1: the gate made the legacy rejections advisory; when the authority
    # then raises, the run must not complete. The legacy verdicts decide,
    # exactly as with no admitted package, and the reason is recorded.
    from ouroboros.boundary.events import BOUNDARY_AGGREGATE_TYPE, LEGACY_FALLBACK

    seed, authority = await _authority(store, repo, tmp_path)
    authority.install(_executor(repo))

    async def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("ouroboros.boundary.authority.verify_check_package", broken)
    gate_only = ACExecutionResult(
        ac_index=0,
        ac_content="criterion 0",
        success=False,
        outcome=ACExecutionOutcome.FAILED,
        error="check_package: ...",
        check_package_repair="counterexample",
        check_package_failure_class="CHECK_PACKAGE_FAIL:abc",
    )
    rejected = [
        replace(_legacy_rejected(index), legacy_rejection="evidence form mismatch")
        for index in (1, 2)
    ]
    parallel = ParallelExecutionResult(
        results=(gate_only, *rejected), success_count=2, failure_count=1
    )
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    assert not decided.all_succeeded
    assert [r.outcome for r in decided.results] == [
        ACExecutionOutcome.SUCCEEDED,  # only the package gate had failed it
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.FAILED,
    ]
    assert decided.results[1].error == "evidence form mismatch"
    assert decided.results[0].check_package_repair is None
    assert (decided.success_count, decided.failure_count) == (1, 2)
    assert authority.outcome.fallback_reason == "authority_error:OSError"
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, authority.state.boundary_id)
    assert [e.data["reason"] for e in events if e.type == LEGACY_FALLBACK] == [
        "authority_error:OSError"
    ]
    run = _run_for(authority)
    assert run.render_outcome() == [
        "Check package could not decide this run (authority_error:OSError); "
        "legacy verification decided this run.",
        "WARNING: insufficient verification: the check package decided 0 of 3 criteria "
        "(verification_coverage=low).",
    ]
    meta = await run.outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="failed"
    )  # type: ignore[arg-type]
    assert meta["reconciliation"] == "fallback_to_legacy"
    assert meta["legacy_verdict"] == "reject"
    assert "unverified_count" not in meta and "check_tier_summary" not in meta


async def test_tier_buckets_are_sent_only_when_the_package_decided(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    # T3: an admitted package whose criteria the worker never attempted is a
    # fallback_to_legacy row without the package-only buckets.
    seed, authority = await _authority(store, repo, tmp_path)
    authority.install(_executor(repo))
    crashed = tuple(
        ACExecutionResult(ac_index=index, ac_content="c", success=False, error="runtime crashed")
        for index in range(3)
    )
    parallel = ParallelExecutionResult(results=crashed, success_count=0, failure_count=3)
    await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    assert authority.outcome.reconciliation is not None and not authority.outcome.package_decided
    meta = await _run_for(authority).outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="failed"
    )  # type: ignore[arg-type]
    assert meta["reconciliation"] == "fallback_to_legacy"
    assert "unverified_count" not in meta and "check_tier_summary" not in meta


TWO_HELD = {
    **REPLY,
    "oracles": [
        REPLY["oracles"][0],
        {
            **REPLY["oracles"][1],
            "cases": [
                *REPLY["oracles"][1]["cases"],
                {
                    "case_id": "held2",
                    "args": {"a": 1, "b": 5, "t": 0.5},
                    "expect": {"kind": "returns", "value": 3.0, "approx": 1e-9},
                },
            ],
        },
    ],
}
WRONG_MIX = "\ndef mix(start, end, weight):\n    return start + end * weight\n"
SPECIAL_CASED_MIX = (
    "\ndef mix(start, end, weight):\n"
    "    if (start, end, weight) == (2, 4, 0.25):\n"
    "        return 2.5\n"
    "    return start + end * weight\n"
)


async def _two_held_authority(
    store: EventStore, repo: Path, tmp_path: Path, retries: int
) -> tuple[Seed, CheckPackageAuthority]:
    seed = _seed()
    constructor = _Constructor(seed, repo)
    constructor.outcome = ConstructionOutcome(
        package_from_reply(
            TWO_HELD, seed, input_digest="1" * 64, generator="fake", base_checkout=repo
        ),
        None,
        "1" * 64,
        "fake",
    )
    settings = CheckPackageSettings(enabled=True)
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=constructor,
        execution_id="exec_two_held",
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=settings,
        store_dir=tmp_path / "store",
    )
    authority = CheckPackageAuthority(state, settings, event_store=store, candidate_checkout=repo)
    authority.install(_executor(repo, retries=retries))
    return seed, authority


async def _reveals(store: EventStore, authority: CheckPackageAuthority) -> list[str]:
    from ouroboros.boundary.events import BOUNDARY_AGGREGATE_TYPE, CASE_REVEALED

    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, authority.state.boundary_id)
    return [e.data["case_id"] for e in events if e.type == CASE_REVEALED]


async def test_at_most_one_held_out_reveal_per_criterion_per_run(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    # A3: after the first reveal the worker special-cases that input; the
    # other held-out case, still failing, is not revealed on a later attempt.
    seed, authority = await _two_held_authority(store, repo, tmp_path, retries=2)
    (repo / "mathutils.py").write_text(FIXED + WRONG_MIX)
    attempt = _legacy_rejected(1, entry=MIX_ENTRY)
    first = await authority.gate(seed=seed, ac_index=1, result=attempt)
    assert "revealed held-out case: mix(start=2, end=4, weight=0.25)" in first.check_package_repair
    (repo / "mathutils.py").write_text(FIXED + SPECIAL_CASED_MIX)
    second = await authority.gate(seed=seed, ac_index=1, result=replace(attempt, retry_attempt=1))
    assert second.success is False
    assert "start=1" not in second.check_package_repair
    assert "1 held-out case(s) also failed" in second.check_package_repair
    assert await _reveals(store, authority) == ["held"]
    assert authority.revealed == {"oracle_2": {"held"}}


@pytest.mark.parametrize(("retries", "attempt_number"), [(0, 0), (2, 2)])
async def test_no_reveal_on_the_final_attempt(
    store: EventStore, repo: Path, tmp_path: Path, retries: int, attempt_number: int
) -> None:
    # A4: no repair follows the final attempt, so nothing is revealed or
    # retired, and the held-out-only diagnostic still counts the failure.
    seed, authority = await _two_held_authority(store, repo, tmp_path, retries=retries)
    (repo / "mathutils.py").write_text(FIXED + WRONG_MIX)
    attempt = replace(_legacy_rejected(1, entry=MIX_ENTRY), retry_attempt=attempt_number)
    gated = await authority.gate(seed=seed, ac_index=1, result=attempt)
    assert gated.success is False
    assert "revealed" not in gated.check_package_repair
    assert "2 held-out case(s) also failed" in gated.check_package_repair
    assert await _reveals(store, authority) == [] and authority.revealed == {}
    parallel = ParallelExecutionResult(
        results=(_legacy_rejected(0), gated), success_count=1, failure_count=1
    )
    await authority(seed=seed, execution_id="exec_two_held", parallel_result=parallel)
    final = authority.outcome.verdict.verdicts[seed_criterion_keys(seed)[1]]
    assert final.status is PackageCriterionStatus.FAIL and final.failed_heldout_only is True


async def test_a_rejected_declaration_gets_a_repair_message_with_its_reason(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    # A5: binding_invalid used to end the run indeterminate without telling
    # the worker why; now the reason goes into the repair.
    seed, authority = await _authority(store, repo, tmp_path)
    authority.install(_executor(repo))
    (repo / "mathutils.py").write_text(FIXED + GOOD_MIX)
    wrong = {"symbol": "mathutils.blend", "arg_map": MIX_ENTRY["arg_map"]}
    gated = await authority.gate(seed=seed, ac_index=1, result=_legacy_rejected(1, entry=wrong))
    assert gated.success is False
    assert gated.check_package_failure_class.startswith("CHECK_PACKAGE_FAIL:")
    repair = gated.check_package_repair
    assert "rejected: binding_invalid:symbol_not_found" in repair
    assert "function mathutils.blend" in repair and "(a, b, t)" in repair
    assert "2.5" not in repair  # no oracle value
    # The worker fixes its declaration within the retry budget.
    fixed = await authority.gate(
        seed=seed, ac_index=1, result=replace(_legacy_rejected(1, entry=MIX_ENTRY), retry_attempt=1)
    )
    assert fixed.success is True


def _decomposed_root(index: int, *, via: str) -> ACExecutionResult:
    """A decomposed root whose second sub-AC the legacy verifier rejected (gate on)."""
    sub_ok = ACExecutionResult(
        ac_index=index, ac_content="sub a", success=True, outcome=ACExecutionOutcome.SUCCEEDED
    )
    if via == "verdict":
        sub_rejected = replace(_legacy_rejected(index), ac_content="sub b")
    else:
        sub_rejected = ACExecutionResult(
            ac_index=index,
            ac_content="sub b",
            success=True,
            outcome=ACExecutionOutcome.SUCCEEDED,
            legacy_rejection="evidence form mismatch",
        )
    return ACExecutionResult(
        ac_index=index,
        ac_content=f"criterion {index}",
        success=True,
        outcome=ACExecutionOutcome.SUCCEEDED,
        is_decomposed=True,
        sub_results=(sub_ok, sub_rejected),
    )


@pytest.mark.parametrize("via", ["verdict", "annotation"])
async def test_the_fallback_uses_the_legacy_verdicts_of_sub_acs(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    # R2-A1: a decomposed root carries no legacy rejection itself; its sub-AC
    # does. When the authority fails, the legacy verdict tree decides, so the
    # root fails and the run does not exit 0.
    seed, authority = await _authority(store, repo, tmp_path)
    authority.install(_executor(repo))

    async def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("ouroboros.boundary.authority.verify_check_package", broken)
    results = (
        _decomposed_root(0, via=via),
        ACExecutionResult(ac_index=1, ac_content="c1", success=True),
        ACExecutionResult(ac_index=2, ac_content="c2", success=True),
    )
    parallel = ParallelExecutionResult(results=results, success_count=3, failure_count=0)
    assert not existing_outcomes_from_results(parallel, gated=True)[0].passed
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    assert authority.outcome.fallback_reason == "authority_error:OSError"
    assert not decided.all_succeeded
    assert decided.results[0].outcome is ACExecutionOutcome.FAILED
    if via == "annotation":
        assert decided.results[0].error == "evidence form mismatch"
    meta = await _run_for(authority).outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="failed"
    )  # type: ignore[arg-type]
    assert meta["legacy_verdict"] == "reject"


async def test_a_failing_fallback_fails_every_attempted_criterion(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # R2-A5: the run never reports a legacy decision it did not apply.
    seed, authority = await _authority(store, repo, tmp_path)
    authority.install(_executor(repo))

    async def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    def broken_apply(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("apply")

    monkeypatch.setattr("ouroboros.boundary.authority.verify_check_package", broken)
    monkeypatch.setattr("ouroboros.boundary.authority.apply_legacy_fallback", broken_apply)
    parallel = ParallelExecutionResult(
        results=tuple(
            ACExecutionResult(ac_index=i, ac_content=f"c{i}", success=True) for i in range(3)
        ),
        success_count=3,
        failure_count=0,
    )
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    assert authority.outcome.fallback_reason == "authority_error:OSError:fallback_failed"
    assert not decided.all_succeeded
    assert (decided.success_count, decided.failure_count) == (0, 3)


DEEP_FRAME = (
    "import os, stat, sys\n"
    "_fd = next(fd for fd in range(3, 64) if os.path.exists(f'/dev/fd/{fd}')"
    " and stat.S_ISFIFO(os.fstat(fd).st_mode))\n"
    "os.write(_fd, ('\\n' + sys.argv[2] + ' ' + '[' * 200000 + ']' * 200000 + '\\n').encode())\n"
)


async def test_a_hostile_frame_is_a_package_fail_not_a_legacy_fallback(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    # R2-S1 end to end: the candidate writes a deeply nested frame while it is
    # imported. Every case of its checks fails with a counterexample; the gate
    # sends a repair, and the authority decides (no legacy fallback): not
    # accepted, so the run exits 1.
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(DEEP_FRAME + BUGGY + GOOD_MIX)
    authority.install(_executor(repo))
    ok = ACExecutionResult(
        ac_index=0, ac_content="c0", success=True, outcome=ACExecutionOutcome.SUCCEEDED
    )
    gated = await authority.gate(seed=seed, ac_index=0, result=ok)
    assert gated.check_package_repair
    assert "observed malformed or oversized output" in gated.check_package_repair
    parallel = ParallelExecutionResult(
        results=(
            ok,
            ACExecutionResult(
                ac_index=1,
                ac_content="c1",
                success=True,
                typed_evidence=EvidenceRecord(data={"entry_points": [MIX_ENTRY]}),
            ),
            ACExecutionResult(ac_index=2, ac_content="c2", success=True),
        ),
        success_count=3,
        failure_count=0,
    )
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    assert authority.outcome.fallback_reason is None
    assert authority.outcome.verdict is not None
    assert authority.outcome.verdict.verdict == "fail"
    assert not decided.all_succeeded


async def test_held_out_values_never_reach_the_boundary_store(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    # S3: after a run (admission, a gate repair, the final verification), no
    # file under the boundary store and no boundary event holds a held-out
    # input or expected value. The values are distinctive so a hit is real.
    import copy

    from ouroboros.boundary.events import BOUNDARY_AGGREGATE_TYPE

    reply: dict[str, Any] = copy.deepcopy(REPLY)
    reply["oracles"][0]["cases"][1] = {
        "case_id": "held",
        "args": {"value": 6173, "low": 1, "high": 4409},
        "expect": {"kind": "returns", "value": 4409},
    }
    reply["oracles"][1]["cases"][1] = {
        "case_id": "held",
        "args": {"a": 3079, "b": 3083, "t": 0.5},
        "expect": {"kind": "returns", "value": 3081, "approx": 1e-9},
    }
    secrets = ("6173", "4409", "3079", "3083", "3081")

    def found(token: str, data: bytes) -> bool:
        # A standalone number: not part of a hex digest, a longer number, or
        # a timestamp's fraction.
        return (
            re.search(rb"(?<![0-9A-Za-z.])" + token.encode() + rb"(?![0-9A-Za-z])", data)
            is not None
        )

    seed = _seed()
    constructor = _Constructor(seed, repo)
    constructor.outcome = ConstructionOutcome(
        package_from_reply(
            reply, seed, input_digest="1" * 64, generator="fake", base_checkout=repo
        ),
        None,
        "1" * 64,
        "fake",
    )
    settings = CheckPackageSettings(enabled=True)
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=constructor,
        execution_id="exec_oracle",
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=settings,
        store_dir=tmp_path / "store",
    )
    assert state.package is not None
    assert [case.held_out for spec in state.package.oracles for case in spec.cases] == [
        False,
        True,
        False,
        True,
    ]
    authority = CheckPackageAuthority(state, settings, event_store=store, candidate_checkout=repo)
    authority.install(_executor(repo))
    (repo / "mathutils.py").write_text(BUGGY + BAD_MIX)
    ok = ACExecutionResult(
        ac_index=0, ac_content="c0", success=True, outcome=ACExecutionOutcome.SUCCEEDED
    )
    gated = await authority.gate(seed=seed, ac_index=0, result=ok)
    assert gated.check_package_repair
    parallel = ParallelExecutionResult(
        results=(
            ok,
            ACExecutionResult(
                ac_index=1,
                ac_content="c1",
                success=True,
                typed_evidence=EvidenceRecord(data={"entry_points": [MIX_ENTRY]}),
            ),
            ACExecutionResult(ac_index=2, ac_content="c2", success=True),
        ),
        success_count=3,
        failure_count=0,
    )
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    assert not decided.all_succeeded

    stored = [path for path in (tmp_path / "store").rglob("*") if path.is_file()]
    assert any(path.parent.name == "packages" for path in stored)
    assert any(path.parent.name == "receipts" for path in stored)
    hits = [
        (str(path), token)
        for path in stored
        for token in secrets
        if found(token, path.read_bytes())
    ]
    assert hits == []
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
    journal = json.dumps([event.data for event in events]).encode()
    assert [token for token in secrets if found(token, journal)] == []
    # The matcher does find a value where it would be leaked.
    assert found("4409", b'{"value": 4409}') and not found("4409", b"ab4409cd")
    # The record names the package by its commitment and keeps held-out cases as ids.
    (record_path,) = (tmp_path / "store" / "packages").iterdir()
    record = json.loads(record_path.read_text())
    assert record["package_commitment"] == state.package.commitment
    assert "package_sha256" not in record
    assert [(item["check_id"], item["case_id"]) for item in record["held_out"]] == [
        ("oracle_1", "held"),
        ("oracle_2", "held"),
    ]
    # The authority reached the final verdict, so it revealed the salt beside
    # the store (never inside it), and the pre-dispatch commitment verifies.
    (salt_path,) = controller_private_dir(tmp_path / "store").iterdir()
    salt = bytes.fromhex(json.loads(salt_path.read_text())["salt"])
    assert verify_commitment(state.package, salt, record["package_commitment"])


async def test_telemetry_never_reads_an_unavailable_transcript_as_a_rejection(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    # R3-A2: the package passes and the legacy verifier had no transcript:
    # that is agreement with what arm off decides (accept), never
    # package_accepted_over_legacy_reject.
    seed, authority = await _authority(store, repo, tmp_path)
    (repo / "mathutils.py").write_text(FIXED + GOOD_MIX)
    authority.install(_executor(repo))
    results = (
        _transcript_unavailable(0),
        replace(
            _transcript_unavailable(1),
            typed_evidence=EvidenceRecord(data={"entry_points": [MIX_ENTRY]}),
        ),
        _transcript_unavailable(2),
    )
    parallel = ParallelExecutionResult(results=results, success_count=3, failure_count=0)
    decided = await authority(seed=seed, execution_id="exec_oracle", parallel_result=parallel)
    assert decided.all_succeeded and authority.outcome.legacy_run_accepted is True
    meta = await _run_for(authority).outcome_meta(
        _NoEvents(), execution_id="exec_oracle", session_id="s", terminal_status="completed"
    )  # type: ignore[arg-type]
    assert meta["package_verdict"] == "pass"
    assert meta["legacy_verdict"] == "accept"
    assert meta["reconciliation"] == "agree"
    assert meta["legacy_failure_class"] == "accepted"


async def test_a_legacy_rejection_of_a_legacy_decided_criterion_drives_the_retry(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    """The README criterion has no admitted check: its legacy rejection fails the attempt."""
    seed, authority = await _authority(store, repo, tmp_path)
    executor = _executor(repo)
    authority.install(executor)
    prompts: list[dict[int, str]] = []

    async def fake_batch(**kwargs: Any) -> list[ACExecutionResult]:
        prompts.append(dict(kwargs.get("retry_prompts") or {}))
        first = len(prompts) == 1
        result = _legacy_rejected(2) if first else _legacy_accepted(2)
        return [replace(result, retry_attempt=len(prompts) - 1)]

    executor._execute_ac_batch = fake_batch  # type: ignore[method-assign]
    results = await _batch(executor, seed, [2])
    assert len(prompts) == 2 and results[0].success is True
    assert authority.gate.legacy_failures == 1
    retry = prompts[1][2]
    assert "### Check package counterexample" not in retry
    assert "LEGACY_DECIDED:EVIDENCE_FORM_MISMATCH" in retry
    assert authority.gate.log == []  # no package verification ran for it
    # A settlement path handing the same attempt to the gate again is not recounted.
    again = await authority.gate(seed=seed, ac_index=2, result=_legacy_rejected(2))
    assert again.success is False and authority.gate.legacy_failures == 1
