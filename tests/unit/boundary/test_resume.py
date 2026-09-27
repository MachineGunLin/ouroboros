"""R3-S3: a controller that dies after the worker stops never hands the decision to legacy.

The crash is simulated after the worker stopped and before the package
decided: the run's in-memory state is dropped (another process) or kept (the
same process), nothing of the authority ran, and a fresh ``CheckPackageRun``
resumes the execution against the same journal and store. The arm resolves
``off`` on resume on purpose: the journal decides that the run was arm on.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest

from ouroboros.boundary.acceptance import PackageCriterionStatus
from ouroboros.boundary.constructor import ConstructionOutcome, package_from_reply
from ouroboros.boundary.events import ACCEPTANCE_RESUMED, BOUNDARY_AGGREGATE_TYPE
from ouroboros.boundary.ledger import verify_boundary_order
from ouroboros.boundary.package import seed_criterion_keys
from ouroboros.boundary.resume import (
    HELD_OUT_UNAVAILABLE,
    PACKAGE_UNAVAILABLE,
    ResumedCheckPackageAuthority,
)
from ouroboros.boundary.rollout import Arm, AssignmentSource, CheckPackageAssignment
from ouroboros.boundary.run_control import CheckPackageRun
from ouroboros.boundary.run_wiring import (
    BoundaryRunState,
    CheckPackageSettings,
    RegenerationPolicy,
    controller_private_dir,
    forget_live_state,
    live_state,
    prepare_check_package,
)
from ouroboros.core.seed import OntologySchema, Seed, SeedMetadata
from ouroboros.orchestrator.parallel_executor_models import (
    ACExecutionOutcome,
    ACExecutionResult,
    ParallelExecutionResult,
)
from ouroboros.persistence.event_store import EventStore

CLAMP_BUGGY = "def clamp(value, low, high):\n    if value > high:\n        return value\n    return max(low, value)\n"
CLAMP_FIXED = "def clamp(value, low, high):\n    return max(low, min(high, value))\n"
DOUBLE = "def double(x):\n    return 2 * x\n"
EXECUTION = "exec_crash"
LEGACY_TEXT = "legacy verifier: evidence form mismatch"


def _seed() -> Seed:
    return Seed(
        goal="math helpers",
        acceptance_criteria=(
            "clamp(15, 0, 10) returns 10",
            "double(3) returns 6 and double(4) returns 8",
            "the helpers are documented in the README",
        ),
        ontology_schema=OntologySchema(name="mathutils", description="math helpers"),
        metadata=SeedMetadata(seed_id="seed_resume", ambiguity_score=0.1),
    )


def _reply() -> dict[str, Any]:
    return {
        "oracles": [
            {
                "criterion": 1,
                "check_id": "oracle_clamp",
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
                        "args": {"value": 99, "low": 1, "high": 7},
                        "expect": {"kind": "returns", "value": 7},
                    },
                ],
            },
            {
                "criterion": 2,
                "check_id": "oracle_double",
                "role": "preservation",
                "call_kind": "function",
                "params": ["x"],
                "default_binding": {"symbol": "mathutils.double"},
                "cases": [
                    {
                        "case_id": "three",
                        "args": {"x": 3},
                        "expect": {"kind": "returns", "value": 6},
                    },
                    {
                        "case_id": "four",
                        "args": {"x": 4},
                        "expect": {"kind": "returns", "value": 8},
                    },
                ],
            },
        ],
        "uncovered": [{"criterion": 3, "reason": "not executable"}],
    }


class _Constructor:
    def __init__(self, seed: Seed, base: Path) -> None:
        self.outcome = ConstructionOutcome(
            package_from_reply(
                _reply(), seed, input_digest="1" * 64, generator="fake", base_checkout=base
            ),
            None,
            "1" * 64,
            "fake",
        )

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
    (root / "mathutils.py").write_text(CLAMP_BUGGY + DOUBLE)
    return root


async def _run_until_the_worker_stops(
    store: EventStore,
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    assignment: CheckPackageAssignment | None = None,
) -> tuple[Seed, BoundaryRunState]:
    """Prepare (arm on), dispatch the worker, and let it stop; the package never decides."""
    monkeypatch.setattr(
        "ouroboros.boundary.resume.default_store_dir", lambda _execution: tmp_path / "store"
    )
    seed = _seed()
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=_Constructor(seed, repo),
        execution_id=EXECUTION,
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=CheckPackageSettings(True, policy=RegenerationPolicy.STUDY, assignment=assignment),
        store_dir=tmp_path / "store",
    )
    assert state.admitted and state.package is not None
    held = [case.held_out for spec in state.package.oracles for case in spec.cases]
    assert held == [False, True, False, False]
    assert live_state(EXECUTION) is state
    return seed, state


def _restored(*, legacy_rejected: tuple[int, ...] = ()) -> ParallelExecutionResult:
    """What the resumed executor restores: every root succeeded before the crash."""
    results = tuple(
        ACExecutionResult(
            ac_index=index,
            ac_content=f"criterion {index}",
            success=True,
            outcome=ACExecutionOutcome.SUCCEEDED,
            legacy_rejection=LEGACY_TEXT if index in legacy_rejected else None,
        )
        for index in range(3)
    )
    return ParallelExecutionResult(results=results, success_count=3, failure_count=0)


async def _resume(store: EventStore, seed: Seed, repo: Path) -> tuple[CheckPackageRun, Any]:
    run = CheckPackageRun(CheckPackageSettings(enabled=False))  # the arm resolves off now
    runner = SimpleNamespace(acceptance_authority=None)
    lines = await run.prepare(
        runner,
        seed,
        event_store=store,
        execution_id=EXECUTION,
        worker_dir=repo,
        runtime_backend="codex",
        model=None,
        resume=True,
    )
    assert isinstance(runner.acceptance_authority, ResumedCheckPackageAuthority)
    assert runner.acceptance_authority is run.resumed and lines
    return run, runner.acceptance_authority


async def test_after_a_crash_the_visible_cases_decide_and_held_out_ones_are_unavailable(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)  # the worker's final workspace
    forget_live_state(state)  # the controller process died: its memory is gone
    run, authority = await _resume(store, seed, repo)
    assert authority.boundary.source == "record"
    assert authority.boundary.held_out_checks == frozenset({"oracle_clamp"})

    # Criterion 1 passes its visible case but its held-out case is gone; the
    # legacy verifier's rejection of criterion 2 decides nothing.
    decided = await authority(
        seed=seed, execution_id=EXECUTION, parallel_result=_restored(legacy_rejected=(1,))
    )
    keys = seed_criterion_keys(seed)
    verdicts = authority.outcome.verdict.verdicts
    assert (verdicts[keys[0]].status, verdicts[keys[0]].reason) == (
        PackageCriterionStatus.INDETERMINATE,
        HELD_OUT_UNAVAILABLE,
    )
    assert verdicts[keys[1]].status is PackageCriterionStatus.PASS
    assert verdicts[keys[2]].status is PackageCriterionStatus.UNCOVERED
    assert [result.outcome for result in decided.results] == [
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.SUCCEEDED,
        ACExecutionOutcome.SUCCEEDED,
    ]
    assert not decided.all_succeeded  # non-zero exit
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
    (resumed,) = [event for event in events if event.type == ACCEPTANCE_RESUMED]
    assert resumed.data["package_commitment"] == state.package.commitment
    assert resumed.data["held_out_checks"] == ["oracle_clamp"]
    assert verify_boundary_order(events) == ()
    assert any("visible cases" in line for line in run.render_outcome())
    # A second call keeps the first decision.
    again = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    assert again.all_succeeded


async def test_after_a_crash_a_visible_failure_fails_a_criterion_legacy_accepted(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    # The worker left clamp broken; the legacy verifier accepted everything.
    forget_live_state(state)
    _run, authority = await _resume(store, seed, repo)
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    assert verdicts[seed_criterion_keys(seed)[0]].status is PackageCriterionStatus.FAIL
    assert decided.results[0].outcome is ACExecutionOutcome.FAILED
    assert "fails the frozen check package" in (decided.results[0].error or "")
    assert not decided.all_succeeded


async def test_without_the_store_every_covered_criterion_is_undecided(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    shutil.rmtree(tmp_path / "store")
    _run, authority = await _resume(store, seed, repo)
    assert authority.boundary.package is None
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    keys = seed_criterion_keys(seed)
    assert [verdicts[key].reason for key in keys[:2]] == [PACKAGE_UNAVAILABLE] * 2
    assert [result.outcome for result in decided.results] == [
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.SUCCEEDED,  # uncovered: unverified, as in the live run
    ]
    assert not decided.all_succeeded


async def test_in_the_same_process_the_held_out_cases_are_re_derived_from_memory(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    # The run's task died but the process did not: the state is still live.
    _run, authority = await _resume(store, seed, repo)
    assert authority.boundary.source == "memory"
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    assert verdicts[seed_criterion_keys(seed)[0]].status is PackageCriterionStatus.PASS
    assert decided.all_succeeded
    # The final verdict exists: the salt is revealed and the memory dropped.
    assert list(controller_private_dir(tmp_path / "store").iterdir())
    assert live_state(EXECUTION) is None


async def test_a_run_never_bound_to_a_package_resumes_as_the_legacy_run(
    store: EventStore, repo: Path
) -> None:
    run = CheckPackageRun(CheckPackageSettings(enabled=False))
    runner = SimpleNamespace(acceptance_authority=None)
    lines = await run.prepare(
        runner,
        _seed(),
        event_store=store,
        execution_id="exec_legacy",
        worker_dir=repo,
        runtime_backend="codex",
        model=None,
        resume=True,
    )
    assert lines == [] and runner.acceptance_authority is None and run.resumed is None


# R4-A1: a resumed run counts attempts exactly as the live arm-on run does.


def _failed(index: int, error: str, **fields: Any) -> ACExecutionResult:
    return ACExecutionResult(
        ac_index=index,
        ac_content=f"criterion {index}",
        success=False,
        outcome=ACExecutionOutcome.FAILED,
        error=error,
        **fields,
    )


async def test_after_a_crash_a_root_that_already_failed_is_never_accepted(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's probe: a failed session and a failed verify command stay failed.

    Criterion 2 passes its visible cases and criterion 3 is uncovered, so the
    package alone would accept both. The live arm-on run does not count
    either as an attempt, and neither does the resumed run.
    """
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    _run, authority = await _resume(store, seed, repo)
    restored = ParallelExecutionResult(
        results=(
            _restored().results[0],
            _failed(1, "Implementation session failed"),
            _failed(2, "Verify gate failed: exit 1"),
        ),
        success_count=1,
        failure_count=2,
    )
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=restored)
    assert [result.outcome for result in decided.results] == [ACExecutionOutcome.FAILED] * 3
    assert decided.results[1].error == "Implementation session failed"
    assert decided.results[2].error == "Verify gate failed: exit 1"
    assert not decided.all_succeeded
    decisions = authority.outcome.reconciliation.decisions
    assert [decision.accepted for decision in decisions] == [False, False, False]
    assert not authority.outcome.legacy_run_accepted


async def test_after_a_crash_a_root_only_the_gate_failed_is_decided_by_the_package(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As in the live run, a root the package gate failed is an attempt the package decides."""
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    _run, authority = await _resume(store, seed, repo)
    base = _restored().results
    restored = ParallelExecutionResult(
        results=(
            base[0],
            _failed(
                1,
                "check_package: the finished workspace fails the frozen check package",
                check_package_repair="counterexample",
                check_package_failure_class="CHECK_PACKAGE_FAIL:abc",
            ),
            base[2],
        ),
        success_count=2,
        failure_count=1,
    )
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=restored)
    assert [result.outcome for result in decided.results] == [
        ACExecutionOutcome.FAILED,  # held-out cases unavailable
        ACExecutionOutcome.SUCCEEDED,  # the finished workspace passes criterion 2
        ACExecutionOutcome.SUCCEEDED,
    ]


# R4-S1: the store must agree with the journal before any of it is used.


def _record_path(tmp_path: Path) -> Path:
    (path,) = list((tmp_path / "store" / "packages").glob("*.json"))
    return path


def _delete_held_out(record: dict[str, Any]) -> None:
    for spec in record["package"]["oracles"]:
        spec["cases"] = [case for case in spec["cases"] if not case.get("held_out")]
    record["held_out"] = []


def _swap_held_out(record: dict[str, Any]) -> None:
    """Counts and ids unchanged: the stated case becomes held out, the held-out one visible."""
    stated, held = record["package"]["oracles"][0]["cases"]
    record["package"]["oracles"][0]["cases"] = [
        {"case_id": "stated", "held_out": True, "hmac_sha256": held["hmac_sha256"]},
        {**stated, "case_id": "held"},
    ]
    record["held_out"] = [{**record["held_out"][0], "case_id": "stated"}]


def _add_case(record: dict[str, Any]) -> None:
    spec = record["package"]["oracles"][0]
    spec["cases"].append({**spec["cases"][0], "case_id": "extra"})


def _edit_file(record: dict[str, Any]) -> None:
    item = record["package"]["files"][0]
    item["content"] += "\n# edited\n"


@pytest.mark.parametrize(
    "tamper", [_delete_held_out, _swap_held_out, _add_case, _edit_file], ids=lambda f: f.__name__
)
async def test_a_store_that_disagrees_with_the_journal_decides_nothing(
    store: EventStore,
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: Any,
) -> None:
    """The reviewer's probe: the target edits the stored record, then kills the controller.

    The workspace still fails the held-out case. Without the check the record
    with its held-out case deleted made criterion 1 a PASS.
    """
    import json

    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED.replace("min(high", "min(high + 1") + DOUBLE)
    path = _record_path(tmp_path)
    record = json.loads(path.read_text())
    tamper(record)
    path.write_text(json.dumps(record))
    forget_live_state(state)
    _run, authority = await _resume(store, seed, repo)
    assert authority.boundary.package is None
    assert authority.boundary.reason == "package_record_tampered"
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    keys = seed_criterion_keys(seed)
    assert [(verdicts[key].status, verdicts[key].reason) for key in keys[:2]] == [
        (PackageCriterionStatus.INDETERMINATE, "package_record_tampered")
    ] * 2
    assert [result.outcome for result in decided.results] == [
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.SUCCEEDED,  # uncovered: unverified, as in the live run
    ]
    assert not decided.all_succeeded


async def test_an_untouched_store_agrees_with_the_journal(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from ouroboros.boundary.resume import record_mismatch

    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
    frozen, admission = events[0], events[1]
    record = json.loads(_record_path(tmp_path).read_text())
    assert record_mismatch(record, frozen.data["manifest"], admission.data) is None
    forget_live_state(state)
    _run, authority = await _resume(store, seed, repo)
    assert authority.boundary.package is not None
    assert authority.boundary.held_out_checks == frozenset({"oracle_clamp"})


# R4-A2: telemetry of a resumed run is the original run's, from the journal.


async def test_resumed_telemetry_reports_the_original_arm_and_the_package_decision(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's probe: arm on (randomized), the resumed package fails the run.

    Before the fix the row read arm ``off``, ``not_run``, ``package_verdict=none``,
    ``reconciliation=none`` and ``legacy_verdict=reject``.
    """
    randomized = CheckPackageAssignment(Arm.ON, AssignmentSource.RANDOMIZED)
    seed, state = await _run_until_the_worker_stops(
        store, repo, tmp_path, monkeypatch, assignment=randomized
    )
    started = [
        event
        for event in await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
        if event.type == "boundary.actor.started"
    ]
    assert [event.data["check_package_assignment"] for event in started] == ["randomized"]
    forget_live_state(state)  # clamp still broken: package FAIL; legacy accepted all
    run, authority = await _resume(store, seed, repo)  # the arm resolves off now
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    assert not decided.all_succeeded and authority.outcome.legacy_run_accepted
    meta = await run.outcome_meta(
        store, execution_id=EXECUTION, session_id="s", terminal_status="failed"
    )
    assert {key: meta[key] for key in list(meta)[:6]} == {
        "check_package_arm": "on",
        "check_package_assignment": "randomized",
        "check_package_status": "admitted",
        "package_verdict": "fail",
        "legacy_verdict": "accept",
        "reconciliation": "package_rejected_over_legacy_accept",
    }
    assert meta["legacy_failure_class"] == "accepted"
    assert meta["unverified_count"] == "1"


async def test_resumed_telemetry_of_an_undecided_run_is_indeterminate(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    run, authority = await _resume(store, seed, repo)
    await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    meta = await run.outcome_meta(
        store, execution_id=EXECUTION, session_id="s", terminal_status="failed"
    )
    assert (
        meta["check_package_arm"],
        meta["check_package_assignment"],
        meta["check_package_status"],
        meta["package_verdict"],
        meta["reconciliation"],
    ) == (
        "on",
        "user_forced_on",
        "admitted",
        "indeterminate",
        "package_rejected_over_legacy_accept",
    )
