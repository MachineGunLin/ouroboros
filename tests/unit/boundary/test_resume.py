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
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        settings=CheckPackageSettings(True, policy=RegenerationPolicy.STUDY),
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
