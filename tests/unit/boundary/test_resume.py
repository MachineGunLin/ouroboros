"""A controller that dies after the worker stops never hands the decision to legacy.

The crash is simulated after the worker stopped and before the package
decided: the run's in-memory state is dropped (another process) or kept (the
same process), nothing of the authority ran, and a fresh ``CheckPackageRun``
resumes the execution against the same journal and store. The switch resolves
``off`` on resume on purpose: the journal decides that the check package was on.
"""

from __future__ import annotations

import copy as copy_module
from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

import pytest

from ouroboros.boundary.acceptance import PackageCriterionStatus
from ouroboros.boundary.check_env import INTERPRETER_CHANGED
from ouroboros.boundary.constructor import ConstructionOutcome
from ouroboros.boundary.events import (
    ACCEPTANCE_RESUMED,
    ADMISSION_COMPLETED,
    BOUNDARY_AGGREGATE_TYPE,
    PACKAGE_FROZEN,
    boundary_version_id,
    package_frozen_event,
)
from ouroboros.boundary.ledger import BoundaryLedger, BoundaryOrderError, verify_boundary_order
from ouroboros.boundary.oracle_build import package_from_reply
from ouroboros.boundary.package import seed_criterion_keys
import ouroboros.boundary.resume as resume_module
from ouroboros.boundary.resume import (
    BOUNDARY_RECORD_MISSING,
    HELD_OUT_UNAVAILABLE,
    PACKAGE_UNAVAILABLE,
    ResumedCheckPackageAuthority,
    load_resumed_boundary,
    recovery_plan,
)
from ouroboros.boundary.run_wiring import (
    BoundaryRunState,
    CheckPackageSettings,
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
                "check_id": "oracle_1",
                "role": "reproduction",
                "call_kind": "function",
                "params": ["value", "low", "high"],
                "default_binding": {"symbol": "mathutils.clamp"},
                "target_named_in_criterion": False,
                "cases": [
                    {
                        "case_id": "c1",
                        "held_out": False,
                        "args": {"value": 15, "low": 0, "high": 10},
                        "expect": {"kind": "returns", "value": 10},
                    },
                    {
                        "case_id": "c2",
                        "held_out": True,
                        "args": {"value": 99, "low": 1, "high": 7},
                        "expect": {"kind": "returns", "value": 7},
                    },
                ],
            },
            {
                "criterion": 2,
                "check_id": "oracle_2",
                "role": "preservation",
                "call_kind": "function",
                "params": ["x"],
                "default_binding": {"symbol": "mathutils.double"},
                "target_named_in_criterion": False,
                "cases": [
                    {
                        "case_id": "three",
                        "held_out": False,
                        "args": {"x": 3},
                        "expect": {"kind": "returns", "value": 6},
                    },
                    {
                        "case_id": "four",
                        "held_out": False,
                        "args": {"x": 4},
                        "expect": {"kind": "returns", "value": 8},
                    },
                ],
            },
        ],
        "uncovered": [{"criterion": 3, "reason": "not executable"}],
    }


class _Constructor:
    def __init__(self, seed: Seed, base: Path, reply: dict[str, Any] | None = None) -> None:
        self.outcome = ConstructionOutcome(
            package_from_reply(
                reply or _reply(), seed, input_digest="1" * 64, generator="fake", base_checkout=base
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
) -> tuple[Seed, BoundaryRunState]:
    """Prepare (switch on), dispatch the worker, and let it stop; the package never decides."""
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
        settings=CheckPackageSettings(True, max_construction_attempts=1),
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


async def _resume(store: EventStore, seed: Seed, repo: Path) -> ResumedCheckPackageAuthority:
    """The resumed authority, as run_control installs it (``test_run_control`` covers that step)."""
    boundary = await load_resumed_boundary(store, EXECUTION)
    assert boundary is not None
    return ResumedCheckPackageAuthority(
        boundary,
        event_store=store,
        candidate_checkout=repo,
    )


async def test_after_a_crash_the_visible_cases_decide_and_held_out_ones_are_unavailable(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)  # the worker's final workspace
    forget_live_state(state)  # the controller process died: its memory is gone
    authority = await _resume(store, seed, repo)
    assert authority.boundary.source == "record"
    assert authority.boundary.held_out_checks == frozenset({"oracle_1"})

    # Criterion 1 passes its visible case but its held-out case is gone.
    # Criterion 2's only check is a preservation check, so its pass verifies
    # nothing and the legacy verifier's rejection decides it.
    decided = await authority(
        seed=seed, execution_id=EXECUTION, parallel_result=_restored(legacy_rejected=(1,))
    )
    keys = seed_criterion_keys(seed)
    verdicts = authority.outcome.verdict.verdicts
    assert (verdicts[keys[0]].status, verdicts[keys[0]].reason) == (
        PackageCriterionStatus.INDETERMINATE,
        HELD_OUT_UNAVAILABLE,
    )
    assert (verdicts[keys[1]].status, verdicts[keys[1]].reason) == (
        PackageCriterionStatus.UNVERIFIED,
        "no_reproduction_check",
    )
    assert verdicts[keys[2]].status is PackageCriterionStatus.UNCOVERED
    assert [result.outcome for result in decided.results] == [
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.SUCCEEDED,
    ]
    assert not decided.all_succeeded  # non-zero exit
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
    (resumed,) = [event for event in events if event.type == ACCEPTANCE_RESUMED]
    assert resumed.data["package_id"] == state.package.package_id
    assert resumed.data["held_out_checks"] == ["oracle_1"]
    assert verify_boundary_order(events) == ()
    assert any("visible cases" in line for line in authority.render())
    # A second call keeps the first decision.
    again = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    assert again.all_succeeded


async def test_after_a_crash_a_visible_failure_fails_a_criterion_legacy_accepted(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    # The worker left clamp broken; the legacy verifier accepted everything.
    forget_live_state(state)
    authority = await _resume(store, seed, repo)
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
    authority = await _resume(store, seed, repo)
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
    authority = await _resume(store, seed, repo)
    assert authority.boundary.source == "memory"
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    assert verdicts[seed_criterion_keys(seed)[0]].status is PackageCriterionStatus.PASS
    assert decided.all_succeeded
    # The final verdict exists: the in-process state is dropped.
    assert live_state(EXECUTION) is None


# a resumed run counts attempts exactly as the live arm-on run does.


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
    """A failed session and a failed verify command stay failed.

    Criterion 2 passes its visible cases and criterion 3 is uncovered, so the
    package alone would accept both. The live arm-on run does not count
    either as an attempt, and neither does the resumed run.
    """
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    authority = await _resume(store, seed, repo)
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
    authority = await _resume(store, seed, repo)
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


# the store must agree with the journal before any of it is used.


def _record_path(tmp_path: Path) -> Path:
    (path,) = list((tmp_path / "store" / "packages").glob("*.json"))
    return path


def _delete_held_out(record: dict[str, Any]) -> None:
    for spec in record["package"]["oracles"]:
        spec["cases"] = [case for case in spec["cases"] if not case.get("held_out")]


def _swap_held_out(record: dict[str, Any]) -> None:
    """Counts and ids unchanged: the stated case becomes held out, the held-out one visible."""
    stated, _held = record["package"]["oracles"][0]["cases"]
    record["package"]["oracles"][0]["cases"] = [
        {"case_id": "c1", "held_out": True},
        {**stated, "case_id": "c2"},
    ]


def _add_case(record: dict[str, Any]) -> None:
    spec = record["package"]["oracles"][0]
    spec["cases"].append({**spec["cases"][0], "case_id": f"c{len(spec['cases']) + 1}"})


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
    """The target edits the stored record, then kills the controller.

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
    authority = await _resume(store, seed, repo)
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
    authority = await _resume(store, seed, repo)
    assert authority.boundary.package is not None
    assert authority.boundary.held_out_checks == frozenset({"oracle_1"})


# the outcome summary of a resumed run is the original run's, from the journal.


# an unreadable journal refuses the resume instead of handing it to legacy.


# R5 follow-up: the record's digest is in the journal, so a visible value edit shows.


def _edit_visible_value(record: dict[str, Any]) -> None:
    """Ids, flags and counts unchanged: only the stated case's expected value moves."""
    stated = record["package"]["oracles"][0]["cases"][0]
    stated["expect"]["value"] = 15


async def test_an_edited_visible_value_is_detected_on_resume(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from ouroboros.boundary.package import sha256_bytes
    from ouroboros.boundary.resume import record_mismatch

    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
    frozen, admission = events[0], events[1]
    path = _record_path(tmp_path)
    assert frozen.data["record_sha256"] == sha256_bytes(path.read_bytes())
    record = json.loads(path.read_text())
    _edit_visible_value(record)
    # The manifest and admission checks alone cannot see this edit.
    assert record_mismatch(record, frozen.data["manifest"], admission.data) is None
    path.write_text(json.dumps(record))
    forget_live_state(state)  # the buggy clamp returns 15 for the stated case
    authority = await _resume(store, seed, repo)
    assert authority.boundary.package is None
    assert authority.boundary.reason == "package_record_tampered"
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    key = seed_criterion_keys(seed)[0]
    assert (verdicts[key].status, verdicts[key].reason) == (
        PackageCriterionStatus.INDETERMINATE,
        "package_record_tampered",
    )
    assert not decided.all_succeeded


async def test_after_a_crash_another_interpreter_leaves_covered_criteria_undecided(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The worker plants a project virtualenv whose python is another binary:
    # the resumed run resolves it, finds it is not the pin the admission
    # recorded, and runs nothing.
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    venv = repo / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "bin" / "python3").symlink_to(shutil.which("true") or "/usr/bin/true")
    forget_live_state(state)
    authority = await _resume(store, seed, repo)
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    keys = seed_criterion_keys(seed)
    assert [verdicts[key].reason for key in keys[:2]] == [INTERPRETER_CHANGED] * 2
    assert verdicts[keys[2]].status is PackageCriterionStatus.UNCOVERED
    assert [result.outcome for result in decided.results[:2]] == [ACExecutionOutcome.FAILED] * 2
    assert not decided.all_succeeded


@pytest.mark.parametrize("field", ["interpreter_sha256", "interpreter_realpath_sha256"])
async def test_after_a_crash_a_missing_interpreter_pin_leaves_covered_criteria_undecided(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    authority = await _resume(store, seed, repo)
    assert authority.boundary.interpreter_realpath_sha256  # recorded before the worker started
    authority.boundary = replace(authority.boundary, **{field: None})
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdicts = authority.outcome.verdict.verdicts
    keys = seed_criterion_keys(seed)
    assert [verdicts[key].reason for key in keys[:2]] == [INTERPRETER_CHANGED] * 2
    assert not decided.all_succeeded


async def test_a_resume_error_leaves_covered_criteria_undecided_and_legacy_decides_the_rest(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The live authority's fail-closed rule, on resume: an error while
    # recomputing never accepts a covered criterion, and never fails an
    # uncovered one the legacy verifier accepted.
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    authority = await _resume(store, seed, repo)

    async def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("ouroboros.boundary.resume.decide_resumed", broken)
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    assert authority.outcome.error == "OSError"
    decisions = authority.outcome.reconciliation.decisions
    assert [d.reason for d in decisions[:2]] == ["authority_error:OSError"] * 2
    assert [d.accepted for d in decisions] == [False, False, True]
    assert [result.outcome for result in decided.results] == [
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.FAILED,
        ACExecutionOutcome.SUCCEEDED,  # uncovered, the legacy verifier accepted it
    ]
    assert any("covered criteria are undecided" in line for line in authority.render())


async def test_a_resume_error_reading_the_legacy_verdicts_fails_every_root(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reading the legacy verdicts is inside the fail-closed scope too: when no
    # decision can be built, every root fails; the restored result is never
    # returned unchanged.
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    forget_live_state(state)
    authority = await _resume(store, seed, repo)

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyError("legacy")

    monkeypatch.setattr("ouroboros.boundary.resume.existing_outcomes_from_results", broken)
    monkeypatch.setattr("ouroboros.boundary.authority.existing_outcomes_from_results", broken)
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    assert authority.outcome.error == "KeyError"
    assert (decided.success_count, decided.failure_count) == (0, 3)
    assert not decided.all_succeeded


# MEDIUM 3: with the check package on, the journal cannot buy a legacy decision
# by omission.


async def _journal_without(
    store: EventStore, drop: Any, *, versions: int = 1, keep_enabled: bool = True
) -> EventStore:
    """A copy of the run's boundary journal without the events ``drop`` selects."""
    copy = EventStore("sqlite+aiosqlite:///:memory:")
    await copy.initialize()
    aggregates = [f"{EXECUTION}/check_package/v{n}" for n in range(1, versions + 1)]
    if keep_enabled:
        aggregates.insert(0, EXECUTION)
    for aggregate in aggregates:
        for event in await store.replay(BOUNDARY_AGGREGATE_TYPE, aggregate):
            if not drop(event):
                await copy.append(event)
    return copy


async def _decide_undecidable(journal: EventStore, seed: Seed, repo: Path) -> Any:
    authority = await _resume(journal, seed, repo)
    assert authority.boundary.reason == BOUNDARY_RECORD_MISSING
    assert authority.boundary.covered is None
    decided = await authority(
        seed=seed, execution_id=EXECUTION, parallel_result=_restored()
    )  # the legacy verifier accepted every root
    verdicts = authority.outcome.verdict.verdicts
    assert {item.status for item in verdicts.values()} == {PackageCriterionStatus.INDETERMINATE}
    assert {item.reason for item in verdicts.values()} == {BOUNDARY_RECORD_MISSING}
    assert [result.outcome for result in decided.results] == [ACExecutionOutcome.FAILED] * 3
    return decided


async def test_deleted_boundary_versions_leave_every_criterion_undecided(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    journal = await _journal_without(store, lambda event: event.aggregate_id != EXECUTION)
    decided = await _decide_undecidable(journal, seed, repo)
    assert not decided.all_succeeded
    (resumed,) = [
        event
        for event in await journal.replay(BOUNDARY_AGGREGATE_TYPE, EXECUTION)
        if event.type == ACCEPTANCE_RESUMED
    ]
    assert resumed.data["package_id"] is None
    assert resumed.data["reason"] == BOUNDARY_RECORD_MISSING


@pytest.mark.parametrize("dropped", [ADMISSION_COMPLETED, PACKAGE_FROZEN])
async def test_a_bound_version_missing_its_seal_or_admission_leaves_every_criterion_undecided(
    store: EventStore,
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dropped: str,
) -> None:
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    forget_live_state(state)
    journal = await _journal_without(store, lambda event: event.type == dropped)
    decided = await _decide_undecidable(journal, seed, repo)
    assert not decided.all_succeeded


async def test_a_deleted_enabled_record_with_the_boundary_intact_leaves_every_criterion_undecided(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The boundary versions prove the package was on, but the settings the run
    # started with went with the enabled record: no check runs with the live
    # config instead, and nothing is left to the legacy verifier.
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    forget_live_state(state)
    journal = await _journal_without(store, lambda _event: False, keep_enabled=False)
    authority = await _resume(journal, seed, repo)
    assert authority.boundary.package is None and authority.boundary.covered is None
    assert authority.boundary.reason == BOUNDARY_RECORD_MISSING
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    assert not decided.all_succeeded
    assert all(r.outcome is ACExecutionOutcome.FAILED for r in decided.results)


async def test_a_resume_uses_the_check_timeout_the_run_started_with(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The live config changed after the run started; the resume keeps the
    # recorded contract, in another process as in the same one.
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    recorded = await BoundaryLedger(store).run_contract(EXECUTION)
    assert recorded is not None
    seen: list[float] = []
    real = resume_module.verify_with_bindings

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["timeout_seconds"])
        return await real(*args, **kwargs)

    monkeypatch.setattr(resume_module, "verify_with_bindings", spy)
    for same_process in (True, False):
        if not same_process:
            forget_live_state(state)
        authority = await _resume(store, seed, repo)
        assert authority.boundary.contract == recorded
        await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    assert seen == [recorded.check_timeout_seconds] * 2


async def test_a_version_gap_leaves_every_criterion_undecided(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #2458 round 3: versions were found by counting up from v1, which a gap
    # stops. Resume reads every version the journal holds; the product writes
    # v1, v2, ... in order, so a gap is a journal it did not write.
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    forget_live_state(state)
    journal = await _journal_without(store, lambda _event: False)
    assert state.package is not None
    await journal.append(package_frozen_event(boundary_version_id(EXECUTION, 3), state.package))
    await _decide_undecidable(journal, seed, repo)


async def test_a_later_version_without_the_first_is_never_a_legacy_run(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With the enabled record and v1 gone, a v2 still proves the package was
    # on: undecided, never the legacy verifier (counting from v1 found nothing).
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    forget_live_state(state)
    journal = await _journal_without(store, lambda _event: True, keep_enabled=False)
    assert state.package is not None
    await journal.append(package_frozen_event(boundary_version_id(EXECUTION, 2), state.package))
    boundary = await load_resumed_boundary(journal, EXECUTION)
    assert boundary is not None and boundary.covered is None
    assert boundary.reason == BOUNDARY_RECORD_MISSING


async def test_a_journal_with_no_record_of_the_run_resumes_as_the_legacy_run(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Documented residual: the journal is writable by the same user; removing
    # every record of the run, the enabled record included, reads as "off".
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    forget_live_state(state)
    journal = await _journal_without(store, lambda _event: True)
    assert await load_resumed_boundary(journal, EXECUTION) is None


async def test_a_worker_bound_to_a_failed_construction_resumes_as_the_legacy_run(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    class _Failing:
        async def construct(self, seed: Seed, base: Path, *, feedback=()) -> ConstructionOutcome:
            return ConstructionOutcome(None, "constructor_timeout", "1" * 64, "fake")

    seed = _seed()
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=_Failing(),
        execution_id=EXECUTION,
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=CheckPackageSettings(True, max_construction_attempts=1),
        store_dir=tmp_path / "store",
    )
    assert not state.admitted
    assert await load_resumed_boundary(store, EXECUTION, store_dir=tmp_path / "store") is None


async def test_a_resume_without_an_execution_id_is_an_error(store: EventStore) -> None:
    with pytest.raises(BoundaryOrderError):
        await load_resumed_boundary(store, "")


# One recovery plan, fixed before any held-out case is dropped.


def _reply_with_a_held_out_only_check() -> dict[str, Any]:
    """Criterion 1: a visible-only oracle plus an oracle whose every case is held out."""
    reply = _reply()
    clamp = reply["oracles"][0]
    visible = {**clamp, "check_id": "oracle_1", "cases": clamp["cases"][:1]}
    hidden = {**clamp, "check_id": "oracle_1_2", "cases": clamp["cases"][1:]}
    reply["oracles"] = [visible, hidden, *reply["oracles"][1:]]
    return reply


async def test_a_check_dropped_for_being_fully_held_out_still_withholds_its_criterion(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review probe: after a crash in another process the fully held-out
    # check is gone from the visible package, and the visible check passes.
    monkeypatch.setattr(
        "ouroboros.boundary.resume.default_store_dir", lambda _execution: tmp_path / "store"
    )
    seed = _seed()
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=_Constructor(seed, repo, _reply_with_a_held_out_only_check()),
        execution_id=EXECUTION,
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=CheckPackageSettings(True, max_construction_attempts=1),
        store_dir=tmp_path / "store",
    )
    assert state.admitted
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    authority = await _resume(store, seed, repo)
    assert authority.boundary.source == "record"
    assert authority.boundary.held_out_checks == frozenset({"oracle_1_2"})
    keys = seed_criterion_keys(seed)
    assert authority.boundary.held_out_criteria == frozenset({keys[0]})
    assert authority.boundary.package is not None
    assert "oracle_1_2" not in {check.check_id for check in authority.boundary.package.checks}
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdict = authority.outcome.verdict.verdicts[keys[0]]
    assert (verdict.status, verdict.reason) == (
        PackageCriterionStatus.INDETERMINATE,
        HELD_OUT_UNAVAILABLE,
    )
    assert decided.results[0].outcome is ACExecutionOutcome.FAILED  # never legacy-decided


async def _journal_with_manifest(
    store: EventStore, edit: Any, *, event_type: str = PACKAGE_FROZEN
) -> EventStore:
    """A copy of the run's journal whose ``event_type`` record ``edit`` changed."""
    copy = EventStore("sqlite+aiosqlite:///:memory:")
    await copy.initialize()
    for aggregate in (EXECUTION, f"{EXECUTION}/check_package/v1"):
        for event in await store.replay(BOUNDARY_AGGREGATE_TYPE, aggregate):
            if event.type == event_type:
                data = copy_module.deepcopy(dict(event.data))
                edit(data)
                event = event.model_copy(update={"data": data})
            await copy.append(event)
    return copy


def _drop_manifest(data: dict[str, Any]) -> None:
    del data["manifest"]


def _shorten_manifest(data: dict[str, Any]) -> None:
    manifest = dict(data["manifest"])
    manifest["checks"] = [c for c in manifest["checks"] if c["check_id"] != "oracle_1"]
    manifest["oracles"] = [o for o in manifest["oracles"] if o["check_id"] != "oracle_1"]
    data["manifest"] = manifest


@pytest.mark.parametrize("edit", [_drop_manifest, _shorten_manifest], ids=["missing", "shortened"])
async def test_a_missing_or_shortened_manifest_leaves_every_criterion_undecided(
    store: EventStore,
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: Any,
) -> None:
    # Never ``covered=()``: an unreadable coverage record must not hand the
    # covered criteria to the legacy verifier.
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    journal = await _journal_with_manifest(store, edit)
    decided = await _decide_undecidable(journal, seed, repo)
    assert not decided.all_succeeded


def _manifest(checks: list[dict[str, Any]], oracles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "package_id": "p" * 64,
        "seed_digest": "s" * 64,
        "criterion_keys": ["k1", "k2"],
        "checks": checks,
        "uncovered": [],
        "oracles": oracles,
    }


def _check(check_id: str, role: str, *keys: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "role": role,
        "criterion_keys": list(keys),
        "assertion_ids": [f"{check_id}.case"],
    }


def _oracle(check_id: str, key: str, held_out: int) -> dict[str, Any]:
    return {"check_id": check_id, "criterion_key": key, "held_out_count": held_out}


_BASE_REASON = {
    "reproduction": "reproduction_passed_on_base",
    "preservation": "preservation_failed",
}
_EXCLUSION = {"reproduction": "repro_passes_on_base", "preservation": "preservation_fails_on_base"}


def _admission(manifest: dict[str, Any], excluded: tuple[str, ...] = ()) -> dict[str, Any]:
    """The admission record ``admit_check_package`` writes for ``manifest``."""
    roles = {check["check_id"]: check["role"] for check in manifest["checks"]}
    return {
        "package_id": manifest["package_id"],
        "seed_digest": manifest["seed_digest"],
        "verdict": "admitted",
        "protected_bytes_mutated": False,
        "base_tree_digest": "b" * 64,
        "base_tree_digest_after": "b" * 64,
        "check_tiers": {check_id: "C" if check_id in excluded else "A" for check_id in roles},
        "excluded_checks": {check_id: _EXCLUSION[roles[check_id]] for check_id in excluded} or None,
        "checks": [
            {
                "check_id": check_id,
                "role": role,
                "status": "violated" if check_id in excluded else "expected",
                "reason": _BASE_REASON[role] if check_id in excluded else "passed",
            }
            for check_id, role in roles.items()
        ],
    }


def test_the_recovery_plan_applies_admission_exclusions_and_the_reproduction_rule() -> None:
    manifest = _manifest(
        [
            _check("repro_1", "reproduction", "k1"),
            _check("keep_1", "preservation", "k1"),
            _check("repro_2", "reproduction", "k2"),
            _check("hidden_2", "reproduction", "k2"),
        ],
        [
            _oracle("repro_1", "k1", 2),
            _oracle("keep_1", "k1", 0),
            _oracle("repro_2", "k2", 0),
            _oracle("hidden_2", "k2", 3),
        ],
    )
    # repro_1 excluded: k1 keeps only a preservation check, so it is not covered
    # (and its held-out cases no longer count); hidden_2 excluded: k2 stays
    # covered by repro_2, which had no held-out case.
    plan = recovery_plan(manifest, _admission(manifest, ("repro_1", "hidden_2")), "p" * 64)
    assert plan is not None
    assert plan.covered == frozenset({"k2"})
    assert plan.held_out_checks == frozenset()
    assert plan.held_out_criteria == frozenset()
    admitted = recovery_plan(manifest, _admission(manifest), "p" * 64)
    assert admitted is not None
    assert admitted.covered == frozenset({"k1", "k2"})
    assert admitted.held_out_criteria == frozenset({"k1", "k2"})


def _set_tier(check_id: str, tier: str) -> Any:
    return lambda a: a["check_tiers"].update({check_id: tier})


@pytest.mark.parametrize(
    ("manifest_edit", "admission_edit"),
    [
        (lambda m: m.update(package_id="q" * 64), lambda _a: None),
        (lambda m: m.update(checks=m["checks"][:1]), lambda _a: None),
        (lambda m: m.update(oracles=[_oracle("ghost", "k1", 1)]), lambda _a: None),
        (lambda m: m.update(oracles=[_oracle("repro_1", "k1", -1)]), lambda _a: None),
        (lambda _m: None, _set_tier("ghost", "C")),
        (lambda _m: None, lambda a: a.pop("check_tiers")),
    ],
    ids=["package_id", "shortened", "unknown_oracle", "held_out_count", "unknown_tier", "no_tiers"],
)
def test_a_disagreeing_record_yields_no_recovery_plan(
    manifest_edit: Any, admission_edit: Any
) -> None:
    manifest = _manifest(
        [_check("repro_1", "reproduction", "k1"), _check("repro_2", "reproduction", "k2")],
        [_oracle("repro_1", "k1", 1), _oracle("repro_2", "k2", 1)],
    )
    admission = _admission(manifest)
    manifest_edit(manifest)
    admission_edit(admission)
    assert recovery_plan(manifest, admission, "p" * 64) is None


def _every_tier_c(admission: dict[str, Any]) -> None:
    """The review probe's edit: every check excluded, the verdict still admitted."""
    admission["check_tiers"] = dict.fromkeys(admission["check_tiers"], "C")


def _result(check_id: str, **update: Any) -> Any:
    def edit(admission: dict[str, Any]) -> None:
        for item in admission["checks"]:
            if item["check_id"] == check_id:
                item.update(update)

    return edit


@pytest.mark.parametrize(
    "edit",
    [
        lambda a: a.update(verdict="rejected"),
        lambda a: a.update(seed_digest="t" * 64),
        lambda a: a.update(protected_bytes_mutated=True),
        lambda a: a.update(base_tree_digest_after="c" * 64),
        lambda a: a["check_tiers"].pop("repro_2"),
        _set_tier("repro_2", "A_prime"),
        _set_tier("repro_2", "C"),
        lambda a: a.update(excluded_checks={**a["excluded_checks"], "repro_2": "x"}),
        _every_tier_c,
        lambda a: a["checks"].append(dict(a["checks"][0])),
        lambda a: a["checks"].pop(),
        _result("repro_2", role="preservation"),
        _result("repro_1", status="expected"),
        _result("repro_1", reason="preservation_failed"),
        lambda a: a["excluded_checks"].update(repro_1="preservation_fails_on_base"),
        _result("repro_2", status="violated", reason="reproduction_passed_on_base"),
        _result("repro_2", status="indeterminate"),
    ],
    ids=[
        "not_admitted",
        "other_seed",
        "protected_bytes_mutated",
        "base_changed",
        "incomplete_tiers",
        "tier_after_admission",
        "tier_c_not_excluded",
        "excluded_without_tier_c",
        "every_check_excluded",
        "duplicate_result",
        "missing_result",
        "role_differs_from_manifest",
        "excluded_check_passed",
        "excluded_for_another_role_reason",
        "exclusion_reason_of_another_role",
        "admitted_check_violated",
        "admitted_check_undecided",
    ],
)
def test_an_admission_record_admission_could_not_write_yields_no_recovery_plan(
    edit: Any,
) -> None:
    manifest = _manifest(
        [_check("repro_1", "reproduction", "k1"), _check("repro_2", "reproduction", "k2")],
        [_oracle("repro_1", "k1", 1), _oracle("repro_2", "k2", 1)],
    )
    admission = _admission(manifest, ("repro_1",))
    assert recovery_plan(manifest, admission, "p" * 64) is not None  # the control
    edit(admission)
    assert recovery_plan(manifest, admission, "p" * 64) is None


async def test_an_admission_record_with_every_check_excluded_leaves_every_criterion_undecided(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review probe: every recorded tier rewritten to C, verdict still
    # admitted; the legacy verifier accepted every root. Never covered=().
    seed, state = await _run_until_the_worker_stops(store, repo, tmp_path, monkeypatch)
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    journal = await _journal_with_manifest(store, _every_tier_c, event_type=ADMISSION_COMPLETED)
    decided = await _decide_undecidable(journal, seed, repo)
    assert not decided.all_succeeded


def _reply_whose_criterion_has_only_a_held_out_check() -> dict[str, Any]:
    """Criterion 1: its only oracle has held-out cases and no visible case."""
    reply = _reply()
    clamp = reply["oracles"][0]
    reply["oracles"] = [
        {**clamp, "check_id": "oracle_1", "cases": clamp["cases"][1:]},
        *reply["oracles"][1:],
    ]
    return reply


async def test_a_criterion_whose_only_check_was_fully_held_out_is_held_out_unavailable(
    store: EventStore, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The visible package has no check left for the criterion; the plan's
    # held-out rule decides it, not the visible package's "uncovered".
    monkeypatch.setattr(
        "ouroboros.boundary.resume.default_store_dir", lambda _execution: tmp_path / "store"
    )
    seed = _seed()
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=_Constructor(seed, repo, _reply_whose_criterion_has_only_a_held_out_check()),
        execution_id=EXECUTION,
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="codex",
        settings=CheckPackageSettings(True, max_construction_attempts=1),
        store_dir=tmp_path / "store",
    )
    assert state.admitted
    (repo / "mathutils.py").write_text(CLAMP_FIXED + DOUBLE)
    forget_live_state(state)
    authority = await _resume(store, seed, repo)
    keys = seed_criterion_keys(seed)
    assert authority.boundary.source == "record"
    assert keys[0] in (authority.boundary.covered or ())
    assert authority.boundary.package is not None
    assert keys[0] not in {
        link.criterion_key
        for check in authority.boundary.package.checks
        for link in check.assertions
    }
    decided = await authority(seed=seed, execution_id=EXECUTION, parallel_result=_restored())
    verdict = authority.outcome.verdict.verdicts[keys[0]]
    assert (verdict.status, verdict.reason) == (
        PackageCriterionStatus.INDETERMINATE,
        HELD_OUT_UNAVAILABLE,
    )
    assert decided.results[0].outcome is ACExecutionOutcome.FAILED  # never legacy-decided
