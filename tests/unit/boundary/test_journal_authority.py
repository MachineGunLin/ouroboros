"""Records with no valid authority never enter an authoritative phase (round 2).

The gateway closes each record's schema; the reducer relates it to what the
journal already holds. A record the product could not have written in that
state is refused on write and flagged on replay: an incomplete final binding,
a decision that accepts what neither the package's recorded results nor the
existing verifier accepted, a frozen manifest that is not the package's
projection, an admission without its interpreter pin, a verification of
checks the bindings did not run, and a later version the product could not
have sealed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.events import (
    ACTOR_STARTED,
    BOUNDARY_AGGREGATE_TYPE,
    CONSTRUCTION_FAILED,
    BindingsPayload,
    ReconciliationPayload,
    RunContract,
    acceptance_reconciled_event,
    actor_started_event,
    admission_completed_event,
    binding_recorded_event,
    boundary_version_id,
    construction_failed_event,
    package_frozen_event,
)
from ouroboros.boundary.ledger import (
    BoundaryLedger,
    BoundaryOrderError,
    Phase,
    RecoveryBound,
    RecoveryUndecidable,
    recovery_projection,
    verify_boundary_order,
    version_state,
)
from ouroboros.boundary.oracle import CaseResult, OracleResult
from ouroboros.boundary.package import CheckPackage, seal_package
from ouroboros.boundary.receipts import CheckStatus
from ouroboros.events.base import BaseEvent
from ouroboros.persistence.event_store import EventStore

from .conftest import REPRO_SCRIPT, build_package
from .journal_fixtures import admission_receipt, expected_execution, verification_receipt
from .test_package_identity import held_out_package

B = "task-1/V1"
TIERS = ("A", "A_prime", "U", "S", "C")
CONTRACT = RunContract(check_timeout_seconds=120)


@pytest.fixture
async def store():
    event_store = EventStore("sqlite+aiosqlite:///:memory:")
    await event_store.initialize()
    yield event_store
    await event_store.close()


def _criterion(index: int, key: str, status: str, **update: Any) -> dict[str, Any]:
    return {
        "root_ac_index": index,
        "criterion_key": key,
        "package_status": status,
        "tier": "U" if status in ("unverified", "uncovered") else "A",
        "reason": status,
        "failed_heldout_only": False,
        "binding": None,
        "existing_outcome": "succeeded",
        "existing_failure_class": None,
        "existing_accepted": True,
        "accepted": status in ("pass", "unverified", "uncovered"),
        "governed_by": "check_package",
        "declared_binding_pass": False,
        **update,
    }


def _decision(criteria: list[dict[str, Any]], **extra: Any) -> ReconciliationPayload:
    """A decision under the product's rule; its summary written out by hand."""
    statuses = [item["package_status"] for item in criteria]
    verdict = next((v for v in ("fail", "indeterminate", "pass") if v in statuses), "unverified")
    not_decided = [i for i in criteria if i["package_status"] in ("unverified", "uncovered")]
    unverified = [i for i in not_decided if i["governed_by"] != "existing_verifier"]
    loose = [i for i in unverified if i["governed_by"] != "execution"]
    coverage = (
        "low"
        if loose or (criteria and len(not_decided) / len(criteria) >= 0.5)
        else "partial"
        if not_decided
        else "full"
    )
    data = {
        "schema_version": "ouroboros.acceptance_reconciliation.v3",
        "run_accepted": bool(criteria) and all(i["accepted"] for i in criteria),
        "existing_run_accepted": True,
        "artifact_verdict": verdict,
        "verified_pass_count": statuses.count("pass"),
        "unverified_count": len(unverified),
        "criterion_count": len(criteria),
        "tier_summary": {tier: sum(1 for i in criteria if i["tier"] == tier) for tier in TIERS},
        "criteria": criteria,
        "legacy_decided_count": sum(1 for i in criteria if i["governed_by"] == "existing_verifier"),
        "verification_coverage": coverage,
        **extra,
    }
    return ReconciliationPayload.model_validate(data)


def _binding(check_id: str, key: str, tier: str, **update: Any) -> dict[str, Any]:
    return {
        "criterion_key": key,
        "check_id": check_id,
        "tier": tier,
        "binding_source": None,
        "binding": None,
        "status_hint": "run",
        "reason": "script_check",
        "declared": None,
        **update,
    }


def _script_bindings(package: CheckPackage, phase: str = "final") -> BindingsPayload:
    checks = [
        _binding(check.check_id, check.assertions[0].criterion_key, "S") for check in package.checks
    ]
    return BindingsPayload.model_validate({"phase": phase, "checks": checks})


async def _started(store: EventStore, package: CheckPackage, checkout: Path) -> BoundaryLedger:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen(B, package)
    await ledger.record_admission(B, admission_receipt(package, checkout))
    await ledger.record_actor_started("actor-1", [B])
    return ledger


# --------------------------------------------------------------------------
# The bot's minimal reproduction, through public BoundaryLedger calls.


async def test_an_empty_final_binding_of_a_two_check_package_is_refused(
    store, package, base_checkout
) -> None:
    ledger = await _started(store, package, base_checkout)
    empty = BindingsPayload(phase="final", checks=())
    with pytest.raises(BoundaryOrderError):
        await ledger.record_bindings(B, package_id=package.package_id, payload=empty)
    # One check of two is incomplete too.
    one = BindingsPayload.model_validate(
        {"phase": "final", "checks": [_script_bindings(package).checks[0].model_dump()]}
    )
    with pytest.raises(BoundaryOrderError):
        await ledger.record_bindings(B, package_id=package.package_id, payload=one)
    await ledger.record_bindings(
        B, package_id=package.package_id, payload=_script_bindings(package)
    )


async def test_an_accepted_decision_without_any_authority_never_reaches_decided(
    store, package, base_checkout
) -> None:
    ledger = await _started(store, package, base_checkout)
    # The empty final binding, written around the ledger.
    empty = BindingsPayload(phase="final", checks=())
    await store.append(binding_recorded_event(B, package_id=package.package_id, payload=empty))
    # Every criterion unverified; the existing verifier rejected each one;
    # the package accepts them all anyway.
    unearned = _decision(
        [
            _criterion(index, key, "unverified", existing_accepted=False)
            for index, key in enumerate(package.criterion_keys)
        ]
    )
    assert unearned.run_accepted
    with pytest.raises(BoundaryOrderError):
        await ledger.record_acceptance_reconciled(
            B, package_id=package.package_id, reconciliation=unearned
        )
    await store.append(
        acceptance_reconciled_event(B, package_id=package.package_id, reconciliation=unearned)
    )
    events = await ledger.events(B)
    assert verify_boundary_order(events) != ()
    with pytest.raises(BoundaryOrderError):
        version_state(events)
    state = version_state(events[:3])
    assert state.phase is Phase.STARTED


# --------------------------------------------------------------------------
# The frozen manifest is exactly the package's projection.

_SEVEN = (
    "schema_version",
    "input_digest",
    "generated_at",
    "generator",
    "files",
    "base_files",
    "scratch_path_count",
)


def _enabled(run: str) -> list[BaseEvent]:
    event = BaseEvent(
        type="boundary.check_package.enabled",
        aggregate_type=BOUNDARY_AGGREGATE_TYPE,
        aggregate_id=run,
        data={"execution_id": run, "contract": CONTRACT.journal_data()},
    )
    return [event.model_copy(update={"timestamp": datetime(2026, 9, 29, tzinfo=UTC)})]


def _timed(events: list[BaseEvent]) -> list[BaseEvent]:
    start = datetime(2026, 9, 29, tzinfo=UTC)
    return [
        event.model_copy(update={"timestamp": start + timedelta(seconds=n + 1)})
        for n, event in enumerate(events)
    ]


def test_a_frozen_manifest_missing_projection_fields_makes_recovery_undecidable(
    package, base_checkout
) -> None:
    run = "run_manifest"
    version = boundary_version_id(run, 1)
    frozen = package_frozen_event(version, package)
    manifest = {k: v for k, v in frozen.data["manifest"].items() if k not in _SEVEN}
    partial = frozen.model_copy(update={"data": {**frozen.data, "manifest": manifest}})
    rest = [
        admission_completed_event(version, admission_receipt(package, base_checkout)),
        actor_started_event(version, actor_id=run, package_id=package.package_id, runtime=None),
    ]
    genuine = recovery_projection(run, _enabled(run), {1: _timed([frozen, *rest])})
    assert isinstance(genuine, RecoveryBound)
    assert verify_boundary_order(_timed([partial, *rest])) != ()
    projection = recovery_projection(run, _enabled(run), {1: _timed([partial, *rest])})
    assert isinstance(projection, RecoveryUndecidable)


@pytest.mark.parametrize("oracles", [False, True], ids=["script_checks", "oracle"])
def test_the_manifest_model_is_exactly_what_manifest_summary_emits(seed, oracles: bool) -> None:
    from pydantic import ValidationError

    from ouroboros.boundary.events import ManifestRecord

    package = seal_package(held_out_package() if oracles else build_package(seed))
    summary = package.manifest_summary()
    record = ManifestRecord.model_validate(summary)
    assert record.model_dump(mode="json", exclude_unset=True) == summary
    if oracles:
        # Every field the model defines is one the projection emits.
        assert set(ManifestRecord.model_fields) == set(summary)
    for key in summary:
        with pytest.raises(ValidationError):
            ManifestRecord.model_validate({k: v for k, v in summary.items() if k != key})
    with pytest.raises(ValidationError):
        ManifestRecord.model_validate({**summary, "argv": ["python3", "x.py"]})


# --------------------------------------------------------------------------
# An admitted record carries both interpreter pins.


async def test_an_admitted_record_without_its_interpreter_pins_is_refused(
    store, package, base_checkout
) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen(B, package)
    unpinned = admission_receipt(package, base_checkout).model_copy(
        update={"interpreter_sha256": None, "interpreter_realpath_sha256": None}
    )
    with pytest.raises(BoundaryOrderError):
        await ledger.record_admission(B, unpinned)
    await store.append(admission_completed_event(B, unpinned))
    await store.append(
        actor_started_event(B, actor_id="actor-1", package_id=package.package_id, runtime=None)
    )
    events = await ledger.events(B)
    assert verify_boundary_order(events) != ()
    with pytest.raises(BoundaryOrderError):
        version_state(events)


# --------------------------------------------------------------------------
# A decision records only the package statuses the recorded results support.


async def _script_verified(
    store: EventStore, package: CheckPackage, checkout: Path, *checks: Any
) -> BoundaryLedger:
    ledger = await _started(store, package, checkout)
    await ledger.record_bindings(
        B, package_id=package.package_id, payload=_script_bindings(package)
    )
    verification = verification_receipt(package, checkout).model_copy(update={"checks": checks})
    await ledger.record_candidate_verification(B, verification)
    return ledger


@pytest.mark.parametrize(
    "first",
    [
        # A failed check recorded as a criterion the package did not verify:
        # it would hand a package failure to the legacy verifier.
        {
            "package_status": "unverified",
            "tier": "U",
            "governed_by": "existing_verifier",
            "accepted": True,
        },
        # A pass the results do not show.
        {"package_status": "pass", "accepted": True},
    ],
    ids=["failure_handed_to_legacy", "pass_without_results"],
)
async def test_a_decision_the_results_do_not_support_is_refused(
    store, package, base_checkout, first: dict[str, Any]
) -> None:
    violated = expected_execution(package.checks[0]).model_copy(
        update={"status": CheckStatus.VIOLATED, "reason": "reproduction_failed"}
    )
    ledger = await _script_verified(store, package, base_checkout, violated)
    keys = package.criterion_keys
    rest = [_criterion(1, keys[1], "indeterminate"), _criterion(2, keys[2], "uncovered")]
    forged = _decision([_criterion(0, keys[0], "fail", **first), *rest])
    with pytest.raises(BoundaryOrderError):
        await ledger.record_acceptance_reconciled(
            B, package_id=package.package_id, reconciliation=forged
        )
    genuine = _decision([_criterion(0, keys[0], "fail"), *rest])
    await ledger.record_acceptance_reconciled(
        B, package_id=package.package_id, reconciliation=genuine
    )
    assert verify_boundary_order(await ledger.events(B)) == ()


def _oracle_run(package: CheckPackage, *, held_out_passed: bool) -> Any:
    check = package.checks[0]
    oracle = OracleResult(
        check_id=check.check_id,
        criterion_key=package.criterion_keys[0],
        binding_source="default",
        symbol="mathutils.clamp",
        call_kind="function",
        resolve="ok",
        cases=(
            CaseResult(case_id="c1", held_out=False, passed=True),
            CaseResult(case_id="c2", held_out=True, passed=held_out_passed),
        ),
    )
    return expected_execution(check).model_copy(
        update={
            "status": CheckStatus.EXPECTED,
            "reason": "reproduction_passed",
            "tier": CheckTier.A,
            "oracle_result": oracle,
        }
    )


async def _oracle_verified(
    store: EventStore, checkout: Path, *, held_out_passed: bool
) -> tuple[BoundaryLedger, CheckPackage]:
    package = seal_package(held_out_package())
    ledger = await _started(store, package, checkout)
    spec = package.oracle_for("oracle_1")
    assert spec is not None
    bound = _binding(
        "oracle_1",
        package.criterion_keys[0],
        "A",
        binding_source="default",
        binding=spec.default_binding.to_dict(),
        reason="default_binding_resolves",
    )
    payload = BindingsPayload.model_validate({"phase": "final", "checks": [bound]})
    await ledger.record_bindings(B, package_id=package.package_id, payload=payload)
    run = _oracle_run(package, held_out_passed=held_out_passed)
    verification = verification_receipt(package, checkout).model_copy(update={"checks": (run,)})
    await ledger.record_candidate_verification(B, verification)
    return ledger, package


async def test_a_verified_pass_needs_a_held_out_case_and_names_its_provenance(
    store, base_checkout
) -> None:
    ledger, package = await _oracle_verified(store, base_checkout, held_out_passed=True)
    (key,) = package.criterion_keys
    declared = _decision([_criterion(0, key, "pass", declared_binding_pass=True)])
    with pytest.raises(BoundaryOrderError):
        await ledger.record_acceptance_reconciled(
            B, package_id=package.package_id, reconciliation=declared
        )
    await ledger.record_acceptance_reconciled(
        B, package_id=package.package_id, reconciliation=_decision([_criterion(0, key, "pass")])
    )


async def test_a_pass_without_a_held_out_case_is_refused(store, base_checkout) -> None:
    ledger, package = await _oracle_verified(store, base_checkout, held_out_passed=False)
    (key,) = package.criterion_keys
    with pytest.raises(BoundaryOrderError):
        await ledger.record_acceptance_reconciled(
            B,
            package_id=package.package_id,
            reconciliation=_decision([_criterion(0, key, "pass")]),
        )
    await ledger.record_acceptance_reconciled(
        B,
        package_id=package.package_id,
        reconciliation=_decision([_criterion(0, key, "unverified")]),
    )


def test_a_decision_whose_summary_disagrees_with_its_criteria_is_refused(package) -> None:
    keys = package.criterion_keys
    genuine = _decision([_criterion(i, k, "indeterminate") for i, k in enumerate(keys)])
    for update in (
        {"verified_pass_count": 1},
        {"artifact_verdict": "pass"},
        {"unverified_count": 2},
    ):
        with pytest.raises(ValueError):
            ReconciliationPayload.model_validate({**genuine.model_dump(mode="json"), **update})


async def test_a_decision_outside_the_products_rule_is_not_journaled(
    store, package, base_checkout
) -> None:
    violated = expected_execution(package.checks[0]).model_copy(
        update={"status": CheckStatus.VIOLATED, "reason": "reproduction_failed"}
    )
    ledger = await _script_verified(store, package, base_checkout, violated)
    keys = package.criterion_keys
    criteria = [
        _criterion(0, keys[0], "fail"),
        _criterion(1, keys[1], "indeterminate"),
        _criterion(2, keys[2], "uncovered"),
    ]
    legacy_off = _decision(criteria, schema_version="ouroboros.acceptance_reconciliation.v2")
    with pytest.raises(BoundaryOrderError):
        await ledger.record_acceptance_reconciled(
            B, package_id=package.package_id, reconciliation=legacy_off
        )


# --------------------------------------------------------------------------
# Bindings and verifications the product could not write.


async def test_a_binding_tier_the_product_cannot_assign_is_refused(
    store, package, base_checkout
) -> None:
    ledger = await _started(store, package, base_checkout)
    good = _script_bindings(package).model_dump()["checks"]
    spec = {
        "criterion_key": package.criterion_keys[0],
        "symbol": "calc.add",
        "call_kind": "function",
    }
    for edit in (
        {"tier": "A", "binding_source": "default", "binding": spec},  # a script as tier A
        {"tier": "C", "status_hint": "excluded"},  # excluded though admitted
        {"criterion_key": package.criterion_keys[2]},  # a criterion the check does not link
    ):
        forged = BindingsPayload.model_validate(
            {"phase": "final", "checks": [{**good[0], **edit}, good[1]]}
        )
        with pytest.raises(BoundaryOrderError):
            await ledger.record_bindings(B, package_id=package.package_id, payload=forged)


async def test_a_rerun_of_a_check_that_met_its_role_is_refused(
    store, package, base_checkout
) -> None:
    ledger = await _started(store, package, base_checkout)
    admitted = [
        _binding(package.checks[0].check_id, package.criterion_keys[0], "S"),
        _binding(
            package.checks[1].check_id,
            package.criterion_keys[1],
            "S",
        ),
    ]
    await ledger.record_bindings(
        B,
        package_id=package.package_id,
        payload=BindingsPayload.model_validate({"phase": "final", "checks": admitted}),
    )
    expected = [expected_execution(check) for check in package.checks]
    receipt = verification_receipt(package, base_checkout)
    await ledger.record_candidate_verification(B, receipt.model_copy(update={"checks": expected}))
    # A re-run of a check that already met its role.
    rerun = receipt.model_copy(update={"checks": expected[:1]})
    with pytest.raises(BoundaryOrderError):
        await ledger.record_candidate_verification(B, rerun)


# --------------------------------------------------------------------------
# A later version the product could not have written (the M3 probe, ported).


async def _bound_run(store: EventStore, package: CheckPackage, checkout: Path, run: str):
    ledger = BoundaryLedger(store)
    await ledger.record_check_package_enabled(run, CONTRACT)
    v1 = boundary_version_id(run, 1)
    await ledger.record_package_frozen(v1, package)
    await ledger.record_admission(v1, admission_receipt(package, checkout))
    await ledger.record_actor_started(run, [v1])
    return ledger


async def _project(ledger: BoundaryLedger, run: str):
    return recovery_projection(run, await ledger.events(run), await ledger.run_versions(run))


async def test_bare_records_on_a_later_version_make_recovery_undecidable(
    store, package, base_checkout
) -> None:
    run = "exec_m3"
    ledger = await _bound_run(store, package, base_checkout, run)
    assert isinstance(await _project(ledger, run), RecoveryBound)
    for event_type in (CONSTRUCTION_FAILED, ACTOR_STARTED):
        await store.append(
            BaseEvent(
                type=event_type,
                aggregate_type=BOUNDARY_AGGREGATE_TYPE,
                aggregate_id=boundary_version_id(run, 2),
                data={},
            )
        )
    assert isinstance(await _project(ledger, run), RecoveryUndecidable)


async def test_a_version_sealed_after_the_worker_started_makes_recovery_undecidable(
    store, package, base_checkout
) -> None:
    run = "exec_m3_sealed"
    ledger = await _bound_run(store, package, base_checkout, run)
    await store.append(
        construction_failed_event(
            boundary_version_id(run, 2),
            seed_digest=package.seed_digest,
            input_digest="1" * 64,
            reason="parse",
        )
    )
    assert isinstance(await _project(ledger, run), RecoveryUndecidable)


async def test_a_gap_in_the_versions_makes_recovery_undecidable(
    store, package, base_checkout
) -> None:
    run = "exec_m3_gap"
    ledger = await _bound_run(store, package, base_checkout, run)
    await store.append(package_frozen_event(boundary_version_id(run, 3), package))
    assert isinstance(await _project(ledger, run), RecoveryUndecidable)


async def test_only_a_superseded_predecessor_may_stand_before_the_bound_version(
    store, seed, package, base_checkout
) -> None:
    run = "exec_chain"
    ledger = BoundaryLedger(store)
    await ledger.record_check_package_enabled(run, CONTRACT)
    v1, v2 = boundary_version_id(run, 1), boundary_version_id(run, 2)
    successor = seal_package(build_package(seed, repro_script=REPRO_SCRIPT + "# v2\n"))
    await ledger.record_package_frozen(v1, package)
    await ledger.record_admission(v1, admission_receipt(package, base_checkout))
    await ledger.record_package_frozen(v2, successor)
    await ledger.record_admission(v2, admission_receipt(successor, base_checkout))
    await ledger.record_actor_started(run, [v2])
    # v1 was never superseded: not a history the product writes.
    assert isinstance(await _project(ledger, run), RecoveryUndecidable)
    await ledger.record_superseded(v1, superseded_by=v2, reason="replacement_checks")
    assert isinstance(await _project(ledger, run), RecoveryBound)


async def test_a_supersession_naming_another_successor_package_is_undecidable(
    store, seed, package, base_checkout
) -> None:
    from ouroboros.boundary.events import superseded_event

    run = "exec_chain_forged"
    ledger = BoundaryLedger(store)
    await ledger.record_check_package_enabled(run, CONTRACT)
    v1, v2 = boundary_version_id(run, 1), boundary_version_id(run, 2)
    successor = seal_package(build_package(seed, repro_script=REPRO_SCRIPT + "# v2\n"))
    await ledger.record_package_frozen(v1, package)
    await ledger.record_package_frozen(v2, successor)
    await ledger.record_admission(v2, admission_receipt(successor, base_checkout))
    await store.append(
        superseded_event(
            v1,
            superseded_by=v2,
            package_id=package.package_id,
            successor_package_id=package.package_id,
            reason="replacement_checks",
        )
    )
    await ledger.record_actor_started(run, [v2])
    assert isinstance(await _project(ledger, run), RecoveryUndecidable)


# --------------------------------------------------------------------------
# The sweep: each record type once more, for what the product never writes.


async def test_an_a_prime_binding_without_its_valid_declaration_is_refused(
    store, base_checkout
) -> None:
    package = seal_package(held_out_package())
    (key,) = package.criterion_keys
    admission = admission_receipt(package, base_checkout).model_copy(
        update={"check_tiers": {"oracle_1": CheckTier.U}}
    )
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen(B, package)
    await ledger.record_admission(B, admission)
    await ledger.record_actor_started("actor-1", [B])
    binding = {"criterion_key": key, "symbol": "mathutils.clamp", "call_kind": "function"}
    declared = {
        "criterion_key": key,
        "valid": True,
        "indeterminate": False,
        "reason": "declared_binding_resolves",
        "binding": binding,
        "base_run": None,
    }
    a_prime = _binding(
        "oracle_1", key, "A_prime", binding_source="declared", binding=binding, reason="r"
    )
    for forged in (
        a_prime,  # no declaration recorded
        {**a_prime, "declared": {**declared, "valid": False}},  # the declaration was invalid
        {**a_prime, "tier": "A", "binding_source": "default", "declared": declared},
    ):
        payload = BindingsPayload.model_validate({"phase": "final", "checks": [forged]})
        with pytest.raises(BoundaryOrderError):
            await ledger.record_bindings(B, package_id=package.package_id, payload=payload)
    genuine = BindingsPayload.model_validate(
        {"phase": "final", "checks": [{**a_prime, "declared": declared}]}
    )
    await ledger.record_bindings(B, package_id=package.package_id, payload=genuine)


async def test_a_reference_check_excluding_cases_of_a_script_check_is_refused(
    store, package
) -> None:
    from ouroboros.boundary.events import ReferenceCheckPayload

    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen(B, package)
    payload = ReferenceCheckPayload.model_validate(
        {
            "schema_version": "x",
            "counts": {},
            "excluded_cases": [
                {"check_id": package.checks[0].check_id, "case_ids": ["c1"], "reason": "r"}
            ],
            "uncovered": [],
        }
    )
    with pytest.raises(BoundaryOrderError):
        await ledger.record_reference_checked(B, package_id=package.package_id, payload=payload)


def test_an_existing_acceptance_of_a_failed_outcome_is_not_journaled() -> None:
    from ouroboros.boundary.events import ReconciledRecord

    item = _criterion(
        0,
        "k",
        "unverified",
        governed_by="existing_verifier",
        existing_outcome="failed",
        existing_accepted=True,
    )
    data = _decision([item]).model_dump(mode="json")
    with pytest.raises(ValueError):
        ReconciledRecord.model_validate({**data, "package_id": None})
    ok = {**item, "existing_outcome": "succeeded"}
    ReconciledRecord.model_validate({**_decision([ok]).model_dump(mode="json"), "package_id": None})
