"""EventStore ordering: package digest persisted before any actor starts."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from ouroboros.boundary.events import (
    ACTOR_STARTED,
    ADMISSION_COMPLETED,
    CANDIDATE_VERIFIED,
    CHECK_PACKAGE_ENABLED,
    PACKAGE_FROZEN,
    BindingsPayload,
    ReconciliationPayload,
    ReferenceCheckPayload,
    RunContract,
    acceptance_reconciled_event,
    actor_started_event,
    admission_completed_event,
    binding_recorded_event,
    boundary_version_id,
    package_frozen_event,
    superseded_event,
)
from ouroboros.boundary.ledger import (
    BoundaryLeakError,
    BoundaryLedger,
    BoundaryOrderError,
    verify_boundary_order,
)
from ouroboros.boundary.package import seal_package
from ouroboros.boundary.receipts import AdmissionResult, CandidateVerification
from ouroboros.persistence.event_store import EventStore

from .conftest import (
    INPUT_DIGEST,
    REPRO_SCRIPT,
    SIGNATURE,
    admission_receipt,
    build_package,
    verification_receipt,
)

CONTRACT = RunContract(check_timeout_seconds=120)


@pytest.fixture
async def store():
    event_store = EventStore("sqlite+aiosqlite:///:memory:")
    await event_store.initialize()
    yield event_store
    await event_store.close()


@pytest.fixture
def admission(base_checkout, package):
    return admission_receipt(package, base_checkout)


async def test_package_hash_event_precedes_actor_start(
    store, tmp_path: Path, seed, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    frozen = await ledger.record_package_frozen("task-1/V1", package, seed=seed)
    await ledger.record_admission("task-1/V1", admission)
    workspace = tmp_path / "worker"
    workspace.mkdir()
    (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    started = await ledger.record_actor_started(
        "actor-1", ["task-1/V1"], workspace=workspace, runtime="codex", packages=[package]
    )

    events = await ledger.events("task-1/V1")
    assert [e.type for e in events] == [PACKAGE_FROZEN, ADMISSION_COMPLETED, ACTOR_STARTED]
    assert frozen.data["package_id"] == package.package_id
    assert frozen.data["seed_digest"] == package.seed_digest
    assert events[0].timestamp < events[-1].timestamp
    assert started[0].data["package_id"] == package.package_id
    assert verify_boundary_order(events) == ()
    # The journal carries ids and digests, never check code or argv.
    journal = repr([e.data for e in events])
    assert SIGNATURE not in journal
    assert REPRO_SCRIPT not in journal


async def test_actor_cannot_start_before_seal_or_admission(store, package) -> None:
    ledger = BoundaryLedger(store)
    with pytest.raises(BoundaryOrderError, match="sealed"):
        await ledger.record_actor_started("actor-1", ["task-1/V1"])
    await ledger.record_package_frozen("task-1/V1", package)
    with pytest.raises(BoundaryOrderError, match="admission"):
        await ledger.record_actor_started("actor-1", ["task-1/V1"])
    assert all(e.type != ACTOR_STARTED for e in await ledger.events("task-1/V1"))


async def test_actor_waits_for_every_bound_boundary(store, package, admission) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V0", package)
    await ledger.record_admission("task-1/V0", admission)
    with pytest.raises(BoundaryOrderError):
        await ledger.record_actor_started("actor-1", ["task-1/V0", "task-1/V1"])
    await ledger.record_construction_failed(
        "task-1/V1", seed_digest=package.seed_digest, input_digest=INPUT_DIGEST, reason="parse"
    )
    started = await ledger.record_actor_started("actor-1", ["task-1/V0", "task-1/V1"])
    assert [e.data["package_id"] for e in started] == [package.package_id, None]


async def test_package_cannot_be_regenerated_after_seal(store, seed, package) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    regenerated = seal_package(build_package(seed, repro_script=REPRO_SCRIPT + "# retry\n"))
    with pytest.raises(BoundaryOrderError, match="regenerated"):
        await ledger.record_package_frozen("task-1/V1", regenerated)


async def test_admission_must_cite_frozen_digest_and_is_single(
    store, seed, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    other = seal_package(build_package(seed, repro_script=REPRO_SCRIPT + "# other\n"))
    await ledger.record_package_frozen("task-1/V1", other)
    with pytest.raises(BoundaryOrderError, match="different package"):
        await ledger.record_admission("task-1/V1", admission)

    await ledger.record_package_frozen("task-2/V1", package)
    await ledger.record_admission("task-2/V1", admission)
    with pytest.raises(BoundaryOrderError, match="already recorded"):
        await ledger.record_admission("task-2/V1", admission)


async def test_workspace_with_generated_check_code_is_refused(
    store, tmp_path: Path, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    await ledger.record_admission("task-1/V1", admission)
    workspace = tmp_path / "worker"
    workspace.mkdir()
    (workspace / "hidden_copy.py").write_text(REPRO_SCRIPT)
    with pytest.raises(BoundaryLeakError):
        await ledger.record_actor_started(
            "actor-1", ["task-1/V1"], workspace=workspace, packages=[package]
        )


async def test_candidate_verification_cites_the_frozen_package(
    store, tmp_path: Path, base_checkout, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    await ledger.record_admission("task-1/V1", admission)
    await ledger.record_actor_started("actor-1", ["task-1/V1"])
    verification = verification_receipt(package, base_checkout)
    event = await ledger.record_candidate_verification("task-1/V1", verification)

    assert event.type == CANDIDATE_VERIFIED
    assert event.data["verdict"] == "fail"
    assert verify_boundary_order(await ledger.events("task-1/V1")) == ()


def test_verify_boundary_order_flags_actor_before_seal(package) -> None:
    from ouroboros.boundary.events import actor_started_event, package_frozen_event

    actor = actor_started_event("b", actor_id="a", package_id=None, runtime=None)
    frozen = package_frozen_event("b", package)
    violations = verify_boundary_order([actor, frozen])
    assert any(item.startswith("actor started before the seal") for item in violations)


async def test_the_enabled_record_is_written_once_per_run_and_needs_an_execution_id(
    store: EventStore,
) -> None:
    ledger = BoundaryLedger(store)
    assert not await ledger.check_package_enabled("exec_enabled")
    event = await ledger.record_check_package_enabled("exec_enabled", CONTRACT)
    assert event.type == CHECK_PACKAGE_ENABLED and event.aggregate_id == "exec_enabled"
    assert await ledger.check_package_enabled("exec_enabled")
    with pytest.raises(BoundaryOrderError):
        await ledger.record_check_package_enabled("exec_enabled", CONTRACT)
    with pytest.raises(BoundaryOrderError):
        await ledger.record_check_package_enabled("", CONTRACT)


def test_a_payload_cannot_name_another_package_or_carry_undefined_fields(package) -> None:
    # The bot's probe: a caller-supplied payload overwrote the cited package id.
    for model in (BindingsPayload, ReconciliationPayload, ReferenceCheckPayload):
        with pytest.raises(ValidationError):
            model.model_validate({"package_id": "forged"})
    with pytest.raises(ValidationError):
        BindingsPayload.model_validate({"phase": "final", "checks": [], "package_id": "forged"})
    with pytest.raises(ValidationError):
        BindingsPayload.model_validate({"phase": "final", "checks": [], "held_out_value": 7})
    with pytest.raises(TypeError):
        binding_recorded_event(
            "b",
            package_id=package.package_id,
            payload={"phase": "final", "checks": [], "package_id": "forged"},  # type: ignore[arg-type]
        )
    event = binding_recorded_event(
        "b", package_id=package.package_id, payload=BindingsPayload(phase="final", checks=())
    )
    assert event.data["package_id"] == package.package_id


async def test_an_admission_receipt_for_another_seed_is_refused(store, package, admission) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    foreign = admission.model_copy(update={"seed_digest": "f" * 64})
    with pytest.raises(BoundaryOrderError, match="different Seed"):
        await ledger.record_admission("task-1/V1", foreign)
    await ledger.record_admission("task-1/V1", admission)


async def test_a_superseded_version_accepts_no_actor_start(
    store, seed, base_checkout, package, admission
) -> None:
    # The bot's probe: V1 frozen and admitted, superseded by V2, then an
    # actor start on V1. The write is refused, and replay flags a journal
    # that holds one anyway.
    ledger = BoundaryLedger(store)
    await ledger.record_check_package_enabled("exec_s", CONTRACT)
    v1, v2 = boundary_version_id("exec_s", 1), boundary_version_id("exec_s", 2)
    await ledger.record_package_frozen(v1, package)
    await ledger.record_admission(v1, admission)
    successor = seal_package(build_package(seed, repro_script=REPRO_SCRIPT + "# v2\n"))
    await ledger.record_package_frozen(v2, successor)
    await ledger.record_admission(v2, admission_receipt(successor, base_checkout))
    await ledger.record_superseded(v1, superseded_by=v2, reason="replacement_checks")
    with pytest.raises(BoundaryOrderError, match="superseded"):
        await ledger.record_actor_started("exec_s", [v1])
    assert verify_boundary_order(await ledger.events(v1)) == ()
    await store.append(
        actor_started_event(v1, actor_id="exec_s", package_id=package.package_id, runtime=None)
    )
    violations = verify_boundary_order(await ledger.events(v1))
    assert f"{ACTOR_STARTED} recorded on a superseded boundary version" in violations
    await ledger.record_actor_started("exec_s", [v2])


def test_replay_flags_an_admission_before_the_seal(package, admission) -> None:
    violations = verify_boundary_order(
        [admission_completed_event("b", admission), package_frozen_event("b", package)]
    )
    assert any(item.startswith("admission recorded before the seal") for item in violations)


async def test_the_enabled_record_carries_the_run_contract(store: EventStore) -> None:
    ledger = BoundaryLedger(store)
    assert await ledger.run_contract("exec_contract") is None  # never on
    contract = RunContract(check_timeout_seconds=37)
    event = await ledger.record_check_package_enabled("exec_contract", contract)
    assert event.data["contract"] == {"check_timeout_seconds": 37}
    assert await ledger.run_contract("exec_contract") == contract
    with pytest.raises(TypeError):
        await ledger.record_check_package_enabled("exec_other", {"check_timeout_seconds": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RunContract(check_timeout_seconds=0)
    with pytest.raises(ValueError):
        RunContract.model_validate({"check_timeout_seconds": 5, "extra": 1})


async def test_a_malformed_run_contract_is_refused_on_read(store: EventStore) -> None:
    from ouroboros.events.base import BaseEvent

    await store.append(
        BaseEvent(
            type=CHECK_PACKAGE_ENABLED,
            aggregate_type="boundary",
            aggregate_id="exec_bad",
            data={"execution_id": "exec_bad"},  # no contract: the run's settings are unknown
        )
    )
    with pytest.raises(BoundaryOrderError):
        await BoundaryLedger(store).run_contract("exec_bad")


async def test_the_enabled_record_comes_before_every_version_of_the_run(store, package) -> None:
    ledger = BoundaryLedger(store)
    v1 = boundary_version_id("exec_e", 1)
    with pytest.raises(BoundaryOrderError, match="enabled record"):
        await ledger.record_package_frozen(v1, package)
    # A version that reached the journal first (written around the ledger)
    # keeps the enabled record out, and replay against the run flags it.
    await store.append(package_frozen_event(v1, package))
    with pytest.raises(BoundaryOrderError, match="precede"):
        await ledger.record_check_package_enabled("exec_e", CONTRACT)
    assert "the run has no enabled record" in verify_boundary_order(
        await ledger.events(v1), run_events=await ledger.events("exec_e")
    )
    # A standalone boundary (not a version of a run) needs no enabled record.
    await ledger.record_package_frozen("task-9/V1", package)


async def test_the_enabled_record_sees_every_version_not_only_the_first(store, package) -> None:
    # #2458 round 3: only v1 was inspected, so a v2 written around the ledger
    # let the enabled record in, and replay then flagged a journal the write
    # path had accepted.
    ledger = BoundaryLedger(store)
    v2 = boundary_version_id("exec_gap", 2)
    await store.append(package_frozen_event(v2, package))
    assert list(await ledger.run_versions("exec_gap")) == [2]
    with pytest.raises(BoundaryOrderError, match="precede"):
        await ledger.record_check_package_enabled("exec_gap", CONTRACT)
    # Other runs' versions are not this run's.
    await store.append(package_frozen_event(boundary_version_id("exec_gap_other", 1), package))
    assert list(await ledger.run_versions("exec_gap")) == [2]
    await ledger.record_check_package_enabled("exec_fresh", CONTRACT)


async def test_a_version_is_superseded_only_by_a_later_version_of_its_own_run(
    store, seed, base_checkout, package, admission
) -> None:
    # The round-2 probe: run_b's version was accepted as the successor of
    # run_a's, and a version could point back to an earlier one.
    ledger = BoundaryLedger(store)
    a1, a2 = boundary_version_id("run_a", 1), boundary_version_id("run_a", 2)
    b1 = boundary_version_id("run_b", 1)
    for run in ("run_a", "run_b"):
        await ledger.record_check_package_enabled(run, CONTRACT)
    for version in (a1, a2, b1):
        await ledger.record_construction_failed(
            version, seed_digest=package.seed_digest, input_digest=INPUT_DIGEST, reason="x"
        )
    with pytest.raises(BoundaryOrderError, match="its own run"):
        await ledger.record_superseded(a1, superseded_by=b1, reason="x")
    with pytest.raises(BoundaryOrderError, match="later version"):
        await ledger.record_superseded(a2, superseded_by=a1, reason="x")
    with pytest.raises(BoundaryOrderError, match="version of a run"):
        await ledger.record_superseded("task-1/V1", superseded_by=a2, reason="x")
    await ledger.record_superseded(a1, superseded_by=a2, reason="x")
    # Replay applies the same rule to a journal written around the ledger.
    forged = superseded_event(
        a2, superseded_by=b1, package_id=None, successor_package_id=None, reason="x"
    )
    await store.append(forged)
    assert "a boundary version is superseded only within its own run" in verify_boundary_order(
        await ledger.events(a2)
    )


async def test_a_version_of_a_run_binds_only_that_runs_worker(store, package) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_check_package_enabled("run_a", CONTRACT)
    a1 = boundary_version_id("run_a", 1)
    await ledger.record_construction_failed(
        a1, seed_digest=package.seed_digest, input_digest=INPUT_DIGEST, reason="x"
    )
    with pytest.raises(BoundaryOrderError, match="that run's worker"):
        await ledger.record_actor_started("run_b", [a1])
    await ledger.record_actor_started("run_a", [a1])


async def test_a_candidate_verification_for_another_seed_is_refused(
    store, base_checkout, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    await ledger.record_admission("task-1/V1", admission)
    await ledger.record_actor_started("actor-1", ["task-1/V1"])
    verification = verification_receipt(package, base_checkout)
    foreign = verification.model_copy(update={"seed_digest": "f" * 64})
    with pytest.raises(BoundaryOrderError, match="different Seed"):
        await ledger.record_candidate_verification("task-1/V1", foreign)
    event = await ledger.record_candidate_verification("task-1/V1", verification)
    assert (event.data["seed_digest"], event.data["package_id"]) == (
        package.seed_digest,
        package.package_id,
    )


def test_receipts_are_closed_and_name_their_seed(base_checkout, package) -> None:
    # The round-2 probe: a supplied seed_digest was silently dropped.
    verification = verification_receipt(package, base_checkout)
    data = verification.model_dump()
    data["package_sha256"] = package.sha256
    with pytest.raises(ValidationError):
        CandidateVerification.model_validate({**data, "unknown": 1})
    with pytest.raises(ValidationError):
        CandidateVerification.model_validate({k: v for k, v in data.items() if k != "seed_digest"})
    with pytest.raises(ValidationError):
        CandidateVerification.model_validate({**data, "seed_digest": "not a digest"})
    admission = admission_receipt(package, base_checkout).model_dump()
    admission["package_sha256"] = package.sha256
    with pytest.raises(ValidationError):
        AdmissionResult.model_validate({**admission, "unknown": 1})
    with pytest.raises(ValidationError):
        AdmissionResult.model_validate({k: v for k, v in admission.items() if k != "package_id"})
    assert verification.event_summary()["seed_digest"] == package.seed_digest


async def test_an_undecided_resume_is_recorded_only_for_a_run_that_was_on(store) -> None:
    from ouroboros.boundary.events import ResumedPayload

    ledger = BoundaryLedger(store)
    payload = ResumedPayload(
        schema_version="s",
        run_accepted=False,
        existing_run_accepted=False,
        artifact_verdict="indeterminate",
        verified_pass_count=0,
        unverified_count=0,
        criterion_count=0,
        tier_summary={},
        criteria=(),
        source="none",
    )
    with pytest.raises(BoundaryOrderError, match="was on"):
        await ledger.record_resumed_undecided("run_x", payload=payload)
    await ledger.record_check_package_enabled("run_x", CONTRACT)
    await ledger.record_resumed_undecided("run_x", payload=payload)


def _bindings(tier: str) -> BindingsPayload:
    return BindingsPayload.model_validate(
        {
            "phase": "final",
            "checks": [
                {
                    "criterion_key": "k",
                    "check_id": "c1",
                    "tier": tier,
                    "binding_source": None,
                    "binding": None,
                    "status_hint": None,
                    "reason": "r",
                    "declared": None,
                }
            ],
        }
    )


def _decision(status: str, *, undecided: str | None = None) -> ReconciliationPayload:
    criterion = {
        "root_ac_index": 0,
        "criterion_key": "k",
        "package_status": status,
        "tier": "A",
        "reason": "r",
        "failed_heldout_only": False,
        "binding": None,
        "existing_outcome": "succeeded",
        "existing_failure_class": None,
        "existing_accepted": True,
        "accepted": status in ("pass", "unverified", "uncovered"),
        "governed_by": "check_package",
    }
    data = {
        "schema_version": "ouroboros.acceptance_reconciliation.v3",
        "run_accepted": status in ("pass", "unverified", "uncovered"),
        "existing_run_accepted": True,
        "artifact_verdict": status,
        "verified_pass_count": int(status == "pass"),
        "unverified_count": 0,
        "criterion_count": 1,
        "tier_summary": {},
        "criteria": [criterion],
    }
    if undecided is not None:
        data["undecided_reason"] = undecided
    return ReconciliationPayload.model_validate(data)


async def _started(store, package, admission) -> BoundaryLedger:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    await ledger.record_admission("task-1/V1", admission)
    await ledger.record_actor_started("actor-1", ["task-1/V1"])
    return ledger


async def test_a_verified_decision_needs_the_candidate_verification_of_runnable_bindings(
    store, package, admission, base_checkout
) -> None:
    # The #2465 probe: final bindings with a runnable (tier A) check, then the
    # verification raised; a reconciliation claiming a verified status must
    # not follow the bindings alone. Refused on write, flagged on replay.
    ledger = await _started(store, package, admission)
    await ledger.record_bindings("task-1/V1", package_id=package.package_id, payload=_bindings("A"))
    for status in ("pass", "fail", "unverified"):
        with pytest.raises(BoundaryOrderError, match="acceptance must cite a verification"):
            await ledger.record_acceptance_reconciled(
                "task-1/V1", package_id=package.package_id, reconciliation=_decision(status)
            )
    forged = acceptance_reconciled_event(
        "task-1/V1", package_id=package.package_id, reconciliation=_decision("pass")
    )
    await store.append(forged)
    assert any(
        "acceptance must cite a verification" in item
        for item in verify_boundary_order(await ledger.events("task-1/V1"))
    )


async def test_an_undecided_decision_after_a_failed_verification_is_recorded(
    store, package, admission
) -> None:
    # The authority's fail-closed shape: covered criteria indeterminate, stated
    # as undecided. It needs no candidate verification, and cannot smuggle a
    # verified status.
    ledger = await _started(store, package, admission)
    await ledger.record_bindings("task-1/V1", package_id=package.package_id, payload=_bindings("A"))
    with pytest.raises(ValueError, match="undecided decision carries only"):
        _decision("pass", undecided="authority_error:OSError")
    await ledger.record_acceptance_reconciled(
        "task-1/V1",
        package_id=package.package_id,
        reconciliation=_decision("indeterminate", undecided="authority_error:OSError"),
    )
    assert verify_boundary_order(await ledger.events("task-1/V1")) == ()


async def test_an_undecided_decision_that_accepts_an_indeterminate_criterion_is_refused(
    store, package, admission
) -> None:
    # #2465 round 2: the undecided exception must not admit an accepted
    # indeterminate criterion. The payload model refuses it, and a payload
    # built without validation is refused on write and flagged on replay.
    ledger = await _started(store, package, admission)
    await ledger.record_bindings("task-1/V1", package_id=package.package_id, payload=_bindings("A"))
    valid = _decision("indeterminate", undecided="authority_error:OSError")
    accepted = valid.criteria[0].model_copy(update={"accepted": True})
    raw = valid.model_dump(mode="json")
    raw["criteria"][0]["accepted"] = True
    raw["run_accepted"] = True
    with pytest.raises(ValueError, match="acceptance disagrees"):
        ReconciliationPayload.model_validate(raw)
    smuggled = ReconciliationPayload.model_construct(
        **{**dict(valid), "criteria": (accepted,), "run_accepted": True}
    )
    with pytest.raises(BoundaryOrderError, match="disagrees with the statuses"):
        await ledger.record_acceptance_reconciled(
            "task-1/V1", package_id=package.package_id, reconciliation=smuggled
        )
    await store.append(
        acceptance_reconciled_event(
            "task-1/V1", package_id=package.package_id, reconciliation=smuggled
        )
    )
    assert any(
        "disagrees with the statuses" in item
        for item in verify_boundary_order(await ledger.events("task-1/V1"))
    )


async def test_bindings_with_no_runnable_check_may_be_followed_by_an_unrun_decision(
    store, package, admission
) -> None:
    ledger = await _started(store, package, admission)
    await ledger.record_bindings("task-1/V1", package_id=package.package_id, payload=_bindings("U"))
    with pytest.raises(BoundaryOrderError, match="claims a verified status"):
        await ledger.record_acceptance_reconciled(
            "task-1/V1", package_id=package.package_id, reconciliation=_decision("pass")
        )
    await ledger.record_acceptance_reconciled(
        "task-1/V1", package_id=package.package_id, reconciliation=_decision("unverified")
    )


async def test_a_verified_decision_follows_the_candidate_verification(
    store, package, admission, base_checkout
) -> None:
    ledger = await _started(store, package, admission)
    await ledger.record_bindings("task-1/V1", package_id=package.package_id, payload=_bindings("A"))
    await ledger.record_candidate_verification(
        "task-1/V1", verification_receipt(package, base_checkout)
    )
    await ledger.record_acceptance_reconciled(
        "task-1/V1", package_id=package.package_id, reconciliation=_decision("fail")
    )
    assert verify_boundary_order(await ledger.events("task-1/V1")) == ()
