"""Event factories for the check-package boundary.

Event Types (aggregate_type ``boundary``, aggregate_id = boundary id):
    boundary.check_package.frozen - package digest and event-safe manifest
    boundary.check_package.construction_failed - no package; verifier indeterminate
    boundary.check_package.admission_completed - whole-package base admission receipt
    boundary.actor.started - a worker bound to this boundary started
    boundary.candidate.verified - frozen package run on a candidate
    boundary.selection.decided - incumbent kept or replaced, with reason and digests
    boundary.check_package.superseded - product regeneration: this boundary
        version was replaced by a later version before any worker bound to it
    boundary.binding.recorded - after the worker stopped: the tier and binding
        of every check (late bindings are data; no code, no cases)
    boundary.oracle.case_revealed - one held-out case was shown to the worker
        in a repair message and is retired from held-out statistics (case id
        only; the inputs stay out of the journal)
    boundary.acceptance.legacy_fallback - the package could not decide the
        run (for example the acceptance authority raised); the legacy
        verifier's verdicts were restored and decided it (reason only)
    boundary.acceptance.resumed - a run resumed after its controller died:
        the package decision recomputed on the current workspace (visible
        cases only when the held-out cases were lost with the controller)

Payloads never carry generated file contents, check argv, or check output, so
the shared journal does not expose check code or counterexamples. The package
and complete receipts are stored separately under their digests
(``write_check_package``, ``write_receipt``).
"""

from __future__ import annotations

from typing import Any

from ouroboros.boundary.admission import AdmissionResult, CandidateVerification
from ouroboros.boundary.package import CheckPackage, persisted_citation
from ouroboros.boundary.selection import SelectionDecision
from ouroboros.events.base import BaseEvent

BOUNDARY_AGGREGATE_TYPE = "boundary"

PACKAGE_FROZEN = "boundary.check_package.frozen"
CONSTRUCTION_FAILED = "boundary.check_package.construction_failed"
ADMISSION_COMPLETED = "boundary.check_package.admission_completed"
ACTOR_STARTED = "boundary.actor.started"
CANDIDATE_VERIFIED = "boundary.candidate.verified"
SELECTION_DECIDED = "boundary.selection.decided"
SUPERSEDED = "boundary.check_package.superseded"
ACCEPTANCE_RECONCILED = "boundary.acceptance.reconciled"
BINDING_RECORDED = "boundary.binding.recorded"
CASE_REVEALED = "boundary.oracle.case_revealed"
LEGACY_FALLBACK = "boundary.acceptance.legacy_fallback"
ACCEPTANCE_RESUMED = "boundary.acceptance.resumed"


def _event(boundary_id: str, event_type: str, data: dict[str, Any]) -> BaseEvent:
    return BaseEvent(
        type=event_type,
        aggregate_type=BOUNDARY_AGGREGATE_TYPE,
        aggregate_id=boundary_id,
        data=data,
    )


def cite(reference: str | None, *, committed: bool, prefix: str = "") -> dict[str, str | None]:
    """``{"<prefix>package_commitment": ref}`` on a committed boundary, else ``..._sha256``."""
    return {f"{prefix}package_{'commitment' if committed else 'sha256'}": reference}


def package_frozen_event(
    boundary_id: str, package: CheckPackage, *, record_sha256: str | None = None
) -> BaseEvent:
    """The package reference (I2) plus the event-safe manifest summary.

    For a committed package the reference is the commitment
    SHA-256(salt || canonical bytes); the unkeyed digest is not recorded.
    ``record_sha256`` is the SHA-256 of the stored package record's bytes
    (visible cases in full, held-out cases as keyed HMACs only), so a resume
    in another process can tell whether the record was edited; omitted when
    not given, so other callers keep their bytes.
    """
    summary = package.manifest_summary()
    extra = {"commitment_scheme": summary["commitment_scheme"]} if package.commitment else {}
    if record_sha256 is not None:
        extra["record_sha256"] = record_sha256
    return _event(
        boundary_id,
        PACKAGE_FROZEN,
        {**package.citation(), **extra, "manifest": summary},
    )


def construction_failed_event(
    boundary_id: str, *, seed_digest: str, input_digest: str, reason: str
) -> BaseEvent:
    """Record that no package exists for this boundary; its verifier is indeterminate."""
    return _event(
        boundary_id,
        CONSTRUCTION_FAILED,
        {"seed_digest": seed_digest, "input_digest": input_digest, "reason": reason},
    )


def admission_completed_event(boundary_id: str, result: AdmissionResult) -> BaseEvent:
    """Whole-package admission receipt."""
    return _event(boundary_id, ADMISSION_COMPLETED, result.event_summary())


def actor_started_event(
    boundary_id: str,
    *,
    actor_id: str,
    package_sha256: str | None,
    runtime: str | None,
    committed: bool = False,
    assignment: str | None = None,
) -> BaseEvent:
    """A worker bound to this boundary started after the boundary was sealed.

    ``assignment`` is the product arm's source (``rollout.AssignmentSource``),
    recorded so that a resumed run reports the original assignment; omitted
    when not given, so other callers keep their earlier bytes.
    """
    return _event(
        boundary_id,
        ACTOR_STARTED,
        {
            "actor_id": actor_id,
            **cite(package_sha256, committed=committed),
            "runtime": runtime,
            **({"check_package_assignment": assignment} if assignment is not None else {}),
        },
    )


def candidate_verified_event(boundary_id: str, verification: CandidateVerification) -> BaseEvent:
    """Frozen package run on one candidate."""
    return _event(boundary_id, CANDIDATE_VERIFIED, verification.event_summary())


def superseded_event(
    boundary_id: str,
    *,
    superseded_by: str,
    package_sha256: str | None,
    successor_package_sha256: str | None,
    reason: str,
    committed: bool = False,
    successor_committed: bool = False,
) -> BaseEvent:
    """Mark a sealed boundary version as replaced by a later version."""
    return _event(
        boundary_id,
        SUPERSEDED,
        {
            "superseded_by": superseded_by,
            **cite(package_sha256, committed=committed),
            **cite(successor_package_sha256, committed=successor_committed, prefix="successor_"),
            "reason": reason,
        },
    )


def selection_decided_event(boundary_id: str, decision: SelectionDecision) -> BaseEvent:
    """Selection reason plus incumbent, candidate, selected, and package digests."""
    return _event(
        boundary_id, SELECTION_DECIDED, persisted_citation(decision.model_dump(mode="json"))
    )


def acceptance_reconciled_event(
    boundary_id: str,
    *,
    package_sha256: str | None,
    reconciliation: dict[str, Any],
    committed: bool = False,
) -> BaseEvent:
    """Per-criterion acceptance: package verdict, existing verdict (advisory), decision."""
    return _event(
        boundary_id,
        ACCEPTANCE_RECONCILED,
        {**cite(package_sha256, committed=committed), **reconciliation},
    )


def binding_recorded_event(
    boundary_id: str, *, package_sha256: str, payload: dict[str, Any], committed: bool = False
) -> BaseEvent:
    """Tiers and bindings of every check, recorded after the worker stopped."""
    return _event(
        boundary_id, BINDING_RECORDED, {**cite(package_sha256, committed=committed), **payload}
    )


def case_revealed_event(
    boundary_id: str,
    *,
    package_sha256: str,
    check_id: str,
    criterion_key: str,
    case_id: str,
    root_ac_index: int | None = None,
    retry_attempt: int | None = None,
    committed: bool = False,
) -> BaseEvent:
    """A held-out case revealed in a repair message (case id, never its values)."""
    return _event(
        boundary_id,
        CASE_REVEALED,
        {
            **cite(package_sha256, committed=committed),
            "check_id": check_id,
            "criterion_key": criterion_key,
            "case_id": case_id,
            "root_ac_index": root_ac_index,
            "retry_attempt": retry_attempt,
        },
    )


def legacy_fallback_event(boundary_id: str, *, reason: str) -> BaseEvent:
    """The legacy verifier decided this run because the package could not."""
    return _event(boundary_id, LEGACY_FALLBACK, {"reason": reason})


def acceptance_resumed_event(
    boundary_id: str, *, package_sha256: str, committed: bool, payload: dict[str, Any]
) -> BaseEvent:
    """The package decision a resumed run recomputed (statuses and reasons, no case values)."""
    return _event(
        boundary_id, ACCEPTANCE_RESUMED, {**cite(package_sha256, committed=committed), **payload}
    )
