"""Event factories for the check-package boundary.

Event Types (aggregate_type ``boundary``, aggregate_id = boundary id):
    boundary.check_package.enabled - aggregate_id is the run's execution id,
        not a boundary version: the check package was on for this run; the
        first boundary event of the run, written before construction
    boundary.check_package.frozen - package digest and event-safe manifest
    boundary.check_package.construction_failed - no package; verifier indeterminate
    boundary.check_package.admission_completed - base admission receipt
    boundary.actor.started - a worker bound to this boundary started
    boundary.candidate.verified - frozen package run on a candidate
    boundary.check_package.superseded - product regeneration: this boundary
        version was replaced by a later version before any worker bound to it
    boundary.binding.recorded - after the worker stopped: the tier and binding
        of every check (late bindings are data; no code, no cases)
    boundary.acceptance.resumed - a run resumed after its controller died:
        the package decision recovered from the journal (recomputed only
        while the same process holds the admitted package, else undecided)

Payloads never carry generated file contents, check argv, or check output, so
the shared journal does not expose check code or counterexamples. Every event
names a package by its opaque id (``package.seal_package``), never by an
unkeyed digest. The package record (held-out cases as ids only) and complete
receipts are stored separately (``package.write_package_record``,
``receipts.write_receipt``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ouroboros.boundary.binding import Binding, BindingSource, CheckTier
from ouroboros.boundary.package import CheckPackage, package_record_bytes, sha256_bytes
from ouroboros.boundary.receipts import (
    AdmissionJournal,
    AdmissionResult,
    CandidateVerification,
    VerificationJournal,
)
from ouroboros.events.base import BaseEvent

BOUNDARY_AGGREGATE_TYPE = "boundary"

PACKAGE_FROZEN = "boundary.check_package.frozen"
CONSTRUCTION_FAILED = "boundary.check_package.construction_failed"
ADMISSION_COMPLETED = "boundary.check_package.admission_completed"
ACTOR_STARTED = "boundary.actor.started"
CANDIDATE_VERIFIED = "boundary.candidate.verified"
SUPERSEDED = "boundary.check_package.superseded"
ACCEPTANCE_RECONCILED = "boundary.acceptance.reconciled"
BINDING_RECORDED = "boundary.binding.recorded"
ACCEPTANCE_RESUMED = "boundary.acceptance.resumed"
REFERENCE_CHECKED = "boundary.oracle.reference_checked"
CHECK_PACKAGE_ENABLED = "boundary.check_package.enabled"

_VERSION_INFIX = "/check_package/v"


def boundary_version_id(execution_id: str, version: int) -> str:
    """The id of a run's ``version``-th boundary version (``<execution_id>/check_package/v<n>``)."""
    if not execution_id or version < 1:
        raise ValueError("a boundary version needs an execution id and a version >= 1")
    return f"{execution_id}{_VERSION_INFIX}{version}"


def parse_boundary_version(boundary_id: str) -> tuple[str, int] | None:
    """``(execution_id, version)`` of an id ``boundary_version_id`` made, else ``None``.

    The inverse of ``boundary_version_id``: a boundary id that it did not
    produce is a standalone boundary (not a version of a run).
    """
    execution_id, infix, tail = boundary_id.rpartition(_VERSION_INFIX)
    if not infix or not execution_id or not tail.isdecimal():
        return None
    version = int(tail)
    if version < 1 or boundary_version_id(execution_id, version) != boundary_id:
        return None
    return execution_id, version


# --------------------------------------------------------------------------
# Typed payloads. Every event a caller supplies content for takes one of these
# models (closed fields, ``extra="forbid"``); the factories add the identity
# fields (the package id) after validation, so a payload can neither name
# another package nor carry fields the journal does not define.


class _Payload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def journal_data(self) -> dict[str, Any]:
        """JSON-safe dump with only the fields that were given."""
        return self.model_dump(mode="json", exclude_unset=True)


class RunContract(_Payload):
    """The settings that decide anything after the worker starts, fixed for the run.

    Written on the run's ``boundary.check_package.enabled`` record before
    construction; a resumed run reads them back instead of the live config,
    so a resume decides with the settings the run started with.
    """

    check_timeout_seconds: int = Field(gt=0)


class BaseRunRecord(_Payload):
    """The base run of one declared binding (status and reason, no output)."""

    status: str
    reason: str
    signature_seen: bool
    return_code: int | None


class DeclaredBindingRecord(_Payload):
    """Grammar check and base run of one worker-declared binding."""

    criterion_key: str
    valid: bool
    indeterminate: bool
    reason: str
    binding: Binding | None
    base_run: BaseRunRecord | None


class CheckBindingRecord(_Payload):
    """The tier and binding one check ran (or would run) through."""

    criterion_key: str
    check_id: str
    tier: CheckTier
    binding_source: BindingSource | None
    binding: Binding | None
    status_hint: str | None
    reason: str
    declared: DeclaredBindingRecord | None


class BindingsPayload(_Payload):
    """``boundary.binding.recorded``: every check's tier and binding (data, no code)."""

    schema_version: Literal["ouroboros.binding_record.v1"] = "ouroboros.binding_record.v1"
    phase: Literal["final", "repair"]
    checks: tuple[CheckBindingRecord, ...]
    root_ac_index: int | None = None
    retry_attempt: int | None = None
    status: str | None = None


class CriterionDecisionRecord(_Payload):
    """One criterion's final acceptance and the signals behind it.

    ``tier`` is for display only (the weakest tier over the passing checks)
    and has no authority: an A' oracle plus an advisory script shows ``S``.
    What decided a pass is ``declared_binding_pass``.
    """

    root_ac_index: int
    criterion_key: str
    package_status: str
    tier: CheckTier
    reason: str
    failed_heldout_only: bool
    binding: Binding | None
    existing_outcome: str | None
    existing_failure_class: str | None
    existing_accepted: bool
    accepted: bool
    governed_by: str
    declared_binding_pass: bool
    """The criterion's package ``pass`` depends on a worker-declared binding (tier A').

    Such a pass only corroborates: it is decided by the existing verifier
    when that verifier rejected the attempt, and it never accepts a criterion
    the existing verifier rejected. ``False`` for every status but ``pass``.
    """


_DECISION_STATUSES = frozenset({"pass", "fail", "indeterminate", "unverified", "uncovered"})
_GOVERNORS = frozenset({"check_package", "execution", "existing_verifier"})
_PACKAGE_ACCEPTS = frozenset({"pass", "unverified", "uncovered"})
_EXISTING_DECIDES = frozenset({"unverified", "uncovered"})
_UNDECIDED_STATUSES = frozenset({"indeterminate", "uncovered"})


class ReconciliationPayload(_Payload):
    """``boundary.acceptance.reconciled``: per-criterion decisions and the run's."""

    schema_version: str
    run_accepted: bool
    existing_run_accepted: bool
    artifact_verdict: str
    verified_pass_count: int
    unverified_count: int
    criterion_count: int
    tier_summary: dict[str, int]
    criteria: tuple[CriterionDecisionRecord, ...]
    legacy_decided_count: int | None = None
    verification_coverage: str | None = None
    undecided_reason: str | None = None
    """Set when the package could not decide (for example the authority raised):
    every covered criterion is then indeterminate. Only such a decision may be
    recorded without a candidate verification of a runnable check."""

    @model_validator(mode="after")
    def _consistent(self) -> ReconciliationPayload:
        """Every acceptance bit agrees with the status and signal that decided it.

        The journal only admits decisions the reconciliation rule can produce
        (``acceptance.reconcile_acceptance``): the package accepts exactly its
        pass, unverified and uncovered criteria and never a fail or an
        indeterminate one; a criterion nobody attempted is never accepted; the
        existing verifier's decision is its own verdict, and of the package's
        passes it decides only one that rests on a worker-declared binding
        (``declared_binding_pass``; the display ``tier`` decides nothing); such
        a pass never accepts over the existing verifier's rejection; the run
        is accepted exactly when every criterion is. An undecided decision
        has no pass and no fail.
        """
        for item in self.criteria:
            status, governor = item.package_status, item.governed_by
            if status not in _DECISION_STATUSES or governor not in _GOVERNORS:
                raise ValueError("a decision names an unknown status or governor")
            if item.declared_binding_pass and status != "pass":
                raise ValueError("only a pass can rest on a worker-declared binding")
            if item.declared_binding_pass and item.accepted and not item.existing_accepted:
                raise ValueError("a declared-binding pass never overrules the existing verifier")
            if governor == "check_package":
                expected = status in _PACKAGE_ACCEPTS
            elif governor == "execution":
                expected = False
            else:
                expected = item.existing_accepted
                if status not in _EXISTING_DECIDES and not item.declared_binding_pass:
                    raise ValueError("the existing verifier decides only what the package did not")
            if item.accepted != expected:
                raise ValueError("a criterion's acceptance disagrees with what decided it")
            if self.undecided_reason and status not in _UNDECIDED_STATUSES:
                raise ValueError("an undecided decision carries only indeterminate or uncovered")
        if self.run_accepted != (bool(self.criteria) and all(i.accepted for i in self.criteria)):
            raise ValueError("the run's acceptance disagrees with its criteria")
        if self.criterion_count != len(self.criteria):
            raise ValueError("criterion_count disagrees with the criteria")
        return self


class ResumedPayload(ReconciliationPayload):
    """``boundary.acceptance.resumed``: the decision a resumed run recomputed."""

    source: str
    held_out_checks: tuple[str, ...] = ()
    reason: str | None = None


class ExcludedCasesRecord(_Payload):
    """Case ids of one oracle that the reference check excluded (ids, never values)."""

    check_id: str
    case_ids: tuple[str, ...]
    reason: str


class UncoveredRecord(_Payload):
    """A criterion left uncovered, with its reason."""

    criterion_key: str
    reason: str


class ReferenceCheckPayload(_Payload):
    """``boundary.oracle.reference_checked``: what the reference check excluded."""

    schema_version: str
    counts: dict[str, int]
    excluded_cases: tuple[ExcludedCasesRecord, ...]
    uncovered: tuple[UncoveredRecord, ...]


def _require(payload: object, model: type[_Payload]) -> None:
    if not isinstance(payload, model):
        raise TypeError(f"{model.__name__} expected, not {type(payload).__name__}")


def _event(boundary_id: str, event_type: str, data: dict[str, Any]) -> BaseEvent:
    return BaseEvent(
        type=event_type,
        aggregate_type=BOUNDARY_AGGREGATE_TYPE,
        aggregate_id=boundary_id,
        data=data,
    )


def cite(package_id: str | None, *, prefix: str = "") -> dict[str, str | None]:
    """``{"<prefix>package_id": package_id}``: how every event names a package."""
    return {f"{prefix}package_id": package_id}


def check_package_enabled_event(execution_id: str, contract: RunContract) -> BaseEvent:
    """The check package is on for the run ``execution_id``, with its run contract."""
    _require(contract, RunContract)
    return _event(
        execution_id,
        CHECK_PACKAGE_ENABLED,
        {"execution_id": execution_id, "contract": contract.journal_data()},
    )


def enabled_contract(data: dict[str, Any]) -> RunContract:
    """The run contract an enabled record carries; ``ValueError`` when it is malformed."""
    return RunContract.model_validate(data.get("contract"))


def package_frozen_event(boundary_id: str, package: CheckPackage) -> BaseEvent:
    """The package id and Seed digest (I2) plus the event-safe manifest summary.

    The package must be sealed (``package.seal_package``); its unkeyed digest
    is never recorded. ``record_sha256`` is computed here, from the same bytes
    the store writes (``package.package_record_bytes``: visible cases in full,
    held-out cases as ids only), so a resume in another process can tell
    whether the stored record was edited, and no caller can put another
    digest in its place.
    """
    return _event(
        boundary_id,
        PACKAGE_FROZEN,
        {
            **cite(package.package_id),
            "seed_digest": package.seed_digest,
            "record_sha256": sha256_bytes(package_record_bytes(package)),
            "manifest": package.manifest_summary(),
        },
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
    """Base admission receipt."""
    return _event(boundary_id, ADMISSION_COMPLETED, result.event_summary())


def actor_started_event(
    boundary_id: str,
    *,
    actor_id: str,
    package_id: str | None,
    runtime: str | None,
) -> BaseEvent:
    """A worker bound to this boundary started after the boundary was sealed."""
    return _event(
        boundary_id,
        ACTOR_STARTED,
        {
            "actor_id": actor_id,
            **cite(package_id),
            "runtime": runtime,
        },
    )


def candidate_verified_event(boundary_id: str, verification: CandidateVerification) -> BaseEvent:
    """Frozen package run on one candidate."""
    return _event(boundary_id, CANDIDATE_VERIFIED, verification.event_summary())


def superseded_event(
    boundary_id: str,
    *,
    superseded_by: str,
    package_id: str | None,
    successor_package_id: str | None,
    reason: str,
) -> BaseEvent:
    """Mark a sealed boundary version as replaced by a later version."""
    return _event(
        boundary_id,
        SUPERSEDED,
        {
            "superseded_by": superseded_by,
            **cite(package_id),
            **cite(successor_package_id, prefix="successor_"),
            "reason": reason,
        },
    )


def acceptance_reconciled_event(
    boundary_id: str,
    *,
    package_id: str | None,
    reconciliation: ReconciliationPayload,
) -> BaseEvent:
    """Per-criterion acceptance: package verdict, existing verdict (advisory), decision."""
    _require(reconciliation, ReconciliationPayload)
    return _event(
        boundary_id,
        ACCEPTANCE_RECONCILED,
        {**reconciliation.journal_data(), **cite(package_id)},
    )


def binding_recorded_event(
    boundary_id: str, *, package_id: str, payload: BindingsPayload
) -> BaseEvent:
    """Tiers and bindings of every check, recorded after the worker stopped."""
    _require(payload, BindingsPayload)
    return _event(boundary_id, BINDING_RECORDED, {**payload.journal_data(), **cite(package_id)})


def reference_checked_event(
    boundary_id: str, *, package_id: str, payload: ReferenceCheckPayload
) -> BaseEvent:
    """Cases excluded and criteria uncovered by the reference check (ids, never values)."""
    _require(payload, ReferenceCheckPayload)
    return _event(boundary_id, REFERENCE_CHECKED, {**payload.journal_data(), **cite(package_id)})


def acceptance_resumed_event(
    boundary_id: str, *, package_id: str | None, payload: ResumedPayload
) -> BaseEvent:
    """The package decision a resumed run recomputed (statuses and reasons, no case values)."""
    _require(payload, ResumedPayload)
    return _event(boundary_id, ACCEPTANCE_RESUMED, {**payload.journal_data(), **cite(package_id)})


# --------------------------------------------------------------------------
# The journal gateway. Every record the ledger appends, and every record replay
# (``ledger.advance``) or the recovery projection reads, passes
# ``validate_record`` first: the envelope, the record's exact closed schema
# (what its factory above writes, no field missing or added), and, once a
# boundary version is sealed, the identities the seal fixed. A record that
# fails is refused on write and flagged on replay, so a record the product
# could not have written reaches no transition and no projection.


class JournalRecordError(ValueError):
    """A journal record is not one the product writes (envelope, schema, or identity)."""


@dataclass(frozen=True, slots=True)
class FrozenIdentity:
    """What a boundary version's seal fixed; every later record must agree with it."""

    package_id: str | None
    seed_digest: str | None
    criterion_keys: frozenset[str] | None
    """The frozen manifest's criterion keys; ``None`` when the version has no manifest."""
    check_ids: frozenset[str]
    """The frozen package's check ids (none without a package)."""


RUN_IDENTITY = FrozenIdentity(
    package_id=None, seed_digest=None, criterion_keys=None, check_ids=frozenset()
)
"""A run aggregate's records cite no package, no Seed and no check."""


@dataclass(frozen=True, slots=True)
class Cited:
    """The identities one record names, compared with the ``FrozenIdentity``."""

    package_id: str | None
    seed_digest: str | None = None
    """Set only by records that carry a Seed digest."""
    criterion_keys: tuple[str, ...] = ()
    check_ids: tuple[str, ...] = ()
    decided: tuple[str, ...] | None = None
    """A decision's criteria: exactly the frozen manifest's, each once."""


class _Citing(Protocol):
    def cited(self) -> Cited: ...


class EnabledRecord(_Payload):
    """``boundary.check_package.enabled`` as journaled."""

    execution_id: str
    contract: RunContract

    def cited(self) -> Cited:
        return Cited(None)


class FrozenRecord(_Payload):
    """``boundary.check_package.frozen`` as journaled.

    ``manifest`` is the package's own summary (``CheckPackage.manifest_summary``);
    ``ledger.frozen_manifest`` validates the identities it holds.
    """

    package_id: str
    seed_digest: str
    record_sha256: str
    manifest: dict[str, Any]

    def cited(self) -> Cited:
        return Cited(self.package_id, self.seed_digest)


class ConstructionFailedRecord(_Payload):
    """``boundary.check_package.construction_failed`` as journaled."""

    seed_digest: str
    input_digest: str
    reason: str

    def cited(self) -> Cited:
        return Cited(None, self.seed_digest)


class ReferenceCheckedRecord(ReferenceCheckPayload):
    """``boundary.oracle.reference_checked`` as journaled."""

    package_id: str

    def cited(self) -> Cited:
        return Cited(
            self.package_id,
            criterion_keys=tuple(item.criterion_key for item in self.uncovered),
            check_ids=tuple(item.check_id for item in self.excluded_cases),
        )


class AdmissionRecord(AdmissionJournal):
    """``boundary.check_package.admission_completed`` as journaled."""

    def cited(self) -> Cited:
        return Cited(self.package_id, self.seed_digest, check_ids=self.check_ids())


class ActorStartedRecord(_Payload):
    """``boundary.actor.started`` as journaled."""

    actor_id: str
    package_id: str | None
    runtime: str | None

    def cited(self) -> Cited:
        return Cited(self.package_id)


class SupersededRecord(_Payload):
    """``boundary.check_package.superseded`` as journaled."""

    superseded_by: str
    package_id: str | None
    successor_package_id: str | None
    reason: str

    def cited(self) -> Cited:
        return Cited(self.package_id)


class BindingsRecord(BindingsPayload):
    """``boundary.binding.recorded`` as journaled."""

    package_id: str

    def cited(self) -> Cited:
        declared = [item.declared for item in self.checks if item.declared is not None]
        return Cited(
            self.package_id,
            criterion_keys=(
                *(item.criterion_key for item in self.checks),
                *(item.criterion_key for item in declared),
            ),
            check_ids=tuple(item.check_id for item in self.checks),
        )


class VerificationRecord(VerificationJournal):
    """``boundary.candidate.verified`` as journaled."""

    def cited(self) -> Cited:
        return Cited(self.package_id, self.seed_digest, check_ids=self.check_ids())


def _decision_cited(package_id: str | None, criteria: tuple[CriterionDecisionRecord, ...]) -> Cited:
    keys = tuple(item.criterion_key for item in criteria)
    return Cited(package_id, criterion_keys=keys, decided=keys)


class ReconciledRecord(ReconciliationPayload):
    """``boundary.acceptance.reconciled`` as journaled."""

    package_id: str | None

    def cited(self) -> Cited:
        return _decision_cited(self.package_id, self.criteria)


class ResumedRecord(ResumedPayload):
    """``boundary.acceptance.resumed`` as journaled (on a version or on the run)."""

    package_id: str | None

    def cited(self) -> Cited:
        cited = _decision_cited(self.package_id, self.criteria)
        return replace(cited, check_ids=self.held_out_checks)


JOURNAL_RECORDS: Mapping[str, type[BaseModel]] = {
    CHECK_PACKAGE_ENABLED: EnabledRecord,
    PACKAGE_FROZEN: FrozenRecord,
    CONSTRUCTION_FAILED: ConstructionFailedRecord,
    REFERENCE_CHECKED: ReferenceCheckedRecord,
    ADMISSION_COMPLETED: AdmissionRecord,
    ACTOR_STARTED: ActorStartedRecord,
    SUPERSEDED: SupersededRecord,
    BINDING_RECORDED: BindingsRecord,
    CANDIDATE_VERIFIED: VerificationRecord,
    ACCEPTANCE_RECONCILED: ReconciledRecord,
    ACCEPTANCE_RESUMED: ResumedRecord,
}
"""The one closed schema of every boundary record, by event type."""

_LABELS = {
    CHECK_PACKAGE_ENABLED: "the run's enabled record",
    PACKAGE_FROZEN: "the frozen record",
    CONSTRUCTION_FAILED: "the construction_failed record",
    REFERENCE_CHECKED: "a reference check",
    ADMISSION_COMPLETED: "admission receipt",
    ACTOR_STARTED: "actor start",
    SUPERSEDED: "a supersession",
    BINDING_RECORDED: "bindings",
    CANDIDATE_VERIFIED: "candidate verification",
    ACCEPTANCE_RECONCILED: "acceptance",
    ACCEPTANCE_RESUMED: "a resumed decision",
}
_DECISIONS = frozenset({ACCEPTANCE_RECONCILED, ACCEPTANCE_RESUMED})


def validate_record(event: BaseEvent, frozen: FrozenIdentity | None) -> Any:
    """The typed record ``event`` holds; ``JournalRecordError`` unless the product wrote it.

    Checks the envelope (a boundary aggregate, a known record type), the
    record's exact closed schema (``JOURNAL_RECORDS``), and, with ``frozen``
    (the identities of a sealed version, or ``RUN_IDENTITY``), that the record
    cites the sealed package and Seed and names only the frozen checks and
    criteria, a decision exactly the frozen criteria.
    """
    model = JOURNAL_RECORDS.get(event.type)
    if event.aggregate_type != BOUNDARY_AGGREGATE_TYPE or not event.aggregate_id or model is None:
        raise JournalRecordError(f"{event.type} is not a boundary record")
    try:
        record = model.model_validate(event.data)
    except ValidationError as exc:
        if event.type in _DECISIONS:
            raise JournalRecordError(
                "a recorded decision disagrees with the statuses that decided it"
            ) from exc
        raise JournalRecordError(
            f"{_LABELS[event.type]} is not a record the product writes"
        ) from exc
    if frozen is not None:
        _bind(_LABELS[event.type], cast(_Citing, record).cited(), frozen)
    return record


def _bind(label: str, cited: Cited, frozen: FrozenIdentity) -> None:
    if cited.package_id != frozen.package_id:
        raise JournalRecordError(f"{label} names a different package than the boundary's seal")
    if cited.seed_digest is not None and cited.seed_digest != frozen.seed_digest:
        raise JournalRecordError(f"{label} names a different Seed than the frozen package")
    if not set(cited.check_ids) <= frozen.check_ids:
        raise JournalRecordError(f"{label} names a check the frozen package does not hold")
    if frozen.criterion_keys is None:
        return
    if not set(cited.criterion_keys) <= frozen.criterion_keys:
        raise JournalRecordError(f"{label} names a criterion the frozen manifest does not hold")
    decided = cited.decided
    if decided is not None and (
        len(set(decided)) != len(decided) or set(decided) != frozen.criterion_keys
    ):
        raise JournalRecordError(f"{label} does not decide exactly the frozen manifest's criteria")


def validate_run_record(event: BaseEvent, execution_id: str) -> EnabledRecord | ResumedRecord:
    """A record of the run's own aggregate; ``JournalRecordError`` unless the product wrote it.

    The product writes there only the enabled record of this run
    (``check_package_enabled_event``) and a resumed decision made without a
    usable boundary (``BoundaryLedger.record_resumed_undecided``): it cites no
    package and no check, states why, and decides nothing (every criterion
    indeterminate or uncovered).
    """
    if event.aggregate_id != execution_id or event.type not in (
        CHECK_PACKAGE_ENABLED,
        ACCEPTANCE_RESUMED,
    ):
        raise JournalRecordError(
            "the run aggregate holds a record the product does not write there"
        )
    record = validate_record(event, RUN_IDENTITY)
    if isinstance(record, EnabledRecord) and record.execution_id != execution_id:
        raise JournalRecordError("the run's enabled record names another run")
    if isinstance(record, ResumedRecord) and (
        not record.reason
        or any(item.package_status not in _UNDECIDED_STATUSES for item in record.criteria)
    ):
        raise JournalRecordError("a run-level resumed decision must be undecided and say why")
    return cast(EnabledRecord | ResumedRecord, record)
