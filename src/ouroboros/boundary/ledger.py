"""EventStore-backed ordering guard for the check-package boundary.

A boundary is one (Seed, verifier source) slot, for example one task and one
check variant. Its lifecycle is one reducer (``advance``), applied at write
time to the event about to be appended and at replay time by
``verify_boundary_order``:

1. exactly one seal: ``record_package_frozen`` (the sealed package's opaque
   id and Seed digest persisted, ``package.seal_package``) or
   ``record_construction_failed``; a second package for the same boundary is
   refused, so admission feedback can never produce a regenerated package;
2. for a frozen package, exactly one ``record_admission`` whose receipt names
   the frozen package id and Seed digest, recorded before any actor starts;
3. ``record_actor_started`` refuses to start a worker until every boundary it
   binds is sealed (and, with a package, admitted) and not superseded; a
   version of a run binds only that run's worker; with a workspace it
   refuses one that contains a generated check file (scanned with the live
   sealed packages, so a renamed copy of the oracle data file is found);
4. every later record (bindings, candidate verification, acceptance) must
   cite the frozen package id (a candidate verification also its Seed
   digest), and a superseded version accepts none;
5. a version is superseded only by a later version of the same run.

A boundary version of a run (``events.boundary_version_id``) is sealed only
after the run's ``record_check_package_enabled``, and that record is refused
once a version of the run exists, so it is always the run's first record.

Regeneration policy. The seal rule above is per boundary id and never
changes. The product run path gives each attempt its own boundary version id
and calls ``record_superseded`` on the old version once the new one is
sealed; the old package stays in the journal, marked superseded. The ledger
assumes one writer per boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from ouroboros.boundary.events import (
    ACCEPTANCE_RECONCILED,
    ACCEPTANCE_RESUMED,
    ACTOR_STARTED,
    ADMISSION_COMPLETED,
    BINDING_RECORDED,
    BOUNDARY_AGGREGATE_TYPE,
    CANDIDATE_VERIFIED,
    CHECK_PACKAGE_ENABLED,
    CONSTRUCTION_FAILED,
    PACKAGE_FROZEN,
    REFERENCE_CHECKED,
    SUPERSEDED,
    BindingsPayload,
    ReconciliationPayload,
    ReferenceCheckPayload,
    ResumedPayload,
    RunContract,
    acceptance_reconciled_event,
    acceptance_resumed_event,
    actor_started_event,
    admission_completed_event,
    binding_recorded_event,
    candidate_verified_event,
    check_package_enabled_event,
    construction_failed_event,
    enabled_contract,
    package_frozen_event,
    parse_boundary_version,
    reference_checked_event,
    superseded_event,
)
from ouroboros.boundary.package import (
    CheckPackage,
    find_workspace_leaks,
    validate_package_for_seed,
)
from ouroboros.boundary.receipts import AdmissionResult, CandidateVerification
from ouroboros.core.errors import OuroborosError
from ouroboros.core.seed import Seed
from ouroboros.events.base import BaseEvent
from ouroboros.persistence.event_store import EventStore


class BoundaryOrderError(OuroborosError):
    """A boundary write would violate the seal, admission, or actor ordering."""


class BoundaryLeakError(BoundaryOrderError):
    """A worker workspace contains generated check files."""


def _utc(value: datetime) -> datetime:
    """Replayed SQLite timestamps are naive UTC; compare everything as aware UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _first(events: Sequence[BaseEvent], event_type: str) -> BaseEvent | None:
    return next((event for event in events if event.type == event_type), None)


def _ref(event: BaseEvent | None) -> str | None:
    """The package id an event cites."""
    return None if event is None else event.data.get("package_id")


# --------------------------------------------------------------------------
# The transition reducer: the one definition of a valid boundary journal.
# Every write path advances the replayed state with the event it is about to
# append, and ``verify_boundary_order`` advances it over replayed events; so a
# record the ledger would refuse is exactly a record replay flags.


@dataclass(frozen=True, slots=True)
class VersionState:
    """What one boundary version's journal has established so far."""

    seal: str | None = None
    """``PACKAGE_FROZEN``, ``CONSTRUCTION_FAILED``, or ``None`` (not sealed)."""
    package_id: str | None = None
    seed_digest: str | None = None
    gate_time: datetime | None = None
    """When the latest seal or admission record was written (an actor starts after it)."""
    admitted: bool = False
    started: bool = False
    superseded: bool = False
    final_bindings: bool = False
    runnable: bool = False
    """The final bindings put at least one check on a runnable tier (``A``, ``A_prime``)."""
    verified: bool = False
    reconciled: bool = False

    @property
    def frozen(self) -> bool:
        return self.seal == PACKAGE_FROZEN


_NOT_SEALED = {
    ACTOR_STARTED: "actor started before the seal (the boundary is not sealed)",
    ADMISSION_COMPLETED: "admission recorded before the seal (no frozen package)",
}


def advance(state: VersionState, event: BaseEvent) -> VersionState:
    """The state after ``event``; ``BoundaryOrderError`` when the journal forbids it."""
    kind = event.type
    if state.superseded:
        if kind == SUPERSEDED:
            raise BoundaryOrderError("boundary version already superseded")
        raise BoundaryOrderError(f"{kind} recorded on a superseded boundary version")
    if kind in (PACKAGE_FROZEN, CONSTRUCTION_FAILED):
        if state.seal is not None:
            raise BoundaryOrderError(
                "boundary already sealed; a package cannot be regenerated or replaced"
            )
        return replace(
            state,
            seal=kind,
            package_id=_ref(event) if kind == PACKAGE_FROZEN else None,
            seed_digest=event.data.get("seed_digest"),
            gate_time=_utc(event.timestamp),
        )
    if state.seal is None:
        raise BoundaryOrderError(_NOT_SEALED.get(kind, f"{kind} recorded before the seal"))
    cites = _ref(event) == state.package_id
    if kind == REFERENCE_CHECKED:
        if not state.frozen or not cites:
            raise BoundaryOrderError("a reference check must cite the boundary's frozen package")
        if state.admitted:
            raise BoundaryOrderError("the reference check precedes admission")
        return state
    if kind == ADMISSION_COMPLETED:
        if not state.frozen:
            raise BoundaryOrderError("admission recorded without a frozen package")
        if not cites:
            raise BoundaryOrderError("admission receipt names a different package")
        if event.data.get("seed_digest") != state.seed_digest:
            raise BoundaryOrderError(
                "admission receipt names a different Seed than the frozen package"
            )
        if state.admitted:
            raise BoundaryOrderError("admission already recorded")
        if state.started:
            raise BoundaryOrderError("admission must be recorded before any actor starts")
        return replace(state, admitted=True, gate_time=_utc(event.timestamp))
    if kind == ACTOR_STARTED:
        run = parse_boundary_version(event.aggregate_id)
        if run is not None and event.data.get("actor_id") != run[0]:
            raise BoundaryOrderError("a boundary version of a run binds only that run's worker")
        if state.frozen and not state.admitted:
            raise BoundaryOrderError("actor started before admission")
        if not cites:
            raise BoundaryOrderError("actor start does not cite the sealed package")
        if state.gate_time is not None and _utc(event.timestamp) <= state.gate_time:
            raise BoundaryOrderError("actor start timestamp does not follow the boundary seal")
        return replace(state, started=True)
    if kind == SUPERSEDED:
        _require_successor(event.aggregate_id, event.data.get("superseded_by"))
        if state.started:
            raise BoundaryOrderError("a boundary version bound to a worker cannot be superseded")
        if not cites:
            raise BoundaryOrderError("a supersession must cite the sealed package")
        return replace(state, superseded=True)
    if kind == BINDING_RECORDED:
        if not state.frozen or not cites:
            raise BoundaryOrderError("bindings must cite the boundary's frozen package")
        if not (state.admitted and state.started):
            raise BoundaryOrderError(
                "bindings are recorded only after admission and the worker start"
            )
        if event.data.get("phase") != "final":
            return state
        if state.final_bindings:
            raise BoundaryOrderError("final bindings already recorded")
        if state.verified:
            raise BoundaryOrderError("final bindings recorded after the candidate verification")
        runnable = any(
            check.get("tier") in _RUNNABLE_TIERS for check in event.data.get("checks") or ()
        )
        return replace(state, final_bindings=True, runnable=runnable)
    if kind == CANDIDATE_VERIFIED:
        if not state.frozen or not state.admitted:
            raise BoundaryOrderError("candidate verification requires a frozen, admitted package")
        if not cites:
            raise BoundaryOrderError("verification ran a package other than the frozen one")
        if event.data.get("seed_digest") != state.seed_digest:
            raise BoundaryOrderError(
                "candidate verification names a different Seed than the frozen package"
            )
        return replace(state, verified=True)
    if kind in (ACCEPTANCE_RECONCILED, ACCEPTANCE_RESUMED):
        _require_consistent_decision(event)
    if kind == ACCEPTANCE_RECONCILED:
        if state.reconciled:
            raise BoundaryOrderError("acceptance already reconciled")
        if state.frozen:
            if not cites:
                raise BoundaryOrderError("acceptance must cite the boundary's frozen package")
            if not state.verified:
                _require_unverified_decision(state, event)
        elif _ref(event) is not None or not state.started:
            raise BoundaryOrderError(
                "a package-less decision needs a construction_failed seal and a worker"
            )
        return replace(state, reconciled=True)
    if kind == ACCEPTANCE_RESUMED:
        if not state.frozen or not cites:
            raise BoundaryOrderError("a resumed decision must cite the boundary's frozen package")
        if not (state.admitted and state.started):
            raise BoundaryOrderError(
                "a resumed decision needs an admitted package and a started worker"
            )
        return state
    raise BoundaryOrderError(f"{kind} is not a boundary version record")


_RUNNABLE_TIERS = frozenset({"A", "A_prime"})
_PAGE = 500
# Statuses a decision may carry without a candidate verification: when the
# final bindings left no check runnable, the checks were not run (unverified or
# indeterminate); when the package could not decide at all, every covered
# criterion is indeterminate and the rest are uncovered.
_UNRUN_STATUSES = frozenset({"unverified", "uncovered", "indeterminate"})
_UNDECIDED_STATUSES = frozenset({"uncovered", "indeterminate"})


def _require_consistent_decision(event: BaseEvent) -> None:
    """A recorded decision is one the reconciliation rule can produce.

    The payload model checks every acceptance bit against the status and the
    signal that decided it (``ReconciliationPayload``); replay applies the same
    check to what the journal holds, so a decision edited in place is flagged.
    """
    model = ResumedPayload if event.type == ACCEPTANCE_RESUMED else ReconciliationPayload
    data = {key: value for key, value in (event.data or {}).items() if key != "package_id"}
    try:
        model.model_validate(data)
    except ValueError as exc:
        raise BoundaryOrderError(
            "a recorded decision disagrees with the statuses that decided it"
        ) from exc


def _require_unverified_decision(state: VersionState, event: BaseEvent) -> None:
    """A decision recorded without a candidate verification claims nothing a run would show.

    Allowed only when the final bindings left no check runnable, or when the
    decision says the package could not decide (``undecided_reason``) after
    the worker started; either way no criterion is a pass or a fail.
    """
    if state.final_bindings and not state.runnable:
        allowed = _UNRUN_STATUSES
    elif event.data.get("undecided_reason") and state.admitted and state.started:
        allowed = _UNDECIDED_STATUSES
    else:
        raise BoundaryOrderError("acceptance must cite a verification of the frozen package")
    statuses = {item.get("package_status") for item in event.data.get("criteria") or ()}
    if not statuses <= allowed:
        raise BoundaryOrderError(
            "a decision recorded without a candidate verification claims a verified status"
        )


def _require_successor(boundary_id: str, successor: object) -> None:
    """A version is superseded only by a later version of the same run."""
    old = parse_boundary_version(boundary_id)
    new = parse_boundary_version(successor) if isinstance(successor, str) else None
    if old is None or new is None:
        raise BoundaryOrderError("only a boundary version of a run can supersede another")
    if new[0] != old[0]:
        raise BoundaryOrderError("a boundary version is superseded only within its own run")
    if new[1] <= old[1]:
        raise BoundaryOrderError("a boundary version is superseded only by a later version")


def version_state(events: Iterable[BaseEvent]) -> VersionState:
    """Advance over a boundary version's replayed events; raises on the first violation."""
    state = VersionState()
    for event in events:
        state = advance(state, event)
    return state


class BoundaryLedger:
    """Write-time ordering guard over an initialized ``EventStore``.

    Every write replays the boundary version's journal, advances the reducer
    (``advance``) with the event it is about to append, and appends only when
    that succeeds; a journal that is already inconsistent refuses every write.
    """

    def __init__(self, store: EventStore) -> None:
        self._store = store

    async def events(self, boundary_id: str) -> list[BaseEvent]:
        """Replay one boundary's events in journal order."""
        return await self._store.replay(BOUNDARY_AGGREGATE_TYPE, boundary_id)

    async def _state(self, boundary_id: str) -> VersionState:
        state = version_state(await self.events(boundary_id))
        return state

    async def _append(self, boundary_id: str, event: BaseEvent) -> BaseEvent:
        advance(await self._state(boundary_id), event)
        await self._store.append(event)
        return event

    async def record_check_package_enabled(
        self, execution_id: str, contract: RunContract
    ) -> BaseEvent:
        """Record, once and before anything else, that the check package is on for a run.

        ``contract`` holds the settings that decide anything after the worker
        starts; a resumed run uses them instead of the live config
        (``run_contract``).

        Refused once any boundary version of the run exists: the record must
        come first. A resumed run reads it back
        (``resume.load_resumed_boundary``): with this record present, a
        missing or malformed boundary makes the resumed decision undecided
        instead of letting the legacy verifier decide the criteria the package
        may cover.
        """
        if not execution_id:
            raise BoundaryOrderError("the check package needs the run's execution id")
        if await self.check_package_enabled(execution_id):
            raise BoundaryOrderError(
                "the check package was already enabled for this run",
                details={"execution_id": execution_id},
            )
        if await self.run_versions(execution_id):
            raise BoundaryOrderError(
                "the enabled record must precede every boundary version of the run",
                details={"execution_id": execution_id},
            )
        event = check_package_enabled_event(execution_id, contract)
        await self._store.append(event)
        return event

    async def run_versions(self, execution_id: str) -> dict[int, list[BaseEvent]]:
        """Every boundary version of the run in the journal, by version number.

        Found by what the journal holds, not by counting up from ``v1``: a
        version recorded without its predecessors (for example written into
        the store directly) is found too, so a run-level rule cannot be
        passed by a gap.
        """
        found: dict[int, str] = {}
        offset = 0
        while True:
            page = await self._store.query_events(
                aggregate_type=BOUNDARY_AGGREGATE_TYPE, limit=_PAGE, offset=offset
            )
            for event in page:
                run = parse_boundary_version(event.aggregate_id)
                if run is not None and run[0] == execution_id:
                    found[run[1]] = event.aggregate_id
            if len(page) < _PAGE:
                break
            offset += _PAGE
        return {version: await self.events(found[version]) for version in sorted(found)}

    async def check_package_enabled(self, execution_id: str) -> bool:
        """Whether ``record_check_package_enabled`` ran for ``execution_id``."""
        return _first(await self.events(execution_id), CHECK_PACKAGE_ENABLED) is not None

    async def run_contract(self, execution_id: str) -> RunContract | None:
        """The run contract recorded for ``execution_id``; ``None`` when the run was never on.

        Raises ``BoundaryOrderError`` when the enabled record exists but its
        contract is missing or malformed: the settings the run started with
        are then unknown.
        """
        event = _first(await self.events(execution_id), CHECK_PACKAGE_ENABLED)
        if event is None:
            return None
        try:
            return enabled_contract(event.data)
        except ValueError as exc:
            raise BoundaryOrderError(
                "the run's check package contract is malformed",
                details={"execution_id": execution_id},
            ) from exc

    async def _require_run_enabled(self, boundary_id: str) -> None:
        run = parse_boundary_version(boundary_id)
        if run is not None and not await self.check_package_enabled(run[0]):
            raise BoundaryOrderError(
                "a boundary version of a run is sealed only after the run's enabled record",
                details={"boundary_id": boundary_id},
            )

    async def record_package_frozen(
        self,
        boundary_id: str,
        package: CheckPackage,
        *,
        seed: Seed | None = None,
    ) -> BaseEvent:
        """Persist the package id, Seed digest and manifest; the boundary's only seal.

        The package id (``package.seal_package``) is what I2 orders before any
        worker start. Every later receipt and event must cite the same one.
        """
        if seed is not None:
            validate_package_for_seed(package, seed)
        await self._require_run_enabled(boundary_id)
        return await self._append(boundary_id, package_frozen_event(boundary_id, package))

    async def record_construction_failed(
        self,
        boundary_id: str,
        *,
        seed_digest: str,
        input_digest: str,
        reason: str,
    ) -> BaseEvent:
        """Seal a boundary whose generation produced no valid package."""
        await self._require_run_enabled(boundary_id)
        event = construction_failed_event(
            boundary_id, seed_digest=seed_digest, input_digest=input_digest, reason=reason
        )
        return await self._append(boundary_id, event)

    async def record_admission(self, boundary_id: str, result: AdmissionResult) -> BaseEvent:
        """Persist the single admission receipt (same package id and Seed as the seal)."""
        return await self._append(boundary_id, admission_completed_event(boundary_id, result))

    async def record_actor_started(
        self,
        actor_id: str,
        boundary_ids: Sequence[str],
        *,
        workspace: Path | None = None,
        runtime: str | None = None,
        packages: Sequence[CheckPackage] = (),
    ) -> list[BaseEvent]:
        """Record a worker start on every boundary it is bound to.

        Raises ``BoundaryOrderError`` unless each boundary is sealed, not
        superseded and, when it holds a package, admitted. With
        ``workspace``, every frozen boundary's live sealed package must be in
        ``packages`` (the one whose id the seal cites), and
        ``BoundaryLeakError`` is raised when the workspace contains a
        generated check file, by path or by the in-memory digest of any
        package file, the oracle data file included (the journal manifest
        cannot find a renamed copy of it). Call this before launching the
        worker; launch only on success.
        """
        if not boundary_ids:
            raise BoundaryOrderError("an actor must be bound to at least one boundary")
        started: list[BaseEvent] = []
        live: list[CheckPackage] = []
        by_id = {package.package_id: package for package in packages}
        for boundary_id in boundary_ids:
            events = await self.events(boundary_id)
            state = version_state(events)
            event = actor_started_event(
                boundary_id, actor_id=actor_id, package_id=state.package_id, runtime=runtime
            )
            try:
                advance(state, event)
            except BoundaryOrderError as exc:
                raise BoundaryOrderError(
                    f"actor cannot start: {exc.message}",
                    details={"boundary_id": boundary_id, "actor_id": actor_id},
                ) from exc
            if state.frozen and workspace is not None:
                package = by_id.get(state.package_id or "")
                if package is None:
                    raise BoundaryOrderError(
                        "actor cannot start: the leak scan needs the boundary's sealed package",
                        details={"boundary_id": boundary_id, "actor_id": actor_id},
                    )
                live.append(package)
            started.append(event)
        if workspace is not None:
            leaks = find_workspace_leaks(workspace, live)
            if leaks:
                raise BoundaryLeakError(
                    "worker workspace contains generated check files",
                    details={"actor_id": actor_id, "paths": list(leaks)},
                )
        await self._store.append_batch(started)
        return started

    async def record_superseded(
        self,
        boundary_id: str,
        *,
        superseded_by: str,
        reason: str,
    ) -> BaseEvent:
        """Mark ``boundary_id`` as replaced by the sealed version ``superseded_by``.

        Refused unless both ids are versions of the same run and
        ``superseded_by`` is a later version, both versions are sealed, the old version
        is not already superseded, and no actor was ever bound to the old
        version (a worker's verdict must cite the package it was bound to).
        A superseded version accepts no later record, an actor start included.
        """
        try:
            _require_successor(boundary_id, superseded_by)
        except BoundaryOrderError as exc:
            raise BoundaryOrderError(
                exc.message, details={"boundary_id": boundary_id, "superseded_by": superseded_by}
            ) from exc
        old = await self._state(boundary_id)
        new = await self._state(superseded_by)
        if old.seal is None or new.seal is None:
            raise BoundaryOrderError(
                "both boundary versions must be sealed before one supersedes the other",
                details={"boundary_id": boundary_id, "superseded_by": superseded_by},
            )
        event = superseded_event(
            boundary_id,
            superseded_by=superseded_by,
            package_id=old.package_id,
            successor_package_id=new.package_id,
            reason=reason,
        )
        return await self._append(boundary_id, event)

    async def record_bindings(
        self, boundary_id: str, *, package_id: str, payload: BindingsPayload
    ) -> BaseEvent:
        """Record every check's tier and binding once the worker has stopped.

        ``package_id`` is the sealed package id (``CheckPackage.package_id``).
        Refused unless the frozen, admitted package is cited and the worker
        started on this boundary. A ``final`` record is single and must precede
        the candidate verification it governs; ``repair`` records may repeat.
        """
        event = binding_recorded_event(boundary_id, package_id=package_id, payload=payload)
        return await self._append(boundary_id, event)

    async def record_candidate_verification(
        self, boundary_id: str, verification: CandidateVerification
    ) -> BaseEvent:
        """Persist a candidate run of the frozen, admitted package."""
        return await self._append(boundary_id, candidate_verified_event(boundary_id, verification))

    async def record_acceptance_reconciled(
        self, boundary_id: str, *, package_id: str | None, reconciliation: ReconciliationPayload
    ) -> BaseEvent:
        """Persist the per-criterion acceptance decision once, after verification.

        With a package it must follow a candidate verification of the frozen
        package; without one it may only record a decision that claims no
        verified status: after final bindings that left no check runnable, or
        a decision the package could not make (``undecided_reason``, every
        covered criterion indeterminate). Without a package (``package_id`` is ``None``) the boundary
        must be sealed as ``construction_failed`` and a worker must have
        started on it.
        """
        event = acceptance_reconciled_event(
            boundary_id, package_id=package_id, reconciliation=reconciliation
        )
        return await self._append(boundary_id, event)

    async def record_reference_checked(
        self, boundary_id: str, *, package_id: str, payload: ReferenceCheckPayload
    ) -> BaseEvent:
        """Record what the reference check excluded, after the seal and before admission."""
        event = reference_checked_event(boundary_id, package_id=package_id, payload=payload)
        return await self._append(boundary_id, event)

    async def record_acceptance_resumed(
        self, boundary_id: str, *, package_id: str, payload: ResumedPayload
    ) -> BaseEvent:
        """Record a resumed run's recomputed package decision (one per resume).

        Refused unless the frozen, admitted package is cited and a worker was
        started on this boundary. The recomputation may run a visible-only
        package, which is not the frozen one; it therefore cites the frozen
        package id here instead of recording a candidate verification.
        """
        event = acceptance_resumed_event(boundary_id, package_id=package_id, payload=payload)
        return await self._append(boundary_id, event)

    async def record_resumed_undecided(
        self, execution_id: str, *, payload: ResumedPayload
    ) -> BaseEvent:
        """Record a resumed decision made without a usable boundary (no package cited).

        Written on the run's own aggregate (``execution_id``), next to the
        record that the check package was on, because no boundary version
        can be cited.
        """
        if not execution_id or parse_boundary_version(execution_id) is not None:
            raise BoundaryOrderError("a resumed decision needs the run's execution id")
        if not await self.check_package_enabled(execution_id):
            raise BoundaryOrderError(
                "an undecided resume is recorded only for a run whose check package was on",
                details={"execution_id": execution_id},
            )
        event = acceptance_resumed_event(execution_id, package_id=None, payload=payload)
        await self._store.append(event)
        return event


def verify_boundary_order(
    events: Sequence[BaseEvent], *, run_events: Sequence[BaseEvent] | None = None
) -> tuple[str, ...]:
    """Return ordering violations in one boundary version's replayed events.

    The same reducer (``advance``) as every write path decides each event; an
    event it refuses is reported and skipped. An empty result means: one
    seal, the seal precedes admission, admission precedes every actor start,
    no record follows a supersession, and every receipt cites the frozen
    package id. With ``run_events`` (the run aggregate's events) the run's
    enabled record must also precede the version's first record.
    """
    violations: list[str] = []
    state = VersionState()
    for event in events:
        try:
            state = advance(state, event)
        except BoundaryOrderError as exc:
            violations.append(exc.message)
    seals = [e for e in events if e.type in {PACKAGE_FROZEN, CONSTRUCTION_FAILED}]
    if len(seals) != 1:
        violations.append(f"expected exactly one seal, found {len(seals)}")
    if state.frozen and not state.superseded:
        admissions = [e for e in events if e.type == ADMISSION_COMPLETED]
        if len(admissions) != 1:
            violations.append(f"expected exactly one admission, found {len(admissions)}")
    if any(key.endswith("package_sha256") for event in events for key in event.data):
        # An unkeyed digest of the full package would let a reader confirm
        # guessed held-out values; events cite the opaque package id only.
        violations.append("a boundary event records an unkeyed package digest")
    if run_events is not None and events:
        enabled = _first(run_events, CHECK_PACKAGE_ENABLED)
        if enabled is None:
            violations.append("the run has no enabled record")
        elif _utc(enabled.timestamp) > _utc(events[0].timestamp):
            violations.append("a boundary version was recorded before the run's enabled record")
    return tuple(violations)
