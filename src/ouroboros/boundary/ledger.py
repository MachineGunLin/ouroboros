"""EventStore-backed ordering guard for the check-package boundary.

A boundary is one (Seed, verifier source) slot, for example one task and one
check variant. Its lifecycle is enforced at write time:

1. exactly one seal: ``record_package_frozen`` (the package reference
   persisted: for a committed package the commitment SHA-256(salt ||
   canonical bytes), else its SHA-256) or ``record_construction_failed``; a second package for the same boundary is
   refused, so admission feedback can never produce a regenerated package;
2. for a frozen package, exactly one ``record_admission`` whose receipt names
   the frozen digest, recorded before any actor starts;
3. ``record_actor_started`` refuses to start a worker until every boundary it
   binds is sealed (and, with a package, admitted), optionally refusing a
   workspace that contains generated check files;
4. ``record_candidate_verification`` and ``record_selection`` must cite the
   frozen digest.

Regeneration policy. The seal rule above is per boundary id and never
changes. A caller that allows regeneration (the product run path) gives each
attempt its own boundary version id and calls ``record_superseded`` on the old
version once the new one is sealed; the old package stays in the journal,
marked superseded. A caller that forbids regeneration (an evaluation harness)
never calls it, so "exactly one package per boundary" holds unchanged.

``verify_boundary_order`` re-checks the same rules over replayed events for
audit and replay. The ledger assumes one writer per boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ouroboros.boundary.admission import AdmissionResult, CandidateVerification
from ouroboros.boundary.events import (
    ACCEPTANCE_RECONCILED,
    ACCEPTANCE_RESUMED,
    ACTOR_STARTED,
    ADMISSION_COMPLETED,
    BINDING_RECORDED,
    BOUNDARY_AGGREGATE_TYPE,
    CANDIDATE_VERIFIED,
    CASE_REVEALED,
    CONSTRUCTION_FAILED,
    LEGACY_FALLBACK,
    PACKAGE_FROZEN,
    SELECTION_DECIDED,
    SUPERSEDED,
    acceptance_reconciled_event,
    acceptance_resumed_event,
    actor_started_event,
    admission_completed_event,
    binding_recorded_event,
    candidate_verified_event,
    case_revealed_event,
    construction_failed_event,
    legacy_fallback_event,
    package_frozen_event,
    reference_checked_event,
    selection_decided_event,
    superseded_event,
)
from ouroboros.boundary.package import (
    CheckPackage,
    cited_reference,
    find_workspace_leaks,
    validate_package_for_seed,
)
from ouroboros.boundary.selection import SelectionDecision
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
    """The package reference an event cites (commitment or plain digest)."""
    return None if event is None else cited_reference(event.data)


def _committed(seal: BaseEvent | None) -> bool:
    return seal is not None and bool(seal.data.get("package_commitment"))


class BoundaryLedger:
    """Write-time ordering guard over an initialized ``EventStore``."""

    def __init__(self, store: EventStore) -> None:
        self._store = store

    async def events(self, boundary_id: str) -> list[BaseEvent]:
        """Replay one boundary's events in journal order."""
        return await self._store.replay(BOUNDARY_AGGREGATE_TYPE, boundary_id)

    async def _require_unsealed(self, boundary_id: str) -> None:
        events = await self.events(boundary_id)
        if _first(events, PACKAGE_FROZEN) or _first(events, CONSTRUCTION_FAILED):
            raise BoundaryOrderError(
                "boundary already sealed; a package cannot be regenerated or replaced",
                details={"boundary_id": boundary_id},
            )

    async def record_package_frozen(
        self,
        boundary_id: str,
        package: CheckPackage,
        *,
        seed: Seed | None = None,
        record_sha256: str | None = None,
    ) -> BaseEvent:
        """Persist the package reference and manifest; the boundary's only seal.

        The reference is what I2 orders before any worker start: the
        commitment of a committed package (``commit_package``), else the
        package SHA-256. Every later receipt and event must cite the same one.
        """
        if seed is not None:
            validate_package_for_seed(package, seed)
        await self._require_unsealed(boundary_id)
        event = package_frozen_event(boundary_id, package, record_sha256=record_sha256)
        await self._store.append(event)
        return event

    async def record_construction_failed(
        self,
        boundary_id: str,
        *,
        seed_digest: str,
        input_digest: str,
        reason: str,
    ) -> BaseEvent:
        """Seal a boundary whose generation produced no valid package."""
        await self._require_unsealed(boundary_id)
        event = construction_failed_event(
            boundary_id, seed_digest=seed_digest, input_digest=input_digest, reason=reason
        )
        await self._store.append(event)
        return event

    async def record_admission(self, boundary_id: str, result: AdmissionResult) -> BaseEvent:
        """Persist the single whole-package admission receipt."""
        events = await self.events(boundary_id)
        frozen = _first(events, PACKAGE_FROZEN)
        if frozen is None:
            raise BoundaryOrderError(
                "admission requires a frozen package", details={"boundary_id": boundary_id}
            )
        if _ref(frozen) != cited_reference(result):
            raise BoundaryOrderError(
                "admission receipt names a different package",
                details={"boundary_id": boundary_id},
            )
        if _first(events, ADMISSION_COMPLETED) is not None:
            raise BoundaryOrderError(
                "admission already recorded", details={"boundary_id": boundary_id}
            )
        if _first(events, ACTOR_STARTED) is not None:
            raise BoundaryOrderError(
                "admission must be recorded before any actor starts",
                details={"boundary_id": boundary_id},
            )
        event = admission_completed_event(boundary_id, result)
        await self._store.append(event)
        return event

    async def record_actor_started(
        self,
        actor_id: str,
        boundary_ids: Sequence[str],
        *,
        workspace: Path | None = None,
        runtime: str | None = None,
        assignment: str | None = None,
    ) -> list[BaseEvent]:
        """Record a worker start on every boundary it is bound to.

        Raises ``BoundaryOrderError`` unless each boundary is sealed and, when
        it holds a package, admitted. Raises ``BoundaryLeakError`` when
        ``workspace`` contains a generated check file (by path or by content
        digest). Call this before launching the worker; launch only on success.
        """
        if not boundary_ids:
            raise BoundaryOrderError("an actor must be bound to at least one boundary")
        seals: list[BaseEvent] = []
        manifests: list[dict] = []
        package_by_boundary: dict[str, str | None] = {}
        committed: dict[str, bool] = {}
        for boundary_id in boundary_ids:
            events = await self.events(boundary_id)
            frozen = _first(events, PACKAGE_FROZEN)
            failed = _first(events, CONSTRUCTION_FAILED)
            admitted = _first(events, ADMISSION_COMPLETED)
            if frozen is None and failed is None:
                raise BoundaryOrderError(
                    "actor cannot start before the boundary is sealed",
                    details={"boundary_id": boundary_id, "actor_id": actor_id},
                )
            if frozen is not None and admitted is None:
                raise BoundaryOrderError(
                    "actor cannot start before package admission is recorded",
                    details={"boundary_id": boundary_id, "actor_id": actor_id},
                )
            seals.extend(event for event in (frozen, failed, admitted) if event is not None)
            if frozen is not None:
                manifests.append(frozen.data["manifest"])
                package_by_boundary[boundary_id] = _ref(frozen)
                committed[boundary_id] = _committed(frozen)
            else:
                package_by_boundary[boundary_id] = None
        if workspace is not None:
            leaks = find_workspace_leaks(workspace, manifests)
            if leaks:
                raise BoundaryLeakError(
                    "worker workspace contains generated check files",
                    details={"actor_id": actor_id, "paths": list(leaks)},
                )
        started = [
            actor_started_event(
                boundary_id,
                actor_id=actor_id,
                package_sha256=package_by_boundary[boundary_id],
                runtime=runtime,
                committed=committed.get(boundary_id, False),
                assignment=assignment,
            )
            for boundary_id in boundary_ids
        ]
        latest_seal = max(_utc(event.timestamp) for event in seals)
        if any(_utc(event.timestamp) <= latest_seal for event in started):
            raise BoundaryOrderError(
                "actor start timestamp does not follow the boundary seal",
                details={"actor_id": actor_id},
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

        Refused unless both versions are sealed, they differ, the old version
        is not already superseded, and no actor was ever bound to the old
        version (a worker's verdict must cite the package it was bound to).
        """
        if superseded_by == boundary_id:
            raise BoundaryOrderError(
                "a boundary cannot supersede itself", details={"boundary_id": boundary_id}
            )
        old = await self.events(boundary_id)
        new = await self.events(superseded_by)
        old_seal = _first(old, PACKAGE_FROZEN) or _first(old, CONSTRUCTION_FAILED)
        new_seal = _first(new, PACKAGE_FROZEN) or _first(new, CONSTRUCTION_FAILED)
        if old_seal is None or new_seal is None:
            raise BoundaryOrderError(
                "both boundary versions must be sealed before one supersedes the other",
                details={"boundary_id": boundary_id, "superseded_by": superseded_by},
            )
        if _first(old, SUPERSEDED) is not None:
            raise BoundaryOrderError(
                "boundary version already superseded", details={"boundary_id": boundary_id}
            )
        if _first(old, ACTOR_STARTED) is not None:
            raise BoundaryOrderError(
                "a boundary version bound to a worker cannot be superseded",
                details={"boundary_id": boundary_id},
            )
        event = superseded_event(
            boundary_id,
            superseded_by=superseded_by,
            package_sha256=_ref(old_seal),
            successor_package_sha256=_ref(new_seal),
            reason=reason,
            committed=_committed(old_seal),
            successor_committed=_committed(new_seal),
        )
        await self._store.append(event)
        return event

    async def record_bindings(
        self, boundary_id: str, *, package_sha256: str, payload: dict[str, Any]
    ) -> BaseEvent:
        """Record every check's tier and binding once the worker has stopped.

        ``package_sha256`` is the package reference (``CheckPackage.reference``).

        Refused unless the frozen, admitted package is cited and an actor
        (the worker) was recorded as started on this boundary: a late binding
        exists only after the oracle hash and the dispatch. A ``final``
        record is single and must precede the candidate verification it
        governs; ``repair`` records (per repair attempt) may repeat.
        """
        events = await self.events(boundary_id)
        frozen = _first(events, PACKAGE_FROZEN)
        if frozen is None or _ref(frozen) != package_sha256:
            raise BoundaryOrderError(
                "bindings must cite the boundary's frozen package",
                details={"boundary_id": boundary_id},
            )
        if _first(events, ADMISSION_COMPLETED) is None or _first(events, ACTOR_STARTED) is None:
            raise BoundaryOrderError(
                "bindings are recorded only after admission and the worker start",
                details={"boundary_id": boundary_id},
            )
        if payload.get("phase") == "final" and any(
            event.type == BINDING_RECORDED and event.data.get("phase") == "final"
            for event in events
        ):
            raise BoundaryOrderError(
                "final bindings already recorded", details={"boundary_id": boundary_id}
            )
        event = binding_recorded_event(
            boundary_id,
            package_sha256=package_sha256,
            payload=payload,
            committed=_committed(frozen),
        )
        await self._store.append(event)
        return event

    async def record_case_revealed(
        self,
        boundary_id: str,
        *,
        package_sha256: str,
        check_id: str,
        criterion_key: str,
        case_id: str,
        root_ac_index: int | None = None,
        retry_attempt: int | None = None,
    ) -> BaseEvent:
        """Record that one held-out case was revealed to the worker (once per case)."""
        events = await self.events(boundary_id)
        frozen = _first(events, PACKAGE_FROZEN)
        if frozen is None or _ref(frozen) != package_sha256:
            raise BoundaryOrderError(
                "a reveal must cite the boundary's frozen package",
                details={"boundary_id": boundary_id},
            )
        if _first(events, ACTOR_STARTED) is None:
            raise BoundaryOrderError(
                "a case is revealed only to a started worker",
                details={"boundary_id": boundary_id},
            )
        if any(
            event.type == CASE_REVEALED
            and event.data.get("check_id") == check_id
            and event.data.get("case_id") == case_id
            for event in events
        ):
            raise BoundaryOrderError(
                "case already revealed", details={"boundary_id": boundary_id, "case_id": case_id}
            )
        event = case_revealed_event(
            boundary_id,
            package_sha256=package_sha256,
            check_id=check_id,
            criterion_key=criterion_key,
            case_id=case_id,
            root_ac_index=root_ac_index,
            retry_attempt=retry_attempt,
            committed=_committed(frozen),
        )
        await self._store.append(event)
        return event

    async def record_candidate_verification(
        self, boundary_id: str, verification: CandidateVerification
    ) -> BaseEvent:
        """Persist a candidate run of the frozen package."""
        events = await self.events(boundary_id)
        frozen = _first(events, PACKAGE_FROZEN)
        if frozen is None or _first(events, ADMISSION_COMPLETED) is None:
            raise BoundaryOrderError(
                "candidate verification requires a frozen, admitted package",
                details={"boundary_id": boundary_id},
            )
        if _ref(frozen) != cited_reference(verification):
            raise BoundaryOrderError(
                "verification ran a package other than the frozen one",
                details={"boundary_id": boundary_id},
            )
        event = candidate_verified_event(boundary_id, verification)
        await self._store.append(event)
        return event

    async def record_selection(self, boundary_id: str, decision: SelectionDecision) -> BaseEvent:
        """Persist the selection reason and all digests."""
        events = await self.events(boundary_id)
        frozen = _first(events, PACKAGE_FROZEN)
        if frozen is None or _ref(frozen) != cited_reference(decision):
            raise BoundaryOrderError(
                "selection must cite the boundary's frozen package",
                details={"boundary_id": boundary_id},
            )
        event = selection_decided_event(boundary_id, decision)
        await self._store.append(event)
        return event

    async def record_acceptance_reconciled(
        self, boundary_id: str, *, package_sha256: str | None, reconciliation: dict[str, Any]
    ) -> BaseEvent:
        """Persist the per-criterion acceptance decision once, after verification.

        With a package it must cite a verification of the frozen package, or
        the final bindings when no check could run (every criterion
        unverified). Without one (``package_sha256`` is ``None``) the boundary
        must be sealed as ``construction_failed`` and a worker must have
        started on it.
        """
        events = await self.events(boundary_id)
        if package_sha256 is None:
            if _first(events, CONSTRUCTION_FAILED) is None or _first(events, ACTOR_STARTED) is None:
                raise BoundaryOrderError(
                    "a package-less decision needs a construction_failed seal and a worker",
                    details={"boundary_id": boundary_id},
                )
        else:
            cited = [
                event
                for event in events
                if event.type in {CANDIDATE_VERIFIED, BINDING_RECORDED}
                and _ref(event) == package_sha256
            ]
            if not any(event.type == CANDIDATE_VERIFIED for event in cited) and not any(
                event.data.get("phase") == "final" for event in cited
            ):
                raise BoundaryOrderError(
                    "acceptance must cite a verification of the frozen package",
                    details={"boundary_id": boundary_id},
                )
        if _first(events, ACCEPTANCE_RECONCILED) is not None:
            raise BoundaryOrderError(
                "acceptance already reconciled", details={"boundary_id": boundary_id}
            )
        event = acceptance_reconciled_event(
            boundary_id,
            package_sha256=package_sha256,
            reconciliation=reconciliation,
            committed=_committed(_first(events, PACKAGE_FROZEN)),
        )
        await self._store.append(event)
        return event

    async def record_reference_checked(
        self, boundary_id: str, *, package_sha256: str, payload: dict[str, Any]
    ) -> BaseEvent:
        """Record what the reference check excluded, after the seal and before admission."""
        events = await self.events(boundary_id)
        frozen = _first(events, PACKAGE_FROZEN)
        if frozen is None or _ref(frozen) != package_sha256:
            raise BoundaryOrderError(
                "a reference check must cite the boundary's frozen package",
                details={"boundary_id": boundary_id},
            )
        if _first(events, ADMISSION_COMPLETED) is not None:
            raise BoundaryOrderError(
                "the reference check precedes admission", details={"boundary_id": boundary_id}
            )
        event = reference_checked_event(
            boundary_id,
            package_sha256=package_sha256,
            committed=_committed(frozen),
            payload=payload,
        )
        await self._store.append(event)
        return event

    async def record_acceptance_resumed(
        self, boundary_id: str, *, package_sha256: str, payload: dict[str, Any]
    ) -> BaseEvent:
        """Record a resumed run's recomputed package decision (one per resume).

        Refused unless the frozen, admitted package is cited and a worker was
        started on this boundary. The recomputation may run a visible-only
        package, which has another digest; it therefore cites the frozen
        reference here instead of recording a candidate verification.
        """
        events = await self.events(boundary_id)
        frozen = _first(events, PACKAGE_FROZEN)
        if frozen is None or _ref(frozen) != package_sha256:
            raise BoundaryOrderError(
                "a resumed decision must cite the boundary's frozen package",
                details={"boundary_id": boundary_id},
            )
        if _first(events, ADMISSION_COMPLETED) is None or _first(events, ACTOR_STARTED) is None:
            raise BoundaryOrderError(
                "a resumed decision needs an admitted package and a started worker",
                details={"boundary_id": boundary_id},
            )
        event = acceptance_resumed_event(
            boundary_id,
            package_sha256=package_sha256,
            committed=_committed(frozen),
            payload=payload,
        )
        await self._store.append(event)
        return event

    async def record_legacy_fallback(self, boundary_id: str, *, reason: str) -> BaseEvent:
        """Record, once, that the legacy verifier decided a run whose worker started."""
        events = await self.events(boundary_id)
        if _first(events, ACTOR_STARTED) is None:
            raise BoundaryOrderError(
                "a fallback is recorded only for a started worker",
                details={"boundary_id": boundary_id},
            )
        if _first(events, LEGACY_FALLBACK) is not None:
            raise BoundaryOrderError(
                "legacy fallback already recorded", details={"boundary_id": boundary_id}
            )
        event = legacy_fallback_event(boundary_id, reason=reason)
        await self._store.append(event)
        return event


def verify_boundary_order(events: Sequence[BaseEvent]) -> tuple[str, ...]:
    """Return ordering violations in one boundary's replayed events.

    An empty result means: one seal, the seal precedes admission, admission
    precedes every actor start, and every receipt cites the frozen digest.
    """
    violations: list[str] = []
    seals = [e for e in events if e.type in {PACKAGE_FROZEN, CONSTRUCTION_FAILED}]
    if len(seals) != 1:
        violations.append(f"expected exactly one seal, found {len(seals)}")
    seal = seals[0] if seals else None
    frozen_sha = _ref(seal) if seal and seal.type == PACKAGE_FROZEN else None
    admissions = [e for e in events if e.type == ADMISSION_COMPLETED]
    if frozen_sha is not None and len(admissions) != 1:
        violations.append(f"expected exactly one admission, found {len(admissions)}")
    if frozen_sha is None and admissions:
        violations.append("admission recorded without a frozen package")
    position = {id(event): index for index, event in enumerate(events)}
    for event in events:
        if event.type in {ADMISSION_COMPLETED, CANDIDATE_VERIFIED, SELECTION_DECIDED}:
            if frozen_sha is None or _ref(event) != frozen_sha:
                violations.append(f"{event.type} does not cite the frozen package")
        if event.type == ACTOR_STARTED:
            if seal is None or position[id(seal)] > position[id(event)]:
                violations.append("actor started before the seal")
            elif _utc(seal.timestamp) >= _utc(event.timestamp):
                violations.append("actor start timestamp does not follow the seal")
            if frozen_sha is not None and (
                not admissions or position[id(admissions[0])] > position[id(event)]
            ):
                violations.append("actor started before admission")
        if event.type in {BINDING_RECORDED, ACCEPTANCE_RESUMED}:
            if frozen_sha is None or _ref(event) != frozen_sha:
                violations.append(f"{event.type} does not cite the frozen package")
            started = [e for e in events if e.type == ACTOR_STARTED]
            if not started or position[id(started[0])] > position[id(event)]:
                violations.append("bindings recorded before the worker started")
    if _committed(seal) and any(
        key.endswith("package_sha256") and value is not None
        for event in events
        for key, value in event.data.items()
    ):
        # Any unkeyed digest, including a successor's. A successor
        # with no package cites none (``successor_package_sha256: null``).
        violations.append("a committed boundary records an unkeyed package digest")
    finals = [e for e in events if e.type == BINDING_RECORDED and e.data.get("phase") == "final"]
    verified = [e for e in events if e.type == CANDIDATE_VERIFIED]
    if finals and verified and position[id(finals[0])] > position[id(verified[-1])]:
        violations.append("final bindings recorded after the candidate verification")
    return tuple(violations)
