"""One run's check package boundary, shared by ``ooo run`` and ``ouroboros_execute_seed``.

``CheckPackageRun`` resolves the switch (``boundary/rollout.py``, on by
default), prepares the package before the worker starts, installs
``CheckPackageAuthority`` on the runner so the package decides the criteria it
covers before the terminal status is persisted, and afterwards renders the
outcome and a closed-value summary of it (``outcome_meta``) that the MCP
``execute_seed`` result carries. Nothing here is sent as telemetry.

With the switch ``off`` nothing here calls a model, writes an event, or
touches the runner: the run is the legacy run.

On resume the switch is read from the journal, not resolved again: when the
original run bound its worker to an admitted package, the resumed run
recomputes the package decision (``boundary/resume.py``) instead of letting
the legacy verifier decide the covered criteria.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ouroboros.boundary.acceptance import VerificationCoverage
from ouroboros.boundary.authority import (
    AuthorityOutcome,
    CheckPackageAuthority,
    reveal_commitment_salts,
)
from ouroboros.boundary.ledger import BoundaryOrderError
from ouroboros.boundary.reference_check import (
    ORACLE_INCONSISTENT,
    REFERENCE_CONTRADICTS_SEED_EXAMPLE,
    REFERENCE_UNAVAILABLE,
)
from ouroboros.boundary.resume import ResumedCheckPackageAuthority, load_resumed_boundary
from ouroboros.boundary.rollout import Arm, AssignmentSource, CheckPackageAssignment
from ouroboros.boundary.run_wiring import (
    BoundaryRunState,
    CheckPackageSettings,
    prepare_check_package,
    render_preparation,
    render_verdict,
    resolve_check_package_settings,
    unavailable_line,
)

if TYPE_CHECKING:
    from ouroboros.core.seed import Seed
    from ouroboros.persistence.event_store import EventStore

log = structlog.get_logger(__name__)

RECOVERY_EXHAUSTED_EVENT_TYPE = "execution.ac.recovery_exhausted"
_EVIDENCE_LIMIT = 5000
# orchestrator/failure_taxonomy.FailureClass values, lower-cased.
_FAILURE_CLASS_VALUES = frozenset(
    {
        "evidence_missing",
        "evidence_form_mismatch",
        "fabrication_suspected",
        "scope_creep",
        "stall",
        "blocked",
        "transcript_missing_infrastructure",
    }
)
_FALLBACK_ASSIGNMENT = CheckPackageAssignment(Arm.OFF, AssignmentSource.FALLBACK)


def _switch_text(source: AssignmentSource) -> str:
    if source is AssignmentSource.DEFAULT:
        return (
            "on by default; opt out with --no-check-package, OUROBOROS_CHECK_PACKAGE=off, "
            "or boundary.check_package: off"
        )
    return f"on, {source.value}"


def _count_bucket(count: int) -> str:
    return "3+" if count >= 3 else str(max(0, count))


TIER_SUMMARY_TIERS = ("A", "A_prime", "U")


def tier_summary_value(counts: dict[str, int]) -> str:
    """``check_tier_summary``: bucketed counts of tiers A, A' and U (closed set)."""
    return ",".join(f"{tier}:{_count_bucket(counts.get(tier, 0))}" for tier in TIER_SUMMARY_TIERS)


def reference_check_meta(state: BoundaryRunState | None) -> dict[str, str]:
    """Bucketed reference-check counts, when this process ran the check (else empty).

    A resumed run in another process reports none: the counts are in the
    journal (``boundary.oracle.reference_checked``), not in its state.
    """
    report = state.reference_check if state is not None else None
    if report is None:
        return {}
    counts = report.counts()
    return {
        "oracle_inconsistent_count": _count_bucket(counts[ORACLE_INCONSISTENT]),
        "reference_contradiction_count": _count_bucket(counts[REFERENCE_CONTRADICTS_SEED_EXAMPLE]),
        "reference_unavailable_count": _count_bucket(counts[REFERENCE_UNAVAILABLE]),
    }


def coverage_meta(state: BoundaryRunState | None) -> dict[str, str]:
    """Bucketed pre-dispatch coverage counts, when this process built the package (else empty).

    ``excluded_check_count``: checks per-check admission excluded, over every
    version; ``replacement_call_count``: replacement constructor calls (0 or 1).
    """
    if state is None:
        return {}
    return {
        "excluded_check_count": _count_bucket(len(state.exclusions)),
        "replacement_call_count": _count_bucket(state.replacement_calls),
    }


def legacy_failure_class_from_annotations(legacy: dict[int, Any]) -> tuple[str, str]:
    """``(class, count bucket)`` from the legacy verdicts the authority annotated."""
    rejected = sorted(
        index
        for index, item in legacy.items()
        if item.outcome == "failed" and item.terminal_status == "failed"
    )
    if not rejected:
        return "other", "0"
    raw = legacy[rejected[0]].failure_class
    first = raw.lower() if isinstance(raw, str) else ""
    return (first if first in _FAILURE_CLASS_VALUES else "other"), _count_bucket(len(rejected))


def legacy_failure_class_from_events(
    events: Iterable[Any], *, session_id: str | None
) -> tuple[str, str]:
    """``(class, count bucket)`` for the criteria the legacy verifier rejected.

    Reads ``execution.ac.recovery_exhausted``, which the executor writes once
    per root criterion it finally rejected, with the worker failure class of
    the last attempt. The class of the lowest criterion index wins; a class
    outside the failure taxonomy is ``other``. ``("other", "0")`` means the
    legacy verdict was a rejection with no per-criterion record.
    """
    by_index: dict[int, str] = {}
    for event in events:
        data = getattr(event, "data", None) or {}
        if session_id is not None and data.get("session_id") not in (None, session_id):
            continue
        index = data.get("root_ac_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            continue
        raw = data.get("last_failure_class")
        by_index[index] = raw.lower() if isinstance(raw, str) else ""
    if not by_index:
        return "other", "0"
    first = by_index[min(by_index)]
    return (first if first in _FAILURE_CLASS_VALUES else "other"), _count_bucket(len(by_index))


async def legacy_failure_dimensions(
    event_store: EventStore,
    *,
    execution_id: str | None,
    session_id: str | None,
    legacy_verdict: str,
) -> dict[str, str]:
    """``legacy_failure_class`` and ``legacy_failure_class_count`` for a run."""
    if legacy_verdict == "accept":
        return {"legacy_failure_class": "accepted", "legacy_failure_class_count": "0"}
    if legacy_verdict != "reject" or not execution_id:
        return {"legacy_failure_class": "none", "legacy_failure_class_count": "0"}
    try:
        events = await event_store.query_events(
            aggregate_id=execution_id,
            event_type=RECOVERY_EXHAUSTED_EVENT_TYPE,
            limit=_EVIDENCE_LIMIT,
        )
    except Exception:  # noqa: BLE001 - enrichment must not drop the outcome
        events = []
    # query_events is newest first; replay oldest first so the latest record
    # for a criterion wins.
    failure_class, count = legacy_failure_class_from_events(
        list(reversed(events)), session_id=session_id
    )
    return {"legacy_failure_class": failure_class, "legacy_failure_class_count": count}


def _no_package_coverage_line(state: BoundaryRunState | None) -> str:
    total = len(state.criterion_keys) if state is not None else 0
    return (
        "WARNING: insufficient verification: the check package decided 0 of "
        f"{total} criteria (verification_coverage=low)."
    )


ConstructorFactory = Callable[..., Any]


@dataclass
class CheckPackageRun:
    """The check package boundary of one run, from the switch to the outcome summary."""

    settings: CheckPackageSettings
    state: BoundaryRunState | None = None
    authority: CheckPackageAuthority | None = None
    attempted: bool = False
    skipped_reason: str | None = None
    preparation_error: str | None = None
    resumed: ResumedCheckPackageAuthority | None = None
    _binding: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def resolve(cls, cli_value: bool | None = None) -> CheckPackageRun:
        """Resolve the switch and budgets; never raises (an error means ``off``)."""
        try:
            return cls(resolve_check_package_settings(cli_value))
        except Exception:  # noqa: BLE001 - resolving the default must not fail a run
            log.warning("boundary.run_control.resolve_failed")
            return cls(CheckPackageSettings(enabled=False, assignment=_FALLBACK_ASSIGNMENT))

    @property
    def assignment(self) -> CheckPackageAssignment:
        if self.resumed is not None:
            # The original run's switch, from the journal (R4-A2): a bound
            # package means on. Journals written before the source was
            # recorded report ``user_forced_on``.
            try:
                source = AssignmentSource(self.resumed.boundary.assignment or "")
            except ValueError:
                source = AssignmentSource.USER_FORCED_ON
            return CheckPackageAssignment(Arm.ON, source)
        return self.settings.assignment or (
            CheckPackageAssignment(Arm.ON, AssignmentSource.USER_FORCED_ON)
            if self.settings.enabled
            else _FALLBACK_ASSIGNMENT
        )

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    async def prepare(
        self,
        runner: Any,
        seed: Seed,
        *,
        event_store: EventStore,
        execution_id: str | None,
        worker_dir: Path,
        runtime_backend: str,
        model: str | None,
        resume: bool,
        constructor_factory: ConstructorFactory | None = None,
    ) -> list[str]:
        """Build, freeze, and admit the package; install the authority on ``runner``.

        Returns lines for the person running the command. Raises
        ``BoundaryOrderError`` (including ``BoundaryLeakError``) when the
        ledger refuses the worker start; the caller must not dispatch then.
        Any other preparation error is recorded and the run continues under
        the legacy verifier.
        """
        if resume and execution_id:
            return await self._prepare_resume(runner, event_store, execution_id, worker_dir)
        if not self.enabled:
            return []
        if resume or not execution_id:
            self.skipped_reason = "resume"
            return ["Check package is not applied on resume; the session keeps its boundary."]
        self.attempted = True
        if constructor_factory is None:
            from ouroboros.boundary.constructor import CheckConstructor

            constructor_factory = CheckConstructor
        lines = [
            "Check package: constructing checks from the acceptance criteria "
            f"(read-only; {_switch_text(self.assignment.source)})..."
        ]
        try:
            constructor = constructor_factory(
                runtime_backend=runtime_backend,
                model=model,
                timeout_seconds=self.settings.constructor_timeout_seconds,
            )
            state = await prepare_check_package(
                seed,
                event_store=event_store,
                constructor=constructor,
                execution_id=execution_id,
                base_checkout=worker_dir,
                worker_workspace=worker_dir,
                runtime_label=runtime_backend,
                settings=self.settings,
            )
        except BoundaryOrderError:
            raise
        except Exception as exc:  # noqa: BLE001 - a preparation fault must not fail the run
            self.preparation_error = type(exc).__name__
            log.warning(
                "boundary.run_control.prepare_failed",
                execution_id=execution_id,
                error_type=self.preparation_error,
            )
            return [
                *lines,
                f"Check package could not be prepared ({self.preparation_error}); "
                "the existing verifier decides this run.",
            ]
        self.state = state
        if not state.admitted:
            # No admitted package: this run is the legacy run, exactly as with
            # the check package off (user decision, 2026-09-27). Nothing is
            # installed on the runner; the outcome summary keeps the failure
            # status and reports reconciliation=fallback_to_legacy.
            return [*lines, *render_preparation(state)]
        self.authority = CheckPackageAuthority(
            state, self.settings, event_store=event_store, candidate_checkout=worker_dir
        )
        runner.acceptance_authority = self.authority
        return [*lines, *render_preparation(state)]

    async def _prepare_resume(
        self, runner: Any, event_store: EventStore, execution_id: str, worker_dir: Path
    ) -> list[str]:
        """Install the resumed authority when the original run was bound to a package.

        Raises ``BoundaryOrderError`` when the journal cannot be read: whether
        the package decides the covered criteria is then unknown, and letting
        the legacy verifier decide them could skip the package (R4-A3). The
        caller must not resume then.
        """
        self.skipped_reason = "resume"
        try:
            boundary = await load_resumed_boundary(event_store, execution_id)
        except Exception as exc:  # noqa: BLE001 - any read failure refuses the resume
            log.warning("boundary.run_control.resume_unreadable", error_type=type(exc).__name__)
            raise BoundaryOrderError(
                "check package state could not be read on resume "
                f"({type(exc).__name__}); retry the resume",
                details={"execution_id": execution_id},
            ) from exc
        if boundary is None:
            if not self.enabled:
                return []
            return ["Check package is not applied on resume: no admitted package was bound."]
        self.resumed = ResumedCheckPackageAuthority(
            boundary, self.settings, event_store=event_store, candidate_checkout=worker_dir
        )
        runner.acceptance_authority = self.resumed
        return [
            "Check package: resumed run; the package decision is recomputed on the workspace "
            f"({'held-out cases in memory' if boundary.source == 'memory' else 'visible cases only'})."
        ]

    # ------------------------------------------------------------------
    # Compact entry points for the MCP ``execute_seed`` handler

    def bind(
        self, runner: Any, event_store: EventStore, worker_dir: Path, runtime_backend: str
    ) -> CheckPackageRun:
        """Remember the handler's runner and workspace for ``prepare_bound``."""
        from ouroboros.config.loader import resolve_execution_model

        self._binding = {
            "runner": runner,
            "event_store": event_store,
            "worker_dir": worker_dir,
            "runtime_backend": runtime_backend,
            "model": resolve_execution_model(runtime_backend),
        }
        return self

    async def prepare_bound(self, seed: Seed, execution_id: str) -> None:
        """``prepare`` for a fresh run with the bound context; lines go to the log."""
        binding = self._binding
        lines = await self.prepare(
            binding["runner"],
            seed,
            event_store=binding["event_store"],
            execution_id=execution_id,
            worker_dir=binding["worker_dir"],
            runtime_backend=binding["runtime_backend"],
            model=binding["model"],
            resume=False,
        )
        for line in lines:
            log.info("boundary.run_control.prepared", execution_id=execution_id, line=line)

    async def prepare_resumed(self, execution_id: str) -> None:
        """``prepare`` for a resumed run with the bound context; lines go to the log.

        Raises ``BoundaryOrderError`` when the journal cannot be read.
        """
        binding = self._binding
        lines = await self._prepare_resume(
            binding["runner"], binding["event_store"], execution_id, binding["worker_dir"]
        )
        for line in lines:
            log.info("boundary.run_control.resumed", execution_id=execution_id, line=line)

    async def meta_for(self, tracker: Any, session_status: Any) -> dict[str, str]:
        """``outcome_meta`` for a finished MCP run; empty while it is still running."""
        status = getattr(session_status, "value", None)
        if not isinstance(status, str):
            return {}
        try:
            return await self.outcome_meta(
                self._binding["event_store"],
                execution_id=tracker.execution_id,
                session_id=tracker.session_id,
                terminal_status=status,
            )
        except Exception:  # noqa: BLE001 - enrichment must not fail the tool result
            return {}

    # ------------------------------------------------------------------
    # Outcome

    @property
    def status(self) -> str:
        """``check_package_status`` of the outcome summary (``outcome_meta``)."""
        if self.resumed is not None:
            # Only a run whose worker was bound to an admitted package resumes
            # with a package decision.
            return "admitted"
        if not self.enabled or not self.attempted:
            return "not_run"
        if self.preparation_error is not None or self.state is None:
            return "construction_failed"
        if self.state.admitted:
            return "admitted"
        reason = self.state.failure_reason or ""
        return "rejected" if reason.startswith("package_") else "construction_failed"

    def render_outcome(self) -> list[str]:
        """Lines describing what the package decided (empty when it did not run)."""
        if self.resumed is not None:
            return self.resumed.render()
        if self.authority is None:
            if self.state is not None and not self.state.admitted:
                return [unavailable_line(self.state), _no_package_coverage_line(self.state)]
            return []
        outcome = self.authority.outcome
        if outcome is None:
            return [
                "Check package was not consulted: this execution path does not support it; "
                "the existing verifier decided the run."
            ]
        if outcome.fallback_reason is not None:
            return [
                f"Check package could not decide this run ({outcome.fallback_reason}); "
                "legacy verification decided this run.",
                _no_package_coverage_line(self.state),
            ]
        if outcome.error is not None:
            return [
                f"Check package verification failed ({outcome.error}); "
                "the existing verifier decided the run."
            ]
        lines = [] if outcome.verdict is None else render_verdict(outcome.verdict)
        repairs = [entry for entry in self.authority.gate.log if entry["status"] == "fail"]
        if repairs:
            lines.append(
                f"Repairs driven by check package counterexamples: {len(repairs)} "
                "(the legacy verifier drives retries only for legacy-decided criteria)."
            )
        if self.authority.gate.legacy_failures:
            lines.append(
                "Attempts the legacy verifier rejected on legacy-decided criteria: "
                f"{self.authority.gate.legacy_failures}."
            )
        reconciliation = outcome.reconciliation
        if reconciliation is not None:
            from ouroboros.boundary.acceptance import render_reconciliation

            lines.extend(render_reconciliation(reconciliation))
            if reconciliation.run_accepted and not outcome.legacy_run_accepted:
                lines.append(
                    "The check package accepted criteria the legacy verifier rejected; "
                    "the legacy verdict is advisory only."
                )
            elif outcome.legacy_run_accepted and not reconciliation.run_accepted:
                lines.append("The finished workspace fails the frozen check package.")
        return lines

    def _coverage(self) -> str | None:
        """``verification_coverage``: ``low`` when the package decided nothing (switch on)."""
        if self.status == "not_run" and self.resumed is None:
            return None
        outcome = self._outcome()
        reconciliation = outcome.reconciliation if outcome is not None else None
        if reconciliation is None or not reconciliation.legacy_rule:
            return VerificationCoverage.LOW.value
        return reconciliation.coverage.value

    def _outcome(self) -> AuthorityOutcome | None:
        """The decision of this run's authority, live or resumed."""
        if self.resumed is not None:
            return self.resumed.outcome
        return self.authority.outcome if self.authority is not None else None

    def _legacy_verdict(self, terminal_status: str | None, *, verdict_available: bool) -> str:
        outcome = self._outcome()
        if outcome is not None:
            return "accept" if outcome.legacy_run_accepted else "reject"
        if not verdict_available:
            return "none"
        return {"completed": "accept", "failed": "reject"}.get(terminal_status or "", "none")

    def _package_verdict(self) -> str:
        outcome = self._outcome()
        if outcome is None:
            return "none"
        if outcome.error is not None:
            return "indeterminate"
        if outcome.verdict is None or outcome.verdict.package_sha256 is None:
            return "none"
        return outcome.verdict.verdict

    def _reconciliation(self, legacy_verdict: str) -> str:
        if self.status == "not_run":
            return "none"
        outcome = self._outcome()
        if outcome is None or outcome.reconciliation is None:
            return "fallback_to_legacy"
        reconciliation = outcome.reconciliation
        rejected = [d for d in reconciliation.decisions if not d.accepted]
        if rejected and all(d.legacy_decided for d in rejected):
            # Every rejection came from the legacy verifier on a criterion the
            # package could not verify (user decision, 2026-09-27).
            return "legacy_decided_unverified"
        if not outcome.package_decided:
            return "fallback_to_legacy"
        accepted = outcome.reconciliation.run_accepted
        legacy_accepted = legacy_verdict == "accept"
        if accepted == legacy_accepted:
            return "agree"
        if accepted:
            return "package_accepted_over_legacy_reject"
        return "package_rejected_over_legacy_accept"

    def finish(self, terminal_status: str | None) -> None:
        """Close the run's check package record once the final verdict exists.

        On a terminal status (``completed``, ``failed``, ``cancelled``; never
        ``paused``) after the authority decided, reveal the commitment salts
        of every frozen version (the authority already did for its own).
        """
        if terminal_status in ("completed", "failed", "cancelled") and (
            self.authority is None or self.authority.outcome is not None
        ):
            reveal_commitment_salts(self.state)

    async def outcome_meta(
        self,
        event_store: EventStore,
        *,
        execution_id: str | None,
        session_id: str | None,
        terminal_status: str | None,
        verdict_available: bool = True,
    ) -> dict[str, str]:
        """The closed-value summary of this run's check package outcome (local only)."""
        self.finish(terminal_status)
        legacy_verdict = self._legacy_verdict(terminal_status, verdict_available=verdict_available)
        meta = {
            "check_package_arm": self.assignment.arm.value,
            "check_package_assignment": self.assignment.source.value,
            "check_package_status": self.status,
            "package_verdict": self._package_verdict(),
            "legacy_verdict": legacy_verdict,
            "reconciliation": self._reconciliation(legacy_verdict),
        }
        outcome = self._outcome()
        if outcome is not None and outcome.legacy and legacy_verdict == "reject":
            failure_class, count = legacy_failure_class_from_annotations(outcome.legacy)
            meta.update(
                {"legacy_failure_class": failure_class, "legacy_failure_class_count": count}
            )
        else:
            meta.update(
                await legacy_failure_dimensions(
                    event_store,
                    execution_id=execution_id,
                    session_id=session_id,
                    legacy_verdict=legacy_verdict,
                )
            )
        reconciliation = outcome.reconciliation if outcome is not None else None
        if reconciliation is not None and outcome is not None and outcome.package_decided:
            # Only when the package decided the run.
            meta["unverified_count"] = _count_bucket(len(reconciliation.unverified))
            meta["check_tier_summary"] = tier_summary_value(reconciliation.tiers)
        meta.update(reference_check_meta(self.state))
        meta.update(coverage_meta(self.state))
        coverage = self._coverage()
        if coverage is not None:
            meta["verification_coverage"] = coverage
        if self.authority is not None and self.authority.installed:
            # Criteria asked once for an entry_points declaration.
            meta["binding_request_count"] = _count_bucket(len(self.authority.binding_requested))
        return meta


__all__ = [
    "TIER_SUMMARY_TIERS",
    "CheckPackageRun",
    "legacy_failure_class_from_annotations",
    "tier_summary_value",
    "legacy_failure_class_from_events",
    "legacy_failure_dimensions",
]
