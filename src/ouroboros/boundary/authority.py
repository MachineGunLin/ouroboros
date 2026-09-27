"""Acceptance authority inside the runner: the frozen check package decides.

``OrchestratorRunner`` installs the authority on its parallel executor
(``install``) and calls it once on the executor's result, after the worker
has stopped and before the terminal acceptance plan and the session status
are persisted (``OrchestratorRunner.acceptance_authority``).

While the worker runs (``CheckPackageGate``, installed as the executor's
``check_package_gate``):

- the legacy per-criterion verifier (typed evidence plus the transcript
  verifier) no longer rejects an attempt of a criterion an admitted check
  covers; its verdict stays on the result as an annotation (advisory reason
  and failure class for telemetry), so it triggers no retry. For a criterion
  no admitted check covers (uncovered, non-behavioral, or every check
  excluded at admission) the legacy verifier decides: its rejection fails
  the attempt and drives the retry, as with the check package off;
- after each attempt of a root criterion the gate runs that criterion's
  checks on the workspace, through the default binding or the entry point
  the worker declared in its evidence; a fail marks the attempt failed with
  the counterexample (``ACExecutionResult.check_package_repair``), which the
  executor's retry loop carries into the next attempt. A failure through a
  worker-declared binding names that binding. Held-out inputs are withheld.

After the worker stops (``CheckPackageAuthority.__call__``):

1. ``verify_check_package`` records the final bindings, verifies the
   finished workspace, and records verification and selection;
2. ``reconcile_acceptance`` decides every criterion (pass accepts, fail and
   indeterminate reject, a criterion the worker never attempted is not
   accepted; an unverified, uncovered or non-behavioral criterion is decided
   by the legacy verifier, and stays unverified and accepted only when the
   legacy verifier has no evidence either); the legacy verdict is kept per
   criterion; ``boundary.acceptance.reconciled`` records it;
3. the executor result is returned with each root result's ``success`` and
   ``outcome`` set to the decision and the counts recomputed, so the durable
   status, the panel, the exit code, and ``workflow_outcome`` carry it.

Neither part ever raises into the run: on an error the gate returns the
attempt unchanged. When the authority cannot decide (it raised, or no package
was admitted) while the gate was installed, the run falls back to the legacy
verifier exactly as with no package: the legacy rejections the gate made
advisory are restored (``ACExecutionResult.legacy_rejection``), attempts the
gate alone failed are accepted again, the reason is recorded
(``boundary.acceptance.legacy_fallback``) and printed. Without the gate the
executor result is already the legacy result and is returned unchanged.

Reveals (held-out reveal-and-retire): at most one per criterion per run, and
never on an attempt after which no repair attempt follows
(``retry_attempt >= ac_retry_attempts``), so a case is retired only when the
worker is actually shown it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ouroboros.boundary.acceptance import (
    AcceptanceReconciliation,
    ExistingOutcome,
    PackageCriterionStatus,
    criterion_verdicts,
    reconcile_acceptance,
)
from ouroboros.boundary.binding import declared_entry_points, entry_points_request
from ouroboros.boundary.binding_flow import (
    assign_tiers,
    bindings_payload,
    retire_revealed,
    verify_with_bindings,
)
from ouroboros.boundary.check_env import resolve_check_interpreter, scrubbed_check_environment
from ouroboros.boundary.ledger import BoundaryLedger
from ouroboros.boundary.package import seed_criterion_keys
from ouroboros.boundary.per_check import criteria_without_admitted_check, excluded_check_ids
from ouroboros.boundary.run_wiring import (
    BoundaryRunState,
    BoundaryVerdict,
    CheckPackageSettings,
    _base_manifest,
    _counterexamples,
    forget_live_state,
    persist_commitment_salts,
    plan_repair,
    verify_check_package,
)

if TYPE_CHECKING:
    from ouroboros.core.seed import Seed
    from ouroboros.persistence.event_store import EventStore

log = structlog.get_logger(__name__)

_ACCEPTED_OUTCOMES = frozenset({"succeeded", "satisfied_externally"})
PACKAGE_REJECTION_ERROR = "check_package: the finished workspace fails the frozen check package"
PACKAGE_INDETERMINATE_ERROR = (
    "check_package: the frozen check package could not decide this criterion"
)
PACKAGE_FAILURE_CLASS_PREFIX = "CHECK_PACKAGE_FAIL"
LEGACY_REJECTION_ERROR = "legacy verifier rejected this criterion"
LEGACY_DECIDED_PREFIX = "legacy-decided"
LEGACY_DECIDED_FAILURE_CLASS_PREFIX = "LEGACY_DECIDED"
NO_BINDING = "no_binding"
NO_BINDING_AFTER_REQUEST = "no_binding_after_request"
NO_BINDING_BUDGET_EXHAUSTED = "no_binding_budget_exhausted"
BINDING_REJECTED_PREFIX = "binding_invalid:"


@dataclass(frozen=True, slots=True)
class AuthorityOutcome:
    """What the authority decided for one run."""

    legacy_run_accepted: bool
    verdict: BoundaryVerdict | None = None
    reconciliation: AcceptanceReconciliation | None = None
    error: str | None = None
    legacy: dict[int, ExistingOutcome] = field(default_factory=dict)
    fallback_reason: str | None = None
    """Set when the legacy verifier decided the run instead of the package."""

    @property
    def package_decided(self) -> bool:
        """Whether the package governed at least one criterion."""
        if self.reconciliation is None:
            return False
        return any(
            decision.governed_by.value == "check_package"
            for decision in self.reconciliation.decisions
        )


def _declared_from(result: Any) -> list[Any]:
    """The first ``entry_points`` found on a result or its sub-results."""
    entries = declared_entry_points(getattr(result, "typed_evidence", None))
    if entries:
        return entries
    for sub in getattr(result, "sub_results", ()) or ():
        entries = _declared_from(sub)
        if entries:
            return entries
    return []


def legacy_verdict_in_tree(result: Any) -> tuple[bool, str | None, str | None]:
    """``(rejected, failure_class, rejection_text)`` over a result and its sub-results.

    A decomposed root is assembled from its sub-ACs' successes; with the gate
    installed a sub-AC's legacy rejection is advisory and stays on that
    sub-result, not on the root. The legacy verdict of the root is therefore
    the whole tree's: rejected when the legacy verifier rejected the root or
    any sub-AC.

    Only the rejection the executor made (``legacy_rejection``) is a legacy
    rejection. A verifier verdict that did not pass without one is not: the
    executor keeps such a result successful when the transcript was
    unavailable (``TRANSCRIPT_MISSING_INFRASTRUCTURE``), when the environment
    was unverifiable, or when a passing ``verify_command`` replaced the
    evidence, and arm ``off`` accepts it. That verdict carries no rejection
    (R3-A2); it only supplies the failure class of a real rejection.
    """
    text = getattr(result, "legacy_rejection", None) or None
    if text:
        verdict = getattr(result, "atomic_verifier_verdict", None)
        failure_class = (
            getattr(verdict, "failure_class", None)
            if verdict is not None and not bool(getattr(verdict, "passed", True))
            else None
        )
        return True, failure_class, text
    for sub in getattr(result, "sub_results", ()) or ():
        rejected, failure_class, sub_text = legacy_verdict_in_tree(sub)
        if rejected:
            return True, failure_class, sub_text
    return False, None, None


def _legacy_evidence(result: Any) -> bool:
    """Whether the legacy verifier accepted ``result`` on evidence.

    A passing verifier verdict, or a passing ``verify_command`` whose
    environment was verifiable; a decomposed root has evidence when every
    sub-result has. A success with no verdict, an unavailable transcript
    (``TRANSCRIPT_MISSING_INFRASTRUCTURE``) or an unverifiable environment
    is an acceptance without evidence.
    """
    gate = getattr(result, "verify_gate_outcome", None)
    if (
        gate is not None
        and bool(getattr(gate, "passed", False))
        and not bool(getattr(gate, "environment_unverifiable", False))
    ):
        return True
    verdict = getattr(result, "atomic_verifier_verdict", None)
    if verdict is not None and bool(getattr(verdict, "passed", False)):
        return True
    subs = tuple(getattr(result, "sub_results", ()) or ())
    return bool(subs) and all(_legacy_evidence(sub) for sub in subs)


def existing_outcomes_from_results(
    parallel_result: Any, *, gated: bool = False
) -> dict[int, ExistingOutcome]:
    """Per root criterion: whether the worker attempted it, and the legacy verdict.

    ``outcome`` carries the legacy verifier's verdict (``failed`` when it
    rejected the attempt, whatever the executor did with that rejection) and
    ``failure_class`` its class. With the gate installed (``gated``), an
    attempt counts as made when the executor result succeeded or failed only
    because the package gate failed it; a runtime failure, a blocked or
    invalid criterion, or a failed ``verify_command`` is not an attempt the
    package may accept. Without the gate, a failed result is a legacy
    rejection of an attempt.
    """
    outcomes: dict[int, ExistingOutcome] = {}
    for result in getattr(parallel_result, "results", ()) or ():
        index = getattr(result, "ac_index", None)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            continue
        base = getattr(getattr(result, "outcome", None), "value", None)
        if not isinstance(base, str):
            base = "succeeded" if getattr(result, "success", False) else "failed"
        # The executor keeps the rejection it made advisory (typed evidence or
        # transcript verifier) on the result, or on a sub-AC's result for a
        # decomposed root.
        legacy_rejected, failure_class, _text = legacy_verdict_in_tree(result)
        if gated:
            judged = bool(getattr(result, "success", False)) or bool(
                getattr(result, "check_package_failure_class", None)
            )
        else:
            judged = base in _ACCEPTED_OUTCOMES or base == "failed"
            legacy_rejected = legacy_rejected or base == "failed"
        if judged:
            outcome = (
                "failed"
                if legacy_rejected
                else (base if base in _ACCEPTED_OUTCOMES else "succeeded")
            )
            terminal = "failed" if outcome == "failed" else "completed"
        else:
            outcome, terminal = base, "not_attempted"
        outcomes[index] = ExistingOutcome(
            root_ac_index=index,
            outcome=outcome,
            disposition="accepted" if outcome in _ACCEPTED_OUTCOMES else outcome,
            terminal_status=terminal,
            failure_class=failure_class,
            no_evidence=outcome in _ACCEPTED_OUTCOMES and not _legacy_evidence(result),
        )
    return outcomes


def apply_reconciliation(parallel_result: Any, reconciliation: AcceptanceReconciliation) -> Any:
    """Return ``parallel_result`` with every root result set to its decision."""
    from ouroboros.orchestrator.parallel_executor_models import ACExecutionOutcome

    decisions = {decision.root_ac_index: decision for decision in reconciliation.decisions}
    results = []
    success_delta = failure_delta = external_delta = 0
    changed = False
    for result in parallel_result.results:
        decision = decisions.get(result.ac_index)
        previous = result.outcome
        if decision is None:
            results.append(result)
            continue
        if decision.accepted and previous is ACExecutionOutcome.FAILED:
            results.append(
                replace(result, success=True, outcome=ACExecutionOutcome.SUCCEEDED, error=None)
            )
            success_delta += 1
            failure_delta -= 1
            changed = True
        elif not decision.accepted and previous in (
            ACExecutionOutcome.SUCCEEDED,
            ACExecutionOutcome.SATISFIED_EXTERNALLY,
        ):
            if decision.legacy_decided:
                error = legacy_decided_error(result, decision.reason)
            elif decision.package_status is PackageCriterionStatus.FAIL:
                error = PACKAGE_REJECTION_ERROR
            else:
                error = f"{PACKAGE_INDETERMINATE_ERROR} ({decision.reason})"
            results.append(
                replace(result, success=False, outcome=ACExecutionOutcome.FAILED, error=error)
            )
            failure_delta += 1
            if previous is ACExecutionOutcome.SUCCEEDED:
                success_delta -= 1
            else:
                external_delta -= 1
            changed = True
        else:
            results.append(result)
    if not changed:
        return parallel_result
    return replace(
        parallel_result,
        results=tuple(results),
        success_count=parallel_result.success_count + success_delta,
        failure_count=parallel_result.failure_count + failure_delta,
        externally_satisfied_count=parallel_result.externally_satisfied_count + external_delta,
    )


def legacy_decided_error(result: Any, reason: str) -> str:
    """The error of a criterion the legacy verifier decided and rejected."""
    text = legacy_verdict_in_tree(result)[2] or LEGACY_REJECTION_ERROR
    return f"{LEGACY_DECIDED_PREFIX} ({reason}): {text}"


def apply_legacy_fallback(parallel_result: Any, legacy: Mapping[int, ExistingOutcome]) -> Any:
    """Return ``parallel_result`` as the legacy verifier decided it.

    For the gated executor: a root result the legacy verifier rejected is
    failed with that rejection, and one that only the package gate failed is
    accepted again. Criteria the worker never attempted are left as they are.
    """
    from ouroboros.orchestrator.parallel_executor_models import ACExecutionOutcome

    accepted_outcomes = (ACExecutionOutcome.SUCCEEDED, ACExecutionOutcome.SATISFIED_EXTERNALLY)
    cleared = {"check_package_repair": None, "check_package_failure_class": None}
    results = []
    success_delta = failure_delta = external_delta = 0
    for result in parallel_result.results:
        item = legacy.get(result.ac_index)
        previous = result.outcome
        gate_failed = bool(getattr(result, "check_package_failure_class", None))
        if item is None or item.terminal_status == "not_attempted":
            results.append(result)
        elif item.passed and previous is ACExecutionOutcome.FAILED and gate_failed:
            results.append(
                replace(
                    result,
                    success=True,
                    outcome=ACExecutionOutcome.SUCCEEDED,
                    error=None,
                    **cleared,
                )
            )
            success_delta += 1
            failure_delta -= 1
        elif not item.passed and (previous in accepted_outcomes or gate_failed):
            error = legacy_verdict_in_tree(result)[2] or (
                f"{LEGACY_REJECTION_ERROR} ({item.failure_class})"
                if item.failure_class
                else LEGACY_REJECTION_ERROR
            )
            results.append(
                replace(
                    result, success=False, outcome=ACExecutionOutcome.FAILED, error=error, **cleared
                )
            )
            if previous is ACExecutionOutcome.SUCCEEDED:
                success_delta -= 1
                failure_delta += 1
            elif previous is ACExecutionOutcome.SATISFIED_EXTERNALLY:
                external_delta -= 1
                failure_delta += 1
        else:
            results.append(result)
    return replace(
        parallel_result,
        results=tuple(results),
        success_count=parallel_result.success_count + success_delta,
        failure_count=parallel_result.failure_count + failure_delta,
        externally_satisfied_count=parallel_result.externally_satisfied_count + external_delta,
    )


class CheckPackageGate:
    """Per-attempt repair signal from the frozen package (see the module docstring)."""

    def __init__(self, authority: CheckPackageAuthority) -> None:
        self._authority = authority
        self.log: list[dict[str, Any]] = []
        # Attempts the legacy verifier failed on a criterion no admitted
        # check covers (it decides those criteria, retries included).
        self.legacy_failures = 0
        # One decision per attempt: settlement paths hand the same attempt to
        # the gate again; they get the stored decision, not a new verification.
        self._decided: dict[tuple[int, int], dict[str, Any] | None] = {}

    async def __call__(self, *, seed: Seed, ac_index: int, result: Any) -> Any:
        attempt = (ac_index, int(getattr(result, "retry_attempt", 0) or 0))
        if attempt in self._decided:
            stored = self._decided[attempt]
            return result if stored is None or not result.success else replace(result, **stored)
        try:
            decided = await self._decide(ac_index, result)
            if decided is result:
                self._decided[attempt] = None
            elif getattr(decided, "check_package_repair", None) or str(
                getattr(decided, "check_package_failure_class", None) or ""
            ).startswith(LEGACY_DECIDED_FAILURE_CLASS_PREFIX):
                if not getattr(decided, "check_package_repair", None):
                    # A legacy-decided rejection, counted once per attempt.
                    self.legacy_failures += 1
                self._decided[attempt] = {
                    "success": False,
                    "outcome": decided.outcome,
                    "error": decided.error,
                    "check_package_repair": decided.check_package_repair,
                    "check_package_failure_class": decided.check_package_failure_class,
                }
            return decided
        except Exception as exc:  # noqa: BLE001 - the gate must never fail an attempt by itself
            log.warning("boundary.gate.failed", ac_index=ac_index, error_type=type(exc).__name__)
            return result

    async def _decide(self, ac_index: int, result: Any) -> Any:
        authority = self._authority
        state = authority.state
        package = state.package
        if package is None or state.admission is None or not getattr(result, "success", False):
            return result
        keys = state.criterion_keys
        if not 0 <= ac_index < len(keys):
            return result
        key = keys[ac_index]
        check_ids = authority.admitted_check_ids(key)
        if not check_ids or key in authority.legacy_decided_keys():
            # No admitted check covers it (uncovered, non-behavioral, or
            # every check excluded at admission): the legacy verifier decides
            # it, so its rejection fails the attempt and drives the retry,
            # exactly as with the check package off.
            return self._legacy_decides(result)
        entries = authority.remember_declaration(key, _declared_from(result))
        options = authority.run_options()
        assignments, results = await assign_tiers(
            package,
            artifact=authority.candidate,
            base=state.base_snapshot,
            declared={key: entries} if entries else None,
            base_manifest=_base_manifest(state),
            expected_base_digest=state.admission.base_tree_digest,
            admitted_tiers=state.admission.check_tiers,
            run_options={"env": options["env"], "interpreter": options["interpreter"]},
            base_run_cache=authority.base_runs,
        )
        subset = {check_id: assignments[check_id] for check_id in check_ids}
        bound = await verify_with_bindings(
            package,
            authority.candidate,
            subset,
            timeout_seconds=authority.settings.check_timeout_seconds,
            **options,
        )
        verification = retire_revealed(bound.effective, authority.revealed)
        verdicts = criterion_verdicts(package, verification, assignments=subset)
        item = verdicts[key]
        await BoundaryLedger(authority.event_store).record_bindings(
            state.boundary_id,
            package_sha256=package.reference,
            payload={
                **bindings_payload(
                    subset,
                    {k: v for k, v in results.items() if k in subset},
                    phase="repair",
                ),
                "root_ac_index": ac_index,
                "retry_attempt": getattr(result, "retry_attempt", 0),
                "status": item.status.value,
            },
        )
        self.log.append(
            {"ac_index": ac_index, "status": item.status.value, "tier": item.tier.value}
        )
        retry_attempt = int(getattr(result, "retry_attempt", 0) or 0)
        if item.status.is_unverified and item.reason == NO_BINDING and not entries:
            # The criterion needs a late binding and the worker declared none.
            # Ask once, declaration only, within the retry budget.
            if key in authority.binding_requested:
                return result
            if not authority.repair_follows(retry_attempt):
                authority.binding_budget_exhausted.add(key)
                return result
            authority.binding_requested.add(key)
            self.log[-1]["binding_requested"] = True
            return self._repair(
                result, _declaration_request_message(key, authority.interfaces().get(ac_index))
            )
        if item.status is PackageCriterionStatus.INDETERMINATE and item.reason.startswith(
            BINDING_REJECTED_PREFIX
        ):
            # A rejected declaration is fixable within the retry budget; the
            # reason names no oracle value.
            return self._repair(result, _binding_rejected_message(item, package))
        if item.status is not PackageCriterionStatus.FAIL:
            return result
        partial = BoundaryVerdict(
            verdict="fail",
            reasons=(),
            boundary_id=state.boundary_id,
            package_sha256=package.reference,
            counterexamples=_counterexamples(verification) if verification is not None else (),
            verdicts=verdicts,
            oracle_results={
                check.check_id: check.oracle_result
                for check in (verification.checks if verification is not None else ())
                if check.oracle_result
            },
        )
        # One reveal per criterion per run, and only when a repair attempt
        # follows (otherwise the worker never sees the case).
        plan = plan_repair(
            partial,
            key,
            allow_reveal=key not in authority.revealed_criteria
            and authority.repair_follows(retry_attempt),
        )
        message = plan.message if plan is not None else PACKAGE_REJECTION_ERROR
        if plan is not None and plan.revealed_check_id and plan.revealed_case_id:
            # Only held-out cases failed: one of them is now shown to the
            # worker and retired from held-out statistics for this run.
            authority.revealed.setdefault(plan.revealed_check_id, set()).add(plan.revealed_case_id)
            authority.revealed_criteria.add(key)
            await BoundaryLedger(authority.event_store).record_case_revealed(
                state.boundary_id,
                package_sha256=package.reference,
                check_id=plan.revealed_check_id,
                criterion_key=key,
                case_id=plan.revealed_case_id,
                root_ac_index=ac_index,
                retry_attempt=getattr(result, "retry_attempt", 0),
            )
            self.log[-1]["revealed_case_id"] = plan.revealed_case_id
        return self._repair(result, message)

    @staticmethod
    def _legacy_decides(result: Any) -> Any:
        from ouroboros.orchestrator.parallel_executor_models import ACExecutionOutcome

        rejected, failure_class, _text = legacy_verdict_in_tree(result)
        if not rejected:
            return result
        return replace(
            result,
            success=False,
            outcome=ACExecutionOutcome.FAILED,
            error=legacy_decided_error(result, "no admitted check"),
            check_package_failure_class=(
                f"{LEGACY_DECIDED_FAILURE_CLASS_PREFIX}:{failure_class or 'rejected'}"
            ),
        )

    @staticmethod
    def _repair(result: Any, message: str) -> Any:
        from ouroboros.orchestrator.parallel_executor_models import ACExecutionOutcome

        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
        return replace(
            result,
            success=False,
            outcome=ACExecutionOutcome.FAILED,
            error=PACKAGE_REJECTION_ERROR,
            check_package_repair=message,
            check_package_failure_class=f"{PACKAGE_FAILURE_CLASS_PREFIX}:{digest}",
        )


def _declaration_request_message(key: str, interface: Mapping[str, Any] | None) -> str:
    """Declaration-only repair: name the criterion and the grammar, nothing about the oracle."""
    return (
        f"The check package could not find the entry point of this criterion ({key}): the "
        "default name it looks for does not exist, and your evidence declared no "
        "entry_points. Keep your implementation unless it is incomplete, and emit the "
        "evidence JSON again with entry_points declared for this criterion."
        + entry_points_request(interface)
    )


def _binding_rejected_message(item: Any, package: Any) -> str:
    """Repair text for a declared entry point that was rejected (no oracle values)."""
    binding = item.binding or {}
    spec = next((o for o in package.oracles if o.check_id in item.check_ids), None)
    params = ", ".join(spec.params) if spec is not None else ""
    lines = [
        f"Your declared entry point for this criterion was rejected: {item.reason}.",
        f"Declared: {binding.get('call_kind')} {binding.get('symbol')}"
        + (
            f" with arg_map {json.dumps(binding.get('arg_map'), sort_keys=True)}"
            if binding.get("arg_map")
            else ""
        )
        + ".",
        "Fix the entry_points declaration: name a function your change introduced or "
        "changed (or one that exists at the base), outside .ouroboros_checks, whose "
        f"inputs ({params}) are mapped only by name or position.",
    ]
    return "\n".join(lines)


class CheckPackageAuthority:
    """The frozen check package as the acceptance authority of one run."""

    def __init__(
        self,
        state: BoundaryRunState,
        settings: CheckPackageSettings,
        *,
        event_store: EventStore,
        candidate_checkout: Path,
    ) -> None:
        self._state = state
        self._settings = settings
        self._event_store = event_store
        self._candidate = candidate_checkout
        self.outcome: AuthorityOutcome | None = None
        self.gate = CheckPackageGate(self)
        self.installed = False
        # One base run per late binding across repair attempts and the end.
        self.base_runs: dict[str, Any] = {}
        # The worker's latest declared entry point per criterion. A later
        # attempt that declares nothing does not withdraw it: otherwise a
        # worker could turn a failing criterion into an unverified one by
        # omitting entry_points after a counterexample.
        self.declared: dict[str, list[Any]] = {}
        # Held-out cases revealed in a repair message, per check id, and the
        # criteria that already had their one reveal in this run.
        self.revealed: dict[str, set[str]] = {}
        self.revealed_criteria: set[str] = set()
        # The executor's same-runtime retry budget (set by ``install``).
        self.max_retry_attempts: int | None = None
        # Criteria that needed a late binding and had no declaration: asked
        # once for a declaration, or not asked because no retry was left.
        self.binding_requested: set[str] = set()
        self.binding_budget_exhausted: set[str] = set()

    def repair_follows(self, retry_attempt: int) -> bool:
        """Whether a repair attempt follows ``retry_attempt`` (unknown budget: yes)."""
        return self.max_retry_attempts is None or retry_attempt < self.max_retry_attempts

    def remember_declaration(self, key: str, entries: list[Any]) -> list[Any]:
        """Record ``entries`` for ``key`` when present; return the declaration in force."""
        if entries:
            self.declared[key] = list(entries)
        return self.declared.get(key, [])

    @property
    def state(self) -> BoundaryRunState:
        return self._state

    @property
    def settings(self) -> CheckPackageSettings:
        return self._settings

    @property
    def event_store(self) -> EventStore:
        return self._event_store

    @property
    def candidate(self) -> Path:
        return self._candidate.resolve()

    def run_options(self) -> dict[str, Any]:
        interpreter = self._state.interpreter or resolve_check_interpreter(self.candidate)
        return {
            "env": scrubbed_check_environment(),
            "interpreter": interpreter.path,
            "interpreter_source": interpreter.source,
        }

    def interfaces(self) -> dict[int, dict[str, Any]]:
        """Root criterion index to the oracle's call kind and input names (no cases)."""
        package = self._state.package
        if package is None:
            return {}
        index = {key: number for number, key in enumerate(self._state.criterion_keys)}
        excluded = self._excluded()
        lost = self.legacy_decided_keys()
        return {
            index[spec.criterion_key]: spec.interface()
            for spec in package.oracles
            if spec.criterion_key in index
            and spec.check_id not in excluded
            and spec.criterion_key not in lost
        }

    def _excluded(self) -> frozenset[str]:
        admission = self._state.admission
        return excluded_check_ids(admission.check_tiers if admission is not None else None)

    def admitted_check_ids(self, key: str) -> list[str]:
        """The admitted (not excluded) checks linked to criterion ``key``."""
        package = self._state.package
        if package is None:
            return []
        excluded = self._excluded()
        return [
            check.check_id
            for check in package.checks
            if check.check_id not in excluded
            and any(link.criterion_key == key for link in check.assertions)
        ]

    def legacy_decided_keys(self) -> frozenset[str]:
        """Criteria that lost their authority to per-check admission (``per_check.py``)."""
        package = self._state.package
        excluded = self._excluded()
        if package is None or not excluded:
            return frozenset()
        return frozenset(criteria_without_admitted_check(package, excluded))

    def install(self, executor: Any) -> None:
        """Make the legacy verifier advisory and the package the repair signal."""
        executor.check_package_gate = self.gate
        executor.check_package_interfaces = self.interfaces()
        budget = getattr(executor, "_ac_retry_attempts", None)
        if isinstance(budget, int) and not isinstance(budget, bool):
            self.max_retry_attempts = max(0, budget)
        self.installed = True

    async def __call__(self, *, seed: Seed, execution_id: str, parallel_result: Any) -> Any:
        if self.outcome is not None:
            # One verdict per run: a second call (for example a resumed
            # parallel pass on the same runner) keeps the first decision.
            return parallel_result
        try:
            return await self._decide_run(seed, execution_id, parallel_result)
        finally:
            if self.outcome is not None:
                reveal_commitment_salts(self._state)

    async def _decide_run(self, seed: Seed, execution_id: str, parallel_result: Any) -> Any:
        legacy = existing_outcomes_from_results(parallel_result, gated=self.installed)
        legacy_accepted = bool(legacy) and all(item.passed for item in legacy.values())
        try:
            declared: Mapping[str, list[Any]] = {}
            keys = seed_criterion_keys(seed)
            for result in getattr(parallel_result, "results", ()) or ():
                index = getattr(result, "ac_index", -1)
                if not 0 <= index < len(keys):
                    continue
                entries = self.remember_declaration(keys[index], _declared_from(result))
                if entries:
                    declared = {**declared, keys[index]: entries}
            verdict = await verify_check_package(
                self._state,
                event_store=self._event_store,
                candidate_checkout=self._candidate,
                settings=self._settings,
                declared_entry_points=declared,
                base_run_cache=self.base_runs,
                revealed=self.revealed,
            )
            if verdict.package_sha256 is None:
                # No admitted package: the legacy verifier decides.
                if not self.installed:
                    self.outcome = AuthorityOutcome(legacy_accepted, verdict=verdict, legacy=legacy)
                    return parallel_result
                return await self._fall_back(
                    parallel_result, legacy, legacy_accepted, "no_admitted_package", verdict=verdict
                )
            verdict = _label_missing_bindings(
                verdict, self.binding_requested, self.binding_budget_exhausted
            )
            reconciliation = reconcile_acceptance(
                keys,
                verdict.verdicts,
                legacy,
                existing_run_accepted=bool(parallel_result.all_succeeded),
                legacy_decides_unverified=True,
            )
            await BoundaryLedger(self._event_store).record_acceptance_reconciled(
                self._state.boundary_id,
                package_sha256=verdict.package_sha256,
                reconciliation=reconciliation.to_dict(),
            )
            self.outcome = AuthorityOutcome(
                legacy_accepted, verdict=verdict, reconciliation=reconciliation, legacy=legacy
            )
            return apply_reconciliation(parallel_result, reconciliation)
        except Exception as exc:  # noqa: BLE001 - the executor result must survive any failure
            log.warning(
                "boundary.authority.failed",
                execution_id=execution_id,
                boundary_id=self._state.boundary_id,
                error_type=type(exc).__name__,
            )
            if not self.installed:
                # Nothing was made advisory: the executor result is the legacy one.
                self.outcome = AuthorityOutcome(
                    legacy_accepted, error=type(exc).__name__, legacy=legacy
                )
                return parallel_result
            return await self._fall_back(
                parallel_result,
                legacy,
                legacy_accepted,
                f"authority_error:{type(exc).__name__}",
                error=type(exc).__name__,
            )

    async def _fall_back(
        self,
        parallel_result: Any,
        legacy: dict[int, ExistingOutcome],
        legacy_accepted: bool,
        reason: str,
        *,
        error: str | None = None,
        verdict: BoundaryVerdict | None = None,
    ) -> Any:
        """Decide the run with the legacy verdicts the gate made advisory (never raises).

        The outcome and the ledger event are written only after the legacy
        verdicts were applied. If applying them fails, no verifier decided the
        run, so every attempted root is failed (``<reason>:fallback_failed``).
        """
        try:
            decided = apply_legacy_fallback(parallel_result, legacy)
        except Exception:  # noqa: BLE001 - never raise into the run
            log.warning("boundary.authority.fallback_failed", reason=reason)
            reason = f"{reason}:fallback_failed"
            decided = _fail_attempted(parallel_result, legacy)
        self.outcome = AuthorityOutcome(
            legacy_accepted, verdict=verdict, error=error, legacy=legacy, fallback_reason=reason
        )
        log.warning(
            "boundary.authority.legacy_fallback",
            boundary_id=self._state.boundary_id,
            reason=reason,
        )
        try:
            await BoundaryLedger(self._event_store).record_legacy_fallback(
                self._state.boundary_id, reason=reason
            )
        except Exception:  # noqa: BLE001 - the fallback itself must stand
            log.warning("boundary.authority.fallback_not_recorded", reason=reason)
        return decided


def _label_missing_bindings(
    verdict: BoundaryVerdict, requested: set[str], exhausted: set[str]
) -> BoundaryVerdict:
    """Say why a criterion that needed a late binding still has none.

    ``no_binding_after_request``: the worker was asked once and declared
    nothing valid; ``no_binding_budget_exhausted``: no retry was left to ask.
    The criterion stays unverified either way.
    """
    relabeled = {}
    for key, item in verdict.verdicts.items():
        if item.reason != NO_BINDING or not item.status.is_unverified:
            continue
        if key in requested:
            relabeled[key] = replace(item, reason=NO_BINDING_AFTER_REQUEST)
        elif key in exhausted:
            relabeled[key] = replace(item, reason=NO_BINDING_BUDGET_EXHAUSTED)
    if not relabeled:
        return verdict
    return replace(verdict, verdicts={**verdict.verdicts, **relabeled})


def reveal_commitment_salts(state: BoundaryRunState | None) -> None:
    """``persist_commitment_salts`` after the final verdict; never raises into the run.

    The run's in-process registry entry (``run_wiring.live_state``) is dropped
    too: after the verdict nothing re-derives the held-out cases from memory.
    """
    if state is None:
        return
    forget_live_state(state)
    try:
        persist_commitment_salts(state)
    except Exception as exc:  # noqa: BLE001 - an unwritten salt only loses auditability
        log.warning("boundary.commitment_salts_not_written", error_type=type(exc).__name__)


def _fail_attempted(parallel_result: Any, legacy: Mapping[int, ExistingOutcome]) -> Any:
    """Every attempted root failed: the run was decided by no verifier."""
    try:
        from ouroboros.orchestrator.parallel_executor_models import ACExecutionOutcome

        results = tuple(
            replace(
                result,
                success=False,
                outcome=ACExecutionOutcome.FAILED,
                error=PACKAGE_INDETERMINATE_ERROR,
            )
            if result.ac_index in legacy
            and legacy[result.ac_index].terminal_status != "not_attempted"
            else result
            for result in parallel_result.results
        )
        failed = sum(1 for result in results if result.outcome is ACExecutionOutcome.FAILED)
        return replace(
            parallel_result,
            results=results,
            success_count=sum(1 for r in results if r.outcome is ACExecutionOutcome.SUCCEEDED),
            failure_count=failed,
            externally_satisfied_count=sum(
                1 for r in results if r.outcome is ACExecutionOutcome.SATISFIED_EXTERNALLY
            ),
        )
    except Exception:  # noqa: BLE001 - never raise into the run
        log.warning("boundary.authority.fail_attempted_failed")
        return parallel_result


__all__ = [
    "PACKAGE_FAILURE_CLASS_PREFIX",
    "PACKAGE_INDETERMINATE_ERROR",
    "PACKAGE_REJECTION_ERROR",
    "AuthorityOutcome",
    "CheckPackageAuthority",
    "CheckPackageGate",
    "apply_legacy_fallback",
    "apply_reconciliation",
    "existing_outcomes_from_results",
    "legacy_verdict_in_tree",
]
