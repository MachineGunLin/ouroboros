"""Opt-in check-package boundary around one ``ooo run`` execution.

Order for a new run (flag ``boundary.check_package: on``):

1. ``prepare_check_package``: construct a package from the frozen Seed with
   one read-only model call, persist it by digest, record
   ``boundary.check_package.frozen``, admit it on isolated copies of the
   worker's starting checkout (the base), record the admission, and finally
   record ``boundary.actor.started`` for the run's execution id. The ledger
   refuses the actor start unless the bound version is sealed and admitted
   and the worker workspace holds no generated check file. The caller
   dispatches the worker only after this returns.
2. ``verify_check_package``: after the worker stops, run the unchanged
   package on the candidate workspace, record ``boundary.candidate.verified``,
   then ``select_incumbent`` (base versus candidate) and record
   ``boundary.selection.decided``.

Regeneration policy (``RegenerationPolicy``). Each attempt is its own boundary
version ``<execution_id>/check_package/v<n>``. With the product policy a
version that is not admitted is superseded by the next attempt, and the
constructor is told why the earlier version was not admitted. When no attempt
is admitted, a final version is sealed as ``construction_failed`` so the
worker can still start (it never depended on the package) and the run's
verdict is indeterminate. The study policy allows exactly one attempt.

The worker never receives the package: it runs from the unchanged Seed, whose
worker prompt already omits ``verify_command`` and ``output_assertion``.
Package records and receipts are stored under ``<store_dir>/packages`` and
``<store_dir>/receipts``, outside every checkout. Held-out inputs and
expected values are never written to disk: they stay in this process's
memory for the run. The package record keeps a held-out case's id and a
keyed hash only (``package.package_record``; the key is per run and in
memory), constructor partial replies keep case ids only, and a stored
receipt keeps a held-out case's id and pass/fail only until the case is
revealed (``oracle.redact_held_out``). The store directory is owner-only
(0700).

Every package is committed before it is frozen (``package.commit_package``,
a fresh 256-bit salt per package): the journal, the record and every receipt
cite SHA-256(salt || canonical package bytes), never the unkeyed digest, so
nothing in the store or the journal lets held-out values be confirmed by
enumeration. The salts stay in memory until the run's final verdict and are
then written to a controller-private directory (``persist_commitment_salts``).
A run that resumes in another process cannot recover the held-out cases; it
recomputes the package decision from the visible cases (``boundary/resume.py``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
import json
import os
from pathlib import Path
import secrets
from typing import TYPE_CHECKING, Any

from ouroboros.boundary.acceptance import (
    ArtifactVerdict,
    CriterionVerdict,
    PackageCriterionStatus,
    artifact_verdict,
    criterion_verdicts,
)
from ouroboros.boundary.admission import (
    AdmissionResult,
    CandidateVerdict,
    CandidateVerification,
    CheckStatus,
    PackageVerdict,
    admit_check_package,
    write_receipt,
)
from ouroboros.boundary.binding import CheckTier, TierAssignment
from ouroboros.boundary.binding_flow import (
    BoundVerification,
    DeclaredBindingResult,
    admission_tiers,
    assign_tiers,
    bindings_payload,
    retire_revealed,
    snapshot_base,
    verify_with_bindings,
)
from ouroboros.boundary.check_env import (
    CheckInterpreter,
    resolve_check_interpreter,
    scrubbed_check_environment,
)
from ouroboros.boundary.constructor import ALL_CRITERIA_UNCOVERED
from ouroboros.boundary.ledger import BoundaryLedger
from ouroboros.boundary.oracle import apply_reveals, first_failing_heldout, repair_lines
from ouroboros.boundary.package import (
    CheckPackage,
    commit_package,
    new_commitment_salt,
    private_directory,
    seed_criterion_keys,
    seed_digest,
    sha256_bytes,
    write_commitment_salt,
    write_package_record,
)
from ouroboros.boundary.reference_check import (
    REFERENCE_LEFT_NO_CHECKS,
    ReferenceCheck,
    check_references,
)
from ouroboros.boundary.rollout import (
    AssignmentSource,
    CheckPackageAssignment,
    resolve_check_package_assignment,
)
from ouroboros.boundary.selection import (
    ArtifactRef,
    SelectionDecision,
    SelectionReason,
    select_incumbent,
)
from ouroboros.boundary.tree import tree_digest

if TYPE_CHECKING:
    from ouroboros.boundary.constructor import CheckConstructor
    from ouroboros.core.seed import Seed
    from ouroboros.persistence.event_store import EventStore

_FEEDBACK_TAIL_CHARS = 400
_COUNTEREXAMPLE_TAIL_CHARS = 1500


class RegenerationPolicy(StrEnum):
    """How many package versions one boundary slot may go through."""

    PRODUCT = "product"
    """Regenerate a non-admitted package as a new, superseding version."""

    STUDY = "study"
    """Exactly one package per boundary; no regeneration."""


@dataclass(frozen=True, slots=True)
class CheckPackageSettings:
    """Resolved configuration for one run."""

    enabled: bool
    constructor_timeout_seconds: int = 600
    check_timeout_seconds: int = 120
    max_construction_attempts: int = 2
    policy: RegenerationPolicy = RegenerationPolicy.PRODUCT
    assignment: CheckPackageAssignment | None = None

    @property
    def attempts(self) -> int:
        if self.policy is RegenerationPolicy.STUDY:
            return 1
        return max(1, self.max_construction_attempts)


def _load_boundary_config() -> Any:
    from ouroboros.config.loader import load_config

    return load_config().boundary


def resolve_check_package_settings(cli_value: bool | None = None) -> CheckPackageSettings:
    """Resolve the arm (``boundary/rollout.py``) and the budgets for one run.

    Precedence: CLI flag, then ``OUROBOROS_CHECK_PACKAGE``, then
    ``boundary.check_package`` in config, then the installation's randomized
    arm; ``off`` when none applies. Budgets always come from ``boundary`` in
    config. An unreadable config contributes no setting (and disables
    telemetry, so no randomized arm either).
    """
    try:
        config = _load_boundary_config()
        configured = config.check_package
        budgets: dict[str, Any] = {
            "constructor_timeout_seconds": config.constructor_timeout_seconds,
            "check_timeout_seconds": config.check_timeout_seconds,
            "max_construction_attempts": config.max_construction_attempts,
        }
    except Exception:
        configured = None
        budgets = {}
    assignment = resolve_check_package_assignment(cli_value, configured=configured)
    return CheckPackageSettings(enabled=assignment.enabled, assignment=assignment, **budgets)


def default_store_dir(execution_id: str) -> Path:
    """``~/.ouroboros/boundary/<execution_id>``: outside every checkout."""
    from ouroboros.config.models import get_config_dir

    return get_config_dir() / "boundary" / execution_id


def controller_private_dir(store: Path) -> Path:
    """Where the commitment salts go after the final verdict: beside the store, not in it.

    The store must never hold enough to confirm a held-out value; a salt next
    to the recorded commitment would. Owner-only (0700) and outside every
    checkout; code running as the same user can still read it (no OS
    sandbox), which is why it is written only after the final verdict.
    """
    return store.parent / f"{store.name}.controller-private"


def private_store_dir(store: Path) -> Path:
    """Create ``store`` owner-only (0700); the ``boundary`` parent too when it is one.

    The store holds the package record, the constructor's partial replies,
    receipts, and the base snapshot, none of which carries a held-out input or
    expected value. The mode keeps other users out; code running as the same
    user can still read it (there is no OS sandbox), which is why nothing in
    it may carry held-out values.
    """
    store.mkdir(parents=True, exist_ok=True)
    targets = [store]
    if store.parent.name == "boundary":
        targets.append(store.parent)
    for target in targets:
        try:
            os.chmod(target, 0o700)
        except OSError:
            continue
    return store


@dataclass(frozen=True, slots=True)
class FrozenCommitment:
    """One frozen package version and the salt of its commitment (controller memory only)."""

    boundary_id: str
    commitment: str
    salt: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class BoundaryRunState:
    """What the run holds between worker dispatch and candidate verification."""

    execution_id: str
    boundary_id: str
    versions: tuple[str, ...]
    seed_digest: str
    base_checkout: Path
    package: CheckPackage | None
    admission: AdmissionResult | None
    failure_reason: str | None
    store_dir: Path
    package_path: Path | None = None
    interpreter: CheckInterpreter | None = None
    criterion_keys: tuple[str, ...] = ()
    base_snapshot: Path | None = None
    base_manifest_path: Path | None = None
    commitments: tuple[FrozenCommitment, ...] = field(default=(), repr=False)
    reference_check: ReferenceCheck | None = None
    """What the reference check excluded from the bound version (``None``: not run)."""

    @property
    def admitted(self) -> bool:
        return self.admission is not None and self.admission.verdict is PackageVerdict.ADMITTED


@dataclass(frozen=True, slots=True)
class Counterexample:
    """One failing check on the candidate, for the person running the command."""

    check_id: str
    role: str
    reason: str
    return_code: int | None
    output_tail: str


@dataclass(frozen=True, slots=True)
class BoundaryVerdict:
    """Final verdict of the boundary for one run: ``pass``, ``fail`` or ``indeterminate``."""

    verdict: str
    reasons: tuple[str, ...]
    boundary_id: str
    # The package reference the journal cites (``CheckPackage.reference``: the
    # commitment of a committed package); ``None`` without an admitted package.
    package_sha256: str | None
    counterexamples: tuple[Counterexample, ...] = ()
    selection: SelectionDecision | None = None
    receipt_path: Path | None = None
    uncovered: tuple[str, ...] = field(default_factory=tuple)
    criteria: dict[str, PackageCriterionStatus] = field(default_factory=dict)
    verdicts: dict[str, CriterionVerdict] = field(default_factory=dict)
    artifact_verdict: ArtifactVerdict | None = None
    assignments: dict[str, TierAssignment] = field(default_factory=dict)
    binding_results: dict[str, DeclaredBindingResult] = field(default_factory=dict)
    oracle_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary for run output (no check code, no argv)."""
        return {
            "verdict": self.verdict,
            "artifact_verdict": self.artifact_verdict.value if self.artifact_verdict else None,
            "tiers": {key: item.tier.value for key, item in self.verdicts.items()},
            "reasons": list(self.reasons),
            "boundary_id": self.boundary_id,
            "package_reference": self.package_sha256,
            "failing_checks": [example.check_id for example in self.counterexamples],
            "uncovered_criteria": list(self.uncovered),
            "criteria": {key: status.value for key, status in self.criteria.items()},
            "selection": (
                None
                if self.selection is None
                else {
                    "replaced": self.selection.replaced,
                    "reason": self.selection.reason.value,
                }
            ),
            "receipt_path": str(self.receipt_path) if self.receipt_path else None,
        }


def _admission_feedback(admission: AdmissionResult) -> list[str]:
    feedback = [f"verdict {admission.verdict.value}", *admission.reasons]
    for check in admission.checks:
        if check.status is not CheckStatus.EXPECTED:
            tail = (check.output_tail or "").strip()[-_FEEDBACK_TAIL_CHARS:]
            feedback.append(
                f"{check.check_id} ({check.role.value}): {check.reason}, "
                f"exit {check.return_code}; output tail: {tail!r}"
            )
    return feedback


async def prepare_check_package(
    seed: Seed,
    *,
    event_store: EventStore,
    constructor: CheckConstructor,
    execution_id: str,
    base_checkout: Path,
    worker_workspace: Path,
    runtime_label: str | None,
    settings: CheckPackageSettings,
    store_dir: Path | None = None,
) -> BoundaryRunState:
    """Construct, freeze, and admit a package, then record the actor start.

    Raises ``BoundaryOrderError`` / ``BoundaryLeakError`` from the ledger; the
    caller must not dispatch the worker in that case.
    """
    store = private_store_dir(store_dir or default_store_dir(execution_id))
    ledger = BoundaryLedger(event_store)
    digest = seed_digest(seed)
    base = base_checkout.resolve()
    versions: list[str] = []
    feedback: list[str] = []
    previous: str | None = None
    previous_reason = ""
    package: CheckPackage | None = None
    admission: AdmissionResult | None = None
    package_path: Path | None = None
    failure_reason: str | None = None
    # Model-written checks run with a scrubbed environment and the project's
    # interpreter when one exists (boundary/check_env.py).
    interpreter = resolve_check_interpreter(base)
    # Keys the held-out case hashes of the stored package record; never stored.
    record_key = secrets.token_bytes(32)
    commitments: list[FrozenCommitment] = []
    reference_check: ReferenceCheck | None = None

    for attempt in range(1, settings.attempts + 1):
        boundary_id = f"{execution_id}/check_package/v{attempt}"
        persist = getattr(constructor, "persist_partials_to", None)
        if persist is not None:
            # Each criterion's oracle is kept as soon as it is produced.
            persist(store / "partial" / f"v{attempt}")
        outcome = await constructor.construct(seed, base, feedback=feedback)
        package, admission, package_path = outcome.package, None, None
        reference_check = None
        references = getattr(outcome, "references", None)
        if package is not None and references is not None and package.oracles:
            # Derived-expectation admission (``boundary/reference_check.py``):
            # before the package is frozen, cases that disagree with the
            # constructor's reference are excluded.
            package, reference_check = await check_references(
                package,
                references,
                seed=seed,
                env=scrubbed_check_environment(),
                interpreter=interpreter.path,
                timeout_seconds=settings.check_timeout_seconds,
            )
            if not package.checks:
                package = None
                outcome = replace(outcome, package=None, failure_reason=REFERENCE_LEFT_NO_CHECKS)
        if package is None:
            failure_reason = outcome.failure_reason or "constructor_failed"
            await ledger.record_construction_failed(
                boundary_id,
                seed_digest=digest,
                input_digest=outcome.input_digest,
                reason=failure_reason,
            )
            feedback = [failure_reason]
        else:
            # A fresh salt per package; the record, the journal and every
            # receipt cite the commitment (I2 orders it before the worker).
            salt = new_commitment_salt()
            package = commit_package(package, salt)
            assert package.commitment is not None
            commitments.append(FrozenCommitment(boundary_id, package.commitment, salt))
            package_path = write_package_record(package, store / "packages", record_key)
            # The digest of the bytes on disk goes into the journal before the
            # worker starts: a resume in another process detects any edit of
            # the record, visible case values included (R5 follow-up).
            await ledger.record_package_frozen(
                boundary_id,
                package,
                seed=seed,
                record_sha256=sha256_bytes(package_path.read_bytes()),
            )
            if reference_check is not None:
                await ledger.record_reference_checked(
                    boundary_id, package_sha256=package.reference, payload=reference_check.payload()
                )
            admission = await admit_check_package(
                package,
                base,
                timeout_seconds=settings.check_timeout_seconds,
                env=scrubbed_check_environment(),
                interpreter=interpreter.path,
                interpreter_source=interpreter.source,
                reject_prose_only_checks=True,
                reject_unsafe_checks=True,
                check_tiers=admission_tiers(package, seed, base),
            )
            write_receipt(admission, store / "receipts")
            await ledger.record_admission(boundary_id, admission)
            failure_reason = (
                None
                if admission.verdict is PackageVerdict.ADMITTED
                else f"package_{admission.verdict.value}"
            )
            feedback = _admission_feedback(admission)
        versions.append(boundary_id)
        if previous is not None:
            await ledger.record_superseded(
                previous, superseded_by=boundary_id, reason=previous_reason
            )
        if failure_reason is None or failure_reason == ALL_CRITERIA_UNCOVERED:
            # Admitted, or every criterion declared not executable (the
            # existing verifier decides them all; regenerating cannot help).
            break
        previous, previous_reason = boundary_id, failure_reason

    bound = versions[-1]
    if package is not None and failure_reason is not None:
        # A frozen but unadmitted version cannot host a worker. Seal a final
        # version that records the absence of an admitted package.
        final_id = f"{execution_id}/check_package/v{len(versions) + 1}"
        await ledger.record_construction_failed(
            final_id,
            seed_digest=digest,
            input_digest=package.input_digest,
            reason=f"no_admitted_package:{failure_reason}",
        )
        await ledger.record_superseded(bound, superseded_by=final_id, reason=failure_reason)
        versions.append(final_id)
        bound = final_id

    # The arm's source goes into the journal: a resumed run (another
    # process) reports the original assignment from it (R4-A2).
    source = (
        settings.assignment.source
        if settings.assignment is not None
        else AssignmentSource.USER_FORCED_ON
    )
    await ledger.record_actor_started(
        execution_id,
        [bound],
        workspace=worker_workspace,
        runtime=runtime_label,
        assignment=source.value,
    )
    admitted = admission is not None and admission.verdict is PackageVerdict.ADMITTED
    snapshot: tuple[Path, Path] | None = None
    if admitted and package is not None and any(not o.default_resolves for o in package.oracles):
        # A late binding is validated against the base after the worker has
        # stopped; keep the base outside every checkout until then.
        snapshot = snapshot_base(base, store)
    state = BoundaryRunState(
        execution_id=execution_id,
        boundary_id=bound,
        versions=tuple(versions),
        seed_digest=digest,
        base_checkout=base,
        package=package if admitted else None,
        admission=admission if admitted else None,
        failure_reason=failure_reason,
        store_dir=store,
        package_path=package_path if admitted else None,
        interpreter=interpreter,
        criterion_keys=seed_criterion_keys(seed),
        base_snapshot=snapshot[0] if snapshot else None,
        base_manifest_path=snapshot[1] if snapshot else None,
        reference_check=reference_check,
        commitments=tuple(commitments),
    )
    if state.admitted:
        _LIVE_STATES[execution_id] = state
    return state


# The admitted boundary of each run still in progress in this process, with
# its held-out cases. A run resumed in the same process (its controller task
# died, the process did not) re-derives the full package decision from here;
# a run resumed in another process finds nothing and decides on the visible
# cases only (``boundary/resume.py``). Dropped at the run's final verdict.
_LIVE_STATES: dict[str, BoundaryRunState] = {}


def live_state(execution_id: str) -> BoundaryRunState | None:
    """The in-memory boundary state of ``execution_id`` in this process, if any."""
    return _LIVE_STATES.get(execution_id)


def forget_live_state(state: BoundaryRunState | None) -> None:
    """Drop ``state`` from the in-process registry (after the final verdict)."""
    if state is not None and _LIVE_STATES.get(state.execution_id) is state:
        del _LIVE_STATES[state.execution_id]


def persist_commitment_salts(state: BoundaryRunState) -> list[Path]:
    """Reveal every frozen version's salt, after the run's final verdict (idempotent).

    Written to ``controller_private_dir(store)`` (0700, files 0600), never to
    the store. With the package retained by the caller, an auditor checks the
    pre-dispatch commitment with ``package.verify_commitment``.
    """
    if not state.commitments:
        return []
    directory = private_directory(controller_private_dir(state.store_dir))
    return [
        write_commitment_salt(item.salt, item.commitment, directory) for item in state.commitments
    ]


def _counterexamples(verification: CandidateVerification) -> tuple[Counterexample, ...]:
    return tuple(
        Counterexample(
            check_id=check.check_id,
            role=check.role.value,
            reason=check.reason,
            return_code=check.return_code,
            output_tail=(check.output_tail or "")[-_COUNTEREXAMPLE_TAIL_CHARS:],
        )
        for check in verification.checks
        if check.status is not CheckStatus.EXPECTED
    )


UNAVAILABLE_VERDICT = "unavailable"


def unavailable_line(state: BoundaryRunState) -> str:
    """The one line a run prints when no package was admitted."""
    return (
        f"Check package unavailable ({state.failure_reason or 'no_admitted_package'}); "
        "legacy verification decided this run."
    )


def _base_manifest(state: BoundaryRunState) -> dict[str, str] | None:
    if state.base_manifest_path is None or not state.base_manifest_path.is_file():
        return None
    data = json.loads(state.base_manifest_path.read_text("utf-8"))
    return data if isinstance(data, dict) else None


async def verify_check_package(
    state: BoundaryRunState,
    *,
    event_store: EventStore,
    candidate_checkout: Path,
    settings: CheckPackageSettings,
    declared_entry_points: Mapping[str, Sequence[Any]] | None = None,
    base_run_cache: dict[str, Any] | None = None,
    revealed: Mapping[str, Collection[str]] | None = None,
) -> BoundaryVerdict:
    """Bind, then run the unchanged package on the candidate; record everything.

    ``declared_entry_points`` maps a criterion key to the worker's declared
    ``entry_points`` (typed evidence). ``revealed`` maps a check id to the
    held-out case ids already shown to the worker in a repair message; they
    are retired from held-out statistics. Order: final bindings
    (``boundary.binding.recorded``), candidate verification (plus one R3
    re-run of transiently indeterminate checks), selection.

    Without an admitted package the verdict is ``unavailable`` with no
    per-criterion verdicts: the run falls back to the legacy verifier exactly
    as if the check package were off (user decision, 2026-09-27).
    """
    if state.package is None or state.admission is None:
        return BoundaryVerdict(
            verdict=UNAVAILABLE_VERDICT,
            reasons=(state.failure_reason or "no_admitted_package",),
            boundary_id=state.boundary_id,
            package_sha256=None,
        )
    ledger = BoundaryLedger(event_store)
    package = state.package
    candidate = candidate_checkout.resolve()
    candidate_ref = ArtifactRef(
        artifact_id=f"{state.execution_id}:candidate",
        tree_digest=tree_digest(candidate),
        seed_digest=package.seed_digest,
    )
    interpreter = state.interpreter or resolve_check_interpreter(candidate)
    run_options = {
        "env": scrubbed_check_environment(),
        "interpreter": interpreter.path,
        "interpreter_source": interpreter.source,
    }
    assignments, results = await assign_tiers(
        package,
        artifact=candidate,
        base=state.base_snapshot,
        declared=declared_entry_points,
        base_manifest=_base_manifest(state),
        expected_base_digest=state.admission.base_tree_digest,
        admitted_tiers=state.admission.check_tiers,
        run_options={"env": run_options["env"], "interpreter": run_options["interpreter"]},
        base_run_cache=base_run_cache,
    )
    await ledger.record_bindings(
        state.boundary_id,
        package_sha256=package.reference,
        payload=bindings_payload(assignments, results, phase="final"),
    )
    bound = await verify_with_bindings(
        package,
        candidate,
        assignments,
        timeout_seconds=settings.check_timeout_seconds,
        **run_options,
    )
    bound = BoundVerification(
        retire_revealed(bound.first, revealed), retire_revealed(bound.rerun, revealed)
    )
    receipt: Path | None = None
    for run in (bound.first, bound.rerun):
        if run is not None:
            receipt = write_receipt(run, state.store_dir / "receipts")
            await ledger.record_candidate_verification(state.boundary_id, run)
    verification = bound.effective
    decision: SelectionDecision | None = None
    if verification is not None:
        decision = select_incumbent(
            incumbent=ArtifactRef(
                artifact_id=f"{state.execution_id}:base",
                tree_digest=state.admission.base_tree_digest,
                seed_digest=package.seed_digest,
            ),
            candidate=candidate_ref,
            package=package,
            admission=state.admission,
            verification=verification,
            candidate_checkout=candidate,
        )
        await ledger.record_selection(state.boundary_id, decision)
    verdicts = criterion_verdicts(
        package,
        verification,
        assignments=assignments,
        candidate_identity_ok=decision is None
        or decision.reason is not SelectionReason.CANDIDATE_IDENTITY_MISMATCH,
    )
    overall = artifact_verdict(item.status for item in verdicts.values())
    if overall is ArtifactVerdict.UNVERIFIED:
        reasons: tuple[str, ...] = ("all_unverified",)
    else:
        reasons = verification.reasons if verification is not None else ("no_bound_checks",)
    return BoundaryVerdict(
        verdict=overall.value,
        reasons=reasons,
        boundary_id=state.boundary_id,
        package_sha256=package.reference,
        counterexamples=_counterexamples(verification) if verification is not None else (),
        selection=decision,
        receipt_path=receipt,
        uncovered=tuple(item.criterion_key for item in package.uncovered),
        criteria={key: item.status for key, item in verdicts.items()},
        verdicts=verdicts,
        artifact_verdict=overall,
        assignments=assignments,
        binding_results=results,
        oracle_results={
            check.check_id: check.oracle_result
            for check in (verification.checks if verification is not None else ())
            if check.oracle_result
        },
    )


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """The repair message for one failing criterion, and the case it reveals (if any)."""

    message: str
    revealed_check_id: str | None = None
    revealed_case_id: str | None = None


def plan_repair(
    verdict: BoundaryVerdict, criterion_key: str, *, allow_reveal: bool = True
) -> RepairPlan | None:
    """Counterexample repair for one failing criterion, or ``None``.

    Visible cases are shown in full. When the criterion failed only on
    held-out cases and ``allow_reveal`` is true, exactly one failing held-out
    case is revealed (input, expected and observed output) and named in the
    plan, so the caller can retire it (``boundary.oracle.case_revealed``); the
    other held-out cases stay hidden and are counted. The caller passes
    ``allow_reveal=False`` when no repair attempt will follow or when this
    criterion already had its one reveal in the run. For a worker-declared
    binding (tier A') the message names the binding the check ran through.
    """
    item = verdict.verdicts.get(criterion_key)
    if item is None or item.status is not PackageCriterionStatus.FAIL:
        return None
    lines = ["The frozen check package failed this criterion on your workspace."]
    if item.tier is CheckTier.A_PRIME and item.binding is not None:
        lines.append(
            "It called your declared entry point: "
            f"{item.binding.get('call_kind')} {item.binding.get('symbol')}"
            + (
                f" with arg_map {json.dumps(item.binding.get('arg_map'), sort_keys=True)}"
                if item.binding.get("arg_map")
                else ""
            )
            + "."
        )
    elif item.binding is not None:
        lines.append(f"It called {item.binding.get('symbol')}.")
    results = {
        check_id: verdict.oracle_results[check_id]
        for check_id in item.check_ids
        if verdict.oracle_results.get(check_id)
    }
    reveal: tuple[str, str] | None = None
    failing = [
        case
        for result in results.values()
        for case in result.get("cases") or ()
        if not case.get("passed")
    ]
    if allow_reveal and failing and all(case.get("held_out") for case in failing):
        for check_id, result in results.items():
            case = first_failing_heldout(result)
            if case is not None:
                reveal = (check_id, str(case.get("case_id")))
                results[check_id] = apply_reveals(result, [reveal[1]]) or result
                lines.append(
                    "Every failing case was held out; one of them is revealed below "
                    "(it no longer counts as held out)."
                )
                break
    shown = False
    for result in results.values():
        counter = repair_lines(result)
        lines.extend(counter)
        shown = shown or bool(counter)
    if not shown:
        # Script checks only: an oracle check's counterexamples come from its
        # structured result above, never from its output.
        for example in verdict.counterexamples:
            if (
                example.check_id in item.check_ids
                and example.check_id not in verdict.oracle_results
                and example.output_tail.strip()
            ):
                lines.append(example.output_tail.strip()[-600:])
    return RepairPlan(
        "\n".join(lines),
        reveal[0] if reveal else None,
        reveal[1] if reveal else None,
    )


def repair_message(verdict: BoundaryVerdict, criterion_key: str) -> str | None:
    """``plan_repair(...).message``; the caller must record a reveal it shows."""
    plan = plan_repair(verdict, criterion_key)
    return None if plan is None else plan.message


def render_preparation(state: BoundaryRunState) -> list[str]:
    """Plain-text lines describing the boundary the worker is bound to."""
    lines = [f"Check package boundary: {state.boundary_id}"]
    if len(state.versions) > 1:
        lines.append(f"Superseded versions: {', '.join(state.versions[:-1])}")
    if state.admitted and state.package is not None:
        package = state.package
        roles = ", ".join(f"{check.check_id} ({check.role.value})" for check in package.checks)
        # The commitment, never the unkeyed digest: even a 64-bit prefix of
        # that digest would confirm guessed held-out values.
        lines.append(f"Package {package.reference[:16]} admitted on the base: {roles}")
        if state.interpreter is not None:
            lines.append(
                f"Checks run with {state.interpreter.path} ({state.interpreter.source}) "
                "and a scrubbed environment"
            )
        if package.uncovered:
            lines.append(
                f"Uncovered criteria: {len(package.uncovered)} of {len(package.criterion_keys)}"
            )
    else:
        lines.append(f"No admitted package ({state.failure_reason}).")
    return lines


def render_verdict(verdict: BoundaryVerdict) -> list[str]:
    """Plain-text lines: verdict first, then each counterexample."""
    lines = [f"Check package verdict: {verdict.verdict}"]
    if verdict.verdicts:
        tiers = ", ".join(
            f"{key}: {item.status.value} (tier {item.tier.label})"
            for key, item in verdict.verdicts.items()
        )
        lines.append(f"Criteria: {tiers}")
    if verdict.selection is not None:
        outcome = "accepted" if verdict.selection.replaced else "not accepted"
        lines.append(f"Candidate {outcome} ({verdict.selection.reason.value})")
    if verdict.verdict not in (CandidateVerdict.PASS.value, ArtifactVerdict.UNVERIFIED.value) and (
        verdict.reasons
    ):
        lines.append(f"Reasons: {', '.join(verdict.reasons)}")
    for example in verdict.counterexamples:
        lines.append(
            f"- {example.check_id} ({example.role}): {example.reason}, exit {example.return_code}"
        )
        if example.output_tail.strip():
            lines.append(example.output_tail.rstrip())
    if verdict.uncovered:
        lines.append(f"Criteria without a check (not verified): {len(verdict.uncovered)}")
    if verdict.receipt_path is not None:
        lines.append(f"Receipt: {verdict.receipt_path}")
    return lines
