"""Arm-on resume: recompute the package decision after the controller died (R3-S3).

A target can kill the controller (it is a child of it), or the controller can
die for any other reason, after the worker stopped and before the package
decided. The resumed run must not fall back to the legacy verifier for the
criteria the package covers: that would turn "kill the controller" into
"skip the package". So on resume the package decision is recomputed on the
current workspace, with no model call:

- in the same process (the run's task died, the process did not) the
  admitted package is still in memory (``run_wiring.live_state``): it is
  re-derived from there after checking it against the frozen commitment, and
  the held-out cases decide as usual;
- in another process the held-out cases are gone (they were never written to
  disk). The package is re-derived from the stored record with its visible
  cases only. A visible failure fails the criterion; otherwise a criterion
  whose check had held-out cases is indeterminate (``held_out_unavailable``),
  because passing only the visible cases is exactly what killing the
  controller would buy. A criterion whose cases were all visible is decided
  normally. The store is as writable as the workspace, so the record is used
  only when it agrees with what the journal recorded before the worker
  started (``record_mismatch``): the manifest recomputed from the record
  (case ids, case and held-out counts, file digests) and the held-out flag
  of every case the admission run saw. Otherwise every covered criterion is
  indeterminate (``package_record_tampered``).

A covered criterion without a package decision (the record or the journal is
unreadable, or the record disagrees with the journal) is indeterminate: not
accepted, a non-zero exit, never a legacy decision. Uncovered criteria keep
the rule of the live run (unverified, accepted when attempted). What counts
as an attempt is the live arm-on rule
(``existing_outcomes_from_results(..., gated=True)``): a root that failed for
any reason other than the package gate (a failed session, a failed verify
command, or a legacy rejection of a resumed attempt, which runs without the
gate) is not accepted, whatever the package says. The recomputed decision is
recorded as ``boundary.acceptance.resumed``; the frozen boundary's single-shot
records (final bindings, candidate verification) are not written again.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import structlog

from ouroboros.boundary.acceptance import (
    CriterionVerdict,
    PackageCriterionStatus,
    artifact_verdict,
    criterion_verdicts,
    reconcile_acceptance,
    render_reconciliation,
)
from ouroboros.boundary.authority import (
    AuthorityOutcome,
    _declared_from,
    _fail_attempted,
    apply_reconciliation,
    existing_outcomes_from_results,
    reveal_commitment_salts,
)
from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.binding_flow import (
    BASE_MANIFEST_FILE,
    BASE_SNAPSHOT_DIR,
    assign_tiers,
    retire_revealed,
    verify_with_bindings,
)
from ouroboros.boundary.check_env import resolve_check_interpreter, scrubbed_check_environment
from ouroboros.boundary.events import (
    ACTOR_STARTED,
    ADMISSION_COMPLETED,
    BOUNDARY_AGGREGATE_TYPE,
    CASE_REVEALED,
    PACKAGE_FROZEN,
)
from ouroboros.boundary.ledger import BoundaryLedger
from ouroboros.boundary.oracle import ORACLE_DATA_PATH, OracleSpec, is_oracle_file
from ouroboros.boundary.package import (
    CHECK_PACKAGE_SCHEMA,
    CheckPackage,
    canonical_json_bytes,
    cited_reference,
    oracle_files,
    seed_criterion_keys,
    sha256_bytes,
)
from ouroboros.boundary.run_wiring import (
    BoundaryVerdict,
    CheckPackageSettings,
    default_store_dir,
    live_state,
    render_verdict,
)

if TYPE_CHECKING:
    from ouroboros.core.seed import Seed
    from ouroboros.persistence.event_store import EventStore

log = structlog.get_logger(__name__)

HELD_OUT_UNAVAILABLE = "held_out_unavailable"
PACKAGE_UNAVAILABLE = "package_unavailable_on_resume"
PACKAGE_RECORD_TAMPERED = "package_record_tampered"
RESUMED_SCHEMA = "ouroboros.acceptance_resumed.v1"
_MAX_VERSIONS = 64
_REFERENCE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ResumedBoundary:
    """What a resumed run can recover of the boundary its worker was bound to."""

    execution_id: str
    boundary_id: str
    reference: str
    covered: tuple[str, ...]
    store_dir: Path
    package: CheckPackage | None
    held_out_checks: frozenset[str] = frozenset()
    base_tree_digest: str | None = None
    check_tiers: dict[str, str] | None = None
    revealed: dict[str, set[str]] | None = None
    source: str = "record"
    reason: str | None = None
    assignment: str | None = None
    """The arm's source recorded at the worker start (``rollout.AssignmentSource``)."""


def visible_package(record: dict[str, Any]) -> tuple[CheckPackage, frozenset[str]]:
    """Re-derive, in memory, the stored package without its held-out cases.

    Returns the package and the ids of the checks that lost held-out cases. A
    check left with no visible case is dropped; its criterion is listed as
    uncovered only when no other check links it (the caller overrides it to
    ``held_out_unavailable`` either way).
    """
    data = copy.deepcopy(record["package"])
    held: set[str] = set()
    dropped: dict[str, str] = {}
    oracles: list[OracleSpec] = []
    for spec in data.get("oracles") or ():
        visible = [case for case in spec["cases"] if not case.get("held_out")]
        if len(visible) != len(spec["cases"]):
            held.add(spec["check_id"])
        if visible:
            oracles.append(OracleSpec.model_validate({**spec, "cases": visible}))
        else:
            dropped[spec["check_id"]] = spec["criterion_key"]
    checks = [check for check in data.get("checks", ()) if check["check_id"] not in dropped]
    linked = {link["criterion_key"] for check in checks for link in check["assertions"]}
    uncovered = list(data.get("uncovered") or ())
    for key in sorted(set(dropped.values()) - linked):
        uncovered.append({"criterion_key": key, "reason": HELD_OUT_UNAVAILABLE})
    files = [
        {key: item[key] for key in ("path", "sha256", "content")}
        for item in data.get("files", ())
        if not is_oracle_file(item["path"])
    ]
    # The harness and the oracle data are rebuilt: this product's harness,
    # and the data of the visible cases only.
    files.extend(item.model_dump() for item in oracle_files(oracles))
    if not oracles:
        data["schema_version"] = CHECK_PACKAGE_SCHEMA
        data.pop("binding_grammar", None)
    data.update(
        {
            "oracles": [spec.model_dump(mode="json") for spec in oracles],
            "checks": checks,
            "uncovered": uncovered,
            "files": files,
        }
    )
    return CheckPackage.model_validate(data), frozenset(held)


async def load_resumed_boundary(
    event_store: EventStore, execution_id: str, *, store_dir: Path | None = None
) -> ResumedBoundary | None:
    """The boundary a resumed run's worker was bound to, or ``None`` when none was admitted.

    ``None`` means the original run never bound a worker to an admitted
    package: the legacy verifier decided it then, and decides it now.
    """
    bound: list[Any] | None = None
    boundary_id = ""
    for version in range(1, _MAX_VERSIONS + 1):
        candidate_id = f"{execution_id}/check_package/v{version}"
        events = await event_store.replay(BOUNDARY_AGGREGATE_TYPE, candidate_id)
        if not events:
            break
        if any(event.type == ACTOR_STARTED for event in events):
            bound, boundary_id = events, candidate_id
    if bound is None:
        return None
    frozen = next((event for event in bound if event.type == PACKAGE_FROZEN), None)
    admission = next((event for event in bound if event.type == ADMISSION_COMPLETED), None)
    if frozen is None or admission is None or admission.data.get("verdict") != "admitted":
        return None
    reference = cited_reference(frozen.data)
    if not isinstance(reference, str) or not _REFERENCE.fullmatch(reference):
        return None
    covered = tuple(
        sorted(
            {
                key
                for check in (frozen.data.get("manifest") or {}).get("checks") or ()
                for key in check.get("criterion_keys") or ()
            }
        )
    )
    revealed: dict[str, set[str]] = {}
    for event in bound:
        if event.type == CASE_REVEALED:
            revealed.setdefault(str(event.data.get("check_id")), set()).add(
                str(event.data.get("case_id"))
            )
    started = next(event for event in bound if event.type == ACTOR_STARTED)
    assignment = started.data.get("check_package_assignment")
    store = store_dir or default_store_dir(execution_id)
    base = {
        "execution_id": execution_id,
        "boundary_id": boundary_id,
        "reference": reference,
        "covered": covered,
        "store_dir": store,
        "base_tree_digest": admission.data.get("base_tree_digest"),
        "check_tiers": admission.data.get("check_tiers"),
        "revealed": revealed,
        "assignment": assignment if isinstance(assignment, str) else None,
    }
    live = live_state(execution_id)
    if live is not None and live.package is not None and live.package.reference == reference:
        # Same process: the admitted package, held-out cases included, is
        # still in memory and is the one the journal committed to.
        return ResumedBoundary(package=live.package, source="memory", **base)
    try:
        record = json.loads((store / "packages" / f"{reference}.json").read_text("utf-8"))
        if cited_reference(record) != reference:
            raise ValueError("record names another package")
        package, held = visible_package(record)
    except Exception as exc:  # noqa: BLE001 - an unreadable record leaves the criteria undecided
        log.warning("boundary.resume.record_unavailable", error_type=type(exc).__name__)
        return ResumedBoundary(package=None, reason=PACKAGE_UNAVAILABLE, **base)
    # The store is as writable as the workspace; the journal recorded the
    # manifest and the admission before the worker started (R4-S1).
    manifest = frozen.data.get("manifest") or {}
    mismatch = record_mismatch(record, manifest, admission.data)
    if mismatch is not None:
        log.warning("boundary.resume.record_tampered", mismatch=mismatch)
        return ResumedBoundary(package=None, reason=PACKAGE_RECORD_TAMPERED, **base)
    held |= {
        str(spec.get("check_id"))
        for spec in manifest.get("oracles") or ()
        if spec.get("held_out_count")
    }
    return ResumedBoundary(package=package, held_out_checks=frozenset(held), **base)


def _record_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """The frozen manifest (``CheckPackage.manifest_summary``) recomputed from a record.

    File digests are recomputed from the stored content, and case counts and
    held-out counts from the stored cases, so an edit to either shows.
    """
    data = record["package"]
    oracles = data.get("oracles") or ()
    manifest: dict[str, Any] = {
        "schema_version": data["schema_version"],
        "package_commitment": record["package_commitment"],
        "commitment_scheme": record["commitment_scheme"],
        "seed_digest": data["seed_digest"],
        "input_digest": data["input_digest"],
        "generated_at": data["generated_at"],
        "generator": data["generator"],
        "criterion_keys": list(data["criterion_keys"]),
        "checks": [
            {
                "check_id": check["check_id"],
                "role": check["role"],
                "criterion_keys": sorted({link["criterion_key"] for link in check["assertions"]}),
                "assertion_ids": [link["assertion_id"] for link in check["assertions"]],
            }
            for check in data["checks"]
        ],
        "files": [
            {"path": item["path"], "held_out_redacted": True}
            if item["path"] == ORACLE_DATA_PATH
            else {
                "path": item["path"],
                "sha256": sha256_bytes(item["content"].encode("utf-8")),
                "size": len(item["content"].encode("utf-8")),
            }
            for item in data["files"]
        ],
        "base_files": list(data["base_files"]),
        "scratch_paths": list(data["scratch_paths"]),
        "uncovered": list(data["uncovered"]),
    }
    if oracles:
        manifest["binding_grammar"] = data.get("binding_grammar")
        manifest["oracles"] = [
            {
                "check_id": spec["check_id"],
                "criterion_key": spec["criterion_key"],
                "call_kind": spec["call_kind"],
                "params": list(spec["params"]),
                "default_symbol": spec["default_binding"]["symbol"],
                "default_resolves": spec["default_resolves"],
                "case_count": len(spec["cases"]),
                "held_out_count": sum(1 for case in spec["cases"] if case.get("held_out")),
            }
            for spec in oracles
        ]
    return manifest


def _manifest_digest(manifest: dict[str, Any]) -> str:
    """Digest of a manifest with ``generated_at`` in one spelling (``Z`` or ``+00:00``)."""
    stamp = manifest.get("generated_at")
    normalized = {
        **manifest,
        "generated_at": datetime.fromisoformat(stamp).isoformat() if stamp else stamp,
    }
    return sha256_bytes(canonical_json_bytes(normalized))


def record_mismatch(
    record: dict[str, Any], manifest: dict[str, Any], admission: dict[str, Any]
) -> str | None:
    """Where the stored record disagrees with the journal, or ``None`` when it agrees.

    The journal holds, from before the worker started: the manifest (every
    case id per check through its assertion ids, case and held-out counts per
    oracle, and the digest of every file but the oracle data), and the
    admission receipt (the held-out flag of every case the base run saw).
    Either may catch a held-out case that was deleted, added, or made
    visible. Returns a short label, never a value from the package.
    """
    try:
        if _manifest_digest(_record_manifest(record)) != _manifest_digest(dict(manifest)):
            return "manifest_digest"
        cases = {
            spec["check_id"]: {
                case["case_id"]: bool(case.get("held_out")) for case in spec["cases"]
            }
            for spec in record["package"].get("oracles") or ()
        }
        assertion_ids = {
            check["check_id"]: list(check["assertion_ids"]) for check in manifest["checks"]
        }
        for spec in record["package"].get("oracles") or ():
            ids = [f"{spec['check_id']}.{case['case_id']}" for case in spec["cases"]]
            if ids != assertion_ids.get(spec["check_id"]):
                return "case_ids"
        for check in admission.get("checks") or ():
            result = check.get("oracle_result") or {}
            if not result.get("cases"):
                continue
            seen = {case.get("case_id"): bool(case.get("held_out")) for case in result["cases"]}
            if cases.get(check.get("check_id")) != seen:
                return "held_out_cases"
        listed = {(item["check_id"], item["case_id"]) for item in record.get("held_out") or ()}
        flagged = {
            (check, case) for check, flags in cases.items() for case, held in flags.items() if held
        }
        if listed != flagged:
            return "held_out_list"
    except (KeyError, TypeError, ValueError, AttributeError):
        return "record_shape"
    return None


def _undecided(key: str, reason: str) -> CriterionVerdict:
    return CriterionVerdict(key, PackageCriterionStatus.INDETERMINATE, CheckTier.A, reason)


async def decide_resumed(
    boundary: ResumedBoundary,
    *,
    seed: Seed,
    candidate: Path,
    settings: CheckPackageSettings,
    declared: dict[str, list[Any]] | None = None,
) -> BoundaryVerdict:
    """The package's per-criterion verdicts on ``candidate`` (see the module docstring)."""
    keys = seed_criterion_keys(seed)
    covered = set(boundary.covered)
    if boundary.package is None:
        verdicts = {
            key: (
                _undecided(key, boundary.reason or PACKAGE_UNAVAILABLE)
                if key in covered
                else CriterionVerdict(
                    key, PackageCriterionStatus.UNCOVERED, CheckTier.U, "uncovered"
                )
            )
            for key in keys
        }
        return _verdict(boundary, verdicts)
    package = boundary.package
    snapshot = boundary.store_dir / BASE_SNAPSHOT_DIR
    manifest_path = boundary.store_dir / BASE_MANIFEST_FILE
    interpreter = resolve_check_interpreter(candidate)
    env = scrubbed_check_environment()
    assignments, _results = await assign_tiers(
        package,
        artifact=candidate,
        base=snapshot if snapshot.is_dir() else None,
        declared=declared,
        base_manifest=json.loads(manifest_path.read_text("utf-8"))
        if manifest_path.is_file()
        else None,
        expected_base_digest=boundary.base_tree_digest,
        admitted_tiers=boundary.check_tiers,
        run_options={"env": env, "interpreter": interpreter.path},
    )
    bound = await verify_with_bindings(
        package,
        candidate,
        assignments,
        timeout_seconds=settings.check_timeout_seconds,
        env=env,
        interpreter=interpreter.path,
        interpreter_source=interpreter.source,
    )
    verification = retire_revealed(bound.effective, boundary.revealed)
    verdicts = criterion_verdicts(package, verification, assignments=assignments)
    lost = {
        link.criterion_key
        for check in package.checks
        if check.check_id in boundary.held_out_checks
        for link in check.assertions
    } | {item.criterion_key for item in package.uncovered if item.reason == HELD_OUT_UNAVAILABLE}
    for key in lost:
        item = verdicts.get(key)
        if item is None or item.status is not PackageCriterionStatus.FAIL:
            verdicts[key] = _undecided(key, HELD_OUT_UNAVAILABLE)
    for key in covered - set(verdicts):
        verdicts[key] = _undecided(key, PACKAGE_UNAVAILABLE)
    return _verdict(boundary, verdicts)


def _verdict(boundary: ResumedBoundary, verdicts: dict[str, CriterionVerdict]) -> BoundaryVerdict:
    overall = artifact_verdict(item.status for item in verdicts.values())
    return BoundaryVerdict(
        verdict=overall.value,
        reasons=(f"resumed:{boundary.source}",),
        boundary_id=boundary.boundary_id,
        package_sha256=boundary.reference,
        criteria={key: item.status for key, item in verdicts.items()},
        verdicts=verdicts,
        artifact_verdict=overall,
    )


class ResumedCheckPackageAuthority:
    """The acceptance authority of a resumed arm-on run (installed as the runner's)."""

    def __init__(
        self,
        boundary: ResumedBoundary,
        settings: CheckPackageSettings,
        *,
        event_store: EventStore,
        candidate_checkout: Path,
    ) -> None:
        self.boundary = boundary
        self._settings = settings
        self._event_store = event_store
        self._candidate = candidate_checkout
        self.outcome: AuthorityOutcome | None = None

    async def __call__(self, *, seed: Seed, execution_id: str, parallel_result: Any) -> Any:
        if self.outcome is not None:
            return parallel_result
        # The live arm-on rule (R4-A1): only a root that succeeded, or that
        # only the package gate failed, is an attempt the package may accept.
        # A runtime failure, a failed verify command, or a resumed attempt the
        # (ungated) legacy verifier rejected is never accepted here.
        legacy = existing_outcomes_from_results(parallel_result, gated=True)
        legacy_accepted = bool(legacy) and all(item.passed for item in legacy.values())
        try:
            keys = seed_criterion_keys(seed)
            declared: dict[str, list[Any]] = {}
            for result in getattr(parallel_result, "results", ()) or ():
                index = getattr(result, "ac_index", -1)
                entries = _declared_from(result) if 0 <= index < len(keys) else []
                if entries:
                    declared[keys[index]] = entries
            verdict = await decide_resumed(
                self.boundary,
                seed=seed,
                candidate=self._candidate.resolve(),
                settings=self._settings,
                declared=declared,
            )
            reconciliation = reconcile_acceptance(
                keys,
                verdict.verdicts,
                legacy,
                existing_run_accepted=bool(parallel_result.all_succeeded),
            )
            await BoundaryLedger(self._event_store).record_acceptance_resumed(
                self.boundary.boundary_id,
                package_sha256=self.boundary.reference,
                payload={
                    "schema_version": RESUMED_SCHEMA,
                    "source": self.boundary.source,
                    "held_out_checks": sorted(self.boundary.held_out_checks),
                    **reconciliation.to_dict(),
                },
            )
            self.outcome = AuthorityOutcome(
                legacy_accepted, verdict=verdict, reconciliation=reconciliation, legacy=legacy
            )
            return apply_reconciliation(parallel_result, reconciliation)
        except Exception as exc:  # noqa: BLE001 - no package decision: undecided, never legacy
            log.warning(
                "boundary.resume.failed",
                execution_id=execution_id,
                boundary_id=self.boundary.boundary_id,
                error_type=type(exc).__name__,
            )
            self.outcome = AuthorityOutcome(
                legacy_accepted, error=type(exc).__name__, legacy=legacy
            )
            return _fail_attempted(parallel_result, legacy)
        finally:
            live = live_state(execution_id)
            if self.outcome is not None and live is not None:
                reveal_commitment_salts(live)

    def render(self) -> list[str]:
        """Lines for the person running the command."""
        outcome = self.outcome
        if outcome is None:
            return []
        if outcome.error is not None:
            return [
                f"Check package could not be recomputed on resume ({outcome.error}); "
                "every attempted criterion is undecided."
            ]
        lines = [
            "Check package recomputed on resume "
            + (
                "from memory (held-out cases available)."
                if self.boundary.source == "memory"
                else "from the visible cases (held-out cases unavailable)."
            )
        ]
        if outcome.verdict is not None:
            lines.extend(render_verdict(outcome.verdict))
        if outcome.reconciliation is not None:
            lines.extend(render_reconciliation(outcome.reconciliation))
        return lines


__all__ = [
    "HELD_OUT_UNAVAILABLE",
    "PACKAGE_RECORD_TAMPERED",
    "PACKAGE_UNAVAILABLE",
    "ResumedBoundary",
    "ResumedCheckPackageAuthority",
    "decide_resumed",
    "load_resumed_boundary",
    "record_mismatch",
    "visible_package",
]
