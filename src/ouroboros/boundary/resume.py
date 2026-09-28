"""Resume with the check package on: recompute the package decision after the controller died.

A target can kill the controller (it is a child of it), or the controller can
die for any other reason, after the worker stopped and before the package
decided. The resumed run must not fall back to the legacy verifier for the
criteria the package covers: that would turn "kill the controller" into
"skip the package". So on resume the package decision is recomputed on the
current workspace, with no model call:

- in the same process (the run's task died, the process did not) the
  admitted package is still in memory (``run_wiring.live_state``): it is
  re-derived from there after checking it against the frozen package id, and
  the held-out cases decide as usual;
- in another process the held-out cases are gone (they were never written to
  disk). The package is re-derived from the stored record with its visible
  cases only. A visible failure fails the criterion; otherwise a criterion
  whose check had held-out cases is indeterminate (``held_out_unavailable``),
  because passing only the visible cases is exactly what killing the
  controller would buy. A criterion whose cases were all visible is decided
  normally. The store is as writable as the workspace, so the record is used
  only when it agrees with what the journal recorded before the worker
  started: the SHA-256 of the record's bytes (``record_sha256`` on the frozen
  event, which covers the visible case values), then the manifest recomputed
  from the record (case ids, case and held-out counts, file digests) and the
  held-out flag of every case the admission run saw (``record_mismatch``).
  Otherwise every covered criterion is indeterminate
  (``package_record_tampered``).

Which criteria the package covers, and which of them had held-out cases, is
decided once, before any held-out case is dropped, by one recovery plan
(``recovery_plan``) read from the journal records written before the worker
started: the frozen manifest (every check, its role and criteria, and each
oracle's held-out count) and the admission receipt (tier ``C`` exclusions),
under the live coverage rule (``per_check.criteria_without_admitted_check``).
A check that had only held-out cases still marks its criteria, even though
the visible package has no such check any more. A manifest or receipt that is
missing, malformed, shortened or disagrees with the frozen package id makes
the whole boundary undecidable (``boundary_record_missing``), never
uncovered.

Whether the package was on is itself in the journal: the run's first
boundary event (``boundary.check_package.enabled`` on the run's execution
id) is written before construction. With that record, a journal that no
longer shows a bound, admitted version (or a construction failure the worker
was bound to) is undecidable: every criterion is indeterminate
(``boundary_record_missing``), because deleting the boundary's records must
not buy a legacy decision. Without it and without any boundary version the
run had the package off and the legacy verifier decides. The journal is as
writable as the workspace; removing every record of the run, the enabled
record included, still reads as "off" (a documented residual).

A covered criterion without a package decision (the record or the journal is
unreadable, or the record disagrees with the journal) is indeterminate: not
accepted, a non-zero exit, never a legacy decision. Uncovered criteria keep
the rule of the live run: the legacy verifier decides them (legacy-decided),
and one it has no evidence for stays unverified and is accepted when
attempted. A check excluded at admission covers nothing. What counts
as an attempt is the live rule with the check package on
(``existing_outcomes_from_results(..., gated=True)``): a root that failed for
any reason other than the package gate (a failed session, a failed verify
command, or a legacy rejection of a resumed attempt, which runs without the
gate) is not accepted, whatever the package says. The recomputed decision is
recorded as ``boundary.acceptance.resumed``; the frozen boundary's single-shot
records (final bindings, candidate verification) are not written again.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, replace
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
    AUTHORITY_ERROR_PREFIX,
    AuthorityOutcome,
    _declared_from,
    _fail_attempted,
    apply_reconciliation,
    decide_without_package,
    existing_outcomes_from_results,
)
from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.binding_flow import (
    BASE_MANIFEST_FILE,
    BASE_SNAPSHOT_DIR,
    assign_tiers,
    verify_with_bindings,
)
from ouroboros.boundary.check_env import (
    INTERPRETER_CHANGED,
    CheckInterpreter,
    resolve_check_interpreter,
)
from ouroboros.boundary.events import (
    ACTOR_STARTED,
    ADMISSION_COMPLETED,
    CONSTRUCTION_FAILED,
    PACKAGE_FROZEN,
    ResumedPayload,
    RunContract,
    boundary_version_id,
)
from ouroboros.boundary.ledger import BoundaryLedger, BoundaryOrderError
from ouroboros.boundary.oracle import ORACLE_DATA_PATH, OracleSpec, is_oracle_file
from ouroboros.boundary.package import (
    CHECK_PACKAGE_SCHEMA,
    CheckPackage,
    CheckRole,
    canonical_json_bytes,
    oracle_files,
    seed_criterion_keys,
    sha256_bytes,
)
from ouroboros.boundary.per_check import (
    EXCLUSION_REASONS,
    criteria_without_admitted_check,
    exclusion_reason_for_role,
)
from ouroboros.boundary.run_wiring import (
    BoundaryVerdict,
    default_store_dir,
    forget_live_state,
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
BOUNDARY_RECORD_MISSING = "boundary_record_missing"
_PACKAGE_ID = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ResumedBoundary:
    """What a resumed run can recover of the boundary its worker was bound to."""

    execution_id: str
    boundary_id: str
    package_id: str
    """Empty for an undecidable boundary (``BOUNDARY_RECORD_MISSING``)."""
    covered: tuple[str, ...] | None
    """The criteria admitted checks cover; ``None`` when unknown (every criterion)."""
    store_dir: Path
    package: CheckPackage | None
    held_out_checks: frozenset[str] = frozenset()
    """Every admitted check that had held-out cases (original ids, from the plan)."""
    held_out_criteria: frozenset[str] = frozenset()
    """Covered criteria with an admitted check that had held-out cases (from the plan)."""
    criterion_keys: tuple[str, ...] | None = None
    """The criterion keys the frozen manifest names (``None``: undecidable boundary)."""
    base_tree_digest: str | None = None
    check_tiers: dict[str, str] | None = None
    source: str = "record"
    reason: str | None = None
    interpreter_sha256: str | None = None
    """The pinned interpreter's digest from the admission receipt."""
    interpreter_realpath_sha256: str | None = None
    """The digest of the pinned interpreter's real path from the admission receipt."""
    interpreter: CheckInterpreter | None = None
    """The run's pinned interpreter, when the package is still in memory."""
    contract: RunContract | None = None
    """The settings the run started with (its enabled record), never the live config."""


@dataclass(frozen=True, slots=True)
class _PlanLink:
    criterion_key: str


@dataclass(frozen=True, slots=True)
class _PlanCheck:
    check_id: str
    role: CheckRole
    assertions: tuple[_PlanLink, ...]


@dataclass(frozen=True, slots=True)
class _PlanChecks:
    criterion_keys: tuple[str, ...]
    checks: tuple[_PlanCheck, ...]


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    """Per criterion, what the journal recorded before the worker started."""

    criterion_keys: tuple[str, ...]
    covered: frozenset[str]
    """Criteria at least one admitted check covers, under the live coverage rule."""
    held_out_checks: frozenset[str]
    """Admitted checks whose oracle had held-out cases (original check ids)."""
    held_out_criteria: frozenset[str]
    """Covered criteria with at least one admitted check that had held-out cases."""


class _Malformed(ValueError):
    """A journal record does not have the shape the frozen package wrote."""


def _strings(value: object) -> tuple[str, ...]:
    """A non-empty list of distinct non-empty strings, or ``_Malformed``."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise _Malformed("expected distinct non-empty strings")
    return tuple(value)


def _plan_checks(manifest: Mapping[str, Any], keys: frozenset[str]) -> list[_PlanCheck]:
    raw = manifest["checks"]
    if not isinstance(raw, list):
        raise _Malformed("checks")
    checks: list[_PlanCheck] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("check_id"), str):
            raise _Malformed("check")
        linked = _strings(item["criterion_keys"])
        _strings(item["assertion_ids"])
        if not set(linked) <= keys or not item["check_id"]:
            raise _Malformed("check links")
        checks.append(
            _PlanCheck(
                item["check_id"],
                CheckRole(item["role"]),
                tuple(_PlanLink(key) for key in linked),
            )
        )
    if len({check.check_id for check in checks}) != len(checks):
        raise _Malformed("duplicate check ids")
    return checks


# Tiers a check can have when admission records it: the base tier of its
# default binding (``admission.admission_tiers``) or ``C`` (excluded).
_ADMISSION_TIERS = frozenset({CheckTier.A.value, CheckTier.U.value, CheckTier.C.value})


def _admitted_exclusions(
    manifest: Mapping[str, Any], admission: Mapping[str, Any], by_id: Mapping[str, _PlanCheck]
) -> frozenset[str]:
    """The excluded checks of an admission record that admission could have written.

    ``admission.admit_check_package`` followed by ``per_check.per_check_admission``
    only ever records an admitted package as: one result per check of the
    frozen manifest, with the manifest's role; a tier (``A``, ``U`` or ``C``)
    for every check; no protected-byte mutation and an unchanged base; every
    check that is not excluded met its contract on the base (``expected``);
    every excluded check (tier ``C``) is exactly an ``excluded_checks`` entry,
    was ``violated`` for the one base reason its role allows, and carries
    that role's exclusion reason; and at least one check is not excluded.
    Anything else is ``_Malformed``: the record was not written by admission,
    so nothing in it may decide which criteria the package covers.
    """
    if admission.get("verdict") != "admitted":
        raise _Malformed("admission verdict")
    if admission.get("seed_digest") != manifest.get("seed_digest"):
        raise _Malformed("admission seed")
    if admission.get("protected_bytes_mutated") is not False or admission.get(
        "base_tree_digest"
    ) != admission.get("base_tree_digest_after"):
        raise _Malformed("admission base")
    tiers = admission["check_tiers"]
    if (
        not isinstance(tiers, Mapping)
        or set(tiers) != set(by_id)
        or not set(tiers.values()) <= _ADMISSION_TIERS
    ):
        raise _Malformed("check tiers")
    excluded = frozenset(check_id for check_id, tier in tiers.items() if tier == CheckTier.C.value)
    recorded_exclusions = admission.get("excluded_checks") or {}
    if not isinstance(recorded_exclusions, Mapping) or set(recorded_exclusions) != excluded:
        raise _Malformed("excluded checks")
    if excluded == set(by_id):
        raise _Malformed("every check excluded")
    results = admission["checks"]
    if not isinstance(results, list) or not all(isinstance(item, Mapping) for item in results):
        raise _Malformed("admission checks")
    ids = [item.get("check_id") for item in results]
    if len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise _Malformed("admission check ids")
    for item in results:
        check = by_id[item["check_id"]]
        if item.get("role") != check.role.value:
            raise _Malformed("admission check role")
        if check.check_id in excluded:
            allowed = exclusion_reason_for_role(check.role)
            if (
                item.get("status") != "violated"
                or EXCLUSION_REASONS.get(str(item.get("reason"))) != allowed
                or recorded_exclusions[check.check_id] != allowed
            ):
                raise _Malformed("excluded check")
        elif item.get("status") != "expected":
            raise _Malformed("admitted check")
    return excluded


def _plan(manifest: object, admission: object, package_id: str) -> RecoveryPlan:
    if not isinstance(manifest, Mapping) or not isinstance(admission, Mapping):
        raise _Malformed("records")
    if manifest.get("package_id") != package_id or admission.get("package_id") != package_id:
        raise _Malformed("package id")
    keys = _strings(manifest["criterion_keys"])
    checks = _plan_checks(manifest, frozenset(keys))
    uncovered_raw = manifest["uncovered"]
    if not isinstance(uncovered_raw, list) or not all(
        isinstance(item, Mapping) for item in uncovered_raw
    ):
        raise _Malformed("uncovered")
    uncovered = [item["criterion_key"] for item in uncovered_raw]
    linked = {link.criterion_key for check in checks for link in check.assertions}
    # The package invariant: every criterion is linked or uncovered, never both.
    if (
        len(set(uncovered)) != len(uncovered)
        or linked & set(uncovered)
        or linked | set(uncovered) != set(keys)
    ):
        raise _Malformed("criterion coverage")
    by_id = {check.check_id: check for check in checks}
    oracles = manifest.get("oracles", [])
    if not isinstance(oracles, list):
        raise _Malformed("oracles")
    held: set[str] = set()
    for spec in oracles:
        check = by_id.get(spec.get("check_id")) if isinstance(spec, Mapping) else None
        count = spec.get("held_out_count") if isinstance(spec, Mapping) else None
        if (
            check is None
            or spec.get("criterion_key") not in {link.criterion_key for link in check.assertions}
            or type(count) is not int
            or count < 0
        ):
            raise _Malformed("oracle")
        if count:
            held.add(check.check_id)
    excluded = _admitted_exclusions(manifest, admission, by_id)
    lost = criteria_without_admitted_check(_PlanChecks(keys, tuple(checks)), excluded)
    admitted = [check for check in checks if check.check_id not in excluded]
    covered = {link.criterion_key for check in admitted for link in check.assertions} - set(lost)
    held_criteria = {
        link.criterion_key
        for check in admitted
        if check.check_id in held
        for link in check.assertions
        if link.criterion_key in covered
    }
    return RecoveryPlan(
        criterion_keys=keys,
        covered=frozenset(covered),
        held_out_checks=frozenset(held - excluded),
        held_out_criteria=frozenset(held_criteria),
    )


def recovery_plan(manifest: object, admission: object, package_id: str) -> RecoveryPlan | None:
    """The criterion-level recovery plan, or ``None`` when a record is malformed.

    ``manifest`` is the frozen event's manifest and ``admission`` the admission
    event's data; both were written before the worker started. ``None`` makes
    the boundary undecidable: every criterion is indeterminate.
    """
    try:
        return _plan(manifest, admission, package_id)
    except (_Malformed, KeyError, TypeError, ValueError, AttributeError):
        return None


def visible_package(record: dict[str, Any]) -> CheckPackage:
    """Re-derive, in memory, the stored package without its held-out cases.

    A check left with no visible case is dropped; its criterion is listed as
    uncovered only when no other check links it, so the package stays valid.
    Which criteria had held-out cases is the recovery plan's, not this
    package's: it was fixed before anything was dropped.
    """
    data = copy.deepcopy(record["package"])
    dropped: dict[str, str] = {}
    oracles: list[OracleSpec] = []
    for spec in data.get("oracles") or ():
        visible = [case for case in spec["cases"] if not case.get("held_out")]
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
    return CheckPackage.model_validate(data)


async def load_resumed_boundary(
    event_store: EventStore, execution_id: str, *, store_dir: Path | None = None
) -> ResumedBoundary | None:
    """The boundary a resumed run's worker was bound to, or ``None`` when legacy decides.

    ``None`` means the original run had the check package off (no
    ``boundary.check_package.enabled`` record and no boundary version), or its
    worker was bound to a version sealed without a package (construction
    failed): the legacy verifier decided it then, and decides it now. With
    the package on, any other shape of the journal (no bound version, or a
    bound version whose seal, admission or package id is missing or
    malformed) is an undecidable boundary: every criterion is undecided
    (``boundary_record_missing``). Raises ``BoundaryOrderError`` without an
    execution id.
    """
    if not execution_id:
        raise BoundaryOrderError("a resumed run needs its execution id to find its boundary")
    ledger = BoundaryLedger(event_store)
    store = store_dir or default_store_dir(execution_id)
    try:
        contract = await ledger.run_contract(execution_id)
    except BoundaryOrderError:
        return _undecidable(execution_id, store, "run_contract")
    # Every version the journal holds for the run, not a count from v1 that a
    # gap would stop (``BoundaryLedger.run_versions``).
    versions = await ledger.run_versions(execution_id)
    if contract is None:
        # Never on: legacy. Versions without the run's enabled record lost the
        # settings the run started with: undecided, never the live config.
        return _undecidable(execution_id, store, "run_contract") if versions else None
    if list(versions) != list(range(1, len(versions) + 1)):
        # The product writes v1, v2, ... in order: a gap is a journal it did not write.
        return _undecidable(execution_id, store, "version_gap")
    bound: list[Any] | None = None
    boundary_id = ""
    for version, events in versions.items():
        if any(event.type == ACTOR_STARTED for event in events):
            bound, boundary_id = events, boundary_version_id(execution_id, version)
    if bound is None:
        return _undecidable(execution_id, store, "no_bound_version")
    frozen = next((event for event in bound if event.type == PACKAGE_FROZEN), None)
    failed = next((event for event in bound if event.type == CONSTRUCTION_FAILED), None)
    admission = next((event for event in bound if event.type == ADMISSION_COMPLETED), None)
    if frozen is None and failed is not None:
        return None
    if frozen is None or admission is None or admission.data.get("verdict") != "admitted":
        return _undecidable(execution_id, store, "bound_version_not_admitted", boundary_id)
    package_id = frozen.data.get("package_id")
    if not isinstance(package_id, str) or not _PACKAGE_ID.fullmatch(package_id):
        return _undecidable(execution_id, store, "bound_package_id", boundary_id)
    # One plan, fixed before any held-out case is dropped: which criteria the
    # admitted checks cover (tier ``C`` exclusions and the reproduction rule
    # applied) and which of them had held-out cases.
    plan = recovery_plan(frozen.data.get("manifest"), admission.data, package_id)
    if plan is None:
        return _undecidable(execution_id, store, "recovery_plan", boundary_id)
    base = {
        "execution_id": execution_id,
        "boundary_id": boundary_id,
        "package_id": package_id,
        "covered": tuple(sorted(plan.covered)),
        "held_out_checks": plan.held_out_checks,
        "held_out_criteria": plan.held_out_criteria,
        "criterion_keys": plan.criterion_keys,
        "store_dir": store,
        "base_tree_digest": admission.data.get("base_tree_digest"),
        "check_tiers": admission.data.get("check_tiers"),
        "interpreter_sha256": admission.data.get("interpreter_sha256"),
        "interpreter_realpath_sha256": admission.data.get("interpreter_realpath_sha256"),
        "contract": contract,
    }
    live = live_state(execution_id)
    if live is not None and live.package is not None and live.package.package_id == package_id:
        # Same process: the admitted package, held-out cases included, is
        # still in memory and is the one the journal froze.
        return ResumedBoundary(
            package=live.package, source="memory", interpreter=live.interpreter, **base
        )
    try:
        raw = (store / "packages" / f"{package_id}.json").read_bytes()
        record = json.loads(raw.decode("utf-8"))
        if record.get("package_id") != package_id:
            raise ValueError("record names another package")
        package = visible_package(record)
    except Exception as exc:  # noqa: BLE001 - an unreadable record leaves the criteria undecided
        log.warning("boundary.resume.record_unavailable", error_type=type(exc).__name__)
        return ResumedBoundary(package=None, reason=PACKAGE_UNAVAILABLE, **base)
    # The store is as writable as the workspace; the journal recorded the
    # manifest and the admission before the worker started.
    manifest = frozen.data.get("manifest") or {}
    recorded = frozen.data.get("record_sha256")
    # Journals written before the record digest was recorded skip this check.
    mismatch = (
        "record_digest"
        if recorded is not None and sha256_bytes(raw) != recorded
        else record_mismatch(record, manifest, admission.data)
    )
    if mismatch is not None:
        log.warning("boundary.resume.record_tampered", mismatch=mismatch)
        return ResumedBoundary(package=None, reason=PACKAGE_RECORD_TAMPERED, **base)
    return ResumedBoundary(package=package, **base)


def _record_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """The frozen manifest (``CheckPackage.manifest_summary``) recomputed from a record.

    File digests are recomputed from the stored content, and case counts and
    held-out counts from the stored cases, so an edit to either shows.
    """
    data = record["package"]
    oracles = data.get("oracles") or ()
    manifest: dict[str, Any] = {
        "schema_version": data["schema_version"],
        "package_id": record["package_id"],
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
    except (KeyError, TypeError, ValueError, AttributeError):
        return "record_shape"
    return None


def _undecidable(
    execution_id: str, store: Path, detail: str, boundary_id: str = ""
) -> ResumedBoundary:
    """A boundary the journal says existed but cannot be recovered: every criterion undecided."""
    log.warning("boundary.resume.boundary_record_missing", detail=detail)
    return ResumedBoundary(
        execution_id=execution_id,
        boundary_id=boundary_id,
        package_id="",
        covered=None,
        store_dir=store,
        package=None,
        reason=BOUNDARY_RECORD_MISSING,
    )


def _undecided(key: str, reason: str) -> CriterionVerdict:
    return CriterionVerdict(key, PackageCriterionStatus.INDETERMINATE, CheckTier.A, reason)


def _uncovered(key: str) -> CriterionVerdict:
    """Not the package's: the legacy verifier decides it."""
    return CriterionVerdict(key, PackageCriterionStatus.UNCOVERED, CheckTier.U, "uncovered")


async def decide_resumed(
    boundary: ResumedBoundary,
    *,
    seed: Seed,
    candidate: Path,
    declared: dict[str, list[Any]] | None = None,
) -> BoundaryVerdict:
    """The package's per-criterion verdicts on ``candidate`` (see the module docstring)."""
    keys = seed_criterion_keys(seed)
    if boundary.criterion_keys is not None and set(boundary.criterion_keys) != set(keys):
        # The manifest names other criteria than this Seed: nothing is known.
        boundary = replace(boundary, covered=None, package=None, reason=BOUNDARY_RECORD_MISSING)
    if boundary.package is not None and boundary.contract is None:
        # Checks would run with settings the run did not start with.
        boundary = replace(boundary, package=None, reason=BOUNDARY_RECORD_MISSING)
    covered = set(keys) if boundary.covered is None else set(boundary.covered)
    if boundary.package is None or boundary.contract is None:
        verdicts = {
            key: (
                _undecided(key, boundary.reason or PACKAGE_UNAVAILABLE)
                if key in covered
                else _uncovered(key)
            )
            for key in keys
        }
        return _verdict(boundary, verdicts)
    package = boundary.package
    snapshot = boundary.store_dir / BASE_SNAPSHOT_DIR
    manifest_path = boundary.store_dir / BASE_MANIFEST_FILE
    interpreter = boundary.interpreter or _repinned(boundary, candidate)
    if interpreter is None:
        return _verdict(
            boundary,
            {
                key: (_undecided(key, INTERPRETER_CHANGED) if key in covered else _uncovered(key))
                for key in keys
            },
        )
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
        run_options={"interpreter": interpreter},
    )
    bound = await verify_with_bindings(
        package,
        candidate,
        assignments,
        timeout_seconds=boundary.contract.check_timeout_seconds,
        interpreter=interpreter,
    )
    verification = bound.effective
    computed = criterion_verdicts(package, verification, assignments=assignments)
    return _verdict(boundary, _apply_plan(boundary, keys, covered, computed))


def _apply_plan(
    boundary: ResumedBoundary,
    keys: tuple[str, ...],
    covered: set[str],
    computed: dict[str, CriterionVerdict],
) -> dict[str, CriterionVerdict]:
    """The recovery plan decides which criteria the package speaks for.

    An uncovered criterion is the legacy verifier's. A covered criterion with
    held-out cases, recomputed from the visible cases only (another process),
    is ``held_out_unavailable`` unless a visible case failed, whether or not
    the visible package still holds one of its checks; any other covered
    criterion the recomputation left without a verdict of its own is
    undecided.
    """
    verdicts: dict[str, CriterionVerdict] = {}
    for key in keys:
        item = computed.get(key)
        failed = item is not None and item.status is PackageCriterionStatus.FAIL
        if key not in covered:
            verdicts[key] = _uncovered(key)
        elif boundary.source == "record" and key in boundary.held_out_criteria and not failed:
            # The plan's held-out rule comes first: whether the visible package
            # still holds a check for the criterion does not matter.
            verdicts[key] = _undecided(key, HELD_OUT_UNAVAILABLE)
        elif item is None or item.status is PackageCriterionStatus.UNCOVERED:
            verdicts[key] = _undecided(key, PACKAGE_UNAVAILABLE)
        else:
            verdicts[key] = item
    return verdicts


def _repinned(boundary: ResumedBoundary, candidate: Path) -> CheckInterpreter | None:
    """The interpreter resolved again in another process, or ``None`` when it is not the pin.

    A resume in another process cannot hold the live run's in-memory pin, so it
    resolves the interpreter once, here, and requires it to be the one the
    admission recorded before the worker started: the same real path and the
    same binary digest. A record without both, or any difference, is
    ``None``: every covered criterion is then undecided
    (``interpreter_changed``), and nothing runs the replacement.
    """
    resolved = resolve_check_interpreter(candidate)
    if (
        not boundary.interpreter_sha256
        or not boundary.interpreter_realpath_sha256
        or resolved.sha256 != boundary.interpreter_sha256
        or resolved.realpath_sha256 != boundary.interpreter_realpath_sha256
    ):
        return None
    return resolved


def _verdict(boundary: ResumedBoundary, verdicts: dict[str, CriterionVerdict]) -> BoundaryVerdict:
    overall = artifact_verdict(item.status for item in verdicts.values())
    return BoundaryVerdict(
        verdict=overall.value,
        reasons=(f"resumed:{boundary.source}",),
        boundary_id=boundary.boundary_id,
        package_id=boundary.package_id or None,
        criteria={key: item.status for key, item in verdicts.items()},
        verdicts=verdicts,
        artifact_verdict=overall,
    )


class ResumedCheckPackageAuthority:
    """The acceptance authority of a resumed run (installed as the runner's)."""

    def __init__(
        self,
        boundary: ResumedBoundary,
        *,
        event_store: EventStore,
        candidate_checkout: Path,
    ) -> None:
        self.boundary = boundary
        self._event_store = event_store
        self._candidate = candidate_checkout
        self.outcome: AuthorityOutcome | None = None

    async def __call__(self, *, seed: Seed, execution_id: str, parallel_result: Any) -> Any:
        if self.outcome is not None:
            return parallel_result
        keys = seed_criterion_keys(seed)
        try:
            return await self._decide(seed, keys, parallel_result)
        except Exception as exc:  # noqa: BLE001 - fail closed below, never open
            log.warning(
                "boundary.resume.failed",
                execution_id=execution_id,
                boundary_id=self.boundary.boundary_id,
                error_type=type(exc).__name__,
            )
            return self._undecided(keys, parallel_result, type(exc).__name__)
        finally:
            live = live_state(execution_id)
            if self.outcome is not None and live is not None:
                forget_live_state(live)

    async def _decide(self, seed: Seed, keys: tuple[str, ...], parallel_result: Any) -> Any:
        # The live rule: only a root that succeeded, or that only the package
        # gate failed, is an attempt the package may accept. A runtime
        # failure, a failed verify command, or a resumed attempt the
        # (ungated) legacy verifier rejected is never accepted here.
        legacy = existing_outcomes_from_results(parallel_result, gated=True)
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
            declared=declared,
        )
        reconciliation = reconcile_acceptance(
            keys,
            verdict.verdicts,
            legacy,
            existing_run_accepted=bool(parallel_result.all_succeeded),
            legacy_decides_unverified=True,
        )
        record = {
            **reconciliation.to_dict(),
            "source": self.boundary.source,
            "held_out_checks": sorted(self.boundary.held_out_checks),
        }
        ledger = BoundaryLedger(self._event_store)
        if self.boundary.package_id:
            await ledger.record_acceptance_resumed(
                self.boundary.boundary_id,
                package_id=self.boundary.package_id,
                payload=ResumedPayload.model_validate(record),
            )
        else:
            # No boundary version can be cited: record it on the run's aggregate.
            await ledger.record_resumed_undecided(
                self.boundary.execution_id,
                payload=ResumedPayload.model_validate({**record, "reason": self.boundary.reason}),
            )
        self.outcome = AuthorityOutcome(
            bool(legacy) and all(item.passed for item in legacy.values()),
            verdict=verdict,
            reconciliation=reconciliation,
            legacy=legacy,
        )
        return apply_reconciliation(parallel_result, reconciliation)

    def _undecided(self, keys: tuple[str, ...], parallel_result: Any, error: str) -> Any:
        """The live authority's fail-closed rule: covered criteria undecided, the rest legacy."""
        try:
            decided, reconciliation, legacy = decide_without_package(
                keys,
                set(keys) if self.boundary.covered is None else set(self.boundary.covered),
                f"{AUTHORITY_ERROR_PREFIX}{error}",
                parallel_result,
                gated=True,
            )
        except Exception:  # noqa: BLE001 - no decision could be built: fail every root
            log.warning("boundary.resume.undecided_failed", error_type=error)
            self.outcome = AuthorityOutcome(False, error=error)
            return _fail_attempted(parallel_result)
        self.outcome = AuthorityOutcome(
            bool(legacy) and all(item.passed for item in legacy.values()),
            reconciliation=reconciliation,
            error=error,
            legacy=legacy,
        )
        return decided

    def render(self) -> list[str]:
        """Lines for the person running the command."""
        outcome = self.outcome
        if outcome is None:
            return []
        if outcome.error is not None:
            lines = [
                f"Check package could not be recomputed on resume ({outcome.error}); "
                "covered criteria are undecided, the legacy verifier decided the rest."
            ]
            if outcome.reconciliation is not None:
                lines.extend(render_reconciliation(outcome.reconciliation))
            return lines
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
    "RecoveryPlan",
    "ResumedBoundary",
    "ResumedCheckPackageAuthority",
    "decide_resumed",
    "load_resumed_boundary",
    "record_mismatch",
    "recovery_plan",
    "visible_package",
]
