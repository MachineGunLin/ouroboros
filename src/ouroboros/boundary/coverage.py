"""Criterion coverage before the worker starts: non-behavioral criteria and replacements.

Two steps of ``run_wiring.prepare_check_package`` that change which criteria
the package covers, both before any worker starts:

- ``strip_criteria``: a non-behavioral criterion (``boundary/behavioral.py``)
  gets no check; any check the constructor linked to it is removed, and it is
  uncovered with reason ``non_behavioral``.
- One replacement call (product policy only). After per-check admission
  (``boundary/per_check.py``), every behavioral criterion without an admitted
  check is a replacement target (``replacement_targets``), with a plain-text
  reason (``why_excluded``) that names no case value. The replacement checks
  are merged with the admitted checks of the earlier version
  (``merge_replacement``) into a new package that goes through the same
  reference check, seal and admission.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from ouroboros.boundary.behavioral import NON_BEHAVIORAL
from ouroboros.boundary.oracle import is_oracle_file
from ouroboros.boundary.oracle_build import assemble_package
from ouroboros.boundary.package import (
    AssertionLink,
    CheckPackage,
    CheckRole,
    CheckSpec,
    PackageFile,
    canonical_json_bytes,
    sha256_bytes,
)
from ouroboros.boundary.per_check import (
    NO_ADMITTED_REPRODUCTION_CHECK,
    PRESERVATION_FAILS_ON_BASE,
    REPRO_PASSES_ON_BASE,
    criteria_without_admitted_check,
)

if TYPE_CHECKING:
    from ouroboros.boundary.oracle import OracleSpec
    from ouroboros.core.seed import Seed

REPLACEMENT_CONFLICT = "replacement_conflict"
_REASON_CHARS = 160

_WHY: dict[str, str] = {
    REPRO_PASSES_ON_BASE: (
        "its reproduction check passes on the base code, so it does not reproduce the bug"
    ),
    PRESERVATION_FAILS_ON_BASE: (
        "its preservation check fails on the base code, so it does not describe behavior "
        "the base code already has"
    ),
    NO_ADMITTED_REPRODUCTION_CHECK: (
        "its reproduction check passes on the base code, so it does not reproduce the bug; "
        "only a preservation check remains, which cannot show the fix"
    ),
    "construction_timeout": "the earlier construction ran out of time before its check",
    "reference_unavailable": "the reference implementation written for its oracle did not run",
    "reference_contradicts_seed_example": (
        "the reference implementation written for its oracle did not reproduce an example "
        "stated in the Seed"
    ),
    "oracle_inconsistent": (
        "every case of its oracle disagreed with the reference implementation written for it"
    ),
    "constructor_omitted": "the earlier reply wrote no check and no reason for it",
}


def why_excluded(reason: str) -> str:
    """Plain-text reason a criterion has no admitted check (never a case value)."""
    known = _WHY.get(reason)
    if known is not None:
        return known
    text = " ".join(reason.split())[:_REASON_CHARS]
    return f"the earlier reply left it without a check ({text})"


def _links(check: CheckSpec) -> set[str]:
    return {link.criterion_key for link in check.assertions}


def _parts(
    package: CheckPackage, keep: Any
) -> tuple[list[tuple[OracleSpec, CheckRole]], list[CheckSpec], list[PackageFile]]:
    """Oracles, script checks and script files of ``package`` for which ``keep(check)`` holds."""
    roles = {check.check_id: check.role for check in package.checks}
    oracles = [
        (spec, roles[spec.check_id])
        for spec in package.oracles
        if keep(next(c for c in package.checks if c.check_id == spec.check_id))
    ]
    oracle_ids = {spec.check_id for spec in package.oracles}
    scripts = [
        check for check in package.checks if check.check_id not in oracle_ids and keep(check)
    ]
    paths = {arg for check in scripts for arg in check.argv[1:]}
    files = [item for item in package.files if not is_oracle_file(item.path) and item.path in paths]
    return oracles, scripts, files


def strip_criteria(package: CheckPackage, seed: Seed, reasons: Mapping[str, str]) -> CheckPackage:
    """``package`` without any check linked to the criteria in ``reasons`` (key to reason).

    A script check that also covers another criterion keeps its other links.
    The stripped criteria are uncovered with their reason. An unchanged
    package is returned as is.
    """
    if not reasons:
        return package
    touched = any(_links(check) & set(reasons) for check in package.checks)
    stated = {item.criterion_key: item.reason for item in package.uncovered}
    if not touched and all(stated.get(key) == reason for key, reason in reasons.items()):
        return package
    oracles, _scripts, _files = _parts(package, lambda check: not (_links(check) & set(reasons)))
    oracle_ids = {spec.check_id for spec in package.oracles}
    scripts: list[CheckSpec] = []
    for check in package.checks:
        if check.check_id in oracle_ids:
            continue
        links = tuple(link for link in check.assertions if link.criterion_key not in reasons)
        if links:
            scripts.append(
                check if len(links) == len(check.assertions) else _relinked(check, links)
            )
    paths = {arg for check in scripts for arg in check.argv[1:]}
    files = [item for item in package.files if not is_oracle_file(item.path) and item.path in paths]
    return assemble_package(
        seed,
        input_digest=package.input_digest,
        generator=package.generator,
        oracles=oracles,
        script_checks=scripts,
        script_files=files,
        uncovered={
            **{item.criterion_key: item.reason for item in package.uncovered},
            **reasons,
        },
        generated_at=package.generated_at,
    )


def _relinked(check: CheckSpec, links: tuple[AssertionLink, ...]) -> CheckSpec:
    return check.model_copy(update={"assertions": links})


def replacement_targets(
    package: CheckPackage, excluded: Collection[str], skip: Collection[str] = ()
) -> dict[str, str]:
    """Behavioral criteria of an admitted package without an admitted check, with the reason.

    ``excluded`` are the check ids per-check admission excluded; ``skip`` are
    criterion keys never to target (non-behavioral). Criteria the
    constructor left uncovered are targets too, with its reason.
    """
    skipped = set(skip)
    targets = {
        item.criterion_key: item.reason
        for item in package.uncovered
        if item.reason != NON_BEHAVIORAL and item.criterion_key not in skipped
    }
    for key, reason in criteria_without_admitted_check(package, excluded).items():
        if key not in skipped:
            targets[key] = reason
    return {key: targets[key] for key in package.criterion_keys if key in targets}


def merge_replacement(
    package: CheckPackage,
    excluded: Collection[str],
    replacement: CheckPackage,
    targets: Mapping[str, str],
    seed: Seed,
) -> tuple[CheckPackage, dict[str, str]]:
    """The admitted checks of ``package`` plus ``replacement``'s checks for ``targets``.

    Returns the merged package and, per target, why it is still uncovered
    (a target the replacement did not link, or whose check id or file path
    collides with a kept check: ``replacement_conflict``). Excluded checks
    are not carried over.
    """
    excluded_ids = set(excluded)
    kept_oracles, kept_scripts, kept_files = _parts(
        package, lambda check: check.check_id not in excluded_ids
    )
    taken_ids = {check.check_id for check in package.checks}
    taken_paths = {item.path for item in package.files}
    target_keys = set(targets)
    new_oracles, new_scripts, new_files = _parts(
        replacement, lambda check: bool(_links(check)) and _links(check) <= target_keys
    )
    still: dict[str, str] = {}
    oracles = list(kept_oracles)
    for spec, role in new_oracles:
        if spec.check_id in taken_ids:
            still[spec.criterion_key] = REPLACEMENT_CONFLICT
            continue
        oracles.append((spec, role))
    scripts = list(kept_scripts)
    files = list(kept_files)
    by_path = {item.path: item for item in new_files}
    for check in new_scripts:
        paths = set(check.argv[1:])
        if check.check_id in taken_ids or paths & taken_paths:
            for key in _links(check):
                still[key] = REPLACEMENT_CONFLICT
            continue
        scripts.append(check)
        files.extend(by_path[path] for path in sorted(paths) if path in by_path)
    linked = {spec.criterion_key for spec, _role in oracles} | {
        key for check in scripts for key in _links(check)
    }
    replacement_uncovered = {item.criterion_key: item.reason for item in replacement.uncovered}
    for key in targets:
        if key not in linked and key not in still:
            still[key] = replacement_uncovered.get(key, "constructor_omitted")
    uncovered = {
        **{item.criterion_key: item.reason for item in package.uncovered},
        **{key: reason for key, reason in targets.items() if key not in linked},
        **{key: reason for key, reason in still.items() if key not in linked},
    }
    merged = assemble_package(
        seed,
        input_digest=sha256_bytes(
            canonical_json_bytes(
                {"kept": package.input_digest, "replacement": replacement.input_digest}
            )
        ),
        generator=replacement.generator,
        oracles=oracles,
        script_checks=scripts,
        script_files=_unique(files),
        uncovered={key: reason for key, reason in uncovered.items() if key not in linked},
        generated_at=replacement.generated_at,
    )
    return merged, {key: reason for key, reason in still.items() if key not in linked}


def _unique(files: list[PackageFile]) -> list[PackageFile]:
    seen: set[str] = set()
    result = []
    for item in files:
        if item.path not in seen:
            seen.add(item.path)
            result.append(item)
    return result


__all__ = [
    "REPLACEMENT_CONFLICT",
    "merge_replacement",
    "replacement_targets",
    "strip_criteria",
    "why_excluded",
]
