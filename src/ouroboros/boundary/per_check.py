"""Per-check admission: exclude a check that contradicts its role on the base.

Admission runs every check of a package on the base checkout. Two role
violations say something about that one check, not about the rest of the
package:

- a reproduction check that passes on the base does not reproduce the bug
  (``repro_passes_on_base``);
- a preservation check that fails on the base does not describe behavior the
  base already has (``preservation_fails_on_base``).

Under this rule each such check is excluded on its own and the rest of the
package is admitted. Anything else leaves the package unadmitted: a setup,
import or other indeterminate failure, a protected-byte mutation, a
precondition failure (prose-only or unsafe checks, path collisions) or a
package whose every check would be excluded is not admitted.

An excluded check stays in the frozen package and
is marked tier ``C`` ("rejected at admission") in the admission's
``check_tiers``, with its reason in ``excluded_checks`` (check ids and
reasons, never values). ``binding_flow.assign_tiers`` carries tier ``C`` into
the check's assignment, so no verification runs it, and
``acceptance.criterion_verdicts`` ignores it.

Coverage after the exclusions (``criteria_without_admitted_check``):

- a criterion keeps authority only if at least one admitted check covers it;
- a reproduction-type criterion (the package links it to at least one
  reproduction check) needs at least one admitted reproduction check: a
  preservation check alone cannot show the bug is fixed, so the criterion is
  uncovered (``no_admitted_reproduction_check``) even when one remains.

The rule is a pure function of the recorded admission
(``per_check_admission``); ``admission.admit_check_package`` always applies
it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.package import CheckRole
from ouroboros.boundary.receipts import AdmissionResult, CheckStatus, PackageVerdict

REPRO_PASSES_ON_BASE = "repro_passes_on_base"
PRESERVATION_FAILS_ON_BASE = "preservation_fails_on_base"
NO_ADMITTED_REPRODUCTION_CHECK = "no_admitted_reproduction_check"
ALL_CHECKS_EXCLUDED = "all_checks_excluded"
EXCLUDED_STATUS_HINT = "excluded"

# Base classification reason (``admission._classify``) to exclusion reason.
EXCLUSION_REASONS: dict[str, str] = {
    "reproduction_passed_on_base": REPRO_PASSES_ON_BASE,
    "preservation_failed": PRESERVATION_FAILS_ON_BASE,
}


def exclusion_reason_for_role(role: CheckRole) -> str:
    """The only exclusion reason a check of ``role`` can have."""
    return REPRO_PASSES_ON_BASE if role is CheckRole.REPRODUCTION else PRESERVATION_FAILS_ON_BASE


def per_check_admission(admission: AdmissionResult) -> AdmissionResult:
    """``admission`` with the per-check rule applied (unchanged when it does not apply).

    Applies only to a ``rejected`` admission whose checks all ran, with no
    protected-byte mutation and an unchanged base, where every check that did
    not meet its contract was violated for one of ``EXCLUSION_REASONS``. The
    result is ``admitted`` with those checks in ``excluded_checks`` and tier
    ``C`` in ``check_tiers``. When every check would be excluded the verdict
    stays ``rejected`` and ``all_checks_excluded`` is added to the reasons.
    """
    if admission.verdict is not PackageVerdict.REJECTED or not admission.checks:
        return admission
    if admission.protected_bytes_mutated or (
        admission.base_tree_digest != admission.base_tree_digest_after
    ):
        return admission
    excluded: dict[str, str] = {}
    for check in admission.checks:
        if check.status is CheckStatus.EXPECTED:
            continue
        reason = EXCLUSION_REASONS.get(check.reason)
        if check.status is not CheckStatus.VIOLATED or reason is None:
            # Setup, import or other failures of the package itself: not admitted.
            return admission
        excluded[check.check_id] = reason
    if not excluded:
        return admission
    if len(excluded) == len(admission.checks):
        return admission.model_copy(update={"reasons": (*admission.reasons, ALL_CHECKS_EXCLUDED)})
    tiers = dict(admission.check_tiers or {})
    tiers.update(dict.fromkeys(excluded, CheckTier.C))
    return admission.model_copy(
        update={
            "verdict": PackageVerdict.ADMITTED,
            "check_tiers": tiers,
            "excluded_checks": excluded,
        }
    )


def excluded_check_ids(check_tiers: Mapping[str, str] | None) -> frozenset[str]:
    """Checks of an admitted package that are excluded (tier ``C``)."""
    return frozenset(
        check_id for check_id, tier in (check_tiers or {}).items() if tier == CheckTier.C.value
    )


class _Link(Protocol):
    @property
    def criterion_key(self) -> str: ...


class _LinkedCheck(Protocol):
    @property
    def check_id(self) -> str: ...

    @property
    def role(self) -> CheckRole: ...

    @property
    def assertions(self) -> Sequence[_Link]: ...


class LinkedChecks(Protocol):
    """What the coverage rule reads: a ``CheckPackage``, or its journal manifest re-read on resume."""

    @property
    def criterion_keys(self) -> Sequence[str]: ...

    @property
    def checks(self) -> Sequence[_LinkedCheck]: ...


def criteria_without_admitted_check(
    package: LinkedChecks, excluded: Mapping[str, str] | frozenset[str] | set[str]
) -> dict[str, str]:
    """Criteria that lose their authority to the exclusions, with the reason.

    Only criteria the package linked to at least one check are considered
    (criteria the constructor left uncovered keep their own reason). The
    reason is the first exclusion reason of the criterion's checks, or
    ``no_admitted_reproduction_check`` when an admitted preservation check
    remains for a reproduction-type criterion.
    """
    excluded_ids = set(excluded)
    lost: dict[str, str] = {}
    for key in package.criterion_keys:
        linked = [
            check
            for check in package.checks
            if any(link.criterion_key == key for link in check.assertions)
        ]
        dropped = [check for check in linked if check.check_id in excluded_ids]
        if not dropped:
            continue
        kept = [check for check in linked if check.check_id not in excluded_ids]
        if not kept:
            lost[key] = exclusion_reason_for_role(dropped[0].role)
        elif any(check.role is CheckRole.REPRODUCTION for check in linked) and not any(
            check.role is CheckRole.REPRODUCTION for check in kept
        ):
            lost[key] = NO_ADMITTED_REPRODUCTION_CHECK
    return lost


__all__ = [
    "ALL_CHECKS_EXCLUDED",
    "EXCLUDED_STATUS_HINT",
    "EXCLUSION_REASONS",
    "LinkedChecks",
    "NO_ADMITTED_REPRODUCTION_CHECK",
    "PRESERVATION_FAILS_ON_BASE",
    "REPRO_PASSES_ON_BASE",
    "criteria_without_admitted_check",
    "exclusion_reason_for_role",
    "excluded_check_ids",
    "per_check_admission",
]
