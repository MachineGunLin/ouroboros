"""A29 per-check admission, routing by admitted check, coverage buckets: the library API.

These are the functions the study harness can call directly (one seal per
boundary): ``per_check_admission`` on a recorded admission, then the usual
``assign_tiers(admitted_tiers=...)`` and ``criterion_verdicts``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ouroboros.boundary.acceptance import (
    RECONCILIATION_SCHEMA,
    ExistingOutcome,
    PackageCriterionStatus,
    VerificationCoverage,
    criterion_verdicts,
    reconcile_acceptance,
    verification_coverage,
)
from ouroboros.boundary.admission import PackageVerdict, admit_check_package
from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.binding_flow import admission_tiers, assign_tiers, verify_with_bindings
from ouroboros.boundary.constructor import package_from_reply
from ouroboros.boundary.per_check import (
    ALL_CHECKS_EXCLUDED,
    REPRO_PASSES_ON_BASE,
    per_check_admission,
)
from ouroboros.core.seed import OntologySchema, Seed, SeedMetadata

BUGGY = "def clamp(value, low, high):\n    if value > high:\n        return value\n    return max(low, value)\n"
FIXED = "def clamp(value, low, high):\n    return max(low, min(high, value))\n"


def _seed(*criteria: str) -> Seed:
    return Seed(
        goal="clamp helper",
        acceptance_criteria=criteria
        or (
            "clamp(15, 0, 10) returns 10",
            "clamp(5, 0, 10) returns 5",
            "clamp(-5, 0, 10) returns 0",
        ),
        ontology_schema=OntologySchema(name="mathutils", description="math helpers"),
        metadata=SeedMetadata(seed_id="seed_per_check", ambiguity_score=0.1),
    )


def _oracle(
    criterion: int, check_id: str, role: str, args: tuple[int, int, int], value: int
) -> dict[str, Any]:
    value_, low, high = args
    return {
        "criterion": criterion,
        "check_id": check_id,
        "role": role,
        "call_kind": "function",
        "params": ["value", "low", "high"],
        "default_binding": {"symbol": "mathutils.clamp"},
        "cases": [
            {
                "case_id": "stated",
                "args": {"value": value_, "low": low, "high": high},
                "expect": {"kind": "returns", "value": value},
            }
        ],
    }


# Base (BUGGY): clamp(15, 0, 10) = 15, clamp(5, 0, 10) = 5, clamp(-5, 0, 10) = 0.
GOOD_REPRO_1 = _oracle(1, "oracle_1", "reproduction", (15, 0, 10), 10)
BAD_REPRO_2 = _oracle(2, "oracle_2", "reproduction", (5, 0, 10), 5)  # passes on base
GOOD_PRESERVE_3 = _oracle(3, "oracle_3", "preservation", (-5, 0, 10), 0)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mathutils.py").write_text(BUGGY)
    return root


async def test_an_indeterminate_check_keeps_the_whole_package_rule(repo: Path) -> None:
    seed = _seed("clamp(15, 0, 10) returns 10", "clamp(5, 0, 10) returns 5")
    crash = _oracle(1, "oracle_1", "reproduction", (15, 0, 10), 10)
    crash["default_binding"] = {"symbol": "mathutils.clamp"}
    package = package_from_reply(
        {"oracles": [crash, BAD_REPRO_2]},
        seed,
        input_digest="1" * 64,
        generator="fake",
        base_checkout=repo,
    )
    (repo / "mathutils.py").write_text("raise ImportError('broken at import')\n")
    admission = await admit_check_package(
        package,
        repo,
        check_tiers=admission_tiers(package, seed, repo),
        exclude_checks_individually=True,
    )
    assert admission.verdict is PackageVerdict.INDETERMINATE
    assert admission.excluded_checks is None


async def test_excluding_every_check_is_no_admission(repo: Path) -> None:
    seed = _seed("clamp(5, 0, 10) returns 5")
    package = package_from_reply(
        {"oracles": [_oracle(1, "oracle_1", "reproduction", (5, 0, 10), 5)]},
        seed,
        input_digest="1" * 64,
        generator="fake",
        base_checkout=repo,
    )
    raw = await admit_check_package(package, repo)
    applied = per_check_admission(raw)
    assert raw.verdict is applied.verdict is PackageVerdict.REJECTED
    assert applied.reasons == (*raw.reasons, ALL_CHECKS_EXCLUDED)
    assert applied.excluded_checks is None


async def test_the_harness_rule_matches_the_product_admission(repo: Path) -> None:
    """Study API: the pure rule on a recorded admission, then the usual tiers and verdicts."""
    seed = _seed()
    package = package_from_reply(
        {"oracles": [GOOD_REPRO_1, BAD_REPRO_2, GOOD_PRESERVE_3]},
        seed,
        input_digest="1" * 64,
        generator="fake",
        base_checkout=repo,
    )
    tiers = admission_tiers(package, seed, repo)
    raw = await admit_check_package(package, repo, check_tiers=tiers)
    assert raw.verdict is PackageVerdict.REJECTED
    applied = per_check_admission(raw)
    product = await admit_check_package(
        package, repo, check_tiers=tiers, exclude_checks_individually=True
    )
    assert applied.verdict is product.verdict is PackageVerdict.ADMITTED
    assert applied.excluded_checks == product.excluded_checks == {"oracle_2": REPRO_PASSES_ON_BASE}
    assert applied.check_tiers == product.check_tiers
    (repo / "mathutils.py").write_text(FIXED)
    assignments, _ = await assign_tiers(
        package, artifact=repo, base=None, admitted_tiers=applied.check_tiers
    )
    assert assignments["oracle_2"].tier is CheckTier.C
    bound = await verify_with_bindings(package, repo, assignments)
    assert {check.check_id for check in bound.effective.checks} == {"oracle_1", "oracle_3"}
    verdicts = criterion_verdicts(package, bound.effective, assignments=assignments)
    assert [v.status for v in verdicts.values()] == [
        PackageCriterionStatus.PASS,
        PackageCriterionStatus.UNCOVERED,
        PackageCriterionStatus.PASS,
    ]


async def test_an_admission_without_exclusions_keeps_its_bytes(repo: Path) -> None:
    seed = _seed("clamp(15, 0, 10) returns 10")
    package = package_from_reply(
        {"oracles": [GOOD_REPRO_1]},
        seed,
        input_digest="1" * 64,
        generator="fake",
        base_checkout=repo,
    )
    admission = await admit_check_package(package, repo)
    assert admission.verdict is PackageVerdict.ADMITTED
    assert per_check_admission(admission) is admission
    assert "excluded_checks" not in admission.event_summary()


DJANGO_UNDER = (
    "When models use custom fields and mixins, generated migration files include the imports "
    "needed to resolve referenced names and do not raise a NameError for undefined names."
)
WILLING = "The user is willing to assist with debugging the issue."


def _dev_seed() -> Seed:
    return _seed(DJANGO_UNDER, WILLING)


def test_an_uncovered_reason_never_routes() -> None:
    """Routing rests on one fact: whether an admitted check covers the criterion.

    The constructor's uncovered reason is descriptive text. A criterion left
    uncovered with any reason, ``non_behavioral`` included, is uncovered, is
    counted as not decided by the package, and the legacy verifier decides it.
    """
    seed = _dev_seed()
    package = package_from_reply(
        {
            "uncovered": [
                {"criterion": 1, "reason": "non_behavioral"},
                {"criterion": 2, "reason": "not executable"},
            ]
        },
        seed,
        input_digest="1" * 64,
        generator="fake",
    )
    verdicts = criterion_verdicts(package, None)
    assert [v.status for v in verdicts.values()] == [PackageCriterionStatus.UNCOVERED] * 2
    rejected = {i: ExistingOutcome(i, "failed", "failed", "failed") for i in range(2)}
    decided = reconcile_acceptance(
        package.criterion_keys,
        verdicts,
        rejected,
        existing_run_accepted=False,
        legacy_decides_unverified=True,
    )
    assert [d.legacy_decided and not d.accepted for d in decided.decisions] == [True, True]
    assert len(decided.not_package_decided) == 2
    assert decided.coverage is VerificationCoverage.LOW
    assert "non_behavioral_count" not in decided.to_dict()


async def test_a_linked_check_decides_whatever_else_the_reply_says(repo: Path) -> None:
    """A reply may still carry a ``labels`` entry (an older prompt): it is ignored.

    The criterion keeps its check, the check is admitted, and the package
    decides the criterion.
    """
    seed = _seed("clamp(15, 0, 10) returns 10")
    reply = {
        "oracles": [GOOD_REPRO_1],
        "labels": [{"criterion": 1, "kind": "context", "evidence_span": "returns 10"}],
    }
    package = package_from_reply(
        reply, seed, input_digest="1" * 64, generator="fake", base_checkout=repo
    )
    assert [check.check_id for check in package.checks] == ["oracle_1"]
    assert package.uncovered == ()
    admission = await admit_check_package(package, repo, exclude_checks_individually=True)
    assert admission.verdict is PackageVerdict.ADMITTED and admission.excluded_checks is None
    (repo / "mathutils.py").write_text(FIXED)
    assignments, _ = await assign_tiers(
        package, artifact=repo, base=None, admitted_tiers=admission.check_tiers
    )
    bound = await verify_with_bindings(package, repo, assignments)
    verdicts = criterion_verdicts(package, bound.effective, assignments=assignments)
    assert [v.status for v in verdicts.values()] == [PackageCriterionStatus.PASS]


@pytest.mark.parametrize(
    ("total", "not_decided", "unverified", "bucket"),
    [
        (3, 0, 0, VerificationCoverage.FULL),
        (3, 1, 0, VerificationCoverage.PARTIAL),
        (4, 2, 0, VerificationCoverage.LOW),
        (2, 1, 0, VerificationCoverage.LOW),
        (5, 1, 1, VerificationCoverage.LOW),
    ],
)
def test_coverage_buckets(total: int, not_decided: int, unverified: int, bucket: Any) -> None:
    assert verification_coverage(total, not_decided, unverified) is bucket


def test_the_default_reconciliation_is_unchanged() -> None:
    """Without the legacy rule (study callers) U is accepted and the schema stays v2."""
    keys = ("k0",)
    rejected = {0: ExistingOutcome(0, "failed", "failed", "failed")}
    before = reconcile_acceptance(
        keys, {"k0": PackageCriterionStatus.UNCOVERED}, rejected, existing_run_accepted=False
    )
    assert before.run_accepted and before.to_dict()["schema_version"] == RECONCILIATION_SCHEMA
    assert set(before.to_dict()) == {
        "schema_version",
        "run_accepted",
        "existing_run_accepted",
        "artifact_verdict",
        "verified_pass_count",
        "unverified_count",
        "criterion_count",
        "tier_summary",
        "criteria",
    }
    after = reconcile_acceptance(
        keys,
        {"k0": PackageCriterionStatus.UNCOVERED},
        rejected,
        existing_run_accepted=False,
        legacy_decides_unverified=True,
    )
    assert not after.run_accepted and after.decisions[0].legacy_decided
