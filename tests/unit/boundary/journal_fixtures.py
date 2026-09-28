"""Receipts built as data, for journal tests (nothing runs)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ouroboros.boundary.binding import CheckTier
from ouroboros.boundary.package import CheckPackage, CheckRole, CheckSpec
from ouroboros.boundary.receipts import (
    AdmissionResult,
    CandidateVerdict,
    CandidateVerification,
    CheckExecution,
    CheckStatus,
    PackageVerdict,
)
from ouroboros.boundary.tree import tree_digest

PIN = "d" * 64
"""A stand-in interpreter digest (binary and real path) for receipts built as data."""


def expected_execution(check: CheckSpec) -> CheckExecution:
    """The base result admission records for a check that met its role (built as data)."""
    return CheckExecution(
        check_id=check.check_id,
        role=check.role,
        argv=check.argv,
        cwd=check.cwd,
        status=CheckStatus.EXPECTED,
        reason=(
            "reached_failing_assertion"
            if check.role is CheckRole.REPRODUCTION
            else "preservation_passed"
        ),
        return_code=1 if check.role is CheckRole.REPRODUCTION else 0,
        timed_out=False,
        duration_seconds=0.0,
        signature_seen=check.role is CheckRole.REPRODUCTION,
        stdout_sha256="0" * 64,
        stderr_sha256="0" * 64,
        output_tail="",
        protected_digest_before="0" * 64,
        protected_digest_after="0" * 64,
        mutated_paths=(),
        scratch_outputs=(),
        undeclared_outputs=(),
    )


def admission_receipt(
    package: CheckPackage, checkout: Path, verdict: PackageVerdict = PackageVerdict.ADMITTED
) -> AdmissionResult:
    """An admission receipt for ``package`` on ``checkout``, built as data (nothing runs).

    An admitted receipt is one admission can write: one expected result and a
    tier per check, and the interpreter pin.
    """
    now = datetime.now(UTC)
    digest = tree_digest(checkout)
    admitted = verdict is PackageVerdict.ADMITTED
    return AdmissionResult(
        package_sha256=package.sha256,
        package_id=package.package_id if package.sealed else None,
        seed_digest=package.seed_digest,
        base_tree_digest=digest,
        base_tree_digest_after=digest,
        verdict=verdict,
        reasons=(),
        protected_bytes_mutated=False,
        timeout_seconds=120,
        checks=tuple(expected_execution(check) for check in package.checks) if admitted else (),
        started_at=now,
        completed_at=now,
        interpreter_sha256=PIN if admitted else None,
        interpreter_realpath_sha256=PIN if admitted else None,
        check_tiers={
            c.check_id: CheckTier.A if package.oracle_for(c.check_id) else CheckTier.S
            for c in package.checks
        }
        if admitted
        else None,
    )


def verification_receipt(
    package: CheckPackage, checkout: Path, verdict: CandidateVerdict = CandidateVerdict.FAIL
) -> CandidateVerification:
    """A candidate verification of ``package`` on ``checkout``, built as data (nothing runs)."""
    now = datetime.now(UTC)
    digest = tree_digest(checkout)
    return CandidateVerification(
        package_sha256=package.sha256,
        package_id=package.package_id if package.sealed else None,
        seed_digest=package.seed_digest,
        artifact_tree_digest=digest,
        artifact_tree_digest_after=digest,
        verdict=verdict,
        reasons=(),
        protected_bytes_mutated=False,
        timeout_seconds=120,
        checks=(),
        started_at=now,
        completed_at=now,
    )
