"""Derived-expectation admission: every oracle case must agree with a reference.

The constructor states each case's expected value by hand. A slip (for
example ``clamp(1234, -2345, 3456)`` expected to return ``3456``) passes
admission when the base also fails that case, and then fails every correct
worker. So the constructor also writes, in the same reply, a pure reference
implementation of each criterion (``reference``: module source plus the
symbol to call; standard library only, no I/O, never the project's code).

Before any worker starts, and before the package is frozen, the controller
runs the reference on every case input through the product harness
(``oracle_run.run_oracle_check``: one isolated target process per case under
the same caps, in a scratch directory holding only the reference module, the
comparison inside the controller) and compares:

- a held-out case whose stated expectation disagrees with the reference is
  excluded (``oracle_inconsistent``): it is not in the frozen package, so it
  can never be held out, revealed, or counted;
- a case whose literals all appear in the Seed text the worker sees (the
  product's definition of a Seed example, ``oracle.mark_held_out``) must be
  reproduced by the reference; otherwise the reference is not trusted and
  the criterion is uncovered (``reference_contradicts_seed_example``);
- a missing reference, or one that does not import or run (for example one
  that imports the project, which is not in the scratch directory), leaves
  the criterion uncovered (``reference_unavailable``); so does an oracle
  whose cases were all excluded (``oracle_inconsistent``).

Limit: this catches a stated value that disagrees with the constructor's own
reading of the criterion. When the cases and the reference share one
misreading of the specification, they agree, and the check passes them.

The reference is a correct implementation of the criterion, so it never
reaches the store, the journal, the partial replies or a log. It exists in
controller memory and, during this check only, in a scratch directory that
is removed before the worker starts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

from ouroboros.boundary.binding import Binding, CallKind
from ouroboros.boundary.oracle import (
    ORACLE_DATA_PATH,
    ORACLE_HARNESS_PATH,
    ORACLE_HARNESS_SOURCE,
    OracleSpec,
    is_oracle_file,
    oracle_data_text,
)
from ouroboros.boundary.oracle_build import assemble_package
from ouroboros.boundary.oracle_run import run_oracle_check

if TYPE_CHECKING:
    from ouroboros.boundary.package import CheckPackage
    from ouroboros.core.seed import Seed

ORACLE_INCONSISTENT = "oracle_inconsistent"
REFERENCE_CONTRADICTS_SEED_EXAMPLE = "reference_contradicts_seed_example"
REFERENCE_UNAVAILABLE = "reference_unavailable"
REFERENCE_LEFT_NO_CHECKS = "reference_check_left_no_checks"
REFERENCE_MODULE = "oracle_reference"
REFERENCE_SCHEMA = "ouroboros.reference_check.v1"
_MAX_SOURCE_CHARS = 100_000
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


@dataclass(frozen=True, slots=True)
class OracleReference:
    """The constructor's reference implementation of one criterion (memory only)."""

    source: str = field(repr=False)
    symbol: str = ""


def parse_reference(raw: object) -> OracleReference | None:
    """``{"source": str, "symbol": str}`` as a reference, or ``None`` when malformed.

    ``symbol`` names a function (``clamp``) or a method (``Clamp.value``) in
    the module; a command's reference is the module run as a script, and its
    symbol is ignored.
    """
    if not isinstance(raw, Mapping):
        return None
    source, symbol = raw.get("source"), raw.get("symbol") or ""
    if not isinstance(source, str) or not source.strip() or len(source) > _MAX_SOURCE_CHARS:
        return None
    if not isinstance(symbol, str) or (symbol and not _SYMBOL.fullmatch(symbol)):
        return None
    return OracleReference(source=source, symbol=symbol)


def references_from_reply(reply: Mapping[str, Any]) -> dict[str, OracleReference | None]:
    """Each oracle's reference by check id (``None`` when missing or malformed)."""
    references: dict[str, OracleReference | None] = {}
    for raw in reply.get("oracles") or ():
        if not isinstance(raw, Mapping):
            continue
        check_id = str(raw.get("check_id") or f"oracle_{raw.get('criterion')}")
        references[check_id] = parse_reference(raw.get("reference"))
    return references


@dataclass(frozen=True, slots=True)
class ReferenceCheck:
    """What the reference check changed: case ids and criteria, never values."""

    excluded: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Check id to the case ids excluded as ``oracle_inconsistent``."""
    uncovered: dict[str, str] = field(default_factory=dict)
    """Criterion key to the reason it became uncovered."""

    def counts(self) -> dict[str, int]:
        """Excluded cases, and criteria made uncovered, by reason."""
        reasons = list(self.uncovered.values())
        return {
            ORACLE_INCONSISTENT: sum(len(ids) for ids in self.excluded.values()),
            REFERENCE_CONTRADICTS_SEED_EXAMPLE: reasons.count(REFERENCE_CONTRADICTS_SEED_EXAMPLE),
            REFERENCE_UNAVAILABLE: reasons.count(REFERENCE_UNAVAILABLE),
            "criteria_left_without_cases": reasons.count(ORACLE_INCONSISTENT),
        }

    def payload(self) -> dict[str, Any]:
        """The journal payload (``boundary.oracle.reference_checked``)."""
        return {
            "schema_version": REFERENCE_SCHEMA,
            "counts": self.counts(),
            "excluded_cases": [
                {"check_id": check_id, "case_ids": list(ids), "reason": ORACLE_INCONSISTENT}
                for check_id, ids in sorted(self.excluded.items())
            ],
            "uncovered": [
                {"criterion_key": key, "reason": reason}
                for key, reason in sorted(self.uncovered.items())
            ],
        }


def _reference_binding(spec: OracleSpec, reference: OracleReference) -> Binding:
    if spec.call_kind is CallKind.CLI:
        symbol = f"{REFERENCE_MODULE}.py"
    else:
        symbol = f"{REFERENCE_MODULE}.{reference.symbol}"
    # Every input by its declared name: the reference takes the params by name.
    return Binding(criterion_key=spec.criterion_key, symbol=symbol, call_kind=spec.call_kind)


async def _disagreeing_cases(
    spec: OracleSpec,
    reference: OracleReference,
    *,
    env: Mapping[str, str],
    interpreter: str | None,
    timeout_seconds: float,
) -> list[str] | None:
    """Ids of the cases the reference disagrees with, or ``None`` when it could not run."""
    if spec.call_kind is not CallKind.CLI and not reference.symbol:
        return None
    scratch = Path(tempfile.mkdtemp(prefix="ouroboros-reference-"))
    try:
        (scratch / f"{REFERENCE_MODULE}.py").write_text(reference.source, encoding="utf-8")
        run = await run_oracle_check(
            {
                ORACLE_HARNESS_PATH: ORACLE_HARNESS_SOURCE,
                ORACLE_DATA_PATH: oracle_data_text([spec]),
            },
            spec,
            scratch,
            timeout_seconds=timeout_seconds,
            on_base=True,
            env=env,
            interpreter=interpreter,
            binding=_reference_binding(spec, reference),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    result = run.result
    if result is None or result.get("resolve") != "ok":
        return None
    passed = {case.get("case_id"): bool(case.get("passed")) for case in result.get("cases") or ()}
    if set(passed) != {case.case_id for case in spec.cases}:
        return None
    return [case.case_id for case in spec.cases if not passed[case.case_id]]


async def check_references(
    package: CheckPackage,
    references: Mapping[str, OracleReference | None],
    *,
    seed: Seed,
    env: Mapping[str, str],
    interpreter: str | None,
    timeout_seconds: float,
) -> tuple[CheckPackage, ReferenceCheck]:
    """The package with inconsistent cases excluded and untrusted oracles uncovered.

    An unchanged package is returned as is. Script checks are untouched.
    """
    kept: list[tuple[OracleSpec, Any]] = []
    excluded: dict[str, tuple[str, ...]] = {}
    uncovered: dict[str, str] = {}
    roles = {check.check_id: check.role for check in package.checks}
    for spec in package.oracles:
        reference = references.get(spec.check_id)
        disagree = (
            None
            if reference is None
            else await _disagreeing_cases(
                spec,
                reference,
                env=env,
                interpreter=interpreter,
                timeout_seconds=timeout_seconds,
            )
        )
        if disagree is None:
            uncovered[spec.criterion_key] = REFERENCE_UNAVAILABLE
            continue
        by_id = {case.case_id: case for case in spec.cases}
        if any(not by_id[case_id].held_out for case_id in disagree):
            uncovered[spec.criterion_key] = REFERENCE_CONTRADICTS_SEED_EXAMPLE
            continue
        cases = tuple(case for case in spec.cases if case.case_id not in disagree)
        if not cases:
            uncovered[spec.criterion_key] = ORACLE_INCONSISTENT
            excluded[spec.check_id] = tuple(disagree)
            continue
        if disagree:
            excluded[spec.check_id] = tuple(disagree)
            spec = spec.model_copy(update={"cases": cases})
        kept.append((spec, roles[spec.check_id]))
    report = ReferenceCheck(excluded=excluded, uncovered=uncovered)
    if not excluded and not uncovered:
        return package, report
    oracle_ids = {spec.check_id for spec in package.oracles}
    rebuilt = assemble_package(
        seed,
        input_digest=package.input_digest,
        generator=package.generator,
        oracles=kept,
        script_checks=_script_checks(package, oracle_ids),
        script_files=tuple(item for item in package.files if not is_oracle_file(item.path)),
        uncovered={
            **{item.criterion_key: item.reason for item in package.uncovered},
            **uncovered,
        },
        generated_at=package.generated_at,
    )
    return rebuilt, report


def _script_checks(package: CheckPackage, oracle_ids: set[str]) -> Sequence[Any]:
    return tuple(check for check in package.checks if check.check_id not in oracle_ids)


__all__ = [
    "ORACLE_INCONSISTENT",
    "REFERENCE_LEFT_NO_CHECKS",
    "REFERENCE_CONTRADICTS_SEED_EXAMPLE",
    "REFERENCE_UNAVAILABLE",
    "OracleReference",
    "ReferenceCheck",
    "check_references",
    "parse_reference",
    "references_from_reply",
]
