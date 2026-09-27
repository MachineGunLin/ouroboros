"""Criterion kind: a model label from the check constructor's reply, validated structurally.

Seed acceptance criteria are the user's contract, so none is ever removed.
The constructor already decides, per criterion, whether it can write an
executable check; in the same reply (no extra model call) it labels each
criterion under ``labels``:

    {"criterion": <1-based number>, "kind": "behavior" | "implementation_preference"
     | "context", "evidence_span": "<verbatim text from the Seed>"}

A criterion labeled anything other than ``behavior`` is non-behavioral: it
gets no check (any check the reply linked to it is removed), it is uncovered
with reason ``non_behavioral``, it is never counted as unverified, and the
existing (legacy) verifier decides it.

Validation is structural only: ``kind`` must be one of the three values and
``evidence_span`` a non-empty verbatim substring of the Seed text (goal,
constraints and criterion descriptions; plain containment). A missing or
invalid label is a parse failure (``CriterionLabel.parse_failure``) and is
treated as ``behavior``, which keeps the check path. No rule here reads the
criterion's wording.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ouroboros.boundary.package import CheckPackage
    from ouroboros.core.seed import Seed

NON_BEHAVIORAL = "non_behavioral"
LABEL_SCHEMA = "ouroboros.criterion_label.v1"


class CriterionKind(StrEnum):
    BEHAVIOR = "behavior"
    IMPLEMENTATION_PREFERENCE = "implementation_preference"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class CriterionLabel:
    """The constructor's label for one criterion, after structural validation."""

    kind: CriterionKind
    evidence_span: str | None = None
    parse_failure: str | None = None
    """Why the label was not usable (``missing``, ``invalid_kind``, ...); ``kind`` is then behavior."""

    @property
    def behavioral(self) -> bool:
        return self.kind is CriterionKind.BEHAVIOR


def _seed_text(seed: Seed) -> str:
    from ouroboros.boundary.oracle import worker_visible_seed_text

    return worker_visible_seed_text(seed)


def _failure(reason: str) -> CriterionLabel:
    return CriterionLabel(CriterionKind.BEHAVIOR, None, reason)


def _label(raw: Any, seed_text: str) -> CriterionLabel:
    if not isinstance(raw, Mapping):
        return _failure("not_an_object")
    try:
        kind = CriterionKind(raw.get("kind"))
    except ValueError:
        return _failure("invalid_kind")
    span = raw.get("evidence_span")
    if not isinstance(span, str) or not span.strip():
        return _failure("missing_evidence_span")
    if span not in seed_text:
        return _failure("evidence_span_not_in_seed")
    return CriterionLabel(kind, span)


def criterion_labels(reply: Mapping[str, Any] | None, seed: Seed) -> dict[str, CriterionLabel]:
    """Every Seed criterion's label from a constructor reply (criterion key to label).

    A criterion with no label, more than one label, or an invalid one gets a
    parse failure and counts as ``behavior``.
    """
    from ouroboros.boundary.package import seed_criterion_keys

    keys = seed_criterion_keys(seed)
    text = _seed_text(seed)
    raw_labels = reply.get("labels") if isinstance(reply, Mapping) else None
    by_number: dict[int, list[Any]] = {}
    if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, str):
        for raw in raw_labels:
            number = raw.get("criterion") if isinstance(raw, Mapping) else None
            if isinstance(number, int) and not isinstance(number, bool):
                by_number.setdefault(number, []).append(raw)
    labels: dict[str, CriterionLabel] = {}
    for number, key in enumerate(keys, start=1):
        entries = by_number.get(number, [])
        if not entries:
            labels[key] = _failure("missing")
        elif len(entries) > 1:
            labels[key] = _failure("duplicate")
        else:
            labels[key] = _label(entries[0], text)
    return labels


def non_behavioral_keys(labels: Mapping[str, CriterionLabel] | None) -> tuple[str, ...]:
    """Criterion keys whose valid label is not ``behavior``, in label order."""
    return tuple(key for key, label in (labels or {}).items() if not label.behavioral)


def non_behavioral_criteria(
    source: CheckPackage | Mapping[str, Any], seed: Seed | None = None
) -> tuple[int, ...]:
    """0-based indices of the non-behavioral criteria.

    ``source`` is a package (criteria uncovered with reason ``non_behavioral``)
    or a constructor reply together with its ``seed`` (the reply's labels).
    """
    from ouroboros.boundary.package import CheckPackage, seed_criterion_keys

    if isinstance(source, CheckPackage):
        marked = {item.criterion_key for item in source.uncovered if item.reason == NON_BEHAVIORAL}
        return tuple(i for i, key in enumerate(source.criterion_keys) if key in marked)
    if seed is None:
        raise ValueError("a constructor reply needs its seed")
    keys = seed_criterion_keys(seed)
    marked = set(non_behavioral_keys(criterion_labels(source, seed)))
    return tuple(i for i, key in enumerate(keys) if key in marked)


def label_summary(labels: Mapping[str, CriterionLabel] | None) -> dict[str, int]:
    """Counts by kind and parse failures (journal- and telemetry-safe)."""
    counts = {kind.value: 0 for kind in CriterionKind}
    failures = 0
    for label in (labels or {}).values():
        counts[label.kind.value] += 1
        failures += label.parse_failure is not None
    return {**counts, "parse_failures": failures}


__all__ = [
    "LABEL_SCHEMA",
    "NON_BEHAVIORAL",
    "CriterionKind",
    "CriterionLabel",
    "criterion_labels",
    "label_summary",
    "non_behavioral_criteria",
    "non_behavioral_keys",
]
