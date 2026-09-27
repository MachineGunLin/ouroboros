"""Behavioral filter: which acceptance criteria a check package can decide.

Seed acceptance criteria are the user's contract, so none is ever removed.
A criterion that names nothing a check could observe (for example "The user
is willing to assist with debugging the issue.") gets no check construction:
it is uncovered with reason ``non_behavioral``, it is never counted as
unverified, and the existing (legacy) verifier decides it.

Pre-registered rule (``BEHAVIORAL_RULE``, fixed; a change is a new version).
A criterion is behavioral when its description, or its declared
``verify_command`` / ``output_assertion``, names an observable input or
invocation, output or return value, raised error, or file or repository
state, tested by any one of:

1. ``code_span``: a backquoted span (```name```);
2. ``call``: a call expression, an identifier (dotted allowed) immediately
   followed by a parenthesized argument list, for example ``clamp(5, 0, 3)``;
3. ``path``: a file name with a common source, data or document extension,
   or a path with at least two ``/`` separators;
4. ``error_name``: a CamelCase name ending in ``Error``, ``Exception`` or
   ``Warning``;
5. ``verb``: a word (any inflection listed in ``OBSERVABLE_VERBS``, case
   insensitive, whole words only) from the fixed list raise, throw, return,
   output, print, emit, display, exit, fail, pass, reject, accept, import,
   generate, write, create, delete, save, call, invoke, give, yield, produce,
   equal, compute, calculate, convert, parse, render, respond;
6. ``declared_command``: the criterion declares a ``verify_command`` or an
   ``output_assertion``.

Otherwise it is non-behavioral. The rule reads only the Seed text; it runs
no model and nothing in the repository.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ouroboros.core.seed import Seed

NON_BEHAVIORAL = "non_behavioral"
BEHAVIORAL_RULE = "ouroboros.behavioral_rule.v1"

OBSERVABLE_VERBS: Mapping[str, tuple[str, ...]] = {
    "raise": ("raise", "raises", "raised", "raising"),
    "throw": ("throw", "throws", "threw", "thrown", "throwing"),
    "return": ("return", "returns", "returned", "returning"),
    "output": ("output", "outputs", "outputted", "outputting"),
    "print": ("print", "prints", "printed", "printing"),
    "emit": ("emit", "emits", "emitted", "emitting"),
    "display": ("display", "displays", "displayed", "displaying"),
    "exit": ("exit", "exits", "exited", "exiting"),
    "fail": ("fail", "fails", "failed", "failing"),
    "pass": ("pass", "passes", "passed", "passing"),
    "reject": ("reject", "rejects", "rejected", "rejecting"),
    "accept": ("accept", "accepts", "accepted", "accepting"),
    "import": ("import", "imports", "imported", "importing"),
    "generate": ("generate", "generates", "generated", "generating"),
    "write": ("write", "writes", "wrote", "written", "writing"),
    "create": ("create", "creates", "created", "creating"),
    "delete": ("delete", "deletes", "deleted", "deleting"),
    "save": ("save", "saves", "saved", "saving"),
    "call": ("call", "calls", "called", "calling"),
    "invoke": ("invoke", "invokes", "invoked", "invoking"),
    "give": ("give", "gives", "gave", "given", "giving"),
    "yield": ("yield", "yields", "yielded", "yielding"),
    "produce": ("produce", "produces", "produced", "producing"),
    "equal": ("equal", "equals", "equaled", "equalled", "equaling", "equalling"),
    "compute": ("compute", "computes", "computed", "computing"),
    "calculate": ("calculate", "calculates", "calculated", "calculating"),
    "convert": ("convert", "converts", "converted", "converting"),
    "parse": ("parse", "parses", "parsed", "parsing"),
    "render": ("render", "renders", "rendered", "rendering"),
    "respond": ("respond", "responds", "responded", "responding"),
}

_EXTENSIONS = (
    "py|pyi|js|jsx|ts|tsx|json|yaml|yml|toml|ini|cfg|conf|txt|md|rst|csv|html|htm|css"
    "|sh|rs|go|java|kt|c|h|cc|cpp|hpp|rb|php|sql|xml|lock|env"
)
_TESTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code_span", re.compile(r"`[^`\n]+`")),
    ("call", re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\([^()\n]*\)")),
    (
        "path",
        re.compile(
            rf"(?:\b[\w-]+(?:/[\w.-]+)*\.(?:{_EXTENSIONS})\b)|(?:[\w.-]*/[\w.-]+/[\w./-]*)",
            re.IGNORECASE,
        ),
    ),
    ("error_name", re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning)\b")),
    (
        "verb",
        re.compile(
            r"\b(?:"
            + "|".join(sorted({form for forms in OBSERVABLE_VERBS.values() for form in forms}))
            + r")\b",
            re.IGNORECASE,
        ),
    ),
)


def behavioral_evidence(text: str) -> str | None:
    """The first rule test ``text`` meets (``code_span``, ``call``, ...), or ``None``."""
    for name, pattern in _TESTS:
        if pattern.search(text or ""):
            return name
    return None


def is_behavioral(text: str) -> bool:
    """Whether ``text`` names an observable (see the module docstring)."""
    return behavioral_evidence(text) is not None


def criterion_evidence(criterion: object) -> str | None:
    """``behavioral_evidence`` for one Seed criterion (a string or a spec)."""
    if getattr(criterion, "verify_command", None) or getattr(criterion, "output_assertion", None):
        return "declared_command"
    description = getattr(criterion, "description", None)
    text = description if isinstance(description, str) else str(criterion)
    return behavioral_evidence(text)


def non_behavioral_criteria(seed: Seed) -> tuple[int, ...]:
    """0-based indices of the Seed's criteria that the rule finds non-behavioral."""
    return tuple(
        index
        for index, criterion in enumerate(seed.acceptance_criteria)
        if criterion_evidence(criterion) is None
    )


__all__ = [
    "BEHAVIORAL_RULE",
    "NON_BEHAVIORAL",
    "OBSERVABLE_VERBS",
    "behavioral_evidence",
    "criterion_evidence",
    "is_behavioral",
    "non_behavioral_criteria",
]
