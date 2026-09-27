"""Derived-expectation admission: cases must agree with the constructor's reference.

The round-5 smoke admitted ``clamp(1234, -2345, 3456)`` expected to return
``3456`` (the value is in range, so ``1234`` is right). It failed on the
buggy base as a reproduction case must, so admission kept it, and it then
failed the worker's correct fix on every attempt.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

from ouroboros.boundary.check_env import scrubbed_check_environment
from ouroboros.boundary.constructor import ConstructionOutcome, package_from_reply
from ouroboros.boundary.events import BOUNDARY_AGGREGATE_TYPE
from ouroboros.boundary.package import seed_criterion_keys
from ouroboros.boundary.reference_check import (
    ORACLE_INCONSISTENT,
    REFERENCE_CONTRADICTS_SEED_EXAMPLE,
    REFERENCE_UNAVAILABLE,
    check_references,
    references_from_reply,
)
from ouroboros.boundary.run_wiring import (
    CheckPackageSettings,
    RegenerationPolicy,
    forget_live_state,
    prepare_check_package,
)
from ouroboros.core.seed import OntologySchema, Seed, SeedMetadata
from ouroboros.persistence.event_store import EventStore

BUGGY = (
    "def clamp(value, low, high):\n"
    "    if value < low:\n"
    "        return low\n"
    "    if value > high:\n"
    "        return value\n"
    "    return value\n"
)
MARKER = "reference_marker_5b1f"
REFERENCE = f"# {MARKER}\ndef clamp(value, low, high):\n    return max(low, min(value, high))\n"
WRONG_REFERENCE = "def clamp(value, low, high):\n    return value\n"
PROJECT_REFERENCE = "from mathutils import clamp  # the project's code is not reachable\n"


def _seed() -> Seed:
    return Seed(
        goal="Fix mathutils.clamp so that a value above the upper bound is clamped.",
        acceptance_criteria=(
            "clamp(value, low, high) returns high when value > high, low when value < low, "
            "and value otherwise; for example clamp(15, 0, 10) == 10, "
            "clamp(-3, 0, 10) == 0, clamp(7, 0, 10) == 7.",
        ),
        ontology_schema=OntologySchema(name="Clamp", description="clamp helper"),
        metadata=SeedMetadata(seed_id="seed_reference", ambiguity_score=0.1),
    )


def _case(case_id: str, value: int, low: int, high: int, expected: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "args": {"value": value, "low": low, "high": high},
        "expect": {"kind": "returns", "value": expected},
    }


def _reply(reference: str | None, *cases: dict[str, Any]) -> dict[str, Any]:
    oracle: dict[str, Any] = {
        "criterion": 1,
        "check_id": "c1_clamp",
        "role": "reproduction",
        "call_kind": "function",
        "params": ["value", "low", "high"],
        "default_binding": {"symbol": "mathutils.clamp"},
        "cases": list(cases)
        or [
            _case("stated_above", 15, 0, 10, 10),
            _case("stated_below", -3, 0, 10, 0),
            _case("held_below", -4567, -3210, 5678, -3210),
            # The smoke's slip: in range, so clamp returns 1234, not 3456.
            _case("held_above", 1234, -2345, 3456, 3456),
        ],
    }
    if reference is not None:
        oracle["reference"] = {"source": reference, "symbol": "clamp"}
    return {"oracles": [oracle], "checks": [], "files": [], "uncovered": []}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mathutils.py").write_text(BUGGY)
    return root


async def _check(reply: dict[str, Any], repo: Path) -> tuple[Any, Any]:
    seed = _seed()
    package = package_from_reply(
        reply, seed, input_digest="1" * 64, generator="fake", base_checkout=repo
    )
    return await check_references(
        package,
        references_from_reply(reply),
        seed=seed,
        env=scrubbed_check_environment(),
        interpreter=sys.executable,
        timeout_seconds=30,
    )


async def test_the_smoke_case_is_excluded_and_consistent_cases_are_kept(repo: Path) -> None:
    package, report = await _check(_reply(REFERENCE), repo)
    (spec,) = package.oracles
    assert [(case.case_id, case.held_out) for case in spec.cases] == [
        ("stated_above", False),
        ("stated_below", False),
        ("held_below", True),
    ]
    assert report.excluded == {"c1_clamp": ("held_above",)}
    assert report.uncovered == {}
    assert report.counts()[ORACLE_INCONSISTENT] == 1
    # The frozen checks name only the kept cases.
    (check,) = package.checks
    assert [link.assertion_id for link in check.assertions] == [
        "c1_clamp.stated_above",
        "c1_clamp.stated_below",
        "c1_clamp.held_below",
    ]


async def test_a_consistent_oracle_is_unchanged(repo: Path) -> None:
    reply = _reply(
        REFERENCE,
        _case("stated_above", 15, 0, 10, 10),
        _case("held_above", 1234, -2345, 1000, 1000),
    )
    seed = _seed()
    original = package_from_reply(
        reply, seed, input_digest="1" * 64, generator="fake", base_checkout=repo
    )
    package, report = await _check(reply, repo)
    assert package.oracles == original.oracles and package.checks == original.checks
    assert report.excluded == {} and report.uncovered == {}


@pytest.mark.parametrize(
    "reply",
    [
        # The reference does not reproduce a Seed example (clamp(15, 0, 10) == 10).
        _reply(WRONG_REFERENCE),
        # A stated slip on a case whose literals all appear in the Seed text.
        _reply(REFERENCE, _case("stated_above", 15, 0, 10, 15), _case("h", 99, 1, 7, 7)),
    ],
    ids=["reference_misses_example", "stated_value_contradicts_reference"],
)
async def test_a_seed_example_contradiction_makes_the_criterion_uncovered(
    repo: Path, reply: dict[str, Any]
) -> None:
    package, report = await _check(reply, repo)
    key = seed_criterion_keys(_seed())[0]
    assert package.oracles == () and package.checks == ()
    assert [(item.criterion_key, item.reason) for item in package.uncovered] == [
        (key, REFERENCE_CONTRADICTS_SEED_EXAMPLE)
    ]
    assert report.uncovered == {key: REFERENCE_CONTRADICTS_SEED_EXAMPLE}


@pytest.mark.parametrize(
    "reference", [None, PROJECT_REFERENCE, "def clamp(:\n"], ids=["missing", "project", "syntax"]
)
async def test_a_reference_that_cannot_run_makes_the_criterion_uncovered(
    repo: Path, reference: str | None
) -> None:
    package, report = await _check(_reply(reference), repo)
    key = seed_criterion_keys(_seed())[0]
    assert package.oracles == ()
    assert report.uncovered == {key: REFERENCE_UNAVAILABLE}


async def test_an_oracle_whose_cases_all_disagree_is_uncovered(repo: Path) -> None:
    reply = _reply(REFERENCE, _case("h1", 1234, -2345, 3456, 3456), _case("h2", 5, 6, 9, 5))
    package, report = await _check(reply, repo)
    key = seed_criterion_keys(_seed())[0]
    assert package.oracles == ()
    assert report.uncovered == {key: ORACLE_INCONSISTENT}
    assert report.excluded == {"c1_clamp": ("h1", "h2")}


class _Constructor:
    def __init__(self, seed: Seed, base: Path, reply: dict[str, Any]) -> None:
        self.outcome = ConstructionOutcome(
            package_from_reply(
                reply, seed, input_digest="1" * 64, generator="fake", base_checkout=base
            ),
            None,
            "1" * 64,
            "fake",
            references=references_from_reply(reply),
        )

    async def construct(self, seed: Seed, base: Path, *, feedback=()) -> ConstructionOutcome:
        return self.outcome


async def test_preparation_freezes_only_consistent_cases_and_records_ids(
    repo: Path, tmp_path: Path
) -> None:
    store_events = EventStore("sqlite+aiosqlite:///:memory:")
    await store_events.initialize()
    try:
        seed = _seed()
        state = await prepare_check_package(
            seed,
            event_store=store_events,
            constructor=_Constructor(seed, repo, _reply(REFERENCE)),
            execution_id="exec_reference",
            base_checkout=repo,
            worker_workspace=repo,
            runtime_label="codex",
            settings=CheckPackageSettings(True, policy=RegenerationPolicy.STUDY),
            store_dir=tmp_path / "store",
        )
        assert state.admitted and state.package is not None
        assert [case.case_id for case in state.package.oracles[0].cases] == [
            "stated_above",
            "stated_below",
            "held_below",
        ]
        assert state.reference_check is not None
        assert state.reference_check.counts()[ORACLE_INCONSISTENT] == 1
        events = await store_events.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
        types = [event.type for event in events]
        checked = types.index("boundary.oracle.reference_checked")
        assert types.index("boundary.check_package.frozen") < checked
        assert checked < types.index("boundary.actor.started")
        payload = events[checked].data
        assert payload["excluded_cases"] == [
            {"check_id": "c1_clamp", "case_ids": ["held_above"], "reason": ORACLE_INCONSISTENT}
        ]
        assert payload["package_commitment"] == state.package.commitment
        # Neither the excluded values nor the reference reach the journal or the store.
        journal = "".join(str(event.data) for event in events)
        stored = "".join(
            path.read_text(errors="replace")
            for path in (tmp_path / "store").rglob("*")
            if path.is_file()
        )
        for text in (journal, stored):
            assert "3456" not in text and "2345" not in text and MARKER not in text
        forget_live_state(state)
    finally:
        await store_events.close()


async def test_a_command_reference_runs_as_a_script(repo: Path) -> None:
    seed = Seed(
        goal="A greeter command.",
        acceptance_criteria=("python greet.py --name Ada prints Hello, Ada",),
        ontology_schema=OntologySchema(name="Greet", description="greeter"),
        metadata=SeedMetadata(seed_id="seed_reference_cli", ambiguity_score=0.1),
    )
    reply = {
        "oracles": [
            {
                "criterion": 1,
                "check_id": "c1_greet",
                "role": "reproduction",
                "call_kind": "cli",
                "params": ["name"],
                "default_binding": {"symbol": "greet.py"},
                "reference": {
                    "source": (
                        "import argparse\n"
                        "parser = argparse.ArgumentParser()\n"
                        "parser.add_argument('--name')\n"
                        "print(f'Hello, {parser.parse_args().name}')\n"
                    ),
                    "symbol": "",
                },
                "cases": [
                    {
                        "case_id": "stated",
                        "args": {"name": "Ada"},
                        "expect": {"kind": "cli", "exit_code": 0, "stdout_contains": "Hello, Ada"},
                    },
                    {
                        "case_id": "held_ok",
                        "args": {"name": "Grace"},
                        "expect": {"kind": "cli", "stdout_contains": "Hello, Grace"},
                    },
                    {
                        "case_id": "held_slip",
                        "args": {"name": "Linus"},
                        "expect": {"kind": "cli", "stdout_contains": "Hello, Linux"},
                    },
                ],
            }
        ],
    }
    package = package_from_reply(
        reply, seed, input_digest="1" * 64, generator="fake", base_checkout=repo
    )
    package, report = await check_references(
        package,
        references_from_reply(reply),
        seed=seed,
        env=scrubbed_check_environment(),
        interpreter=sys.executable,
        timeout_seconds=30,
    )
    assert report.uncovered == {}
    assert report.excluded == {"c1_greet": ("held_slip",)}
    assert [case.case_id for case in package.oracles[0].cases] == ["stated", "held_ok"]
