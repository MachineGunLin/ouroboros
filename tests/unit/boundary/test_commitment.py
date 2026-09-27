"""R3-S1: the package identity is a salted commit-reveal.

Before the final verdict nothing in the store, the journal, or the printed
preparation lines lets anyone confirm guessed held-out values offline (the
reviewer's probe rebuilt the package around guessed expected values and
compared its unkeyed SHA-256 with the stored one: 90 guesses). After the
verdict the salt is revealed and the pre-dispatch commitment verifies.
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import re
import stat
from typing import Any

import pytest

from ouroboros.boundary.constructor import ConstructionOutcome, package_from_reply
from ouroboros.boundary.events import (
    BOUNDARY_AGGREGATE_TYPE,
    PACKAGE_FROZEN,
    package_frozen_event,
)
from ouroboros.boundary.ledger import BoundaryLedger, BoundaryOrderError, verify_boundary_order
from ouroboros.boundary.oracle import ORACLE_DATA_PATH, OracleSpec, oracle_data_text
from ouroboros.boundary.package import (
    COMMITMENT_SCHEME,
    CheckPackage,
    CheckPackageError,
    PackageFile,
    commit_package,
    commitment_digest,
    load_reveal_record,
    new_commitment_salt,
    package_record,
    reveal_record,
    sha256_bytes,
    verify_commitment,
    write_reveal_record,
)
from ouroboros.boundary.run_wiring import (
    CheckPackageSettings,
    RegenerationPolicy,
    controller_private_dir,
    persist_commitment_salts,
    prepare_check_package,
    render_preparation,
    verify_check_package,
)
from ouroboros.core.seed import OntologySchema, Seed, SeedMetadata
from ouroboros.persistence.event_store import EventStore

BUGGY = "def clamp(value, low, high):\n    if value > high:\n        return value\n    return max(low, value)\n"
FIXED = "def clamp(value, low, high):\n    return max(low, min(high, value))\n"
HELD = {"held_1": -2, "held_2": 7}
DOMAIN = range(-10, 11)  # 21 x 21 = 441 joint guesses, a superset of the reviewer's 90


def _seed() -> Seed:
    return Seed(
        goal="clamp helper",
        acceptance_criteria=("clamp(15, 0, 10) returns 10",),
        ontology_schema=OntologySchema(name="mathutils", description="math helpers"),
        metadata=SeedMetadata(seed_id="seed_commit", ambiguity_score=0.1),
    )


def _reply() -> dict[str, Any]:
    return {
        "oracles": [
            {
                "criterion": 1,
                "check_id": "oracle_1",
                "role": "reproduction",
                "call_kind": "function",
                "params": ["value", "low", "high"],
                "default_binding": {"symbol": "mathutils.clamp"},
                "cases": [
                    {
                        "case_id": "stated",
                        "args": {"value": 15, "low": 0, "high": 10},
                        "expect": {"kind": "returns", "value": 10},
                    },
                    {
                        "case_id": "held_1",
                        "args": {"value": -3, "low": -2, "high": 4},
                        "expect": {"kind": "returns", "value": HELD["held_1"]},
                    },
                    {
                        "case_id": "held_2",
                        "args": {"value": 9, "low": 1, "high": 7},
                        "expect": {"kind": "returns", "value": HELD["held_2"]},
                    },
                ],
            }
        ],
    }


class _Constructor:
    def __init__(self, seed: Seed, base: Path) -> None:
        self.outcome = ConstructionOutcome(
            package_from_reply(
                _reply(), seed, input_digest="1" * 64, generator="fake", base_checkout=base
            ),
            None,
            "1" * 64,
            "fake",
        )

    async def construct(self, seed: Seed, base: Path, *, feedback=()) -> ConstructionOutcome:
        return self.outcome


@pytest.fixture
async def store():
    event_store = EventStore("sqlite+aiosqlite:///:memory:")
    await event_store.initialize()
    yield event_store
    await event_store.close()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mathutils.py").write_text(BUGGY)
    return root


def _guess(package: CheckPackage, values: dict[str, int]) -> CheckPackage:
    """The package an attacker rebuilds from the record around guessed held-out values."""
    oracles: list[OracleSpec] = []
    for spec in package.oracles:
        cases = tuple(
            case.model_copy(
                update={"expect": case.expect.model_copy(update={"value": values[case.case_id]})}
            )
            if case.case_id in values
            else case
            for case in spec.cases
        )
        oracles.append(spec.model_copy(update={"cases": cases}))
    files = tuple(
        PackageFile.from_content(item.path, oracle_data_text(oracles))
        if item.path == ORACLE_DATA_PATH
        else item
        for item in package.files
    )
    return package.model_copy(update={"oracles": tuple(oracles), "files": files})


def _probe(package: CheckPackage, haystack: bytes) -> list[tuple[int, int]]:
    """Every joint guess whose unkeyed package or oracle-data digest appears in ``haystack``.

    A 16-hex-character prefix counts too: 64 bits confirm a guess as well.
    """
    tokens = set(re.findall(rb"[0-9a-f]{16,}", haystack))
    prefixes = {token[:16] for token in tokens}
    hits = []
    for first, second in product(DOMAIN, DOMAIN):
        guessed = _guess(package, {"held_1": first, "held_2": second})
        data_file = next(item for item in guessed.files if item.path == ORACLE_DATA_PATH)
        for digest in (guessed.sha256, data_file.sha256):
            encoded = digest.encode()
            if encoded in tokens or encoded[:16] in prefixes:
                hits.append((first, second))
    return hits


async def _journal(store: EventStore, boundary_id: str) -> bytes:
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, boundary_id)
    return json.dumps([event.data for event in events], sort_keys=True).encode()


def _store_bytes(root: Path) -> bytes:
    return b"\n".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())


def test_commitment_api_round_trip(tmp_path: Path, repo: Path) -> None:
    package = _Constructor(_seed(), repo).outcome.package
    assert package is not None
    salt = new_commitment_salt()
    assert len(salt) == 32 and salt != new_commitment_salt()
    digest = commitment_digest(package, salt)
    assert digest == sha256_bytes(salt + package.to_json_bytes())
    assert digest != package.sha256
    assert verify_commitment(package, salt, digest)
    assert not verify_commitment(package, new_commitment_salt(), digest)
    assert not verify_commitment(_guess(package, {"held_1": -1, "held_2": 7}), salt, digest)
    assert not verify_commitment(package, b"short", digest)
    with pytest.raises(CheckPackageError):
        commitment_digest(package, b"short")

    committed = commit_package(package, salt)
    assert committed.commitment == committed.reference == digest
    assert committed.sha256 == package.sha256 and package.commitment is None
    assert package.reference == package.sha256
    assert committed.citation() == {"package_commitment": digest}
    summary = json.dumps(committed.manifest_summary())
    assert package.sha256 not in summary and digest in summary
    assert COMMITMENT_SCHEME in summary

    record = reveal_record(package, salt)
    assert record["package_commitment"] == digest and record["salt"] == salt.hex()
    path = write_reveal_record(package, salt, tmp_path / "private" / "reveals")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    loaded, loaded_salt, loaded_digest = load_reveal_record(path)
    assert (loaded.sha256, loaded_salt, loaded_digest) == (package.sha256, salt, digest)
    tampered = json.loads(path.read_text())
    tampered["package_commitment"] = "0" * 64
    other = tmp_path / "tampered.json"
    other.write_text(json.dumps(tampered))
    with pytest.raises(CheckPackageError):
        load_reveal_record(other)


def test_only_a_committed_package_is_recorded(repo: Path) -> None:
    package = _Constructor(_seed(), repo).outcome.package
    assert package is not None
    with pytest.raises(CheckPackageError):
        package_record(package, b"k" * 32)
    record = package_record(commit_package(package, new_commitment_salt()), b"k" * 32)
    assert "package_sha256" not in record and package.sha256 not in json.dumps(record)


def test_the_probe_recovers_held_out_values_from_an_unkeyed_digest(repo: Path) -> None:
    # Negative control: the probe works. Against what the journal carried
    # before this fix (the frozen event of an uncommitted package) it finds
    # the held-out values.
    package = _Constructor(_seed(), repo).outcome.package
    assert package is not None
    old_event = json.dumps(package_frozen_event("b", package).data).encode()
    assert _probe(package, old_event) == [(HELD["held_1"], HELD["held_2"])] * 2


async def test_nothing_before_the_verdict_confirms_a_held_out_guess(
    store: EventStore, repo: Path, tmp_path: Path
) -> None:
    seed = _seed()
    settings = CheckPackageSettings(True, policy=RegenerationPolicy.STUDY)
    state = await prepare_check_package(
        seed,
        event_store=store,
        constructor=_Constructor(seed, repo),
        execution_id="exec_commit",
        base_checkout=repo,
        worker_workspace=repo,
        runtime_label="test",
        settings=settings,
        store_dir=tmp_path / "store",
    )
    package = state.package
    assert package is not None and package.commitment is not None
    assert [case.held_out for case in package.oracles[0].cases] == [False, True, True]
    # The rebuild is exact: the true values give the true digest.
    assert _guess(package, HELD).sha256 == package.sha256

    (repo / "mathutils.py").write_text(FIXED)
    verdict = await verify_check_package(
        state, event_store=store, candidate_checkout=repo, settings=settings
    )
    assert verdict.verdict == "pass"

    # Everything a same-user worker could read before the reveal: the store,
    # the journal, and the preparation lines (console and log).
    private = controller_private_dir(tmp_path / "store")
    assert not private.exists()
    haystack = b"\n".join(
        (
            _store_bytes(tmp_path / "store"),
            await _journal(store, state.boundary_id),
            "\n".join(render_preparation(state)).encode(),
            json.dumps(verdict.summary()).encode(),
        )
    )
    assert _probe(package, haystack) == []
    assert package.sha256.encode()[:16] not in haystack
    frozen = (await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id))[0]
    assert frozen.type == PACKAGE_FROZEN
    assert frozen.data["package_commitment"] == package.commitment
    assert "package_sha256" not in frozen.data
    events = await store.replay(BOUNDARY_AGGREGATE_TYPE, state.boundary_id)
    assert verify_boundary_order(events) == ()
    assert all("package_sha256" not in event.data for event in events)

    # The final verdict exists: reveal. The salt goes beside the store, not
    # into it, and the pre-dispatch commitment verifies.
    (salt_path,) = persist_commitment_salts(state)
    assert salt_path.parent == private and not salt_path.is_relative_to(tmp_path / "store")
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE(salt_path.stat().st_mode) == 0o600
    revealed = json.loads(salt_path.read_text())
    salt = bytes.fromhex(revealed["salt"])
    assert verify_commitment(package, salt, frozen.data["package_commitment"])
    assert not verify_commitment(
        _guess(package, {"held_1": -2, "held_2": 6}), salt, frozen.data["package_commitment"]
    )
    assert persist_commitment_salts(state) == [salt_path]  # idempotent


async def test_a_committed_boundary_refuses_an_uncommitted_receipt(
    store: EventStore, repo: Path
) -> None:
    seed = _seed()
    package = _Constructor(seed, repo).outcome.package
    assert package is not None
    committed = commit_package(package, new_commitment_salt())
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("b1", committed, seed=seed)
    with pytest.raises(BoundaryOrderError):
        await ledger.record_bindings("b1", package_sha256=package.sha256, payload={})
