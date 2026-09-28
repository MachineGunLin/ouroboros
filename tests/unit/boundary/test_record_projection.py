"""The stored record and the journal summary carry product-computed values only.

Anything the constructor wrote (a file path, a script, argv, a failure
signature, a binding symbol, a parameter name, a case, a locator, a reason)
can carry a held-out value. Neither ``package_record_bytes`` nor
``manifest_summary`` may contain any of it, wherever the constructor put it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import hashlib
import json
from typing import Any

import pytest

from ouroboros.boundary.oracle_build import package_from_reply
from ouroboros.boundary.package import (
    AssertionLink,
    BaseFileRef,
    CheckPackage,
    CheckRole,
    CheckSpec,
    PackageFile,
    package_record,
    package_record_bytes,
    seal_package,
    seed_criterion_keys,
    seed_digest,
)

from .clamp_fixtures import _seed

SENTINEL = "6173"
FIXED_TIME = datetime(2026, 9, 26, tzinfo=UTC)
SEED = _seed()  # one Seed: its metadata carries a creation time


def _script_path(secret: str) -> str:
    return f".ouroboros_checks/expected_{secret}.py"


def _script(secret: str) -> str:
    # The bot's probe: the held-out value copied into a referenced preservation script.
    return f"from mathutils import clamp\nassert clamp(-3, -2, {secret}) == -2\n"


def _reply(secret: str = SENTINEL) -> dict[str, Any]:
    """A valid reply with ``secret`` in every constructor-controlled free-form field."""
    oracle = {
        "criterion": 1,
        "check_id": f"oracle_{secret}",
        "role": "reproduction",
        "call_kind": "method",
        "params": [f"value_{secret}", "low", "high"],
        "default_binding": {
            "symbol": f"mathutils_{secret}.Clamp{secret}.clamp",
            "arg_map": {f"value_{secret}": 0, "low": 1, "high": f"high_{secret}"},
        },
        "target_named_in_criterion": False,
        "cases": [
            {
                "case_id": f"stated_{secret}",
                "held_out": False,
                "args": {f"value_{secret}": 15, "low": 0, "high": f"{secret}"},
                "init": {"note": f"{secret}"},
                "expect": {"kind": "returns", "value": f"visible {secret}"},
            },
            {
                "case_id": f"held_{secret}",
                "held_out": True,
                "args": {f"value_{secret}": -3, "low": -2, "high": int(secret)},
                "expect": {"kind": "raises", "exception": f"Error{secret}"},
            },
        ],
    }
    preservation = {
        "check_id": f"s{secret}",
        "role": "preservation",
        "argv": ["python3", _script_path(secret)],
        "assertions": [{"criterion": 2, "assertion_id": f"a{secret}", "locator": secret}],
    }
    reproduction = {
        "check_id": f"r{secret}",
        "role": "reproduction",
        "argv": ["python3", _script_path(secret)],
        "failure_signature": f"CLAMP_FAILED_{secret}_SIGNATURE",
        "assertions": [{"criterion": 3, "assertion_id": f"b{secret}"}],
    }
    return {
        "oracles": [oracle],
        "checks": [preservation, reproduction],
        "files": [{"path": _script_path(secret), "content": _script(secret)}],
        "uncovered": [{"criterion": 3, "reason": f"reason {secret}"}],
    }


def _from_reply(secret: str = SENTINEL) -> CheckPackage:
    return package_from_reply(
        _reply(secret),
        SEED,
        input_digest="1" * 64,
        generator="constructor-model",
        generated_at=FIXED_TIME,
    )


def _assert_sentinel_absent(package: CheckPackage) -> None:
    sealed = seal_package(package)
    record = package_record_bytes(sealed)
    summaries = (
        json.dumps(sealed.manifest_summary()),
        json.dumps(package.manifest_summary()),
    )
    assert SENTINEL.encode() not in record
    assert json.loads(record) == package_record(sealed)
    for text in summaries:
        assert SENTINEL not in text


def test_no_constructor_text_reaches_the_record_or_the_summary() -> None:
    package = _from_reply()
    # The sentinel is really in the package: in the script, its path and argv,
    # the signature, the binding, the params and the cases.
    assert SENTINEL.encode() in package.to_json_bytes()
    assert any(SENTINEL in item.content for item in package.files)
    _assert_sentinel_absent(package)


def _assembled(secret: str = SENTINEL) -> CheckPackage:
    """A package a caller assembled, with ``secret`` in base_files and scratch_paths too."""
    seed = SEED
    keys = seed_criterion_keys(seed)
    script = PackageFile.from_content(_script_path(secret), _script(secret))
    return CheckPackage(
        seed_digest=seed_digest(seed),
        criterion_keys=keys,
        input_digest="1" * 64,
        generated_at=FIXED_TIME,
        generator="assembler",
        checks=(
            CheckSpec(
                check_id="script_1_1",
                role=CheckRole.PRESERVATION,
                argv=("python3", _script_path(secret)),
                assertions=tuple(
                    AssertionLink(
                        assertion_id="script_1_1.a1",
                        criterion_key=key,
                        file=_script_path(secret),
                        locator=f"line {secret}",
                    )
                    for key in keys
                ),
            ),
        ),
        files=(script,),
        base_files=(
            BaseFileRef(
                path=f"src/value_{secret}.py", sha256=hashlib.sha256(secret.encode()).hexdigest()
            ),
        ),
        scratch_paths=(f"scratch_{secret}",),
    )


def test_paths_the_package_carries_from_any_caller_are_not_persisted() -> None:
    # base_files and scratch_paths are not set by the reply parser; a caller
    # that assembles a package can still put a held-out value in them.
    _assert_sentinel_absent(_assembled())


def test_the_record_and_the_summary_are_one_projection() -> None:
    package = seal_package(_from_reply())
    record = package_record(package)
    assert record["package"] == package.manifest_summary()
    assert (record["package_id"], record["seed_digest"]) == (
        package.package_id,
        package.seed_digest,
    )


def _normalized(package: CheckPackage) -> tuple[bytes, str, str]:
    """Record bytes and summaries with the random package id replaced by a placeholder."""
    sealed = seal_package(package)
    placeholder = "0" * len(sealed.package_id)

    def blind(text: str) -> str:
        return text.replace(sealed.package_id, placeholder)

    return (
        blind(package_record_bytes(sealed).decode("utf-8")).encode("utf-8"),
        blind(json.dumps(sealed.manifest_summary(), sort_keys=True)),
        json.dumps(package.manifest_summary(), sort_keys=True),
    )


@pytest.mark.parametrize("build", [_from_reply, _assembled])
@pytest.mark.parametrize("other", ["4409", "12", "987654321"])
def test_packages_differing_only_in_constructor_bytes_persist_identically(
    build: Callable[[str], CheckPackage], other: str
) -> None:
    # The real guard: nothing persisted is a function of what the constructor
    # wrote (not its text, not a digest or a length of it), so no persisted
    # byte can confirm or narrow a guessed held-out value.
    first, second = build(SENTINEL), build(other)
    assert first.to_json_bytes() != second.to_json_bytes()
    assert _normalized(first) == _normalized(second)


def test_a_held_out_value_in_a_script_cannot_be_enumerated_from_the_record() -> None:
    # The bot's probe: try every 4-digit value against the persisted bytes.
    sealed = seal_package(_from_reply())
    haystack = package_record_bytes(sealed) + json.dumps(sealed.manifest_summary()).encode()
    recovered = [
        guess
        for guess in (f"{number:04d}" for number in range(10_000))
        if hashlib.sha256(_script(guess).encode()).hexdigest().encode() in haystack
    ]
    assert recovered == []
