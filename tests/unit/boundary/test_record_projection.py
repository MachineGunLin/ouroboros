"""The stored record and the journal summary carry product-computed values only.

Anything the constructor wrote (a file path, a script, argv, a failure
signature, a binding symbol, a parameter name, a case, a locator, a reason)
can carry a held-out value. Neither ``package_record_bytes`` nor
``manifest_summary`` may contain any of it, wherever the constructor put it.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

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
SCRIPT_PATH = f".ouroboros_checks/expected_{SENTINEL}.py"
# The bot's probe: the held-out value copied into a referenced preservation script.
SCRIPT = f"from mathutils import clamp\nassert clamp(-3, -2, {SENTINEL}) == -2\n"


def _reply() -> dict[str, Any]:
    """A valid reply with the sentinel in every constructor-controlled free-form field."""
    oracle = {
        "criterion": 1,
        "check_id": f"oracle_{SENTINEL}",
        "role": "reproduction",
        "call_kind": "method",
        "params": [f"value_{SENTINEL}", "low", "high"],
        "default_binding": {
            "symbol": f"mathutils_{SENTINEL}.Clamp{SENTINEL}.clamp",
            "arg_map": {f"value_{SENTINEL}": 0, "low": 1, "high": f"high_{SENTINEL}"},
        },
        "target_named_in_criterion": False,
        "cases": [
            {
                "case_id": f"stated_{SENTINEL}",
                "held_out": False,
                "args": {f"value_{SENTINEL}": 15, "low": 0, "high": f"{SENTINEL}"},
                "init": {"note": f"{SENTINEL}"},
                "expect": {"kind": "returns", "value": f"visible {SENTINEL}"},
            },
            {
                "case_id": f"held_{SENTINEL}",
                "held_out": True,
                "args": {f"value_{SENTINEL}": -3, "low": -2, "high": int(SENTINEL)},
                "expect": {"kind": "raises", "exception": f"Error{SENTINEL}"},
            },
        ],
    }
    preservation = {
        "check_id": f"s{SENTINEL}",
        "role": "preservation",
        "argv": ["python3", SCRIPT_PATH],
        "assertions": [{"criterion": 2, "assertion_id": f"a{SENTINEL}", "locator": SENTINEL}],
    }
    reproduction = {
        "check_id": f"r{SENTINEL}",
        "role": "reproduction",
        "argv": ["python3", SCRIPT_PATH],
        "failure_signature": f"CLAMP_FAILED_{SENTINEL}_SIGNATURE",
        "assertions": [{"criterion": 3, "assertion_id": f"b{SENTINEL}"}],
    }
    return {
        "oracles": [oracle],
        "checks": [preservation, reproduction],
        "files": [{"path": SCRIPT_PATH, "content": SCRIPT}],
        "uncovered": [{"criterion": 3, "reason": f"reason {SENTINEL}"}],
    }


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
    package = package_from_reply(
        _reply(), _seed(), input_digest="1" * 64, generator="constructor-model"
    )
    # The sentinel is really in the package: in the script, its path and argv,
    # the signature, the binding, the params and the cases.
    assert SENTINEL.encode() in package.to_json_bytes()
    assert any(SENTINEL in item.content for item in package.files)
    _assert_sentinel_absent(package)


def test_paths_the_package_carries_from_any_caller_are_not_persisted() -> None:
    # base_files and scratch_paths are not set by the reply parser; a caller
    # that assembles a package can still put a held-out value in them.
    seed = _seed()
    keys = seed_criterion_keys(seed)
    script = PackageFile.from_content(SCRIPT_PATH, SCRIPT)
    package = CheckPackage(
        seed_digest=seed_digest(seed),
        criterion_keys=keys,
        input_digest="1" * 64,
        generated_at=datetime(2026, 9, 26, tzinfo=UTC),
        generator="assembler",
        checks=(
            CheckSpec(
                check_id="script_1_1",
                role=CheckRole.PRESERVATION,
                argv=("python3", SCRIPT_PATH),
                assertions=tuple(
                    AssertionLink(
                        assertion_id="script_1_1.a1",
                        criterion_key=key,
                        file=SCRIPT_PATH,
                        locator=f"line {SENTINEL}",
                    )
                    for key in keys
                ),
            ),
        ),
        files=(script,),
        base_files=(BaseFileRef(path=f"src/value_{SENTINEL}.py", sha256="2" * 64),),
        scratch_paths=(f"scratch_{SENTINEL}",),
    )
    _assert_sentinel_absent(package)


def test_the_record_and_the_summary_are_one_projection() -> None:
    package = seal_package(
        package_from_reply(_reply(), _seed(), input_digest="1" * 64, generator="constructor-model")
    )
    record = package_record(package)
    assert record["package"] == package.manifest_summary()
    assert (record["package_id"], record["seed_digest"]) == (
        package.package_id,
        package.seed_digest,
    )
