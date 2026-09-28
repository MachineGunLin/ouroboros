"""Bindings: closed grammar, static validation, and tier assignment."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from ouroboros.boundary.binding import (
    Binding,
    BindingError,
    BindingValidation,
    CallKind,
    CheckTier,
    assign_tier,
    default_binding_resolves,
    entry_points_request,
    locate_symbol,
    parse_binding,
    script_check_tier,
    tier_summary,
    validate_declared_binding_static,
)
from ouroboros.boundary.oracle import ORACLE_DATA_PATH
from ouroboros.boundary.oracle_build import assemble_package, build_oracle_spec
from ouroboros.boundary.package import CheckPackage, CheckRole, sha256_bytes
from ouroboros.boundary.tree import tree_manifest

PARAMS = ("value", "low", "high")


def _parse(raw: object, kind: CallKind = CallKind.FUNCTION) -> Binding:
    return parse_binding(raw, criterion_key="k1", params=PARAMS, call_kind=kind)


def test_grammar_accepts_renames_and_permutations() -> None:
    binding = _parse(
        {"symbol": "mathutils.clamp", "arg_map": {"value": 0, "low": "lo", "high": "hi"}}
    )
    assert binding.arg_map == {"value": 0, "low": "lo", "high": "hi"}
    assert _parse({"symbol": "pkg.mod.fn"}).arg_map == {}
    permuted = _parse({"symbol": "m.f", "arg_map": {"value": 2, "low": 0, "high": 1}})
    assert permuted.describe() == "function m.f (high->1, low->0, value->2)"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        # A literal injected as an argument (the value never reaches the target).
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 1, "high": 10}}, "position_out_of_range"),
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 1, "high": "10"}}, "not_a_position"),
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 1, "high": {"literal": 10}}}, "not_a"),
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 1, "high": True}}, "not_a_position"),
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 1, "high": 2.0}}, "not_a_position"),
        # A lambda, in the map or as the symbol.
        ({"symbol": "m.f", "arg_map": {"value": "lambda v: 10", "low": 1, "high": 2}}, "not_a"),
        ({"symbol": "lambda v, lo, hi: min(hi, v)", "arg_map": {}}, "symbol_not_dotted_path"),
        # A nested call, in the map or as the symbol.
        ({"symbol": "m.f", "arg_map": {"value": "min(high, value)", "low": 1, "high": 2}}, "not_a"),
        ({"symbol": "m.wrap(m.f)"}, "symbol_not_dotted_path"),
        # Defaults, routing, and anything else outside the grammar.
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 1}}, "keys_differ"),
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 0, "high": 1}}, "not_unique"),
        ({"symbol": "m.f", "arg_map": {"value": 0, "low": 2, "high": "high"}}, "not_contiguous"),
        ({"symbol": "m.f", "defaults": {"high": 10}}, "unknown_keys"),
        ({"symbol": "m.f", "call_kind": "cli"}, "call_kind_differs"),
        ({"symbol": "f"}, "symbol_not_dotted_path"),
        ("m.f", "binding_not_an_object"),
    ],
)
def test_grammar_rejects_everything_but_rename_or_permute(raw: object, reason: str) -> None:
    with pytest.raises(BindingError) as caught:
        _parse(raw)
    assert reason in caught.value.reason


def test_cli_grammar() -> None:
    binding = _parse(
        {
            "symbol": "tools/clamp.py",
            "call_kind": "cli",
            "arg_map": {"value": 0, "low": "--low", "high": "--high"},
        },
        CallKind.CLI,
    )
    assert binding.call_kind is CallKind.CLI
    assert _parse({"symbol": "-m pkg.cli", "call_kind": "cli"}, CallKind.CLI).symbol == "-m pkg.cli"
    for bad in ("../x.py", "/abs/x.py", "x.py; rm -rf /", "-m pkg; ls"):
        with pytest.raises(BindingError):
            _parse({"symbol": bad, "call_kind": "cli"}, CallKind.CLI)
    with pytest.raises(BindingError):
        _parse(
            {
                "symbol": "x.py",
                "call_kind": "cli",
                "arg_map": {"value": 0, "low": 1, "high": "high"},
            },
            CallKind.CLI,
        )


def _repo(root: Path, files: dict[str, str]) -> Path:
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return root


def test_locate_symbol_statically(tmp_path: Path) -> None:
    root = _repo(
        tmp_path / "r",
        {
            "mathutils.py": "def clamp(v, lo, hi):\n    return v\n\nclass Box:\n    def area(self):\n        return 1\n",
            "src/geo/__init__.py": "from .shapes import circle\n",
            "src/geo/shapes.py": "def circle(r):\n    return 3 * r * r\n",
            "tools/run.py": "print(1)\n",
        },
    )
    assert locate_symbol(root, "mathutils.clamp", CallKind.FUNCTION).path == "mathutils.py"
    assert locate_symbol(root, "mathutils.Box.area", CallKind.METHOD).qualname == "Box.area"
    reexported = locate_symbol(root, "geo.circle", CallKind.FUNCTION)
    assert reexported is not None and reexported.path == "src/geo/shapes.py"
    assert locate_symbol(root, "tools/run.py", CallKind.CLI) is not None
    assert locate_symbol(root, "mathutils.lerp", CallKind.FUNCTION) is None
    assert locate_symbol(root, "mathutils.clamp", CallKind.METHOD) is None
    assert locate_symbol(root, "nope/run.py", CallKind.CLI) is None


def test_default_binding_resolves_at_base_or_by_declaration(tmp_path: Path) -> None:
    base = _repo(tmp_path / "base", {"mathutils.py": "def clamp(v, lo, hi):\n    return v\n"})
    existing = _parse({"symbol": "mathutils.clamp"})
    absent = _parse({"symbol": "mathutils.lerp"})
    assert default_binding_resolves(existing, base, named_in_criterion=False)
    # Declared named by the criterion: tier A even though the base lacks it.
    assert default_binding_resolves(absent, base, named_in_criterion=True)
    assert default_binding_resolves(absent, None, named_in_criterion=True)
    # Neither declared nor present at the base: not tier A.
    assert not default_binding_resolves(absent, base, named_in_criterion=False)
    assert not default_binding_resolves(existing, None, named_in_criterion=False)


def _validate(raw: object, artifact: Path, base: Path, **kwargs: object) -> BindingValidation:
    return validate_declared_binding_static(
        raw,
        criterion_key="k1",
        params=PARAMS,
        call_kind=CallKind.FUNCTION,
        artifact=artifact,
        base=base,
        base_manifest=tree_manifest(base),
        **kwargs,
    )


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    base = _repo(
        tmp_path / "base",
        {
            "mathutils.py": "def clamp(v, lo, hi):\n    return v\n",
            "other.py": "def keep(x):\n    return x\n",
        },
    )
    artifact = _repo(
        tmp_path / "artifact",
        {
            "mathutils.py": "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))\n",
            "other.py": "def keep(x):\n    return x\n",
            "interp.py": "def mix(a, b, t):\n    return a + (b - a) * t\n",
            ".ouroboros_checks/fake.py": "def mix(a, b, t):\n    return 0\n",
        },
    )
    return base, artifact


def test_declared_binding_introduced_by_the_diff_is_valid(pair: tuple[Path, Path]) -> None:
    base, artifact = pair
    result = _validate(
        {"symbol": "interp.mix", "arg_map": {"value": 0, "low": 1, "high": 2}}, artifact, base
    )
    assert result.valid and result.changed_by_artifact and result.exists_at_base is False
    assert result.location == "interp.py"


def test_declared_binding_modified_by_the_diff_is_valid(pair: tuple[Path, Path]) -> None:
    base, artifact = pair
    result = _validate({"symbol": "mathutils.clamp"}, artifact, base)
    assert result.valid and result.exists_at_base and result.changed_by_artifact


def test_declared_binding_existing_at_base_is_statically_valid(pair: tuple[Path, Path]) -> None:
    base, artifact = pair
    # Unchanged base code passes the static rule; the base run
    # (admission.admit_binding) is what refuses it when it already passes.
    result = _validate({"symbol": "other.keep"}, artifact, base)
    assert result.valid and result.exists_at_base and not result.changed_by_artifact


def test_declared_binding_to_a_standard_library_name_is_refused(tmp_path: Path) -> None:
    # a workspace json.py satisfies the static lookup, but the harness
    # has already imported the real json module.
    base = _repo(tmp_path / "base", {"mathutils.py": "def clamp(v, lo, hi):\n    return v\n"})
    artifact = _repo(
        tmp_path / "artifact",
        {
            "mathutils.py": "def clamp(v, lo, hi):\n    return v\n",
            "json.py": "def loads(v, lo, hi):\n    return v\n",
        },
    )
    result = _validate({"symbol": "json.loads"}, artifact, base)
    assert not result.valid and result.reason == "binding_invalid:stdlib_symbol"


@pytest.mark.parametrize("module", ["__main__", "sitecustomize", "usercustomize"])
def test_declared_binding_to_an_interpreter_module_is_refused(tmp_path: Path, module: str) -> None:
    # a workspace __main__.py satisfies the static lookup, but in the
    # target process __main__ is the harness itself.
    base = _repo(tmp_path / "base", {"mathutils.py": "def clamp(v, lo, hi):\n    return v\n"})
    artifact = _repo(
        tmp_path / "artifact",
        {
            "mathutils.py": "def clamp(v, lo, hi):\n    return v\n",
            f"{module}.py": "def _short(v, lo, hi):\n    return v\n",
        },
    )
    result = _validate({"symbol": f"{module}._short"}, artifact, base)
    assert not result.valid and result.reason == "binding_invalid:interpreter_module"


def test_declared_binding_under_the_check_dir_is_refused(pair: tuple[Path, Path]) -> None:
    base, artifact = pair
    result = _validate({"symbol": ".ouroboros_checks.fake.mix"}, artifact, base)
    assert not result.valid and result.reason.startswith("binding_invalid:")
    cli = validate_declared_binding_static(
        {"symbol": ".ouroboros_checks/fake.py", "call_kind": "cli"},
        criterion_key="k1",
        params=PARAMS,
        call_kind=CallKind.CLI,
        artifact=artifact,
        base=base,
    )
    assert not cli.valid


def test_wrong_symbol_is_refused(pair: tuple[Path, Path]) -> None:
    base, artifact = pair
    result = _validate({"symbol": "interp.blend"}, artifact, base)
    assert not result.valid and result.reason == "binding_invalid:symbol_not_found"
    bad_grammar = _validate(
        {"symbol": "interp.mix", "arg_map": {"value": "1+1", "low": 1, "high": 2}}, artifact, base
    )
    assert bad_grammar.reason.startswith("binding_invalid:arg_map")


def test_tier_assignment_matrix() -> None:
    default = _parse({"symbol": "m.f"})
    declared = _parse({"symbol": "m.g"})
    valid = BindingValidation(
        criterion_key="k1", valid=True, reason="binding_admitted_on_base", binding=declared
    )
    invalid = BindingValidation(
        criterion_key="k1",
        valid=False,
        reason="binding_invalid:binding_passes_on_base",
        binding=declared,
    )
    timeout = BindingValidation(
        criterion_key="k1", valid=False, indeterminate=True, reason="binding_admission_timeout"
    )

    a = assign_tier(
        criterion_key="k1",
        check_id="o1",
        default_binding=default,
        default_resolves=True,
        declared=valid,
    )
    assert (a.tier, a.binding, a.status_hint) == (CheckTier.A, default, "run")
    a_prime = assign_tier(
        criterion_key="k1",
        check_id="o1",
        default_binding=default,
        default_resolves=False,
        declared=valid,
    )
    assert (a_prime.tier, a_prime.binding, a_prime.binding_source.value) == (
        CheckTier.A_PRIME,
        declared,
        "declared",
    )
    bad = assign_tier(
        criterion_key="k1",
        check_id="o1",
        default_binding=default,
        default_resolves=False,
        declared=invalid,
    )
    assert (bad.tier, bad.status_hint, bad.reason) == (
        CheckTier.U,
        "indeterminate",
        "binding_invalid:binding_passes_on_base",
    )
    late = assign_tier(
        criterion_key="k1",
        check_id="o1",
        default_binding=default,
        default_resolves=False,
        declared=timeout,
    )
    assert (late.status_hint, late.reason) == ("indeterminate", "binding_admission_timeout")
    none = assign_tier(
        criterion_key="k1",
        check_id="o1",
        default_binding=default,
        default_resolves=False,
        declared=None,
    )
    assert (none.tier, none.status_hint, none.reason) == (CheckTier.U, "unverified", "no_binding")
    assert CheckTier.A_PRIME.label == "A'" and CheckTier.A_PRIME.value == "A_prime"
    assert tier_summary([CheckTier.A, CheckTier.A, CheckTier.U]) == {
        "A": 2,
        "A_prime": 0,
        "U": 1,
        "C": 0,
    }


def test_script_check_tier_follows_its_imports(tmp_path: Path) -> None:
    base = _repo(tmp_path / "base", {"mathutils.py": "def clamp(v, lo, hi):\n    return v\n"})
    existing = "import sys\nfrom mathutils import clamp\n"
    absent = "from mathutils import lerp\n"
    assert script_check_tier(existing, base=base, named_in_criterion=False)[0] is CheckTier.A
    assert script_check_tier(absent, base=base, named_in_criterion=True)[0] is CheckTier.A
    tier, reason = script_check_tier(absent, base=base, named_in_criterion=False)
    assert tier is CheckTier.U and reason == "no_binding:mathutils.lerp"


def test_entry_points_request_names_the_inputs_only() -> None:
    text = entry_points_request({"call_kind": "function", "params": ["a", "b", "t"]})
    assert '"entry_points"' in text and "(a, b, t)" in text
    assert "expect" not in text and "held" not in text
    assert "(a, b, t)" not in entry_points_request(None)


def test_the_binding_model_owns_the_closed_grammar() -> None:
    # Direct construction is held to the grammar parse_binding enforces.
    for symbol in ("/tmp/foreign.py", "../outside.py", "./x.py"):
        with pytest.raises(ValidationError):
            Binding(criterion_key="k1", symbol=symbol, call_kind=CallKind.CLI)
    with pytest.raises(ValidationError):
        Binding(criterion_key="k1", symbol="not a path", call_kind=CallKind.FUNCTION)
    with pytest.raises(ValidationError):
        Binding(criterion_key="k1", symbol="m.f", arg_map={"a": 1}, call_kind=CallKind.FUNCTION)
    assert Binding(criterion_key="k1", symbol="tool.py", call_kind=CallKind.CLI).symbol == (
        "tool.py"
    )


def test_a_stored_package_cannot_smuggle_an_absolute_cli_target(seed) -> None:
    # The bot's probe: build a valid CLI oracle package, rewrite the stored
    # binding to an absolute path, and load it canonically.
    spec = build_oracle_spec(
        seed,
        criterion_index=0,
        check_id="oracle_1",
        call_kind="cli",
        params=("a", "b"),
        default_binding={"symbol": "tool.py"},
        cases=[
            {
                "case_id": "c1",
                "held_out": False,
                "args": {"a": 1, "b": 2},
                "expect": {"kind": "cli", "exit_code": 0},
            }
        ],
    )
    package = assemble_package(
        seed,
        input_digest="1" * 64,
        generator="test",
        oracles=((spec, CheckRole.REPRODUCTION),),
    )
    data = json.loads(package.to_json_bytes())
    assert CheckPackage.model_validate(data).sha256 == package.sha256
    data["oracles"][0]["default_binding"]["symbol"] = "/tmp/foreign.py"
    # Keep the frozen oracle data file consistent with the edit, so only the
    # binding grammar can refuse the package.
    for item in data["files"]:
        if item["path"] == ORACLE_DATA_PATH:
            oracle_data = json.loads(item["content"])
            oracle_data["oracles"][0]["default_binding"]["symbol"] = "/tmp/foreign.py"
            item["content"] = (
                json.dumps(oracle_data, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
            )
            item["sha256"] = sha256_bytes(item["content"].encode("utf-8"))
    with pytest.raises(ValidationError, match="cli_path_invalid"):
        CheckPackage.model_validate(data)
    with pytest.raises(ValidationError, match="cli_path_invalid"):
        CheckPackage.model_validate_json(json.dumps(data))


def test_an_unparseable_script_has_no_known_binding(tmp_path: Path) -> None:
    base = _repo(tmp_path / "base", {"mathutils.py": "def clamp(v, lo, hi):\n    return v\n"})
    tier, reason = script_check_tier("def broken(:\n", base=base, named_in_criterion=True)
    assert tier is CheckTier.U and reason == "script_unparsed"


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("mod = __import__('new_impl')\n", "script_dynamic_import"),
        ("import importlib\nmod = importlib.import_module('new_impl')\n", "script_dynamic_import"),
        ("from importlib import import_module\n", "script_dynamic_import"),
        ("import runpy\nrunpy.run_path('x.py')\n", "script_dynamic_import"),
        ("import builtins\nbuiltins.exec('import new_impl')\n", "script_dynamic_import"),
        ("exec('from new_impl import target')\n", "script_dynamic_import"),
        ("from .new_impl import target\n", "script_relative_import"),
        ("from . import new_impl\n", "script_relative_import"),
        ("from new_impl import *\n", "script_star_import"),
    ],
)
@pytest.mark.parametrize("named", [False, True])
def test_a_script_whose_target_its_imports_do_not_show_is_never_tier_a(
    tmp_path: Path, source: str, reason: str, named: bool
) -> None:
    # #2458 round 3: these forms were tier A (``script_imports_resolve``)
    # against an empty base. A declared named target does not change that:
    # the script does not bind to it through a static import.
    base = _repo(tmp_path / "base", {"mathutils.py": "def clamp(v, lo, hi):\n    return v\n"})
    assert script_check_tier(source, base=base, named_in_criterion=named) == (CheckTier.U, reason)


def test_ordinary_standard_library_use_keeps_a_static_script_tier_a(tmp_path: Path) -> None:
    base = _repo(tmp_path / "base", {"mathutils.py": "def clamp(v, lo, hi):\n    return v\n"})
    source = (
        "import re\nfrom os.path import *\nfrom mathutils import clamp\n"
        "assert re.compile('x').match('x') and clamp(1, 0, 2) == 1\n"
    )
    assert script_check_tier(source, base=base, named_in_criterion=False) == (
        CheckTier.A,
        "script_imports_resolve",
    )
