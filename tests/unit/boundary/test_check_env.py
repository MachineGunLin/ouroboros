"""Environment and interpreter for model-written checks (boundary/check_env.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from ouroboros.boundary import admit_check_package, verify_candidate
from ouroboros.boundary.admission import _run_argv
from ouroboros.boundary.check_env import (
    check_environment,
    check_process_environment,
    resolve_check_interpreter,
)
from ouroboros.boundary.package import CheckRole
from ouroboros.boundary.run_wiring import CheckPackageSettings, prepare_check_package
from ouroboros.persistence.event_store import EventStore

from .test_oracle import FIXED, _repo
from .test_oracle import _package as _oracle_package
from .test_oracle import _seed as _oracle_seed
from .test_run_wiring import FakeConstructor, _ok, _package, _seed

SECRETS = {
    "OPENAI_API_KEY": "sk-test",
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "GH_TOKEN": "ghp_test",
    "GITHUB_TOKEN": "ghs_test",
    "AWS_SECRET_ACCESS_KEY": "aws-test",
    "MY_SERVICE_PASSWORD": "hunter2",
}
# A parent variable no name pattern would call a credential: only an
# environment built from an allowlist keeps it out.
SENTINEL_NAME = "OUROBOROS_TEST_PARENT_ONLY"
SENTINEL_VALUE = "parent-credential-sentinel-7f3a"

# Preservation check: passes only when no credential-like variable is visible.
NO_SECRET_SCRIPT = """import os, sys
leaked = sorted(k for k in os.environ if any(t in k for t in ("KEY", "TOKEN", "SECRET", "PASSWORD")))
print("leaked:", leaked)
sys.exit(1 if leaked else 0)
"""

POSIX_KEYS = {"PATH", "HOME", "TMPDIR"}


def test_the_environment_is_built_from_the_allowlist_not_copied(tmp_path: Path) -> None:
    source = {
        "PATH": "/bin",
        "HOME": "/home/user",
        "LC_ALL": "C",
        "VIRTUAL_ENV": "/other/venv",
        "PYTHONPATH": ".",
        "LC_SECRET_TOKEN": "a name pattern would have kept this",
        SENTINEL_NAME: SENTINEL_VALUE,
        **SECRETS,
    }
    env = check_environment(tmp_path, source=source)
    expected = POSIX_KEYS | {"LC_ALL"}
    if sys.platform == "win32":
        expected |= {"USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"}
    assert set(env) == expected
    assert env["PATH"] == "/bin" and env["LC_ALL"] == "C"
    # HOME and the temp directory are the scratch directory, never the parent's.
    assert env["HOME"] == str(tmp_path / "home") and (tmp_path / "home").is_dir()
    assert env["TMPDIR"] == str(tmp_path / "tmp") and (tmp_path / "tmp").is_dir()
    assert SENTINEL_VALUE not in json.dumps(env)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv layout")
def test_a_virtualenv_interpreter_names_its_venv_and_leads_path(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    python = venv / "bin" / "python3"
    python.symlink_to(sys.executable)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = check_environment(scratch, interpreter=str(python), source={"PATH": "/bin"})
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"] == os.pathsep.join((str(venv / "bin"), "/bin"))
    plain = check_environment(scratch, interpreter="python3", source={"PATH": "/bin"})
    assert "VIRTUAL_ENV" not in plain and plain["PATH"] == "/bin"


def test_the_scratch_directory_is_removed_afterwards() -> None:
    with check_process_environment({"PATH": "/bin"}) as env:
        scratch = Path(env["HOME"]).parent
        assert scratch.is_dir()
    assert not scratch.exists()


async def test_a_script_check_never_sees_a_parent_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real child process, launched the way every script check is (``_run_argv``)."""
    monkeypatch.setenv(SENTINEL_NAME, SENTINEL_VALUE)
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    parent_home = os.environ.get("HOME", "")
    argv = ["python3", "-c", "import json, os; print(json.dumps(dict(os.environ)))"]
    # No ``env``: the library default must not hand the child this process's environment.
    completed = await _run_argv(argv, tmp_path, 30, interpreter=sys.executable)
    assert completed.return_code == 0, completed.stderr
    child = json.loads(completed.stdout)
    assert SENTINEL_NAME not in child
    assert not set(SECRETS) & set(child)
    assert SENTINEL_VALUE not in completed.stdout.decode()
    assert not any(value in completed.stdout.decode() for value in SECRETS.values())
    assert child["HOME"] != parent_home
    assert not Path(child["HOME"]).exists()  # the scratch directory is gone
    if sys.platform != "win32":
        # Only the built names (VIRTUAL_ENV: this interpreter's venv); the OS
        # itself may add LC_CTYPE or macOS's __CF_USER_TEXT_ENCODING at exec.
        assert set(child) - {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"} <= (
            POSIX_KEYS | {"VIRTUAL_ENV"} | set(_copied_names())
        )


def _copied_names() -> tuple[str, ...]:
    from ouroboros.boundary.check_env import CHECK_ENV_COPIED

    return CHECK_ENV_COPIED


async def test_an_oracle_target_process_never_sees_a_parent_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The implementation under an oracle check runs in the same built environment."""
    monkeypatch.setenv(SENTINEL_NAME, SENTINEL_VALUE)
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    report = tmp_path / "target-env.json"
    probe = f"import json, os\nopen({str(report)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
    base = _repo(tmp_path / "base", {"mathutils.py": FIXED})
    candidate = _repo(tmp_path / "cand", {"mathutils.py": probe + FIXED})
    package = _oracle_package(_oracle_seed(), base)
    result = await verify_candidate(package, candidate)
    assert result.verdict.value == "pass"
    seen = json.loads(report.read_text())
    assert SENTINEL_NAME not in seen and not set(SECRETS) & set(seen)
    assert SENTINEL_VALUE not in report.read_text()
    assert seen["HOME"] != os.environ.get("HOME")


def _preservation_package(seed):
    package = _package(seed, "keep_env", NO_SECRET_SCRIPT)
    check = package.checks[0]
    return package.model_copy(
        update={
            "checks": (
                check.model_copy(
                    update={"role": CheckRole.PRESERVATION, "failure_signature": None}
                ),
            )
        }
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    return root


async def _prepare(seed, package, repo: Path, tmp_path: Path):
    store = EventStore("sqlite+aiosqlite:///:memory:")
    await store.initialize()
    try:
        return await prepare_check_package(
            seed,
            event_store=store,
            constructor=FakeConstructor(_ok(package)),
            execution_id="exec_env",
            base_checkout=repo,
            worker_workspace=repo,
            runtime_label="codex",
            settings=CheckPackageSettings(enabled=True, max_construction_attempts=1),
            store_dir=tmp_path / "store",
        )
    finally:
        await store.close()


async def test_product_admission_hides_credentials_from_checks(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credential hiding only; independent of which interpreter is detected."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    seed = _seed("add(2, 3) returns 5")
    package = _preservation_package(seed)

    # The library default builds the same environment: no caller can hand a
    # check the parent's variables.
    library = await admit_check_package(package, repo)
    assert library.verdict.value == "admitted", library.reasons

    state = await _prepare(seed, package, repo, tmp_path)
    assert state.admitted, state.failure_reason


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv layout")
@pytest.mark.parametrize("active", [True, False], ids=["active_venv", "python3_fallback"])
async def test_product_admission_records_the_detected_interpreter(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, active: bool
) -> None:
    """Interpreter selection follows VIRTUAL_ENV (the detector's only env input)."""
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    if active:
        venv = tmp_path / "active-env"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python3").symlink_to(sys.executable)
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    else:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    seed = _seed("add(2, 3) returns 5")
    state = await _prepare(seed, _preservation_package(seed), repo, tmp_path)

    expected = "active_venv" if active else "python3_fallback"
    # Admitted means the credential check passed under either interpreter.
    assert state.admitted, state.failure_reason
    assert state.admission is not None
    assert state.admission.interpreter_source == expected
    if active:
        assert state.admission.interpreter == str(venv / "bin" / "python3")
    summary = state.admission.event_summary()
    assert summary["interpreter_source"] == expected
    assert "interpreter" not in summary  # the absolute path stays in the stored receipt


def _fake_venv(root: Path) -> Path:
    bin_dir = root / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python3"
    python.symlink_to(sys.executable)
    return python


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv layout")
def test_interpreter_prefers_the_project_venv(tmp_path: Path) -> None:
    checkout = tmp_path / "project"
    python = _fake_venv(checkout)
    chosen = resolve_check_interpreter(checkout, environ={})
    assert (chosen.path, chosen.source) == (str(python), "project_venv")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv layout")
def test_linked_worktree_uses_the_main_tree_venv(tmp_path: Path) -> None:
    main = tmp_path / "main"
    python = _fake_venv(main)
    (main / ".git" / "worktrees" / "task").mkdir(parents=True)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'task'}\n")
    chosen = resolve_check_interpreter(worktree, environ={})
    assert (chosen.path, chosen.source) == (str(python), "project_venv")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv layout")
def test_active_virtualenv_then_python3_fallback(tmp_path: Path) -> None:
    active = tmp_path / "active-env"
    (active / "bin").mkdir(parents=True)
    (active / "bin" / "python3").symlink_to(sys.executable)
    chosen = resolve_check_interpreter(tmp_path / "plain", environ={"VIRTUAL_ENV": str(active)})
    assert chosen.source == "active_venv"
    fallback = resolve_check_interpreter(tmp_path / "plain", environ={})
    assert fallback.source == "python3_fallback"
    assert os.path.basename(fallback.path).startswith("python3")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv layout")
async def test_admission_runs_checks_with_the_resolved_interpreter(
    repo: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "used-interpreter"
    wrapper = repo / ".venv" / "bin" / "python3"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\necho used > "{marker}"\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    seed = _seed("add(2, 3) returns 5")
    package = _preservation_package(seed)
    chosen = resolve_check_interpreter(repo, environ={})
    result = await admit_check_package(
        package,
        repo,
        env={"PATH": os.environ.get("PATH", "")},
        interpreter=chosen.path,
        interpreter_source=chosen.source,
    )
    assert result.verdict.value == "admitted", result.reasons
    assert marker.read_text().strip() == "used"
    assert result.interpreter_source == "project_venv"
