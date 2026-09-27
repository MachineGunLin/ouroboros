"""Process environment and interpreter for model-written check scripts.

Admission and candidate verification execute Python scripts that a model
wrote, and an oracle check runs the implementation under test in target
processes. They always run on a throwaway copy of the checkout, without a
shell, under the per-check timeout (``boundary/admission.py``,
``boundary/oracle_run.py``). This module adds two product-path rules:

- **Environment built from scratch.** Every such process gets
  ``check_environment``: a new mapping, not a filtered copy of this
  process's environment. Only the variables named in ``CHECK_ENV_COPIED``
  (``PATH``, the locale, Python I/O settings, and on Windows the system
  variables any process needs) take their value from the value source (this
  process by default); ``HOME`` (``USERPROFILE``, ``APPDATA`` and
  ``LOCALAPPDATA`` on Windows) and the temp directory point to a scratch
  directory created for the process and removed afterwards; ``VIRTUAL_ENV``
  names the interpreter's virtualenv, whose scripts directory leads ``PATH``.
  No other variable exists in the child, so a credential held in an
  environment variable, or in a file found through ``HOME``, is not in a
  check's environment. ``PYTHONPATH`` is never copied: a relative entry would
  resolve inside the checkout copy ahead of the standard library.
  ``check_process_environment`` owns the scratch directory's lifetime; the
  two spawn points (``admission._run_argv`` and
  ``oracle_run.run_oracle_check``) use it, so no caller can hand a check the
  parent environment.
- **Project interpreter.** A check's ``python3``/``python`` runs with the
  project's virtualenv interpreter when one is found (the checkout's, or the
  main working tree's when the checkout is a linked git worktree, then an
  active ``VIRTUAL_ENV``), else ``python3`` from ``PATH``. The choice is
  recorded in the admission and verification receipts.

Residual risk, not addressed here: there is no OS sandbox. A check runs as
the user, so it can read files the user can read (including, on Linux,
another of the user's processes' ``/proc/<pid>/environ``), write outside its
copy, and use the network, as the worker agent can. Opt out with
``--no-check-package``, ``OUROBOROS_CHECK_PACKAGE=off``, or
``boundary.check_package: off``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile

# The only variables whose value a check process takes from the value source.
CHECK_ENV_COPIED: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_COLLATE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
)
# Windows cannot start a process, or find its system directories, without these.
CHECK_ENV_COPIED_WINDOWS: tuple[str, ...] = (
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "PROGRAMDATA",
)
CHECK_INTERPRETER_NAMES = frozenset({"python3", "python"})


def _interpreter_venv(interpreter: str | None) -> Path | None:
    """The virtualenv that ``interpreter`` belongs to (its ``pyvenv.cfg``), if any."""
    if not interpreter or not Path(interpreter).is_absolute():
        return None
    root = Path(interpreter).parent.parent
    return root if (root / "pyvenv.cfg").is_file() else None


def check_environment(
    scratch: Path,
    *,
    interpreter: str | None = None,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The complete environment of a check process, rooted at the directory ``scratch``.

    ``source`` (default: this process's environment) supplies values for the
    names in ``CHECK_ENV_COPIED`` (and ``CHECK_ENV_COPIED_WINDOWS`` on
    Windows) only; nothing else is read from it. ``scratch`` must exist; the
    temp directory is created inside it.
    """
    values = os.environ if source is None else source
    names = CHECK_ENV_COPIED + (CHECK_ENV_COPIED_WINDOWS if sys.platform == "win32" else ())
    env = {name: values[name] for name in names if values.get(name)}
    home = scratch / "home"
    temp = scratch / "tmp"
    home.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["TMPDIR"] = str(temp)
    if sys.platform == "win32":
        env["USERPROFILE"] = str(home)
        env["APPDATA"] = str(home / "AppData" / "Roaming")
        env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
        env["TEMP"] = env["TMP"] = str(temp)
    venv = _interpreter_venv(interpreter)
    if venv is not None:
        env["VIRTUAL_ENV"] = str(venv)
        scripts = Path(interpreter or "").parent
        env["PATH"] = os.pathsep.join(filter(None, (str(scripts), env.get("PATH"))))
    return env


@contextmanager
def check_process_environment(
    source: Mapping[str, str] | None = None,
    *,
    interpreter: str | None = None,
    parent: Path | None = None,
) -> Iterator[dict[str, str]]:
    """``check_environment`` on a fresh owner-only scratch directory, removed on exit.

    The directory is created in ``parent`` (the run's work directory, beside
    the checkout copy and never inside it) or, without one, in the system
    temp directory.
    """
    scratch = Path(tempfile.mkdtemp(prefix="ouroboros-check-env-", dir=parent))
    try:
        yield check_environment(scratch, interpreter=interpreter, source=source)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class CheckInterpreter:
    """The interpreter that runs check scripts, and why it was chosen."""

    path: str
    source: str
    """``project_venv``, ``active_venv``, or ``python3_fallback``."""


def _python_in_venv(venv: Path) -> Path | None:
    names = ("Scripts/python.exe",) if sys.platform == "win32" else ("bin/python3", "bin/python")
    for name in names:
        candidate = venv / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _venv_python(root: Path) -> Path | None:
    for venv in (".venv", "venv"):
        found = _python_in_venv(root / venv)
        if found is not None:
            return found
    return None


def _main_worktree_root(checkout: Path) -> Path | None:
    """The main working tree of a linked git worktree, read from its ``.git`` file."""
    marker = checkout / ".git"
    try:
        if not marker.is_file():
            return None
        text = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text.removeprefix("gitdir:").strip())
    if not gitdir.is_absolute():
        gitdir = (checkout / gitdir).resolve()
    # <main>/.git/worktrees/<name>
    if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
        return gitdir.parent.parent.parent
    return None


def resolve_check_interpreter(
    checkout: Path, environ: Mapping[str, str] | None = None
) -> CheckInterpreter:
    """Pick the interpreter for ``python3``/``python`` in a check's argv."""
    roots = [checkout]
    main_root = _main_worktree_root(checkout)
    if main_root is not None:
        roots.append(main_root)
    for root in roots:
        found = _venv_python(root)
        if found is not None:
            return CheckInterpreter(str(found), "project_venv")
    source = os.environ if environ is None else environ
    active = source.get("VIRTUAL_ENV", "").strip()
    if active:
        found = _python_in_venv(Path(active))
        if found is not None:
            return CheckInterpreter(str(found), "active_venv")
    return CheckInterpreter(shutil.which("python3") or "python3", "python3_fallback")


__all__ = [
    "CHECK_ENV_COPIED",
    "CHECK_ENV_COPIED_WINDOWS",
    "CHECK_INTERPRETER_NAMES",
    "CheckInterpreter",
    "check_environment",
    "check_process_environment",
    "resolve_check_interpreter",
]
