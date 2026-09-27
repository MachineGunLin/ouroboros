"""Controller-private directory for the oracle comparator.

The comparator of an oracle check (``boundary/oracle_run.py``) runs with this
directory as its working directory. It is created fresh for each check with
``tempfile.mkdtemp`` (mode 0700, owner only), in the system temporary
directory, never beside the checkout copy the target runs in, so its location
cannot be derived from the target's working directory. Nothing in it holds an
expected value: the comparator receives the frozen oracle data over stdin.

It is digested before and after the check, so a file the target manages to
create or change there is reported as a protected-byte mutation
(``ctrl:<path>``), which makes the check indeterminate.

Reading is not prevented beyond the mode: there is no OS sandbox, and code
running as the same user can read any file that user can read.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil
import tempfile

from ouroboros.boundary.tree import tree_manifest

CONTROLLER_PREFIX = "ctrl:"


def create_controller_dir() -> Path:
    """A new owner-only (0700) directory outside every checkout copy."""
    return Path(tempfile.mkdtemp(prefix="ouroboros-ctrl-"))


def controller_mutations(ctrl: Path, before: Mapping[str, str]) -> tuple[str, ...]:
    """Controller files that changed, disappeared, or appeared during the run."""
    after = tree_manifest(ctrl, unprotected_names=())
    changed = {path for path, digest in before.items() if after.get(path) != digest}
    changed.update(set(after) - set(before))
    return tuple(f"{CONTROLLER_PREFIX}{path}" for path in sorted(changed))


def remove_controller_dir(ctrl: Path) -> None:
    shutil.rmtree(ctrl, ignore_errors=True)


__all__ = [
    "CONTROLLER_PREFIX",
    "controller_mutations",
    "create_controller_dir",
    "remove_controller_dir",
]
