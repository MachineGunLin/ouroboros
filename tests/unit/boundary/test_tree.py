"""Byte manifests never follow a link or block on a special file."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from ouroboros.boundary import tree
from ouroboros.boundary.tree import UNREADABLE, tree_manifest, unreadable_paths


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX named pipes")
def test_a_named_pipe_is_unreadable_and_never_blocks(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("x = 1\n")
    os.mkfifo(tmp_path / "pipe")

    manifest = tree_manifest(tmp_path)

    assert manifest["pipe"] == UNREADABLE
    assert unreadable_paths(manifest) == ("pipe",)
    assert manifest["code.py"] != UNREADABLE


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_hashing_a_path_swapped_for_a_link_refuses_it(tmp_path: Path) -> None:
    # A file replaced by a link after it was listed is not read through the link.
    secret = tmp_path / "secret.txt"
    secret.write_text("outside\n")
    swapped = tmp_path / "swapped.py"
    swapped.symlink_to(secret)

    with pytest.raises(OSError):
        tree._file_sha256(swapped)
