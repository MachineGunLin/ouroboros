"""Byte-level manifests of checkout trees for protected-byte checks.

A manifest maps every workspace-relative file path to the SHA-256 of its bytes
(or to ``symlink:<target>`` for a symbolic link). A file or directory that
cannot be read, and anything that is neither a regular file, a directory nor a
symbolic link (a named pipe, a socket, a device), maps to ``UNREADABLE``: the
manifest is still computed, and a caller that must copy or judge the tree
sees which paths it cannot use (``unreadable_paths``). Paths whose components match
an unprotected name (by default version-control metadata and regenerable
interpreter or tool caches) are left out: running a test legitimately rewrites
them, and flagging them would make every run look like a mutation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import os
from pathlib import Path
import shutil
import stat

DEFAULT_UNPROTECTED_NAMES: frozenset[str] = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"}
)

_CHUNK = 1024 * 1024
UNREADABLE = "unreadable"


_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


def _file_sha256(path: Path) -> str:
    """SHA-256 of a regular file, read without following a link or blocking on a pipe.

    A path swapped for a link, a pipe or a device after it was listed raises
    ``OSError`` (the caller records it as ``UNREADABLE``).
    """
    descriptor = os.open(path, _READ_FLAGS)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(
    root: Path,
    *,
    unprotected_names: Iterable[str] = DEFAULT_UNPROTECTED_NAMES,
) -> dict[str, str]:
    """Return ``{relative_path: digest}`` for every protected file under ``root``."""
    skip = frozenset(unprotected_names)
    base = root.resolve()
    manifest: dict[str, str] = {}

    def unreadable_directory(error: OSError) -> None:
        if error.filename is not None:
            path = Path(error.filename)
            if path != base and not any(part in skip for part in path.relative_to(base).parts):
                manifest[path.relative_to(base).as_posix()] = UNREADABLE

    walk = os.walk(base, followlinks=False, onerror=unreadable_directory)
    for dirpath, dirnames, filenames in walk:
        current = Path(dirpath)
        kept: list[str] = []
        for name in dirnames:
            if name in skip:
                continue
            full = current / name
            if full.is_symlink():
                manifest[full.relative_to(base).as_posix()] = f"symlink:{os.readlink(full)}"
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            if name in skip:
                continue
            full = current / name
            relative = full.relative_to(base).as_posix()
            if full.is_symlink():
                manifest[relative] = f"symlink:{os.readlink(full)}"
            elif full.is_file():
                try:
                    manifest[relative] = _file_sha256(full)
                except OSError:
                    manifest[relative] = UNREADABLE
            else:
                manifest[relative] = UNREADABLE
    return manifest


def unreadable_paths(manifest: Mapping[str, str]) -> tuple[str, ...]:
    """Paths of ``manifest`` whose bytes could not be read (``UNREADABLE``)."""
    return tuple(sorted(path for path, value in manifest.items() if value == UNREADABLE))


def manifest_digest(manifest: Mapping[str, str]) -> str:
    """Return one SHA-256 over a manifest, independent of insertion order."""
    digest = hashlib.sha256()
    for path in sorted(manifest):
        digest.update(path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(manifest[path].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def tree_digest(
    root: Path,
    *,
    unprotected_names: Iterable[str] = DEFAULT_UNPROTECTED_NAMES,
) -> str:
    """Return the manifest digest of ``root``; the identity of a checkout artifact."""
    return manifest_digest(tree_manifest(root, unprotected_names=unprotected_names))


def changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    """Return paths present in ``before`` that are modified or missing in ``after``."""
    return tuple(sorted(path for path, value in before.items() if after.get(path) != value))


def added_paths(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    """Return paths present in ``after`` but not in ``before``."""
    return tuple(sorted(set(after) - set(before)))


def copy_checkout(source: Path, destination: Path) -> None:
    """Copy a checkout, preserving symlinks, into a new ``destination``.

    ``__pycache__`` is not copied: it is outside the protected digest, so a
    planted ``.pyc`` matching a source file's mtime and size could otherwise
    run in place of the reviewed source.
    """
    shutil.copytree(
        source, destination, symlinks=True, ignore=shutil.ignore_patterns("__pycache__")
    )
