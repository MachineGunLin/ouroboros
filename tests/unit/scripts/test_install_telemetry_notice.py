"""scripts/install.sh records the notice version only when a person saw the notice.

The installer's ``_telemetry_notice`` and ``_telemetry_notice_reaches_a_person``
are extracted from the script and run in bash with stubbed identity and
output helpers, so the real recording logic is exercised (R2-T1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pty
import re
import subprocess

import pytest

INSTALL_SH = Path(__file__).resolve().parents[3] / "scripts" / "install.sh"
STUBS = """
BOLD=""; RESET=""; DIM=""
_telemetry_enabled() { return 0; }
_telemetry_distinct_id() { printf '%s' "11111111-1111-4111-8111-111111111111"; }
_say() { printf '%s\\n' "$*"; }
_blank() { printf '\\n'; }
_info() { _say "$1"; }
"""


def _function(source: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", source, re.MULTILINE | re.DOTALL)
    assert match is not None, name
    return match.group(0)


def _notice_script() -> str:
    source = INSTALL_SH.read_text(encoding="utf-8")
    version = re.search(r"^TELEMETRY_NOTICE_VERSION=\d+$", source, re.MULTILINE)
    assert version is not None
    return "\n".join(
        [
            STUBS,
            version.group(0),
            _function(source, "_telemetry_notice_reaches_a_person"),
            _function(source, "_telemetry_notice"),
            "_telemetry_notice",
        ]
    )


def _run(home: Path, *, ci: str | None, tty: bool) -> str:
    state = home / ".ouroboros" / "telemetry.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps({"distinct_id": "11111111-1111-4111-8111-111111111111", "notice_shown": False})
    )
    env = {key: value for key, value in os.environ.items() if key != "CI"}
    env["HOME"] = str(home)
    if ci is not None:
        env["CI"] = ci
    script = _notice_script()
    if tty:
        leader, follower = pty.openpty()
        try:
            subprocess.run(
                ["bash", "-c", script],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=follower,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=30,
            )
        finally:
            os.close(follower)
            os.close(leader)
    else:
        output = subprocess.run(
            ["bash", "-c", script],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        assert "Anonymous usage stats" in output  # printed either way
    return state.read_text()


@pytest.mark.parametrize(
    ("ci", "tty", "recorded"),
    [
        (None, True, True),
        ("false", True, True),
        ("true", True, False),
        ("1", True, False),
        (None, False, False),
        ("true", False, False),
    ],
)
def test_notice_is_recorded_only_on_a_terminal_outside_ci(
    tmp_path: Path, ci: str | None, tty: bool, recorded: bool
) -> None:
    state = json.loads(_run(tmp_path, ci=ci, tty=tty))
    # Printing is recorded either way; the version that randomized defaults
    # require only when a person saw the notice.
    assert state["notice_shown"] is True
    if recorded:
        assert state["notice_version"] == 4
    else:
        assert "notice_version" not in state


@pytest.mark.parametrize(
    ("before", "version_after"),
    [
        ('{"distinct_id": "11111111-1111-4111-8111-111111111111", "notice_shown": false}', None),
        (
            '{"distinct_id": "11111111-1111-4111-8111-111111111111", "notice_shown": true, '
            '"notice_version": 3}',
            3,
        ),
    ],
)
def test_without_python3_no_version_is_recorded_without_a_person(
    tmp_path: Path, before: str, version_after: int | None
) -> None:
    # The sed fallback follows the same rule: a piped install marks the
    # notice printed and leaves notice_version as it was.
    from tests.unit.scripts.test_install_runtime_selection import _build_no_python3_path

    home = tmp_path / "home"
    state = home / ".ouroboros" / "telemetry.json"
    state.parent.mkdir(parents=True)
    state.write_text(before + "\n")
    env = {key: value for key, value in os.environ.items() if key != "CI"}
    env["HOME"] = str(home)
    env["PATH"] = _build_no_python3_path(tmp_path / "no-python-bin")
    output = subprocess.run(
        ["/bin/bash", "-c", _notice_script()],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    assert "Anonymous usage stats" in output
    after = json.loads(state.read_text())
    assert after["notice_shown"] is True
    assert after.get("notice_version") == version_after
