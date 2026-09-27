"""The check package switch: on by default, explicit settings win (boundary/rollout.py)."""

from __future__ import annotations

import pytest

from ouroboros.boundary.rollout import (
    Arm,
    AssignmentSource,
    resolve_check_package_assignment,
)
from ouroboros.config.untrusted_env import is_untrusted_env_denied_key


def _resolve(
    cli: bool | None = None,
    *,
    env: str | None = None,
    configured: str | None = None,
):
    environ = {} if env is None else {"OUROBOROS_CHECK_PACKAGE": env}
    return resolve_check_package_assignment(cli, configured=configured, environ=environ)


@pytest.mark.parametrize(
    ("kwargs", "arm", "source"),
    [
        ({"cli": True, "env": "off", "configured": "off"}, "on", "user_forced_on"),
        ({"cli": False, "env": "on", "configured": "on"}, "off", "user_forced_off"),
        ({"env": "on", "configured": "off"}, "on", "user_forced_on"),
        ({"env": "off", "configured": "on"}, "off", "user_forced_off"),
        # T2: a set but unreadable value is an opt-out (with a warning), never the default.
        ({"env": "garbage", "configured": "on"}, "off", "user_forced_off"),
        ({"env": "disable"}, "off", "user_forced_off"),
        ({"configured": "off"}, "off", "user_forced_off"),
        ({"configured": "on"}, "on", "user_forced_on"),
        ({}, "on", "default"),
        ({"env": ""}, "on", "default"),
    ],
)
def test_explicit_settings_override_the_default(kwargs: dict, arm: str, source: str) -> None:
    assignment = _resolve(kwargs.pop("cli", None), **kwargs)
    assert (assignment.arm.value, assignment.source.value) == (arm, source)


def test_the_default_depends_on_nothing_but_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """No telemetry, no anonymous ID, no notice: the check package is still on."""
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("OUROBOROS_TELEMETRY", "0")
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("OUROBOROS_CHECK_PACKAGE", raising=False)
    assignment = resolve_check_package_assignment(None, configured=None)
    assert (assignment.arm, assignment.source) == (Arm.ON, AssignmentSource.DEFAULT)


def test_project_env_cannot_set_the_switch() -> None:
    assert is_untrusted_env_denied_key("OUROBOROS_CHECK_PACKAGE")
