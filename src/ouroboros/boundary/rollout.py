"""The on/off switch for the check package boundary of ``ooo run``.

The check package is on by default for every eligible run. An explicit user
setting turns it off (or on). Precedence, first match wins:

1. the CLI flag ``--check-package`` / ``--no-check-package``;
2. the environment variable ``OUROBOROS_CHECK_PACKAGE=on|off``;
3. ``boundary.check_package: on|off`` in ``~/.ouroboros/config.yaml``;
4. otherwise ``on``, recorded as ``default``.

The switch depends on nothing else: not on telemetry, not on the anonymous
ID, not on any notice. The switch and its source are recorded in the local
journal when the worker starts (``boundary.actor.started``), so a resumed run
reports them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import os

CHECK_PACKAGE_ENV = "OUROBOROS_CHECK_PACKAGE"


class Arm(StrEnum):
    """Whether the check package boundary runs for this run."""

    ON = "on"
    OFF = "off"


class AssignmentSource(StrEnum):
    """Why the switch has its value."""

    DEFAULT = "default"
    USER_FORCED_ON = "user_forced_on"
    USER_FORCED_OFF = "user_forced_off"
    # The settings could not be resolved (an unexpected error): off, so a
    # fault in the switch never fails a run (``run_control.CheckPackageRun``).
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class CheckPackageAssignment:
    """The resolved switch for one run and where it came from."""

    arm: Arm
    source: AssignmentSource

    @property
    def enabled(self) -> bool:
        return self.arm is Arm.ON


def parse_switch(value: str) -> bool | None:
    """Parse an on/off spelling; ``None`` for anything else (including empty)."""
    normalized = value.strip().lower()
    if normalized in {"on", "1", "true", "yes"}:
        return True
    if normalized in {"off", "0", "false", "no"}:
        return False
    return None


def _forced(enabled: bool) -> CheckPackageAssignment:
    if enabled:
        return CheckPackageAssignment(Arm.ON, AssignmentSource.USER_FORCED_ON)
    return CheckPackageAssignment(Arm.OFF, AssignmentSource.USER_FORCED_OFF)


def resolve_check_package_assignment(
    cli_value: bool | None,
    *,
    configured: str | None,
    environ: Mapping[str, str] | None = None,
) -> CheckPackageAssignment:
    """Resolve the switch with the precedence in the module docstring.

    ``configured`` is ``boundary.check_package`` from config (``"on"``,
    ``"off"``, or ``None`` when unset).
    """
    if cli_value is not None:
        return _forced(cli_value)
    env = os.environ if environ is None else environ
    raw = env.get(CHECK_PACKAGE_ENV, "")
    from_env = parse_switch(raw)
    if from_env is not None:
        return _forced(from_env)
    if raw.strip():
        # A set but unreadable value (a typo such as "disable") is an attempt
        # to opt out: treat it as off, never as the default.
        import structlog

        structlog.get_logger(__name__).warning(
            "boundary.rollout.unparsable_switch", variable=CHECK_PACKAGE_ENV
        )
        return _forced(False)
    if configured in {"on", "off"}:
        return _forced(configured == "on")
    return CheckPackageAssignment(Arm.ON, AssignmentSource.DEFAULT)


__all__ = [
    "CHECK_PACKAGE_ENV",
    "Arm",
    "AssignmentSource",
    "CheckPackageAssignment",
    "parse_switch",
    "resolve_check_package_assignment",
]
