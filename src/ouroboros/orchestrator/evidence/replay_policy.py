"""Which transcript commands the harness may replay, and what each one runs.

Replay executes commands the leaf ran, so the default is to refuse. The
program a command really runs is found by peeling known wrappers (``timeout``,
``env``, ``nice``, ...) and launchers (``uv run``, ``npx``, ``bundle exec``,
...) with per-program option tables; when an option or operand cannot be
classified with certainty, the program is unknown and nothing is replayed.

A command is replayed only when that program is one of:

(a) a test or build runner or an interpreter on the allowlist below, with a
    subcommand, target or script name that does not install, deploy or
    publish:

    - ``python``/``python3``/``pythonX.Y`` with ``-m pytest``, ``-m unittest``,
      ``-m tox``, ``-m nox``, ``-m django test``, or a script inside the
      workspace (never ``-c``, ``-`` or stdin);
    - ``pytest``, ``py.test``, ``tox``, ``nox``, ``django-admin test``;
    - ``make``/``gmake`` (targets not ``install``, ``uninstall``, ``deploy``,
      ``publish``, ``release``, ``upload`` or ``push``; no ``-C``/``-f``, no
      dry run);
    - ``npm``/``pnpm``/``yarn``/``bun`` ``test`` or ``run <script>`` (not an
      install, publish or deploy script); ``bun test``;
    - ``go test|vet|build``, ``cargo test|check|build|clippy|nextest``,
      ``mvn``/``mvnw`` with a ``test`` or ``verify`` goal, ``gradle``/
      ``gradlew`` with a ``test``, ``check`` or ``build`` task,
      ``dotnet test|build``, ``deno test``, ``swift test|build``, ``mix test``;
    - ``rspec``, ``jest``, ``vitest``, ``mocha``, ``ava``, ``phpunit``,
      ``ctest``;
    - the same runners reached through ``uv run``, ``uvx``, ``poetry run``,
      ``pipenv run``, ``pdm run``, ``bundle exec``, ``npx`` or ``bunx``;

(b) a script inside the workspace, run directly (``./run_tests.sh``,
    ``bin/test``) or by ``python``/``sh``/``bash`` (``tests/runtests.py``).

An absolute-path program outside the workspace is admitted only when its
name is an allowlisted interpreter or runner (``python3.9``, ``pytest``,
``make``, ...) and its resolved real path lies inside a known environment
root (``environment_roots``: the verifying interpreter's ``sys.prefix`` and
``sys.base_prefix``, ``VIRTUAL_ENV``, ``CONDA_PREFIX``, and the directories
on the replay environment's ``PATH``), as with
``/opt/miniconda3/envs/testbed/bin/python -m pytest`` in a SWE-bench image.

Refused: every other program, including the file viewers and text utilities
in ``VIEWER_PROGRAMS``, version control, package managers, any other
absolute-path program outside the workspace, an absolute-path argument
outside the workspace (other than such an environment program), ``xargs`` (its argv comes from stdin), and a runner in a mode that
runs no tests (``--help``, ``--collect-only``, ``make -n``, ...). The
denylist in ``command_replay`` stays as a second layer.

The same resolution decides the test-target linkage rule
(``claim_target_operands``): a claim naming a test file or label is linked to
a command only when it is a positional operand (not an option value) of a
test runner that executes it, and the command neither excludes or narrows the
tests it runs nor changes what the runner collects, loads or imports through
configuration. Each of these disables the target rule and the runner-output
rules (``run_may_back_test_claim``); where a runner's semantics are unclear,
an option is treated as narrowing:

- an option that excludes or selects tests: ``--deselect``, ``--ignore``,
  ``-k`` and ``-m`` (pytest), ``-k`` (unittest, Django, project runner
  scripts), ``--tag``, ``--exclude-tag`` and ``--start-at``/``--start-after``
  (Django), and the entries of ``_TARGET_EXCLUDING_OPTIONS`` and
  ``_SHORT_EXCLUDING_OPTIONS``;
- pytest ``-o``/``--override-ini`` setting ``addopts``, ``python_files``,
  ``python_classes``, ``python_functions``, ``testpaths`` or
  ``norecursedirs`` (``addopts`` covers ``--deselect``, ``-k`` and ``-m``
  inside it);
- pytest ``-c``/``--config-file``, ``--rootdir`` and ``--confcutdir``, in any
  form: whether a named file or directory is the one pytest would use by
  default cannot be decided from the command, so any use counts;
- pytest ``-p`` unless the plugin is in ``NO_OP_PYTEST_PLUGINS``
  (``no:cacheprovider``); ``-p``/``--pattern`` for unittest, Django and
  project runner scripts, where it is a discovery pattern;
- an assignment on the command line (leading, or consumed by an ``env``
  wrapper) of a variable in ``NARROWING_ENVIRONMENT`` (the pytest variables,
  ``PYTHONPATH``, ``PYTHONHOME``, ``PYTHONSTARTUP``, ``PYTHONSAFEPATH``,
  ``DJANGO_SETTINGS_MODULE``, ``NODE_OPTIONS``, ``NODE_PATH``, ``RUBYOPT``,
  ``RUBYLIB``, ``BUNDLE_GEMFILE``, ``GOFLAGS``, ``CGO_ENABLED``) or with a
  prefix in ``NARROWING_ENVIRONMENT_PREFIXES`` (``JEST_``, ``VITEST_``),
  whatever the runner (``narrowing_variable``);
- the Python interpreter flags ``-P`` and ``-I``;
- an entry of ``_RUNNER_CONFIG_OPTIONS`` for the runner: Django ``--settings``
  (except ``runtests.py``'s documented default ``test_sqlite``,
  ``_DEFAULT_OPTION_VALUES``), ``--pythonpath`` and ``--testrunner``; jest and vitest
  configuration, selection, sharding and module-mapping options; mocha,
  phpunit and rspec configuration, filter, group and load-path options;
  ``go test`` ``-run``, ``-skip``, ``-tags``, ``-short``, ``-list``,
  ``-exec``, ``-mod``, ``-overlay``; Maven ``-P``, ``-pl``, ``-s``, ``-f``
  and test-selecting ``-D`` properties (``-Dtest=``); Gradle ``--tests``,
  ``-P``, ``-x`` and build-file options; ``cargo test`` and ``cargo nextest``
  target and feature options, and any positional filter or option outside
  ``_CARGO_FLAG_OPTIONS`` and ``_CARGO_VALUE_OPTIONS``.

An option narrows only when it changes which tests run or how modules and
settings resolve relative to the runner's documented default: a process
count (Django ``--parallel N``, pytest-xdist ``-n``, ``cargo -j``) never
does, and an option set to a documented default does not either.

Narrowing has two classes. ``SELECTION`` options only choose which tests run
(``-k``, ``--tests``, ``-Dtest=``, ``-run``, ``_RUNNER_SELECTION_OPTIONS``);
``CONFIGURATION`` (every other entry above: the environment, interpreter
flags, configuration files, settings, module resolution, plugin loading)
can make a named test pass against code other than the workspace's, so the
transcript-only rules do not accept even output that names the claimed test
(``alters_configuration``).

Replay also removes every narrowing variable from the environment the replay
inherits (``command_replay``). ``uv run --env-file`` is refused, since the
file may set those variables. Configuration the workspace itself carries
(``pytest.ini`` ``addopts``, ``conftest.py``, ``jest.config.js``) is part of
the work under review and out of scope here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import PurePosixPath
import re
import sys

from ouroboros.orchestrator.evidence.shell_parsing import (
    _PYTEST_NON_EXECUTING_OPTIONS,
    _UV_FLAG_OPTIONS,
    _UV_SHORT_FLAG_OPTIONS,
    _UV_SHORT_VALUE_OPTIONS,
    _UV_VALUE_OPTIONS,
    _has_gradle_or_maven_test_skip,
    _is_env_assignment,
    _is_python_executable,
    _project_test_runner_script,
)

_MAX_RESOLVE_DEPTH = 6
# Narrowing classes (``_selection``). ``SELECTION``: an option selects, skips
# or excludes tests by name, path or group. ``CONFIGURATION``: the runner's
# configuration, module resolution or environment is replaced.
SELECTION = "selection"
CONFIGURATION = "configuration"

# File viewers and text utilities: they read or list files, they never
# execute a test. Never replayed, and never a test-target runner.
VIEWER_PROGRAMS = frozenset(
    {
        "cat",
        "bat",
        "tac",
        "nl",
        "sed",
        "head",
        "tail",
        "less",
        "more",
        "most",
        "view",
        "vi",
        "vim",
        "nvim",
        "nano",
        "emacs",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ag",
        "ack",
        "awk",
        "gawk",
        "mawk",
        "wc",
        "ls",
        "tree",
        "find",
        "fd",
        "stat",
        "file",
        "du",
        "diff",
        "cmp",
        "comm",
        "git",
        "hg",
        "svn",
        "sort",
        "uniq",
        "cut",
        "tr",
        "paste",
        "column",
        "od",
        "xxd",
        "hexdump",
        "strings",
        "jq",
        "yq",
        "echo",
        "printf",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        "pwd",
        "which",
        "type",
        "true",
        "false",
        "test",
    }
)

# Programs whose argv cannot be known from the transcript, or that run a
# command line through a shell of their own. Never replayed.
REFUSED_WRAPPERS = frozenset({"xargs", "watch", "script", "flock", "parallel"})

# Target, script and task names that install, deploy or publish. A name is
# refused when any of its words (split on ``-``, ``_``, ``:`` and ``.``) is one
# of these.
_DENIED_TASK_WORDS = frozenset(
    {
        "install",
        "uninstall",
        "preinstall",
        "postinstall",
        "deploy",
        "publish",
        "prepublish",
        "prepublishonly",
        "release",
        "upload",
        "push",
    }
)
_TASK_WORD_SPLIT_RE = re.compile(r"[-_:.]+")
_GENERIC_NON_EXECUTING_OPTIONS = frozenset({"-h", "--help", "--version", "--dry-run"})
_DURATION_RE = re.compile(r"\d+(?:\.\d+)?[smhd]?")
_NUMERIC_OPTION_RE = re.compile(r"-\d+")


@dataclass(frozen=True, slots=True)
class _OptionSpec:
    """Options a wrapper or launcher accepts before the program it runs."""

    values: frozenset[str] = frozenset()
    flags: frozenset[str] = frozenset()
    refused: frozenset[str] = frozenset()
    positionals: int = 0
    positional_re: re.Pattern[str] | None = None
    assignments: bool = False
    numeric_flags: bool = False


def _spec(
    values: set[str] | frozenset[str] = frozenset(),
    flags: set[str] | frozenset[str] = frozenset(),
    refused: set[str] | frozenset[str] = frozenset(),
    **options: object,
) -> _OptionSpec:
    return _OptionSpec(
        values=frozenset(values),
        flags=frozenset(flags),
        refused=frozenset(refused),
        **options,  # type: ignore[arg-type]
    )


# Wrappers: they run the rest of the argv. Options taking a separate value are
# listed so the value is never mistaken for the program.
_WRAPPERS: Mapping[str, _OptionSpec] = {
    "timeout": _spec(
        {"-s", "--signal", "-k", "--kill-after"},
        {"--preserve-status", "--foreground", "-v", "--verbose", "-f", "-p"},
        positionals=1,
        positional_re=_DURATION_RE,
    ),
    "stdbuf": _spec({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "time": _spec(
        {"-o", "--output", "-f", "--format"},
        {"-p", "-a", "--append", "-v", "--verbose", "-q", "--quiet", "-l", "--portability"},
    ),
    "nice": _spec({"-n", "--adjustment"}, numeric_flags=True),
    "ionice": _spec(
        {"-c", "--class", "-n", "--classdata"},
        {"-t", "--ignore"},
        {"-p", "--pid", "-P", "--pgid", "-u", "--uid"},
    ),
    "env": _spec(
        {"-u", "--unset"},
        {"-i", "--ignore-environment", "-0", "--null", "-v", "--debug"},
        {"-C", "--chdir", "-S", "--split-string", "-P"},
        assignments=True,
    ),
    "nohup": _spec(),
    "command": _spec(flags={"-p"}, refused={"-v", "-V"}),
    "exec": _spec({"-a"}, {"-c", "-l"}),
    "setsid": _spec(flags={"-c", "--ctty", "-w", "--wait", "-f", "--fork"}),
}

_UV_RUN_SPEC = _spec(
    (_UV_VALUE_OPTIONS - {"--directory", "--project"})
    | {f"-{option}" for option in _UV_SHORT_VALUE_OPTIONS},
    _UV_FLAG_OPTIONS | {f"-{option}" for option in _UV_SHORT_FLAG_OPTIONS},
    {"--directory", "--project", "--script", "-s", "--gui-script", "-m", "--module"}
    | {"--env-file"},
)
# Launchers: they run another program by name. The launched program must pass
# the allowlist itself.
_LAUNCHERS: Mapping[tuple[str, ...], _OptionSpec] = {
    ("uv", "run"): _UV_RUN_SPEC,
    ("uvx",): _UV_RUN_SPEC,
    ("poetry", "run"): _spec(
        flags={"-q", "--quiet", "-v", "-vv", "-vvv", "--verbose", "-n", "--no-interaction"}
        | {"--ansi", "--no-ansi"},
        refused={"-C", "--directory", "-P", "--project"},
    ),
    ("pipenv", "run"): _spec(),
    ("pdm", "run"): _spec(
        flags={"-v", "-q", "--verbose", "--quiet"},
        refused={"-p", "--project", "-g", "--global"},
    ),
    ("bundle", "exec"): _spec(flags={"--keep-file-descriptors"}),
    ("npx",): _spec(
        {"-p", "--package"},
        {"-y", "--yes", "--no", "-q", "--quiet", "--no-install", "--ignore-existing"}
        | {"--prefer-offline", "--offline"},
        {"-c", "--call"},
    ),
    ("bunx",): _spec({"-p", "--package"}, {"--bun"}),
}

_PYTHON_FLAGS = frozenset(
    {"-u", "-B", "-O", "-OO", "-E", "-s", "-S", "-I", "-b", "-bb", "-q", "-P", "-d", "-R"}
)
_PYTHON_VALUE_OPTIONS = frozenset({"-W", "-X"})
_SHELL_INTERPRETERS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_DIRECT_RUNNERS = frozenset(
    {"pytest", "py.test", "tox", "nox", "rspec", "jest", "vitest", "mocha", "ava", "phpunit"}
    | {"ctest"}
)
# Runners with an allowed first subcommand; ``cargo`` also takes a
# ``+toolchain`` token before it.
_SUBCOMMAND_RUNNERS: Mapping[str, frozenset[str]] = {
    "go": frozenset({"test", "vet", "build"}),
    "cargo": frozenset({"test", "check", "build", "clippy", "nextest"}),
    "dotnet": frozenset({"test", "build"}),
    "deno": frozenset({"test"}),
    "swift": frozenset({"test", "build"}),
    "mix": frozenset({"test"}),
    "django-admin": frozenset({"test"}),
}
_JS_PACKAGE_RUNNERS = frozenset({"npm", "pnpm", "yarn", "bun"})
_JS_TEST_SUBCOMMANDS = frozenset({"test", "t", "tst"})
_JS_RUN_SUBCOMMANDS = frozenset({"run", "run-script"})
_JS_GLOBAL_FLAGS = frozenset({"-s", "--silent", "-q", "--quiet"})
_MAKE_REFUSED_OPTIONS = frozenset(
    {
        "-C",
        "--directory",
        "-f",
        "--file",
        "--makefile",
        "-n",
        "--just-print",
        "--dry-run",
        "--recon",
        "-q",
        "--question",
        "-t",
        "--touch",
        "-p",
        "--print-data-base",
    }
)
_MAVEN_TEST_GOALS = frozenset({"test", "verify"})
_GRADLE_TEST_TASKS = frozenset({"test", "check", "build"})

# Kinds whose positional operands select the tests that run.
TARGET_RUNNER_KINDS = frozenset(
    {"pytest", "unittest", "django", "test-script", "jest", "vitest", "mocha", "rspec"}
    | {"go-test", "phpunit"}
)
# Options that exclude or narrow the tests a runner executes. Any of them in
# the command disables the target rule and the runner-output rules: the named
# test may not have run.
_TARGET_EXCLUDING_OPTIONS = frozenset(
    {
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--exclude",
        "--exclude-tag",
        "--exclude-dir",
        "--exclude-pattern",
        "--exclude-group",
        "--skip",
        "--tag",
        "--start-at",
        "--start-after",
        "--testNamePattern",
        "--testPathIgnorePatterns",
        "--grep",
        "--invert",
        "--filter",
        "--group",
        "--example",
    }
)
# Short selection options, by runner (``-k`` is a keyword filter for pytest,
# unittest and Django; ``-m`` a pytest marker filter; ``-run``/``-skip`` for
# ``go test``; ``-t`` a name pattern for jest and vitest; ``-g`` for mocha;
# ``-e`` an example filter for rspec).
_SHORT_EXCLUDING_OPTIONS: Mapping[str, frozenset[str]] = {
    "pytest": frozenset({"-k", "-m"}),
    "unittest": frozenset({"-k"}),
    "django": frozenset({"-k"}),
    "test-script": frozenset({"-k"}),
    "go-test": frozenset({"-run", "-skip"}),
    "jest": frozenset({"-t"}),
    "vitest": frozenset({"-t"}),
    "mocha": frozenset({"-g"}),
    "rspec": frozenset({"-e"}),
}
# Options of the target runners that take a separate value.
_TARGET_VALUE_OPTIONS = frozenset(
    {
        "-m",
        "-p",
        "-c",
        "-o",
        "-W",
        "-n",
        "-r",
        "--rootdir",
        "--basetemp",
        "--junitxml",
        "--junit-xml",
        "--cov",
        "--cov-report",
        "--tb",
        "--maxfail",
        "--durations",
        "--confcutdir",
        "--log-level",
        "--color",
        "--import-mode",
        "--capture",
        "--dist",
        "--numprocesses",
        "--settings",
        "--parallel",
        "--timeout",
        "--config-file",
        "--override-ini",
        "--pattern",
    }
)
# Options of the target runners that take no value.
_TARGET_FLAG_OPTIONS = frozenset(
    {
        "-q",
        "-qq",
        "-v",
        "-vv",
        "-vvv",
        "-x",
        "-s",
        "-l",
        "-b",
        "-f",
        "-ra",
        "-rA",
        "-rf",
        "-rs",
        "-rx",
        "-rX",
        "-rE",
        "-rP",
        "--lf",
        "--ff",
        "--nf",
        "--sw",
        "--last-failed",
        "--failed-first",
        "--new-first",
        "--stepwise",
        "--exitfirst",
        "--quiet",
        "--verbose",
        "--no-header",
        "--no-summary",
        "--disable-warnings",
        "--showlocals",
        "--strict-markers",
        "--strict",
        "--failfast",
        "--buffer",
        "--noinput",
        "--no-input",
        "--keepdb",
        "--debug-sql",
        "--reverse",
        "--timing",
        "--runxfail",
        "--no-cov",
        "-race",
    }
)
# Django-style runners read ``-v`` as a verbosity level with a value.
_LABEL_RUNNER_VALUE_OPTIONS = frozenset({"-v", "--verbosity"})
# Runners whose ``-p``/``--pattern`` is a test discovery pattern.
_PATTERN_RUNNER_KINDS = frozenset({"unittest", "django", "test-script"})

# Per runner, options that replace its configuration, change where it imports
# modules from, or select, shard or skip tests. Any of them disables test-target
# linkage and the runner-output rules. Where a runner's semantics are unclear
# the option is listed (conservative). jest and vitest long options are
# compared case-insensitively without dashes (``--testNamePattern`` and
# ``--test-name-pattern`` alike); ``go test`` accepts ``--name`` for ``-name``.
_JS_CONFIG_OPTIONS = frozenset(
    {"-c", "--config", "-t", "--testnamepattern", "--testpathignorepatterns"}
    | {"--selectprojects", "--ignoreprojects", "--shard", "-o", "--onlychanged"}
    | {"--changedsince", "--changed", "--lastcommit", "--findrelatedtests", "--related"}
    | {"--passwithnotests", "--testpathpattern", "--testpathpatterns", "--testmatch"}
    | {"--testregex", "--roots", "--rootdir", "-r", "--root", "--dir", "--project"}
    | {"--exclude", "--setupfiles", "--setupfilesafterenv", "--modulepaths"}
    | {"--moduledirectories", "--modulenamemapper", "--testrunner", "--testsequencer"}
)
_DJANGO_CONFIG_OPTIONS = frozenset({"--settings", "--pythonpath", "--testrunner"})
# Option values equal to a runner's documented default: they neither select
# tests nor change how modules and settings resolve, so they do not narrow.
# Django's ``tests/runtests.py`` sets ``DJANGO_SETTINGS_MODULE`` to
# ``test_sqlite`` when neither ``--settings`` nor the variable is given
# (``os.environ.setdefault("DJANGO_SETTINGS_MODULE", "test_sqlite")``); a
# command-line ``DJANGO_SETTINGS_MODULE`` narrows on its own, and replay scrubs
# an inherited one. ``manage.py`` and ``django-admin`` have no such default.
_DEFAULT_OPTION_VALUES: Mapping[tuple[str, str], frozenset[str]] = {
    ("runtests.py", "--settings"): frozenset({"test_sqlite"}),
}
_CARGO_CONFIG_OPTIONS = frozenset(
    {"--skip", "--exact", "--ignored", "--features", "-F", "--no-default-features"}
    | {"--all-features", "--lib", "--bins", "--bin", "--tests", "--test", "--examples"}
    | {"--example", "--benches", "--bench", "--doc", "-p", "--package", "--exclude"}
    | {"--manifest-path", "--config", "--no-run", "--list", "-Z"}
    | {"-E", "--filter-expr", "--filterset", "--partition", "-P", "--profile", "--run-ignored"}
)
_RUNNER_CONFIG_OPTIONS: Mapping[str, frozenset[str]] = {
    "jest": _JS_CONFIG_OPTIONS,
    "vitest": _JS_CONFIG_OPTIONS,
    "mocha": frozenset(
        {"--config", "--package", "--opts", "--grep", "-g", "--fgrep", "-f", "--invert"}
        | {"-i", "--ignore", "--exclude", "--file", "--require", "-r", "--extension"}
    ),
    "phpunit": frozenset(
        {"-c", "--configuration", "--no-configuration", "--filter", "--exclude-filter"}
        | {"--group", "--exclude-group", "--testsuite", "--exclude-testsuite", "--covers"}
        | {"--uses", "--test-suffix", "--bootstrap", "--include-path", "-d"}
    ),
    "rspec": frozenset(
        {"-O", "--options", "-e", "--example", "-E", "--example-matches", "-t", "--tag"}
        | {"-P", "--pattern", "--exclude-pattern", "-I", "-r", "--require"}
        | {"--default-path", "--only-failures", "-n", "--next-failure"}
    ),
    "go-test": frozenset(
        {"-run", "-skip", "-tags", "-short", "-list", "-exec", "-toolexec", "-overlay"}
        | {"-mod", "-modfile", "-C"}
    ),
    "cargo-test": _CARGO_CONFIG_OPTIONS,
    "cargo-nextest": _CARGO_CONFIG_OPTIONS,
    "mvn": frozenset(
        {"-P", "--activate-profiles", "-pl", "--projects", "-s", "--settings", "-gs"}
        | {"--global-settings", "-f", "--file"}
    ),
    "gradle": frozenset(
        {"--tests", "-P", "--project-prop", "-p", "--project-dir", "-b", "--build-file"}
        | {"-c", "--settings-file", "-I", "--init-script", "-x", "--exclude-task"}
    ),
    "django": _DJANGO_CONFIG_OPTIONS,
    "test-script": _DJANGO_CONFIG_OPTIONS,
}
# The entries of ``_RUNNER_CONFIG_OPTIONS`` that only select or skip tests by
# name, path or group; the others replace configuration or module resolution
# (see ``SELECTION`` and ``CONFIGURATION``).
_JS_SELECTION_OPTIONS = frozenset(
    {"-t", "--testnamepattern", "--testpathignorepatterns", "--testpathpattern"}
    | {"--testpathpatterns", "--selectprojects", "--ignoreprojects", "--shard", "-o"}
    | {"--onlychanged", "--changedsince", "--changed", "--lastcommit", "--findrelatedtests"}
    | {"--related", "--passwithnotests", "--exclude", "--project"}
)
_RUNNER_SELECTION_OPTIONS: Mapping[str, frozenset[str]] = {
    "jest": _JS_SELECTION_OPTIONS,
    "vitest": _JS_SELECTION_OPTIONS,
    "mocha": frozenset(
        {"--grep", "-g", "--fgrep", "-f", "--invert", "-i", "--ignore"} | {"--exclude"}
    ),
    "phpunit": frozenset(
        {"--filter", "--exclude-filter", "--group", "--exclude-group", "--testsuite"}
        | {"--exclude-testsuite", "--covers", "--uses", "--test-suffix"}
    ),
    "rspec": frozenset(
        {"-e", "--example", "-E", "--example-matches", "-t", "--tag", "-P", "--pattern"}
        | {"--exclude-pattern", "--only-failures", "-n", "--next-failure"}
    ),
    "go-test": frozenset({"-run", "-skip"}),
    "cargo-test": frozenset({"--skip", "--exact", "--ignored"}),
    "cargo-nextest": frozenset({"--skip", "--exact", "--ignored", "-E", "--filter-expr"}),
    "mvn": frozenset({"-pl", "--projects"}),
    "gradle": frozenset({"--tests", "-x", "--exclude-task"}),
}
_CAMEL_OPTION_KINDS = frozenset({"jest", "vitest"})
# Runners whose short options take an attached value (``-cjest.config.js``,
# ``-Ilib``): a table entry of two characters also matches as a prefix.
_ATTACHED_SHORT_KINDS = frozenset({"jest", "vitest", "mocha", "phpunit", "rspec"})
# Runners whose single-dash options are whole words (``go test -mod=vendor``,
# ``mvn -pl core``): the option is matched as written, up to any ``=``.
_WORD_OPTION_KINDS = frozenset({"go-test", "mvn", "gradle"})
# Kinds whose positional arguments are test-name filters (``cargo test foo``,
# ``cargo test -- foo``), not targets: any positional narrows, and an option
# outside the two tables below narrows too, since its value may be a filter.
_FILTER_OPERAND_KINDS = frozenset({"cargo-test", "cargo-nextest"})
_CARGO_FLAG_OPTIONS = frozenset(
    {"-q", "--quiet", "-v", "-vv", "--verbose", "-r", "--release", "--workspace", "--all"}
    | {"--all-targets", "--no-fail-fast", "--frozen", "--locked", "--offline", "--timings"}
    | {"--nocapture", "--no-capture", "--show-output", "--include-ignored"}
)
_CARGO_VALUE_OPTIONS = frozenset(
    {"-j", "--jobs", "--target", "--target-dir", "--color", "--message-format"} | {"--test-threads"}
)
# Maven and Gradle ``-D`` properties that select tests.
_BUILD_SELECTION_PROPERTIES = frozenset({"test", "it.test", "groups", "excludedgroups"})
_BUILD_SELECTION_PROPERTY_PREFIXES = ("test.", "surefire.", "failsafe.", "maven.test.")
# Interpreter flags that change where Python imports modules from (``-P``
# and ``-I`` stop prepending the script or working directory to ``sys.path``).
_NARROWING_PYTHON_FLAGS = frozenset({"-P", "-I"})

# Environment variables that change what a test runner collects, selects or
# loads, or where the code under test is imported from. Assigned on the
# command line they disable test-target linkage; replay also removes them from
# the environment it inherits. ``NARROWING_ENVIRONMENT_PREFIXES`` extends the
# set to every variable with one of those prefixes.
NARROWING_ENVIRONMENT = frozenset(
    {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    | {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONSAFEPATH"}
    | {"DJANGO_SETTINGS_MODULE", "NODE_OPTIONS", "NODE_PATH", "RUBYOPT", "RUBYLIB"}
    | {"BUNDLE_GEMFILE", "GOFLAGS", "CGO_ENABLED"}
)
NARROWING_ENVIRONMENT_PREFIXES = ("JEST_", "VITEST_")
# An assignment to one of those variables anywhere in a command's text: a
# prefix, an ``env`` or ``export`` form, a preamble segment, or inside a shell
# wrapper's quoted body.
_NARROWING_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(sorted(NARROWING_ENVIRONMENT))
    + "".join(f"|{prefix}[A-Za-z0-9_]*" for prefix in NARROWING_ENVIRONMENT_PREFIXES)
    + r")="
)


def narrowing_variable(name: str) -> bool:
    """Return True when environment variable ``name`` narrows a test run."""
    upper = name.upper()
    return upper in NARROWING_ENVIRONMENT or upper.startswith(NARROWING_ENVIRONMENT_PREFIXES)


def narrowing_assignments(command: str) -> tuple[str, ...]:
    """Return the narrowing variables ``command``'s text assigns anywhere, sorted."""
    return tuple(sorted(set(_NARROWING_ASSIGNMENT_RE.findall(command))))


# pytest ini keys that decide which tests are collected or selected; setting
# one through ``-o``/``--override-ini`` disables test-target linkage.
_PYTEST_SELECTION_INI_KEYS = frozenset(
    {"addopts", "python_files", "python_classes", "python_functions", "testpaths"}
    | {"norecursedirs"}
)
# pytest options that replace the configuration or where it is looked up.
_PYTEST_CONFIG_OPTIONS = frozenset({"-c", "--config-file", "--rootdir", "--confcutdir"})
_PYTEST_OVERRIDE_OPTIONS = frozenset({"-o", "--override-ini"})
# ``-p`` plugins that cannot change which tests run or pass: disabling the
# cache provider only removes ``--lf``/``--ff`` and the ``cache`` fixture.
NO_OP_PYTEST_PLUGINS = frozenset({"no:cacheprovider"})


# Programs admitted by absolute path when they resolve inside an environment
# root (see ``environment_roots``), besides the Python interpreters.
_ENVIRONMENT_PROGRAMS = (
    _DIRECT_RUNNERS
    | _JS_PACKAGE_RUNNERS
    | frozenset(_SUBCOMMAND_RUNNERS)
    | {"py.test", "make", "gmake", "mvn", "gradle"}
)


def environment_roots(environment: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return the real paths of the known environment roots.

    The verifying interpreter's ``sys.prefix`` and ``sys.base_prefix``,
    ``VIRTUAL_ENV``, ``CONDA_PREFIX`` and every absolute ``PATH`` directory of
    ``environment`` (the process environment when None). The filesystem root
    itself is never a root.
    """
    source = os.environ if environment is None else environment
    candidates = [
        sys.prefix,
        sys.base_prefix,
        source.get("VIRTUAL_ENV", ""),
        source.get("CONDA_PREFIX", ""),
        *source.get("PATH", "").split(os.pathsep),
    ]
    roots = {
        os.path.realpath(candidate)
        for candidate in candidates
        if candidate and os.path.isabs(candidate)
    }
    roots.discard(os.sep)
    return tuple(sorted(roots))


def _environment_program(program: str, roots: Sequence[str] | None) -> bool:
    """Return True for an absolute allowlisted program inside an environment root.

    With ``roots`` None (lexical linkage on an admitted command) the name
    alone decides; otherwise the program's real path, symlinks resolved,
    must be an executable file inside one of ``roots``.
    """
    if not os.path.isabs(program):
        return False
    name = _program_name(program)
    if not (_is_python_executable(name) or name in _ENVIRONMENT_PROGRAMS):
        return False
    if roots is None:
        return True
    real = os.path.realpath(program)
    return os.path.isfile(real) and os.access(real, os.X_OK) and _inside(real, roots)


@dataclass(frozen=True, slots=True)
class ResolvedRunner:
    """The program a command really runs and the arguments it receives.

    ``kind`` names the runner (``pytest``, ``make``, ``script``, ...);
    ``arguments`` are the runner's own arguments, after the module, script or
    subcommand that selected it. ``narrowing_interpreter`` is True when the
    Python interpreter ran with a flag in ``_NARROWING_PYTHON_FLAGS``.
    ``script`` is the file name of a ``test-script`` runner (``runtests.py``).
    """

    kind: str
    arguments: tuple[str, ...]
    narrowing_interpreter: bool = False
    script: str = ""


def _program_name(value: str) -> str:
    name = PurePosixPath(value.replace("\\", "/")).name.lower()
    return name[:-4] if name.endswith(".exe") else name


def _skip_options(parts: Sequence[str], index: int, spec: _OptionSpec) -> int | None:
    """Return the index of the program after ``parts[index:]``'s options, or None.

    None when an option is refused or unknown, a value is missing, or no
    program follows: the program cannot be identified with certainty.
    """
    remaining = spec.positionals
    while index < len(parts):
        token = parts[index]
        if token == "--":
            index += 1
            return index if remaining == 0 and index < len(parts) else None
        if spec.assignments and _is_env_assignment(token):
            index += 1
            continue
        if token.startswith("-") and token != "-":
            name, separator, _ = token.partition("=")
            if token.startswith("--"):
                if name in spec.refused:
                    return None
                if name in spec.values:
                    index += 1 if separator else 2
                elif name in spec.flags and not separator:
                    index += 1
                else:
                    return None
                continue
            short = token[:2]
            if short in spec.refused or token in spec.refused:
                return None
            if short in spec.values:
                index += 1 if len(token) > 2 else 2
            elif token in spec.flags or (
                spec.numeric_flags and _NUMERIC_OPTION_RE.fullmatch(token)
            ):
                index += 1
            else:
                return None
            continue
        if remaining:
            if spec.positional_re is not None and not spec.positional_re.fullmatch(token):
                return None
            remaining -= 1
            index += 1
            continue
        return index
    return None


def peel_wrappers(argv: Sequence[str]) -> tuple[str, ...] | None:
    """Return ``argv`` from the program its wrappers run, or None when unknown."""
    parts = tuple(argv)
    for _ in range(_MAX_RESOLVE_DEPTH):
        if not parts:
            return None
        name = _program_name(parts[0])
        if name in REFUSED_WRAPPERS:
            return None
        spec = _WRAPPERS.get(name)
        if spec is None:
            return parts
        index = _skip_options(parts, 1, spec)
        if index is None:
            return None
        parts = parts[index:]
    return None


def _environment_bin_path(program: str) -> bool:
    """Return True when ``program``'s parent directory is an environment's
    ``bin`` (``Scripts`` on Windows) or ``node_modules/.bin``."""
    parts = PurePosixPath(program.replace("\\", "/")).parts
    if len(parts) < 3:
        return False
    return parts[-2] in {"bin", "Scripts"} or parts[-3:-1] == ("node_modules", ".bin")


def _denied_task_name(name: str) -> bool:
    return any(word in _DENIED_TASK_WORDS for word in _TASK_WORD_SPLIT_RE.split(name.lower()))


def _inside(path: str, roots: Sequence[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _workspace_roots(workspace: str) -> tuple[str, ...]:
    return tuple({os.path.normpath(os.path.abspath(workspace)), os.path.realpath(workspace)} - {""})


def _workspace_file(token: str, workspace: str | None, cwd_relative: str) -> bool:
    """Return True when ``token`` names a regular file inside the workspace.

    The path is normalized lexically (``..`` cannot climb out). With no
    workspace (linkage on an already admitted command), only a relative path
    that does not climb out is accepted.
    """
    if not token or "\\" in token:
        return False
    if workspace is None:
        path = PurePosixPath(token)
        return not path.is_absolute() and ".." not in path.parts
    roots = _workspace_roots(workspace)
    base = os.path.normpath(os.path.join(os.path.realpath(workspace), cwd_relative))
    candidate = os.path.normpath(token if os.path.isabs(token) else os.path.join(base, token))
    return _inside(candidate, roots) and os.path.isfile(candidate)


def _python_runner(
    parts: Sequence[str], workspace: str | None, cwd_relative: str
) -> ResolvedRunner | None:
    index = 1
    narrowing = False
    while index < len(parts):
        token = parts[index]
        if token == "-m":
            if index + 1 >= len(parts):
                return None
            module, arguments = parts[index + 1], tuple(parts[index + 2 :])
            if module in {"pytest", "unittest", "tox", "nox"}:
                return ResolvedRunner(module, arguments, narrowing)
            if module == "django" and arguments[:1] == ("test",):
                return ResolvedRunner("django", arguments[1:], narrowing)
            return None
        narrowing = narrowing or token in _NARROWING_PYTHON_FLAGS
        if token in _PYTHON_FLAGS:
            index += 1
            continue
        if token in _PYTHON_VALUE_OPTIONS:
            index += 2
            continue
        if token[:2] in _PYTHON_VALUE_OPTIONS and len(token) > 2:
            index += 1
            continue
        if token.startswith("-"):
            # -c, -, -i and unknown options: an inline or interactive program.
            return None
        if not _workspace_file(token, workspace, cwd_relative):
            return None
        script = _project_test_runner_script(["python", *parts[index:]])
        if script is None:
            return ResolvedRunner("script", tuple(parts[index + 1 :]), narrowing)
        rest = tuple(parts[index + 1 :])
        if PurePosixPath(script).name == "manage.py":
            rest = rest[1:]
        return ResolvedRunner("test-script", rest, narrowing, PurePosixPath(script).name)
    return None


def _make_runner(arguments: Sequence[str]) -> ResolvedRunner | None:
    for token in arguments:
        name = token.partition("=")[0]
        if name in _MAKE_REFUSED_OPTIONS or (
            not token.startswith("--") and token[:2] in {"-C", "-f"}
        ):
            return None
        if token in _GENERIC_NON_EXECUTING_OPTIONS:
            return None
    targets = [
        token
        for token in arguments
        if not token.startswith("-") and "=" not in token and not token.isdigit()
    ]
    if any(_denied_task_name(target) for target in targets):
        return None
    return ResolvedRunner("make", tuple(arguments))


def _js_runner(name: str, arguments: Sequence[str]) -> ResolvedRunner | None:
    index = 0
    while index < len(arguments) and arguments[index] in _JS_GLOBAL_FLAGS:
        index += 1
    if index >= len(arguments):
        return None
    subcommand = arguments[index]
    rest = tuple(arguments[index + 1 :])
    if subcommand in _JS_TEST_SUBCOMMANDS:
        return ResolvedRunner("bun-test" if name == "bun" else "js-test", rest)
    if subcommand in _JS_RUN_SUBCOMMANDS:
        if not rest or rest[0].startswith("-") or _denied_task_name(rest[0]):
            return None
        return ResolvedRunner("js-run", rest)
    return None


def _build_tool_runner(name: str, arguments: Sequence[str]) -> ResolvedRunner | None:
    parts = list(arguments)
    if _has_gradle_or_maven_test_skip(parts) or any(
        token in _GENERIC_NON_EXECUTING_OPTIONS or token in {"-m", "--dry-run"} for token in parts
    ):
        return None
    tasks = [token for token in parts if not token.startswith("-") and "=" not in token]
    if any(_denied_task_name(task) for task in tasks):
        return None
    wanted = _MAVEN_TEST_GOALS if name.startswith("mvn") else _GRADLE_TEST_TASKS
    if not any(task.rsplit(":", 1)[-1] in wanted for task in tasks):
        return None
    return ResolvedRunner(name.removesuffix("w"), tuple(arguments))


def _subcommand_runner(name: str, arguments: Sequence[str]) -> ResolvedRunner | None:
    index = 0
    if name == "cargo" and arguments[:1] and arguments[0].startswith("+"):
        index = 1
    if index >= len(arguments) or arguments[index] not in _SUBCOMMAND_RUNNERS[name]:
        return None
    subcommand = arguments[index]
    kind = "django" if name == "django-admin" else f"{name}-{subcommand}"
    return ResolvedRunner(kind, tuple(arguments[index + 1 :]))


def _runner(
    name: str, parts: Sequence[str], workspace: str | None, cwd_relative: str
) -> ResolvedRunner | None:
    arguments = parts[1:]
    if _is_python_executable(name):
        return _python_runner(parts, workspace, cwd_relative)
    if name in _SHELL_INTERPRETERS:
        # Only ``sh <workspace script> [args]``: no options, no inline program.
        if not arguments or not _workspace_file(arguments[0], workspace, cwd_relative):
            return None
        return ResolvedRunner("script", tuple(arguments[1:]))
    if name in {"py.test", "pytest"}:
        return ResolvedRunner("pytest", tuple(arguments))
    if name in _DIRECT_RUNNERS:
        return ResolvedRunner(name, tuple(arguments))
    if name in {"make", "gmake"}:
        return _make_runner(arguments)
    if name in _JS_PACKAGE_RUNNERS:
        return _js_runner(name, arguments)
    if name in {"mvn", "mvnw", "gradle", "gradlew"}:
        return _build_tool_runner(name, arguments)
    if name in _SUBCOMMAND_RUNNERS:
        return _subcommand_runner(name, arguments)
    return None


def _resolve(
    parts: tuple[str, ...],
    workspace: str | None,
    cwd_relative: str,
    depth: int,
    roots: Sequence[str] | None,
) -> ResolvedRunner | None:
    if depth > _MAX_RESOLVE_DEPTH:
        return None
    peeled = peel_wrappers(parts)
    if not peeled:
        return None
    program = peeled[0]
    name = _program_name(program)
    if "/" in program or "\\" in program:
        # A path program must be a file inside the workspace: a runner the
        # project ships (``./gradlew``, ``.venv/bin/pytest``) or its script;
        # or an allowlisted interpreter or runner inside an environment root.
        if not _workspace_file(program, workspace, cwd_relative):
            if _environment_program(program, roots):
                return _runner(name, peeled, workspace, cwd_relative)
            return None
        interpreter = _is_python_executable(name) or name in _SHELL_INTERPRETERS
        if (interpreter or name in _DIRECT_RUNNERS) and not _environment_bin_path(program):
            # A workspace file named like an interpreter or a test runner
            # (``./pytest``) is that program only inside an environment's
            # ``bin`` directory (``.venv/bin/pytest``, ``node_modules/.bin/jest``);
            # elsewhere an interpreter name is refused and a runner name is a
            # plain script, whose operands are never test targets.
            return None if interpreter else ResolvedRunner("script", peeled[1:])
        resolved = _runner(name, peeled, workspace, cwd_relative)
        if resolved is not None:
            return resolved
        if interpreter:
            return None
        script = _project_test_runner_script(list(peeled))
        if script is None:
            return ResolvedRunner("script", peeled[1:])
        rest = peeled[1:]
        if PurePosixPath(script).name == "manage.py":
            rest = rest[1:]
        return ResolvedRunner("test-script", rest, script=PurePosixPath(script).name)
    if name in VIEWER_PROGRAMS or name in REFUSED_WRAPPERS:
        return None
    for launcher, spec in _LAUNCHERS.items():
        if tuple(token.lower() for token in peeled[: len(launcher)]) != launcher:
            continue
        index = _skip_options(peeled, len(launcher), spec)
        if index is None:
            return None
        # The launched program passes the same allowlist (``uv run pytest``).
        return _resolve(peeled[index:], workspace, cwd_relative, depth + 1, roots)
    return _runner(name, peeled, workspace, cwd_relative)


def _non_executing(runner: ResolvedRunner) -> bool:
    if any(argument in _GENERIC_NON_EXECUTING_OPTIONS for argument in runner.arguments):
        return True
    if runner.kind == "pytest":
        return any(argument in _PYTEST_NON_EXECUTING_OPTIONS for argument in runner.arguments)
    if runner.kind == "tox":
        return any(a in {"-l", "--listenvs", "--showconfig"} for a in runner.arguments)
    if runner.kind == "nox":
        return any(argument in {"-l", "--list"} for argument in runner.arguments)
    return False


def resolve_replay_program(
    argv: Sequence[str],
    *,
    workspace: str | None,
    cwd_relative: str = ".",
    environment: Mapping[str, str] | None = None,
) -> ResolvedRunner | None:
    """Return the allowlisted runner ``argv`` really runs, or None.

    ``workspace`` None resolves lexically (no file checks); replay admission
    always passes the workspace, and ``environment`` (the replay environment,
    the process environment when None) supplies the environment roots.
    """
    roots = None if workspace is None else environment_roots(environment)
    resolved = _resolve(tuple(argv), workspace, cwd_relative, 0, roots)
    if resolved is None or _non_executing(resolved):
        return None
    return resolved


def _absolute_argument_outside(token: str, roots: Sequence[str]) -> bool:
    value = token.partition("=")[2] if token.startswith("-") and "=" in token else token
    if not value.startswith("/") or value == "/dev/null":
        return False
    return not _inside(os.path.normpath(value), roots)


def outside_known_roots(
    value: str, *, workspace: str, environment: Mapping[str, str] | None = None
) -> bool:
    """Return True when an absolute ``value`` lies outside the workspace.

    ``/dev/null`` and an admitted environment program are not outside.
    """
    return _absolute_argument_outside(value, _workspace_roots(workspace)) and not (
        _environment_program(value, environment_roots(environment))
    )


def replay_allowed(
    argv: Sequence[str],
    *,
    workspace: str,
    cwd_relative: str = ".",
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return True when ``argv`` may be replayed (see module docstring)."""
    if (
        resolve_replay_program(
            argv, workspace=workspace, cwd_relative=cwd_relative, environment=environment
        )
        is None
    ):
        return False
    return not any(
        outside_known_roots(token, workspace=workspace, environment=environment) for token in argv
    )


def _excluding_option(token: str, kind: str) -> bool:
    if not token.startswith("-") or token == "-":
        return False
    name = token.partition("=")[0]
    if name in _TARGET_EXCLUDING_OPTIONS:
        return True
    short = _SHORT_EXCLUDING_OPTIONS.get(kind, frozenset())
    if name in short:
        return True
    return not token.startswith("--") and any(
        len(option) == 2 and token.startswith(option) for option in short
    )


def _option_key(name: str, kind: str) -> str:
    """Return ``name`` as ``_RUNNER_CONFIG_OPTIONS`` spells it for ``kind``."""
    if kind in _CAMEL_OPTION_KINDS and name.startswith("--"):
        return "--" + name[2:].replace("-", "").lower()
    if kind == "go-test" and name.startswith("--"):
        return name[1:]
    return name


def _build_property_narrows(name: str, value: str | None) -> bool:
    """Return True for a Maven or Gradle ``-D`` property that selects tests."""
    if not name.startswith("-D"):
        return False
    key = name[2:] if len(name) > 2 else (value or "").partition("=")[0]
    key = key.lower()
    return key in _BUILD_SELECTION_PROPERTIES or key.startswith(_BUILD_SELECTION_PROPERTY_PREFIXES)


def _runner_config_option(token: str, name: str, value: str | None, kind: str) -> str | None:
    """Return the narrowing class of ``token`` among ``kind``'s
    ``_RUNNER_CONFIG_OPTIONS``, or None when it is not one of them."""
    options = _RUNNER_CONFIG_OPTIONS.get(kind)
    if options is None:
        return None
    selection = _RUNNER_SELECTION_OPTIONS.get(kind, frozenset())
    if kind in _WORD_OPTION_KINDS:
        name = token.partition("=")[0]
    if kind in {"mvn", "gradle"}:
        if _build_property_narrows(name, value):
            return SELECTION
        if name.startswith("-P"):
            return CONFIGURATION
    key = _option_key(name, kind)
    if key not in options and kind in _ATTACHED_SHORT_KINDS and not token.startswith("--"):
        key = next((o for o in options if len(o) == 2 and token.startswith(o)), key)
    if key in options:
        return SELECTION if key in selection else CONFIGURATION
    if kind in _FILTER_OPERAND_KINDS and name not in (_CARGO_FLAG_OPTIONS | _CARGO_VALUE_OPTIONS):
        # An option this table does not know: its value may be a filter, and
        # the option may change the build.
        return CONFIGURATION
    return None


def _narrowing_option(name: str, value: str | None, kind: str) -> str | None:
    """Return the narrowing class of option ``name`` (with ``value``, if it took
    one) for pytest and the pattern runners, or None."""
    if kind == "pytest":
        if name in _PYTEST_CONFIG_OPTIONS:
            return CONFIGURATION
        if name in _PYTEST_OVERRIDE_OPTIONS:
            if value is None or "=" not in value:
                return CONFIGURATION
            key = value.partition("=")[0].strip().lower()
            return CONFIGURATION if not key or key in _PYTEST_SELECTION_INI_KEYS else None
        if name == "-p" and value not in NO_OP_PYTEST_PLUGINS:
            return CONFIGURATION
    elif kind in _PATTERN_RUNNER_KINDS and name in {"-p", "--pattern"}:
        return SELECTION
    return None


def command_line_assignments(argv: Sequence[str]) -> tuple[str, ...]:
    """Return the ``NAME=value`` tokens ``argv`` sets for the program it runs.

    Leading assignments, and those consumed by an ``env`` wrapper anywhere in
    the chain of wrappers and launchers (``timeout 60 env X=1 pytest``,
    ``uv run env X=1 pytest``).
    """
    parts = tuple(argv)
    found: list[str] = []
    index = 0
    while index < len(parts) and _is_env_assignment(parts[index]):
        found.append(parts[index])
        index += 1
    parts = parts[index:]
    for _ in range(_MAX_RESOLVE_DEPTH):
        if not parts:
            break
        name = _program_name(parts[0])
        spec = _WRAPPERS.get(name)
        launcher = next(
            (
                (key, option_spec)
                for key, option_spec in _LAUNCHERS.items()
                if tuple(token.lower() for token in parts[: len(key)]) == key
            ),
            None,
        )
        if spec is not None:
            program_index = _skip_options(parts, 1, spec)
        elif launcher is not None:
            program_index = _skip_options(parts, len(launcher[0]), launcher[1])
        else:
            break
        if program_index is None:
            break
        if spec is not None and spec.assignments:
            found.extend(token for token in parts[1:program_index] if _is_env_assignment(token))
        parts = parts[program_index:]
    return tuple(found)


def _narrowing_environment(argv: Sequence[str], environment: Sequence[str]) -> bool:
    names = set(environment)
    names.update(token.partition("=")[0] for token in command_line_assignments(argv))
    return any(narrowing_variable(name) for name in names)


def _selection(
    argv: Sequence[str], environment: Sequence[str]
) -> tuple[ResolvedRunner | None, str | None, frozenset[str]]:
    """Return ``(runner, narrowing, operands)`` for ``argv``.

    ``narrowing`` is ``CONFIGURATION`` when configuration, the environment or
    an interpreter flag changes what the runner collects, loads or imports;
    otherwise ``SELECTION`` when an option excludes or narrows the tests it
    executes; otherwise None (see the module docstring). ``operands`` are
    empty unless ``narrowing`` is None. ``environment``
    names the variables the command line assigns outside ``argv`` (a replay
    candidate's ``env_delta``). Option values are never operands; a token after
    an option the tables do not know is not an operand either, since it may be
    that option's value. Tokens after ``--`` are checked for narrowing but are
    never operands.
    """
    runner = resolve_replay_program(argv, workspace=None)
    if runner is None:
        return None, None, frozenset()
    if runner.narrowing_interpreter or _narrowing_environment(argv, environment):
        return runner, CONFIGURATION, frozenset()
    narrowing: str | None = None
    value_options = set(_TARGET_VALUE_OPTIONS)
    flag_options = set(_TARGET_FLAG_OPTIONS)
    if runner.kind in _PATTERN_RUNNER_KINDS:
        value_options |= _LABEL_RUNNER_VALUE_OPTIONS
        flag_options -= _LABEL_RUNNER_VALUE_OPTIONS
    if runner.kind in _FILTER_OPERAND_KINDS:
        value_options |= _CARGO_VALUE_OPTIONS
        flag_options |= _CARGO_FLAG_OPTIONS
    operands: set[str] = set()
    after_separator = False
    arguments = runner.arguments
    index = 0
    while index < len(arguments):
        token = arguments[index]
        index += 1
        if token == "--":
            after_separator = True
            continue
        if not token.startswith("-") or token == "-":
            if runner.kind in _FILTER_OPERAND_KINDS:
                # ``cargo test foo`` or ``cargo test -- foo``: a name filter.
                narrowing = SELECTION
            elif not after_separator:
                operands.add(token)
            continue
        if _excluding_option(token, runner.kind):
            narrowing = SELECTION
            continue
        name, separator, inline = token.partition("=")
        value: str | None = None
        consumes_next = False
        if token.startswith("--"):
            if separator:
                value = inline
            elif name not in flag_options:
                # A known value option, or an unknown one whose value may follow.
                consumes_next = True
        elif token[:2] in value_options and len(token) > 2:
            name, value = token[:2], token[2:]
        elif separator:
            value = inline
        elif name not in flag_options:
            consumes_next = True
        if consumes_next and index < len(arguments):
            following = arguments[index]
            if not following.startswith("-") or following == "-":
                # An option-like token is never taken as a value: it is
                # examined as an option of its own (``--x --ignore t.py``).
                value = following
                index += 1
        if value in _DEFAULT_OPTION_VALUES.get((runner.script, name), frozenset()):
            continue
        found = _narrowing_option(name, value, runner.kind) or _runner_config_option(
            token, name, value, runner.kind
        )
        if found == CONFIGURATION:
            return runner, CONFIGURATION, frozenset()
        narrowing = narrowing or found
    if narrowing is not None:
        return runner, narrowing, frozenset()
    return runner, None, frozenset(operands)


def excludes_tests(argv: Sequence[str], environment: Sequence[str] = ()) -> bool:
    """Return True when ``argv`` resolves to a runner that an option or the
    command-line configuration narrows (``--ignore``, ``-k``, ``-o addopts=``,
    ``PYTEST_ADDOPTS=``, ...). ``environment`` names variables assigned
    outside ``argv``."""
    runner, narrowing, _ = _selection(argv, environment)
    return runner is not None and narrowing is not None


def alters_configuration(argv: Sequence[str], environment: Sequence[str] = ()) -> bool:
    """Return True when ``argv`` resolves to a runner whose configuration, module
    resolution or environment the command changes (``CONFIGURATION``): its
    output cannot show that a named test passed against the workspace's code,
    even when it names that test. ``environment`` names variables assigned
    outside ``argv``."""
    runner, narrowing, _ = _selection(argv, environment)
    return runner is not None and narrowing == CONFIGURATION


def run_may_back_test_claim(
    argv: Sequence[str], claim_file: str | None, environment: Sequence[str] = ()
) -> bool:
    """Gate the runner-output rules for a replayed test run.

    False when the program cannot be resolved, when an option or the
    command-line configuration excludes or narrows the tests it runs, or when
    ``claim_file`` (the claimed test file, if any) appears in the command
    without being an executed operand, as in ``pytest --rootdir
    tests/test_x.py``. ``environment`` names variables assigned outside
    ``argv`` (a replayed run's ``env_delta``).
    """
    runner, narrowing, operands = _selection(argv, environment)
    if runner is None or narrowing is not None:
        return False
    if claim_file is None:
        return True
    needle = claim_file.lower()
    if not any(needle in token.lower() for token in argv):
        return True
    return runner.kind in TARGET_RUNNER_KINDS and any(
        operand.split("::", 1)[0].lower() == needle for operand in operands
    )


def claim_target_operands(argv: Sequence[str], environment: Sequence[str] = ()) -> frozenset[str]:
    """Return the tests ``argv`` names as positional operands of a test runner.

    Empty unless the program is a test runner whose operands select tests
    (``TARGET_RUNNER_KINDS``) and neither an option nor the command-line
    configuration narrows them (see the module docstring). ``environment``
    names variables assigned outside ``argv``.
    """
    runner, narrowing, operands = _selection(argv, environment)
    if runner is None or narrowing is not None or runner.kind not in TARGET_RUNNER_KINDS:
        return frozenset()
    return operands


__all__ = [
    "REFUSED_WRAPPERS",
    "TARGET_RUNNER_KINDS",
    "VIEWER_PROGRAMS",
    "ResolvedRunner",
    "alters_configuration",
    "NARROWING_ENVIRONMENT",
    "NARROWING_ENVIRONMENT_PREFIXES",
    "NO_OP_PYTEST_PLUGINS",
    "claim_target_operands",
    "command_line_assignments",
    "environment_roots",
    "excludes_tests",
    "narrowing_assignments",
    "narrowing_variable",
    "outside_known_roots",
    "peel_wrappers",
    "replay_allowed",
    "resolve_replay_program",
    "run_may_back_test_claim",
]
