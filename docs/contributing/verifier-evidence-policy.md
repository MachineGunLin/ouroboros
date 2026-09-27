# Verifier Evidence Policy

This policy governs fixes to the atomic verifier and fat-harness evidence
boundary.

The core verifier must stay domain- and language-agnostic. It decides whether
typed evidence is backed by the runtime transcript. It must not grow bespoke
parsers for each test runner, language ecosystem, framework, or report format.

## Core rule

When valid work appears to be rejected, first classify the failure shape:

| Shape | Meaning | Correct direction |
| --- | --- | --- |
| No runtime transcript evidence exists for the claim | The claim may be fabricated. | Keep or use `FABRICATION_SUSPECTED`. |
| Related runtime work exists, but the evidence form cannot prove the claim | The evidence contract is mismatched. | Use `EVIDENCE_FORM_MISMATCH` with retry guidance. |
| The evidence needs runner-specific interpretation | The core verifier is the wrong layer. | Add a profile/adapter-level contract or require a runner-agnostic proof form. |

## Do not add runner-specific parsers to core

Avoid fixes that teach `parallel_executor.py` to understand one ecosystem's
result format, such as JUnit XML, pytest JUnit XML, TAP, Go test JSON, Vitest
JSON, Maven Surefire XML, or Gradle-specific report layouts.

Those fixes make one stack pass while creating an implicit obligation to support
every other stack in the same core path. They also make anti-fabrication
semantics depend on language-specific parsing details that the core verifier
cannot own consistently.

## Preferred evidence forms

For test commands whose output is filtered or paged, the preferred form is
the plain pipeline, which replay supports:

```sh
<test command> 2>&1 | tail -100
```

Replay runs `<test command>` alone and judges its own exit status, so the
filter cannot mask a failure. A `set -o pipefail && ...` preamble is not
replayable (`set` is a shell builtin), so claims resting on it keep the
transcript-only rules.

For richer result formats, prefer one of these approaches:

- Emit a typed, runner-neutral proof field from a profile or adapter that owns
  that runner.
- Keep the core verdict as `EVIDENCE_FORM_MISMATCH` and retry with clearer
  evidence.
- Promote a new cross-runner evidence contract only after it has an explicit
  design issue and acceptance criteria across multiple ecosystems.

## Replay-first corroboration

Replay is the runner-agnostic proof the core verifier uses when the transcript
alone cannot prove a `tests_passed` or `commands_run` claim. It needs no
knowledge of the runner: the harness re-runs a command the leaf ran and judges
only the exit status.

It runs only when command verification is on (`execution.run_verify_commands`,
the default) and the leaf held Bash authority. The rules:

- **Source.** Candidates are commands from the transcript's structured Bash
  calls, never text from a claim. Runtime shell wrappers (`/bin/zsh -lc '...'`)
  are peeled. Commands linked to an unproven claim come first, then recognized
  test runs that the runner-output rules judge (for example a
  `pytest tests/test_a.py` run behind a `tests/test_a.py::test_x` claim). At
  most 3 per criterion. The most recent run of a command decides: when the
  transcript recorded a non-zero exit for it (or a failed tool result), it is
  not replayed.
- **Allowlist.** Only these programs are replayed, found after peeling
  wrappers (`timeout`, `stdbuf`, `time`, `nice`, `ionice`, `env`, `nohup`,
  `command`, `exec`, `setsid`) with per-wrapper option tables; an option or
  operand the tables do not know leaves the program unknown, and nothing is
  replayed:
  - `python`/`python3`/`pythonX.Y` with `-m pytest`, `-m unittest`, `-m tox`,
    `-m nox`, `-m django test`, or a script inside the workspace (never `-c`,
    `-` or stdin);
  - `pytest`, `py.test`, `tox`, `nox`, `django-admin test`, `rspec`, `jest`,
    `vitest`, `mocha`, `ava`, `phpunit`, `ctest`;
  - `make`/`gmake`, except targets whose words include `install`, `deploy`,
    `publish`, `release`, `upload` or `push`, and except `-C`, `-f` and dry
    runs;
  - `npm`/`pnpm`/`yarn`/`bun` `test`, or `run <script>` with the same
    exceptions for the script name; `bun test`;
  - `go test|vet|build`, `cargo test|check|build|clippy|nextest`,
    `mvn`/`mvnw` with a `test` or `verify` goal, `gradle`/`gradlew` with a
    `test`, `check` or `build` task (not `install`/`publish`/`deploy` tasks,
    not with tests skipped), `dotnet test|build`, `deno test`,
    `swift test|build`, `mix test`;
  - any of these reached through `uv run`, `uvx`, `poetry run`, `pipenv run`,
    `pdm run`, `bundle exec`, `npx` or `bunx`;
  - a script inside the workspace, run directly (`./run_tests.sh`,
    `bin/test`) or by `python`, `sh` or `bash`.

  Everything else is refused: file viewers and text utilities (`cat`, `sed`,
  `head`, `tail`, `less`, `grep`, `rg`, `awk`, `wc`, `ls`, `find`, `stat`,
  `file`, `diff`, `git`, ...), package managers, `xargs`, an absolute-path
  program or argument outside the workspace, an environment assignment in
  the command naming an absolute path outside the workspace (`PATH=/tmp/x`),
  `uv run --env-file`, and a runner in a mode that runs no tests
  (`--help`, `--collect-only`, `make -n`, ...). One exception: an absolute-path program whose name is an
  allowlisted interpreter or runner (`python3.9`, `pytest`, `make`, ...) is
  admitted when its real path, symlinks resolved, lies inside a known
  environment root: `sys.prefix` or `sys.base_prefix` of the verifying
  process, `VIRTUAL_ENV`, `CONDA_PREFIX`, or a directory on the replay
  environment's `PATH` (for example `/opt/miniconda3/envs/testbed/bin/python`
  in a SWE-bench image). The denylist below is a second layer.
- **Isolation.** Each command runs as a direct argv (no shell) in a fresh copy
  of the workspace, with the verify gate's scrubbed environment plus
  `PYTHONDONTWRITEBYTECODE=1`, and `execution.verify_command_timeout_seconds`.
  `PYTEST_ADDOPTS`, `PYTEST_PLUGINS` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD` are
  removed from the inherited environment, and each replayed run records them
  in `scrubbed_environment`. Assignments the command itself makes are kept
  (recorded in `env_delta`); they disable target linkage instead (below).
  `.git` and caches are not copied; `.venv`, `venv`, `node_modules`, `.tox`
  and `.nox` are linked, not copied. A workspace over 50,000 files or 1 GiB is
  not replayed. Absolute workspace paths in the command point at the copy.
- **Network.** Network access must be denied: `sandbox-exec` on macOS
  (loopback allowed), an unprivileged network namespace on Linux, or a Linux
  process that already has only a loopback interface (a container started
  with `--network none`). Where none of these works, nothing is replayed, the
  claims keep the transcript-only rules, and the observation records
  `replay_skipped: network_isolation_unavailable`.
- **Live paths.** The copy reaches live paths through its links: the linked
  dependency trees and the targets of copied symlinks. On macOS the sandbox
  denies writes to them and to the live workspace. Elsewhere their metadata
  (type, size, mtime and ctime of every entry) is fingerprinted before and
  after the run, and any change marks it `mutated`; a tree over 250,000
  entries is not replayed. Writes to other absolute paths outside the copy are
  not confined.
- **Success.** Exit 0, no timeout, the transcript's recorded exit (when it
  recorded one) equal to the replay's, and no change to or deletion of a
  pre-existing file (SHA-256 of every file in the copy before and after,
  outside build outputs, caches and dependency directories) or of a live
  linked path. Anything else leaves the claim unsupported.
- **Linkage.** A claim is corroborated by a successful replay only when the
  whitespace-normalized claim:
  - equals the transcript command or its replayed core;
  - equals one of them plus one trailing parenthetical annotation
    (`make test (12 passed)`); or
  - minus a trailing `(N tests)` count, is one test target that is a
    positional operand (not an option value) of a test runner that executes
    it (pytest, unittest, Django-style runners, `bin/test`, jest, vitest,
    mocha, rspec, `go test`, phpunit), and nothing in the command narrows
    what the runner collects or selects (see "Narrowing" below). For example
    `migrations (578 tests)` is linked to `python tests/runtests.py
    migrations`, and `tests/test_x.py` is not linked to `sed -n 1,40p
    tests/test_x.py` or to `pytest --ignore tests/test_x.py`.

  A claim that merely contains a command (`make` inside `make test`) is not
  linked to it. The runner-output rules apply to a replayed run only under the
  same conditions: nothing narrows it, and a claimed file named in the
  command must be one of its executed operands.
- **Narrowing.** Any of these in a command disables the target rule and the
  runner-output rules for it, on replay and on the transcript-only path:
  - an option that excludes or selects tests, in any spelling (`--opt value`,
    `--opt=value`, `-kvalue`): pytest `--deselect`, `--ignore`,
    `--ignore-glob`, `-k`, `-m`; unittest `-k`; Django and project runner
    scripts `-k`, `--tag`, `--exclude-tag`, `--start-at`, `--start-after`;
    and `--exclude*`, `--skip`, `--filter`, `--grep`, `-t`, `-g`, `-e`,
    `-run`, `-skip` for the other runners;
  - pytest `-o`/`--override-ini` setting `addopts`, `python_files`,
    `python_classes`, `python_functions`, `testpaths` or `norecursedirs`
    (`addopts` covers a `--deselect`, `-k` or `-m` inside it); other keys,
    such as `cache_dir`, do not narrow;
  - pytest `-c`/`--config-file`, `--rootdir` and `--confcutdir`, whatever
    they name: whether it is the file or directory pytest would pick by
    default cannot be decided from the command;
  - pytest `-p` loading or disabling a plugin, except the no-op
    `no:cacheprovider`; `-p`/`--pattern` for unittest, Django and project
    runner scripts (a discovery pattern);
  - an assignment of `PYTEST_ADDOPTS`, `PYTEST_PLUGINS` or
    `PYTEST_DISABLE_PLUGIN_AUTOLOAD` on the command line: a leading
    assignment, one consumed by an `env` wrapper, or, on the transcript-only
    path, anywhere in the command text (`export PYTEST_ADDOPTS=... && ...`).

  An option the tables do not know never takes an option-like token as its
  value, so `pytest --x --ignore tests/t.py tests/t.py` is still narrowed.
- **Output filters.** `CMD | tail ...`, `CMD 2>&1 | grep ...` and chains of
  output filters (`tail`, `head`, `grep`, `egrep`, `fgrep`, `sed`, `cat`,
  `cut`, `sort`, `uniq`, `wc`, `tr`) replay `CMD` alone and use `CMD`'s own
  exit status. The filters are never run. Any other shell construct (`||`,
  `;`, `&&` other than a leading `cd <relative-dir> &&`, redirection to a
  file, substitution, `tee`, a pipe into a program) is not replayed.
- **Denylist.** Commands that escalate privilege, reach the network or other
  hosts, drive containers, delete files, write to version control, or install
  packages (`sudo`, `ssh`, `curl`, `wget`, `docker`, `rm`, `git` other than
  read-only subcommands, `pip install`, `npm install`, `uv add`, `brew`,
  `twine`, `gh`, ...) and inline shell programs (`bash -c`) are never
  replayed, whatever the allowlist says.

A claim that no successful replay backs falls through to the transcript-only
rules and failure classes below. Two of those rules follow the same
principle: a claimed test file backed by a transcript run links only as an
executed operand of a runner that nothing narrows (not
`pytest --ignore tests/x.py`), and the functional tier does not accept a
recorded exit that belongs to a pipeline without `pipefail`
(`./run_tests.sh | tail -5`), since it is the last stage's status.

## Failure class semantics

`FABRICATION_SUSPECTED` is reserved for claims with no supporting runtime event
or artifact reference. It should not be used when the transcript clearly shows
related work but the evidence shape is contract-incompatible.

`EVIDENCE_FORM_MISMATCH` means:

- related runtime work is visible;
- the current evidence form cannot prove the typed claim;
- retrying with a contract-compliant proof form is reasonable;
- the verifier must still reject the claim until that proof form exists.

This distinction keeps issue triage honest: the implementer did not necessarily
invent work, but the harness still cannot accept the evidence.

## Review checklist

Before accepting a verifier-evidence fix, ask:

1. Does this add language-, framework-, or runner-specific parsing to core?
2. Could the same pattern appear in another ecosystem tomorrow?
3. Is the fix preserving anti-fabrication semantics, or merely making one report
   format pass?
4. Would `EVIDENCE_FORM_MISMATCH` plus retry guidance be the correct smaller
   response?
5. If structured parsing is required, is it owned by a profile/adapter or a
   cross-runner evidence contract rather than by the core verifier?
