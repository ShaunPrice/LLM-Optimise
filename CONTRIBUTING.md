# Contributing

Contributions should make the lab's measurements, task-quality decisions and user workflows more reliable. Small, reviewable changes with clear evidence are easier to assess than broad feature claims.

Start with [the architecture](docs/architecture.md), [the user guide](docs/user-guide.md) and the module relevant to your change. For significant behaviour changes, describe the user problem and proposed scope before implementing a large rewrite.

## Development setup

Use Python 3.10 or newer in a virtual environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests
python -m build
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m build
```

Core tests should remain runnable without model downloads, paid provider requests, GPU hardware or Soup. Keep heavy runtime/training dependencies outside the core package. Use separate environments for live accelerator or training checks.

## Validation expectations

Match validation to the behaviour changed:

- Configuration/routing changes: test boundaries, invalid inputs, unknown metrics, hard constraints and deterministic selection.
- Provider changes: use a local controlled fixture for request/response contracts, missing usage, truncation and errors. Label live provider checks separately.
- Runtime/measurement changes: verify process cleanup, resource scope, missing telemetry, warmup exclusion and reproducibility. Record any hardware/runtime needed for a live check.
- File/proposal changes: cover traversal, symlinks, stale originals, duplicate paths and response size limits.
- Docker changes: verify the structured command, resource settings and cleanup. A mocked command test is not live Docker execution.
- GUI changes: inspect the running interface, keyboard operation, empty/error states and the complete affected interaction. Ensure labels and defaults match the Python API.
- Documentation changes: check actual `--help`, file paths, defaults, links and the distinction between preparation and execution.

Do not replace unavailable telemetry with zero or infer a test passed because no exception occurred. Inspect actual test counts and assertions. Reversible wording/style-only changes do not need invented tests that merely mirror the implementation.

## Measurement and data discipline

Maintain quality gates before declaring an optimisation success. Use the same task dataset for competing configurations, preserve the original input hashes and retain a held-out evaluation for selection claims. Tiny bundled tasks are functional smoke tests; describe them that way.

Keep estimates, measured observations and untested ideas explicit in UI copy, reports and code comments. Apple unified-memory observations must not be represented as dedicated VRAM. A remote endpoint's missing process memory must remain unavailable. Changing an experiment or runtime should invalidate resume rather than silently combine incompatible evidence.

Use small sanitised fixtures. Do not commit API keys, private evaluation data, generated project source belonging to someone else, model weights, `.llm-optimise` state or machine-specific run output accidentally. Confirm licence/redistribution rights before adding a dataset or other artifact.

## Pull requests

Explain the user-facing problem, the resulting behaviour and the evidence for it. State the operating system, Python/runtime version and exact checks for relevant live tests. Distinguish unit/contract tests from local inference, hardware, cloud calls, Docker execution and training.

A useful description answers:

1. What concrete workflow or failure changes?
2. What is the final implementation and why is it appropriate?
3. What checks ran and what did they establish?
4. What material limits remain untested?

Keep changes focused, preserve existing interfaces where practical and update affected help alongside code. Avoid unrelated formatting churn, secret-bearing logs, dependency additions without a clear need, and unsupported performance claims. Do not alter external accounts, billing or providers as part of a test without explicit authorisation.

## Reporting a bug

Include a minimal reproduction, expected/actual behaviour, the exact command or GUI steps, version information, and sanitised relevant logs. For experiments, include the configuration and the failing trial's status/rejection reasons when shareable. For rendering problems, include the browser/version and the interaction that fails.

Do not publish credentials or private prompts in an issue. Reduce the reproduction to synthetic inputs when possible. A report identifying a missing measurement is more useful than filling the field with an assumed value.
