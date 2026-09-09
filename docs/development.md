# Develop with your selected model

Develop turns an instruction and selected project files into a reviewable change. Choose a local or cloud model, describe the work, inspect proposed files and diffs, then apply the proposal. Running the code is a separate, explicit Docker action.

This is a bounded coding assistant. It does not autonomously browse the repository, install dependencies, run arbitrary shell commands, or claim that generated code has passed tests.

## GUI workflow

1. In **Model router**, configure an endpoint with the `code` capability. Start a managed local server if appropriate.
2. Open **Develop**, enter a project name and choose **Create / open project**. GUI projects live in `projects/<name>/` under the lab workspace.
3. Select relevant text files for context. For a new project, start with a concrete specification and no files.
4. Choose a model or automatic routing policy. Use **Local only** when the request and file context should stay on configured local/private endpoints.
5. Describe the desired behaviour and how it should be tested. Choose **Generate changes**.
6. Review the summary, complete file contents and diffs. Choose **Apply changes** only for the proposal you want.
7. In **Build & test in Docker**, choose runtime, action and resource limits, then explicitly run it. Inspect logs and test count.

A useful first request is: “Create a small Python module that converts Celsius to Fahrenheit. Add unittest cases for freezing, boiling and a negative temperature. Use only the standard library, place tests under tests/, and describe the test command.” This is an illustrative request, not a claim about a generated or tested result.

Projects may contain existing code placed there with your normal editor or file tools. Refresh the file list after external edits. GUI project names use 1–64 letters, digits, underscores or dashes and begin with a letter or digit.

## CLI workflow

The CLI can target a specific project directory using `--project`. Relative context paths are resolved within that project. Store proposal output outside the source project when practical.

```bash
llm-optimise code --models .llm-optimise/models.json --placement local --model local-demo --project projects/temperature-tool --message "Create a Celsius to Fahrenheit module and standard-library unittest coverage under tests/." --max-tokens 1024 --output runs/temperature-proposal.json
```

Open `runs/temperature-proposal.json` in your editor and inspect `proposal.summary` and every `proposal.files[].diff`. Applying the proposal is explicit:

```bash
llm-optimise apply runs/temperature-proposal.json --project projects/temperature-tool
llm-optimise container --project projects/temperature-tool --runtime python --action test --memory-mib 512 --cpus 1 --timeout-s 60 --output runs/temperature-tests
```

The container image must already exist unless `--pull` is selected. Use a new container output directory for each run. See [Docker](docker.md) for images, dependencies and execution boundaries.

For a follow-up with selected context:

```bash
llm-optimise code --models .llm-optimise/models.json --placement local --model local-demo --project projects/temperature-tool --context temperature.py tests/test_temperature.py --message "Add Fahrenheit to Celsius conversion and matching tests." --max-tokens 1024 --output runs/temperature-proposal-2.json
```

Replace the context filenames with the files actually created in your project. Supplying a model ID pins the route; it does not override hard constraints.

## Response and file contract

The provider is instructed to return one JSON object with exactly `summary` and `files`. Each file contains only `path` and its complete new `content`:

```json
{
  "summary": "Describe the change and suggested validation without claiming tests ran.",
  "files": [
    {"path": "example.py", "content": "def answer():\n    return 42\n"}
  ]
}
```

The application validates the response and creates the preview. It does not accept a prose patch as a substitute for this contract. An empty files array can describe missing information without changing anything. Proposals add or replace files; they do not delete files.

| Boundary | Current limit |
| --- | --- |
| Selected context | At most 20 files, 128 KiB each, 256 KiB combined |
| Prompt | 1–64,000 characters |
| Proposed files | At most 20; 1 MiB combined file content |
| Proposal response | At most 2 MiB before parsing |
| Paths | Portable project-relative paths; no traversal, hidden paths, absolute paths, symlinks or Windows reserved names |

Model context and output limits may be tighter than these application limits. Large file rewrites can exceed the output budget even when the input fits. Narrow the request or choose a model with appropriate context/output capacity.

## Review and stale-file protection

Each preview records a hash of the original file, or its absence for a new file. Apply checks all originals before writing. If a file changed since preview, the proposal is rejected and must be regenerated against current content. A proposal cannot be applied twice. Parent/file path collisions and case-variant duplicates are rejected.

These checks protect the review boundary but do not replace version control. Review the actual diff in your editor, keep your own commits, and inspect tests for meaningful assertions. A model's suggested test command is not execution evidence. A successful Python compile check demonstrates syntax compilation, not correctness or a distributable package build.

GUI pending proposals are held in the current server process; completed job records are saved under `runs/jobs/`. Apply the reviewed GUI proposal before stopping the session, or generate a fresh proposal later. The CLI saves an explicit proposal record when `--output` is supplied.

## Iterate from evidence

After a failed container run, read stdout/stderr and supply the relevant failure in a new request along with the affected files. Do not paste credentials or unrelated private files into context. Keep each request focused enough that its diff and tests can be reviewed together.

When using cloud placement, both your instruction and selected file contents go to the configured provider. Changing routing later does not retract previously sent context. See [model routing](agent-routing.md) for credentials and destination rules.
