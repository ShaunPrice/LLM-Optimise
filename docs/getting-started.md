# Getting started

LLM-Optimise is a local experiment lab with a browser interface and a Python CLI. Use it to measure a model against your task, compare configurations within memory and quality limits, and use your selected local or cloud endpoint for chat and code proposals.

The lightweight application does not include model weights, a GPU runtime, or Soup. Install those separately when you need them. A successful application install is not a successful model or accelerator test.

## 1. Install the application

Start in the repository directory. Python 3.10 or newer is required; the core dependency is `psutil`. Use a virtual environment so the installation does not modify your operating system's Python.

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
llm-optimise doctor
llm-optimise ui --workspace . --open
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\llm-optimise.exe doctor
.\.venv\Scripts\llm-optimise.exe ui --workspace . --open
```

Invoking the environment's executables directly avoids needing to change PowerShell's execution policy. In later examples, use `.\.venv\Scripts\llm-optimise.exe` wherever `llm-optimise` appears if the environment is not activated.

Open [the local lab](http://127.0.0.1:8765) if a browser does not open automatically. The default server listens on loopback. Keep the terminal open while using the application; press Ctrl+C to stop it. To use a different port:

```bash
llm-optimise ui --workspace . --port 8766 --open
```

For a containerised application, follow [Docker](docker.md). Run the application on the host when you want to launch a native accelerated model or use its Docker project testing controls.

## 2. Choose an inference path

| Path | What you provide | What the lab manages |
| --- | --- | --- |
| Managed llama.cpp | A compatible `llama-server` executable and local GGUF file | Server lifecycle, explicit inference settings, process measurements and experiment artifacts |
| Existing compatible endpoint | A running server, exact model ID and API base URL | Requests and response measurements; server process memory is unavailable |
| Cloud endpoint for Chat or Develop | Your provider's model ID, endpoint and API key environment variable | One routed request and its result |

For managed experiments, verify your `llama-server` installation independently. The accelerator build must match your operating system, driver and hardware. Start with CPU execution if accelerator support is uncertain. Detection in `doctor` means an executable or device was found; it does not prove a successful model load.

Place a licensed GGUF model in `models/` to make it discoverable in the GUI, or enter its absolute path. Model downloads are not automatic. The sample experiment's model filename is a configuration example; edit it to match a file you actually have.

An existing llama.cpp, MLX or Ollama server may be usable through its OpenAI-compatible interface. Compatibility depends on the server's supported request fields and response format. Use `provider: "openai"`, the exact served model ID, and a base URL including its API prefix, commonly `/v1`. See [model routing](agent-routing.md) before adding a provider.

## 3. Run a small baseline in the GUI

1. Open **Experiment lab**. Enter a name, your GGUF path, and the evaluation dataset.
2. Choose **Baseline**. Keep one configuration, CPU only, modest context, and a small output limit for the first runtime check.
3. Expand **Runtime & memory limits** and set the actual `llama-server` executable if it is not on `PATH`. Set a RAM budget appropriate to your machine and retain free system memory.
4. Start the experiment and follow **Activity**. Inspect failures before increasing the search size.
5. Review quality, latency and memory together. Open a trial for its configuration and rejection reasons.

The bundled dataset is a six-task smoke test. Passing it establishes a narrow functional result, not specialist competence. A model that loads and responds but misses the quality gate has not met your optimisation objective.

Do not run a managed chat server at the same time as a benchmark. The GUI requires it to be stopped first to reduce resource contention.

## 4. Run the same workflow from the CLI

Copy [the sample experiment](../examples/experiment.json), then edit its model, dataset, runtime and limits. Relative paths are resolved from the configuration file's directory, so keeping the copy under `examples/` preserves the sample dataset reference.

Linux/macOS:

```bash
cp examples/experiment.json examples/my-experiment.json
llm-optimise validate examples/my-experiment.json
llm-optimise run examples/my-experiment.json --output runs/my-baseline
```

Windows PowerShell:

```powershell
Copy-Item examples/experiment.json examples/my-experiment.json
.\.venv\Scripts\llm-optimise.exe validate examples/my-experiment.json
.\.venv\Scripts\llm-optimise.exe run examples/my-experiment.json --output runs/my-baseline
```

Before the first run, remove the sample candidate's `sweep` for a single baseline. Validation expands a sweep and checks the dataset without loading a model. It does not check that the model can load or that the runtime supports the requested settings.

The run produces `results.json`, request-level JSONL, runtime logs, a CSV and a standalone HTML report. See [the user guide](user-guide.md) for how to interpret them. Use a new output directory for each changed experiment.

## 5. Use a model for chat or development

In **Model router**, either start a managed local GGUF server or add your existing endpoint. A started managed server appears as `managed-local`. In **Chat** or **Develop**, explicitly select the model and placement you intend to use.

The GUI initially uses mixed placement with balanced ranking; the CLI initially uses local placement with cost ranking. Automatic balanced ranking needs known prices, latency and quality. For a first local connection with incomplete metrics, pin the model and keep hard constraints limited to facts you know.

Chat can explain tools and measured results and suggest a bounded local experiment. A suggestion needs your visible action before it runs. Develop creates a file proposal and diff; review it, apply it, and separately choose a Docker test or build. Generation alone does not execute the proposed code.

Continue with [the user guide](user-guide.md), [routing and credentials](agent-routing.md), [development workflow](development.md), and [troubleshooting](troubleshooting.md).
