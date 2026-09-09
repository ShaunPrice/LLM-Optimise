# Model routing and provider setup

The router chooses from endpoints you configure. It does not discover subscriptions, obtain credentials, download models, or maintain a live price catalogue. A route preview makes no inference request. Chat, `agent` and code generation send one request to the selected provider; there are no hidden retries or automatic cloud fallbacks.

## Add a model

Use **Model router → Add model** or edit **Advanced model registry JSON**. GUI settings are saved in `.llm-optimise/models.json` under the selected workspace. CLI commands accept any catalogue file through `--models`.

This local catalogue is a template. Replace `model-served-by-your-runtime` and the limits with the exact values of your running server. The zero prices explicitly mean no provider API charge; they exclude electricity and hardware cost. Unknown measurements remain `null`.

```json
[
  {
    "id": "local-demo",
    "model": "model-served-by-your-runtime",
    "provider": "openai",
    "location": "local",
    "base_url": "http://127.0.0.1:8080/v1",
    "context_window": 4096,
    "max_output_tokens": 2048,
    "api_key_env": null,
    "input_cost_per_million": 0,
    "output_cost_per_million": 0,
    "latency_ms": null,
    "quality": null,
    "ram_gib": null,
    "gpu_gib": null,
    "capabilities": ["chat", "code"]
  }
]
```

`id` is the lab's unique registry identifier; `model` is the name your provider accepts. `provider` selects the wire protocol: `openai` for an OpenAI-compatible chat-completions endpoint or `anthropic` for the messages protocol. It does not determine whether execution is local or cloud; `location` does that.

OpenAI-compatible requests append `/chat/completions` to `base_url`. Anthropic requests append `/messages`. Include the appropriate API prefix, usually `/v1`, but not the final operation path. Local llama.cpp, MLX and Ollama services must expose the required compatible interface. The app's agent requests use `max_completion_tokens`; the benchmark adapter has a separate streaming request shape. A server accepting one path does not prove full compatibility with the other.

Live cloud execution is separate from protocol contract tests. Verify your own provider/model combination with a small explicit request before using it for substantive work. There is no built-in list of current commercial model IDs or prices.

## Credentials and destination handling

For a keyed endpoint, set `api_key_env` to an environment variable name such as `LAB_MODEL_API_KEY`. Set the value in the environment that launches the application. Never place the key itself in catalogue JSON, a URL, a prompt, or a repository file.

Linux/macOS, using a value provided securely in your shell session:

```bash
export LAB_MODEL_API_KEY
llm-optimise ui --workspace .
```

PowerShell can request it without echoing it:

```powershell
$labCredential = Read-Host "Provider API key" -AsSecureString
$env:LAB_MODEL_API_KEY = [System.Net.NetworkCredential]::new('', $labCredential).Password
.\.venv\Scripts\llm-optimise.exe ui --workspace .
```

The Unix `export` example exports an existing shell variable; it does not populate a missing key. Use your normal credential manager or secure shell input to provide the value first. A desktop application launched separately will not inherit variables set later in a terminal. Restart the lab from the intended environment after changing a key.

Cloud endpoints require HTTPS and a populated API key environment variable. Local endpoints must resolve to loopback or private-network addresses. A public provider cannot be made local merely by changing its label. Redirects are refused, and URL-embedded credentials are rejected.

When cloud or mixed placement selects a cloud model, the provider receives the request and supplied context. Develop includes the files you select; Chat includes lab context and conversation history. Local placement includes explicitly configured private-network servers, so it does not necessarily mean the same machine.

## Placement, objectives and hard constraints

| Setting | Effect |
| --- | --- |
| `local` | Only models labelled local are eligible. Public destinations are rejected when the request is made. |
| `cloud` | Only cloud models are eligible. |
| `mixed` | Both locations can compete. It still chooses one endpoint for the request. |
| `cost` | Rank by configured input/output token price estimates. Both prices must be known. |
| `performance` | Rank by configured request latency. Latency must be known. |
| `balanced` | Rank using normalised cost and latency plus quality deficit, with weights `0.4`, `0.4`, `0.2`. All three metrics must be known. |

Balanced normalisation uses the feasible candidates in that decision. Adding or removing an alternative can change the ranking. Ties are broken deterministically by registry ID.

Hard constraints check capability, placement, context size, maximum output, optional cost, optional latency, minimum quality, and local RAM/GPU estimates. A required metric that is unknown cannot establish compliance. Local RAM and GPU constraints use catalogue estimates, not live hardware measurements or runtime enforcement; cloud models are not tested against local memory limits.

Pinning a model bypasses objective ranking and its missing-metric requirements, but still obeys all hard constraints. This is useful for an initial endpoint check before you have calibrated latency and quality.

```bash
llm-optimise route --models .llm-optimise/models.json --placement local --objective cost --capability chat --input-tokens 800 --max-tokens 256
llm-optimise agent --models .llm-optimise/models.json --placement local --model local-demo --message "Reply with the word connected." --max-tokens 64 --output runs/connection-check.json
```

The GUI initially selects mixed/balanced; the CLI initially selects local/cost. Choose these settings deliberately before sending your first request. A managed server appears as `managed-local` while running. Pin it for initial use if its catalogue lacks metrics required by the chosen objective.

GUI placement, objective and hard-constraint settings govern Chat and Develop as well as the route preview. Recheck the current settings when switching views. The GUI requests up to 1,024 output tokens for Chat and 2,048 for Develop; select a model with enough output and context capacity. CLI `--max-tokens` allows a different explicit output bound.

## Calibrate the catalogue

Use prices from your actual provider agreement and date your working notes. Populate latency from comparable prompts and output sizes on the same execution path. Use quality from a representative evaluation with a consistent definition across models. Populate local RAM/GPU estimates only when you have suitable observations; leave unsupported Apple GPU telemetry unknown.

The lab does not automatically convert benchmark results into catalogue metrics. Review the experiment and enter suitable values. Do not compare a short cached response's latency against a long uncached generation as though they represented the same workload.

Before a request, token planning uses a conservative UTF-8 byte proxy plus framing reserve, not the provider's exact tokenizer. Estimated cost is based on that input estimate and the requested output limit. After a response, `accounted_cost_usd` uses reported token counts and your configured prices; it is not a provider billing receipt. Missing usage or prices leaves it unavailable.

A cost or latency constraint controls selection against supplied estimates. It cannot guarantee a final invoice amount or actual completion latency. No feasible route produces an error with per-model rejection reasons, allowing you to correct a catalogue, choose a model, or revise an explicit constraint.

## Structured code output

Managed llama.cpp models enable `supports_json_schema: true`. Develop and CLI `code` then request schema-constrained JSON through the OpenAI-compatible adapter. Other endpoints default to false; enable the field in advanced registry JSON only if your endpoint supports `response_format` with `json_schema`. This improves output syntax, not semantic correctness: review and test generated code. Anthropic requests currently use the prompted JSON format without this extension.

## OpenRouter

Register an OpenAI-compatible model with base URL `https://openrouter.ai/api/v1`, the full provider/model identifier and `api_key_env: "OPENROUTER_API_KEY"`. [Example registry](../examples/openrouter-models.json) contains the two models used in live validation; its prices are a dated snapshot, so refresh them from your agreement or OpenRouter before treating them as current.

For Chat, Develop and CLI `agent`/`code` requests to this official endpoint, the application disables gateway provider fallbacks and requires support for requested parameters. It records OpenRouter's returned `usage.cost` separately from its own token-price calculation, along with response ID and resolved model. The GUI displays reported cost when available. Other compatible gateways may have their own internal routing policies; configure those separately. [Authentication](https://openrouter.ai/docs/api_reference/authentication), [provider controls](https://openrouter.ai/docs/guides/routing/provider-selection), [response schema](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).

Keep the actual credential in your environment or existing credential manager. The registry contains only its variable name. Testing uses synthetic tasks; no source repository or private document needs to be sent to a cloud model.


## Experimental workbench additions

The [Workbench guide](workbench.md) documents executable dataset, adapter, search, calibration, caching, context, distillation, component, repair and model-lifecycle workflows shared by the GUI and CLI. Native process supervision and desktop packaging are documented under [native](../native/README.md) and [desktop](../desktop/README.md).
