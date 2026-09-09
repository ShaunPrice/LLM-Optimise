# OpenRouter live validation

On 9 September 2026 the application's own routing and completion adapters made **14 successful cloud requests** using an existing user credential held outside this repository. No credential was printed or committed. Both `openai/gpt-4.1-nano` and `openai/gpt-4.1-mini` passed all six bundled smoke tasks. Mini also generated a valid schema-constrained code proposal and a chat response.

The code proposal was reviewed, applied to an isolated project and executed in Docker with networking disabled, 128 MiB RAM and one CPU. Three unittest methods passed, covering four temperature conversion/error cases. This is one coding workflow, not a coding-model capability benchmark.

OpenRouter reported **US$0.001015** total across the 14 responses. The test used a US$0.10 preflight estimate ceiling and no automatic retry. Prices were read from the live model catalogue before requests. The response cost is recorded separately from the application calculation; shared-account balance changes are not used to infer this test's cost.

- [Summary and container output](results.json)
- [Synthetic request results, route decisions, response IDs and costs](requests.json)
- [Reusable model registry](../../examples/openrouter-models.json)

A further GUI chat request verified the selected cloud model, successful response and displayed provider-reported cost. It cost US$0.0003436, taking the full 15-call validation to **US$0.0013586**. [GUI response evidence](gui-chat.json). An initial local report-format error was corrected before this request; it did not make a cloud call.

## Reproduce with your credential

Set `OPENROUTER_API_KEY` in the launching process using your existing credential manager, then run:

```bash
llm-optimise agent --models examples/openrouter-models.json --model openrouter-gpt-4.1-nano --placement cloud --objective cost --max-cost-usd 0.01 --max-tokens 128 --message "Reply only with the word ready."
```

The bounded [reproduction script](reproduce.py) refreshes prices and runs up to 12 task calls with a US$0.10 preflight estimate ceiling. It requires an explicit `--run` flag and a new `--output` directory. It does not execute generated code.

For the full workflow, use the application's `code`, `apply` and `container` commands described in the [development guide](../../docs/development.md). Refresh prices before running new tests. Do not copy credentials into example JSON.

The adapter sends `allow_fallbacks: false` and `require_parameters: true` to the official OpenRouter endpoint. Model-level placement decisions remain in LLM-Optimise; OpenRouter's initial upstream provider selection is still its own mechanism. See [OpenRouter authentication](https://openrouter.ai/docs/api_reference/authentication), [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection) and [chat response/cost fields](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).
