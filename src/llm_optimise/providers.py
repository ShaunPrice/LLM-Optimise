"""Bring-your-own-endpoint generation; exactly one request, no hidden provider fallback."""

from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError
from urllib.request import Request

from .network import open_request, validate_local_destination


def estimate_input_tokens(messages):
    # Conservative planning proxy, intentionally not presented as a tokenizer count.
    return sum(len(m["content"].encode("utf-8")) + 32 for m in messages) + 256


def complete(model, messages, max_tokens, *, timeout=120, json_schema=None):
    if model.location == "local":
        validate_local_destination(model.base_url)
    key = os.environ.get(model.api_key_env) if model.api_key_env else None
    if model.api_key_env and not key:
        raise ValueError(
            f"set {model.api_key_env} in the launching environment; do not paste keys into model settings"
        )
    if model.location == "cloud" and not key:
        raise ValueError("cloud models require an API key environment variable")
    if max_tokens <= 0 or max_tokens > model.max_output_tokens:
        raise ValueError("requested output exceeds model limit")
    headers = {"Content-Type": "application/json"}
    if model.provider == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if key:
            headers["x-api-key"] = key
        body = {
            "model": model.model,
            "max_tokens": max_tokens,
            "system": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
            "messages": [m for m in messages if m["role"] != "system"],
        }
        url = model.base_url.rstrip("/") + "/messages"
    elif model.provider == "openai":
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {
            "model": model.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }
        if json_schema is not None and model.supports_json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "code_proposal", "strict": True, "schema": json_schema},
            }
        url = model.base_url.rstrip("/") + "/chat/completions"
    else:
        raise ValueError("unsupported provider")
    request = Request(url, data=json.dumps(body).encode(), headers=headers)
    started = time.monotonic()
    try:
        with open_request(request, timeout=timeout) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("provider response exceeded 4 MiB")
            data = json.loads(raw)
    except HTTPError as exc:
        # Provider error bodies can echo confidential prompts and tokens; don't persist them.
        raise RuntimeError(
            f"{model.id} returned HTTP {exc.code}; check model ID, credentials and endpoint compatibility"
        ) from None
    if data.get("error"):
        raise RuntimeError(
            "provider returned an error; no automatic retry or fallback was attempted"
        )
    usage = data.get("usage") or {}
    if model.provider == "anthropic":
        if data.get("stop_reason") == "max_tokens":
            raise RuntimeError("model reached output limit; incomplete response rejected")
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    else:
        choice = data["choices"][0]
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise RuntimeError(
                "model response was truncated or filtered; incomplete response rejected"
            )
        text = choice.get("message", {}).get("content") or ""
        input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("model returned no text")
    cost = None
    if (
        input_tokens is not None
        and output_tokens is not None
        and model.input_cost_per_million is not None
        and model.output_cost_per_million is not None
    ):
        cost = (
            input_tokens * model.input_cost_per_million
            + output_tokens * model.output_cost_per_million
        ) / 1e6
    return {
        "text": text,
        "model_id": model.id,
        "latency_ms": (time.monotonic() - started) * 1000,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "accounted_cost_usd": cost,
        "cost_note": "computed from provider token counts and configured prices; not a provider billing receipt",
    }
