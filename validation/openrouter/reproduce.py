"""Explicit opt-in live cloud smoke test; reads OPENROUTER_API_KEY from the environment.

Usage from the repository: python validation/openrouter/reproduce.py --run --output runs/cloud-check
Requires an installed llm-optimise and performs at most 12 inference calls. No code execution.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
from urllib.request import Request

from llm_optimise.agent import estimate_input_tokens, run_agent
from llm_optimise.network import open_request
from llm_optimise.quality import load_tasks
from llm_optimise.routing import ProviderModel, RoutePolicy
from llm_optimise.runner import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="Explicitly make billable cloud requests"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run to make live requests")
    if args.output.exists():
        parser.error("use a new output directory")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("set OPENROUTER_API_KEY in the launching environment")
    args.output.mkdir(parents=True)
    request = Request(
        "https://openrouter.ai/api/v1/models", headers={"Authorization": "Bearer " + key}
    )
    with open_request(request, timeout=30) as response:
        catalogue = json.load(response)["data"]
    tasks = load_tasks(Path(__file__).resolve().parents[2] / "examples" / "tasks.jsonl")
    if len(tasks) > 6:
        parser.error("this bounded fixture supports at most six tasks")
    result = {
        "scope": "Live synthetic smoke test, not domain quality",
        "requests": [],
        "estimated_maximum_cost_usd": 0.0,
    }
    for model_id in ["openai/gpt-4.1-nano", "openai/gpt-4.1-mini"]:
        info = next(m for m in catalogue if m["id"] == model_id)
        model = ProviderModel(
            id=model_id,
            model=model_id,
            provider="openai",
            location="cloud",
            base_url="https://openrouter.ai/api/v1",
            context_window=info["context_length"],
            max_output_tokens=128,
            api_key_env="OPENROUTER_API_KEY",
            input_cost_per_million=float(info["pricing"]["prompt"]) * 1e6,
            output_cost_per_million=float(info["pricing"]["completion"]) * 1e6,
        )
        for task in tasks:
            messages = [
                {
                    "role": "system",
                    "content": task.system
                    + " Return the requested answer directly. Do not wrap JSON in markdown.",
                },
                {"role": "user", "content": task.prompt},
            ]
            estimate = model.estimated_cost_usd(estimate_input_tokens(messages), 128)
            if result["estimated_maximum_cost_usd"] + estimate > 0.10:
                raise SystemExit("US$0.10 preflight estimate ceiling reached; no further requests")
            result["estimated_maximum_cost_usd"] += estimate
            response = run_agent(
                [model],
                messages,
                RoutePolicy(placement="cloud", objective="cost", max_cost_usd=0.01),
                model.id,
                128,
            )
            result["requests"].append(
                {
                    "task": task.id,
                    "model": dataclasses.asdict(model),
                    "score": task.score(response["completion"]["text"]),
                    **response,
                }
            )
            write_json(args.output / "results.json", result)
            print(task.id, model_id, result["requests"][-1]["score"])


if __name__ == "__main__":
    main()
