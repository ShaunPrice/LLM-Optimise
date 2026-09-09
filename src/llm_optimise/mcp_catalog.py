"""Discoverable MCP action schemas. Imported only by the optional MCP entry point."""

from pydantic import TypeAdapter

from .config import Experiment
from .routing import ProviderModel, RoutePolicy


def obj(properties=None, required=()):
    return {
        "type": "object",
        "properties": properties or {},
        "required": list(required),
        "additionalProperties": False,
    }


def string(description="", **kwargs):
    return {"type": "string", "description": description, "maxLength": 64000, **kwargs}


def number(minimum=0, maximum=3600):
    return {"type": "number", "minimum": minimum, "maximum": maximum}


def integer(minimum=1, maximum=10000):
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def array(items=None, maximum=1000):
    return {"type": "array", "items": items or {}, "maxItems": maximum}


def schema(cls):
    value = TypeAdapter(cls).json_schema()
    definitions = value.pop("$defs", {})

    def inline(item):
        if isinstance(item, list):
            return [inline(x) for x in item]
        if isinstance(item, dict):
            if "$ref" in item:
                return inline(definitions[item["$ref"].rsplit("/", 1)[-1]])
            return {key: inline(child) for key, child in item.items()}
        return item

    return inline(value)


JSON = {"type": "object"}
BOOL = {"type": "boolean"}
PATH = string("Existing path inside the configured workspace or an operator-allowed read root.")
PROJECT = string(
    "Project name below workspace/projects.", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
)
POLICY = schema(RoutePolicy)
EXPERIMENT = schema(Experiment)
EXPERIMENT["properties"]["candidates"]["items"]["properties"]["sweep"] = JSON
TEACHER = {
    "selected_model": string(),
    "policy": POLICY,
    "task_class": string(),
    "calibrated": BOOL,
    "max_tokens": integer(1, 8192),
}
BOUNDS = {
    "max_rss_gib": number(0.01, 4096),
    "max_gpu_gib": {"anyOf": [number(0.01, 4096), {"type": "null"}]},
    "min_available_gib": number(0, 4096),
    "timeout_s": number(1, 3600),
}
SOURCE = {"rows": array(JSON), "source": PATH}
COMPARE = {"baseline": JSON, "candidate": JSON, "gates": JSON}
EXPLORE = {
    "mode": string(enum=["capacity", "halving", "kv", "speculative", "accelerators"]),
    "experiment": EXPERIMENT,
    "capacity": JSON,
    "search": JSON,
    "kv": JSON,
    "speculative": JSON,
    "accelerators": array(JSON, 10),
    "max_duration_s": number(0.1, 3600),
}


def action(name, description, properties=None, required=(), *, read=False, endpoint=None):
    return {
        "id": name,
        "description": description,
        "read_only": read,
        "input_schema": obj(properties, required),
        "endpoint": endpoint or f"/api/workbench/{name}",
    }


CATALOG = [
    action(
        "status",
        "Inspect the current GUI session, available data, jobs, models and workbench.",
        read=True,
    ),
    action(
        "models",
        "Read configured provider IDs, costs and constraints; credentials are never returned.",
        read=True,
    ),
    action(
        "lifecycle", "Read managed model residency, leases and resource requirements.", read=True
    ),
    action("results", "List experiment result summaries with run IDs.", read=True),
    action(
        "result",
        "Read a recorded experiment by run ID.",
        {"id": string(pattern=r"^[a-zA-Z0-9_-]+$")},
        ("id",),
        read=True,
    ),
    action(
        "workbench-status",
        "List datasets, immutable adapters and calibration/cache status.",
        read=True,
    ),
    action(
        "route",
        "Preview one deterministic model choice without sending a provider request.",
        {
            **TEACHER,
            "input_tokens": integer(),
            "output_tokens": integer(1, 8192),
            "capability": string(enum=["chat", "code"]),
        },
        read=True,
        endpoint="/api/route",
    ),
    action(
        "experiment",
        "Start the GUI's measured experiment job, with quality and memory gates.",
        {"config": EXPERIMENT},
        ("config",),
        endpoint="/api/experiment",
    ),
    action(
        "explore-plan",
        "Plan bounded capacity, halving, KV, speculation or accelerator exploration; may probe allowed runtimes. Returns a GUI job ID.",
        EXPLORE,
        ("mode", "experiment"),
    ),
    action(
        "explore-run",
        "Run bounded isolated exploration workers. Held-out results never select candidates. Returns a GUI job ID.",
        EXPLORE,
        ("mode", "experiment"),
    ),
    action(
        "dataset-prepare",
        "Validate, deduplicate and split specialised rows into train/tune/held-out artifacts.",
        {
            **SOURCE,
            "name": string(),
            "seed": {"type": ["integer", "string"]},
            "ratios": obj({k: number(0, 1) for k in ("train", "tune", "held_out")}),
        },
        ("name",),
    ),
    action(
        "dataset-evaluate",
        "Score supplied predictions against known task rows; writes an evaluation artifact.",
        {**SOURCE, "predictions": JSON},
        ("predictions",),
    ),
    action(
        "dataset-compare",
        "Compare evaluations on identical task IDs and fingerprint with regression gates.",
        COMPARE,
        ("baseline", "candidate"),
    ),
    action(
        "training-probe",
        "Verify an operator-allowed Soup Python environment and pinned source. Returns a job ID.",
        {"python": PATH},
        ("python",),
    ),
    action(
        "training-run",
        "Run bounded Soup SFT in a verified environment; registration is separate from held-out proof.",
        {
            **BOUNDS,
            "python": PATH,
            "config": JSON,
            "max_steps": integer(1, 100000),
            "allow_network": BOOL,
        },
        ("python", "config"),
    ),
    action(
        "adapter-register",
        "Snapshot an existing adapter into the immutable registry.",
        {
            "path": PATH,
            "base_model": PATH,
            "python": PATH,
            "backend": string(enum=["mlx", "transformers"]),
        },
        ("path", "base_model", "python", "backend"),
    ),
    action(
        "adapter-reload",
        "Reload a registered adapter and generate a bounded smoke answer.",
        {
            **BOUNDS,
            "python": PATH,
            "adapter_id": string(),
            "prompt": string(),
            "expected": string(),
            "max_tokens": integer(1, 4096),
            "allow_network": BOOL,
        },
        ("adapter_id", "python"),
    ),
    action(
        "adapter-evaluate",
        "Evaluate a registered adapter or its base model on supplied task rows.",
        {
            **BOUNDS,
            **SOURCE,
            "python": PATH,
            "adapter_id": string(),
            "baseline": BOOL,
            "max_tokens": integer(1, 4096),
            "allow_network": BOOL,
        },
        ("adapter_id", "python"),
    ),
    action(
        "adapter-compare",
        "Compare completed base and adapter evaluations with regression gates.",
        COMPARE,
        ("baseline", "candidate"),
    ),
    action(
        "distill",
        "Generate bounded teacher data from authorised rows; records measured/estimated costs and rejects invalid outputs.",
        {
            **TEACHER,
            **SOURCE,
            "data_authorized": BOOL,
            "max_requests": integer(1, 1000),
            "max_cost_usd": number(0, 1000),
            "timeout_s": number(1, 3600),
            "min_score": number(0, 1),
            "allow_unscored": BOOL,
        },
        ("data_authorized", "max_requests", "max_cost_usd"),
    ),
    action(
        "calibrate",
        "Measure configured providers on a named task class under explicit request/cost budgets.",
        {
            **TEACHER,
            "dataset": PATH,
            "model_ids": array(string(), 100),
            "repeats": integer(1, 10),
            "max_requests": integer(1, 1000),
            "max_cost_usd": number(0, 1000),
        },
        ("dataset", "max_cost_usd"),
    ),
    action(
        "context",
        "Select relevant project spans, preserve adjacent facts and report omitted required facts.",
        {
            "project": PROJECT,
            "context_files": array(string(), 20),
            "query": string(),
            "max_chars": integer(1, 256000),
            "required_facts": array(string(), 100),
        },
        ("project", "query"),
    ),
    action(
        "feedback",
        "Stage bounded generated repairs and run protected acceptance gates in Docker. Original project changes require proposal apply.",
        {
            **TEACHER,
            "project": PROJECT,
            "prompt": string(),
            "context_files": array(string(), 20),
            "runtime": string(enum=["python", "node", "rust"]),
            "reviewed_execution": BOOL,
            "max_iterations": integer(1, 10),
            "max_cost_usd": number(0, 1000),
            "timeout_s": number(1, 3600),
            "memory_mib": integer(64, 16384),
            "cpus": number(0.1, 16),
            "cases": array(JSON, 100),
            "entrypoint": string(),
            "repeats": integer(1, 20),
            "min_quality": number(0, 1),
            "case_timeout_s": number(0.01, 30),
            "max_duration_s": number(1, 3600),
            "pull": BOOL,
        },
        ("project", "prompt", "runtime", "reviewed_execution", "max_cost_usd"),
    ),
    action(
        "component",
        "Compile/run a reviewed Python or Rust deterministic component inside Docker and score expected labels on the host.",
        {
            "project": PROJECT,
            "runtime": string(enum=["python", "rust"]),
            "reviewed_execution": BOOL,
            "cases": array(JSON, 100),
            "entrypoint": string(),
            "repeats": integer(1, 20),
            "min_quality": number(0, 1),
            "case_timeout_s": number(0.01, 30),
            "timeout_s": number(1, 3600),
            "max_duration_s": number(1, 3600),
            "memory_mib": integer(64, 16384),
            "cpus": number(0.1, 16),
            "pull": BOOL,
        },
        ("project", "runtime", "reviewed_execution", "cases"),
    ),
    action(
        "specialist",
        "Run a contract-checked specialist with explicit escalation policy and bounded cost.",
        {
            **TEACHER,
            "message": string(),
            "contract": JSON,
            "escalation_model": string(),
            "cache": BOOL,
        },
        ("message", "contract", "policy"),
    ),
    action(
        "cache-clear",
        "Delete stored exact-response cache entries; measurements remain.",
        endpoint="/api/workbench/cache-clear",
    ),
    action(
        "chat",
        "Ask a selected/routed model to interpret tools, experiments and results; no hidden cloud fallback.",
        {**TEACHER, "message": string(), "history": array(JSON, 50), "optimisation": JSON},
        ("message", "policy"),
        endpoint="/api/chat",
    ),
    action(
        "code",
        "Generate a JSON code proposal and diff against selected project files; does not execute or apply code.",
        {
            **TEACHER,
            "project": PROJECT,
            "prompt": string(),
            "context_files": array(string(), 20),
            "context_selection": JSON,
            "optimisation": JSON,
        },
        ("project", "prompt", "policy"),
        endpoint="/api/code",
    ),
    action(
        "proposal-apply",
        "Apply an existing reviewed code proposal using the core's stale-source guards.",
        {"proposal_id": string(pattern=r"^[a-f0-9]{32}$")},
        ("proposal_id",),
        endpoint="/api/apply",
    ),
    action(
        "project-open",
        "Open or create a named project and list visible file names.",
        {"name": PROJECT},
        ("name",),
        endpoint="/api/project",
    ),
    action(
        "models-configure",
        "Replace the provider catalogue. Existing trusted credential/endpoint bindings are preserved; no secret values accepted.",
        {"models": array(schema(ProviderModel), 100)},
        ("models",),
        endpoint="/api/models",
    ),
    action(
        "model-start",
        "Start/register an allowed local GGUF runtime under lifecycle memory constraints.",
        {
            "id": string(),
            "model": PATH,
            "executable": PATH,
            "supervisor_executable": PATH,
            "threads": integer(1, 256),
            "context": integer(128, 131072),
            "gpu_layers": integer(-1, 1000),
            "batch_size": integer(1, 8192),
            "ubatch_size": integer(1, 8192),
            "cache_type_k": string(),
            "cache_type_v": string(),
            "flash_attention": string(),
            "accelerator": string(),
            "device": string(),
            "ram_gib": number(0.01, 4096),
            "gpu_gib": number(0, 4096),
            "working_ram_gib": number(0, 4096),
            "max_concurrency": integer(1, 8),
            "max_rss_gib": number(0.01, 4096),
            "max_gpu_gib": number(0.01, 4096),
        },
        ("model", "executable", "max_rss_gib"),
        endpoint="/api/local/start",
    ),
    action(
        "model-unload",
        "Unload idle managed models, optionally removing their registrations.",
        {"id": string(), "remove": BOOL},
        endpoint="/api/lifecycle/unload",
    ),
    action(
        "lifecycle-settings",
        "Configure residency, idle eviction and headroom before registering managed models.",
        {
            "max_active_leases": integer(1, 8),
            "max_models": integer(1, 8),
            "min_ram_gib": number(0, 4096),
            "min_gpu_gib": number(0, 4096),
            "idle_timeout_s": number(1, 3600),
        },
        endpoint="/api/lifecycle/settings",
    ),
    action(
        "container",
        "Run the core's fixed build/test command in a bounded Docker container, with networking disabled.",
        {
            "project": PROJECT,
            "runtime": string(enum=["python", "node", "rust"]),
            "action": string(enum=["build", "test"]),
            "memory_mib": integer(64, 16384),
            "cpus": number(0.1, 16),
            "timeout_s": number(1, 3600),
            "pull": BOOL,
        },
        ("project", "runtime", "action"),
        endpoint="/api/container",
    ),
]

BY_ID = {item["id"]: item for item in CATALOG}
