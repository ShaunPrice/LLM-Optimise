"use strict";

(() => {
  const lab = window.LLMOptimise;
  const $ = (id) => document.getElementById(id);
  const el = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined && text !== null) node.textContent = String(text); if (cls) node.className = cls; return node; };
  const pretty = (value) => JSON.stringify(value, null, 2);
  const active = (status) => ["queued", "running", "pending", "starting", "cancelling"].includes(status);
  const state = { area: "explore", tool: "capacity", request: null, result: null, pending: new Map(), completed: new Map(), seen: new Set(), reports: new Map(), preferences: { task_class: "general", cache: false, calibrated: false, prefix_cache: false }, dirty: new Set() };
  const field = (key, label, type = "text", options = {}) => ({ key, label, type, ...options });
  const number = (key, label, value, min = 0, max = null) => field(key, label, "number", { value, min, max });
  const text = (key, label, options = {}) => field(key, label, "text", { wide: true, ...options });
  const select = (key, label, options, value) => field(key, label, "select", { options, value });
  const check = (key, label, value = false, options = {}) => field(key, label, "checkbox", { value, wide: true, ...options });
  const json = (key, label, value = "", options = {}) => field(key, label, "json", { value, wide: true, ...options });
  const lines = (key, label, options = {}) => field(key, label, "lines", { wide: true, ...options });
  const source = (label = "Task dataset path", options = {}) => text("source", label, { required: true, list: "wb-task-paths", ...options });
  const project = () => text("project", "Project", { required: true, list: "project-names", value: () => lab.project, placeholder: "Open or create a project in Develop" });
  const teacherFields = () => [field("selected_model", "Selected model", "model", { required: true, wide: true }), select("placement", "Placement", [["local", "Local only"], ["cloud", "Cloud only"], ["mixed", "Local or cloud"]], "local"), select("objective", "Optimisation target", [["cost", "Lowest cost"], ["performance", "Fastest response"], ["balanced", "Balanced"]], "cost"), text("task_class", "Task class", { value: () => state.preferences.task_class, placeholder: "classification, extraction, coding…" })];
  const runtimeBudget = () => [number("max_rss_gib", "Maximum process RAM · GiB", 4, 0.1), number("timeout_s", "Job timeout · seconds", 600, 1, 86400), number("max_gpu_gib", "Maximum GPU memory · GiB", "", 0.1), number("min_available_gib", "Keep RAM available · GiB", 1, 0)];
  const engineOptions = [["mlx", "Apple silicon · MLX"], ["transformers", "CUDA / Transformers"]];
  const python = () => text("python", "Soup environment Python", { required: true, placeholder: ".venv-soup/bin/python or C:\\…\\python.exe" });
  const policy = (v) => ({ placement: v.placement || "local", objective: v.objective || "cost", ...(v.max_cost_usd !== undefined ? { max_cost_usd: v.max_cost_usd } : {}) });
  const teacherRequest = (v) => ({ selected_model: v.selected_model, task_class: v.task_class || "general", policy: policy(v) });
  const evalReport = (id) => { const record = state.reports.get(id); if (!record) throw new Error("Choose a completed report first."); return record.result.evaluation || record.result; };
  const fullReport = (id) => { const record = state.reports.get(id); if (!record) throw new Error("Choose a completed evaluation first."); return record.result; };
  const compareFields = () => [field("baseline", "Baseline result", "report", { required: true, wide: true }), field("candidate", "Candidate result", "report", { required: true, wide: true }), number("max_quality_drop", "Allowed quality drop", 0, 0, 1), number("max_regressed_tasks", "Allowed individual regressions", 0, 0), number("max_error_cost_increase", "Allowed error-cost increase", 0, 0), number("max_schema_failure_increase", "Allowed new schema failures", 0, 0)];
  const gates = (v) => Object.fromEntries(["max_quality_drop", "max_regressed_tasks", "max_error_cost_increase", "max_schema_failure_increase"].map((key) => [key, v[key]]));
  const boundRequest = (v) => ({ max_rss_gib: v.max_rss_gib, max_gpu_gib: v.max_gpu_gib, min_available_gib: v.min_available_gib, timeout_s: v.timeout_s });
  const exploreFields = () => [text("model", "GGUF model", { required: true, list: "gguf-paths", autofill: "model" }), text("executable", "llama-server executable", { required: true, value: () => $("experiment-executable").value, autofill: "executable" }), text("supervisor_executable", "Native supervisor executable · optional", { placeholder: "/path/to/llm-supervisor", note: "Leave blank for the Python process supervisor." }), source("Tuning task dataset", { autofill: "tasks" }), number("threads", "CPU threads", 4, 1), select("gpu_layers", "Execution", [["0", "CPU only"], ["-1", "All supported GPU layers"]], "0"), number("min_quality", "Minimum task quality", 0.9, 0, 1), number("max_rss_gib", "Maximum process RAM · GiB", 4, 0.1), number("min_available_gib", "Keep RAM available · GiB", 1, 0.1), number("max_duration_s", "Whole search · seconds", 600, 1, 3600), number("max_tokens", "Output tokens per task", 64, 1, 2048), number("repeats", "Final repeats", 1, 1, 10), check("constrain_json", "Use dataset JSON schemas to constrain output", true)];
  function experiment(v) {
    return { name: tools[state.tool].title, dataset: v.source, candidates: [{ name: "base", model: v.model, executable: v.executable, ...(v.supervisor_executable ? { supervisor_executable: v.supervisor_executable } : {}), threads: v.threads, gpu_layers: Number(v.gpu_layers), context: 512, batch_size: 256, ubatch_size: 64, constrain_json: v.constrain_json }], repeats: v.repeats, warmup: 1, max_tokens: v.max_tokens, max_trials: 32, timeout_s: 120, startup_timeout_s: 120, limits: { min_quality: v.min_quality, max_rss_gib: v.max_rss_gib, min_available_gib: v.min_available_gib } };
  }
  const exploreRequest = (v) => ({ mode: state.tool, experiment: experiment(v), max_duration_s: v.max_duration_s });
  const parseNumbers = (value) => String(value).split(/[\s,]+/).filter(Boolean).map((part) => { const n = Number(part); if (!Number.isFinite(n)) throw new Error("Enter comma-separated numbers."); return n; });

  const tools = {
    capacity: { area: "explore", title: "Capacity explorer", operation: "explore-run", action: "Run capacity search", description: "Raise one resource dimension in fresh workers. Retain the last configuration that passes your quality and memory gates.", fields: () => [...exploreFields(), select("dimension", "Dimension to increase", [["context", "Context tokens"], ["batch_size", "Batch size"], ["ubatch_size", "Microbatch size"], ["threads", "CPU threads"]], "context"), text("values", "Increasing values", { value: "512, 1024, 2048, 4096", required: true }), check("verify_recovery", "Verify memory recovery between workers", true)], build: (v) => ({ ...exploreRequest(v), capacity: { dimensions: { [v.dimension]: parseNumbers(v.values) }, verify_recovery: v.verify_recovery, recovery_timeout_s: 10 } }) },
    halving: { area: "explore", title: "Progressive experiment search", operation: "explore-run", action: "Run progressive search", description: "Start with smaller task samples, then give promising candidates more evidence. The optional held-out split is evaluated only after selection.", fields: () => [...exploreFields(), text("thread_values", "Candidate CPU thread counts", { value: "2, 4", required: true }), text("context_values", "Candidate context lengths", { value: "512, 2048", required: true }), text("task_budgets", "Tasks per round", { value: "2, 6", required: true, note: "Each value must fit the tuning dataset; the final round includes all tuning tasks." }), number("eta", "Keep one in this many candidates", 2, 2, 8), text("heldout_dataset", "Held-out task dataset · optional", { list: "wb-task-paths" })], build: (v) => { const request = exploreRequest(v); request.experiment.candidates[0].sweep = { threads: parseNumbers(v.thread_values), context: parseNumbers(v.context_values) }; request.search = { task_budgets: parseNumbers(v.task_budgets), eta: v.eta, ...(v.heldout_dataset ? { heldout_dataset: v.heldout_dataset } : {}) }; return request; } },
    kv: { area: "explore", title: "KV cache & context", operation: "explore-run", action: "Compare cache settings", description: "Compare context length and KV precision at the same task quality. Quantised value caches enable Flash Attention; installed runtime support is checked.", fields: () => [...exploreFields(), text("contexts", "Context lengths", { value: "512, 1024, 2048", required: true }), field("cache_types", "Cache precisions", "multi", { options: [["f16", "f16 · full precision"], ["q8_0", "q8_0 · 8-bit"], ["q4_0", "q4_0 · 4-bit"]], value: ["f16", "q8_0"], required: true, wide: true })], build: (v) => ({ ...exploreRequest(v), kv: { contexts: parseNumbers(v.contexts), cache_types: v.cache_types } }) },
    speculative: { area: "explore", title: "Speculative decoding", operation: "explore-run", action: "Compare draft models", description: "Measure a baseline against explicitly selected draft models. Acceptance counters stay unavailable when the runtime does not expose them.", fields: () => [...exploreFields(), lines("draft_models", "Draft GGUF paths · one per line", { required: true, placeholder: "/path/to/small-draft.gguf" }), text("draft_max", "Draft lengths", { value: "4, 8", required: true })], build: (v) => ({ ...exploreRequest(v), speculative: { draft_models: v.draft_models, draft_max: parseNumbers(v.draft_max) } }) },
    accelerators: { area: "explore", title: "Accelerator comparison", operation: "explore-run", action: "Compare installed backends", description: "Use identical weights and tasks across available CPU, Metal, CUDA, Vulkan or ROCm builds. Unverified devices are excluded with a reason.", fields: () => [...exploreFields(), field("accelerators", "Backends to compare", "multi", { options: ["cpu", "metal", "cuda", "vulkan", "rocm"].map((x) => [x, x === "cpu" ? "CPU" : x === "metal" ? "Metal" : x.toUpperCase()]), value: ["cpu", "metal"], wide: true, required: true }), text("alternate_executable", "Other backend executable · optional", { note: "Leave blank to probe the main executable for every selected backend. Advanced JSON supports a distinct executable/device per backend." })], build: (v) => ({ ...exploreRequest(v), accelerators: v.accelerators.map((accelerator) => ({ accelerator, executable: accelerator === "cpu" ? v.executable : v.alternate_executable || v.executable })) }) },
    "dataset-prepare": { area: "data", title: "Import & split a dataset", action: "Validate and create splits", description: "Deduplicate prompts and keep every related source group in one split. Conflicting answers fail validation; grouping and task error costs are preserved.", fields: () => [text("name", "Dataset name", { value: "My specialised task", required: true }), text("source", "Existing JSONL / JSON path · optional", { list: "wb-task-paths", note: "Use a path or paste rows below. Rows can use the task format or Alpaca instruction/input/output." }), field("rows", "Or paste dataset rows", "rows", { wide: true, placeholder: '{"id":"case-1","group":"source-document-1","prompt":"Classify…","expected":"normal","error_cost":5}' }), text("seed", "Stable split seed", { value: "42", required: true }), number("train", "Training fraction", 0.7, 0.01, 0.98), number("tune", "Tuning fraction", 0.15, 0.01, 0.98), number("held_out", "Held-out fraction", 0.15, 0.01, 0.98)], build: (v) => ({ name: v.name, seed: v.seed, ratios: { train: v.train, tune: v.tune, held_out: v.held_out }, ...(v.rows?.length ? { rows: v.rows } : { source: v.source }) }) },
    "dataset-evaluate": { area: "data", title: "Score task predictions", action: "Evaluate predictions", description: "Score expected answers, missing outputs, schema validity and task-specific error costs. Use a metadata split to retain its error-cost settings.", fields: () => [source("Dataset or split metadata path", { list: "wb-metadata-paths" }), json("predictions", "Predictions by task ID", '{"case-1": "normal"}', { required: true, note: "A JSON object mapping task IDs to the model's output strings." })], build: (v) => ({ source: v.source, predictions: v.predictions }) },
    "dataset-compare": { area: "data", title: "Regression gate", action: "Compare task results", description: "Require the same dataset fingerprint and task IDs. Detect regressions that an unchanged overall score could otherwise hide.", fields: compareFields, build: (v) => ({ baseline: evalReport(v.baseline), candidate: evalReport(v.candidate), gates: gates(v) }) },
    "training-probe": { area: "train", title: "Verify Soup environment", action: "Verify installed source", description: "Inspect the selected Python environment and match its installed Soup source against the reviewed pinned revision. No model is loaded.", fields: () => [python()], build: (v) => ({ python: v.python }) },
    "training-run": { area: "train", title: "Train a Soup adapter", action: "Start bounded training", description: "Run actual Soup SFT in an isolated environment. Save provenance, register an immutable adapter snapshot, then reload and evaluate it separately.", fields: () => [python(), select("backend", "Training backend", engineOptions, "mlx"), check("stream_layers", "Stream frozen layers · Transformers only", false), text("base", "Downloaded base model directory", { required: true, placeholder: "/path/to/local/model" }), text("train_data", "Training JSONL", { required: true, list: "wb-sft-paths" }), select("format", "Training data format", [["chatml", "Chat messages"], ["alpaca", "Alpaca instruction/input/output"]], "chatml"), number("epochs", "Epochs", 1, 1, 100), number("rank", "LoRA rank", 8, 1, 256), number("max_length", "Maximum sequence length", 512, 32, 131072), number("batch_size", "Batch size", 1, 1, 256), number("max_steps", "Maximum planned microsteps", 100, 1), ...runtimeBudget()], build: (v) => { if (v.stream_layers && v.backend === "mlx") throw new Error("Layer streaming and MLX are separate paths. Select Transformers for streaming."); return { python: v.python, max_steps: v.max_steps, ...boundRequest(v), config: { base: v.base, task: "sft", backend: v.backend, data: { train: v.train_data, format: v.format, val_split: 0, max_length: v.max_length, train_on_responses_only: true }, training: { epochs: v.epochs, lr: 0.0002, batch_size: v.batch_size, gradient_accumulation_steps: 1, gradient_checkpointing: true, quantization: "4bit", lora: { r: v.rank, alpha: 2 * v.rank }, ...(v.stream_layers ? { stream_layers: true, stream_source: "auto", stream_buffers: 2 } : {}) } } }; } },
    "adapter-register": { area: "train", title: "Register an existing adapter", action: "Snapshot and register", description: "Copy final adapter files into an immutable registry and record the base-model fingerprint. Registration alone does not verify reload or task quality.", fields: () => [text("path", "Adapter output directory", { required: true }), text("base_model", "Downloaded base model directory", { required: true }), select("backend", "Adapter backend", engineOptions, "mlx"), python()], build: (v) => v },
    "adapter-reload": { area: "train", title: "Reload adapter & generate", action: "Verify reload", description: "Start a fresh process, verify that every saved adapter tensor attaches, and generate a bounded answer. This is a smoke test; use held-out evaluation next.", fields: () => [field("adapter_id", "Registered adapter", "adapter", { required: true, wide: true }), field("prompt", "Smoke-test prompt", "textarea", { value: "Reply with the word ready.", required: true, wide: true }), text("expected", "Expected answer", { value: "ready", required: true }), number("max_tokens", "Maximum answer tokens", 64, 1, 4096), ...runtimeBudget()], build: (v) => v },
    "adapter-evaluate": { area: "train", title: "Evaluate held-out tasks", action: "Evaluate selected model", description: "Run the adapter or its base model in a fresh process against the same task rows. Keep tuning and final held-out decisions separate.", fields: () => [field("adapter_id", "Registered adapter", "adapter", { required: true, wide: true }), source("Held-out metadata / task dataset", { list: "wb-metadata-paths" }), check("baseline", "Evaluate the base model without the adapter", false), number("max_tokens", "Maximum answer tokens", 128, 1, 4096), ...runtimeBudget()], build: (v) => v },
    "adapter-compare": { area: "train", title: "Compare adapter quality", action: "Apply regression gates", description: "Compare completed base/adapter evaluations on identical tasks. A completed training job alone is not an optimisation success.", fields: compareFields, build: (v) => ({ baseline: fullReport(v.baseline), candidate: fullReport(v.candidate), gates: gates(v) }) },
    distill: { area: "train", title: "Distil specialist labels", action: "Generate and curate labels", description: "Use a selected teacher to produce training labels within an explicit request and cost budget. Preserve rejected responses and separate schema-only labels from reference-scored labels.", fields: () => [...teacherFields(), source("Training metadata / task rows", { list: "wb-metadata-paths" }), number("max_requests", "Maximum teacher requests", 10, 1, 10000), number("max_cost_usd", "Total provider budget · USD", 0.02, 0), number("max_tokens", "Answer tokens", 128, 1, 4096), number("min_score", "Minimum reference score", 1, 0, 1), check("data_authorized", "I authorise sending these training rows to the selected teacher", false, { required: true }), check("allow_unscored", "Accept labels without reference answers or schema gates", false, { note: "Keep disabled for evaluated curation. This does not make unscored labels verified." })], build: (v) => ({ ...v, ...teacherRequest(v) }) },
    preferences: { area: "intelligence", title: "Chat & coding optimisation", action: "Use these preferences", local: true, description: "Set the task class and optional reuse controls for subsequent Chat and Develop requests. Exact caching needs an immutable model revision; calibrated routing needs comparable measurements.", fields: () => [text("task_class", "Task class", { value: () => state.preferences.task_class, required: true }), check("cache", "Reuse exact matching results", state.preferences.cache), check("prefix_cache", "Reuse eligible exact instruction prefixes", state.preferences.prefix_cache), check("calibrated", "Route using measured task performance", state.preferences.calibrated)], build: (v) => v },
    calibrate: { area: "intelligence", title: "Calibrate model routing", action: "Measure selected models", description: "Measure quality, latency and cost for a task class and prompt-size range. Calibration sends real requests to the selected local or cloud models.", fields: () => [field("model_ids", "Models to measure", "models", { required: true, wide: true }), text("dataset", "Representative task dataset", { required: true, list: "wb-task-paths" }), text("task_class", "Task class", { value: () => state.preferences.task_class, required: true }), number("repeats", "Repeats per task", 1, 1, 10), number("max_requests", "Total request budget", 20, 1, 1000), number("max_cost_usd", "Total provider budget · USD", 0.02, 0), number("max_tokens", "Maximum answer tokens", 128, 1, 4096)], build: (v) => v },
    specialist: { area: "intelligence", title: "Specialist response contract", action: "Run contracted request", description: "Accept an answer only when it passes your exact output contract. Otherwise abstain, or try one explicitly selected escalation model within the same total budget.", fields: () => [...teacherFields(), field("message", "Task request", "textarea", { required: true, wide: true, value: "Classify temperature=70. Return normal below 80, otherwise alarm." }), select("contract_type", "Acceptance contract", [["allowed", "Allowed exact outputs"], ["schema", "JSON schema"]], "allowed"), lines("allowed_outputs", "Allowed answers · one per line", { value: "normal\nalarm" }), json("json_schema", "JSON schema · if selected", '{"type":"object","required":["label"],"properties":{"label":{"enum":["normal","alarm"]}},"additionalProperties":false}'), field("escalation_model", "Escalation model · optional", "model", { wide: true, emptyLabel: "Abstain if the first answer fails" }), number("max_cost_usd", "Total provider budget · USD", 0.01, 0), number("max_tokens", "Maximum answer tokens", 128, 1, 4096)], build: (v) => ({ ...teacherRequest(v), message: v.message, contract: v.contract_type === "schema" ? { json_schema: v.json_schema } : { allowed_outputs: v.allowed_outputs }, ...(v.escalation_model ? { escalation_model: v.escalation_model } : {}), max_tokens: v.max_tokens }) },
    context: { area: "intelligence", title: "Select useful context", action: "Preview selected context", description: "Retrieve relevant excerpts from selected project files before generation. Inspect required-fact coverage and omitted context alongside size savings.", fields: () => [project(), lines("context_files", "Project files · one relative path per line", { required: true, value: () => lab.selectedFiles().join("\n") }), field("query", "Task or retrieval query", "textarea", { required: true, wide: true }), number("max_chars", "Context character budget", 12000, 256), lines("required_facts", "Facts that must be retained · optional", { placeholder: "One exact fact per line" })], build: (v) => v },
    "cache-clear": { area: "intelligence", title: "Manage exact cache", action: "Clear saved cache entries", description: "Inspect cache usage in Lab inventory. Clearing removes reusable cached outputs; calibration observations remain available.", fields: () => [], build: () => ({}) },
    feedback: { area: "build", title: "Test-guided coding loop", action: "Run staged repair loop", description: "Generate changes in a staging copy, run container tests, and return one final proposal for review. Existing acceptance tests are protected and your project is changed only after you apply the proposal.", fields: () => [project(), ...teacherFields(), field("prompt", "Implementation or repair request", "textarea", { required: true, wide: true }), lines("context_files", "Context files · one relative path per line", { value: () => lab.selectedFiles().join("\n") }), select("runtime", "Project runtime", [["python", "Python"], ["node", "Node.js"], ["rust", "Rust"]], "python"), number("max_iterations", "Maximum repair iterations", 3, 1, 10), number("max_cost_usd", "Total provider budget · USD", 0.05, 0), number("max_tokens", "Generated tokens per iteration", 2048, 1, 32768), number("memory_mib", "Container memory · MiB", 512, 64), number("timeout_s", "Each container run · seconds", 60, 1), check("pull", "Allow Docker image downloads", false), check("reviewed_execution", "Allow generated changes to be tested inside the isolated container", false, { required: true })], build: (v) => ({ ...v, ...teacherRequest(v), cpus: 1 }) },
    component: { area: "build", title: "Deterministic task component", action: "Benchmark reviewed component", description: "Run a precise JSON-in/JSON-out component without a model call. Compare task correctness and process latency; Rust is compiled inside Docker before measurement.", fields: () => [project(), select("runtime", "Component language", [["python", "Python"], ["rust", "Rust"]], "python"), text("entrypoint", "Relative entrypoint path", { value: "main.py", required: true }), json("cases", "Input / expected-output cases", '[{"id":"example","input":{"temperature":70},"expected":{"label":"normal"}}]', { required: true }), number("repeats", "Repeats per case", 3, 1, 20), number("min_quality", "Minimum quality", 1, 0, 1), number("memory_mib", "Container memory · MiB", 512, 64), number("timeout_s", "Container timeout · seconds", 60, 1), check("pull", "Allow Docker image downloads", false), check("reviewed_execution", "I have reviewed the component source and authorise container execution", false, { required: true })], build: (v) => ({ ...v, cpus: 1 }) },
    "lifecycle-settings": { area: "runtime", title: "Memory & concurrency policy", operation: "/api/lifecycle/settings", action: "Save lifecycle policy", description: "Reserve machine headroom, limit simultaneous work and unload idle models. Remove existing managed registrations before changing this policy.", fields: () => { const s = lab.state.lifecycle?.settings || {}; return [number("max_models", "Maximum loaded models", s.max_models ?? 2, 1, 20), number("max_active_leases", "Concurrent requests", s.max_active_leases ?? 2, 1, 32), number("min_ram_gib", "Reserve system RAM · GiB", s.min_ram_gib ?? 1, 0), number("min_gpu_gib", "Reserve discrete GPU memory · GiB", s.min_gpu_gib ?? 0.25, 0), number("idle_timeout_s", "Unload after idle · seconds", s.idle_timeout_s ?? 60, 1)]; }, build: (v) => v },
    "managed-start": { area: "runtime", title: "Load a managed model", operation: "/api/local/start", action: "Register and load model", description: "Give each model a managed ID and an explicit resource allowance. Identical runtime configurations share residency; leases control simultaneous requests.", fields: () => [text("id", "Managed registry ID", { value: "managed-specialist", required: true, pattern: "managed-[A-Za-z0-9_-]+" }), text("model", "GGUF model", { required: true, list: "gguf-paths", autofill: "model" }), text("executable", "llama-server executable", { required: true, value: () => $("experiment-executable").value, autofill: "executable" }), number("threads", "CPU threads", 4, 1), number("context", "Context tokens", 4096, 128), select("gpu_layers", "GPU layers", [["-1", "All supported"], ["0", "CPU only"]], "-1"), text("supervisor_executable", "Native supervisor executable · optional", { placeholder: "/path/to/llm-supervisor" }), number("ram_gib", "Expected resident RAM · GiB", 2, 0.1), number("gpu_gib", "Expected discrete GPU memory · GiB", "", 0), number("max_rss_gib", "Stop above process RAM · GiB", 4, 0.1), number("max_gpu_gib", "Stop above GPU memory · GiB", "", 0.1), number("max_concurrency", "Concurrent requests for this model", 1, 1, 8), number("working_ram_gib", "Extra RAM per request · GiB", 0.125, 0)], build: (v) => ({ ...v, supervisor_executable: v.supervisor_executable || null, gpu_layers: Number(v.gpu_layers) }) },
    "lifecycle-unload": { area: "runtime", title: "Unload or remove models", operation: "/api/lifecycle/unload", action: "Unload selected models", description: "Release idle model residency. Removing a registration also removes it from the model router; active leases must finish or be cancelled first.", fields: () => [field("id", "Managed model", "managed", { wide: true, emptyLabel: "All managed models" }), check("remove", "Also remove the model registrations", false)], build: (v) => ({ ...(v.id ? { id: v.id } : {}), remove: v.remove }) },
  };
  const areas = [
    ["explore", "⌁", "Explore"], ["data", "▦", "Datasets"], ["train", "▤", "Adapters"],
    ["intelligence", "⇄", "Intelligence"], ["build", "⌘", "Build & test"], ["runtime", "◈", "Runtime"],
  ];

  function optionsFor(f) {
    if (f.type === "model" || f.type === "models") return (lab.state.models || []).map((model) => [model.id, `${model.id} · ${model.location}`]);
    if (f.type === "adapter") return (lab.state.workbench?.adapters || []).map((adapter) => [adapter.id, `${adapter.id.slice(0, 10)} · ${adapter.backend} · ${adapter.reload_status}`]);
    if (f.type === "managed") return (lab.state.lifecycle?.registered || []).map((model) => [model.id, `${model.id} · ${model.context} context`]);
    if (f.type === "report") return [...state.reports].filter(([, item]) => item.result.evaluation || item.result.dataset_sha256 && item.result.results).map(([id, item]) => [id, item.title]);
    return f.options || [];
  }
  function setOptions(input, f) {
    const selected = new Set(input.multiple ? [...input.selectedOptions].map((o) => o.value) : [input.value]);
    input.replaceChildren();
    if (!input.multiple) { const option = el("option", f.emptyLabel || "Choose…"); option.value = ""; input.append(option); }
    for (const [value, label] of optionsFor(f)) { const option = el("option", label); option.value = value; option.selected = selected.has(value); input.append(option); }
  }
  function control(f) {
    const label = el("label", null, f.wide ? "wb-wide" : "");
    if (f.type === "checkbox") label.classList.add("checkbox-label");
    const input = el(["json", "rows", "lines", "textarea"].includes(f.type) ? "textarea" : ["select", "multi", "model", "models", "adapter", "managed", "report"].includes(f.type) ? "select" : "input");
    input.id = "wb-f-" + f.key;
    input.name = f.key;
    if (input.tagName === "INPUT") input.type = f.type === "checkbox" ? "checkbox" : f.type === "number" ? "number" : "text";
    if (input.tagName === "SELECT") { input.multiple = ["multi", "models"].includes(f.type); setOptions(input, f); }
    if (["json", "rows"].includes(f.type)) { input.classList.add("code-input"); input.spellcheck = false; input.rows = 6; }
    if (f.required) input.required = true;
    if (f.placeholder) input.placeholder = f.placeholder;
    if (f.list) input.setAttribute("list", f.list);
    if (f.pattern) input.pattern = f.pattern;
    if (f.min !== undefined && f.min !== null) input.min = f.min;
    if (f.max !== undefined && f.max !== null) input.max = f.max;
    if (f.type === "number") input.step = ["threads", "repeats", "max_tokens", "max_requests", "max_iterations", "max_models", "max_active_leases", "max_concurrency", "context", "epochs", "batch_size", "rank", "max_length", "max_steps", "memory_mib", "max_regressed_tasks", "max_schema_failure_increase", "eta", "max_chars"].includes(f.key) ? "1" : "any";
    const defaultValue = typeof f.value === "function" ? f.value() : f.value;
    if (defaultValue !== undefined) {
      if (f.type === "checkbox") input.checked = defaultValue;
      else if (input.multiple) [...input.options].forEach((option) => { option.selected = defaultValue.includes(option.value); });
      else input.value = defaultValue;
    }
    if (f.type === "checkbox") label.append(input, document.createTextNode(f.label));
    else label.append(document.createTextNode(f.label), input);
    if (f.note) { const note = el("span", f.note, "wb-field-note"); note.id = input.id + "-note"; input.setAttribute("aria-describedby", note.id); label.append(note); }
    input.addEventListener("input", () => state.dirty.add(f.key));
    return label;
  }
  function sampleRows() {
    return Array.from({ length: 24 }, (_, i) => ({ id: `sensor-${i + 1}`, group: `source-${Math.floor(i / 2) + 1}`, task_class: "temperature-classification", prompt: `Classify this fictional reading: temperature=${i * 5}. Reply normal below 80, otherwise alarm.`, expected: i * 5 < 80 ? "normal" : "alarm", error_cost: i * 5 < 80 ? 1 : 5 }));
  }
  function extraActions() {
    const row = el("div", null, "wb-inline-action");
    const button = (label, callback) => { const b = el("button", label, "text-button"); b.type = "button"; b.addEventListener("click", callback); row.append(b); };
    if (tools[state.tool].area === "explore") button("Use current Experiment lab inputs ↗", () => {
      try { const exp = lab.experiment(); const candidate = exp.candidates[0]; for (const [key, value] of Object.entries({ model: candidate.model, executable: candidate.executable, source: exp.dataset, threads: candidate.threads, gpu_layers: candidate.gpu_layers, min_quality: exp.limits?.min_quality, max_tokens: exp.max_tokens })) if ($("wb-f-" + key) && value !== undefined) $("wb-f-" + key).value = value; lab.toast("Lab inputs copied. Choose the search settings below."); } catch (error) { lab.toast(error.message, true); }
    });
    if (state.tool === "dataset-prepare") {
      button("Insert example rows", () => { $("wb-f-rows").value = sampleRows().map((r) => JSON.stringify(r)).join("\n"); $("wb-f-source").value = ""; lab.toast("Inserted synthetic format examples. Use representative data for domain claims."); });
      const upload = el("input"); upload.type = "file"; upload.accept = ".json,.jsonl,.txt"; upload.setAttribute("aria-label", "Import dataset file, up to 1 MiB");
      upload.addEventListener("change", async () => { const file = upload.files[0]; if (!file) return; if (file.size > 1024 * 1024) { lab.toast("Use a local dataset path for files larger than 1 MiB.", true); return; } $("wb-f-rows").value = await file.text(); $("wb-f-source").value = ""; }); row.append(upload);
    }
    if (state.tool === "training-run") button("Use prepared Training recipe ↗", () => {
      const config = lab.training()?.config;
      if (!config) { lab.toast("Prepare a recipe in Training first."); return; }
      const values = { backend: config.backend, base: config.base, train_data: config.data?.train, format: config.data?.format, max_length: config.data?.max_length, rank: config.training?.lora?.r, epochs: config.training?.epochs, batch_size: config.training?.batch_size };
      for (const [key, value] of Object.entries(values)) if ($("wb-f-" + key) && value !== undefined) $("wb-f-" + key).value = value;
      $("wb-f-stream_layers").checked = Boolean(config.training?.stream_layers); lab.toast("Recipe fields copied. Select the pinned Soup environment and job limits.");
    });
    if (["feedback", "context", "component"].includes(state.tool)) button("Use open Develop project & files ↗", () => { if (!lab.project) { lab.showView("develop"); return; } $("wb-f-project").value = lab.project; if ($("wb-f-context_files")) $("wb-f-context_files").value = lab.selectedFiles().join("\n"); });
    return row.childNodes.length ? row : null;
  }
  function renderForm() {
    const tool = tools[state.tool]; state.dirty.clear();
    $("wb-tool-heading").textContent = tool.title;
    $("wb-area-label").textContent = areas.find((a) => a[0] === tool.area)[2].toUpperCase();
    $("wb-description").textContent = tool.description;
    $("wb-submit").textContent = tool.action;
    $("wb-submit").disabled = false;
    $("wb-use-json").checked = false; $("wb-json").value = ""; $("wb-form").noValidate = false;
    const fields = $("wb-fields"); fields.replaceChildren();
    const actions = extraActions(); if (actions) fields.append(actions);
    for (const f of tool.fields()) fields.append(control(f));
    if (["distill", "feedback", "calibrate", "specialist"].includes(state.tool)) fields.append(el("p", "Selected cloud models receive your task text and selected context. Provider budgets apply to this run; no model fallback is implicit.", "wb-cloud-note"));
    if (state.tool === "preferences") fields.append(el("p", "Preferences apply to the next Chat and Develop request, and measured routing also applies to the router preview. They do not run a model by themselves.", "wb-section-note"));
    $("wb-preview").textContent = tool.area === "explore" ? "Validate search plan" : "Inspect request";
    $("wb-run-note").textContent = tool.area === "explore" || ["training-run", "adapter-evaluate", "adapter-reload"].includes(state.tool) ? "Resource experiments use fresh workers. Active work must finish first; idle managed models are unloaded before measurement." : state.tool === "preferences" ? "These preferences stay in this browser session." : "Results record what ran, what passed, and what remains unverified.";
    if ($("wb-f-runtime") && state.tool === "component") $("wb-f-runtime").addEventListener("change", () => { const entry = $("wb-f-entrypoint"); if (["main.py", "main.rs"].includes(entry.value)) entry.value = $("wb-f-runtime").value === "rust" ? "main.rs" : "main.py"; });
    if ($("wb-f-contract_type")) $("wb-f-contract_type").addEventListener("change", contractVisibility);
    contractVisibility(); refreshDynamicFields();
  }
  function contractVisibility() {
    if (state.tool !== "specialist") return;
    const schema = $("wb-f-contract_type").value === "schema";
    $("wb-f-json_schema").closest("label").hidden = !schema;
    $("wb-f-allowed_outputs").closest("label").hidden = schema;
  }
  function chooseArea(area) {
    state.area = area;
    for (const button of $("wb-tabs").children) button.setAttribute("aria-pressed", button.dataset.area === area ? "true" : "false");
    $("wb-tool").replaceChildren();
    for (const [id, tool] of Object.entries(tools)) if (tool.area === area) { const option = el("option", tool.title); option.value = id; $("wb-tool").append(option); }
    state.tool = $("wb-tool").value; renderForm();
  }
  function readFields() {
    const result = {};
    for (const f of tools[state.tool].fields()) {
      const input = $("wb-f-" + f.key);
      if (input.closest("label").hidden) continue;
      if (f.type === "checkbox") result[f.key] = input.checked;
      else if (input.multiple) result[f.key] = [...input.selectedOptions].map((option) => option.value);
      else if (f.type === "number") { if (input.value !== "") { const n = Number(input.value); if (!Number.isFinite(n)) throw new Error(f.label + " must be a finite number."); result[f.key] = n; } }
      else if (f.type === "json") { try { result[f.key] = input.value.trim() ? JSON.parse(input.value) : undefined; } catch { throw new Error(f.label + " is not valid JSON."); } }
      else if (f.type === "rows") { const value = input.value.trim(); try { result[f.key] = !value ? [] : value.startsWith("[") ? JSON.parse(value) : value.split("\n").filter((line) => line.trim()).map((line) => JSON.parse(line)); } catch { throw new Error("Dataset rows must be a JSON array or one JSON object per line."); } }
      else if (f.type === "lines") result[f.key] = input.value.split("\n").map((line) => line.trim()).filter(Boolean);
      else result[f.key] = input.value.trim();
    }
    return result;
  }
  function requestFromForm() {
    if ($("wb-use-json").checked) { const request = JSON.parse($("wb-json").value); if (!request || Array.isArray(request) || typeof request !== "object") throw new Error("The request must be a JSON object."); return request; }
    return tools[state.tool].build(readFields());
  }
  function remember(result, title, id = "local-" + Date.now()) { state.reports.set(id, { result, title }); if (state.reports.size > 80) state.reports.delete(state.reports.keys().next().value); return id; }
  const fmt = (value, digits = 3) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
  function metrics(entries) { const grid = el("div", null, "wb-metrics"); for (const [label, value] of entries) if (value !== undefined && value !== null) { const card = el("div", null, "wb-metric"); card.append(el("span", label), el("strong", value)); grid.append(card); } return grid; }
  function showResult(result, title, request = null) {
    state.result = result; $("wb-export").disabled = false;
    const target = $("wb-result"); target.replaceChildren();
    const status = result.status || (result.passed === false ? "Gate failed" : result.passed === true ? "Gate passed" : result.abstained ? "Abstained" : "Result ready");
    const bad = ["failed", "error", "timeout", "cancelled", "resource_limit", "budget_exceeded", "cost_unavailable", "Gate failed"].includes(status);
    const tag = el("div", status, "wb-status " + (bad ? "bad" : active(status) ? "" : "good"));
    target.append(tag, el("h3", title));
    if (result.error || result.reason || result.stop_reason) target.append(el("p", result.error || result.reason || result.stop_reason));
    const evaluation = result.evaluation || (result.quality !== undefined && result.results ? result : null);
    if (evaluation) target.append(metrics([["Task quality", fmt(evaluation.quality * 100, 1) + "%"], ["Tasks", evaluation.tasks], ["Schema failures", evaluation.schema_failures], ["Weighted error cost", fmt(evaluation.total_error_cost)]]));
    if (result.soup_revision && result.versions) {
      target.append(el("p", "Installed Soup source matches reviewed revision " + result.soup_revision.slice(0, 12) + "."));
      const table = el("table", null, "wb-results-table"); const header = el("tr"); ["Installed package", "Version"].forEach((label) => header.append(el("th", label))); table.append(header);
      for (const [name, version] of Object.entries(result.versions)) { const row = el("tr"); row.append(el("td", name), el("td", version)); table.append(row); } target.append(table);
      target.append(el("p", "Source verification loads no model. Run a bounded training or reload job to verify the selected backend."));
    }
    if (Array.isArray(result.candidates)) {
      target.append(metrics([["Planned configurations", result.candidates.length], ["Maximum requests", result.max_requests_including_warmup], ["Time budget · seconds", result.max_duration_s], ["Excluded backends", result.excluded_candidates?.length || 0]]));
      target.append(el("p", "Configuration and input checks passed. Backend availability is shown where the plan probed it. No inference has run."));
    }
    if (Array.isArray(result.trials)) {
      target.append(metrics([["Configurations tested", result.trials.length], ["Eligible configurations", result.trials.filter((trial) => trial.eligible).length], ["Elapsed seconds", fmt(result.elapsed_s)], ["Search stages", result.stages?.length]]));
      target.append(el("p", result.selected_candidate ? "Selected from measured candidates: " + result.selected_candidate : "No candidate passed all selection gates."));
      const table = el("table", null, "wb-results-table"); const header = el("tr"); ["Configuration", "Quality", "Latency", "RAM", "Gate"].forEach((label) => header.append(el("th", label))); table.append(header);
      for (const trial of result.trials.slice(0, 64)) { const row = el("tr"); const metric = trial.metrics || {}; const rss = trial.memory?.rss_peak_gib ?? metric.rss_peak_gib; [trial.name, metric.quality === undefined ? "—" : fmt(metric.quality * 100, 1) + "%", metric.latency_p50_s == null ? "—" : fmt(metric.latency_p50_s * 1000, 1) + " ms", rss == null ? "—" : fmt(rss, 2) + " GiB", trial.eligible ? "Pass" : trial.failure_class || "Below limits"].forEach((value) => row.append(el("td", value))); table.append(row); } target.append(table);
      if (result.boundary) target.append(el("p", "Observed search boundary: " + (result.boundary.failure_class || result.boundary.reason || result.boundary.candidate)));
      if (result.heldout) target.append(el("p", "Final held-out gate: " + (result.heldout.passed ? "passed" : "did not pass") + ". Held-out results were not used for selection."));
    }
    if (result.input_chars !== undefined && result.output_chars !== undefined) {
      target.append(metrics([["Original characters", result.input_chars], ["Selected characters", result.output_chars], ["Selected excerpts", result.selected?.length], ["Required-fact gate", result.required_fact_gate ? "Passed" : "Missing facts"]]));
      if (result.required_facts_missing?.length) target.append(el("p", "Missing from selected context: " + result.required_facts_missing.join("; ")));
      target.append(el("pre", result.text || "No relevant context selected."));
    }
    if (result.measurements?.quality !== undefined) target.append(metrics([["Component quality", fmt(result.measurements.quality * 100, 1) + "%"], ["Median latency", fmt(result.measurements.latency_p50_s * 1000, 2) + " ms"], ["Provider calls", result.provider_calls], ["Provider cost", "$" + fmt(result.provider_cost_usd, 6)]]));
    if (Array.isArray(result.iterations)) target.append(metrics([["Repair iterations", result.iterations.length], ["Tests passed", result.passed ? "Yes" : "No"], ["Provider USD", "$" + fmt(result.cost_usd, 6)], ["Project changed", result.applied_to_project ? "Yes" : "No"]]));
    if (result.splits) {
      target.append(metrics([["Unique tasks", result.unique_rows], ["Duplicates removed", result.duplicates?.length], ["Training", result.splits.train?.rows], ["Held-out", result.splits.held_out?.rows]]));
      const table = el("table", null, "wb-results-table"); const head = el("tr"); ["Split", "Rows", "Groups"].forEach((x) => head.append(el("th", x))); table.append(head);
      for (const [key, split] of Object.entries(result.splits)) { const tr = el("tr"); [key.replace("_", " "), split.rows, split.groups].forEach((x) => tr.append(el("td", x))); table.append(tr); } target.append(table);
      target.append(el("p", "Use tuning tasks to choose settings. Preserve held-out tasks for the final decision."));
    }
    if (result.accepted_rows !== undefined) target.append(metrics([["Accepted labels", result.accepted_rows], ["Rejected labels", result.rejected_rows], ["Requests", result.requests_attempted], ["Accounted USD", "$" + fmt(result.accounted_cost_usd, 6)]]));
    if (result.process) target.append(metrics([["Elapsed seconds", fmt(result.process.elapsed_s)], ["Peak process RAM · GiB", fmt(result.process.resources?.rss_peak_gib)], ["GPU memory · GiB", fmt(result.process.resources?.gpu_peak_gib)], ["Exit code", result.process.exit_code]]));
    if (result.worker?.adapter_verification) target.append(el("p", `${result.worker.adapter_verification.tensor_count} adapter tensors attached and checked in a fresh process.`));
    if (result.worker?.predictions?.length) { const detail = el("details"); detail.append(el("summary", "Generated answers"), el("pre", pretty(result.worker.predictions))); target.append(detail); }
    if (result.output !== undefined && result.output !== null) target.append(el("pre", result.output));
    if (result.abstained) target.append(el("p", "No answer passed the contract. No unsupported result was accepted."));
    if (result.delta) target.append(metrics([["Quality change", fmt(result.delta.quality)], ["Error-cost change", fmt(result.delta.error_cost)], ["Schema failure change", result.delta.schema_failures], ["Regressed tasks", result.regressed_task_ids?.length]]));
    const warnings = [...(result.warnings || []), ...(result.failures || [])];
    if (warnings.length) { const list = el("ul", null, "reason-list"); warnings.forEach((warning) => list.append(el("li", typeof warning === "string" ? warning : pretty(warning)))); target.append(list); }
    if (result.proposal_id && result.proposal) {
      const review = el("button", "Review final changes in Develop ↗", "button primary"); review.type = "button";
      review.addEventListener("click", async () => { const name = request?.project; if (name) await lab.loadProject(name); lab.displayCode(result); lab.showView("develop"); });
      target.append(el("p", "The staged test result is recorded. Your project has not been changed."), review);
    }
    if (result.manifest_path || result.result_path || result.path) target.append(el("p", "Saved evidence: " + (result.manifest_path || result.result_path || result.path), "field-help"));
    const raw = el("details"); raw.append(el("summary", "Inspect complete evidence JSON"), el("pre", pretty(result))); target.append(raw);
    if (request) { const details = el("details"); details.append(el("summary", "Request used for this run"), el("pre", pretty(request))); target.append(details); }
  }
  async function execute(operation, request, title) {
    const button = $("wb-submit"); button.disabled = true;
    try {
      const result = await lab.api(operation.startsWith("/") ? operation : "/api/workbench/" + operation, request);
      if (result.id && !result.identity_sha256 && (result.status === "running" || result.status === "queued" || Object.keys(result).every((key) => ["id", "status", "kind"].includes(key)))) {
        state.pending.set(result.id, { title, request, operation }); showResult({ status: "running", id: result.id }, title, request); lab.toast(title + " started. Progress and cancellation are available below.");
      } else { remember(result, title); showResult(result, title, request); }
      await lab.refresh();
    } catch (error) { showResult({ status: "failed", error: error.message }, title, request); lab.toast(error.message, true); }
    finally { button.disabled = false; }
  }
  function refreshDynamicFields() {
    for (const f of tools[state.tool].fields()) {
      const input = $("wb-f-" + f.key); if (!input) continue;
      if (["model", "models", "adapter", "managed", "report"].includes(f.type)) setOptions(input, f);
      if (f.autofill === "executable" && !state.dirty.has(f.key)) input.value = $("experiment-executable").value;
      if (!input.value && !state.dirty.has(f.key)) {
        if (f.autofill === "model") input.value = $("experiment-model").value || lab.state.gguf_models?.[0] || "";
        if (f.autofill === "tasks") input.value = $("experiment-dataset-path").value || $("experiment-dataset").value || "";
      }
    }
    for (const [id, kind] of [["wb-task-paths", "tasks"], ["wb-metadata-paths", "rows_path"], ["wb-sft-paths", "supervised"]]) {
      let list = $(id); if (!list) { list = el("datalist"); list.id = id; document.body.append(list); }
      list.replaceChildren(); const values = new Set();
      for (const dataset of lab.state.workbench?.datasets || []) for (const [split, data] of Object.entries(dataset.splits || {})) { const path = data[kind] || (kind === "rows_path" ? data.tasks : null); if (path && !values.has(path)) { const option = el("option", `${dataset.name} · ${split}`); option.value = path; list.append(option); values.add(path); } }
      if (kind !== "supervised") for (const task of lab.state.tasks || []) if (task.path && !values.has(task.path)) { const option = el("option", task.label || task.path); option.value = task.path; list.append(option); }
    }
  }
  const jobTitles = { "explore-plan": "Search plan", "explore-run": "Experiment search", model: "Managed model" };
  const jobDetails = (job) => state.pending.get(job.id) || state.completed.get(job.id);
  const jobTitle = (job) => jobDetails(job)?.title || tools[job.kind]?.title || jobTitles[job.kind] || job.kind;
  function renderState() {
    const operations = new Set([...(lab.state.workbench?.operations || []), "explore-plan", "explore-run", "model", ...Object.entries(tools).filter(([, tool]) => !tool.local).map(([key, tool]) => tool.operation || key)]);
    const jobs = (lab.state.jobs || []).filter((job) => operations.has(job.kind) || state.pending.has(job.id));
    for (const job of jobs) {
      const pending = state.pending.get(job.id);
      if (!active(job.status) && job.result) remember(job.result, `${jobTitle(job)} · ${job.id.slice(0, 6)}`, "job-" + job.id);
      if (pending && !active(job.status) && !state.seen.has(job.id)) {
        state.seen.add(job.id); state.completed.set(job.id, pending); state.pending.delete(job.id);
        if (state.completed.size > 80) state.completed.delete(state.completed.keys().next().value);
        showResult(job.error ? { status: job.status, error: job.error } : job.result || { status: job.status }, pending.title, pending.request);
        lab.toast(pending.title + (job.error ? " failed. Inspect the result." : " finished. Inspect its evidence."), Boolean(job.error));
      }
    }
    const direct = [...state.reports].filter(([id]) => !id.startsWith("job-")).slice(-6).reverse();
    $("wb-job-count").textContent = jobs.length + direct.length;
    const list = $("wb-jobs"); list.replaceChildren();
    for (const job of jobs.slice().reverse().slice(0, 12)) {
      const row = el("div", null, "wb-job"); const main = el("button", null, "wb-job-main"); main.type = "button";
      main.append(el("strong", jobTitle(job)), el("span", `${job.status} · ${job.id.slice(0, 8)}${job.progress?.elapsed_s ? " · " + fmt(job.progress.elapsed_s, 1) + "s" : ""}`));
      main.addEventListener("click", () => showResult(job.result || { status: job.status, progress: job.progress, error: job.error }, jobTitle(job), jobDetails(job)?.request)); row.append(main);
      if (active(job.status)) { const cancel = el("button", "Cancel", "text-button"); cancel.type = "button"; cancel.addEventListener("click", async () => { cancel.disabled = true; try { await lab.api("/api/cancel", { id: job.id }); lab.toast("Cancellation requested."); await lab.refresh(); } catch (error) { lab.toast(error.message, true); } }); row.append(cancel); }
      list.append(row);
    }
    for (const [id, report] of direct) { const row = el("div", null, "wb-job"); const button = el("button", null, "wb-job-main"); button.type = "button"; button.append(el("strong", report.title), el("span", "Saved result · this browser session")); button.addEventListener("click", () => showResult(report.result, report.title)); row.append(button); list.append(row); }
    if (!jobs.length && !direct.length) list.append(el("p", "Your workbench jobs will appear here, with their results and cancellation controls.", "small-empty"));
    const inventory = $("wb-inventory"); inventory.replaceChildren();
    const data = lab.state.workbench || {}; const cache = data.intelligence?.cache || {};
    for (const [label, value] of [["Domain datasets", data.datasets?.length || 0], ["Registered adapters", data.adapters?.length || 0], ["Managed models", lab.state.lifecycle?.registered?.length || 0], ["Active model leases", lab.state.lifecycle?.active_leases || 0], ["Exact cache entries", cache.entries || 0], ["Cache size · MiB", fmt((cache.bytes || 0) / 1024 ** 2)], ["Calibrated task / model groups", data.intelligence?.measurements?.length || 0]]) { const row = el("div"); row.append(el("span", label), el("strong", value)); inventory.append(row); }
    if (state.area === "runtime") for (const model of lab.state.lifecycle?.registered || []) { const card = el("div", null, "wb-lifecycle-card"); card.append(el("strong", model.id), el("p", `${model.context} context · RAM estimate ${fmt(model.requirements?.ram_gib)} GiB`)); const button = el("button", "Unload", "button secondary"); button.type = "button"; button.addEventListener("click", () => execute("/api/lifecycle/unload", { id: model.id }, "Unload " + model.id)); card.append(button); inventory.append(card); }
    refreshDynamicFields();
  }

  for (const [id, glyph, label] of areas) { const button = el("button"); button.type = "button"; button.dataset.area = id; button.setAttribute("aria-pressed", "false"); const icon = el("span", glyph); icon.setAttribute("aria-hidden", "true"); button.append(icon, document.createTextNode(label)); button.addEventListener("click", () => { chooseArea(id); renderState(); }); $("wb-tabs").append(button); }
  $("wb-tool").addEventListener("change", () => { state.tool = $("wb-tool").value; renderForm(); });
  $("wb-help").addEventListener("click", () => { $("wb-guide").hidden = !$("wb-guide").hidden; $("wb-help").setAttribute("aria-expanded", String(!$("wb-guide").hidden)); });
  $("wb-use-json").addEventListener("change", () => { $("wb-form").noValidate = $("wb-use-json").checked; });
  $("wb-build-json").addEventListener("click", () => { try { $("wb-json").value = pretty(tools[state.tool].build(readFields())); } catch (error) { lab.toast(error.message, true); } });
  $("wb-preview").addEventListener("click", async () => {
    try { const request = requestFromForm(); $("wb-json").value = pretty(request); if (tools[state.tool].area === "explore") { if (!$("wb-form").reportValidity()) return; await execute("explore-plan", request, "Search plan"); } else { const detail = el("details"); detail.open = true; detail.append(el("summary", "Request preview · nothing has run"), el("pre", pretty(request))); $("wb-result").replaceChildren(detail); } } catch (error) { lab.toast(error.message, true); }
  });
  $("wb-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try { const request = requestFromForm(); const tool = tools[state.tool]; if (tool.local) { state.preferences = request; showResult({ status: "Preferences saved", ...request }, tool.title); lab.toast("Optimisation preferences apply to subsequent Chat and Develop requests."); return; } await execute(tool.operation || state.tool, request, tool.title); } catch (error) { lab.toast(error.message, true); }
  });
  $("wb-export").addEventListener("click", () => { if (state.result) lab.downloadData("workbench-evidence.json", state.result); });
  window.addEventListener("llm-state", renderState);
  lab.getOptimisation = () => ({ ...state.preferences });
  chooseArea("explore"); renderState();
})();
