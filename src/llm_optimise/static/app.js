"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const token = document.querySelector('meta[name="csrf-token"]').content;
  const app = { state: { models: [], results: [], jobs: [] }, view: "lab", preset: "baseline", result: null, resultId: "", selectedTrial: null, sort: { key: "quality", direction: -1 }, policy: { placement: "mixed", objective: "balanced" }, model: "", project: "", history: [], proposal: null, training: null, pending: new Map(), delivered: new Set(), polling: false, timer: null, modelsDirty: false, resultsSignature: "", modelsSignature: "", projectsSignature: "", defaultsSet: false, editingModel: null };
  const activeStatus = (status) => ["queued", "running", "pending", "starting", "cancelling"].includes(status);
  const safeArray = (value) => Array.isArray(value) ? value : [];
  const number = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)) ? Number(value) : null;
  const format = (value, digits = 2) => number(value) === null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
  const value = (id) => $(id).value.trim();
  const numeric = (id) => Number($(id).value);
  const optionalNumeric = (id) => value(id) === "" ? null : numeric(id);
  const element = (tag, attrs = {}, text = null) => {
    const node = document.createElement(tag);
    for (const [key, val] of Object.entries(attrs)) {
      if (key === "class") node.className = val;
      else if (key === "hidden") node.hidden = Boolean(val);
      else node.setAttribute(key, val);
    }
    if (text !== null) node.textContent = String(text);
    return node;
  };
  const option = (label, val) => element("option", { value: val }, label);
  const listen = (id, event, fn) => $(id).addEventListener(event, fn);
  const pretty = (data) => JSON.stringify(data, null, 2);
  const errorText = (error) => error instanceof Error ? error.message : String(error);

  function toast(message, isError = false) {
    const box = element("div", { class: `toast${isError ? " error" : ""}` });
    box.append(element("span", {}, message));
    const close = element("button", { type: "button", "aria-label": "Dismiss notification" }, "×");
    close.addEventListener("click", () => box.remove());
    box.append(close);
    $("toasts").append(box);
    setTimeout(() => box.remove(), isError ? 18000 : 6500);
  }

  async function api(path, payload) {
    const options = { method: payload === undefined ? "GET" : "POST", headers: { Accept: "application/json" }, cache: "no-store" };
    if (payload !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.headers["X-LLM-Token"] = token;
      options.body = JSON.stringify(payload);
    }
    const response = await fetch(path, options);
    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json") ? await response.json() : { error: await response.text() };
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }

  async function busy(button, label, action) {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = label;
    try { return await action(); }
    catch (error) { toast(errorText(error), true); return null; }
    finally {
      button.disabled = false; button.textContent = original;
      if (["run-experiment", "start-server", "stop-server"].includes(button.id) && app.state.hardware) renderHardware();
    }
  }

  function showView(view, updateHash = true) {
    if (!$("view-" + view)) view = "lab";
    app.view = view;
    $$(".view").forEach((section) => { section.hidden = section.id !== "view-" + view; section.classList.toggle("active", !section.hidden); });
    $$(".nav-link").forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle("active", active);
      if (active) { button.setAttribute("aria-current", "page"); $("view-title").textContent = button.textContent.trim().replace(/^[^A-Za-z]+/, ""); }
      else button.removeAttribute("aria-current");
    });
    if (updateHash) history.replaceState(null, "", "#" + view);
  }

  function statusBadge(status, eligible) {
    let kind = "";
    if (eligible === true || ["complete", "completed", "success", "running-server", "ready"].includes(status)) kind = "good";
    if (eligible === false || ["failed", "error", "cancelled"].includes(status)) kind = "bad";
    if (activeStatus(status)) kind = "running";
    return element("span", { class: "status-badge " + kind }, status || "Unknown");
  }

  function runtimePaths(hardware) {
    const runtimes = hardware.runtimes || {};
    if (Array.isArray(runtimes)) return runtimes.map((r) => typeof r === "string" ? r : r.path || r.executable || r.name || "");
    return Object.entries(runtimes).map(([name, info]) => {
      if (!info) return "";
      if (typeof info === "string") return info;
      return typeof info === "object" ? info.path || info.executable || name : name;
    }).filter(Boolean);
  }

  function renderHardware() {
    const h = app.state.hardware || {};
    $("hw-cpu").textContent = h.cpu || "Processor unavailable";
    $("hw-platform").textContent = [h.architecture, h.logical_cores ? h.logical_cores + " logical cores" : h.os].filter(Boolean).join(" · ") || "Hardware information unavailable";
    $("hw-memory").textContent = number(h.ram_total_gib) === null ? "Memory unavailable" : `${format(h.ram_available_gib, 1)} / ${format(h.ram_total_gib, 1)} GiB`;
    $("hw-memory-detail").textContent = h.unified_memory ? "Available / total · unified memory" : "Available / total system memory";
    const gpus = safeArray(h.gpus);
    $("hw-gpu").textContent = gpus.length ? gpus.map((gpu) => typeof gpu === "string" ? gpu : gpu.name || gpu.model || "GPU").join(", ") : h.unified_memory ? "Apple unified memory" : "CPU execution";
    $("hw-gpu-detail").textContent = gpus.length ? gpus.map((gpu) => typeof gpu === "object" ? [gpu.backend || gpu.vendor, number(gpu.memory_gib) !== null ? format(gpu.memory_gib, 1) + " GiB" : ""].filter(Boolean).join(" · ") : "").filter(Boolean).join(", ") || "GPU detected · support depends on runtime" : h.unified_memory ? "Metal support depends on runtime build" : "No GPU reported";
    const paths = runtimePaths(h);
    const llama = paths.find((path) => /llama[-_]server/.test(path));
    $("hw-runtime").textContent = llama ? "llama-server detected" : paths.length ? paths.map((path) => path.split("/").pop()).slice(0, 2).join(" · ") : "No runtime detected";
    $("hw-runtime").title = paths.join("\n");
    const server = app.state.local_server || {};
    const serverStatus = server.status || "stopped";
    $("hw-server").textContent = ["running", "ready"].includes(serverStatus) ? "Managed model is running" : "Managed server: " + serverStatus;
    $("server-status").replaceWith(Object.assign(statusBadge(serverStatus, ["running", "ready"].includes(serverStatus) ? true : undefined), { id: "server-status" }));
    $("server-detail").textContent = [server.endpoint, server.model, server.error].filter(Boolean).join(" · ");
    $("start-server").disabled = ["running", "ready", "starting"].includes(serverStatus);
    $("stop-server").disabled = !["running", "ready", "starting"].includes(serverStatus);
    const benchmarking = safeArray(app.state.jobs).some((job) => ["experiment", "benchmark"].includes(job.kind) && activeStatus(job.status));
    $("run-experiment").disabled = ["running", "ready", "starting"].includes(serverStatus) || benchmarking;
    $("experiment-run-note").textContent = ["running", "ready", "starting"].includes(serverStatus) ? "Stop the managed local server in Model router before benchmarking." : benchmarking ? "An experiment is in progress. Follow its status in Activity." : "Runs use your machine. Keep other heavy workloads idle for comparable measurements.";
    if (!app.defaultsSet && h.cpu) {
      if (llama) { $("experiment-executable").value = llama; $("server-executable").value = llama; }
      if (h.logical_cores) { const threads = Math.min(4, h.logical_cores); $("experiment-threads").value = threads; $("server-threads").value = threads; }
      app.defaultsSet = true;
    }
    $("workspace-path").textContent = app.state.workspace || "Local workspace";
  }

  function renderDatasets() {
    const dataset = $("experiment-dataset");
    const selected = dataset.value;
    const tasks = safeArray(app.state.tasks);
    const signature = pretty(tasks);
    if (dataset.dataset.signature !== signature) {
      dataset.replaceChildren(option("Choose a dataset", ""), ...tasks.map((task) => option(task.label || task.path, task.path)));
      if (tasks.some((task) => task.path === selected)) dataset.value = selected;
      else if (!selected && tasks.length) dataset.value = tasks[0].path;
      dataset.dataset.signature = signature;
    }
    const discoveredModels = safeArray(app.state.gguf_models);
    $("gguf-paths").replaceChildren(...discoveredModels.map((path) => option(path, path)));
    const modelInput = $("experiment-model");
    if (!modelInput.value && !modelInput.dataset.userEdited && discoveredModels.length) {
      modelInput.value = discoveredModels.find((path) => /qwen.*coder|coder.*qwen/i.test(path.split("/").pop())) || discoveredModels[0];
    }
    const projects = safeArray(app.state.projects);
    const projectSignature = pretty(projects);
    if (app.projectsSignature !== projectSignature) {
      $("project-names").replaceChildren(...projects.map((project) => option(typeof project === "string" ? project : project.name, typeof project === "string" ? project : project.name)));
      app.projectsSignature = projectSignature;
    }
  }

  function buildPolicyControls() {
    const limitFields = [
      ["max_cost_usd", "max-cost", "Max estimated cost · USD", "0.000001", null],
      ["max_latency_ms", "max-latency", "Max latency · ms", "1", null],
      ["min_quality", "min-quality", "Minimum quality", "0.01", "1"],
      ["max_ram_gib", "max-ram", "Max model RAM · GiB", "0.1", null],
      ["max_gpu_gib", "max-gpu", "Max model GPU · GiB", "0.1", null],
    ];
    $$("[data-policy-host]").forEach((host) => {
      const name = host.dataset.policyHost;
      const modelLabel = element("label", {}, "Model");
      const modelSelect = element("select", { id: name + "-model", "data-shared": "model" });
      modelSelect.append(option("Automatic · use routing policy", ""));
      modelLabel.append(modelSelect);
      const grid = element("div", { class: "two-fields" });
      for (const [key, label, choices] of [["placement", "Where to run", [["mixed", "Local + cloud"], ["local", "Local only"], ["cloud", "Cloud only"]]], ["objective", "Prioritise", [["balanced", "Balanced"], ["cost", "Lowest cost"], ["performance", "Performance"]]]]) {
        const field = element("label", {}, label);
        const select = element("select", { id: name + "-" + key, "data-shared": key });
        choices.forEach(([val, text]) => select.append(option(text, val)));
        field.append(select); grid.append(field);
      }
      const limits = element("details", { class: "advanced policy-limits" });
      limits.append(element("summary", {}, "Routing limits (optional)"), element("p", { class: "field-help" }, "Shared across Chat, Develop, and routing previews. Leave a field blank for no limit."));
      const prefix = name === "router" ? "route" : name;
      const form = { router: "route-form", chat: "chat-form", develop: "code-form" }[name];
      for (const [key, suffix, label, step, maximum] of limitFields) {
        const field = element("label", {}, label);
        const input = element("input", { id: prefix + "-" + suffix, type: "number", min: "0", step, placeholder: "No limit", "data-shared": key, form });
        if (maximum !== null) input.max = maximum;
        input.addEventListener("invalid", () => { limits.open = true; });
        field.append(input); limits.append(field);
      }
      host.append(modelLabel, grid, limits);
    });
    $$("[data-shared]").forEach((control) => control.addEventListener(control.type === "number" ? "input" : "change", () => {
      const key = control.dataset.shared;
      if (key === "model") app.model = control.value;
      else if (control.type === "number") {
        if (control.value.trim() === "") delete app.policy[key];
        else app.policy[key] = Number(control.value);
      }
      else app.policy[key] = control.value;
      $$("[data-shared='" + key + "']").forEach((peer) => { if (peer !== control) peer.value = control.value; });
      const count = limitFields.filter(([limit]) => app.policy[limit] !== undefined).length;
      $$(".policy-limits>summary").forEach((summary) => { summary.textContent = count ? `Routing limits · ${count} active` : "Routing limits (optional)"; });
    }));
  }

  function renderModels() {
    const models = safeArray(app.state.models);
    const signature = pretty(models);
    if (signature === app.modelsSignature) return;
    app.modelsSignature = signature;
    $("model-count").textContent = models.length;
    const list = $("model-list"); list.replaceChildren();
    if (!models.length) list.append(element("p", { class: "small-empty" }, "Add a local or cloud model, or start the managed local server, to begin."));
    models.forEach((model) => {
      const card = element("article", { class: "model-card" });
      const heading = element("div", { class: "model-card-heading" });
      const description = element("div");
      const title = element("h3", {}, model.id); title.append(statusBadge(model.location));
      description.append(title, element("p", {}, model.model), element("p", {}, model.base_url));
      const actions = element("div", { class: "button-row" });
      const edit = element("button", { class: "text-button", type: "button", "aria-label": "Edit " + model.id }, "Edit");
      edit.addEventListener("click", () => openModel(model));
      const remove = element("button", { class: "text-button", type: "button", "aria-label": "Remove " + model.id }, "Remove");
      remove.addEventListener("click", () => busy(remove, "Removing…", async () => {
        await api("/api/models", { models: safeArray(app.state.models).filter((candidate) => candidate.id !== model.id) });
        toast("Removed " + model.id + " from the registry."); await refreshState();
      }));
      actions.append(edit, remove); heading.append(description, actions);
      const facts = element("div", { class: "model-facts" });
      const cost = (cost) => number(cost) === null ? "Unknown" : "$" + format(cost, 6);
      [["Input / 1M tokens", cost(model.input_cost_per_million)], ["Output / 1M tokens", cost(model.output_cost_per_million)], ["Latency", number(model.latency_ms) === null ? "Unknown" : format(model.latency_ms) + " ms"], ["Quality", number(model.quality) === null ? "Unknown" : format(model.quality)], ["Context", format(model.context_window, 0) + " tokens"], ["Capabilities", safeArray(model.capabilities).join(", ") || "None"]].forEach(([label, val]) => { const fact = element("div"); fact.append(element("span", {}, label), element("strong", {}, val)); facts.append(fact); });
      card.append(heading, facts); list.append(card);
    });
    if (!app.modelsDirty) $("models-json").value = pretty(models);
    $$("[data-shared='model']").forEach((select) => {
      select.replaceChildren(option("Automatic · use routing policy", ""), ...models.map((model) => option(`${model.id} · ${model.location}`, model.id)));
      if (!models.some((model) => model.id === app.model)) app.model = "";
      select.value = app.model;
    });
  }

  function candidateConfig() {
    const candidate = { name: "baseline", model: value("experiment-model"), executable: value("experiment-executable"), threads: numeric("experiment-threads"), context: numeric("experiment-context"), gpu_layers: numeric("experiment-gpu"), cache_type_k: value("experiment-cache"), cache_type_v: value("experiment-cache"), flash_attention: $("experiment-flash").checked ? "on" : "off", constrain_json: $("experiment-constrain-json").checked, batch_size: 256, ubatch_size: 64 };
    if (app.preset === "compact") {
      candidate.name = "small-sweep";
      candidate.sweep = { threads: [...new Set([Math.max(1, Math.floor(candidate.threads / 2)), candidate.threads])], context: [...new Set([Math.min(1024, candidate.context), candidate.context])] };
    }
    const limits = { min_quality: numeric("experiment-quality"), min_available_gib: 1 };
    if (optionalNumeric("experiment-ram") !== null) limits.max_rss_gib = numeric("experiment-ram");
    if (optionalNumeric("experiment-vram") !== null) limits.max_gpu_gib = numeric("experiment-vram");
    return { name: value("experiment-name"), dataset: value("experiment-dataset-path") || value("experiment-dataset"), repeats: numeric("experiment-repeats"), warmup: 1, max_tokens: numeric("experiment-tokens"), max_trials: 32, limits, candidates: [candidate] };
  }

  function experimentConfig() { return $("experiment-use-json").checked ? JSON.parse(value("experiment-json")) : candidateConfig(); }

  function configurationCount(config) {
    return safeArray(config.candidates).reduce((total, candidate) => total + Object.values(candidate.sweep || {}).reduce((count, options) => count * (Array.isArray(options) ? options.length : 1), 1), 0);
  }

  function updateCandidateCount() {
    try { $("candidate-count").textContent = configurationCount(experimentConfig()); }
    catch { $("candidate-count").textContent = "Invalid JSON"; }
  }

  async function runExperiment(config) {
    if (!config.dataset) throw new Error("Choose an evaluation dataset or enter its path.");
    if (!Array.isArray(config.candidates) || !config.candidates.length) throw new Error("Add at least one model configuration.");
    if (configurationCount(config) > (config.max_trials || 32)) throw new Error("The sweep exceeds max_trials. Reduce the search size or explicitly increase the limit in JSON.");
    const job = await api("/api/experiment", { config });
    app.pending.set(job.id, "experiment");
    toast("Experiment queued. Live progress appears in Activity.");
    await refreshState();
    return job;
  }

  function loadExperiment(config) {
    $("experiment-json").value = pretty(config);
    $("experiment-use-json").checked = true;
    $("experiment-form").noValidate = true;
    $("experiment-use-json").closest("details").open = true;
    const fields = { name: "experiment-name", dataset: "experiment-dataset-path", repeats: "experiment-repeats", max_tokens: "experiment-tokens" };
    Object.entries(fields).forEach(([key, id]) => { if (config[key] !== undefined) $(id).value = config[key]; });
    const candidate = safeArray(config.candidates)[0];
    if (candidate) { $("experiment-model").value = candidate.model || ""; $("experiment-model").dataset.userEdited = "true"; $("experiment-executable").value = candidate.executable || "llama-server"; }
    updateCandidateCount(); showView("lab"); toast("Suggestion loaded. Review the JSON configuration before running.");
  }

  function metric(trial, key) { return number((trial.metrics || {})[key] ?? (trial.memory || {})[key] ?? trial[key]); }
  function trials() { return safeArray(app.result && app.result.trials); }
  function resultId(result) { return String(result.id || result.run_id || result.name || ""); }

  async function selectResult(id) {
    if (!id) return;
    try {
      const full = await api("/api/results/" + encodeURIComponent(id));
      app.result = full; app.resultId = id; app.selectedTrial = null;
      $("result-select").value = id; renderResults();
    } catch (error) { toast(errorText(error), true); }
  }

  function syncResults() {
    const results = safeArray(app.state.results);
    const signature = pretty(results);
    if (signature === app.resultsSignature) return;
    app.resultsSignature = signature;
    const select = $("result-select");
    select.replaceChildren(...(results.length ? results.map((result) => option(result.name || resultId(result), resultId(result))) : [option("No experiments yet", "")]));
    if (app.resultId && results.some((result) => resultId(result) === app.resultId)) {
      select.value = app.resultId;
      const current = results.find((result) => resultId(result) === app.resultId);
      if (Array.isArray(current.trials)) { app.result = current; renderResults(); }
    } else if (results.length) {
      const latest = results[0];
      app.resultId = resultId(latest); select.value = app.resultId;
      if (Array.isArray(latest.trials)) { app.result = latest; renderResults(); }
      else selectResult(app.resultId);
    }
  }

  function eligibleTrial(trial) { return trial.eligible === true; }
  function frontierTrials(items) {
    const names = new Set(safeArray(app.result && app.result.frontier).map((entry) => typeof entry === "string" ? entry : entry.name));
    return items.filter((trial) => names.has(trial.name));
  }

  function svgElement(tag, attrs = {}, text = null) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, val]) => node.setAttribute(key, val));
    if (text !== null) node.textContent = String(text);
    return node;
  }

  function renderChart() {
    const chart = $("frontier-chart");
    const key = value("chart-metric");
    const allTrials = trials();
    const shown = allTrials.filter((trial) => !$("chart-eligible-only").checked || eligibleTrial(trial));
    const measured = shown.filter((trial) => metric(trial, key) !== null && metric(trial, "quality") !== null);
    const boundary = frontierTrials(measured);
    const maxX = (Math.max(0, ...measured.map((trial) => metric(trial, key))) || 1) * 1.12;
    const xTickDigits = maxX < 1 ? Math.min(8, Math.max(2, Math.ceil(-Math.log10(maxX / 4)))) : 1;
    const maxY = Math.max(1, ...measured.map((trial) => metric(trial, "quality")));
    const left = 62, right = 722, top = 32, bottom = 293;
    const x = (val) => left + val / maxX * (right - left);
    const y = (val) => bottom - val / maxY * (bottom - top);
    const metricLabel = { latency_p50_s: "MEDIAN LATENCY · SECONDS", rss_peak_gib: "PEAK RAM · GiB", decode_tokens_s: "DECODE · TOKENS / SECOND" }[key];
    chart.replaceChildren(svgElement("title", { id: "chart-title" }, "Measured quality versus " + metricLabel.toLowerCase()), svgElement("desc", { id: "chart-description" }, measured.length ? `${measured.length} measured configurations. Teal dots mark the experiment's saved frontier across all evaluated objectives, including metrics outside these axes. Crosses indicate rejected configurations. Select a point to inspect its metrics.` : "No measured configurations are available for these axes. Run an experiment to plot measured results."));
    for (let i = 0; i <= 4; i++) {
      const gridY = top + (bottom - top) * i / 4;
      chart.append(svgElement("line", { x1: left, x2: right, y1: gridY, y2: gridY, class: "chart-grid" }), svgElement("text", { x: left - 13, y: gridY + 3, "text-anchor": "end", class: "chart-axis-text" }, format(maxY * (1 - i / 4), 2)));
      const gridX = left + (right - left) * i / 4;
      chart.append(svgElement("line", { x1: gridX, x2: gridX, y1: top, y2: bottom, class: "chart-grid" }), svgElement("text", { x: gridX, y: bottom + 20, "text-anchor": "middle", class: "chart-axis-text" }, measured.length ? (maxX * i / 4).toLocaleString(undefined, { minimumFractionDigits: maxX < 1 ? xTickDigits : 0, maximumFractionDigits: xTickDigits }) : "—"));
    }
    chart.append(svgElement("text", { x: left, y: 15, class: "chart-axis-label" }, "QUALITY ↑"), svgElement("text", { x: (left + right) / 2, y: 337, class: "chart-axis-label", "text-anchor": "middle" }, metricLabel));
    if (boundary.length > 1) {
      const ordered = [...boundary].sort((a, b) => metric(a, key) - metric(b, key));
      chart.append(svgElement("polyline", { class: "chart-frontier", points: ordered.map((trial) => `${x(metric(trial, key))},${y(metric(trial, "quality"))}`).join(" ") }));
    }
    const resultLimits = app.result && (app.result.limits || (app.result.experiment && app.result.experiment.limits));
    const limit = number(resultLimits && resultLimits.min_quality);
    if (limit !== null) {
      chart.append(svgElement("line", { x1: left, x2: right, y1: y(limit), y2: y(limit), stroke: "#c38b79", "stroke-width": 1, "stroke-dasharray": "6 5", opacity: .65 }), svgElement("text", { x: right - 3, y: y(limit) - 7, "text-anchor": "end", class: "chart-axis-text" }, "Minimum quality " + format(limit)));
    }
    measured.forEach((trial) => {
      const cx = x(metric(trial, key)), cy = y(metric(trial, "quality"));
      const selected = app.selectedTrial === trial.name;
      const point = svgElement("g", { class: "chart-point", "data-trial": trial.name, tabindex: "0", role: "button", "aria-label": `${trial.name}: quality ${format(metric(trial, "quality"))}, ${key} ${format(metric(trial, key))}, ${eligibleTrial(trial) ? "eligible" : "rejected"}`, transform: `translate(${cx} ${cy})` });
      point.append(svgElement("title", {}, `${trial.name}\nQuality: ${format(metric(trial, "quality"))}\n${key}: ${format(metric(trial, key))}\n${safeArray(trial.rejection_reasons).join("; ")}`));
      point.append(svgElement("circle", { r: 13, fill: "transparent" }));
      if (selected) point.append(svgElement("circle", { r: 12, fill: "none", stroke: "#3465cf", "stroke-width": 1.5 }));
      if (eligibleTrial(trial)) point.append(svgElement("circle", { r: boundary.includes(trial) ? 6 : 4.5, fill: boundary.includes(trial) ? "#007e80" : "#a5bbd9", stroke: "white", "stroke-width": 2 }));
      else point.append(svgElement("path", { d: "M -4 -4 L 4 4 M -4 4 L 4 -4", stroke: "#b75047", "stroke-width": 2, "stroke-linecap": "round" }));
      if (selected) point.append(svgElement("text", { y: -18, x: cx > 590 ? -8 : 8, "text-anchor": cx > 590 ? "end" : "start", class: "chart-point-label" }, trial.name));
      const select = () => { app.selectedTrial = trial.name; renderResults(); };
      point.addEventListener("click", select);
      point.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); select(); $$(".chart-point").find((node) => node.dataset.trial === trial.name)?.focus(); } });
      chart.append(point);
    });
    $("chart-empty").hidden = measured.length > 0;
    if (allTrials.length && !measured.length) {
      $("chart-empty").querySelector("h3").textContent = "No measured points for this view.";
      $("chart-empty").querySelector("p").textContent = "Try another metric or include rejected trials. Failures remain visible in the table below.";
    }
    $("chart-direction").textContent = key === "decode_tokens_s" ? "More throughput →" : key === "rss_peak_gib" ? "Less memory ←" : "Less latency ←";
  }

  function renderResults() {
    const allTrials = trials();
    $("trial-count").textContent = allTrials.length;
    $("export-results").disabled = !app.resultId;
    const body = $("results-body"); body.replaceChildren();
    if (!allTrials.length) body.append(element("tr", {}, null));
    if (!allTrials.length) body.firstChild.append(element("td", { colspan: 6, class: "table-empty" }, "Every tested configuration will appear here, including failures."));
    const sorted = [...allTrials].sort((a, b) => {
      if (app.sort.key === "name") return String(a.name).localeCompare(String(b.name)) * app.sort.direction;
      const av = metric(a, app.sort.key), bv = metric(b, app.sort.key);
      if (av === null) return bv === null ? 0 : 1;
      if (bv === null) return -1;
      return (av - bv) * app.sort.direction;
    });
    const frontier = frontierTrials(allTrials);
    sorted.forEach((trial) => {
      const row = element("tr", { "data-trial": trial.name, tabindex: "0", "aria-label": "Inspect " + trial.name, class: app.selectedTrial === trial.name ? "selected" : "" });
      row.append(element("td", {}, trial.name), ...["quality", "latency_p50_s", "decode_tokens_s", "rss_peak_gib"].map((key) => element("td", { class: "number" }, format(metric(trial, key), key === "quality" ? 3 : 2))));
      const status = element("td");
      status.append(statusBadge(eligibleTrial(trial) ? frontier.includes(trial) ? "Frontier" : "Eligible" : trial.status === "failed" ? "Failed" : "Rejected", eligibleTrial(trial)));
      row.append(status);
      const choose = () => { app.selectedTrial = trial.name; renderResults(); };
      row.addEventListener("click", choose); row.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); choose(); $$("#results-body tr").find((node) => node.dataset.trial === trial.name)?.focus(); } });
      body.append(row);
    });
    $$("th button[data-sort]").forEach((button) => { const th = button.closest("th"); if (button.dataset.sort === app.sort.key) th.setAttribute("aria-sort", app.sort.direction === 1 ? "ascending" : "descending"); else th.removeAttribute("aria-sort"); });
    const detail = $("trial-detail"); detail.replaceChildren();
    const selected = allTrials.find((trial) => trial.name === app.selectedTrial);
    detail.hidden = !selected;
    if (selected) {
      detail.append(element("h3", {}, selected.name));
      const reasons = safeArray(selected.rejection_reasons);
      if (reasons.length) detail.append(element("p", {}, "Excluded: " + reasons.join("; ")));
      if (selected.error) detail.append(element("p", {}, selected.error));
      const grid = element("div", { class: "trial-detail-grid" });
      [["Latency p95", "latency_p95_s", " s"], ["Time to first token", "ttft_p50_s", " s"], ["End-to-end throughput", "end_to_end_tokens_s", " tokens/s"], ["Peak GPU memory", "gpu_peak_gib", " GiB"]].forEach(([label, key, unit]) => { const item = element("div"); item.append(element("span", {}, label), element("strong", {}, metric(selected, key) === null ? "Unknown" : format(metric(selected, key)) + unit)); grid.append(item); });
      detail.append(grid);
    }
    renderChart();
  }

  function renderActivity() {
    const jobs = safeArray(app.state.jobs);
    const active = jobs.filter((job) => activeStatus(job.status));
    $("activity-count").textContent = active.length ? active.length + " active " + (active.length === 1 ? "job" : "jobs") : "No active jobs";
    const list = $("activity-list"); list.replaceChildren();
    if (!jobs.length) { list.append(element("p", { class: "small-empty" }, "Ready when you are. Start an experiment or work with a model.")); return; }
    [...jobs].sort((a, b) => Number(activeStatus(b.status)) - Number(activeStatus(a.status))).slice(0, 5).forEach((job) => {
      const item = element("div", { class: "job-item" });
      const info = element("div", { class: "job-info" });
      const title = element("div", { class: "job-title" });
      title.append(element("span", {}, job.name || job.kind || "Job"), statusBadge(job.status));
      info.append(title);
      if (job.error) info.append(element("p", { class: "job-error" }, job.error));
      if (activeStatus(job.status)) {
        const progress = element("progress", { class: "job-progress", max: "100", "aria-label": "Progress for " + (job.name || job.kind) });
        if (typeof job.progress === "number") progress.value = job.progress <= 1 ? job.progress * 100 : job.progress;
        else if (job.progress && typeof job.progress === "object") {
          const current = number(job.progress.requests ?? job.progress.completed ?? job.progress.current);
          const total = number(job.progress.request_total ?? job.progress.total);
          if (current !== null && total) progress.value = current / total * 100;
          if (job.progress.message) info.append(element("p", { class: "field-help" }, job.progress.message));
          else if (job.progress.candidate) info.append(element("p", { class: "field-help" }, `${job.progress.candidate}${current !== null && total ? " · " + current + " / " + total : ""}`));
        }
        info.append(progress);
        const cancel = element("button", { class: "text-button", type: "button", "aria-label": "Cancel " + (job.name || job.kind) }, "Cancel");
        cancel.addEventListener("click", () => busy(cancel, "Cancelling…", async () => { await api("/api/cancel", { id: job.id }); toast("Cancellation requested."); await refreshState(); }));
        item.append(info, cancel);
      } else item.append(info);
      list.append(item);
    });
  }

  function routingChip(route, completion = {}) {
    const pieces = [route && route.model_id ? route.model_id : "Configured model"];
    if (number(completion.latency_ms) !== null) pieces.push(format(completion.latency_ms) + " ms");
    if (number(completion.accounted_cost_usd) !== null) pieces.push("$" + format(completion.accounted_cost_usd, 6));
    else if (route && number(route.estimated_cost_usd) !== null) pieces.push("estimated $" + format(route.estimated_cost_usd, 6));
    return element("div", { class: "route-chip" }, pieces.join(" · "));
  }

  function displayCode(result) {
    const output = $("code-output"); output.replaceChildren();
    output.append(routingChip(result.route, result.completion));
    const proposal = result.proposal;
    app.proposal = result.proposal_id ? { id: result.proposal_id, proposal, project: app.project } : null;
    if (proposal) {
      output.append(element("p", {}, proposal.summary || "Review the proposed changes below."));
      safeArray(proposal.files).forEach((file) => {
        const section = element("section", { class: "file-proposal" });
        section.append(element("h3", {}, file.path));
        const diff = element("pre", {}, file.diff || "No unified diff supplied.");
        section.append(diff);
        const details = element("details", { class: "advanced" }); details.append(element("summary", {}, "View proposed file"), element("pre", {}, file.content || ""));
        section.append(details); output.append(section);
      });
    } else output.append(element("p", {}, (result.completion && result.completion.text) || result.reply || "The model returned no proposed file changes."));
    $("code-status").textContent = proposal && proposal.applied ? "Changes applied" : "Ready for review";
    $("code-apply-bar").hidden = !app.proposal || !proposal || proposal.applied || !safeArray(proposal.files).length;
    $("apply-code").disabled = false;
  }

  function displayContainer(result) {
    const output = $("container-output"); output.replaceChildren();
    const successful = result.exit_code === 0 && !result.timed_out && !result.cancelled && !result.output_limit_exceeded;
    const label = result.timed_out ? "Timed out" : result.cancelled ? "Cancelled" : result.output_limit_exceeded ? "Output limit reached" : successful ? "Command passed" : "Command failed";
    $("container-status").textContent = label;
    output.append(statusBadge(label, successful), element("p", {}, `Exit code: ${result.exit_code ?? "unknown"}${result.image ? " · Image: " + result.image : ""}`));
    if (result.evidence) output.append(element("p", {}, typeof result.evidence === "string" ? result.evidence : pretty(result.evidence)));
    if (result.artifact_dir) output.append(element("p", {}, "Evidence saved: " + result.artifact_dir));
    if (result.command) output.append(element("h3", {}, "Executed command"), element("pre", {}, Array.isArray(result.command) ? result.command.join(" ") : result.command));
    output.append(element("h3", {}, "Standard output"), element("pre", {}, result.stdout || "No standard output."));
    if (result.stderr) output.append(element("h3", {}, "Standard error"), element("pre", {}, result.stderr));
    if (result.limits) { const details = element("details", { class: "advanced" }); details.append(element("summary", {}, "Applied resource limits"), element("pre", {}, pretty(result.limits))); output.append(details); }
  }

  function appendMessage(role, content, route, completion, experiment) {
    $("chat-welcome")?.remove();
    const message = element("article", { class: "chat-message " + role });
    message.append(element("div", { class: "message-role" }, role === "user" ? "You" : "Lab assistant"), element("div", { class: "message-body" }, content));
    if (route) message.append(routingChip(route, completion));
    if (experiment) {
      const suggestion = element("div", { class: "experiment-suggestion" });
      suggestion.append(element("h3", {}, experiment.name || "Suggested experiment"), element("p", {}, "Review this configuration in the lab, or explicitly run the suggested experiment."));
      const details = element("details", { class: "advanced" }); details.append(element("summary", {}, "Inspect suggested configuration"), element("pre", {}, pretty(experiment))); suggestion.append(details);
      const buttons = element("div", { class: "button-row" });
      const load = element("button", { type: "button", class: "button secondary" }, "Load in lab"); load.addEventListener("click", () => loadExperiment(experiment));
      const run = element("button", { type: "button", class: "button primary" }, "Run suggested experiment");
      run.addEventListener("click", () => busy(run, "Queuing…", async () => { await runExperiment(experiment); showView("lab"); }));
      buttons.append(load, run); suggestion.append(buttons); message.append(suggestion);
    }
    $("chat-transcript").append(message); $("chat-transcript").scrollTop = $("chat-transcript").scrollHeight;
    return message;
  }

  function completeJobs() {
    for (const job of safeArray(app.state.jobs)) {
      if (!app.pending.has(job.id) || app.delivered.has(job.id) || activeStatus(job.status)) continue;
      app.delivered.add(job.id);
      const kind = app.pending.get(job.id); app.pending.delete(job.id);
      if (kind === "code") { $("generate-code").disabled = false; $("generate-code").textContent = "Generate changes ↗"; }
      if (kind === "chat") { $("send-chat").disabled = false; $("send-chat").textContent = "Send message ↑"; $("chat-pending")?.remove(); }
      if (kind === "container") { $("run-container").disabled = false; $("run-container").textContent = "Run in container"; }
      if (["failed", "error", "cancelled"].includes(job.status) || job.error) {
        const message = job.error || "The job was " + job.status + ".";
        toast(message, true);
        if (kind === "code") $("code-status").textContent = "Request " + job.status;
        if (kind === "chat") appendMessage("assistant", "Request " + job.status + ": " + message);
        if (kind === "container") {
          $("container-status").textContent = "Run " + job.status;
          $("container-output").replaceChildren(element("p", {}, message), element("p", { class: "field-help" }, "Check that Docker is running and the selected runtime image is available. Enable image pull only if you want to download it, then retry."));
        }
        continue;
      }
      const result = job.result || {};
      if (kind === "code") { displayCode(result); toast("Proposed changes are ready to review."); }
      if (kind === "container") { displayContainer(result); toast(result.exit_code === 0 && !result.timed_out && !result.cancelled && !result.output_limit_exceeded ? "Container command passed. Inspect its captured evidence." : "Container command did not pass. Inspect the output.", result.exit_code !== 0 || result.timed_out); }
      if (kind === "chat") {
        const reply = result.reply || (result.completion && result.completion.text) || "The model returned an empty response.";
        app.history.push({ role: "assistant", content: reply });
        appendMessage("assistant", reply, result.route, result.completion, result.experiment);
      }
      if (kind === "experiment") {
        toast("Experiment complete. Measurements are ready.");
        if (Array.isArray(result.trials)) { app.result = result; app.resultId = result.id || result.run_id || job.id; $("result-select").value = app.resultId; renderResults(); }
        else if (result.id || result.run_id) selectResult(result.id || result.run_id);
        else if (safeArray(app.state.results).length) selectResult(resultId(app.state.results[0]));
      }
    }
  }

  async function refreshState() {
    if (app.polling) return;
    app.polling = true;
    clearTimeout(app.timer);
    try {
      app.state = await api("/api/state");
      $("connection-dot").className = "status-dot connected";
      $("connection-label").textContent = "Workspace connected";
      renderHardware(); renderDatasets(); renderModels(); syncResults(); renderActivity(); completeJobs();
    } catch (error) {
      $("connection-dot").className = "status-dot error";
      $("connection-label").textContent = "Connection unavailable";
      $("connection-label").title = errorText(error);
      if (!app.state.hardware) $("hw-cpu").textContent = "Unable to connect";
    } finally {
      app.polling = false;
      const active = app.pending.size || safeArray(app.state.jobs).some((job) => activeStatus(job.status));
      app.timer = setTimeout(refreshState, active ? 1000 : 5000);
    }
  }

  function openModel(model = null) {
    $("model-form").reset();
    app.editingModel = model && model.id;
    $("model-dialog-heading").textContent = model ? "Edit model" : "Add model";
    if (model) {
      const fields = { id: "model-id", model: "model-name", provider: "model-provider", location: "model-location", base_url: "model-base-url", context_window: "model-context", max_output_tokens: "model-output", api_key_env: "model-key-env", input_cost_per_million: "model-input-cost", output_cost_per_million: "model-output-cost", latency_ms: "model-latency", quality: "model-quality", ram_gib: "model-ram", gpu_gib: "model-gpu" };
      Object.entries(fields).forEach(([key, id]) => { $(id).value = model[key] ?? ""; });
      $("model-chat").checked = safeArray(model.capabilities).includes("chat");
      $("model-code").checked = safeArray(model.capabilities).includes("code");
      $("model-schema").checked = Boolean(model.supports_json_schema);
    }
    $("model-dialog").showModal();
    $("model-id").focus();
  }

  async function loadProject(name, create = false) {
    if (!name) throw new Error("Enter a project name first.");
    const project = create ? await api("/api/project", { name }) : await api("/api/project?name=" + encodeURIComponent(name));
    app.project = project.name || name;
    $("project-name").value = app.project;
    const list = $("project-files");
    const checked = new Set($$("input:checked", list).map((input) => input.value));
    list.replaceChildren();
    if (!safeArray(project.files).length) list.append(element("p", { class: "small-empty" }, "This project is empty. Describe what you want to create."));
    safeArray(project.files).forEach((file) => {
      const path = typeof file === "string" ? file : file.path;
      const label = element("label", { class: "checkbox-label" });
      const checkbox = element("input", { type: "checkbox", value: path }); checkbox.checked = checked.has(path);
      label.append(checkbox, document.createTextNode(path)); list.append(label);
    });
    return project;
  }

  async function sendChat(message) {
    if (!message.trim()) return;
    for (const control of $$("[data-policy-host='chat'] input")) {
      if (!control.checkValidity()) { control.closest("details").open = true; control.reportValidity(); return; }
    }
    const button = $("send-chat");
    if (button.disabled) return;
    button.disabled = true; button.textContent = "Thinking…";
    const history = app.history.slice();
    appendMessage("user", message);
    app.history.push({ role: "user", content: message });
    $("chat-message").value = "";
    const pending = appendMessage("assistant", "Choosing a model and preparing a response…"); pending.id = "chat-pending";
    try {
      const job = await api("/api/chat", { message, history, policy: { ...app.policy }, ...(app.model ? { selected_model: app.model } : {}), max_tokens: 1024 });
      app.pending.set(job.id, "chat"); await refreshState();
    } catch (error) {
      pending.remove(); toast(errorText(error), true);
      appendMessage("assistant", "The request failed: " + errorText(error));
      button.disabled = false; button.textContent = "Send message ↑";
    }
  }

  function renderRoute(route) {
    const output = $("route-result"); output.replaceChildren();
    output.append(element("h3", {}, route.model_id ? "Selected: " + route.model_id : "No eligible model"));
    const cost = number(route.estimated_cost_usd) === null ? "Cost unknown" : "Estimated cost $" + format(route.estimated_cost_usd, 6);
    const latency = number(route.estimated_latency_ms) === null ? "latency unknown" : "estimated latency " + format(route.estimated_latency_ms) + " ms";
    output.append(element("p", {}, cost + " · " + latency));
    if (safeArray(route.reasons).length) { const reasons = element("ul", { class: "reason-list" }); route.reasons.forEach((reason) => reasons.append(element("li", {}, typeof reason === "string" ? reason : pretty(reason)))); output.append(reasons); }
    const alternatives = Array.isArray(route.alternatives) ? route.alternatives : Object.entries(route.alternatives || {}).map(([model_id, reasons]) => ({ model_id, reasons }));
    alternatives.forEach((alternative) => {
      const row = element("div", { class: "route-alternative" });
      if (typeof alternative === "string") row.textContent = alternative;
      else {
        row.append(element("strong", {}, alternative.model_id || alternative.id || "Alternative"));
        const reasons = alternative.reasons || alternative.rejection_reasons || alternative.reason || alternative.status;
        row.append(document.createTextNode(Array.isArray(reasons) ? reasons.join("; ") : reasons ? String(reasons) : pretty(alternative)));
      }
      output.append(row);
    });
    Object.entries(route.rejections || {}).forEach(([modelId, reasons]) => {
      const row = element("div", { class: "route-alternative" });
      row.append(element("strong", {}, modelId + " · rejected"), document.createTextNode(Array.isArray(reasons) ? reasons.join("; ") : String(reasons)));
      output.append(row);
    });
  }

  function downloadData(name, data) {
    const blob = new Blob([typeof data === "string" ? data : pretty(data)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = element("a", { href: url, download: name });
    document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function bindEvents() {
    $$(".nav-link").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
    window.addEventListener("hashchange", () => showView(location.hash.slice(1), false));
    listen("refresh-state", "click", () => busy($("refresh-state"), "↻", async () => { await refreshState(); if ($("connection-dot").classList.contains("error")) throw new Error("Unable to connect to the workspace server."); toast("Hardware and workspace refreshed."); }));
    $$("[data-preset]").forEach((button) => button.addEventListener("click", () => {
      app.preset = button.dataset.preset;
      $$("[data-preset]").forEach((peer) => { const selected = peer === button; peer.classList.toggle("selected", selected); peer.setAttribute("aria-pressed", selected); });
      $("preset-description").textContent = { baseline: "One configuration. Establish a measured starting point.", compact: "Compare thread counts and context sizes in a small, bounded sweep.", custom: "Adjust runtime settings below, or use JSON to define your own search." }[app.preset];
      updateCandidateCount();
    }));
    listen("experiment-form", "input", updateCandidateCount);
    listen("experiment-model", "input", () => { $("experiment-model").dataset.userEdited = "true"; });
    listen("experiment-form", "change", () => { $("experiment-form").noValidate = $("experiment-use-json").checked; updateCandidateCount(); });
    listen("experiment-to-json", "click", () => { $("experiment-json").value = pretty(candidateConfig()); updateCandidateCount(); toast("Form copied to JSON. Enable “Use JSON configuration” to run it."); });
    listen("experiment-form", "submit", (event) => { event.preventDefault(); busy($("run-experiment"), "Queuing experiment…", async () => { await runExperiment(experimentConfig()); }); });
    listen("chart-metric", "change", renderResults);
    listen("chart-eligible-only", "change", renderChart);
    listen("result-select", "change", () => selectResult(value("result-select")));
    listen("load-latest", "click", () => { const latest = safeArray(app.state.results)[0]; if (latest) selectResult(resultId(latest)); else toast("No measurements yet. Configure and run your first experiment."); });
    listen("export-results", "click", () => { if (app.resultId) { const anchor = element("a", { href: "/api/export/" + encodeURIComponent(app.resultId), download: app.resultId + ".json" }); document.body.append(anchor); anchor.click(); anchor.remove(); } });
    $$("[data-sort]").forEach((button) => button.addEventListener("click", () => { app.sort.direction = app.sort.key === button.dataset.sort ? -app.sort.direction : button.dataset.sort === "name" ? 1 : -1; app.sort.key = button.dataset.sort; renderResults(); }));
    listen("project-form", "submit", (event) => { event.preventDefault(); busy(event.submitter, "Opening…", async () => { await loadProject(value("project-name"), true); toast("Opened " + app.project + "."); await refreshState(); }); });
    listen("project-refresh", "click", () => busy($("project-refresh"), "Refreshing…", () => loadProject(value("project-name"))));
    listen("code-form", "submit", async (event) => {
      event.preventDefault();
      const button = $("generate-code"); button.disabled = true; button.textContent = "Generating…";
      try {
        if (!app.project || value("project-name") !== app.project) throw new Error("Create or open the project before generating changes.");
        $("code-apply-bar").hidden = true; app.proposal = null; $("code-status").textContent = "Generating proposal";
        const job = await api("/api/code", { project: app.project, prompt: value("code-prompt"), context_files: $$("input:checked", $("project-files")).map((input) => input.value), policy: { ...app.policy }, ...(app.model ? { selected_model: app.model } : {}), max_tokens: 2048 });
        app.pending.set(job.id, "code"); toast("Generating changes. Review the proposal when it is ready."); await refreshState();
      } catch (error) { button.disabled = false; button.textContent = "Generate changes ↗"; $("code-status").textContent = "Request failed"; toast(errorText(error), true); }
    });
    listen("apply-code", "click", () => busy($("apply-code"), "Applying…", async () => {
      if (!app.proposal) throw new Error("No proposal is ready to apply.");
      const applied = await api("/api/apply", { proposal_id: app.proposal.id });
      $("code-apply-bar").hidden = true; $("code-status").textContent = "Changes applied";
      toast(`Applied ${safeArray(applied.files).length} file changes. Generated code has not been tested.`);
      await loadProject(app.proposal.project); app.proposal = null;
    }));
    listen("container-form", "submit", async (event) => {
      event.preventDefault();
      const button = $("run-container"); button.disabled = true; button.textContent = "Running in container…";
      try {
        if (!app.project || value("project-name") !== app.project) throw new Error("Create or open the project before running it in Docker.");
        const job = await api("/api/container", { project: app.project, runtime: value("container-runtime"), action: value("container-action"), memory_mib: numeric("container-memory"), cpus: numeric("container-cpus"), timeout_s: numeric("container-timeout"), network: $("container-network").checked, pull: $("container-pull").checked });
        app.pending.set(job.id, "container"); $("container-status").textContent = "Running";
        toast("Container job started. Resource limits and output will be recorded."); await refreshState();
      } catch (error) { button.disabled = false; button.textContent = "Run in container"; $("container-status").textContent = "Could not start"; toast(errorText(error), true); }
    });
    listen("chat-form", "submit", (event) => { event.preventDefault(); sendChat(value("chat-message")); });
    listen("chat-message", "keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); sendChat(value("chat-message")); } });
    $$("[data-starter]").forEach((button) => button.addEventListener("click", () => sendChat(button.dataset.starter)));
    listen("clear-chat", "click", () => {
      if ([...app.pending.values()].includes("chat")) { toast("Wait for the current reply before clearing the conversation."); return; }
      app.history = []; $("chat-transcript").replaceChildren(element("p", { class: "small-empty" }, "Conversation cleared. Ask about your hardware or results to start again.")); toast("Conversation cleared.");
    });
    listen("add-model", "click", () => openModel());
    listen("close-model-dialog", "click", () => $("model-dialog").close());
    listen("cancel-model-dialog", "click", () => $("model-dialog").close());
    listen("models-json", "input", () => { app.modelsDirty = true; });
    listen("save-models-json", "click", () => busy($("save-models-json"), "Saving…", async () => {
      const models = JSON.parse(value("models-json")); if (!Array.isArray(models)) throw new Error("The model registry must be a JSON array.");
      await api("/api/models", { models }); app.modelsDirty = false; app.modelsSignature = ""; await refreshState(); toast("Model registry saved.");
    }));
    listen("model-form", "submit", (event) => {
      event.preventDefault(); busy(event.submitter, "Saving…", async () => {
        const model = { id: value("model-id"), model: value("model-name"), provider: value("model-provider"), location: value("model-location"), base_url: value("model-base-url"), context_window: numeric("model-context"), max_output_tokens: numeric("model-output"), api_key_env: value("model-key-env") || null, supports_json_schema: $("model-schema").checked, input_cost_per_million: optionalNumeric("model-input-cost"), output_cost_per_million: optionalNumeric("model-output-cost"), latency_ms: optionalNumeric("model-latency"), quality: optionalNumeric("model-quality"), ram_gib: optionalNumeric("model-ram"), gpu_gib: optionalNumeric("model-gpu"), capabilities: [$("model-chat").checked ? "chat" : null, $("model-code").checked ? "code" : null].filter(Boolean) };
        if (!model.capabilities.length) throw new Error("Choose at least one model capability.");
        const models = safeArray(app.state.models).filter((candidate) => candidate.id !== app.editingModel);
        if (models.some((candidate) => candidate.id === model.id)) throw new Error("This registry ID already exists. Use a unique ID.");
        models.push(model); await api("/api/models", { models }); app.modelsDirty = false; app.modelsSignature = ""; $("model-dialog").close(); await refreshState(); toast("Model saved.");
      });
    });
    listen("model-provider", "change", () => {
      if (value("model-provider") === "anthropic") { $("model-base-url").value = "https://api.anthropic.com/v1"; $("model-key-env").value = "ANTHROPIC_API_KEY"; $("model-location").value = "cloud"; }
    });
    listen("route-form", "submit", (event) => { event.preventDefault(); busy(event.submitter, "Comparing…", async () => {
      const policy = { ...app.policy };
      const route = await api("/api/route", { policy, input_tokens: numeric("route-input"), output_tokens: numeric("route-output"), capability: value("route-capability"), ...(app.model ? { selected_model: app.model } : {}) }); renderRoute(route);
    }); });
    listen("local-server-form", "submit", (event) => { event.preventDefault(); busy($("start-server"), "Starting…", async () => {
      await api("/api/local/start", { model: value("server-model"), executable: value("server-executable"), threads: numeric("server-threads"), context: numeric("server-context"), gpu_layers: numeric("server-gpu") }); toast("Local server start requested."); await refreshState();
    }); });
    listen("stop-server", "click", () => busy($("stop-server"), "Stopping…", async () => { await api("/api/local/stop", {}); toast("Local server stopped."); await refreshState(); }));
    listen("training-form", "submit", (event) => { event.preventDefault(); busy(event.submitter, "Preparing…", async () => {
      const recipe = await api("/api/training", { engine: value("training-engine"), model: value("training-model"), data: value("training-data"), output: value("training-output"), max_length: numeric("training-length"), rank: numeric("training-rank") });
      app.training = recipe;
      const output = $("training-result"); output.replaceChildren();
      output.append(statusBadge("Recipe prepared", true), element("p", {}, "Training has not started. Review the configuration and run the command in your terminal when ready."));
      if (recipe.path) output.append(element("p", {}, "Configuration saved: " + recipe.path));
      const command = safeArray(recipe.command).map((part) => /^[A-Za-z0-9_/.=:@+-]+$/.test(String(part)) ? String(part) : "'" + String(part).replaceAll("'", "'\\''") + "'").join(" ");
      output.append(element("h3", {}, "CLI command"), element("pre", {}, command), element("h3", {}, "Configuration"), element("pre", {}, pretty(recipe.config)));
      if (safeArray(recipe.notes).length) { const notes = element("ul", { class: "reason-list" }); recipe.notes.forEach((note) => notes.append(element("li", {}, note))); output.append(notes); }
      $("download-recipe").disabled = false; toast("Training recipe prepared. No training process was started.");
    }); });
    listen("download-recipe", "click", () => { if (app.training) downloadData("soup-training-recipe.json", app.training.config); });
    window.addEventListener("unhandledrejection", (event) => { toast(errorText(event.reason), true); });
  }

  buildPolicyControls();
  bindEvents();
  showView(location.hash.slice(1) || "lab", false);
  renderChart();
  updateCandidateCount();
  refreshState();
})();
