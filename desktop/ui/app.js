"use strict";
const invoke = window.__TAURI__.core.invoke;
const byId = (id) => document.getElementById(id);
const form = byId("launch");
const status = byId("status");
const button = byId("start");
let ready = false;
let bundledAvailable = false;
let bundledError = "";
let busy = false;

function message(value, error = false) {
  status.textContent = value;
  status.classList.toggle("error", error);
}
function runtimeSelection() {
  const custom = byId("runtime-mode").value === "custom";
  byId("custom-runtime").hidden = !custom;
  byId("python").disabled = !custom;
  byId("python").required = custom;
  byId("runtime-title").textContent = custom ? "Your custom Python environment" : bundledAvailable ? "Bundled runtime is ready" : "Bundled runtime unavailable";
  byId("runtime-note").textContent = custom ? "Use an existing development environment with LLM-Optimise installed." : bundledAvailable ? "Included with this app. No Python installation or setup required." : bundledError;
  byId("runtime-badge").textContent = custom ? "CUSTOM" : bundledAvailable ? "INCLUDED" : "SETUP";
  byId("runtime-dot").classList.toggle("unavailable", !custom && !bundledAvailable);
  button.disabled = !ready || busy || !custom && !bundledAvailable;
}
invoke("defaults").then((value) => {
  byId("python").value = value.interpreter || "";
  byId("workspace").value = value.workspace || "";
  byId("runtime-mode").value = value.runtime_mode || "bundled";
  bundledAvailable = value.bundled_available;
  bundledError = value.bundled_error || "Install the full desktop package, or select a custom Python environment in Advanced.";
  ready = true;
  if (value.runtime_mode === "custom" || !bundledAvailable) byId("advanced").open = true;
  runtimeSelection();
  if (!bundledAvailable && value.runtime_mode !== "custom") message(bundledError, true);
}).catch((error) => message(String(error), true));
byId("runtime-mode").addEventListener("change", runtimeSelection);
form.addEventListener("submit", async (event) => {
  event.preventDefault(); busy = true; runtimeSelection(); message("Opening your laboratory…");
  try {
    const result = await invoke("start_service", { interpreter: byId("python").value.trim(), workspace: byId("workspace").value.trim(), runtimeMode: byId("runtime-mode").value });
    byId("service-url").value = result.url;
    byId("service-address").hidden = false;
    message("Your laboratory is open. Use the tray menu to return here or quit and release owned models.");
  } catch (error) { message(String(error), true); }
  finally { busy = false; runtimeSelection(); }
});
byId("stop").addEventListener("click", async () => {
  byId("stop").disabled = true;
  try { await invoke("stop_service"); byId("service-url").value = ""; byId("service-address").hidden = true; message("Local service stopped and owned models released."); }
  catch (error) { message(String(error), true); }
  finally { byId("stop").disabled = false; }
});
