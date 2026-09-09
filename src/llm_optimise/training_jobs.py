"""Pinned external Soup jobs and content-addressed adapters; no tensor dependencies in core."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import psutil

from .config import Limits, digest, file_digest, positive
from .datasets import TASK_KEYS, compare_regression, evaluate_predictions, normalise_rows
from .hardware import ResourceMonitor
from .runner import write_json

SOUP_REVISION = "254351e10331e360b5d75e26aabe0a4e591e16a4"
# SHA256 over sorted relative Python paths + NUL + each file's SHA256 digest.
# Independently matched the reviewed source tarball and its installed wheel (500 files).
SOUP_SOURCE_SHA256 = "f6c71ebe6d8dee66f781964efcac90ee843471f27ec00b0c00e9e711b258c55e"
MAX_LOG_BYTES = 8 * 1024 * 1024

_PROBE = r"""
import hashlib,importlib.metadata as m,json,sys
from pathlib import Path
import soup_cli
root=Path(soup_cli.__file__).parent
h=hashlib.sha256()
paths=sorted(root.rglob('*.py'))
for p in paths:
 h.update(p.relative_to(root).as_posix().encode()+b'\0'+hashlib.sha256(p.read_bytes()).digest())
versions={}
for name in ['soup-cli','mlx','mlx-lm','torch','transformers','peft']:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:pass
print(json.dumps({'python':sys.version,'versions':versions,'soup_source_sha256':h.hexdigest(),'python_files':len(paths)}))
"""

_WORKER = r"""
import gc,importlib.metadata as metadata,json,sys,time,traceback
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text())
result_path=Path(sys.argv[2])
started=time.perf_counter()
result={'status':'failed'}
backend=request['backend']
mx=None
torch=None
try:
 if backend=='mlx':
  import mlx.core as mx
  assert mx.metal.is_available(),'Metal unavailable'
  mx.set_memory_limit(int(request['max_rss_gib']*1024**3))
  mx.set_cache_limit(256*1024**2)
  mx.random.seed(42)
  mx.reset_peak_memory()
 elif backend=='transformers':
  import torch
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
 if request['operation']=='train':
  from soup_cli.cli import run
  sys.argv=['soup','train','--config',request['config_path'],'--yes']
  try:run()
  except SystemExit as exc:
   if exc.code not in (None,0):raise RuntimeError('Soup CLI exited with code '+str(exc.code))
  result={'status':'passed','operation':'train'}
 else:
  rows=request['rows'];predictions=[]
  adapter=request.get('adapter_path')
  if backend=='mlx':
   from mlx_lm import load,generate
   from mlx_lm.sample_utils import make_sampler
   from mlx.utils import tree_flatten
   model,tokenizer=load(request['base_model'],adapter_path=adapter)
   if adapter:
    saved=mx.load(str(Path(adapter)/'adapters.safetensors'))
    loaded=dict(tree_flatten(model.parameters()))
    assert saved and all(k in loaded and bool(mx.array_equal(v,loaded[k]).item()) for k,v in saved.items()),'Adapter tensor attachment mismatch'
    assert all(bool(mx.all(mx.isfinite(v)).item()) for v in saved.values()),'Nonfinite adapter values'
    result['adapter_verification']={'tensor_count':len(saved),'all_saved_tensors_attached_exactly':True,'nonzero_lora_b_tensors':sum(k.endswith('lora_b') and bool(mx.any(v!=0).item()) for k,v in saved.items())}
   for row in rows:
    prompt=tokenizer.apply_chat_template([{'role':'system','content':row['system']},{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
    before=time.perf_counter()
    text=generate(model,tokenizer,prompt=prompt,max_tokens=request['max_tokens'],sampler=make_sampler(temp=0.0),verbose=False)
    predictions.append({'id':row['id'],'output':text,'latency_s':time.perf_counter()-before,'generated_tokens':len(tokenizer.encode(text))})
  else:
   from transformers import AutoModelForCausalLM,AutoTokenizer
   from peft import PeftModel
   kwargs={'trust_remote_code':False,'local_files_only':not request['allow_network']}
   tokenizer=AutoTokenizer.from_pretrained(request['base_model'],**kwargs)
   model=AutoModelForCausalLM.from_pretrained(request['base_model'],**kwargs)
   if torch.cuda.is_available():model=model.to('cuda')
   if adapter:
    from peft.utils.save_and_load import load_peft_weights,get_peft_model_state_dict
    model=PeftModel.from_pretrained(model,adapter)
    saved=load_peft_weights(adapter,device='cpu')
    loaded=get_peft_model_state_dict(model)
    assert saved and all(k in loaded and torch.equal(v.to(loaded[k].dtype),loaded[k].detach().cpu()) for k,v in saved.items()),'Adapter tensor attachment mismatch'
    assert all(torch.isfinite(v).all().item() for v in saved.values()),'Nonfinite adapter values'
    result['adapter_verification']={'tensor_count':len(saved),'all_saved_tensors_attached_exactly':True}
   model.eval()
   for row in rows:
    text=tokenizer.apply_chat_template([{'role':'system','content':row['system']},{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(text,return_tensors='pt').to(model.device)
    before=time.perf_counter()
    with torch.no_grad():tokens=model.generate(**inputs,max_new_tokens=request['max_tokens'],do_sample=False)
    output=tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    predictions.append({'id':row['id'],'output':output,'latency_s':time.perf_counter()-before,'generated_tokens':int(tokens.shape[1]-inputs['input_ids'].shape[1])})
  result.update(status='passed',operation='evaluate',predictions=predictions)
except BaseException as exc:
 result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
 traceback.print_exc()
finally:
 result['elapsed_s']=time.perf_counter()-started
 if mx is not None:result['mlx_peak_memory_bytes']=mx.get_peak_memory()
 if torch is not None and torch.cuda.is_available():
  result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
  result['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
 result_path.write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
sys.exit(0 if result['status']=='passed' else 1)
"""


def _python(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("provide an explicit Python executable from the Soup environment")
    path = shutil.which(value) if not Path(value).is_file() else value
    if not path:
        raise ValueError("Soup Python executable does not exist")
    # Do not resolve a venv Python symlink to its base interpreter: venv identity matters.
    return os.path.abspath(path)


def probe_environment(python):
    python = _python(python)
    reply = subprocess.run(
        [python, "-c", _PROBE], capture_output=True, text=True, timeout=30, check=False
    )
    if reply.returncode or len(reply.stdout) > 65536:
        raise ValueError(
            "could not inspect Soup environment; install the pinned Soup package in that environment"
        )
    try:
        result = json.loads(reply.stdout)
    except ValueError as exc:
        raise ValueError("Soup environment returned an invalid probe result") from exc
    if result.get("soup_source_sha256") != SOUP_SOURCE_SHA256:
        raise ValueError(
            "Soup source differs from the reviewed pinned revision; use the documented isolated environment"
        )
    return {
        **result,
        "python_executable": python,
        "soup_revision": SOUP_REVISION,
        "revision_verification": "installed Python source fingerprint matches reviewed pinned source",
    }


def _fingerprint(path):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"provenance input does not exist: {path}")
    paths = (
        [path]
        if path.is_file()
        else sorted(
            p for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts
        )
    )
    if len(paths) > 20000:
        raise ValueError("provenance directory exceeds 20000 files")
    records = []
    for file in paths:
        records.append(
            {
                "path": file.name if path.is_file() else file.relative_to(path).as_posix(),
                "sha256": file_digest(file),
                "bytes": file.stat().st_size,
            }
        )
    if not records:
        raise ValueError("provenance input contains no files")
    return {"path": str(path), "sha256": digest(records), "files": records}


def _limits(request):
    if request.get("max_rss_gib", 4) is None:
        raise ValueError("an explicit RSS budget is required")
    timeout = request.get("timeout_s", 600)
    positive(timeout, "timeout_s")
    if timeout > 86400:
        raise ValueError("timeout_s exceeds one day")
    limits = Limits(
        max_rss_gib=request.get("max_rss_gib", 4),
        max_gpu_gib=request.get("max_gpu_gib"),
        min_available_gib=request.get("min_available_gib", 1),
    )
    return timeout, limits


def _stop_process(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
        _, alive = psutil.wait_procs([*children, parent], timeout=3)
        for item in alive:
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def run_command(
    command,
    output_dir,
    *,
    timeout_s=600,
    max_rss_gib=4,
    max_gpu_gib=None,
    min_available_gib=1,
    cancel=None,
    progress=None,
    env=None,
):
    """Supervise an explicit argv (never a shell), including stalled-worker cancellation."""
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(x, str) or "\x00" in x for x in command)
    ):
        raise ValueError("command must be a nonempty argument list")
    timeout, limits = _limits(locals())
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path, log_path = out / "process.json", out / "process.log"
    if log_path.exists() or result_path.exists():
        raise ValueError("process output already exists; choose a new output directory")
    started = time.monotonic()
    if cancel is not None and cancel.is_set():
        result = {
            "status": "cancelled",
            "exit_code": None,
            "elapsed_s": 0,
            "command": command,
            "resources": None,
        }
        write_json(result_path, result)
        return result
    if psutil.virtual_memory().available < limits.min_available_gib * 1024**3:
        raise ValueError("insufficient host memory headroom to start job")
    status, reason, monitor = "failed", None, None
    with log_path.open("wb") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=out,
            env=env,
            start_new_session=os.name != "nt",
        )
        try:
            monitor = ResourceMonitor(child.pid, limits).start()
            last_update = 0.0
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if cancel is not None and cancel.is_set():
                    status, reason = "cancelled", "cancel requested"
                elif elapsed > timeout:
                    status, reason = "timeout", "wall-clock timeout"
                elif monitor.violation:
                    status, reason = "resource_limit", monitor.violation
                elif log_path.stat().st_size > MAX_LOG_BYTES:
                    status, reason = "resource_limit", "log exceeded 8 MiB"
                if reason:
                    _stop_process(child)
                    break
                if progress is not None and elapsed - last_update >= 0.5:
                    progress(
                        {"stage": "running", "elapsed_s": elapsed, "resources": monitor.result()}
                    )
                    last_update = elapsed
                time.sleep(0.05)
            if reason is None:
                status = "passed" if child.returncode == 0 else "failed"
                if monitor.violation:
                    status, reason = "resource_limit", monitor.violation
                elif log_path.stat().st_size > MAX_LOG_BYTES:
                    status, reason = "resource_limit", "log exceeded 8 MiB"
        finally:
            if child.poll() is None:
                _stop_process(child)
            resources = monitor.stop() if monitor else None
    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - 16384))
        tail = log.read(16384).decode("utf-8", errors="replace")
    result = {
        "status": status,
        "reason": reason,
        "exit_code": child.returncode,
        "elapsed_s": time.monotonic() - started,
        "command": command,
        "resources": resources,
        "log_path": str(log_path),
        "log_tail": tail,
    }
    write_json(result_path, result)
    if progress:
        progress({"stage": status, "elapsed_s": result["elapsed_s"]})
    return result


def _runtime_request(request, out, *, operation):
    timeout, limits = _limits(request)
    allow_network = request.get("allow_network", False)
    if type(allow_network) is not bool:
        raise ValueError("allow_network must be a boolean")
    runtime = probe_environment(request.get("python"))
    payload = {
        "operation": operation,
        "backend": request["backend"],
        "max_rss_gib": limits.max_rss_gib,
        "allow_network": allow_network,
    }
    if payload["backend"] not in ("mlx", "transformers"):
        raise ValueError("backend must be mlx or transformers")
    env = dict(os.environ)
    env.update(
        SOUP_DB_PATH=str(out / "experiments.db"),
        HF_HOME=str(out / "hub-cache"),
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    if not allow_network:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    return (
        runtime,
        payload,
        env,
        {
            "timeout_s": timeout,
            "max_rss_gib": limits.max_rss_gib,
            "max_gpu_gib": limits.max_gpu_gib,
            "min_available_gib": limits.min_available_gib,
        },
    )


def _execute_worker(request, out, payload, env, limits, *, cancel, progress):
    worker = out / "worker.py"
    worker.write_text(_WORKER, encoding="utf-8")
    input_path, result_path = out / "worker-request.json", out / "worker-result.json"
    write_json(input_path, payload)
    process = run_command(
        [_python(request["python"]), str(worker), str(input_path), str(result_path)],
        out / "execution",
        cancel=cancel,
        progress=progress,
        env=env,
        **limits,
    )
    worker_result = (
        json.loads(result_path.read_text())
        if result_path.is_file() and result_path.stat().st_size < 4 * 1024 * 1024
        else None
    )
    return process, worker_result


def _new_output(path):
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("job output directory must be empty")
    return out


def _training_config(request, out):
    config = json.loads(json.dumps(request.get("config"), allow_nan=False))
    if not isinstance(config, dict) or set(config) - {
        "base",
        "task",
        "backend",
        "data",
        "training",
        "output",
        "experiment_name",
    }:
        raise ValueError("provide a supported Soup configuration")
    if config.get("task") != "sft" or config.get("backend") not in ("mlx", "transformers"):
        raise ValueError("training jobs support Soup SFT with mlx or transformers")
    data, training = config.get("data", {}), config.get("training", {})
    if not isinstance(data, dict) or set(data) - {
        "train",
        "val",
        "format",
        "val_split",
        "max_length",
        "train_on_responses_only",
    }:
        raise ValueError("unsupported data configuration")
    if data.get("format") not in ("alpaca", "chatml"):
        raise ValueError("job dataset format must be alpaca or chatml")
    allowed = {
        "epochs",
        "lr",
        "batch_size",
        "lora",
        "quantization",
        "gradient_checkpointing",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_steps",
        "stream_layers",
        "stream_source",
        "stream_buffers",
        "seed",
        "data_seed",
    }
    if not isinstance(training, dict) or set(training) - allowed:
        raise ValueError("unsupported training configuration")
    base = Path(config.get("base", ""))
    if not str(config.get("base", "")) or not base.exists():
        raise ValueError(
            "training requires a downloaded local base model; model downloads are separate explicit actions"
        )
    config["base"] = str(base.resolve())
    fingerprints = {"base_model": _fingerprint(base)}
    for key in ("train", "val"):
        if key in data:
            path = Path(data[key]).resolve()
            if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("training data must be a local file <=16 MiB")
            data[key] = str(path)
            fingerprints[key] = _fingerprint(path)
    if "train" not in data:
        raise ValueError("training dataset is required")
    records = [
        json.loads(line)
        for line in Path(data["train"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("training dataset is empty")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
        training.setdefault(key, 1)
        positive(training[key], key, integer=True)
    steps = ((len(records) + training["batch_size"] - 1) // training["batch_size"]) * training[
        "epochs"
    ]
    max_steps = request.get("max_steps", 100)
    positive(max_steps, "max_steps", integer=True)
    if steps > max_steps:
        raise ValueError(
            "estimated microsteps exceed max_steps; reduce rows/epochs or explicitly raise the budget"
        )
    config["output"] = str(out / "adapter")
    fingerprints["estimated_microsteps"] = steps
    return config, fingerprints


def run_training(request, output_dir, *, registry_dir=None, cancel=None, progress=None):
    """Run actual Soup CLI training in a pinned explicit Python environment."""
    out = _new_output(output_dir)
    config, provenance = _training_config(request, out)
    request = {**request, "backend": config["backend"]}
    runtime, payload, env, limits = _runtime_request(request, out, operation="train")
    config_path = out / "soup.json"
    write_json(config_path, config)
    payload["config_path"] = str(config_path)
    process, worker = _execute_worker(
        request, out, payload, env, limits, cancel=cancel, progress=progress
    )
    result = {
        "schema_version": 1,
        "status": process["status"],
        "backend": config["backend"],
        "runtime": runtime,
        "provenance": provenance,
        "config": config,
        "config_sha256": digest(config),
        "process": process,
        "worker": worker,
        "adapter_path": str(out / "adapter"),
        "result_path": str(out / "training-result.json"),
    }
    if process["status"] == "passed" and (not worker or worker.get("status") != "passed"):
        result["status"] = "failed"
    write_json(out / "training-result.json", result)
    if result["status"] == "passed" and registry_dir is not None:
        result["adapter"] = register_adapter(
            {
                "path": result["adapter_path"],
                "base_model": config["base"],
                "backend": config["backend"],
                "python": runtime["python_executable"],
                "training_result": result,
            },
            registry_dir,
        )
        write_json(out / "training-result.json", result)
    return result


def register_adapter(request, registry_dir):
    """Snapshot final adapter files; never register mutable arbitrary paths as verified weights."""
    backend = request.get("backend")
    if backend not in ("mlx", "transformers"):
        raise ValueError("adapter backend must be mlx or transformers")
    source = Path(request["path"]).resolve()
    required = [
        "adapter_config.json",
        "adapters.safetensors" if backend == "mlx" else "adapter_model.safetensors",
    ]
    files = []
    for name in required:
        file = source / name
        if file.is_symlink() or not file.is_file() or not 0 < file.stat().st_size < 2 * 1024**3:
            raise ValueError(f"required adapter file is missing, linked or too large: {name}")
        files.append({"path": name, "sha256": file_digest(file), "bytes": file.stat().st_size})
    base = _fingerprint(request["base_model"])
    identity = digest({"backend": backend, "files": files, "base_sha256": base["sha256"]})
    directory = Path(registry_dir).resolve() / identity[:24]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = _adapter(identity[:24], registry_dir)
        return existing
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "weights"
    weights.mkdir()
    for item in files:
        shutil.copyfile(source / item["path"], weights / item["path"])
        if file_digest(weights / item["path"]) != item["sha256"]:
            raise ValueError("adapter changed during registration")
    manifest = {
        "schema_version": 1,
        "id": identity[:24],
        "identity_sha256": identity,
        "backend": backend,
        "path": str(weights),
        "files": files,
        "base_model": base,
        "python": request.get("python"),
        "training_result": json.loads(json.dumps(request.get("training_result"), allow_nan=False)),
        "reload_status": "unverified",
        "registered_at": time.time(),
    }
    write_json(manifest_path, manifest)
    return manifest


def list_adapters(registry_dir):
    root = Path(registry_dir)
    if not root.exists():
        return []
    result = []
    for path in sorted(root.glob("*/manifest.json")):
        if path.is_symlink() or not re.fullmatch(r"[a-f0-9]{24}", path.parent.name):
            continue
        try:
            manifest = json.loads(path.read_text())
            result.append(
                {
                    key: manifest.get(key)
                    for key in (
                        "id",
                        "backend",
                        "path",
                        "base_model",
                        "reload_status",
                        "registered_at",
                    )
                }
            )
        except (ValueError, OSError):
            continue
    return result


def _adapter(adapter_id, registry_dir):
    if not isinstance(adapter_id, str) or not re.fullmatch(r"[a-f0-9]{24}", adapter_id):
        raise ValueError("invalid adapter ID")
    directory = Path(registry_dir).resolve() / adapter_id
    manifest = json.loads((directory / "manifest.json").read_text())
    for file in manifest["files"]:
        path = directory / "weights" / file["path"]
        if path.is_symlink() or not path.is_file() or file_digest(path) != file["sha256"]:
            raise ValueError("registered adapter was modified; register a new revision")
    if _fingerprint(manifest["base_model"]["path"])["sha256"] != manifest["base_model"]["sha256"]:
        raise ValueError("base model changed after adapter registration")
    manifest["path"] = str(directory / "weights")
    return manifest


def evaluate_adapter(request, registry_dir, output_dir, *, cancel=None, progress=None):
    """Fresh-process MLX or Transformers/PEFT inference, followed by domain task scoring."""
    manifest = _adapter(request["adapter_id"], registry_dir)
    rows = normalise_rows(request["rows"])
    max_tokens = request.get("max_tokens", 128)
    positive(max_tokens, "max_tokens", integer=True)
    if max_tokens > 4096 or len(rows) > 1000:
        raise ValueError("adapter evaluation is bounded to 1000 rows and 4096 completion tokens")
    out = _new_output(output_dir)
    request = {
        **request,
        "backend": manifest["backend"],
        "python": request.get("python") or manifest.get("python"),
    }
    runtime, payload, env, limits = _runtime_request(request, out, operation="evaluate")
    payload.update(
        base_model=manifest["base_model"]["path"],
        adapter_path=None if request.get("baseline", False) else manifest["path"],
        rows=[{k: v for k, v in row.items() if k in TASK_KEYS} for row in rows],
        max_tokens=max_tokens,
    )
    process, worker = _execute_worker(
        request, out, payload, env, limits, cancel=cancel, progress=progress
    )
    predictions = worker.get("predictions", []) if worker else []
    report = evaluate_predictions(rows, predictions)
    result = {
        "status": process["status"],
        "adapter_id": manifest["id"],
        "baseline": bool(request.get("baseline", False)),
        "runtime": runtime,
        "process": process,
        "worker": worker,
        "evaluation": report,
        "result_path": str(out / "evaluation.json"),
    }
    if process["status"] == "passed" and (not worker or worker.get("status") != "passed"):
        result["status"] = "failed"
    if result["status"] == "passed" and not result["baseline"]:
        manifest["reload_status"] = "verified"
        manifest["last_evaluation"] = {
            "path": result["result_path"],
            "quality": report["quality"],
            "dataset_sha256": report["dataset_sha256"],
        }
        write_json(Path(registry_dir) / manifest["id"] / "manifest.json", manifest)
    write_json(out / "evaluation.json", result)
    return result


def reload_adapter(request, registry_dir, output_dir, *, cancel=None, progress=None):
    row = {
        "id": "reload-smoke",
        "prompt": request.get("prompt", "Reply with the word ready."),
        "expected": request.get("expected", "ready"),
        "evaluator": "exact",
    }
    result = evaluate_adapter(
        {**request, "rows": [row], "baseline": False},
        registry_dir,
        output_dir,
        cancel=cancel,
        progress=progress,
    )
    result["scope"] = "adapter attachment and bounded generation smoke; quality is separate"
    return result


def compare_adapters(baseline, candidate, gates=None):
    if baseline.get("status") != "passed" or candidate.get("status") != "passed":
        raise ValueError("adapter comparison requires successful evaluation executions")
    return compare_regression(baseline["evaluation"], candidate["evaluation"], gates)
