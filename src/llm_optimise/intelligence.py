"""Bounded exact caching and task-scoped empirical routing. No invented measurements."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import sqlite3
import statistics
import threading
import time
from contextlib import nullcontext
from pathlib import Path

from .agent import route_request
from .providers import complete, estimate_input_tokens
from .quality import Task
from .workspace import CODE_SCHEMA


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def model_identity(model):
    return fingerprint(
        {
            k: getattr(model, k)
            for k in (
                "id",
                "model",
                "provider",
                "base_url",
                "location",
                "revision",
                "adapter_revision",
                "supports_json_schema",
            )
        }
    )


def positive(value, name, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ValueError(f"{name} must be positive and <= {maximum}")
    return value


def wilson_lower(scores):
    if not scores:
        return None
    n, p, z = len(scores), statistics.mean(scores), 1.96
    return max(
        0.0,
        (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
        / (1 + z * z / n),
    )


def quality_score(task, text):
    if task is None:
        return None
    if task.json_schema is not None:
        from .datasets import validate_json_schema

        try:
            if not validate_json_schema(json.loads(text), task.json_schema)["valid"]:
                return 0.0
        except ValueError:
            return 0.0
    return task.score(text)


class Intelligence:
    def __init__(
        self,
        workspace,
        *,
        hardware_id=None,
        max_cache_bytes=32 * 1024**2,
        max_records=5000,
        ttl_s=86400,
        min_samples=3,
    ):
        directory = Path(workspace).resolve() / ".llm-optimise"
        if directory.is_symlink():
            raise ValueError("state directory cannot be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "intelligence.sqlite3"
        if self.path.is_symlink():
            raise ValueError("state database cannot be a symlink")
        self.hardware_id = hardware_id or fingerprint(
            [platform.system(), platform.machine(), platform.node()]
        )
        self.max_cache_bytes = int(positive(max_cache_bytes, "cache size", 1024**3))
        self.max_records = int(positive(max_records, "record count", 100000))
        self.ttl_s = positive(ttl_s, "cache TTL", 30 * 86400)
        self.min_samples = int(positive(min_samples, "minimum samples", 10000))
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, size INTEGER NOT NULL, expires REAL NOT NULL, touched REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY, scope TEXT NOT NULL, model_id TEXT NOT NULL, task_class TEXT NOT NULL, created REAL NOT NULL, value TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS observation_scope ON observations(scope, created);
                CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def scope(self, model, task_class, input_tokens):
        if not isinstance(task_class, str) or not task_class.strip() or len(task_class) > 128:
            raise ValueError("task class must be 1–128 characters")
        return fingerprint(
            [
                model_identity(model),
                task_class,
                max(0, math.ceil(math.log2(max(1, input_tokens)))),
                self.hardware_id,
            ]
        )

    def cache_get(self, key):
        now = time.time()
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM cache WHERE expires < ?", (now,))
            row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            counter = "hits" if row else "misses"
            db.execute(
                "INSERT INTO counters VALUES (?,1) ON CONFLICT(name) DO UPDATE SET value=value+1",
                (counter,),
            )
            if row:
                db.execute("UPDATE cache SET touched=? WHERE key=?", (now, key))
                return json.loads(row[0])
        return None

    def cache_put(self, key, value):
        raw = json.dumps(value, allow_nan=False)
        size = len(raw.encode())
        if size > min(self.max_cache_bytes, 4 * 1024**2):
            return
        now = time.time()
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM cache WHERE expires < ?", (now,))
            db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                (key, raw, size, now + self.ttl_s, now),
            )
            while (
                db.execute("SELECT COALESCE(SUM(size),0) FROM cache").fetchone()[0]
                > self.max_cache_bytes
            ):
                db.execute(
                    "DELETE FROM cache WHERE key=(SELECT key FROM cache ORDER BY touched LIMIT 1)"
                )

    def clear_cache(self):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM cache")
            db.execute("DELETE FROM counters")
        return {"status": "cleared"}

    def record(self, model, task_class, input_tokens, completion, score=None, *, error=None):
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score not in (0, 1)
        ):
            raise ValueError("quality observations must be binary pass/fail for Wilson confidence")
        record = {
            k: completion.get(k)
            for k in (
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "provider_reported_cost_usd",
                "accounted_cost_usd",
                "cached_input_tokens",
            )
        }
        record.update(
            score=score,
            error=error,
            model_identity=model_identity(model),
            revision=model.revision,
            adapter_revision=model.adapter_revision,
            input_bucket=max(0, math.ceil(math.log2(max(1, input_tokens)))),
            hardware_id=self.hardware_id,
        )
        with self.lock, self.connect() as db:
            db.execute(
                "INSERT INTO observations(scope,model_id,task_class,created,value) VALUES (?,?,?,?,?)",
                (
                    self.scope(model, task_class, input_tokens),
                    model.id,
                    task_class,
                    time.time(),
                    json.dumps(record, allow_nan=False),
                ),
            )
            while db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] > self.max_records:
                db.execute(
                    "DELETE FROM observations WHERE id=(SELECT MIN(id) FROM observations WHERE scope=(SELECT scope FROM observations GROUP BY scope ORDER BY COUNT(*) DESC, MIN(id) LIMIT 1))"
                )

    def measurements(self, model, task_class, input_tokens):
        with self.connect() as db:
            rows = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT value FROM observations WHERE scope=? AND created>? ORDER BY id",
                    (self.scope(model, task_class, input_tokens), time.time() - 30 * 86400),
                )
            ]
        latencies = [
            r["latency_ms"]
            for r in rows
            if r.get("error") is None
            and isinstance(r.get("latency_ms"), (int, float))
            and r["latency_ms"] > 0
        ]
        scores = [r["score"] for r in rows if r.get("score") is not None]
        return {
            "samples": len(rows),
            "latency_samples": len(latencies),
            "quality_samples": len(scores),
            "latency_ms": statistics.median(latencies) if latencies else None,
            "quality_mean": statistics.mean(scores) if scores else None,
            "quality_lower_95": wilson_lower(scores),
            "failures": sum(bool(r.get("error")) for r in rows),
        }

    def calibrated_models(self, models, task_class, input_tokens):
        calibrated, evidence = [], {}
        for model in models:
            if model.task_classes and task_class not in model.task_classes:
                continue
            metrics = self.measurements(model, task_class, input_tokens)
            evidence[model.id] = metrics
            # Opting into empirical routing removes stale/manual quality and latency.
            calibrated.append(
                dataclasses.replace(
                    model,
                    latency_ms=metrics["latency_ms"]
                    if metrics["latency_samples"] >= self.min_samples
                    else None,
                    quality=metrics["quality_lower_95"]
                    if metrics["quality_samples"] >= self.min_samples
                    else None,
                )
            )
        return calibrated, evidence

    def status(self):
        with self.connect() as db:
            size, count = db.execute(
                "SELECT COALESCE(SUM(size),0),COUNT(*) FROM cache WHERE expires>?", (time.time(),)
            ).fetchone()
            counters = dict(db.execute("SELECT name,value FROM counters"))
            groups = [
                {"model_id": m, "task_class": t, "samples": n}
                for m, t, n in db.execute(
                    "SELECT model_id,task_class,COUNT(*) FROM observations GROUP BY model_id,task_class"
                )
            ]
        return {
            "cache": {
                "entries": count,
                "bytes": size,
                "max_bytes": self.max_cache_bytes,
                "ttl_s": self.ttl_s,
                **counters,
            },
            "measurements": groups,
            "minimum_samples": self.min_samples,
            "quality_policy": "95% Wilson lower bound from scored requests; unscored chat does not establish quality",
            "hardware_id": self.hardware_id,
            "retention": "bounded to the configured record count; evict oldest from the most populated scope first; routing evidence expires after 30 days",
        }

    def run(
        self,
        models,
        messages,
        policy=None,
        selected_model=None,
        max_tokens=1024,
        capability="chat",
        *,
        task_class="general",
        cache=False,
        calibrated=False,
        prefix_cache=False,
        json_schema=None,
        quality_task=None,
        cancel=None,
        lease=None,
    ):
        for name, value in (
            ("cache", cache),
            ("calibrated", calibrated),
            ("prefix_cache", prefix_cache),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        started = time.monotonic()
        positive(max_tokens, "max_tokens", 32768)
        if type(max_tokens) is not int:
            raise ValueError("max_tokens must be an integer")
        input_tokens = estimate_input_tokens(messages)
        models = [m for m in models if not m.task_classes or task_class in m.task_classes]
        evidence = {}
        if calibrated:
            models, evidence = self.calibrated_models(models, task_class, input_tokens)
        route = route_request(models, messages, policy, selected_model, max_tokens, capability)
        model = next(m for m in models if m.id == route["model_id"])
        if cache and not model.revision:
            raise ValueError(
                "exact caching requires an explicit model revision; register the pinned model revision first"
            )
        route["calibration"] = {
            "enabled": calibrated,
            "task_class": task_class,
            "evidence": evidence,
        }
        schema = (
            json_schema
            if json_schema is not None
            else CODE_SCHEMA
            if capability == "code"
            else None
        )
        key = fingerprint(
            {
                "model": model_identity(model),
                "messages": messages,
                "output": max_tokens,
                "schema": schema,
                "temperature": 0 if cache else None,
                "prefix_cache": prefix_cache,
                "protocol": 2,
            }
        )
        if cancel is not None and cancel.is_set():
            raise InterruptedError("request cancelled")
        if cache:
            completion = self.cache_get(key)
            if completion is not None:
                completion.update(
                    cache_hit=True,
                    cache_lookup_ms=(time.monotonic() - started) * 1000,
                    source_latency_ms=completion.get("latency_ms"),
                    latency_ms=(time.monotonic() - started) * 1000,
                    source_cost_usd=completion.get(
                        "provider_reported_cost_usd", completion.get("accounted_cost_usd")
                    ),
                    provider_reported_cost_usd=None,
                    accounted_cost_usd=0.0,
                    response_id=None,
                    cost_note="exact result cache hit; no provider request or new provider cost",
                )
                return {
                    "route": route,
                    "completion": completion,
                    "quality": quality_score(quality_task, completion["text"]),
                }
        options = {"json_schema": schema, "prefix_cache": prefix_cache}
        if cache:
            options["temperature"] = 0
        try:
            with lease(model) if lease else nullcontext(model) as live_model:
                completion = complete(live_model, messages, max_tokens, **options)
        except Exception as exc:
            self.record(
                model,
                task_class,
                input_tokens,
                {},
                0.0 if quality_task else None,
                error=type(exc).__name__,
            )
            raise
        score = quality_score(quality_task, completion["text"])
        self.record(model, task_class, input_tokens, completion, score)
        completion["cache_hit"] = False
        if cache:
            self.cache_put(key, completion)
        return {"route": route, "completion": completion, "quality": score}

    def respond(self, models, messages, *, contract, escalation_model=None, **options):
        """At most two explicitly selected attempts; deterministic contract or abstention."""
        from .datasets import validate_json_schema, validate_schema_definition

        if (
            not isinstance(contract, dict)
            or set(contract) - {"allowed_outputs", "json_schema"}
            or len(contract) != 1
        ):
            raise ValueError("response contract needs allowed_outputs or json_schema")
        if "json_schema" in contract:
            validate_schema_definition(contract["json_schema"])
            options["json_schema"] = contract["json_schema"]
        else:
            allowed = contract["allowed_outputs"]
            if (
                not isinstance(allowed, list)
                or not 1 <= len(allowed) <= 100
                or any(not isinstance(v, str) or not v for v in allowed)
            ):
                raise ValueError("allowed_outputs must contain 1–100 nonempty strings")
        policy = options.get("policy") or {"placement": "local", "objective": "cost"}
        policy = dataclasses.asdict(policy) if dataclasses.is_dataclass(policy) else dict(policy)
        cap = policy.get("max_cost_usd")
        if escalation_model and (cap is None or not any(m.id == escalation_model for m in models)):
            raise ValueError("explicit escalation requires a known model and a total cost budget")
        attempts = []
        spent = 0.0
        cost_known = True
        for attempt in range(2 if escalation_model else 1):
            if attempt:
                if attempts[0]["route"]["model_id"] == escalation_model or spent >= cap:
                    break
                options["selected_model"] = escalation_model
                options["policy"] = {**policy, "max_cost_usd": cap - spent}
            result = self.run(models, messages, **options)
            text = result["completion"]["text"].strip()
            if "allowed_outputs" in contract:
                accepted = text in contract["allowed_outputs"]
            else:
                try:
                    accepted = validate_json_schema(json.loads(text), contract["json_schema"])[
                        "valid"
                    ]
                except ValueError:
                    accepted = False
            result["contract_passed"] = accepted
            attempts.append(result)
            cost = result["completion"].get("provider_reported_cost_usd")
            if cost is None:
                cost = result["completion"].get("accounted_cost_usd")
            if cost is None:
                cost = result["route"].get("estimated_cost_usd")
            cost_known = cost_known and cost is not None
            spent += cost or 0
            if accepted:
                return {
                    "accepted": True,
                    "abstained": False,
                    "output": text,
                    "attempts": attempts,
                    "cost_usd": spent if cost_known else None,
                    "budget_exceeded": cap is not None and spent > cap,
                }
            if not cost_known:
                break
        return {
            "accepted": False,
            "abstained": True,
            "output": None,
            "attempts": attempts,
            "cost_usd": spent if cost_known else None,
            "budget_exceeded": cap is not None and spent > cap,
            "reason": "No response passed the explicit output contract",
        }

    def calibrate(
        self,
        models,
        tasks,
        *,
        task_class,
        model_ids=None,
        repeats=1,
        max_requests=100,
        max_cost_usd=0.1,
        max_tokens=128,
        cancel=None,
        progress=None,
        lease=None,
    ):
        for name, v, cap in (
            ("repeats", repeats, 20),
            ("max_requests", max_requests, 1000),
            ("max_tokens", max_tokens, 32768),
        ):
            positive(v, name, cap)
            if type(v) is not int:
                raise ValueError(f"{name} must be an integer")
        positive(max_cost_usd, "max_cost_usd", 100)
        if model_ids is not None and (
            not isinstance(model_ids, list)
            or any(not isinstance(v, str) for v in model_ids)
            or set(model_ids) - {m.id for m in models}
        ):
            raise ValueError("calibration model IDs must identify configured models")
        selected = [m for m in models if model_ids is None or m.id in model_ids]
        if any(m.task_classes and task_class not in m.task_classes for m in selected):
            raise ValueError("selected specialist does not support this calibration task class")
        if not selected or not tasks or len(selected) * len(tasks) * repeats > max_requests:
            raise ValueError("no models/tasks selected or calibration exceeds request limit")
        if not all(isinstance(t, Task) for t in tasks):
            raise ValueError("calibration requires validated tasks")
        from .datasets import validate_schema_definition

        for task in tasks:
            if task.json_schema is not None:
                validate_schema_definition(task.json_schema)
        reserved = 0.0
        for m in selected:
            for t in tasks:
                msgs = [
                    {"role": "system", "content": t.system},
                    {"role": "user", "content": t.prompt},
                ]
                cost = m.estimated_cost_usd(estimate_input_tokens(msgs), max_tokens)
                if cost is None:
                    raise ValueError(f"{m.id} needs prices to bound calibration")
                reserved += cost * repeats
        if reserved > max_cost_usd:
            raise ValueError("calibration maximum estimated cost exceeds budget")
        results = []
        spent = 0.0
        for model in selected:
            for _ in range(repeats):
                for task in tasks:
                    if cancel is not None and cancel.is_set():
                        return {"cancelled": True, "requests": results, "accounted_cost_usd": spent}
                    messages = [
                        {"role": "system", "content": task.system},
                        {"role": "user", "content": task.prompt},
                    ]
                    estimate = model.estimated_cost_usd(estimate_input_tokens(messages), max_tokens)
                    if spent + estimate > max_cost_usd:
                        return {
                            "cancelled": False,
                            "budget_exhausted": True,
                            "budget_exceeded": spent > max_cost_usd,
                            "requests": results,
                            "accounted_cost_usd": spent,
                            "reserved_estimate_usd": reserved,
                        }
                    try:
                        result = self.run(
                            selected,
                            messages,
                            {
                                "placement": model.location,
                                "objective": "cost",
                                "max_cost_usd": max_cost_usd - spent,
                            },
                            model.id,
                            max_tokens,
                            task_class=task_class,
                            quality_task=task,
                            json_schema=task.json_schema,
                            cancel=cancel,
                            lease=lease,
                        )
                    except Exception as exc:
                        return {
                            "status": "cancelled"
                            if isinstance(exc, InterruptedError)
                            else "failed",
                            "cancelled": isinstance(exc, InterruptedError),
                            "error": type(exc).__name__,
                            "failed_task_id": task.id,
                            "failed_model_id": model.id,
                            "requests": results,
                            "accounted_cost_usd": spent,
                            "reserved_estimate_usd": reserved,
                        }
                    cost = result["completion"].get("provider_reported_cost_usd")
                    if cost is None:
                        cost = result["completion"].get("accounted_cost_usd")
                    spent += cost if cost is not None else estimate
                    results.append(
                        {
                            "model_id": model.id,
                            "task_id": task.id,
                            "score": result["quality"],
                            "latency_ms": result["completion"]["latency_ms"],
                            "cost_usd": cost,
                        }
                    )
                    if progress:
                        progress(
                            {
                                "phase": "calibrating",
                                "completed": len(results),
                                "total": len(selected) * len(tasks) * repeats,
                                "cost_usd": spent,
                            }
                        )
        return {
            "cancelled": False,
            "budget_exceeded": spent > max_cost_usd,
            "requests": results,
            "accounted_cost_usd": spent,
            "reserved_estimate_usd": reserved,
            "note": "Budget is an estimate; provider-reported costs are not a billing cap. Latency excludes cache hits; quality is task-specific.",
        }
