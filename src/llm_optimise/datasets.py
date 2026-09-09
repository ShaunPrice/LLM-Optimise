"""Bounded domain datasets, group-disjoint splits and deterministic regression gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import digest, finite
from .quality import Task, _same_json
from .runner import write_json

MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100000
TASK_KEYS = {"id", "prompt", "expected", "evaluator", "system", "tolerance", "json_schema"}
ROW_KEYS = TASK_KEYS | {"group", "task_class", "error_cost", "schema_error_cost", "split"}
SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "description",
    "title",
}
SCHEMA_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def validate_schema_definition(schema, *, _depth=0):
    """Validate our explicitly supported JSON-schema subset; unsupported keywords fail closed.

    Supported: type (single type), properties, required, additionalProperties (bool),
    items (one schema), enum, const, numeric bounds, string/array length bounds,
    title and description. References, regex, formats and composition are rejected.
    """
    if not isinstance(schema, dict) or _depth > 20:
        raise ValueError("schema must be an object with nesting depth <=20")
    unknown = set(schema) - SCHEMA_KEYS
    if unknown:
        raise ValueError(f"unsupported schema keywords: {sorted(unknown)}")
    if "type" in schema and (
        not isinstance(schema["type"], str) or schema["type"] not in SCHEMA_TYPES
    ):
        raise ValueError("schema type must be one supported type string")
    for key in ("description", "title"):
        if key in schema and not isinstance(schema[key], str):
            raise ValueError(f"schema {key} must be a string")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        raise ValueError("additionalProperties must be a boolean")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict) or len(schema["properties"]) > 200:
            raise ValueError("properties must be an object with <=200 entries")
        for key, value in schema["properties"].items():
            if not isinstance(key, str):
                raise ValueError("property names must be strings")
            validate_schema_definition(value, _depth=_depth + 1)
    if "required" in schema:
        values = schema["required"]
        if (
            not isinstance(values, list)
            or any(not isinstance(x, str) for x in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("required must list unique property names")
    if "items" in schema:
        validate_schema_definition(schema["items"], _depth=_depth + 1)
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("enum must be a nonempty list")
    for key in ("minimum", "maximum"):
        if key in schema:
            finite(schema[key], key)
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    for lo, hi in (("minimum", "maximum"), ("minLength", "maxLength"), ("minItems", "maxItems")):
        if lo in schema and hi in schema and schema[lo] > schema[hi]:
            raise ValueError(f"{lo} exceeds {hi}")
    try:
        json.dumps(schema, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("schema values must be finite JSON") from exc
    return schema


def validate_json_schema(value, schema):
    """Return {valid, errors}; reject unsupported schemas instead of silently ignoring them."""
    validate_schema_definition(schema)
    errors = []
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError):
        return {"valid": False, "errors": ["$: value must be finite JSON"]}

    def visit(item, rule, path):
        kind = rule.get("type")
        checks = {
            "object": isinstance(item, dict),
            "array": isinstance(item, list),
            "string": isinstance(item, str),
            "number": type(item) in (int, float) and math.isfinite(item),
            "integer": type(item) in (int, float) and math.isfinite(item) and int(item) == item,
            "boolean": type(item) is bool,
            "null": item is None,
        }
        if kind and not checks[kind]:
            errors.append(f"{path}: expected {kind}")
            return
        if "enum" in rule and not any(_same_json(item, option) for option in rule["enum"]):
            errors.append(f"{path}: value is outside enum")
        if "const" in rule and not _same_json(item, rule["const"]):
            errors.append(f"{path}: value differs from const")
        if isinstance(item, dict):
            props = rule.get("properties", {})
            for key in rule.get("required", []):
                if key not in item:
                    errors.append(f"{path}.{key}: required property missing")
            if rule.get("additionalProperties") is False:
                for key in item.keys() - props.keys():
                    errors.append(f"{path}.{key}: additional property")
            for key, child in props.items():
                if key in item:
                    visit(item[key], child, f"{path}.{key}")
        if isinstance(item, list):
            for key, compare in (
                ("minItems", len(item) < rule.get("minItems", 0)),
                ("maxItems", len(item) > rule.get("maxItems", len(item))),
            ):
                if compare:
                    errors.append(f"{path}: {key} violated")
            if "items" in rule:
                for i, child in enumerate(item):
                    visit(child, rule["items"], f"{path}[{i}]")
        if isinstance(item, str):
            if len(item) < rule.get("minLength", 0) or len(item) > rule.get("maxLength", len(item)):
                errors.append(f"{path}: string length outside bounds")
        if type(item) in (int, float):
            if (
                not math.isfinite(item)
                or item < rule.get("minimum", item)
                or item > rule.get("maximum", item)
            ):
                errors.append(f"{path}: number outside bounds")

    visit(value, schema, "$")
    return {"valid": not errors, "errors": errors[:100]}


def load_rows(source):
    """Read a bounded list, JSON file or JSONL file. Never execute dataset content."""
    if isinstance(source, list):
        rows = source
        raw = json.dumps(source, allow_nan=False).encode()
    else:
        if not isinstance(source, (str, Path)):
            raise ValueError("source must be a row list or dataset file path")
        path = Path(source)
        if not path.is_file() or path.stat().st_size > MAX_DATASET_BYTES:
            raise ValueError("dataset must be a file no larger than 16 MiB")
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
            rows = (
                json.loads(text)
                if text.lstrip().startswith("[")
                else [json.loads(line) for line in text.splitlines() if line.strip()]
            )
        except (ValueError, UnicodeError) as exc:
            raise ValueError("dataset must contain UTF-8 JSON or JSONL") from exc
    if len(raw) > MAX_DATASET_BYTES or not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("dataset requires 1–100000 rows and <=16 MiB")
    return rows


def normalise_rows(rows, *, allow_duplicate_ids=False):
    """Validate canonical tasks and accept Alpaca data as exact-answer tasks."""
    result, ids = [], {}
    for index, source in enumerate(load_rows(rows)):
        if not isinstance(source, dict):
            raise ValueError(f"row {index + 1} must be an object")
        row = dict(source)
        if "instruction" in row:
            allowed = {
                "instruction",
                "input",
                "output",
                "id",
                "group",
                "task_class",
                "error_cost",
                "schema_error_cost",
                "split",
            }
            if (
                set(row) - allowed
                or not isinstance(row.get("instruction"), str)
                or not isinstance(row.get("input", ""), str)
                or not isinstance(row.get("output"), str)
            ):
                raise ValueError("Alpaca rows require instruction/input/output strings")
            prompt = row.pop("instruction") + (
                "\n\n" + row.pop("input", "") if row.get("input") else ""
            )
            row.pop("input", None)
            row.update(prompt=prompt, expected=row.pop("output"), evaluator="exact")
        if set(row) - ROW_KEYS:
            raise ValueError(
                f"row {index + 1} contains unsupported fields: {sorted(set(row) - ROW_KEYS)}"
            )
        row.setdefault("id", "task-" + digest({k: v for k, v in row.items() if k != "id"})[:16])
        task = Task(**{key: value for key, value in row.items() if key in TASK_KEYS})
        if task.id in ids and (not allow_duplicate_ids or row != ids[task.id]):
            raise ValueError("dataset task IDs must be unique or exact duplicate rows")
        ids[task.id] = dict(row)
        if len(task.prompt) > 64000 or len(task.system) > 64000:
            raise ValueError("task prompt/system exceeds 64000 characters")
        if task.json_schema is not None:
            validate_schema_definition(task.json_schema)
            if (
                task.evaluator != "json_subset"
                and not validate_json_schema(task.expected, task.json_schema)["valid"]
            ):
                raise ValueError(f"task {task.id} expected value does not match its schema")
        row.setdefault("group", task.id)
        row.setdefault("task_class", "general")
        row.setdefault("error_cost", 1.0)
        row.setdefault("schema_error_cost", row["error_cost"])
        for key in ("group", "task_class"):
            if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 256:
                raise ValueError(f"{key} must contain 1–256 characters")
        for key in ("error_cost", "schema_error_cost"):
            finite(row[key], key)
            if row[key] < 0:
                raise ValueError(f"{key} must be nonnegative")
        if "split" in row and row["split"] not in ("train", "tune", "held_out"):
            raise ValueError("split must be train, tune or held_out")
        # Make task defaults explicit in the preserved metadata.
        row.update(evaluator=task.evaluator, system=task.system, tolerance=task.tolerance)
        result.append(row)
    return result


def prepare_dataset(request, output_dir):
    """Import/deduplicate/split a dataset. Split membership depends only on seed and group."""
    if not isinstance(request, dict) or ("rows" in request) == ("source" in request):
        raise ValueError("provide exactly one of rows or source")
    rows = normalise_rows(
        load_rows(request.get("rows", request.get("source"))), allow_duplicate_ids=True
    )
    seed = request.get("seed", 42)
    if type(seed) not in (int, str):
        raise ValueError("seed must be an integer or string")
    ratios = request.get("ratios", {"train": 0.7, "tune": 0.15, "held_out": 0.15})
    if not isinstance(ratios, dict) or set(ratios) != {"train", "tune", "held_out"}:
        raise ValueError("ratios require train, tune and held_out")
    for key, value in ratios.items():
        finite(value, key)
        if not 0 < value < 1:
            raise ValueError("each split ratio must be greater than zero and below one")
    if not math.isclose(sum(ratios.values()), 1, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    parents = {row["group"]: row["group"] for row in rows}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    unique, duplicates = {}, []
    for row in sorted(rows, key=lambda value: value["id"]):
        identity = digest({"prompt": row["prompt"].strip(), "system": row["system"].strip()})
        if identity in unique:
            previous = unique[identity]
            semantic = (
                "expected",
                "evaluator",
                "tolerance",
                "json_schema",
                "task_class",
                "error_cost",
                "schema_error_cost",
            )
            if any(not _same_json(row.get(key), previous.get(key)) for key in semantic):
                raise ValueError(
                    f"conflicting duplicate labels or scoring rules: {previous['id']}, {row['id']}"
                )
            left, right = sorted((find(previous["group"]), find(row["group"])))
            parents[right] = left
            duplicates.append({"removed": row["id"], "retained": previous["id"]})
        else:
            unique[identity] = row
    splits = {key: [] for key in ("train", "tune", "held_out")}
    all_rows = []
    for row in unique.values():
        row = dict(row)
        row["group"] = find(row["group"])
        fraction = (
            int(hashlib.sha256(f"{seed}\0{row['group']}".encode()).hexdigest()[:16], 16) / 2**64
        )
        split = (
            "train"
            if fraction < ratios["train"]
            else "tune"
            if fraction < ratios["train"] + ratios["tune"]
            else "held_out"
        )
        row["split"] = split
        splits[split].append(row)
        all_rows.append(row)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("dataset output directory must be empty; keep imports immutable")
    paths = {}
    for split, values in splits.items():
        path = out / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({k: v for k, v in row.items() if k in TASK_KEYS}, allow_nan=False) + "\n"
                for row in values
            ),
            encoding="utf-8",
        )
        supervised = out / f"{split}-sft.jsonl"
        supervised.write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": row["system"]},
                            {"role": "user", "content": row["prompt"]},
                            {
                                "role": "assistant",
                                "content": row["expected"]
                                if isinstance(row["expected"], str)
                                else json.dumps(row["expected"], allow_nan=False),
                            },
                        ]
                    }
                )
                + "\n"
                for row in values
            ),
            encoding="utf-8",
        )
        metadata_path = out / f"{split}-rows.jsonl"
        metadata_path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in values), encoding="utf-8"
        )
        paths[split] = {
            "rows_path": str(metadata_path),
            "tasks": str(path),
            "supervised": str(supervised),
            "rows": len(values),
            "groups": len({r["group"] for r in values}),
        }
    manifest = {
        "schema_version": 1,
        "name": str(request.get("name", "Domain dataset"))[:256],
        "seed": seed,
        "ratios": ratios,
        "input_rows": len(rows),
        "unique_rows": len(all_rows),
        "duplicates": duplicates,
        "dataset_sha256": digest(all_rows),
        "splits": paths,
        "rows": all_rows,
        "group_disjoint": True,
        "warnings": [
            f"{key} is empty; provide more independent groups before evaluating"
            for key, values in splits.items()
            if not values
        ],
    }
    write_json(out / "manifest.json", manifest)
    return {**manifest, "manifest_path": str(out / "manifest.json")}


def evaluate_predictions(rows, predictions):
    """Score missing outputs as failures; schema failures are tracked and costed explicitly."""
    rows = normalise_rows(rows)
    if isinstance(predictions, list):
        if any(not isinstance(x, dict) or "id" not in x or "output" not in x for x in predictions):
            raise ValueError("predictions require id and output")
        if len({x["id"] for x in predictions}) != len(predictions):
            raise ValueError("prediction IDs must be unique")
        predictions = {x["id"]: x["output"] for x in predictions}
    if not isinstance(predictions, dict) or set(predictions) - {r["id"] for r in rows}:
        raise ValueError("predictions must map known task IDs to output strings")
    results = []
    for row in rows:
        output = predictions.get(row["id"])
        if output is not None and not isinstance(output, str):
            raise ValueError("prediction output must be a string")
        task = Task(**{k: v for k, v in row.items() if k in TASK_KEYS})
        schema_valid, schema_errors = True, []
        if task.json_schema is not None or task.evaluator in ("json", "json_subset"):
            try:
                value = json.loads(output) if output is not None else None
                if output is None:
                    raise ValueError("missing output")
                if task.json_schema is not None:
                    validation = validate_json_schema(value, task.json_schema)
                    schema_valid, schema_errors = validation["valid"], validation["errors"]
            except (ValueError, TypeError):
                schema_valid, schema_errors = False, ["response is missing or is not valid JSON"]
        score = task.score(output) if output is not None and schema_valid else 0.0
        cost = (1 - score) * row["error_cost"]
        if not schema_valid:
            cost += row["schema_error_cost"]
        results.append(
            {
                "id": row["id"],
                "task_class": row["task_class"],
                "score": score,
                "schema_valid": schema_valid,
                "schema_errors": schema_errors,
                "missing": output is None,
                "error_cost": cost,
                "output": output,
            }
        )
    classes = {
        key: [r for r in results if r["task_class"] == key]
        for key in {r["task_class"] for r in results}
    }
    return {
        "schema_version": 1,
        "dataset_sha256": digest(sorted(rows, key=lambda x: x["id"])),
        "tasks": len(results),
        "quality": sum(r["score"] for r in results) / len(results),
        "schema_failures": sum(not r["schema_valid"] for r in results),
        "missing_outputs": sum(r["missing"] for r in results),
        "total_error_cost": sum(r["error_cost"] for r in results),
        "by_task_class": {
            key: {
                "tasks": len(values),
                "quality": sum(r["score"] for r in values) / len(values),
                "error_cost": sum(r["error_cost"] for r in values),
            }
            for key, values in sorted(classes.items())
        },
        "results": results,
    }


def compare_regression(baseline, candidate, gates=None):
    gates = {
        "max_quality_drop": 0.0,
        "max_error_cost_increase": 0.0,
        "max_schema_failure_increase": 0,
        "max_regressed_tasks": 0,
        **(gates or {}),
    }
    allowed = {
        "max_quality_drop",
        "max_error_cost_increase",
        "max_schema_failure_increase",
        "max_regressed_tasks",
        "min_quality",
    }
    if set(gates) - allowed:
        raise ValueError("unsupported regression gate")
    for key, value in gates.items():
        finite(value, key)
        if value < 0 or (key in ("max_quality_drop", "min_quality") and value > 1):
            raise ValueError(f"invalid {key}")
    if baseline.get("dataset_sha256") != candidate.get("dataset_sha256") or not baseline.get(
        "dataset_sha256"
    ):
        raise ValueError("regression comparison requires the same dataset fingerprint")
    before = {row["id"]: row for row in baseline["results"]}
    after = {row["id"]: row for row in candidate["results"]}
    if set(before) != set(after):
        raise ValueError("regression comparison requires identical task IDs")
    delta = {
        "quality": candidate["quality"] - baseline["quality"],
        "error_cost": candidate["total_error_cost"] - baseline["total_error_cost"],
        "schema_failures": candidate["schema_failures"] - baseline["schema_failures"],
    }
    failures = []
    if delta["quality"] < -gates["max_quality_drop"] - 1e-12:
        failures.append("quality regression")
    if delta["error_cost"] > gates["max_error_cost_increase"] + 1e-12:
        failures.append("task error cost increased")
    if delta["schema_failures"] > gates["max_schema_failure_increase"]:
        failures.append("schema failures increased")
    if candidate["quality"] < gates.get("min_quality", 0):
        failures.append("minimum quality not met")
    regressed = [
        key
        for key in before
        if after[key]["score"] < before[key]["score"]
        or after[key]["error_cost"] > before[key]["error_cost"]
    ]
    if len(regressed) > gates["max_regressed_tasks"]:
        failures.append("individual task regressions exceed budget")
    return {
        "passed": not failures,
        "failures": failures,
        "delta": delta,
        "regressed_task_ids": regressed,
        "gates": gates,
        "dataset_sha256": baseline["dataset_sha256"],
    }
