"""Deterministic lexical context selection with source spans and omission checks."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter


def select_context(query, files, *, max_chars=12000, chunk_lines=30, required_facts=()):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("context query is required")
    if type(max_chars) is not int or not 256 <= max_chars <= 256 * 1024:
        raise ValueError("context size must be 256–262144 characters")
    if type(chunk_lines) is not int or not 1 <= chunk_lines <= 200:
        raise ValueError("chunk_lines must be 1–200")
    if (
        not isinstance(files, dict)
        or len(files) > 20
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in files.items())
    ):
        raise ValueError("context must contain at most twenty text files")
    if sum(len(v.encode()) for v in files.values()) > 256 * 1024:
        raise ValueError("context input exceeds 256 KiB")
    if (
        not isinstance(required_facts, (list, tuple))
        or len(required_facts) > 100
        or any(not isinstance(v, str) or not v for v in required_facts)
    ):
        raise ValueError("required facts must be up to 100 nonempty strings")
    started = time.perf_counter()
    terms = set(re.findall(r"[\w]+", query.casefold()))
    chunks = []
    for name, text in sorted(files.items()):
        lines = text.splitlines(keepends=True)
        for start in range(0, len(lines), chunk_lines):
            content = "".join(lines[start : start + chunk_lines])
            counts = Counter(re.findall(r"[\w]+", content.casefold()))
            chunks.append(
                {
                    "path": name,
                    "start_line": start + 1,
                    "end_line": min(start + chunk_lines, len(lines)),
                    "content": content,
                    "counts": counts,
                }
            )
    frequency = Counter(term for c in chunks for term in terms if term in c["counts"])
    for c in chunks:
        c["score"] = sum(
            math.log(1 + len(chunks) / (1 + frequency[t])) * c["counts"][t] / (c["counts"][t] + 1.2)
            for t in terms
            if t in c["counts"]
        )
    selected = []
    size = 0
    for c in sorted(chunks, key=lambda c: (-c["score"], c["path"], c["start_line"])):
        if c["score"] <= 0:
            continue
        label = f"\n--- {c['path']}:{c['start_line']}-{c['end_line']} ---\n"
        if size + len(label) + len(c["content"]) > max_chars:
            continue
        selected.append({k: v for k, v in c.items() if k != "counts"})
        size += len(label) + len(c["content"])
    selected_chunk_count = len(selected)
    # Merge only contiguous selected source ranges. Labels must neither hide a
    # fact spanning neighbouring chunks nor manufacture a fact across a gap.
    merged = []
    for chunk in sorted(selected, key=lambda c: (c["path"], c["start_line"])):
        if (
            merged
            and merged[-1]["path"] == chunk["path"]
            and merged[-1]["end_line"] + 1 == chunk["start_line"]
        ):
            merged[-1]["content"] += chunk["content"]
            merged[-1]["end_line"] = chunk["end_line"]
            merged[-1]["score"] += chunk["score"]
        else:
            merged.append(dict(chunk))
    selected = sorted(merged, key=lambda c: (-c["score"], c["path"], c["start_line"]))
    text = "".join(
        f"\n--- {c['path']}:{c['start_line']}-{c['end_line']} ---\n{c['content']}" for c in selected
    )
    omitted = [
        fact for fact in required_facts if not any(fact in span["content"] for span in selected)
    ]
    return {
        "text": text,
        "selected": selected,
        "input_chars": sum(len(v) for v in files.values()),
        "output_chars": len(text),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "source_hashes": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
        "required_facts_missing": omitted,
        "required_fact_gate": not omitted,
        "required_facts_absent_from_sources": [
            fact for fact in required_facts if not any(fact in source for source in files.values())
        ],
        "zero_relevance_chunks_skipped": sum(c["score"] <= 0 for c in chunks),
        "unselected_relevant_chunks": sum(c["score"] > 0 for c in chunks) - selected_chunk_count,
        "note": "Lexical retrieval can omit relevant facts. No model is called; source spans and optional known-fact checks are retained.",
    }
