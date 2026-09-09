"""Portable, self-contained HTML and CSV exports."""

import csv
import html
from pathlib import Path


def export_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "status",
        "eligible",
        "quality",
        "latency_p50_s",
        "latency_p95_s",
        "ttft_p50_s",
        "decode_tokens_s",
        "end_to_end_tokens_s",
        "rss_peak_gib",
        "gpu_peak_gib",
    ]
    rows = [
        {
            **{k: t[k] for k in ("name", "status", "eligible")},
            **t.get("metrics", {}),
            **t.get("memory", {}),
        }
        for t in result["trials"]
    ]
    with open(output / "results.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def fmt(x):
        return "unavailable" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)

    header = "".join(f"<th>{html.escape(k)}</th>" for k in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(row.get(k)))}</td>" for k in fields) + "</tr>"
        for row in rows
    )
    reasons = "".join(
        f"<li><b>{html.escape(t['name'])}</b>: {html.escape('; '.join(t.get('rejection_reasons', [])) or 'eligible')}</li>"
        for t in result["trials"]
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{html.escape(result["name"])}</title>
<style>body{{font:15px system-ui;margin:4vw;color:#203847;background:#f3f7fa}}h1{{font-size:34px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;background:white}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #dbe5ec;white-space:nowrap}}th{{background:#dce8f0}}small{{color:#526c7e}}</style>
<h1>{html.escape(result["name"])}</h1><p>Measured speed · memory · specialised-task quality</p>
<p><b>Frontier:</b> {html.escape(", ".join(result["frontier"]) or "No configuration met every gate.")}</p>
<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div><ul>{reasons}</ul>
<small>RSS is sampled managed-process memory. GPU values are NVIDIA process samples or an explicit CPU-only zero; unsupported telemetry stays unavailable. Apple RAM and GPU share memory. Latency includes prompt preparation and generation. Decode rate is backend-reported. This finite sweep does not establish the hardware's global limit.</small></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")
    return output / "report.html"
