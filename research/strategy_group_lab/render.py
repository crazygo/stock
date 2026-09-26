"""Self-contained, offline research comparison report."""
from __future__ import annotations
import json
from pathlib import Path

def render(output, cfg, results, details, router, summary):
    payload = {"config": cfg, "results": results, "details": details, "router": router, "summary": summary,
               "availability": json.loads((output/"availability.json").read_text())}
    template = (Path(__file__).parent/"report.html").read_text()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output/"index.html").write_text(template.replace("__PAYLOAD__", data), encoding="utf-8")
