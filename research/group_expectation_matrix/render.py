"""Render the reusable low-fidelity, fully offline matrix wireframe."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render(run: Path):
    bundle = json.loads((run/"bundle.json").read_text())
    insight_path = run/"model_insights.json"
    bundle["insights"] = json.loads(insight_path.read_text()) if insight_path.exists() else None
    template = (HERE/"wireframe.html").read_text()
    data = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    result = template.replace("/*__BUNDLE__*/null", data)
    (run/"index.html").write_text(result, encoding="utf-8")
    paths = [run/"bundle.json", HERE/"wireframe.html", HERE/"render.py"]
    if insight_path.exists():
        paths.append(insight_path)
    provenance = {"rendered_at": datetime.now(timezone.utc).isoformat(),
                  "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
                  "html_sha256": hashlib.sha256(result.encode()).hexdigest(),
                  "has_independent_model_insights": insight_path.exists()}
    (run/"render_manifest.json").write_text(json.dumps(provenance, indent=2))
    print(run/"index.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    render(args.run.resolve())
