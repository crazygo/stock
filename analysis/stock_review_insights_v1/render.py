"""Render price evidence and derive descriptive paths from frozen daily inputs; never fetch."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
from zoneinfo import ZoneInfo
from price_paths import METHOD, OVERLAY_METHOD, attach

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = "industry_price_evidence_v4"
DAYS = ("5", "20", "60")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Tables(HTMLParser):
    """Read displayed text, including the breadth labels separated by <br>."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.table, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("th", "td") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" | ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def section_bounds(source: str) -> tuple[int, int]:
    if 'id="industry-evidence-viz"' in source:
        raise ValueError("This page is already visualized; render from its original source report.")
    match = re.search(r"<h2\b[^>]*>\s*行业环境\s*·\s*价格证据\s*</h2>", source)
    if not match:
        raise ValueError("Industry price evidence heading was not found.")
    following = re.search(r"<h2\b|</main>", source[match.end():])
    if not following:
        raise ValueError("Industry price evidence section end was not found.")
    return match.start(), match.end() + following.start()


def price(cell: str, raw: float | None) -> dict:
    display = cell.split("|")[0].strip()
    expected = "未知" if raw is None else f"{raw * 100:+.2f}%"
    if display != expected:
        raise ValueError(f"Displayed price differs from P08 evidence: {display!r} != {expected!r}")
    value = None if display == "未知" else float(display.rstrip("%"))
    if value is not None and not math.isfinite(value):
        raise ValueError("Non-finite displayed price.")
    return {"value": value, "display": display}


def table_data(source: str, industry: dict) -> tuple[dict, str]:
    start, end = section_bounds(source)
    section = source[start:end]
    parser = Tables()
    parser.feed(section)
    if len(parser.tables) != 2:
        raise ValueError("Expected the benchmark and AI value-chain tables in the target section.")
    fund_table, group_table = parser.tables
    if not fund_table or fund_table[0] != ["宽基/基金", "5日", "20日", "60日"]:
        raise ValueError("Unexpected benchmark table columns.")
    if not group_table or group_table[0] != ["AI价值链样本", "5日等权均值 / 广度", "20日", "60日", "解释"]:
        raise ValueError("Unexpected value-chain table columns.")
    benchmarks = {r["code"]: r for r in industry["benchmarks"]}
    groups = {(r["market"], r["group_name"]): r for r in industry["groups"] if r["sample_type"] == "AI价值链样本"}
    funds, records, identities = [], [], set()
    for row in fund_table[1:]:
        if len(row) != 4 or row[0] not in benchmarks or row[0] in identities:
            raise ValueError(f"Unrecognized or duplicate benchmark table row: {row!r}")
        code, *cells = row
        evidence = benchmarks[code]
        identities.add(code)
        funds.append({"id": code, "name": code.split(".", 1)[-1], "market": evidence["market"], "n": None,
                      "values": {day: price(cell, evidence["price_returns"][day]["value"]) for day, cell in zip(DAYS, cells)}})
    for row in group_table[1:]:
        if len(row) != 5:
            raise ValueError(f"Unexpected group table row: {row!r}")
        label = row[0].split("|")[0].strip()
        market, name = (v.strip() for v in label.split("·", 1))
        evidence = groups[(market, name)]
        identity = market + "-" + evidence["group_id"]
        if identity in identities:
            raise ValueError(f"Duplicate group table row: {identity}")
        identities.add(identity)
        n = evidence["securities"]
        values = {}
        for day, cell in zip(DAYS, row[1:4]):
            raw = evidence["windows"][day]
            v = price(cell, raw["equal_weight_mean"])
            match = re.search(r"(\d+)/(\d+)上涨；未知(\d+)", cell)
            if not match:
                raise ValueError(f"Missing breadth denominator or unknown count: {cell!r}")
            up, scored, unknown = map(int, match.groups())
            if (up, scored, unknown) != (raw["positive"], raw["scored"], raw["unknown"]) or not 0 <= up <= scored or scored + unknown != n:
                raise ValueError(f"Breadth/sample count differs from P08: {identity}, {day}")
            values[day] = {**v, "up": up, "scored": scored, "unknown": unknown}
        if row[4] != evidence["interpretation"]:
            raise ValueError(f"Interpretation differs from P08: {identity}")
        records.append({"id": identity, "name": name, "market": market, "n": n, "values": values, "interpretation": row[4]})
    market_names = {"US": "美股", "HK": "港股"}
    markets = sorted({r["market"] for r in funds + records}, key=lambda m: (m != "US", m != "HK", m))
    sections = []
    for market in markets:
        rows = [r for r in funds if r["market"] == market]
        if rows:
            sections.append({"id": market + "-funds", "title": market_names.get(market, market) + "宽基与行业基金", "fund": True, "rows": rows})
    for market in markets:
        for small in (False, True):
            rows = [r for r in records if r["market"] == market and (r["n"] <= 3) == small]
            if rows:
                title = market_names.get(market, market) + ("小样本与待核实主题 · 每组 ≤3 只" if small else " AI 价值链样本")
                sections.append({"id": market + ("-small" if small else "-groups"), "title": title, "fund": False, "rows": rows})
    etfs = industry["sector_etfs"]
    data = {"schema_version": VERSION, "source_table_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "market_cutoff": industry["market_cutoffs"], "funds": funds, "groups": records, "sections": sections,
            "sector_etf_coverage": {"present": sum(bool(r["present"]) for r in etfs), "requested": len(etfs)},
            "basis": "source table displayed percentages; price change, not total return; overlapping current member samples; breadth is up/scored, not a forecast"}
    tables_html = "".join('<div class="scroll">' + t.group() + "</div>" for t in re.finditer(r"<table\b.*?</table>", section, re.S))
    return data, tables_html


def render_html(source: str, industry: dict, prices: dict | None = None,
                context: dict | None = None, value_chain: dict | None = None) -> tuple[str, dict]:
    """For an unsealed report builder: transform the HTML, preserving all other content."""
    data, tables = table_data(source, industry)
    attach(data, industry, prices, context, value_chain)
    start, end = section_bounds(source)
    cutoff = " · ".join(f"{m} 截至 {d or '未知'}" for m, d in data["market_cutoff"].items()) or "行情截止未知"
    coverage = data["sector_etf_coverage"]
    replacements = {"__CUTOFF__": html.escape(cutoff), "__ETF_COVERAGE__": f'{coverage["present"]}/{coverage["requested"]}',
                    "__SOURCE_TABLES__": tables, "__DATA__": json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"),
                    "__OVERLAY__": (HERE / "industry_overlay.html").read_text()}
    template = (HERE / "industry_price_evidence.html").read_text()
    fragment = re.sub(r"__CUTOFF__|__ETF_COVERAGE__|__SOURCE_TABLES__|__DATA__|__OVERLAY__", lambda m: replacements[m.group()], template)
    return source[:start] + fragment + source[end:], data


def validation(data: dict) -> dict:
    rows = data["funds"] + data["groups"]
    return {"status": "pass", "source_table_vs_industry_analysis": "all displayed rows, percentages, breadth counts and interpretations match",
            "funds": len(data["funds"]), "groups": len(data["groups"]), "price_cells": len(rows) * len(DAYS),
            "scored_price_cells": sum(r["values"][d]["value"] is not None for r in rows for d in DAYS),
            "breadth_cells": len(data["groups"]) * len(DAYS), "browser": "not_verified",
            "daily_path_method": METHOD, "overlay_method": OVERLAY_METHOD,
            "overlay_center_coverage": {"parents": sum(r["chart"]["overlay"]["status"] == "complete" for r in rows),
                "securities": sum(r["chart"]["overlay"]["status"] == "complete" for r in data["security_rows"].values()),
                "securities_total": len(data["security_rows"])}, "group_path_coverage": [
                {"id":r["id"], "used":r["chart"]["used"], "members":r["chart"]["members"], "status":r["chart"]["status"]}
                for r in data["groups"]]}


def load_path_inputs(path: Path) -> tuple:
    names = {"prices": next((n for n in ["base/results/trend/prices.json", "results/trend/prices.json"] if (path/n).is_file()), None),
             "context": "context.json", "value_chain": "inputs/value_chain.json"}
    values, refs = {}, {}
    for key, name in names.items():
        if name and (path/name).is_file():
            values[key] = json.loads((path/name).read_text())
            refs[key] = {"path": name, "sha256": sha(path/name)}
        else:
            values[key] = None
    if values["prices"] and values["context"]:
        if values["prices"].get("data_run_id") != values["context"].get("data_run_id"):
            raise ValueError("Daily paths and source context belong to different data runs.")
    return values, refs


def render_staging(path: Path) -> Path:
    if (path / "manifest.json").exists() or (path / "checksums.sha256").exists():
        raise ValueError("Staging render refuses sealed runs; use --from-run to create a new render view.")
    index = path / "index.html"
    industry = json.loads((path / "results/industry_analysis.json").read_text())
    values, refs = load_path_inputs(path)
    output, data = render_html(index.read_text(), industry, **values)
    data["daily_path_input_refs"] = refs
    (path / "validation").mkdir(exist_ok=True)
    write_json(path / "results/industry_table_data.json", data)
    write_json(path / "validation/industry_price_evidence.json", validation(data))
    index.write_text(output)
    return index


def render_view(source: Path, output: Path) -> Path:
    if output.exists():
        raise ValueError("Render output already exists; use a new render ID.")
    source_html = (source / "index.html").read_text()
    industry_file = source / "results/industry_analysis.json"
    values, refs = load_path_inputs(source)
    rendered, data = render_html(source_html, json.loads(industry_file.read_text()), **values)
    data["daily_path_input_refs"] = refs
    output.mkdir(parents=True)
    excluded = {"index.html", "manifest.json", "checksums.sha256", "run.json", "validation", "repro", "README.md"}
    for item in source.iterdir():
        if item.name not in excluded:
            # Relative links preserve the full report navigation without changing the archive.
            (output / item.name).symlink_to(os.path.relpath(item, output), target_is_directory=item.is_dir())
    (output / "index.html").write_text(rendered)
    write_json(output / "industry_table_data.json", data)
    (output / "validation").mkdir()
    write_json(output / "validation/industry_price_evidence.json", validation(data))
    snapshot = output / "repro/source"
    snapshot.mkdir(parents=True)
    for name in ("render.py", "price_paths.py", "industry_price_evidence.html", "industry_overlay.html", "verify_browser.cjs", "verify_overlay.cjs", "verify_overlay_math.py"):
        shutil.copy2(HERE / name, snapshot / name)
    metadata = {"schema_version": VERSION, "run_kind": "render", "run_id": output.name,
                "parent_run_ids": [source.name], "source_report_run_id": source.name,
                "source_path": os.path.relpath(source, output), "source_index_sha256": sha(source / "index.html"),
                "source_manifest_sha256": sha(source / "manifest.json"), "source_industry_analysis_sha256": sha(industry_file),
                "market_cutoffs": data["market_cutoff"], "rendered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "facts_recalculated": False, "new_descriptive_statistics": True, "daily_path_method": METHOD,
                "overlay_method": data["overlay_method"],
                "daily_path_input_refs": refs,
                "dependencies": "local links to the preserved source run; materialize before standalone sharing"}
    write_json(output / "run.json", metadata)
    (output / "README.md").write_text("# 行业价格证据展示版本\n\n本目录是 run_kind=render。沿用原分析和行情截止，没有重新取数或重新分析。原表百分比保留；从同批冻结日线新增日K、走势指标及居中同屏折线，计算口径保存于 industry_table_data.json。默认最近60交易日，0线是各自 T−30 至 T 的31个收盘价几何均值；可切换为30次日涨跌幅均值，均使用对数比值坐标。中心窗口不足时不画线，旧缺口不填补。\n\n除新首页、绘图数据与渲染源码外，其余页面和证据通过相对链接复用原封存运行。\n从单股详情返回首页进入本展示版本；独立分享前必须物化链接并重新验收。\n")
    hashes = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and not p.is_symlink() and not any(parent.is_symlink() for parent in p.parents if parent != output)}
    write_json(output / "manifest.json", {**metadata, "files": hashes, "validation": "table consistency passed; browser acceptance separate"})
    hashes["manifest.json"] = sha(output / "manifest.json")
    (output / "checksums.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
    return output / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-run", type=Path, help="Preserve a sealed report and create a new local render view.")
    mode.add_argument("--staging", type=Path, help="Replace the industry tables in an unsealed report before checksums/package creation.")
    parser.add_argument("--output", type=Path, help="New directory for --from-run; never overwrite an existing directory.")
    args = parser.parse_args()
    if args.staging:
        if args.output:
            parser.error("--output applies only to --from-run")
        index = render_staging(args.staging.resolve())
    else:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        output = args.output or ROOT / ".cache/stock_review_insights_v1/renders" / f"{stamp}__render-{uuid.uuid4()}"
        index = render_view(args.from_run.resolve(), output.resolve())
    print(json.dumps({"index": str(index), "view_version": VERSION, "facts_recalculated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
