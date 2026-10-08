"""Validate scoped agent reviews and merge industry evidence without inventing grades."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json
from urllib.parse import urlparse

OUT = Path(__file__).resolve().parent
WORK = OUT / "label_work"


def main():
    universe = json.loads((WORK / "universe.json").read_text())
    expected = {r["code"] for r in universe}
    sectors = json.loads((WORK / "sector_evidence.json").read_text())
    records, errors, jobs = {}, [], []
    for filename in ["review_remaining.json", "review_priority_1.json", "review_priority_2.json", "review_luna.json",
                     "review_tech_expansion_1.json", "review_tech_expansion_2.json", "review_cross_sector.json",
                     "review_final_gaps.json", "review_sec_tech_gaps.json", "review_med_fin_gaps.json"]:
        path = WORK / filename
        if not path.exists():
            jobs.append(dict(file=filename, status="not_completed"))
            continue
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("records", [])
        input_name = filename.removeprefix("review_")
        input_path = WORK / input_name
        declared = {"review_luna.json": 27, "review_cross_sector.json": 60,
                    "review_final_gaps.json": 30}.get(filename)
        if input_path.exists():
            declared = len(json.loads(input_path.read_text()))
        jobs.append(dict(file=filename, status="completed" if declared is None or len(rows) == declared else "partial",
                         records=len(rows), expected=declared))
        for raw in rows:
            code = raw.get("code")
            if code not in expected:
                errors.append(dict(code=code, reason="unknown or duplicate code"))
                continue
            if code in records and records[code].get("tier") is not None and raw.get("tier") is None:
                continue
            if code in records and records[code].get("tier") is not None and raw.get("tier") is not None:
                errors.append(dict(code=code, reason="duplicate graded code"))
                continue
            item = dict(raw)
            sources = item.get("sources") or []
            sources = [s for s in sources if isinstance(s, dict) and
                       urlparse(s.get("url", "")).scheme in ("http", "https")]
            item["sources"] = sources
            status = item.get("review_status", "unresolved")
            tier = item.get("tier")
            if tier not in (0, 1, 2, 3, None):
                errors.append(dict(code=code, reason="invalid grade"))
                item["tier"] = None
            # Issuer industry fields verify industries, never AI pathways.
            business_sources = [s for s in sources if s.get("status") not in ("issuer_sector_field", "prior_sec_profile")
                                and not any(host in urlparse(s["url"]).netloc
                                    for host in ("ishares.com", "globalxetfs.com"))]
            if tier is not None and (status != "official_review" or not business_sources):
                item["proposed_tier"] = tier
                item["tier"] = None
                item["review_status"] = "industry_screened" if code in sectors else "unresolved"
                errors.append(dict(code=code, reason="grade withheld without company evidence"))
            records[code] = item
    # Preserve incomplete reviews explicitly; a missing output never becomes zero.
    for r in universe:
        code = r["code"]
        if code in records:
            continue
        sector = sectors.get(code, {})
        records[code] = dict(code=code, name=sector.get("name", r["name"]),
            industry=sector.get("industry"), primary=r["current_business"]["primary"],
            groups=r["current_business"]["groups"], tier=None,
            reasons=["行业资料已取得；AI 商业路径尚无逐公司核查结果。"],
            sources=sector.get("sources", [])[:1],
            review_status="industry_screened" if sector.get("industry") else "unresolved",
            confidence="industry_evidence_only" if sector.get("industry") else "unreviewed")
    adjustments = []
    for code, item in records.items():
        if not item.get("industry") and code in sectors:
            item["industry"] = sectors[code].get("industry")
        # Keep AI users distinct from suppliers of computing chips. This is a
        # taxonomy review, not a change to the source-backed relevance grade.
        before = list(item.get("groups", []))
        primary = item.get("primary")
        if code == "US.TER" and item.get("tier") is not None:
            item["primary"], item["groups"] = "design", ["design", "edge"]
        else:
            remove = set()
            if primary in ("health", "cloud", "power", "design") or code in ("US.KDK", "US.KOPN", "US.AIP"):
                remove.add("compute")
            if primary in ("health", "edge", "design", "power"):
                remove.add("applications")
            item["groups"] = [g for g in before if g not in remove]
        if item.get("groups") != before:
            adjustments.append(dict(code=code, before=before, after=item["groups"],
                reason="产业链角色校正：AI 使用者不当作计算芯片供应商；医疗/工业产品保留专属应用群。评级与业务证据不变。"))
    counts = Counter(r.get("review_status", "unresolved") for r in records.values())
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(),
            agent="Codex 6 Luna；agyd 初筛运行因重复读取而停止", records=records,
        industry_evidence=sectors,
        summary=dict(total=len(records), industry_named=sum(bool(r.get("industry")) for r in records.values()),
            statuses=dict(counts), agent_graded=sum(r.get("tier") is not None for r in records.values())),
        jobs=jobs, validation_errors=errors, taxonomy_adjustments=adjustments,
        method="发行人行业字段 + Codex 6 Luna 逐股官方业务资料；初筛和已验证业务证据分开。")
    (OUT / "reviewed_labels.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": result["summary"], "jobs": jobs, "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
