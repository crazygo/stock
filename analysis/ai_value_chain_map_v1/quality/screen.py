#!/usr/bin/env python3
"""Build a deterministic, evidence-gated AI research-priority screen.

This is a triage layer, not a business-quality rating or return forecast. It
reuses current value-chain labels, official sources, the formal quality record,
and the read-only financial snapshot; it performs no network access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parents[1]
POOL_PATH = PROJECT / "data.json"
QUALITY_PATH = HERE / "current.json"
FINANCIALS_PATH = HERE / "financial_snapshot.json"
CURRENT_PATH = HERE / "screen_current.json"
CSV_PATH = HERE / "screen_current.csv"
RULE_VERSION = "ai_research_screen_v1"
ALLOWED_BUSINESS_STATUS = {"official_review", "prior_business_review", "reviewed", "luna_official_review"}
ALLOWED_SOURCE_STATUS = {"luna_official_review", "prior_official_snapshot", "reviewed_this_run", "official_review", "reviewed"}
DIRECT_GROUPS = {"design", "compute", "memory", "connect", "optics", "cloud"}
ENABLER_GROUPS = {"systems", "power", "data", "health", "edge", "applications"}
EARLY_STAGE_TERMS = ("尚未证实收入", "收入尚未证实", "不等于收入兑现", "不证明规模化收入", "仍需检验订单规模", "订单规模仍待验证", "商业化尚未兑现", "产品关联不证明规模", "尚待验证", "尚未验证规模", "收入规模待验证", "产品验证但未规模化", "临床前", "试点阶段")
GENERIC_REASON_TERMS = ("已提出业务分类", "尚未完成逐公司核查", "ai 关联等级尚未完成", "已提出业务分类，ai 关联等级尚未")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def parse_day(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def taxonomy_groups() -> list[dict[str, Any]]:
    path = PROJECT / "taxonomy.py"
    spec = importlib.util.spec_from_file_location("ai_value_chain_taxonomy_for_screen", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load taxonomy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [{"id": row[0], "label": row[1], "order": row[2], "product_chain": row[3], "research_value": row[4]} for row in module.GROUPS]


def _approved_source(source: Any, as_of: date) -> tuple[bool, str | None]:
    if not isinstance(source, dict):
        return False, "来源不是对象"
    url = source.get("url")
    if not isinstance(url, str) or urlsplit(url).scheme not in {"https", "http"} or not urlsplit(url).netloc:
        return False, "来源URL无效"
    if source.get("status") not in ALLOWED_SOURCE_STATUS:
        return False, "来源缺少已核查/官方来源状态"
    observed = parse_day(source.get("observed_at"))
    if observed is None:
        return False, "来源实际观察日期未知"
    if observed > as_of:
        return False, "来源观察日期晚于筛选截止日"
    if (as_of - observed).days > 90:
        return False, "来源观察日期超过90日"
    return True, None


def _fresh_sources(business: dict[str, Any], as_of: date) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    failures: list[str] = []
    for source in business.get("sources", []) if isinstance(business.get("sources"), list) else []:
        valid, reason = _approved_source(source, as_of)
        if valid:
            # Keep source dates/status exactly as received; never backfill them.
            kept.append({k: source.get(k) for k in ("url", "title", "observed_at", "status", "evidence", "date_basis") if k in source})
        elif reason:
            failures.append(reason)
    unique: dict[str, dict[str, Any]] = {}
    for source in kept:
        unique[source["url"]] = source
    return list(unique.values()), sorted(set(failures))


def _valid_business(row: dict[str, Any], as_of: date, group_ids: set[str]) -> tuple[bool, list[str], list[str], list[dict[str, Any]]]:
    business = row.get("business") or {}
    missing: list[str] = []
    tier = business.get("tier")
    if tier not in (1, 2, 3):
        missing.append("缺少已确认的AI业务关联等级")
    if business.get("review_status") not in ALLOWED_BUSINESS_STATUS:
        missing.append("业务标签缺少已完成核查标志")
    reasons = [x.strip() for x in business.get("reasons", []) if isinstance(x, str) and len(x.strip()) >= 12]
    reasons = [x for x in reasons if not any(term in x.casefold() for term in GENERIC_REASON_TERMS)]
    if not reasons:
        missing.append("缺少具体业务与AI传导路径说明")
    groups = business.get("groups", []) if isinstance(business.get("groups"), list) else []
    primary = business.get("primary")
    valid_groups = [x for x in groups if isinstance(x, str) and x in group_ids]
    if primary not in group_ids and not valid_groups:
        missing.append("缺少taxonomy.py中有效的业务群组")
    sources, source_failures = _fresh_sources(business, as_of)
    if not sources:
        missing.append("缺少90日内实际观察的已核查官方业务来源")
        missing.extend(source_failures[:3])
    return not missing, reasons, sorted(set(missing)), sources


def _financial_view(fin_snapshot: dict[str, Any] | None, code: str, as_of: date) -> tuple[str, dict[str, Any], dict[str, Any]]:
    raw = (fin_snapshot or {}).get("records", {}).get(code)
    if not raw:
        freshness = {"status": "unknown", "source_capture_at": None, "latest_period_end": None, "latest_filed": None, "age_days": None, "reason": "财务记录缺失"}
        return "unknown", freshness, {"metrics": {}, "derived": {}}
    capture = parse_day(raw.get("source_capture_at"))
    period_end = parse_day(raw.get("latest_period_end"))
    filed = parse_day(raw.get("latest_filed"))
    age = (as_of - period_end).days if period_end else None
    status, reason = "fresh", None
    if capture is None:
        status, reason = "source_unknown", "原始缓存取得日期未保留；未以复制/文件时间代替"
    elif capture > as_of:
        status, reason = "future_source", "原始缓存取得日晚于筛选截止日"
    elif (as_of - capture).days > 30:
        status, reason = "stale_source", "原始缓存取得日期超过30日"
    elif period_end is None:
        status, reason = "period_unknown", "最新财务期末未知"
    elif period_end > as_of:
        status, reason = "future_period", "财务期末晚于筛选截止日"
    elif age is not None and age > 210:
        status, reason = "stale_period", "最新财务期末超过210日"
    elif filed is None:
        status, reason = "filing_unknown", "最近申报日期未知"
    elif filed > as_of:
        status, reason = "future_filing", "财务申报日期晚于筛选截止日"
    freshness = {"status": status, "source_capture_at": raw.get("source_capture_at"), "latest_period_end": raw.get("latest_period_end"), "latest_filed": raw.get("latest_filed"), "age_days": age, "reason": reason}
    metric_names = ("revenue", "gross_profit", "operating_income", "net_income", "operating_cashflow", "capex", "cash", "total_debt", "assets", "diluted_shares", "stock_compensation")
    metrics: dict[str, Any] = {}
    for name in metric_names:
        slot = raw.get("metrics", {}).get(name, {})
        selected = slot.get("latest") if name in {"cash", "total_debt", "assets"} else (slot.get("ttm") or slot.get("annual"))
        if not selected:
            metrics[name] = None
            continue
        metrics[name] = {k: selected.get(k) for k in ("value", "unit", "period", "start", "end", "filed", "tag", "source_url", "formula", "caveat", "coverage")}
        if not metrics[name].get("source_url"):
            components = selected.get("components") or []
            if components:
                metrics[name]["source_url"] = components[0].get("source_url")
    derived = {k: (raw.get("derived", {}).get(k) if isinstance(raw.get("derived", {}).get(k), dict) else None) for k in ("revenue_yoy", "net_margin", "fcf", "fcf_margin", "cash_runway_months")}
    derived = {k: ({a: v.get(a) for a in ("value", "unit", "formula", "period_end")} if v else None) for k, v in derived.items()}
    return raw.get("status", "unknown"), freshness, {"metrics": metrics, "derived": derived}


def _specific_direct_support(text: str) -> bool:
    terms = ("数据中心专用", "为数据中心提供", "支持数据中心建设", "数据中心供电", "数据中心配电", "AI数据中心", "AI 数据中心", "AI机柜", "AI 机柜", "AI集群", "AI 集群", "GPU云", "GPU 云")
    value = text.casefold()
    return any(term.casefold() in value for term in terms)


def _value_role(primary: str | None, groups: list[str], tier: int, reasons: list[str]) -> str:
    path = primary if primary in groups else (groups[0] if groups else None)
    evidence = " ".join(reasons)
    if path in DIRECT_GROUPS:
        return "core"
    if tier == 3 and path in ENABLER_GROUPS:
        # Tier 3 is already an explicit AI product/mechanism or dedicated supply label.
        return "core"
    if path in ENABLER_GROUPS:
        return "enabler"
    if path in {"energy", "adjacent"}:
        return "enabler" if _specific_direct_support(evidence) else "indirect"
    if path in {"other", "pending"}:
        return "indirect" if tier == 1 else "unknown"
    return "unknown"


def _early_or_unproven(business: dict[str, Any], quality: dict[str, Any], reasons: list[str], as_of: date) -> bool:
    reviewed_at = parse_day(quality.get("business_reviewed_at"))
    quality_judgment_current = (
        quality.get("review_status") == "reviewed"
        and reviewed_at is not None
        and reviewed_at <= as_of
        and (as_of - reviewed_at).days <= 90
    )
    if quality_judgment_current and (quality.get("stage") == "precommercial" or quality.get("commercial") == 1):
        return True
    text = " ".join(reasons)
    return any(term in text for term in EARLY_STAGE_TERMS)


def _initial_grade(tier: int, role: str, quality: dict[str, Any], business: dict[str, Any], reasons: list[str], finance_support: str, confidence: Any, as_of: date) -> tuple[str | None, list[str]]:
    basis = [f"business.tier={tier}；该字段是AI业务关联等级，不是上涨概率或AI收入占比"]
    if role == "unknown":
        return None, basis + ["无法把业务标签映射到明确价值链角色"]
    if tier == 1 or role == "indirect":
        return "B", basis + [f"价值链角色={role}；属于间接/外围传导，保留研究记录"]
    early = _early_or_unproven(business, quality, reasons, as_of)
    if confidence == "low":
        if tier == 1:
            return "B", basis + ["业务标签置信度低，按低优先级观察"]
        return "A-", basis + ["业务标签置信度低；未核查初筛最高限制为A−"]
    if tier == 3 and role == "core":
        if early:
            return "A-", basis + ["tier 3有直接产品/专用供给路径，但来源明示商业规模或兑现仍早期"]
        return "A", basis + ["tier 3已标注直接AI产品机制或专用供给，列为高研究优先级；不代表护城河/收入兑现"]
    if tier == 3 and role == "enabler":
        return ("A-" if early else "A"), basis + ["tier 3的专用部署支持路径；研究优先级不代表实际价值捕获"]
    if tier == 2 and role == "core":
        if finance_support in {"strong", "sound"}:
            return "A", basis + [f"tier 2核心传导路径且正式财务支持为{finance_support}"]
        return "A-", basis + ["tier 2核心传导路径；财务未知/未达strong不降低业务路径优先级，但需后续核实"]
    if tier == 2 and role == "enabler":
        return "A-", basis + ["tier 2明确部署配套路径，作为研究观察，不将其等同AI专属收入"]
    return "B", basis + [f"业务路径有限或间接（role={role}）"]


def _verified_grade(quality: dict[str, Any], quality_snapshot: dict[str, Any] | None, as_of: date) -> str | None:
    if not isinstance(quality_snapshot, dict) or quality_snapshot.get("preview") is not False:
        return None
    snapshot_day = parse_day(quality_snapshot.get("as_of"))
    if snapshot_day is None or snapshot_day > as_of:
        return None
    grade = quality.get("grade")
    if grade not in {"A+", "A", "A-", "B"} or quality.get("decision") != "current":
        return None
    expires = parse_day(quality.get("expires_at"))
    return grade if expires and expires >= as_of else None


def _record(row: dict[str, Any], quality: dict[str, Any], fin_snapshot: dict[str, Any] | None, groups: list[dict[str, Any]], as_of: date, valid_group_ids: set[str]) -> dict[str, Any]:
    code = row.get("code")
    business = row.get("business") or {}
    valid, reasons, missing, sources = _valid_business(row, as_of, valid_group_ids)
    value_groups = list(dict.fromkeys(g for g in business.get("groups", []) if g in valid_group_ids))
    primary = business.get("primary") if business.get("primary") in valid_group_ids else (value_groups[0] if value_groups else None)
    tier = business.get("tier")
    role = _value_role(primary, value_groups, tier if tier in (1, 2, 3) else 0, reasons) if valid else "unknown"
    raw_fin_status, financial_freshness, financial_metrics = _financial_view(fin_snapshot, code, as_of)
    historical_finance_support = quality.get("financial_support", "unknown")
    finance_support = historical_finance_support if financial_freshness.get("status") == "fresh" else "unknown"
    # Formal quality is gated centrally in build(), which inspects the
    # snapshot-level preview/as_of metadata.
    formal_grade = None
    initial_basis: list[str] = []
    if valid:
        grade, initial_basis = _initial_grade(tier, role, quality, business, reasons, finance_support, business.get("confidence"), as_of)
    else:
        grade = None
    if business.get("confidence") == "low" and grade not in (None, "B") and grade == "A":
        grade = "A-"
        initial_basis.append("低置信业务标签对未核查初筛等级设A−上限")
    verified_grade = formal_grade
    if verified_grade:
        grade = verified_grade
        status = "verified"
        initial_basis = ["正式quality/current.json中同截止日、未过期且decision=current的已核查等级"]
    else:
        status = "screened" if grade is not None else "needs_evidence"
    source_days = [parse_day(s.get("observed_at")) for s in sources]
    source_days = [x for x in source_days if x]
    expires = as_of + timedelta(days=30)
    if source_days:
        expires = min(expires, min(source_days) + timedelta(days=90))
    if verified_grade and parse_day(quality.get("expires_at")):
        expires = min(expires, parse_day(quality.get("expires_at")))
    limitations = ["这是AI业务研究优先级初筛，不是公司质量/投资建议/上涨空间/未来收益评级。", "价值链位置不证明AI收入占比、护城河、订单持续性或价值最终由本公司取得。"]
    if verified_grade is None:
        limitations.append("尚无同截止日有效的正式质量等级；A/A−/B为初筛工作顺序，未核查初筛最高A，不生成A+。")
    if financial_freshness.get("status") != "fresh":
        limitations.append("财务支持不能作为当前财务质量结论；原始取得时间/期间未知或已过时会明确保留。")
    if business.get("confidence") == "low":
        limitations.append("原业务标签置信度低；已显式标记并限制未核查等级。")
    next_actions = []
    if status == "needs_evidence":
        next_actions.append("补齐可复查、日期有效的官方业务来源及具体AI传导说明")
    if status == "screened":
        next_actions.append("核实商业化、客户集中、竞争替代及资金风险后再进入正式quality评级")
    if financial_freshness.get("status") != "fresh":
        next_actions.append("核实财务核心期间和原始SEC缓存取得时间；不要用文件复制时间代替")
    if verified_grade:
        next_actions.append("按正式quality记录的到期日/触发条件复核")
    group_lookup = {g["id"]: g for g in groups}
    if valid:
        priority_reason = f"tier {tier}业务路径，经{group_lookup[primary]['label'] if primary in group_lookup else primary}传导，定位为{role}；该档仅表示研究优先级。"
    else:
        priority_reason = "当前业务/官方证据未通过初筛门槛，暂不推断研究等级。"
    industry = row.get("industry") or business.get("industry") or "未分类"
    industry_detail = f"{industry} · 价值链：{group_lookup[primary]['label'] if primary in group_lookup else '待核'}（非GICS行业细分）"
    return {
        "code": code, "name": row.get("name"), "tier": tier,
        "grade": grade, "status": status, "as_of": as_of.isoformat(), "expires_at": expires.isoformat(),
        "industry": industry, "industry_detail": industry_detail,
        "value_groups": value_groups, "primary_group": primary, "value_role": role,
        "business_evidence": reasons, "business_sources": sources,
        "financial_support": finance_support, "financial_freshness": financial_freshness,
        "financial_metrics": financial_metrics,
        "verified_grade": verified_grade, "initial_basis": initial_basis,
        "limitations": limitations, "missing": missing,
        "next_actions": list(dict.fromkeys(next_actions)), "priority_reason": priority_reason,
        "confidence_low": business.get("confidence") == "low",
        "business_confidence": business.get("confidence"),
        "business_review_status": business.get("review_status"),
        "business_reviewed_at": quality.get("business_reviewed_at"),
        "formal_quality_as_of": quality.get("as_of") if verified_grade else None,
        "historical_financial_support": historical_finance_support,
        "financial_record_status": raw_fin_status,
    }


def _pool_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        b = row.get("business") or {}
        out.append({"code": row.get("code"), "kind": row.get("kind"), "name": row.get("name"), "industry": row.get("industry"), "business": {k: b.get(k) for k in ("primary", "groups", "tier", "reasons", "sources", "confidence", "review_status", "industry")}})
    return out


def _quality_projection(q: dict[str, Any] | None, codes: list[str]) -> dict[str, Any]:
    q = q or {}
    return {"as_of": q.get("as_of"), "run_id": q.get("run_id"), "rule_version": q.get("rule_version"), "preview": q.get("preview"), "records": {code: {k: (q.get("records", {}).get(code, {}) or {}).get(k) for k in ("grade", "decision", "expires_at", "financial_support", "business_reviewed_at", "review_status", "stage", "commercial")} for code in codes}}


def _financial_projection(f: dict[str, Any] | None, codes: list[str]) -> dict[str, Any]:
    f = f or {}
    metric_fields = ("revenue", "gross_profit", "operating_income", "net_income", "operating_cashflow", "capex", "cash", "total_debt", "assets", "diluted_shares", "stock_compensation")
    fact_fields = ("value", "unit", "period", "start", "end", "filed", "tag", "source_url", "formula", "caveat", "coverage", "received_at", "cache_sha256")
    records = {}
    for code in codes:
        raw = f.get("records", {}).get(code, {}) or {}
        financials = raw.get("financials", raw)
        records[code] = {"status": financials.get("status"), "source_capture_at": financials.get("source_capture_at"), "latest_period_end": financials.get("latest_period_end"), "latest_filed": financials.get("latest_filed"), "metrics": {name: {slot: ({k: v.get(k) for k in fact_fields} if isinstance(v, dict) else None) for slot, v in (financials.get("metrics", {}).get(name, {}) or {}).items()} for name in metric_fields}, "derived": financials.get("derived"), "sources": [{k: s.get(k) for k in ("url", "cik", "received_at", "cache_sha256")} for s in financials.get("sources", []) if isinstance(s, dict)], "caveats": financials.get("caveats", [])}
    return {"as_of": f.get("as_of"), "records": records}


def semantic_input_hashes(rows: list[dict[str, Any]], quality: dict[str, Any] | None, financials: dict[str, Any] | None) -> dict[str, str]:
    """Public deterministic hash helper for builders/UI invalidation checks."""
    codes = [r.get("code") for r in rows]
    return {
        "membership_business_sha256": canonical_hash(_pool_projection(rows)),
        "quality_current_sha256": canonical_hash(_quality_projection(quality, codes)),
        "financial_snapshot_sha256": canonical_hash(_financial_projection(financials, codes)),
        "screen_code_sha256": sha256_file(Path(__file__)),
        "taxonomy_sha256": sha256_file(PROJECT / "taxonomy.py"),
    }


def build(as_of: date | None = None, pool_path: Path = POOL_PATH, quality_path: Path = QUALITY_PATH, financials_path: Path = FINANCIALS_PATH) -> dict[str, Any]:
    as_of = as_of or date.today()
    pool = read_json(pool_path)
    rows = [r for r in pool.get("rows", []) if r.get("kind") == "STOCK" and r.get("business", {}).get("tier") in (1, 2, 3)]
    rows.sort(key=lambda r: r.get("code", ""))
    codes = [r["code"] for r in rows]
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate security code in initial AI-labelled pool")
    quality = read_json(quality_path) if quality_path.exists() else None
    financials = read_json(financials_path) if financials_path.exists() else None
    tax_groups = taxonomy_groups()
    group_ids = {g["id"] for g in tax_groups}
    input_hashes = semantic_input_hashes(rows, quality, financials)
    seed = {"rule_version": RULE_VERSION, "as_of": as_of.isoformat(), "input_hashes": input_hashes}
    run_id = "screen_" + canonical_hash(seed)[:16]
    records: dict[str, Any] = {}
    for row in rows:
        code = row["code"]
        q = (quality or {}).get("records", {}).get(code, {}) or {}
        item = _record(row, q, financials, tax_groups, as_of, group_ids)
        # Formal status depends on the whole current quality snapshot, not only a row.
        formal = _verified_grade(q, quality, as_of)
        if formal and item["business_evidence"] and item["business_sources"] and not item["missing"]:
            item["verified_grade"] = formal
            item["grade"] = formal
            item["status"] = "verified"
            item["formal_quality_as_of"] = quality.get("as_of")
            item["initial_basis"] = [f"正式quality/current.json（核查日{q.get('as_of')}，截至筛选日仍未过期且decision=current）中的已核查等级"]
            item["limitations"] = [x for x in item["limitations"] if "尚无同截止日有效" not in x]
            item["next_actions"] = [x for x in item["next_actions"] if "进入正式quality评级" not in x]
            item["next_actions"] = list(dict.fromkeys(item["next_actions"] + ["按正式quality记录的原核查日期、到期日及触发条件复核"]))
            q_expiry = parse_day(q.get("expires_at"))
            if q_expiry:
                item["expires_at"] = min(parse_day(item["expires_at"]), q_expiry).isoformat()
        records[code] = item
    grade_keys = ("A+", "A", "A-", "B")
    grade_counts = {k: sum(1 for r in records.values() if r["grade"] == k) for k in grade_keys}
    grade_counts["ungraded"] = sum(1 for r in records.values() if r["grade"] is None)
    status_counts = {k: sum(1 for r in records.values() if r["status"] == k) for k in ("verified", "screened", "needs_evidence")}
    industry_count: dict[str, int] = {}
    industry_detail_count: dict[str, int] = {}
    business_evidence_count = 0
    financial_known = financial_unknown = 0
    group_summary: dict[str, Any] = {}
    for row in records.values():
        industry_count[row["industry"]] = industry_count.get(row["industry"], 0) + 1
        industry_detail_count[row["industry_detail"]] = industry_detail_count.get(row["industry_detail"], 0) + 1
        business_evidence_count += bool(row["business_evidence"] and row["business_sources"])
        if row["financial_support"] in {"strong", "sound", "developing", "strained"}:
            financial_known += 1
        else:
            financial_unknown += 1
    for group in tax_groups:
        members = [r for r in records.values() if group["id"] in r["value_groups"]]
        group_summary[group["id"]] = {**group, "count": len(members), "tier3_count": sum(1 for r in members if r.get("tier") == 3), "financial_known_count": sum(1 for r in members if r["financial_support"] in {"strong", "sound", "developing", "strained"}), "grade_counts": {**{k: sum(1 for r in members if r["grade"] == k) for k in grade_keys}, "ungraded": sum(1 for r in members if r["grade"] is None)}, "non_additive": True, "interpretation": "证券可同时属于多个业务群；计数是证券覆盖数，不是收入、市场份额或价值捕获比例。"}
    total = len(records)
    summary = {"pool_count": total, "security_count": total, "graded_count": total - grade_counts["ungraded"], "coverage": (total - grade_counts["ungraded"]) / total if total else 0.0, "verified_count": status_counts["verified"], "screened_count": status_counts["screened"], "needs_evidence_count": status_counts["needs_evidence"], "status_counts": status_counts, "grade_counts": grade_counts, "industry_count": dict(sorted(industry_count.items())), "industry_detail_count": dict(sorted(industry_detail_count.items())), "confidence_low_count": sum(1 for r in records.values() if r["confidence_low"]), "business_evidence_count": business_evidence_count, "group_counts_non_additive": {k: v["count"] for k, v in group_summary.items()}, "financial_known": financial_known, "financial_unknown": financial_unknown, "as_of": as_of.isoformat(), "interpretation": "研究优先级初筛覆盖率；不是公司质量合格率、上涨概率、预期收益或AI收入份额。"}
    return {"rule_version": RULE_VERSION, "rating_type": "research_priority", "as_of": as_of.isoformat(), "run_id": run_id, "preview": False, "summary": summary, "groups": group_summary, "input_hashes": input_hashes, "records": records}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, data: dict[str, Any]) -> None:
    fields = ("code", "name", "grade", "status", "verified_grade", "industry", "industry_detail", "value_role", "primary_group", "value_groups", "business_confidence", "confidence_low", "financial_support", "financial_freshness", "latest_period_end", "latest_filed", "source_capture_at", "expires_at", "priority_reason", "initial_basis", "missing", "next_actions", "business_evidence", "business_sources")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for code, record in sorted(data["records"].items()):
            row = next((x for x in data.get("_input_rows", []) if x.get("code") == code), {})
            fin = record.get("financial_freshness", {})
            writer.writerow({"code": code, "name": row.get("name"), "grade": record.get("grade") or "待初筛", "status": record.get("status"), "verified_grade": record.get("verified_grade"), "industry": record.get("industry"), "industry_detail": record.get("industry_detail"), "value_role": record.get("value_role"), "primary_group": record.get("primary_group"), "value_groups": "|".join(record.get("value_groups", [])), "business_confidence": record.get("business_confidence"), "confidence_low": record.get("confidence_low"), "financial_support": record.get("financial_support"), "financial_freshness": fin.get("status"), "latest_period_end": fin.get("latest_period_end"), "latest_filed": fin.get("latest_filed"), "source_capture_at": fin.get("source_capture_at"), "expires_at": record.get("expires_at"), "priority_reason": record.get("priority_reason"), "initial_basis": "；".join(record.get("initial_basis", [])), "missing": "；".join(record.get("missing", [])), "next_actions": "；".join(record.get("next_actions", [])), "business_evidence": "；".join(record.get("business_evidence", [])), "business_sources": " | ".join(s.get("url", "") for s in record.get("business_sources", []))})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    ap.add_argument("--output", type=Path, default=CURRENT_PATH)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    args = ap.parse_args()
    result = build(args.as_of)
    # Read the input name map only for CSV output; it does not affect the JSON schema/hash.
    pool = read_json(POOL_PATH)
    result["_input_rows"] = [{"code": r.get("code"), "name": r.get("name")} for r in pool.get("rows", []) if r.get("kind") == "STOCK" and r.get("business", {}).get("tier") in (1, 2, 3)]
    run_dir = HERE / "screen_runs" / result["run_id"]
    snapshot = run_dir / "snapshot.json"
    immutable = dict(result)
    immutable.pop("_input_rows", None)
    immutable["preview"] = False
    if snapshot.exists():
        old = read_json(snapshot)
        if canonical_hash(old) != canonical_hash(immutable):
            raise FileExistsError(f"Refusing to overwrite changed immutable screen snapshot: {snapshot}")
    else:
        write_json(snapshot, immutable)
    write_json(args.output, immutable)
    write_csv(args.csv, result)
    print(json.dumps({"run_id": result["run_id"], "as_of": result["as_of"], "summary": result["summary"], "output": str(args.output), "csv": str(args.csv), "snapshot": str(snapshot)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
