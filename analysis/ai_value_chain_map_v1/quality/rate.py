#!/usr/bin/env python3
"""Deterministic implementation of ai_quality_v1; preview is the default.

No web access, price data, model output, or inferred AI revenue is used here.
The only business judgments are reviewed records in quality/reviews_*.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parents[1]
PROTOCOL_PATH = ROOT / ".agents/skills/ai-stock-rating/references/protocol.md"
POOL_PATH = PROJECT / "data.json"
FINANCIALS_PATH = HERE / "financial_snapshot.json"
CURRENT_PATH = HERE / "current.json"
CSV_PATH = HERE / "current.csv"
RULE_VERSION = "ai_quality_v1"
GRADE_RANK = {None: 0, "B": 1, "A-": 2, "A": 3, "A+": 4}
ALLOWED_STAGES = {"mature", "scaling", "precommercial"}
SEVERE_TERMS = ("fraud", "default", "bankruptcy", "insolvency", "going concern", "restatement", "delisting", "欺诈", "违约", "破产", "资不抵债", "持续经营", "财务重述", "退市", "重大监管处罚")
FINANCIAL_SECTORS = ("bank", "insurance", "financial", "银行", "保险", "金融")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_day(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def load_reviews(paths: list[Path], as_of: date, pool_codes: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Choose the strongest/latest review deterministically; keep every source file in run hash."""
    chosen: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    for path in sorted(paths, key=lambda p: p.name):
        raw = read_json(path)
        records = raw.get("records", raw) if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            continue
        for r in records:
            if not isinstance(r, dict) or r.get("code") not in pool_codes:
                continue
            code = r["code"]
            observed_dates = [d for s in r.get("sources", []) if isinstance(s, dict) and (d := parse_day(s.get("observed_at"))) is not None]
            observed = parse_day(r.get("observed_at")) or max(observed_dates, default=None)
            explicit_priority = int(r.get("review_priority", 1))
            file_priority = 2 if path.name in {"reviews_core_top.json", "reviews_core_other.json"} else 1
            rank = (max(explicit_priority, file_priority), observed.isoformat() if observed else "", int(r.get("risk_reviewed") is True), int(r.get("review_status") in ("reviewed", "official_review")), len(r.get("sources", [])), path.name)
            previous = chosen.get(code)
            if previous is None or rank > previous["_rank"]:
                chosen[code] = {**r, "_rank": rank}
                origins[code] = path.name
    return chosen, origins


def _fact_value(financials: dict[str, Any], metric: str) -> dict[str, Any] | None:
    entry = financials.get("metrics", {}).get(metric, {})
    return entry.get("ttm") or entry.get("annual")


def _same_period(*facts: dict[str, Any] | None) -> bool:
    if not facts or any(not x for x in facts):
        return False
    return len({(x.get("start"), x.get("end"), x.get("unit")) for x in facts}) == 1


def finance_assessment(snapshot: dict[str, Any] | None, code: str, as_of: date, industry: str = "") -> dict[str, Any]:
    reasons: list[str] = []
    missing: list[str] = []
    finance = snapshot.get("records", {}).get(code) if isinstance(snapshot, dict) else None
    if not finance:
        return {"support": "unknown", "reasons": [], "missing": ["缺少该证券的SEC财务记录"], "periods": {}, "selected": {}, "compact": _compact_financials(None)}
    compact = _compact_financials(finance)
    snapshot_day = parse_day(snapshot.get("as_of"))
    if snapshot_day != as_of:
        missing.append("财务快照as_of与评级截止日不一致")
    if any(term in (industry or "").casefold() for term in FINANCIAL_SECTORS):
        missing.append("银行/保险/金融机构需行业适配财务分析；不套用工业FCF标准")
        return {"support": "unknown", "reasons": [], "missing": missing, "periods": {}, "selected": {}, "compact": compact}
    capture = parse_day(finance.get("source_capture_at"))
    if capture is None:
        missing.append("原始Company Facts缓存取得时间未知")
    elif capture > as_of:
        missing.append("Company Facts缓存取得日晚于评级截止日")
    elif (as_of - capture).days > 30:
        missing.append("Company Facts缓存超过30日，来源快照过旧")
    latest_filed = parse_day(finance.get("latest_filed"))
    if latest_filed is None:
        missing.append("缺少最近申报日期")
    elif latest_filed > as_of:
        missing.append("存在晚于评级截止日的申报日期")
    if missing:
        return {"support": "unknown", "reasons": [], "missing": missing, "periods": {}, "selected": {}, "compact": compact}

    selected = {name: _fact_value(finance, name) for name in ("revenue", "gross_profit", "operating_income", "net_income", "operating_cashflow", "capex", "diluted_shares", "stock_compensation")}
    selected.update({name: finance.get("metrics", {}).get(name, {}).get("latest") for name in ("cash", "total_debt", "assets")})
    revenue, net_income, opcf, capex = (selected.get(k) for k in ("revenue", "net_income", "operating_cashflow", "capex"))
    support_notes: list[str] = []
    required_core = {"revenue": revenue, "net_income": net_income, "operating_cashflow": opcf}
    for name, val in required_core.items():
        if val is None:
            missing.append(f"缺少{ name }可比期间数值")
    if missing:
        return {"support": "unknown", "reasons": [], "missing": missing, "periods": {}, "selected": selected, "compact": compact}
    if not _same_period(revenue, net_income, opcf):
        missing.append("收入、净利润、经营现金流期间或币种不匹配")
    if capex is not None and not _same_period(revenue, opcf, capex):
        support_notes.append("资本开支事实期间或币种与核心期间不匹配，本轮排除该项，FCF保持未知；原始期间仍保留供查看")
        capex = None
        selected["capex"] = None
    core_end = parse_day(revenue.get("end")) if revenue else None
    if core_end is None:
        missing.append("核心财务事实缺少收入期末")
    elif core_end > as_of:
        missing.append("核心财务事实期末晚于评级截止日")
    elif (as_of - core_end).days > 210:
        missing.append("收入/净利润/经营现金流核心期间超过210日")
    for name, fact in selected.items():
        if fact is None:
            continue
        if name in {"revenue", "net_income", "operating_cashflow", "capex"} and not parse_day(fact.get("start")):
            missing.append(f"所选{name}期间起点缺失")
        if not parse_day(fact.get("end")):
            missing.append(f"所选{name}期间终点缺失")
        if not parse_day(fact.get("filed")):
            missing.append(f"所选{name}申报日期缺失或无效")
        if not fact.get("unit"):
            missing.append(f"所选{name}单位/币种缺失")
        for date_field in ("start", "end", "filed"):
            fact_day = parse_day(fact.get(date_field))
            if fact_day and fact_day > as_of:
                missing.append(f"所选{name}事实的{date_field}晚于评级截止日")
        if fact.get("filed") and parse_day(fact.get("filed")) is None:
            missing.append(f"所选{name}事实申报日期无效")
    if revenue.get("value", 0) <= 0:
        missing.append("收入分母非正，无法复算利润率")
    if missing:
        return {"support": "unknown", "reasons": [], "missing": missing, "periods": {}, "selected": selected, "compact": compact}

    net_margin = net_income["value"] / revenue["value"]
    fcf = opcf["value"] - abs(capex["value"]) if capex is not None else None
    fcf_margin = fcf / revenue["value"] if fcf is not None else None
    yoy_obj = finance.get("derived", {}).get("revenue_yoy")
    yoy = yoy_obj.get("value") if isinstance(yoy_obj, dict) and yoy_obj.get("unit") == "ratio" and yoy_obj.get("period_end") == revenue.get("end") and parse_day(yoy_obj.get("period_end")) and parse_day(yoy_obj.get("period_end")) <= as_of else None
    periods = {"start": revenue.get("start"), "end": revenue.get("end"), "unit": revenue.get("unit"), "net_margin": net_margin, "fcf": fcf, "fcf_margin": fcf_margin, "revenue_yoy": yoy}
    if core_end and (as_of - core_end).days > 180:
        reasons.append("财务期末超过180日，不符合A/A+新鲜度门槛")
    if net_income["value"] > 0 and opcf["value"] > 0 and capex is not None and fcf is not None and fcf > 0 and net_margin >= 0.05 and fcf_margin >= 0.05 and yoy is not None and yoy > 0:
        support = "strong"
        reasons.append("同币种、同期间净利润与经营现金流为正；净利率、FCF率均至少5%，同日同比为正")
    elif net_income["value"] > 0 and opcf["value"] > 0:
        support = "sound"
        reasons.append("同币种、同期间净利润和经营现金流为正；强支持门槛未全部达到")
        if capex is None:
            support_notes.append("资本开支缺失或不可比；FCF及FCF率保持未知")
        if yoy is None:
            support_notes.append("缺少可比同日同比TTM；增长保持未知")
    else:
        runway_obj = finance.get("derived", {}).get("cash_runway_months")
        cash_latest = selected.get("cash")
        runway_period = parse_day(runway_obj.get("period_end")) if isinstance(runway_obj, dict) else None
        runway = runway_obj.get("value") if isinstance(runway_obj, dict) and runway_obj.get("unit") == "months" and runway_period and runway_period <= as_of and cash_latest and cash_latest.get("end") == opcf.get("end") == runway_obj.get("period_end") else None
        if runway is not None and runway >= 12:
            support = "developing"
            reasons.append("盈利/现金流未成熟，但同期间现金与经营现金消耗显示至少12个月跑道")
        elif net_income["value"] < 0 and opcf["value"] < 0 and runway is not None and runway < 12:
            support = "strained"
            reasons.append("同期间净亏损、经营现金流为负；仅按可用现金/CFO计算的现金跑道不足12个月，该值不含短期证券与最近融资")
        else:
            support = "unknown"
            missing.append("财务支持既未满足盈利/经营现金流门槛，也缺可复算的资金跑道")
    return {"support": support, "reasons": reasons, "missing": missing + support_notes, "periods": periods, "selected": selected, "compact": compact}


def _compact_financials(finance: dict[str, Any] | None) -> dict[str, Any]:
    fields = ("revenue", "gross_profit", "operating_income", "net_income", "operating_cashflow", "capex", "cash", "total_debt", "assets", "diluted_shares", "stock_compensation")
    metrics: dict[str, Any] = {}
    source_capture_at = latest_filed = latest_period_end = None
    derived: dict[str, Any] = {k: None for k in ("revenue_yoy", "net_margin", "fcf", "fcf_margin", "cash_runway_months")}
    if not finance:
        return {"metrics": {k: None for k in fields}, "derived": derived, "latest_period_end": None, "latest_filed": None, "source_capture_at": None}
    source_capture_at = finance.get("source_capture_at")
    latest_filed = finance.get("latest_filed")
    latest_period_end = finance.get("latest_period_end")
    for name in fields:
        slot = finance.get("metrics", {}).get(name, {})
        value = slot.get("latest") if name in ("cash", "total_debt", "assets") else (slot.get("ttm") or slot.get("annual"))
        if not value:
            metrics[name] = None
            continue
        components = value.get("components") or []
        first_component = components[0] if components else {}
        metrics[name] = {k: value.get(k) for k in ("value", "unit", "period", "start", "end", "filed", "tag", "source_url", "formula", "caveat", "coverage")}
        if metrics[name].get("source_url") is None:
            metrics[name]["source_url"] = first_component.get("source_url")
        if metrics[name].get("tag") is None:
            metrics[name]["tag"] = first_component.get("tag")
    for name in derived:
        v = finance.get("derived", {}).get(name)
        derived[name] = {k: v.get(k) for k in ("value", "unit", "formula", "period_end")} if isinstance(v, dict) else None
    return {"metrics": metrics, "derived": derived, "latest_period_end": latest_period_end, "latest_filed": latest_filed, "source_capture_at": source_capture_at}


def _valid_review(review: dict[str, Any] | None, as_of: date) -> tuple[bool, list[str], date | None, list[dict[str, Any]]]:
    missing: list[str] = []
    if not review:
        return False, ["缺少逐公司业务质量核查"], None, []
    sources = [s for s in review.get("sources", []) if isinstance(s, dict)]
    observed_dates = [d for s in sources if (d := parse_day(s.get("observed_at"))) is not None]
    observed = parse_day(review.get("observed_at")) or max(observed_dates, default=None)
    if review.get("review_status") not in ("reviewed", "official_review"):
        missing.append("业务核查未完成")
    if review.get("stage") not in ALLOWED_STAGES:
        missing.append("缺少有效发展阶段")
    if review.get("materiality") not in (1, 2, 3):
        missing.append("缺少AI业务重要性判断")
    if review.get("commercial") not in (1, 2, 3):
        missing.append("缺少商业兑现判断")
    if review.get("moat") not in (0, 1, 2):
        missing.append("缺少竞争优势判断")
    if not review.get("reasons"):
        missing.append("缺少业务判断理由")
    if not sources:
        missing.append("缺少官方业务资料来源")
    if observed is None:
        missing.append("业务核查取得日期未知")
    elif observed > as_of:
        missing.append("业务核查日期晚于评级截止日")
    elif (as_of - observed).days > 90:
        missing.append("业务核查超过90日")
    for source in sources:
        observed_source = parse_day(source.get("observed_at"))
        published = parse_day(source.get("published_at"))
        if observed_source and observed_source > as_of or published and published > as_of:
            missing.append("存在晚于评级截止日的业务来源")
            break
        if not str(source.get("url", "")).startswith(("https://", "http://")):
            missing.append("业务来源缺少可复查URL")
            break
    return not missing, missing, observed, sources


def _unexplained_severe_risks(review: dict[str, Any]) -> list[str]:
    risks = [str(x) for x in review.get("risk_flags", [])]
    resolutions = [x for x in review.get("risk_resolutions", []) if isinstance(x, dict)]
    unresolved = []
    for risk in risks:
        if not any(term in risk.casefold() for term in SEVERE_TERMS):
            continue
        matched = [x for x in resolutions if str(x.get("risk", "")).casefold() == risk.casefold() and x.get("resolution") and x.get("source_url", "").startswith("https://")]
        if not matched:
            unresolved.append(risk)
    return unresolved


def _risk_domains_complete(review: dict[str, Any]) -> bool:
    """A/A+ need cited coverage of customer, competition, and funding risks."""
    urls = {s.get("url") for s in review.get("sources", []) if isinstance(s, dict)}
    return {"customer", "competition", "funding"}.issubset(_risk_evidence_domains(review, urls))


def _risk_evidence_domains(review: dict[str, Any], valid_urls: set[str] | None = None) -> set[str]:
    entries = [x for x in review.get("risk_evidence", []) if isinstance(x, dict)]
    urls = valid_urls if valid_urls is not None else {s.get("url") for s in review.get("sources", []) if isinstance(s, dict)}
    return {x.get("domain") for x in entries if x.get("domain") in {"customer", "competition", "funding"} and x.get("risk") and x.get("evidence") and x.get("source_url") in urls}


def _funding_insufficiency_reviewed(review: dict[str, Any] | None) -> bool:
    if not review:
        return False
    urls = {s.get("url") for s in review.get("sources", []) if isinstance(s, dict)}
    return any(
        item.get("domain") == "funding" and item.get("risk") and item.get("evidence")
        and item.get("source_url") in urls and item.get("liquid_assets_reviewed") is True
        and item.get("latest_financing_reviewed") is True and item.get("funding_capacity") == "insufficient"
        for item in review.get("risk_evidence", []) if isinstance(item, dict)
    )


def _validated_funding_protection(review: dict[str, Any] | None, fin: dict[str, Any], as_of: date) -> bool:
    if not review:
        return False
    protection = review.get("funding_protection")
    if not isinstance(protection, dict):
        return False
    source_type = protection.get("source_type")
    if source_type not in {"cash_raise", "committed_facility", "parent_support"}:
        return False
    if source_type != "cash_raise" and protection.get("committed") is not True:
        return False
    urls = {s.get("url") for s in review.get("sources", []) if isinstance(s, dict)}
    if protection.get("source_url") not in urls or not protection.get("evidence"):
        return False
    opcf = fin.get("selected", {}).get("operating_cashflow")
    if not opcf or opcf.get("value", 0) >= 0:
        return False
    if protection.get("amount_unit") != opcf.get("unit"):
        return False
    # Avoid double-counting a financing already included in the latest cash
    # balance, and exclude restricted/project-only or uncommitted capital.
    if protection.get("unrestricted_for_operations") is not True:
        return False
    if source_type == "cash_raise":
        received = parse_day(protection.get("received_at"))
        cash = fin.get("selected", {}).get("cash")
        cash_end = parse_day(cash.get("end")) if cash else None
        amount = protection.get("amount_value")
        return bool(received and cash_end and received > cash_end and isinstance(amount, (int, float)) and amount >= abs(opcf["value"]))
    if protection.get("committed") is not True:
        return False
    available_through = parse_day(protection.get("available_through"))
    if not available_through or (available_through - as_of).days < 365:
        return False
    amount = protection.get("undrawn_amount_value")
    if not isinstance(amount, (int, float)) or amount <= 0:
        return False
    # A facility is considered only when the amount remaining undrawn and
    # operationally usable covers the next twelve months of reported burn.
    return amount >= abs(opcf["value"])


def evaluate_stock(row: dict[str, Any], review: dict[str, Any] | None, financial_snapshot: dict[str, Any] | None, as_of: date) -> dict[str, Any]:
    code = row["code"]
    valid, review_missing, review_day, business_sources = _valid_review(review, as_of)
    fin = finance_assessment(financial_snapshot, code, as_of, row.get("industry", ""))
    if fin["support"] == "strained" and _validated_funding_protection(review, fin, as_of):
        fin["support"] = "developing"
        fin["reasons"].append("已核实的已承诺资金按最新TTM经营现金消耗可覆盖至少12个月，财务支持列为developing")
    missing = list(review_missing) + list(fin["missing"])
    reasons = list(review.get("reasons", [])) if review else []
    if review and review.get("inherited_from"):
        reasons.append(f"同发行人股权类别复用{review['inherited_from']}业务核查：{review.get('share_class_basis', '同CIK发行人证据共享')}；该证券财务数据仍独立。")
    reasons.extend(fin["reasons"])
    risks = list(review.get("risk_flags", [])) if review else []
    risks.extend((financial_snapshot or {}).get("records", {}).get(code, {}).get("caveats", []))
    grade: str | None = None
    severe_unexplained = _unexplained_severe_risks(review) if review else []
    comprehensive_risk_review = bool(review and _risk_domains_complete(review))
    cited_risk_count = len(_risk_evidence_domains(review)) if review else 0
    business_fresh = review_day is not None and review_day <= as_of and (as_of - review_day).days <= 90
    finance_known = fin["support"] != "unknown"
    strained_funding_verified = _funding_insufficiency_reviewed(review)
    if valid and business_fresh and finance_known:
        m, c, moat = review["materiality"], review["commercial"], review["moat"]
        support = fin["support"]
        financial_period_end = parse_day(fin.get("periods", {}).get("end"))
        period_age = (as_of - financial_period_end).days if financial_period_end else None
        if m == 3 and c == 3 and moat == 2 and support == "strong" and review.get("risk_reviewed") is True and comprehensive_risk_review and not severe_unexplained and period_age is not None and period_age <= 180:
            grade = "A+"
        elif m >= 2 and c >= 2 and moat >= 1 and support in ("strong", "sound") and review.get("risk_reviewed") is True and comprehensive_risk_review and not severe_unexplained:
            grade = "A"
        elif m >= 2 and c >= 1 and support in ("strong", "sound", "developing") and review.get("risk_reviewed") is True and cited_risk_count >= 1 and (support == "developing" or c < 2 or moat < 1):
            grade = "A-"
        elif m == 1 and any(s.get("evidence") for s in business_sources):
            grade = "B"
        elif support == "strained" and (m == 1 or strained_funding_verified):
            grade = "B"
    if review and severe_unexplained:
        missing.append("严重风险未被核查理由解释：" + "；".join(severe_unexplained))
        grade = None
    elif review and not comprehensive_risk_review:
        missing.append("A/A+要求客户、竞争、资金风险各有可复查来源证据")
    if review and valid and cited_risk_count == 0 and review.get("materiality", 0) >= 2:
        missing.append("A−以上至少需要一项带来源引用的具体风险核查")
    if fin["support"] == "strained" and review and review.get("materiality", 0) > 1 and not strained_funding_verified:
        missing.append("仅现金跑道不足是资金承受下界；可流动投资与最新融资未核实，不能据此评B")
    if review and not valid and not missing:
        missing.append("业务核查尚未达到评级门槛")
    if grade is None and not missing:
        missing.append("业务、商业化或财务证据未达到已核查等级门槛")
    expires = as_of + timedelta(days=30)
    if review_day:
        expires = min(expires, review_day + timedelta(days=90))
    finance_end = parse_day(fin.get("periods", {}).get("end"))
    if finance_end:
        expires = min(expires, finance_end + timedelta(days=180 if grade == "A+" else 210))
    capture_day = parse_day(fin.get("compact", {}).get("source_capture_at"))
    if capture_day:
        expires = min(expires, capture_day + timedelta(days=30))
    sources = list(business_sources)
    raw_fin = (financial_snapshot or {}).get("records", {}).get(code, {})
    sources.extend(raw_fin.get("sources", []))
    return {
        "grade": grade, "decision": "current" if grade is not None else "pending",
        "stage": review.get("stage") if review else None,
        "materiality": review.get("materiality") if review else None,
        "commercial": review.get("commercial") if review else None,
        "moat": review.get("moat") if review else None,
        "financial_support": fin["support"], "financials": fin["compact"],
        "risk_reviewed": bool(review and review.get("risk_reviewed") is True),
        "risk_evidence": review.get("risk_evidence", []) if review else [],
        "risk_resolutions": review.get("risk_resolutions", []) if review else [],
        "funding_protection": review.get("funding_protection") if review else None,
        "review_status": review.get("review_status") if review else None,
        "review_observed_at": review.get("observed_at") if review else None,
        "business_reviewed_at": review_day.isoformat() if review_day else None,
        "reasons": reasons, "missing": sorted(set(missing)), "risks": risks,
        "sources": sources, "as_of": as_of.isoformat(), "expires_at": expires.isoformat(),
        "rule_version": RULE_VERSION,
        "issuer_review_shared": bool(review and review.get("inherited_from")),
        "inherited_from": review.get("inherited_from") if review else None,
        "review_trigger": ["新财报或AI收入/订单披露", "客户集中、竞争替代、产能/资本开支、债务与稀释变化", "业务证据超过90日或财务期资料超过阈值"],
        "previous_grade": None, "change": None,
    }


def _change_type(old: dict[str, Any], new: dict[str, Any]) -> str | None:
    old_grade, new_grade = old.get("grade"), new.get("grade")
    if old_grade == new_grade:
        return None
    if new_grade is None:
        return "转待评级"
    if old_grade is None:
        return "评级完成"
    return "升级" if GRADE_RANK[new_grade] > GRADE_RANK[old_grade] else "降级"


def attach_history(records: dict[str, dict[str, Any]], previous: dict[str, Any] | None, run_id: str, input_hashes: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Attach only real transitions; exact replay preserves prior history verbatim."""
    previous_records = previous.get("records", {}) if isinstance(previous, dict) else {}
    changes: list[dict[str, Any]] = []
    if not previous:
        for quality in records.values():
            quality["history_status"] = "first_run"
        return changes
    if previous and previous.get("run_id") == run_id:
        for code, quality in records.items():
            old_same = previous_records.get(code, {})
            quality["previous_grade"] = old_same.get("previous_grade")
            quality["change"] = old_same.get("change")
            quality["history_status"] = old_same.get("history_status", "unchanged")
        return previous.get("changes", [])
    old_hashes = previous.get("summary", {}).get("input_hashes", {})
    new_hashes = input_hashes or {}
    rule_changed = (previous.get("rule_version") != RULE_VERSION
                    or (old_hashes.get("rate_code_sha256") and old_hashes.get("rate_code_sha256") != new_hashes.get("rate_code_sha256"))
                    or (old_hashes.get("protocol_sha256") and old_hashes.get("protocol_sha256") != new_hashes.get("protocol_sha256")))
    if previous:
        for code, quality in records.items():
            old = previous_records.get(code)
            if old is None:
                quality["history_status"] = "new_member"
                quality["change"] = "新增"
                changes.append({"code": code, "change": "新增", "previous_grade": None, "grade": quality.get("grade"), "reason": "证券新进入当前原始股票池", "previous_record": None})
                continue
            quality["previous_grade"] = old.get("grade")
            quality["history_status"] = "rule_version_changed" if rule_changed else "compared"
            change = _change_type(old, quality)
            quality["change"] = ("规则版本变化（不可解读为基本面变化）" if rule_changed and change else change)
            if change:
                changes.append({"code": code, "change": quality["change"], "previous_grade": old.get("grade"), "grade": quality.get("grade"), "reason": "规则版本变化，等级差异不可解读为基本面变化" if rule_changed else "本次输入与上一 current.json 的评级结果不同", "previous_record": old})
        for code, old in previous_records.items():
            if code not in records:
                changes.append({"code": code, "change": "退出", "previous_grade": old.get("grade"), "grade": None, "reason": "证券已不在当前原始股票池", "previous_record": old})
    return changes


def compute_run_id(as_of: date, pool_path: Path, review_paths: list[Path], financial_path: Path) -> tuple[str, dict[str, str]]:
    pool = read_json(pool_path)
    rows = [r for r in pool.get("rows", []) if r.get("kind") == "STOCK" and r.get("business", {}).get("tier") in (1, 2, 3)]
    pool_core = [{k: r.get(k) for k in ("code", "name", "industry", "business")} for r in rows]
    finance = read_json(financial_path) if financial_path.exists() else {}
    finance_core = {"as_of": finance.get("as_of"), "records": finance.get("records", {})}
    review_payloads = {p.name: read_json(p) for p in sorted(review_paths)}
    hashes = {"pool_membership_business_sha256": hashlib.sha256(json.dumps(pool_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "financial_facts_sha256": hashlib.sha256(json.dumps(finance_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    for name, value in review_payloads.items():
        hashes[f"review:{name}"] = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    hashes["protocol_sha256"] = sha256(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else "missing"
    hashes["rate_code_sha256"] = sha256(Path(__file__))
    prepare_code = HERE / "prepare_financials.py"
    hashes["financial_extractor_code_sha256"] = sha256(prepare_code) if prepare_code.exists() else "missing"
    seed = json.dumps({"as_of": as_of.isoformat(), "rule_version": RULE_VERSION, "input_hashes": hashes}, sort_keys=True, separators=(",", ":"))
    return "aiq_" + hashlib.sha256(seed.encode()).hexdigest()[:16], hashes


def build(as_of: date, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    pool = read_json(POOL_PATH)
    rows = [r for r in pool.get("rows", []) if r.get("kind") == "STOCK" and r.get("business", {}).get("tier") in (1, 2, 3)]
    codes = [r["code"] for r in rows]
    if len(set(codes)) != len(codes):
        raise ValueError(f"Duplicate stock code in current tier 1/2/3 pool: {len(codes)} rows")
    preferred = [HERE / n for n in ("reviews_core_top.json", "reviews_core_other.json", "reviews_core.json", "reviews_secondary.json")]
    reviews_paths = [p for p in preferred if p.exists()]
    reviews_paths += sorted(p for p in HERE.glob("reviews_*.json") if p not in reviews_paths and p.name not in ("reviews_backup.json",))
    reviews, origins = load_reviews(reviews_paths, as_of, set(codes))
    financial_snapshot = read_json(FINANCIALS_PATH) if FINANCIALS_PATH.exists() else None
    # GOOG and GOOGL are share classes of Alphabet (SEC CIK 1652044). Reuse
    # the higher-priority reviewed issuer evidence where one class is absent.
    if "US.GOOG" in codes and "US.GOOGL" in codes:
        issuer_ciks = {}
        for share in ("US.GOOG", "US.GOOGL"):
            entry = (financial_snapshot or {}).get("records", {}).get(share, {})
            issuer_ciks[share] = next((s.get("cik") for s in entry.get("sources", []) if isinstance(s, dict) and s.get("cik")), None)
        same_verified_issuer = issuer_ciks == {"US.GOOG": 1652044, "US.GOOGL": 1652044}
        for target, source in (("US.GOOG", "US.GOOGL"), ("US.GOOGL", "US.GOOG")):
            target_review, source_review = reviews.get(target), reviews.get(source)
            target_incomplete = target_review is None or target_review.get("review_status") not in ("reviewed", "official_review") or target_review.get("risk_reviewed") is not True
            source_complete = source_review is not None and source_review.get("review_status") in ("reviewed", "official_review") and source_review.get("risk_reviewed") is True
            if same_verified_issuer and target_incomplete and source_complete:
                inherited = dict(source_review)
                inherited["reasons"] = list(inherited.get("reasons", [])) + (["目标股权类别旧核查不完整；无发行人层面相反证据，继承另一类别核查。"] if target_review else [])
                inherited["inherited_from"] = source
                inherited["share_class_basis"] = "Alphabet same issuer/CIK 1652044; class-specific financial records remain separate"
                reviews[target] = inherited
                origins[target] = origins.get(source, "")
    run_id, hashes = compute_run_id(as_of, POOL_PATH, reviews_paths, FINANCIALS_PATH)
    records = {r["code"]: evaluate_stock(r, reviews.get(r["code"]), financial_snapshot, as_of) for r in rows}
    for code, quality in records.items():
        quality["review_source_file"] = origins.get(code)
    changes = attach_history(records, previous, run_id, hashes)
    grade_counts: dict[str, int] = {k: 0 for k in ("A+", "A", "A-", "B", "pending")}
    for quality in records.values():
        key = quality["grade"] if quality["grade"] is not None else "pending"
        grade_counts[key] += 1
    return {
        "as_of": as_of.isoformat(), "run_id": run_id, "rule_version": RULE_VERSION, "preview": True,
        "summary": {"pool_count": len(records), "grade_counts": grade_counts,
                    "business_reviewed": sum(1 for code in codes if reviews.get(code, {}).get("review_status") in ("reviewed", "official_review")),
                    "financial_source_unknown": sum(1 for r in records.values() if "原始Company Facts缓存取得时间未知" in r["missing"]),
                    "pending": grade_counts["pending"], "input_hashes": hashes,
                    "review_files": {p.name: sum(1 for code in codes if origins.get(code) == p.name) for p in reviews_paths}},
        "records": records, "changes": changes,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(data, encoding="utf-8")
    temp.replace(path)


def write_preview(path: Path, result: dict[str, Any], current_path: Path = CURRENT_PATH) -> None:
    if path.resolve() == current_path.resolve():
        raise ValueError("Preview output must not overwrite formal current.json")
    result["preview"] = True
    _write_json(path, result)
    write_csv(path.with_suffix(".csv"), result)


def write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["code", "grade", "decision", "stage", "materiality", "commercial", "moat", "risk_reviewed", "financial_support", "financial_currency", "revenue", "net_income", "operating_cashflow", "capex", "revenue_yoy", "latest_period_end", "latest_filed", "source_capture_at", "source_urls", "as_of", "rule_version", "expires_at", "history_status", "change", "reasons", "missing", "risks"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for code, row in sorted(result["records"].items()):
            financials = row.get("financials", {})
            metrics = financials.get("metrics", {})
            revenue = metrics.get("revenue") or {}
            unit = revenue.get("unit")
            urls = sorted({s.get("url") for s in row.get("sources", []) if isinstance(s, dict) and s.get("url")})
            writer.writerow({"code": code, "grade": row.get("grade") or "待评级", "decision": row["decision"], "stage": row.get("stage"), "materiality": row.get("materiality"), "commercial": row.get("commercial"), "moat": row.get("moat"), "risk_reviewed": row.get("risk_reviewed"), "financial_support": row.get("financial_support"), "financial_currency": unit, "revenue": revenue.get("value"), "net_income": (metrics.get("net_income") or {}).get("value"), "operating_cashflow": (metrics.get("operating_cashflow") or {}).get("value"), "capex": (metrics.get("capex") or {}).get("value"), "revenue_yoy": (financials.get("derived", {}).get("revenue_yoy") or {}).get("value"), "latest_period_end": financials.get("latest_period_end"), "latest_filed": financials.get("latest_filed"), "source_capture_at": financials.get("source_capture_at"), "source_urls": " | ".join(urls), "as_of": row.get("as_of"), "rule_version": row.get("rule_version"), "expires_at": row.get("expires_at"), "history_status": row.get("history_status"), "change": row.get("change"), "reasons": "；".join(row.get("reasons", [])), "missing": "；".join(row.get("missing", [])), "risks": "；".join(row.get("risks", []))})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--final", action="store_true", help="Write current.json and immutable runs/<run_id>/snapshot.json; default writes preview.json only")
    parser.add_argument("--preview-output", type=Path, default=HERE / "preview.json", help="Preview JSON destination; does not change current.json/history")
    args = parser.parse_args()
    previous = read_json(CURRENT_PATH) if CURRENT_PATH.exists() else None
    result = build(args.as_of, previous)
    result["preview"] = not args.final
    if args.final:
        run_snapshot = HERE / "runs" / result["run_id"] / "snapshot.json"
        if run_snapshot.exists():
            existing = read_json(run_snapshot)
            if existing.get("run_id") != result["run_id"] or existing.get("summary", {}).get("input_hashes") != result["summary"].get("input_hashes"):
                raise FileExistsError(f"Immutable snapshot identity/input hashes differ; refusing overwrite: {run_snapshot}")
            # The immutable record is authoritative on an exact input replay;
            # later current history must not rewrite its original transition.
            result = existing
        else:
            _write_json(run_snapshot, result)
        snapshot_csv = run_snapshot.parent / "ratings.csv"
        if snapshot_csv.exists():
            import hashlib
            expected = snapshot_csv.with_name(".ratings.expected.csv")
            write_csv(expected, result)
            if hashlib.sha256(expected.read_bytes()).digest() != hashlib.sha256(snapshot_csv.read_bytes()).digest():
                expected.unlink(missing_ok=True)
                raise FileExistsError(f"Immutable CSV differs from recomputed content; refusing overwrite: {snapshot_csv}")
            expected.unlink(missing_ok=True)
        else:
            write_csv(snapshot_csv, result)
        _write_json(CURRENT_PATH, result)
        write_csv(CSV_PATH, result)
    else:
        write_preview(args.preview_output, result)
    print(json.dumps({"as_of": result["as_of"], "run_id": result["run_id"], "preview": not args.final,
                      "summary": result["summary"], "changes": len(result["changes"]), "current": str(CURRENT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
