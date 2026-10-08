"""Populate industry-only labels for the entire ungraded component pool.

Fund sector / SEC SIC metadata never assigns an AI grade.
"""
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent / "label_work"
SECTORS = {
    "Health Care": "医疗健康", "Information Technology": "信息技术",
    "Communication": "通信服务", "Communication Services": "通信服务",
    "Industrials": "工业", "Financials": "金融", "Materials": "材料",
    "Consumer Discretionary": "可选消费", "Consumer Staples": "必需消费",
    "Utilities": "公用事业", "Energy": "能源", "Real Estate": "房地产",
}


def main():
    evidence = json.loads((WORK / "sector_evidence.json").read_text())
    universe = {r["code"]: r for r in json.loads((WORK / "universe.json").read_text())}
    rows = json.loads((WORK / "remaining.json").read_text())
    records = []
    for r in rows:
        s = evidence.get(r["code"], {})
        raw = s.get("industry")
        industry = SECTORS.get(raw, raw)
        sic = str(s.get("sic", ""))
        # Coarse industries are displayed alongside SIC; they do not imply AI.
        if raw and raw not in SECTORS and sic.isdigit():
            n = int(sic)
            coarse = ("医疗健康" if 2830 <= n <= 2839 or 3840 <= n <= 3859 or 8000 <= n <= 8099 else
                      "金融" if 6000 <= n <= 6799 else
                      "消费与商业服务" if 5000 <= n <= 5999 or 7000 <= n <= 7299 or 7800 <= n <= 7999 else
                      "软件与计算服务" if 7370 <= n <= 7379 else
                      "电子与计算设备" if 3570 <= n <= 3579 or 3600 <= n <= 3699 else
                      "能源与公用事业" if 4900 <= n <= 4999 or 1300 <= n <= 1399 else
                      "工业与制造" if 1500 <= n <= 1799 or 2000 <= n <= 3999 else
                      "运输与通信" if 4000 <= n <= 4899 else "其他行业")
            industry = coarse + " · " + raw
        b = universe[r["code"]]["current_business"]
        primary, groups = b["primary"], b["groups"]
        if primary == "pending" and industry:
            if industry.startswith("医疗健康"):
                primary, groups = "health", ["health"]
            elif industry.startswith(("金融", "消费", "可选消费", "必需消费", "材料", "房地产", "其他行业")):
                primary, groups = "other", ["other"]
        records.append(dict(code=r["code"], name=s.get("name", r["name"]),
            industry=industry, primary=primary, groups=groups, tier=None,
            reasons=["发行人行业 / SEC SIC 资料支持行业分类；AI 产品、订单或效率增量仍待逐公司核查。"
                     if industry else "证券身份或行业证据仍待补充，未生成 AI 等级。"],
            sources=s.get("sources", [])[:1],
            review_status="industry_screened" if industry else "unresolved",
            confidence="industry_evidence_only" if industry else "unreviewed",
            agent="发行人行业与既有 SEC 行业资料整理",
            caveat="行业标签不等于 AI 关联；本条未赋予 AI 等级，未核查不能记为零。"))
    (WORK / "review_remaining.json").write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2))
    print(json.dumps({"records": len(records), "industry_named": sum(bool(r["industry"]) for r in records)}))


if __name__ == "__main__":
    main()
