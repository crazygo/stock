"""Build a standalone, read-only AI business map with descriptive price clusters.

Only this directory is written. Raw Parquet remains outside Git. Existing
research models, thresholds and membership snapshots are never modified.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from taxonomy import GROUPS, labels

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CORE = ["NVDA", "AMD", "AVGO", "ARM", "MRVL", "MU"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def universe():
    snap = json.loads((OUT / "pool_snapshot.json").read_text())
    old = json.loads((ROOT / "analysis/preopen_ranked_policy_v5/universe.json").read_text())
    rows = {}
    excluded = []
    anomalies = []
    def add(code, name, source, kind="STOCK", fund=None):
        if kind not in ("STOCK", "ETF"):
            excluded.append(dict(code=code,name=name,reason="非股票/基金实体：" + kind))
            return
        if not re.fullmatch(r"(US\.[A-Z][A-Z0-9.\-]{0,9}|HK\.\d{5}|JP\.\d{4})", code):
            excluded.append(dict(code=code, name=name, reason="非证券代码或未识别市场"))
            return
        row = rows.setdefault(code, dict(code=code, symbol=code.removeprefix("US."),
            name=name or code, kind=kind, sources=[], funds=[], accounts=[],
            market=code.split(".")[0]))
        if source not in row["sources"]:
            row["sources"].append(source)
        if fund and fund not in row["funds"]:
            row["funds"].append(fund)
        if name and name != code and name != code.removeprefix("US."):
            row["name"] = name
        return row
    for group, source in [("全部", "watch"), ("特别关注", "favorite"), ("ETF", "etf_list")]:
        for r in snap["watchlists"].get(group, []):
            row = add(r["code"], r["name"], source, r["stock_type"])
            if row:
                row["kind"] = r["stock_type"]
                row["listing_date"] = r.get("listing_date")
    # OpenD classifies some REITs as ETF. Retain the raw type and correct only
    # identities explicitly verified via issuer materials in this run.
    for symbol in ["EQIX", "CCI"]:
        row = rows.get("US." + symbol)
        if row:
            row["raw_kind"] = row["kind"]
            row["kind"] = "STOCK"
            row["identity_note"] = "OpenD 标为 ETF；官方发行人资料显示为 REIT，公司股票单列。"
            anomalies.append(dict(code=row["code"], reason=row["identity_note"]))
    live_types = {r["code"]:r["kind"] for r in rows.values()}
    for p in snap["positions"]:
        row = add(p["code"], p["name"], "held", live_types.get(p["code"], "ETF" if p["code"] in ["US.IHE"] else "STOCK"))
        if row and p["account"] not in row["accounts"]:
            row["accounts"].append(p["account"])
    for p in snap["qqq"]["equities"]:
        add("US." + p["symbol"], p["name"], "qqq", fund="QQQ")
    # ETF entities and component relationships remain separate. Invalid IYM
    # issuer identity is rejected instead of blindly recycling the old capture.
    funds = []
    for etf in sorted([r for r in rows.values() if r["kind"] == "ETF"], key=lambda r:r["code"]):
        symbol = etf["symbol"]
        prior = old["funds"].get(symbol, {}) if etf["market"] == "US" else {}
        members = prior.get("members", [])
        state = "dated_snapshot" if prior.get("status", "").startswith("full_issuer") else "unresolved"
        if symbol == "IYM":
            members = []
            state = "identity_mismatch"
        if symbol == "QQQ":
            members = [r["symbol"] for r in snap["qqq"]["equities"]]
            state = snap["qqq"]["status"]
        header = str(prior.get("source_header", ""))
        match = re.search(r"(?:10/02/2026|Oct 01, 2026)", header)
        source_date = prior.get("source_business_date") or prior.get("source_date")
        if not source_date and match:
            source_date = "2026-10-02" if match.group() == "10/02/2026" else "2026-10-01"
        if symbol == "LAZR":
            source_date = "2026-10-02"
        if symbol == "QQQ":
            source_date = snap["qqq"].get("business_date")
        count = 0
        for s in members:
            name = old["members"].get(s, {}).get("name", s)
            row = add("US." + s, name, "component", live_types.get("US."+s, "STOCK"), fund=symbol)
            if row and row["kind"] == "STOCK":
                count += 1
        funds.append(dict(code=etf["code"], symbol=symbol, name=etf["name"], status=state,
            known_us_stocks=count, source_date=source_date,
            observed_at=prior.get("observed_at", old["observed_at"]) if prior else None,
            source_url=prior.get("source_url"), excluded_non_equity=prior.get("excluded_non_equity", []),
            note="基金身份不一致，排除旧成员" if state == "identity_mismatch" else
                 "复用旧发行人成分快照；只含当时文件可识别的美股" if state=="dated_snapshot" else
                 "全球股票、现金和衍生品未按股票回填；不完整展开" if state=="unresolved" else "发行人文件已重新读取，业务日期保留"))
    taxonomy = labels()
    review_path = OUT / "reviewed_labels.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    reviewed = reviews.get("records", {})
    sector_evidence = reviews.get("industry_evidence", {})
    names_path = OUT / "label_work/futu_names.json"
    names_capture = json.loads(names_path.read_text()) if names_path.exists() else {}
    fresh_names = {r["code"]:r["name"] for r in names_capture.get("records", [])
                   if r.get("name") not in (None, "", "未知股票", "Unknown")}
    unknown_identity = {r["code"] for r in names_capture.get("records", [])
                        if r.get("name") == "未知股票"}
    valid_groups = {g[0] for g in GROUPS}
    company_names = json.loads((ROOT / "data/ai_sec_annual_facts.json").read_text())["companies"]
    watched_issuers = {("US.GOOG" if c in ("US.GOOG", "US.GOOGL") else c)
        for c,r in rows.items() if set(r["sources"]) & {"held", "watch", "favorite"}}
    for row in rows.values():
        if row["name"] == row["symbol"] and row["symbol"] in company_names:
            row["name"] = company_names[row["symbol"]].get("name",row["name"])
        verified_names = {"AI":"C3 AI", "CRM":"Salesforce", "HPE":"Hewlett Packard Enterprise",
                          "SNOW":"Snowflake", "NOW":"ServiceNow"}
        if row["name"] == row["symbol"] and row["symbol"] in verified_names:
            row["name"] = verified_names[row["symbol"]]
        row["sources"].sort()
        row["funds"].sort()
        issuer = "US.GOOG" if row["code"] in ("US.GOOG", "US.GOOGL") else row["code"]
        row["issuer_already_watched"] = issuer in watched_issuers
        row["blind"] = row["kind"] == "STOCK" and issuer not in watched_issuers
        row["business"] = deepcopy(taxonomy.get(row["symbol"], dict(primary="pending",groups=["pending"],tier=None,
                reasons=["本版尚未完成该公司的业务与 AI 暴露核查。"],sources=[],confidence="unreviewed"))
        )
        review = reviewed.get(row["code"], {})
        sector = sector_evidence.get(row["code"], {})
        if row["kind"] == "STOCK":
            if row["code"] in unknown_identity:
                row["identity_status"] = "unresolved"
                row["identity_note"] = "来自旧发行人成分记录；本次 OpenD 基本信息返回未知股票，证券身份待核实。"
                anomalies.append(dict(code=row["code"], reason=row["identity_note"]))
            if row["name"] == row["symbol"] and row["code"] in fresh_names:
                row["name"] = fresh_names[row["code"]]
                row["name_source"] = "Futu OpenD get_stock_basicinfo / " + names_capture.get("observed_at", "")
            if sector.get("name") and row["name"] == row["symbol"]:
                row["name"] = sector["name"]
            if review.get("name") and row["name"] == row["symbol"]:
                row["name"] = review["name"]
            row["industry"] = review.get("industry") or sector.get("industry") or (
                next((g[1] for g in GROUPS if g[0] == row["business"]["primary"]), "行业资料待核实")
                if row["business"]["tier"] is not None else "行业资料待核实")
            row["industry_sources"] = sector.get("sources", [])[:1]
            b = row["business"]
            # Current positive business reviews survive a new sector-only scan.
            if review and (review.get("tier") is not None or b["tier"] is None):
                groups = [g for g in review.get("groups", []) if g in valid_groups]
                primary = review.get("primary")
                if primary not in valid_groups:
                    primary = groups[0] if groups else b["primary"]
                if not groups:
                    groups = [primary]
                row["business"] = dict(primary=primary, groups=groups, tier=review.get("tier"),
                    reasons=review.get("reasons") or ["行业资料已初筛，AI 商业路径仍待核查。"],
                    sources=review.get("sources", []),
                    confidence=review.get("confidence") or "agyd_screening",
                    review_status=review.get("review_status", "unresolved"),
                    caveat=review.get("caveat", ""), agent=review.get("agent", "codex / gpt-6-luna"))
            else:
                b["review_status"] = "prior_business_review" if b["tier"] is not None else "unresolved"
            row["business"]["industry"] = row["industry"]
            if not row["business"]["sources"] and row["industry_sources"]:
                row["business"]["sources"] = row["industry_sources"]
        if row["kind"] != "STOCK":
            row["business"] = dict(primary="fund", groups=[], tier=None, reasons=["基金和成分股分开；不把基金名称当作成员的 AI 证据。"], sources=[],confidence="instrument")
        # Constituents without preserved names remain identifiable by symbol;
        # no generated company names or identities are supplied.
        if row["business"]["primary"] == "pending":
            known = set(row["funds"])
            row["check_direction"] = ("医药与生物来源" if known & {"IBB","XBI","ARKG","BBH"} else
               "软件与金融科技来源" if known & {"IGV","FINX"} else
               "工业与电网来源" if known & {"GRID","PAVE","ITA","VIS"} else
               "半导体来源" if known & {"SOXX","SMH"} else
               "能源与材料来源" if known & {"IYE","IXC","COPX","SLVP"} else "宽基或其他来源")
    return snap, sorted(rows.values(),key=lambda r:r["code"]), funds, excluded, anomalies


def prices(rows):
    calendar = json.loads((ROOT / "analysis/preopen_ranked_policy_v5/calendar.json").read_text())["sessions"]
    if isinstance(calendar, list):
        calendar = {d["session_date"]:d for d in calendar}
    series, provenance, rejected = {}, [], []
    symbols = sorted({r["symbol"] for r in rows if r["market"]=="US" and r["kind"]=="STOCK"} | set(CORE) | {"QQQ"})
    for symbol in symbols:
        p = ROOT / "market_data/us_60m" / symbol / "2026.parquet"
        if not p.exists():
            continue
        f = pd.read_parquet(p, columns=["time_key","open","high","low","close","volume"])
        f["time"] = pd.to_datetime(f.time_key)
        f["day"] = f.time.dt.strftime("%Y-%m-%d")
        f["minute"] = f.time.dt.hour*60 + f.time.dt.minute
        returns = []
        for d,g in f[(f.day>="2026-03-01")&(f.day<="2026-09-30")&(f.minute>=570)&(f.minute<960)].groupby("day"):
            if d not in calendar:
                continue
            duration = calendar[d]["duration_minutes"]
            expected = list(range(570, 570+duration, 60))
            g = g[g.minute < 570+duration].sort_values("minute")
            if list(g.minute) != expected:
                continue
            px = g[["open","high","low","close"]].to_numpy(float)
            if not np.isfinite(px).all() or (px<=0).any():
                continue
            if (px[:,1] < np.maximum(px[:,0],px[:,3])).any() or (px[:,2] > np.minimum(px[:,0],px[:,3])).any():
                continue
            # Intraday ratio cancels a day's uniform adjustment factor. No
            # overnight split jump enters this descriptive return definition.
            returns.append([d,round(float(np.log(g.close.iloc[-1]/g.open.iloc[0])),8)])
        if returns:
            series[symbol] = returns
            provenance.append(dict(symbol=symbol,path=str(p.relative_to(ROOT)),sha256=sha(p),complete_sessions=len(returns)))
    required = [s for s in CORE+["QQQ"] if s in series]
    # Reuse the existing complete-session 5m aggregation, never a partial-day
    # snapshot. Existing build.py only emits a daily row if every scheduled
    # 5m RTH bar is present, including official early closes. This finer source
    # takes precedence on overlapping dates. Missing dates remain missing.
    native_path = ROOT / "analysis/preopen_stock_cycle_v3/daily.json"
    native = json.loads(native_path.read_text())
    native_symbols = []
    for symbol, daily in native.items():
        if symbol not in symbols:
            continue
        values = dict(series.get(symbol, []))
        added = 0
        for r in daily:
            day=r["day"]
            if day not in calendar or not "2026-03-01" <= day <= "2026-09-30":
                continue
            o,c=float(r["o"]),float(r["c"])
            if not np.isfinite([o,c]).all() or min(o,c)<=0:
                continue
            values[day]=round(float(np.log(c/o)),8)
            added+=1
        if added:
            series[symbol]=sorted(values.items())
            native_symbols.append(symbol)
    provenance.append(dict(source="既有完整5m常规盘日聚合，覆盖时优先于60m",
        path=str(native_path.relative_to(ROOT)),sha256=sha(native_path),symbols=native_symbols))
    required = [s for s in CORE+["QQQ"] if s in series]
    end = min(series[s][-1][0] for s in required)
    start = (date.fromisoformat(end) - timedelta(days=89)).isoformat()
    returns = pd.DataFrame({s:dict(v) for s,v in series.items()}).sort_index()
    returns = returns.loc[(returns.index>=start)&(returns.index<=end)]
    residual = {}
    for s in returns:
        if s == "QQQ":
            continue
        f = returns[[s,"QQQ"]].dropna()
        if len(f) < 40 or f[s].std() < 1e-8:
            continue
        x=np.column_stack([np.ones(len(f)),f.QQQ])
        residual[s]=pd.Series(f[s].to_numpy()-x@np.linalg.lstsq(x,f[s].to_numpy(),rcond=None)[0],index=f.index)
    frame=pd.DataFrame(residual)
    corr=frame.corr(min_periods=40)
    # Reject insufficient pairwise coverage rather than imputing correlation.
    while len(corr) and corr.isna().any().any():
        counts=corr.isna().sum(); victim=sorted(counts.index, key=lambda s:(-counts[s],s))[0]
        rejected.append(dict(symbol=victim,reason="聚类两两共同完整交易日不足40或残差无方差"))
        corr=corr.drop(index=victim,columns=victim)
    cluster_by={}; clusters=[]
    if len(corr)>=6:
        dist=np.sqrt(np.maximum(0,2*(1-corr.to_numpy())));np.fill_diagonal(dist,0)
        z=linkage(squareform(dist,checks=False),method="average")
        ids=fcluster(z,6,criterion="maxclust")
        for s,i in zip(corr.index,ids):cluster_by[s]=int(i)
        for i in sorted(set(ids)):
            members=sorted(s for s in cluster_by if cluster_by[s]==i)
            c=corr.loc[members,members].to_numpy(); vals=c[np.triu_indices(len(members),1)]
            clusters.append(dict(id=int(i),symbols=members,n=len(members),mean_pair_corr=round(float(vals.mean()),4) if len(vals) else None))
    return dict(series=series,core=CORE,default_start=start,default_end=end,
       earliest=min(v[0][0] for v in series.values()),latest=max(v[-1][0] for v in series.values()),
       definition="常规盘 log(收盘 Close / 09:30 Open)，仅保留官方日历要求的完整常规盘路径；既有完整5m日聚合优先，60m档案补充",
       source="本地 Futu 60m 档案与既有完整5m日聚合；本次未下载行情，未上传 R2", minimum_correlation_days=20,
       clusters=clusters,cluster_by=cluster_by,cluster_minimum_days=40,
       cluster_period=[start,end],cluster_algorithm="Agglomerative hierarchical clustering / average linkage / 固定6群",
       cluster_distance="sqrt(2 × (1 − corr))，输入为逐股收益对QQQ回归后的残差；不填补缺失",
       clustering_rejected=rejected,provenance=provenance)


def main():
    review_path = OUT / "reviewed_labels.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    snap, rows, funds, excluded, anomalies=universe()
    quality_path = OUT / "quality/current.json"
    quality_current = json.loads(quality_path.read_text()) if quality_path.exists() else None
    quality_records = (quality_current or {}).get("records", {})
    if isinstance(quality_records, list):
        quality_records = {r.get("code"): r for r in quality_records if r.get("code")}
    screen_path = OUT / "quality/screen_current.json"
    research_screen = json.loads(screen_path.read_text()) if screen_path.exists() else None
    screen_records = (research_screen or {}).get("records", {})
    eligible_codes = {r["code"] for r in rows if r["kind"] == "STOCK" and r["business"]["tier"] in (1, 2, 3)}
    screen_status = "current" if research_screen else "not_run"
    if research_screen and set(screen_records) != eligible_codes:
        research_screen = None
        screen_records = {}
        screen_status = "stale_input"
    if research_screen:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ai_research_screen_input_guard", OUT / "quality/screen.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("无法读取批量初评输入校验器")
        screen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(screen_module)
        screen_rows = sorted((r for r in rows if r["code"] in eligible_codes), key=lambda r:r["code"])
        financial_path = OUT / "quality/financial_snapshot.json"
        financial_input = json.loads(financial_path.read_text()) if financial_path.exists() else None
        input_hashes = screen_module.semantic_input_hashes(screen_rows, quality_current, financial_input)
        if research_screen.get("input_hashes") != input_hashes:
            research_screen = None
            screen_records = {}
            screen_status = "stale_input"
    for row in rows:
        row["quality"] = quality_records.get(row["code"])
        row["research_screen"] = screen_records.get(row["code"])
    price=prices(rows)
    groups=[dict(id=i,name=n,lane=l,chain=c,transmission=t) for i,n,l,c,t in GROUPS]
    stock=[r for r in rows if r["kind"]=="STOCK"]
    summary=dict(securities=len(rows),stocks=len(stock),held=sum("held" in r["sources"] for r in rows),
       watch=sum("watch" in r["sources"] for r in rows),qqq=sum("qqq" in r["sources"] for r in stock),
       business_located=sum(r["business"]["primary"]!="pending" for r in stock),
       graded=sum(r["business"]["tier"] is not None for r in stock),
       direct=sum(r["business"]["tier"]==3 for r in stock),
       ai_related=sum(r["business"]["tier"] in (1,2,3) for r in stock),
       industry_named=sum(r.get("industry") not in (None,"行业资料待核实") for r in stock),
       officially_reviewed=sum(r["business"].get("review_status") in ("official_review","prior_business_review") for r in stock),
       blind_direct=sum(r["blind"] and r["business"]["tier"]==3 for r in stock),
       resolved_funds=sum(f["status"] in ["dated_snapshot","fresh_issuer_capture","dated_issuer_fallback"] for f in funds),
       funds=len(funds),priced=sum(r["symbol"] in price["series"] for r in stock))
    quality_rating = ({k:v for k,v in quality_current.items() if k != "records"} if quality_current else
       dict(as_of=None,run_id=None,rule_version=None,summary={},changes=[],status="not_run"))
    if quality_current and "first_run" not in quality_rating:
        histories = [r.get("history_status") for r in quality_records.values() if isinstance(r, dict)]
        quality_rating["first_run"] = bool(histories) and all(x == "first_run" for x in histories)
    bundle=dict(version="ai_value_chain_map_v1",built_at=datetime.now(timezone.utc).isoformat(),
       membership_at=snap["observed_at"],qqq=snap["qqq"],summary=summary,groups=groups,
       rows=rows,funds=funds,excluded=excluded,identity_anomalies=anomalies,capture_errors=snap["errors"],price=price,
       quality_rating=quality_rating,
       research_screen=({k:v for k,v in research_screen.items() if k != "records"} if research_screen else
           dict(as_of=None,run_id=None,rule_version=None,summary={},groups=[],status=screen_status)),
       label_review={k:v for k,v in reviews.items() if k not in ("records","industry_evidence")},
       watchlist_action=json.loads((OUT/"ai_basket_action.json").read_text()) if (OUT/"ai_basket_action.json").exists() else {},
       scope="当前股票池 + 已取得的发行人美股成分快照。持仓合并两个券商实体的真实账户；仅显示成员，不展示数量或盈亏。",
       grade_method="业务关联等级为可审核规则推断，不是收入占比、概率、因果或自动文本聚类；未核实为null。多标签以单证券去重。",
       sources_notes=["继承的官方业务证据保留2026-09-25的观察时间，没有改称本次重新核验。",
         "价格分群是固定窗口的探索性描述，没有训练预测模型、没有收益标签或交易信号。",
         "GOOG/GOOGL分别是证券，统计发行人时合并；所有股票池为当前快照，不是历史PIT。",
         "成分只覆盖当时发行人文件中可识别的美股；基金未展开不代表没有AI成分。"])
    (OUT/"data.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False))
    encoded=json.dumps(bundle,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("</","<\\/")
    html=(OUT/"template.html").read_text().replace("__DATA__",encoded)
    (OUT/"index.html").write_text(html)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":main()
