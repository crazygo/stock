"""Read current OpenD pools. Save names and membership only; never trade.

Runs in a bounded subprocess via the build command. Existing fund files keep
their original source dates. Does not touch other research snapshots.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from urllib.request import Request, urlopen

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    observed = datetime.now(timezone.utc).isoformat()
    result = dict(observed_at=observed, watchlists={}, positions=[], errors=[])
    q = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        ret, groups = q.get_user_security_group()
        if ret != ft.RET_OK:
            raise RuntimeError("Unable to read OpenD group names")
        result["group_names"] = list(groups.group_name)
        for name in ["全部", "特别关注", "ETF"]:
            if name not in result["group_names"]:
                result["errors"].append("Missing watchlist group: " + name)
                continue
            ret, rows = q.get_user_security(group_name=name)
            if ret != ft.RET_OK:
                raise RuntimeError("Unable to read watchlist: " + name)
            result["watchlists"][name] = [dict(code=str(r.code), name=str(r.name),
                 stock_type=str(r.stock_type), listing_date=str(r.listing_date))
                 for r in rows.itertuples()]
    finally:
        q.close()
    for firm, label in [(ft.SecurityFirm.FUTUINC, "Moomoo US"),
                        (ft.SecurityFirm.FUTUSECURITIES, "富途证券")]:
        ctx = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US,
              host="127.0.0.1", port=11111, security_firm=firm)
        try:
            ret, accounts = ctx.get_acc_list()
            if ret != ft.RET_OK:
                result["errors"].append(label + " account list unavailable")
                continue
            for a in accounts.itertuples():
                if str(a.trd_env) != "REAL":
                    continue
                ret, rows = ctx.position_list_query(acc_id=int(a.acc_id),
                                                    trd_env=ft.TrdEnv.REAL)
                if ret != ft.RET_OK:
                    result["errors"].append(label + " positions unavailable")
                    continue
                for r in rows.itertuples():
                    if float(r.qty) == 0:
                        continue
                    result["positions"].append(dict(code=str(r.code),
                      name=str(r.stock_name), account=label + " / " + str(a.acc_type)))
        finally:
            ctx.close()
    # Issuer QQQ holdings, independent of OpenD's own ETF entity.
    prior = json.loads((ROOT / "analysis/preopen_ranked_policy_v5/universe.json").read_text())
    fund = prior["funds"]["QQQ"]
    try:
        raw = urlopen(Request(fund["source_url"], headers={"User-Agent": "Mozilla/5.0"}), timeout=5).read()
        j = json.loads(raw)
        equities = [r for r in j["holdings"] if r.get("securityTypeCode") in ("COM", "ADR", "DRNY")
                    and re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", r.get("ticker", ""))]
        if len(equities) < 90 or len(set(r["ticker"] for r in equities)) != len(equities):
            raise ValueError("Incomplete or duplicate issuer QQQ holdings")
        result["qqq"] = dict(source_url=fund["source_url"], observed_at=observed,
           business_date=j.get("effectiveBusinessDate"), effective_date=j.get("effectiveDate"),
           equities=[dict(symbol=r["ticker"], name=r["issuerName"]) for r in equities],
           excluded_non_equity=len(j["holdings"])-len(equities), status="fresh_issuer_capture")
    except Exception as e:
        j = json.loads((ROOT / "analysis/preopen_ranked_policy_v5" / fund["raw_file"]).read_text())
        result["qqq"] = dict(source_url=fund["source_url"], observed_at=fund["observed_at"],
           business_date=fund.get("source_business_date"), status="dated_issuer_fallback",
           equities=[dict(symbol=r["ticker"], name=r["issuerName"]) for r in j["holdings"]
                      if r.get("securityTypeCode") in ("COM", "ADR", "DRNY")],
           excluded_non_equity=len(fund.get("excluded_non_equity", [])))
        result["errors"].append("QQQ refresh: " + type(e).__name__)
    (OUT / "pool_snapshot.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(observed_at=observed,
      groups={k:len(v) for k,v in result["watchlists"].items()},
      position_securities=len(set(r["code"] for r in result["positions"])),
      qqq_equities=len(result["qqq"]["equities"]), errors=result["errors"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
