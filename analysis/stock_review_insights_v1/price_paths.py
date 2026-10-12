"""Descriptive paths from frozen daily OHLC. No network, forecasts or orders."""
from __future__ import annotations
import math

METHOD = "fixed_initial_equal_weight_daily_bounds_v1"
OVERLAY_METHOD = "cutoff_fixed_30_change_center_log_overlay_v1"


def centered_series(chart: dict) -> dict:
    """Log coordinates around a fixed cutoff reference; gaps remain gaps.

    Price reference uses 31 closes from T-30 to T. The alternate daily-return
    reference is the arithmetic mean of the 30 corresponding simple returns.
    Both references require the whole official-session window, independently
    of how many earlier display points are available.
    """
    dates = chart.get("dates", [])
    closes = {b[0]: b[4] for b in chart.get("bars", [])}
    valid = lambda v: isinstance(v, (int, float)) and math.isfinite(v) and v > 0
    closes = {d: v for d, v in closes.items() if valid(v)}
    previous = None
    if dates and chart.get("baseline_date") and chart["baseline_date"] < dates[0]:
        previous = 100.0
    returns = {}
    for day in dates:
        current = closes.get(day)
        returns[day] = current / previous - 1 if current is not None and previous is not None else None
        previous = current
    window = dates[-31:]
    complete = len(window) == 31 and all(d in closes for d in window)
    result = {"method": OVERLAY_METHOD, "center_changes": 30, "center_dates": window,
              "status": "complete" if complete else "missing_center", "price_reference": None,
              "daily_return_reference": None, "points": []}
    if complete:
        result["price_reference"] = math.exp(math.fsum(math.log(closes[d]) for d in window) / 31)
        result["daily_return_reference"] = math.fsum(returns[d] for d in window[1:]) / 30
    for day in dates:
        close, daily = closes.get(day), returns.get(day)
        price_y = 100 * math.log(close / result["price_reference"]) if complete and close is not None else None
        daily_y = 100 * math.log((1 + daily) / (1 + result["daily_return_reference"])) if complete and daily is not None else None
        result["points"].append([day, close, price_y, daily, daily_y])
    return result


def metrics(closes: list[float | None], changes: int) -> dict:
    result = {"changes": changes, "points": len(closes), "status": "missing", "direction": None,
              "slope": None, "r2": None, "er": None}
    if len(closes) != changes + 1 or any(v is None or not math.isfinite(v) or v <= 0 for v in closes):
        return result
    y = [v / closes[0] * 100 for v in closes]
    x_mean = changes / 2
    y_mean = sum(y) / len(y)
    slope = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(y)) / sum((i - x_mean) ** 2 for i in range(len(y)))
    intercept = y_mean - slope * x_mean
    total = sum((v - y_mean) ** 2 for v in y)
    residual = sum((v - intercept - slope * i) ** 2 for i, v in enumerate(y))
    distance = sum(abs(b - a) for a, b in zip(y, y[1:]))
    result.update(status="complete", direction="up" if slope > 1e-10 else "down" if slope < -1e-10 else "flat",
                  slope=slope, r2=None if total <= 1e-20 else min(1, max(0, 1 - residual / total)),
                  er=0.0 if distance <= 1e-10 else min(1, abs(y[-1] - y[0]) / distance))
    return result


def daily_rows(record: dict) -> dict:
    result = {}
    for bar in record.get("bars", []):
        day, *ohlc = bar[:5]
        if day in result:
            raise ValueError("Duplicate daily date: " + record.get("code", "") + " " + day)
        if len(ohlc) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in ohlc):
            continue
        o, h, l, c = ohlc
        if l > min(o, c) + 1e-8 or h < max(o, c) - 1e-8 or l > h:
            continue
        result[day] = ohlc
    return result


def basket(codes: list[str], rows: dict, dates: list[str]) -> dict:
    """Equal initial capital, fixed shares. High/low are range bounds, not synchronized extrema."""
    eligible = [code for code in codes if len(dates) == 61 and all(day in rows.get(code, {}) for day in dates)]
    result = {"kind": "basket_bounds", "method": METHOD, "members": len(codes), "used": len(eligible),
              "used_ids": eligible, "excluded_ids": [c for c in codes if c not in eligible],
              "baseline_date": dates[0] if dates else None, "dates": dates[1:], "bars": [],
              "metrics": {str(n): metrics([], n) for n in [20, 60]}, "status": "missing"}
    if not eligible:
        return result
    bases = {code: rows[code][dates[0]][3] for code in eligible}
    bars = [[day] + [100 * sum(rows[code][day][field] / bases[code] for code in eligible) / len(eligible)
                    for field in range(4)] for day in dates[1:]]
    closes = [100.0] + [b[4] for b in bars]
    result.update(bars=bars, metrics={str(n): metrics(closes[-n-1:], n) for n in [20, 60]},
                  status="complete" if len(eligible) == len(codes) else "partial",
                  kind="single_member" if len(eligible) == 1 else "basket_bounds",
                  baseline_close=bases[eligible[0]] if len(eligible)==1 else None)
    return result


def attach(data: dict, industry: dict, prices: dict | None, context: dict | None, value_chain: dict | None) -> None:
    calendars = (prices or {}).get("calendars", {})
    by_price = {r["code"]: r for r in (prices or {}).get("records", [])}
    raw = {code: daily_rows(record) for code, record in by_price.items()}
    by_context = {r["code"]: r for r in (context or {}).get("securities", [])}
    by_member = {r["code"]: r for r in (value_chain or {}).get("rows", [])}
    group_members = {r["market"] + "-" + r["group_id"]: r["security_ids"]
                     for r in industry["groups"] if r["sample_type"] == "AI价值链样本" and "security_ids" in r}
    funds = {r["code"]: r for r in (value_chain or {}).get("funds", [])}
    dates = {market: [d for d in calendars.get(market, {}).get("days", []) if d <= cutoff][-61:]
             for market, cutoff in data["market_cutoff"].items()}
    securities = {}

    def security(code: str, market: str) -> dict:
        if code in securities:
            return securities[code]
        record = by_context.get(code, {})
        row = by_member.get(code, {})
        sessions = dates.get(market, [])
        bars = raw.get(code, {})
        observed = [day for day in sessions if day in bars]
        base_date = observed[0] if observed else None
        base = bars[base_date][3] if base_date else None
        chart = {"kind": "security", "status": "complete" if len(observed) == 61 else "partial" if observed else "missing",
                 "dates": sessions[1:], "bars": [[day] + [v / base * 100 for v in bars[day]] for day in sessions[1:] if day in bars] if base else [],
                 "baseline_date": base_date, "used": 1 if observed else 0, "members": 1,
                 "baseline_close": base, "currency": by_price.get(code,{}).get("currency"),
                 "metrics": {str(n): metrics([bars.get(day, [None]*4)[3] for day in sessions[-n-1:]], n) for n in [20, 60]}}
        values = {}
        for n in [5, 20, 60]:
            source_return = record.get("price_returns", {}).get(str(n))
            if source_return is not None:
                value = source_return["value"]
            else:
                window = sessions[-n-1:]
                value = bars[window[-1]][3] / bars[window[0]][3] - 1 if len(window) == n+1 and all(day in bars for day in window) else None
            values[str(n)] = {"value": None if value is None else float(f"{value*100:.2f}"),
                              "display": "未知" if value is None else f"{value*100:+.2f}%"}
        result = {"id": code, "name": record.get("name") or row.get("name") or code, "code": code, "market": market,
                  "n": None, "values": values, "chart": chart, "children": [], "child_status": "leaf",
                  "price_source_sha256": by_price.get(code, {}).get("sha256"), "price_basis": by_price.get(code, {}).get("price_basis")}
        securities[code] = result
        return result

    for parent in data["funds"] + data["groups"]:
        market = parent["market"]
        if parent["n"] is None:
            parent["chart"] = security(parent["id"], market)["chart"]
            fund = funds.get(parent["id"], {})
            symbol = parent["id"].split(".", 1)[-1]
            children = [r["code"] for r in by_member.values() if symbol in r.get("funds", []) and r.get("kind") == "STOCK"]
            parent["child_status"] = fund.get("status", "source_missing") if children else "source_missing"
            parent["child_source_date"] = fund.get("source_date")
            parent["child_source_url"] = fund.get("source_url")
            parent["child_note"] = "发行人成分快照；未还原现金、衍生品或持仓权重。" if children else "本批次没有可核实的成分名单。"
        else:
            children = group_members.get(parent["id"], [])
            parent["chart"] = basket(children, raw, dates.get(market, []))
            if parent["chart"]["kind"]=="single_member":
                parent["chart"]["currency"] = by_price.get(parent["chart"]["used_ids"][0],{}).get("currency")
            parent["child_status"] = "current_members_snapshot" if children else "source_missing"
            parent["child_note"] = "当前分组成员回看，不是历史 PIT 成分。组图固定使用连续61个收盘时点完整的成员。"
        parent["children"] = children
        for code in children:
            security(code, code.split(".", 1)[0])
    data["security_rows"] = securities
    for row in data["funds"] + data["groups"] + list(securities.values()):
        row["chart"]["overlay"] = centered_series(row["chart"])
    data["overlay_method"] = {"version": OVERLAY_METHOD, "default_sessions": 60,
        "default_mode": "price", "center_changes": 30, "center_closes": 31,
        "price": "G=exp(mean(log(C))) on 31 official closes T-30 through T; y=100*log(C_t/G)",
        "daily": "mu=mean(C_t/C_previous-1) on the 30 changes T-30 through T; y=100*log((1+r_t)/(1+mu))",
        "axis": "zero denotes each series own reference; coordinates are log ratios, tick labels are percent relative to reference",
        "missing": "whole center window required; earlier gaps break lines and are never filled; dates use a union of actual displayed market sessions",
        "cutoff": "reference fixed at source market cutoff and unchanged by category, display window or selected series",
        "group": "same fixed-initial-equal-weight close basket as the daily K column, not an average of individual centered paths"}
    data["path_method"] = {"version": METHOD, "chart_default_changes": 20, "chart_alternate_changes": 60,
        "group_ohlc": "100 * mean(member OHLC / that member close on the common baseline date); fixed initial equal allocation, no daily rebalance",
        "group_wicks": "High is an upper bound and Low is a lower bound because member extrema need not be simultaneous; dashed for multi-member baskets",
        "cohort": "fixed members complete for 61 official sessions; excluded members remain visible as children; no missing values filled",
        "direction": "OLS slope of 100*close/initial_close versus session index; up/down/flat",
        "r2": "1-SSE/SST of that OLS fit; constant close => undefined",
        "er": "abs(last-first)/sum(abs(successive close changes)); constant path => 0",
        "metrics_window": "20 or 60 trading-day changes use 21 or 61 closes; computed on the basket path, not averaged per-stock metrics",
        "returns": "original displayed 5/20/60 return percentages retained; do not equate them with the fixed-cohort basket's returns"}
