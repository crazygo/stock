#!/usr/bin/env python3
"""Build a self-contained HTML audit of the frozen pre-open experiment."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from backtest_ai_strategies import Backtest
from identify_growth_windows import (FOLLOW_THROUGH_SESSIONS, MAX_ONE_DAY_GAIN,
                                     MAX_SESSIONS, MIN_RETAINED_GAIN, MIN_SESSIONS,
                                     TARGET_RETURN, find_growth_windows)
from preopen_three_week import FundMembership


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:+.{digits}f}%"


def build_payload(root: Path, result: dict) -> dict:
    membership = FundMembership(root / "data/ai_fund_holdings.json")
    with (root / "analysis/composer_ai_2026_pool/candidate_stocks.csv").open(newline="") as handle:
        composer_candidates = {row["ticker"]: row for row in csv.DictReader(handle)}
    market = Backtest(
        root / "market_data/us", root / "ai_universe_candidates.json",
        root / "data/ai_sec_annual_facts.json",
        extra_tickers=membership.all_tickers | composer_candidates.keys(),
        extra_snapshot=root / "data/futu_snapshot_2026-09-22.json",
        minimum_history=60,
    )
    holdings = json.loads((root / "data/ai_fund_holdings.json").read_text())
    names = {}
    for snapshot in holdings["snapshots"]:
        if snapshot["as_of"] == result["asof_session"]:
            names.update({h["ticker"]: h["name"] for h in snapshot["holdings"]})
    for ticker, candidate in composer_candidates.items():
        names.setdefault(ticker, candidate["company"])
    current = [{**row, "composer_candidate": row["ticker"] in composer_candidates}
               for row in result["current"]]
    present = {row["ticker"] for row in current}
    for ticker in sorted(composer_candidates.keys() - present):
        latest = next((bar for bar in reversed(market.bars[ticker]) if bar), None)
        current.append({
            "ticker": ticker, "funds": [], "quality_pass": None,
            "action": "无法判断", "close": float(latest["c"]) if latest else None,
            "test_opportunities": 0, "test_opportunity_mean_net": None,
            "test_opportunity_hit_rate": None, "portfolio_test_buys": 0,
            "reason": "Composer 候选股，尚未纳入当前开盘前模型的回测。",
            "composer_candidate": True, "composer_only": True,
        })
    result = {**result, "current": current}
    trades = {}
    for period in ("validation", "test"):
        for trade in result[period + "_trades"]:
            trades.setdefault(trade["ticker"], []).append({**trade, "period": period})
    symbols = {}
    for row in result["current"]:
        symbol = row["ticker"]
        candles = []
        first_market_index = max(0, len(market.days) - 160)
        dated_bars = []
        for i in range(first_market_index, len(market.days)):
            bar = market.bar(symbol, i)
            if bar:
                candles.append({"d": market.days[i].isoformat(), "o": bar["o"], "h": bar["h"],
                                "l": bar["l"], "c": bar["c"], "v": bar["v"],
                                "mi": i - first_market_index})
                dated_bars.append({**bar, "d": market.days[i].isoformat()})
            else:
                dated_bars.append(None)
        eligible_starts = {}
        for local_i, bar in enumerate(dated_bars):
            if bar and market.eligible(symbol, first_market_index + local_i,
                                       require_next_bar=False):
                quality = market.quality(symbol, first_market_index + local_i)
                eligible_starts[local_i] = {
                    "revenue": quality["revenue"],
                    "operating_cashflow": quality["operating_cashflow"],
                    "net_income": quality["net_income"],
                    "latest_filed": quality["latest_filed"],
                }
        windows = find_growth_windows(
            dated_bars,
            lambda local_i: local_i in eligible_starts,
        )
        for window in windows:
            window["quality"] = eligible_starts[window["start_index"]]
            # Chart indices refer to the compact list of present candles.
            window["start_index"] = next(i for i, bar in enumerate(candles)
                                         if bar["d"] == window["start_date"])
            window["end_index"] = next(i for i, bar in enumerate(candles)
                                       if bar["d"] == window["end_date"])
        symbols[symbol] = {"name": names.get(symbol, symbol), "candles": candles,
                           "trades": sorted(trades.get(symbol, []), key=lambda x: x["entry_date"]),
                           "growth_windows": windows, "eligible_starts": eligible_starts}
    return {"result": result, "symbols": symbols,
            "groups": {"composer_ai_2026": sorted(composer_candidates)}}


def growth_labels(payload: dict) -> dict:
    symbols = payload["symbols"]
    stocks = [{
        "ticker": row["ticker"],
        "name": symbols[row["ticker"]]["name"],
        "funds": row["funds"],
        "current_quality": row.get("quality"),
        "windows": symbols[row["ticker"]]["growth_windows"],
    } for row in payload["result"]["current"]
        if row.get("quality_pass") and symbols[row["ticker"]]["growth_windows"]]
    stocks.sort(key=lambda stock: (-max(w["return"] for w in stock["windows"]), stock["ticker"]))
    return {
        "label": "historical_growth_window_v1",
        "asof_session": payload["result"]["asof_session"],
        "source": "Most recent 160 U.S. market sessions shown in the dashboard; adjusted daily bars",
        "rules": {
            "target_close_return_at_least": TARGET_RETURN,
            "min_trading_close_intervals": MIN_SESSIONS,
            "max_trading_close_intervals": MAX_SESSIONS,
            "calendar_days_strictly_between": [3, 14],
            "max_single_session_close_gain": MAX_ONE_DAY_GAIN,
            "follow_through_sessions": FOLLOW_THROUGH_SESSIONS,
            "min_gain_retained_vs_start": MIN_RETAINED_GAIN,
            "fundamentals_and_liquidity": "Existing SEC annual-facts quality gate at the start and current quality_pass",
        },
        "stock_count": len(stocks),
        "window_count": sum(len(stock["windows"]) for stock in stocks),
        "stocks": stocks,
    }


def render(payload: dict) -> str:
    result = payload["result"]
    test = result["test"]
    valid = result["validation"]
    gate = result["quality_gate"]
    composer_count = len(payload["groups"]["composer_ai_2026"])
    checks = "".join(
        f'<li class="{"pass" if passed else "fail"}"><b>{"✓" if passed else "×"}</b> {html.escape(label)}</li>'
        for label, passed in gate["checks"].items()
    )
    # Avoid closing the inert JSON script block from a ticker/name string.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 股票开盘前策略｜样本外审计</title>
<style>
:root{{--bg:#0a101a;--panel:#111c2b;--panel2:#162436;--edge:#28394d;--text:#e8eff7;--muted:#96a9bf;--cyan:#7edbe8;--red:#ff8b8e;--green:#78d4a6;--amber:#ffd490}}
*{{box-sizing:border-box}}[hidden]{{display:none!important}}body{{margin:0;background:radial-gradient(circle at 80% 0%,#17334b 0,#0a101a 36%);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
button,input,select{{font:inherit}}button{{cursor:pointer}}a{{color:var(--cyan)}}.wrap{{max-width:1530px;margin:auto;padding:30px 26px 48px}}
.eyebrow{{color:var(--cyan);font-weight:700;letter-spacing:.14em;font-size:11px;text-transform:uppercase}}h1{{font-size:clamp(26px,3vw,40px);letter-spacing:-.035em;line-height:1.15;margin:8px 0 12px}}h2{{font-size:17px;margin:0}}p{{margin:0}}.sub{{color:var(--muted);max-width:980px}}
.head{{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}}.stamp{{border:1px solid var(--edge);border-radius:13px;padding:10px 14px;min-width:214px;color:var(--muted);font-size:12px}}.stamp strong{{display:block;color:var(--text);font-size:15px}}
.alert{{border:1px solid #8a4549;background:linear-gradient(110deg,#352026,#1b1d2b);padding:18px 21px;border-radius:15px;margin:24px 0 20px;display:flex;gap:14px;align-items:flex-start}}.alert-icon{{background:#9f4149;border-radius:8px;color:white;font-weight:900;padding:4px 9px;font-size:18px}}.alert strong{{font-size:17px}}.alert p{{color:#ddc3c6;margin-top:4px}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:22px}}.metric{{background:var(--panel);border:1px solid var(--edge);border-radius:14px;padding:15px 17px}}.metric small{{color:var(--muted);display:block;font-size:12px}}.metric strong{{display:block;font-size:26px;font-variant-numeric:tabular-nums;letter-spacing:-.05em;margin-top:4px}}.metric span{{color:var(--muted);font-size:11px}}.bad{{color:var(--red)!important}}.good{{color:var(--green)!important}}.amb{{color:var(--amber)!important}}
.section-title{{display:flex;justify-content:space-between;gap:12px;align-items:end;margin-bottom:12px}}.section-title p{{color:var(--muted);font-size:12px}}
.layout{{display:block}}.panel{{background:var(--panel);border:1px solid var(--edge);border-radius:16px;overflow:hidden}}.panel-head{{padding:16px 18px;border-bottom:1px solid var(--edge)}}.filter{{display:grid;grid-template-columns:minmax(0,1fr) 180px 235px;gap:8px;margin-top:13px}}.filter input,.filter select{{background:#0b1522;color:var(--text);border:1px solid var(--edge);border-radius:9px;padding:9px 11px;outline:none;min-width:0}}.filter input:focus,.filter select:focus{{border-color:var(--cyan)}}.hint{{color:var(--muted);font-size:11px;margin-top:9px}}
.explorer-grid{{display:grid;grid-template-columns:238px minmax(0,1fr);gap:18px;align-items:start}}.explorer-main{{min-width:0}}.sidebar{{position:sticky;top:14px;background:var(--panel);border:1px solid var(--edge);border-radius:15px;padding:16px;display:grid;gap:14px}}.sidebar h3{{font-size:14px;margin:0 0 6px}}.sidebar-title{{color:var(--cyan);font-size:10px;letter-spacing:.12em;font-weight:700}}.sidebar-group{{display:grid;gap:8px}}.sidebar-label{{display:grid;gap:4px;color:var(--muted);font-size:11px}}.sidebar input[type=number]{{width:100%;min-width:0;background:#0b1522;color:var(--text);border:1px solid var(--edge);border-radius:8px;padding:7px 9px}}.sidebar input[type=range]{{width:100%;accent-color:#55dfc1}}.sidebar .visual-option{{display:flex;align-items:center;gap:8px;color:var(--text);font-size:12px;cursor:pointer}}.sidebar .visual-option input{{accent-color:#55dfc1}}.sidebar-pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.sidebar small{{color:var(--muted);font-size:10px;line-height:1.5}}.sidebar-summary{{border-top:1px solid var(--edge);padding-top:10px;color:#b7f2dd;font-size:11px}}.sidebar button{{background:#193940;border:1px solid #346a65;color:#b8f5e2;border-radius:8px;padding:7px;font-size:11px}}.sidebar button:hover{{border-color:#55dfc1}}
.stock-rows{{display:grid;gap:12px;margin-top:12px}}.stock-row{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start}}.stock{{width:100%;min-width:0;display:flex;flex-direction:column;gap:7px;text-align:left;background:var(--panel);color:var(--text);padding:14px 15px;border:1px solid var(--edge);border-radius:12px}}.stock-head{{display:flex;justify-content:space-between;align-items:start;gap:10px}}.stock-title{{min-width:0}}.stock-return{{text-align:right;white-space:nowrap;font-size:16px;font-variant-numeric:tabular-nums}}.stock-return small{{display:block;color:var(--muted);font-size:10px;font-weight:400}}.stock-title-line{{display:flex;align-items:center;gap:8px}}.pin-button{{flex:none;background:#182638;color:var(--muted);border:1px solid var(--edge);border-radius:7px;padding:3px 8px;font-size:11px;line-height:1.4}}.pin-button:hover{{border-color:var(--cyan);color:var(--cyan)}}.pin-button.pinned{{background:#3b3020;color:var(--amber);border-color:#80663a}}.ticker{{font-size:17px;font-weight:750;letter-spacing:.015em}}.name{{color:var(--muted);font-size:12px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}}.stock-meta{{display:flex;align-items:center;gap:5px;flex-wrap:wrap}}.stock-chart-label{{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:10px;margin-top:2px}}.stock-chart-label b{{font-size:11px;font-variant-numeric:tabular-nums}}.stock-chart{{display:block;width:100%;height:310px;border-bottom:1px solid #28394d}}.stats{{font-size:11px;color:var(--muted)}}.tag{{font-size:10px;border-radius:5px;padding:2px 6px;background:#274050;color:#a9dce1}}.tag.warn{{color:#ffce92;background:#473329}}.hover{{min-height:21px;color:var(--muted);font-size:11px;margin-top:3px}}.legend{{color:var(--muted);font-size:11px;margin:0 0 8px;display:flex;gap:15px;flex-wrap:wrap}}.legend i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
.tag.growth{{background:#164744;color:#8ff4d9}}.hover-price{{margin-bottom:5px}}.hover .growth-window{{background:#112e35;border-left:3px solid #55dfc1;border-radius:5px;padding:5px 8px;color:#c9f7e9;font-size:11px;line-height:1.45;margin-top:5px}}.hover .growth-window small{{display:block;color:#95bbb4;font-size:10px}}.hover .growth-window b{{font-variant-numeric:tabular-nums}}.method-note{{border:1px solid #2b645c;background:#10272b;border-radius:9px;padding:9px 11px;color:#b6dcd3;font-size:11px;margin-top:10px}}
.trades{{padding:0 18px 18px}}.trades h3{{font-size:13px;margin:2px 0 9px}}.trade{{display:grid;grid-template-columns:92px 1fr 85px;gap:6px;border-top:1px solid var(--edge);padding:7px 0;font-size:11px;color:var(--muted)}}.trade b{{color:var(--text)}}.trade span:last-child{{text-align:right}}.none{{color:var(--muted);font-size:12px}}.explainer{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}}.body{{padding:16px 18px;color:var(--muted);font-size:12px}}.body p+p{{margin-top:10px}}.body strong{{color:var(--text)}}ul.checks{{list-style:none;margin:0;padding:16px 18px;display:grid;grid-template-columns:1fr 1fr;gap:8px}}.checks li{{font-size:12px;color:var(--muted)}}.checks li b{{font-size:17px;margin-right:5px}}.checks .pass b{{color:var(--green)}}.checks .fail b{{color:var(--red)}}.compare{{width:100%;border-collapse:collapse}}.compare th,.compare td{{text-align:right;border-bottom:1px solid #28394d;padding:7px 8px;font-size:11px}}.compare th:first-child,.compare td:first-child{{text-align:left}}.compare th{{color:var(--muted);font-weight:500}}.foot{{color:var(--muted);font-size:11px;margin-top:20px;line-height:1.65}}.foot p+p{{margin-top:8px}}
@media(max-width:1100px){{.metrics{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:900px){{.explorer-grid{{grid-template-columns:1fr}}.sidebar{{position:static;grid-template-columns:repeat(2,minmax(0,1fr))}}.sidebar-summary{{grid-column:1/-1}}}}@media(max-width:760px){{.wrap{{padding:20px 13px 32px}}.head{{display:block}}.stamp{{margin-top:14px;width:max-content}}.metrics{{grid-template-columns:repeat(2,1fr)}}.explainer{{grid-template-columns:1fr}}.section-title{{display:block}}.filter{{grid-template-columns:1fr 105px}}.stock-row{{gap:8px}}.stock{{padding:10px 11px}}.stock-chart{{height:270px}}ul.checks{{grid-template-columns:1fr}}}}@media(max-width:520px){{.sidebar{{grid-template-columns:1fr}}.stock-row{{grid-template-columns:1fr}}.stock-chart{{height:300px}}}}@media(max-width:400px){{.metrics{{grid-template-columns:1fr 1fr}}.metric strong{{font-size:20px}}}}
</style>
</head>
<body><main class="wrap">
<header class="head"><div><div class="eyebrow">RESEARCH DESK / 01 · PRE-OPEN</div><h1>AI 股票开盘前策略审计</h1><p class="sub">这是上一收盘日线的盘前决策代理版，尚无真实盘前新闻或成交。用当时可见的 ETF 成分与已披露财报形成信号；目标是 15 个交易日内先触及 +20%，并在 -10% 风险线之前退出。</p></div><div class="stamp">信号依据：美股已完成交易日<strong>{result['asof_session']}</strong>下一次开盘前仍需确认真实盘前信息</div></header>
<div class="alert"><div class="alert-icon">!</div><div><strong>策略未通过样本外质量门槛，当前没有可执行的买入或卖出建议</strong><p>独立测试期组合收益 {pct(test['portfolio_return'])}，68 笔交易平均净收益 {pct(test['mean_net_return'])}。图中蓝色买点仅为历史模拟，也只用于排查失败原因。</p></div></div>
<section class="metrics">
<div class="metric"><small>测试期目标先达成率</small><strong class="bad">{pct(test['target_rate']).replace('+','')}</strong><span>{test['trades']} 笔组合交易 / +20% 先于 -10%</span></div>
<div class="metric"><small>测试期平均交易净收益</small><strong class="bad">{pct(test['mean_net_return'])}</strong><span>同日可买候选平均 {pct(test['matched_day_mean_net_return'])}</span></div>
<div class="metric"><small>10 槽位组合收益</small><strong class="bad">{pct(test['portfolio_return'])}</strong><span>最大按日标记回撤 {pct(test['marked_max_drawdown'])}</span></div>
<div class="metric"><small>买入覆盖 / 亏损交易</small><strong class="amb">{pct(test['days_with_buy_rate']).replace('+','')} / {pct(test['loss_trade_rate']).replace('+','')}</strong><span>40 / 54 个测试日有买入；亏损 43 / 68 笔</span></div>
<div class="metric"><small>基金主题池 / 业务筛选通过</small><strong>{result['coverage']['latest_members']} / {result['coverage']['latest_quality_pass']}</strong><span>AIQ · BOTZ · CLOU · DTCR</span></div>
</section>
<div class="explorer-grid"><aside class="sidebar" aria-label="图表可视化选项">
<div class="sidebar-group"><div class="sidebar-title">CHART OPTIONS</div><h3>可视化选项</h3><label class="visual-option"><input id="show-growth" type="checkbox" checked>显示增长区间</label><label class="visual-option"><input id="show-trades" type="checkbox" checked>显示旧策略买点</label><label class="visual-option"><input id="show-volume" type="checkbox" checked>显示成交量</label></div>
<div class="sidebar-group"><div class="sidebar-title">GROWTH WINDOW</div><h3>增长区间条件</h3><label class="sidebar-label" for="growth-threshold">收盘涨幅至少 <span id="threshold-value">20%</span></label><input id="threshold-slider" type="range" min="5" max="80" step="1" value="20" aria-label="增长阈值滑块"><input id="growth-threshold" type="number" min="5" max="80" step="1" value="20" aria-label="增长阈值百分比"><div class="sidebar-pair"><label class="sidebar-label" for="min-calendar">最短自然日<input id="min-calendar" type="number" min="1" max="30" step="1" value="4"></label><label class="sidebar-label" for="max-calendar">最长自然日<input id="max-calendar" type="number" min="1" max="30" step="1" value="13"></label></div><label class="sidebar-label" for="min-sessions">最少交易日收盘间隔<input id="min-sessions" type="number" min="1" max="20" step="1" value="4"></label><small>首日收盘到达标日收盘计时。单日涨幅上限 12%，达标后两日须保留至少一半目标涨幅。</small><button id="reset-growth" type="button">恢复默认条件</button></div>
<div class="sidebar-summary" id="growth-summary">正在计算区间…</div>
</aside><div class="explorer-main">
<div class="section-title"><div><div class="eyebrow">UNIVERSE EXPLORER</div><h2>逐股查看历史增长区间与模拟买点</h2></div><p>增长区间按左侧条件即时重算；<a href="growth_windows.json">默认 20% 标签 JSON</a>。</p></div>
<section class="layout"><div class="panel"><div class="panel-head"><h2>股票浏览器 <span id="count" style="color:var(--muted);font-size:12px"></span></h2><div class="filter"><input id="search" aria-label="搜索股票代码或名称" placeholder="搜索代码或公司名"><select id="fund" aria-label="基金筛选"><option value="">所有基金</option><option>AIQ</option><option>BOTZ</option><option>CLOU</option><option>DTCR</option></select><select id="group" aria-label="股票分组"><option value="">全部股票</option><option value="pinned">只看置顶</option><option value="growth">历史增长区间</option><option value="composer_ai_2026">Composer AI 候选组（{composer_count}）</option></select></div><p class="hint">任意分组都可显示符合左侧条件的绿色增长区间；分组只决定列出哪些股票。蓝/紫色三角是旧策略模拟买点。Composer AI 候选组来自三个共享策略的公开股票范围，不代表 2026 年逐日实际持仓。</p><p class="method-note">起点按当时已披露 SEC 年报核验：年营收 ≥10 亿美元、经营现金流与净利润为正，股价 ≥5 美元、近 20 日平均成交额 ≥1000 万美元。增长分组还要求当前业务筛选通过。该条件用于人工审查，不能证明没有市场操纵，也不代表预判信号。</p></div></div><div class="legend"><span><i style="background:var(--red)"></i>收涨</span><span><i style="background:var(--green)"></i>收跌</span><span id="growth-legend"><i style="background:#55dfc1"></i><span id="growth-legend-label">历史 +20% 区间</span></span><span id="trade-legend-test"><i style="background:#54c3f5"></i>测试期组合买点</span><span id="trade-legend-validation"><i style="background:#9b95ef"></i>验证期组合买点</span></div><div class="stock-rows" id="stocks"></div></section></div></div>
<section class="explainer"><div class="panel"><div class="panel-head"><h2>为什么当前统一为“无法判断”</h2></div><ul class="checks">{checks}</ul><div class="body">四类行动的规则：<strong>买入</strong>需通过全局样本外质量门槛、达到冻结的阈值且仓位可用；<strong>卖出</strong>需验证持仓卖出模型，目前尚无通过的卖出模型；<strong>不动</strong>用于模型通过但个股无买卖条件；<strong>无法判断</strong>用于资料不足或模型整体失效。这次没有通过全局门槛，不会为了凑出手次数强行推荐买入。</div></div>
<div class="panel"><div class="panel-head"><h2>验证期很好，测试期却失效</h2></div><div class="body"><table class="compare"><thead><tr><th>指标</th><th>验证期</th><th>测试期</th></tr></thead><tbody><tr><td>组合买入</td><td>{valid['trades']} 笔</td><td>{test['trades']} 笔</td></tr><tr><td>先达 +20% 比例</td><td>{pct(valid['target_rate']).replace('+','')}</td><td>{pct(test['target_rate']).replace('+','')}</td></tr><tr><td>平均交易净收益</td><td>{pct(valid['mean_net_return'])}</td><td>{pct(test['mean_net_return'])}</td></tr><tr><td>组合收益</td><td>{pct(valid['portfolio_return'])}</td><td>{pct(test['portfolio_return'])}</td></tr><tr><td>最大回撤</td><td>{pct(valid['marked_max_drawdown'])}</td><td>{pct(test['marked_max_drawdown'])}</td></tr></tbody></table><p>测试期逐日可买股票的 +20% 基础比例是 {pct(test['test_base_target_rate']).replace('+','')}；模型概率 Brier {test['brier']:.4f}，还差于只报常数基础比例的 {test['constant_base_brier']:.4f}。训练期基础比例仅 {pct(result['train_base_hit_rate']).replace('+','')}，市场状态变化让训练期概率失准。</p><p>逐股较漂亮的测试数字是事后看到的结果，不能据此挑出“赢家股票”再把同一测试期当证明。</p></div></div></section>
<footer class="foot"><p>口径：训练 {result['periods']['train'][0]}—{result['periods']['train'][1]}；验证 {result['periods']['validation'][0]}—{result['periods']['validation'][1]}；测试 {result['periods']['test'][0]}—{result['periods']['test'][1]}，区间有 15 交易日隔离。信号收盘后计算，下一交易日开盘模拟买入；开盘跳空高于上日收盘 3% 的信号跳过。最多每天 3 笔、同时 10 槽位。目标 +20%、风险线 -10%，同日同时触及按先止损，每边 0.1% 成本。K 线按拆股调整，未计股息、税和真实冲击成本。</p><p>数据：<a href="https://www.globalxetfs.com/funds/aiq">Global X AIQ</a>、<a href="https://www.globalxetfs.com/funds/BOTZ">BOTZ</a>、<a href="https://www.globalxetfs.com/funds/clou">CLOU</a>、<a href="https://www.globalxetfs.com/funds/dtcr">DTCR</a> 日期化成分，<a href="https://www.sec.gov/search-filings/edgar-application-programming-interfaces">SEC EDGAR 公司财报</a>，本地 Massive 日线及 2026-09-22 富途收盘快照。本研究没有历史盘前新闻、盘前成交和订单簿，不能检验“盘前消息先动、开盘后散户跟进”的假说；盘中、盘后需要另建模型。基金回溯文件仍可能缺少后来退市/被收购的公司，保留残余幸存者偏差。</p></footer>
</main><script id="payload" type="application/json">{data}</script><script>
const payload=JSON.parse(document.getElementById('payload').textContent), result=payload.result, symbols=payload.symbols;
const byTicker=Object.fromEntries(result.current.map(x=>[x.ticker,x]));
const composerTickers=new Set(payload.groups.composer_ai_2026);
const pct=(x,d=1)=>x==null?'—':`${{x>=0?'+':''}}${{(x*100).toFixed(d)}}%`;
const rate=(x,d=1)=>x==null?'—':`${{(x*100).toFixed(d)}}%`;
const nice=(x)=>x==null?'—':Number(x).toLocaleString('en-US',{{maximumFractionDigits:2}});
const visualDefaults={{threshold:20,minCalendar:4,maxCalendar:13,minSessions:4,showGrowth:true,showTrades:true,showVolume:true}};
let visual={{...visualDefaults}};
try{{const saved=JSON.parse(localStorage.getItem('preopen-growth-visual-v1')||'null');if(saved&&typeof saved==='object')visual={{...visual,...saved}}}}catch(error){{console.warn('无法读取本地可视化选项',error)}}
function clampInt(value,min,max,fallback){{const n=Number(value);return Number.isFinite(n)?Math.min(max,Math.max(min,Math.round(n))):fallback}}
function normalizeVisual(){{visual.threshold=clampInt(visual.threshold,5,80,20);visual.minCalendar=clampInt(visual.minCalendar,1,30,4);visual.maxCalendar=clampInt(visual.maxCalendar,visual.minCalendar,30,13);visual.minSessions=clampInt(visual.minSessions,1,Math.min(20,visual.maxCalendar),4);for(const key of ['showGrowth','showTrades','showVolume'])visual[key]=visual[key]!==false}}
function syncVisualControls(){{document.getElementById('growth-threshold').value=visual.threshold;document.getElementById('threshold-slider').value=visual.threshold;document.getElementById('threshold-value').textContent=visual.threshold+'%';document.getElementById('min-calendar').value=visual.minCalendar;document.getElementById('max-calendar').value=visual.maxCalendar;document.getElementById('min-sessions').value=visual.minSessions;document.getElementById('show-growth').checked=visual.showGrowth;document.getElementById('show-trades').checked=visual.showTrades;document.getElementById('show-volume').checked=visual.showVolume}}
function calendarSpan(a,b){{return Math.round((Date.parse(b+'T00:00:00Z')-Date.parse(a+'T00:00:00Z'))/86400000)}}
function findGrowthWindows(item){{
 const bars=item.candles, candidates=[], target=visual.threshold/100;
 for(let start=0;start<bars.length-2-visual.minSessions;start++){{
  const first=bars[start],quality=item.eligible_starts[first.mi];if(!quality||first.c<=0)continue;
  let crossing=-1,spike=false;
  for(let end=start+1;end<bars.length-2;end++){{
   const previous=bars[end-1],current=bars[end];
   if(current.mi!==previous.mi+1||calendarSpan(first.d,current.d)>visual.maxCalendar)break;
   if(current.c/previous.c-1>0.12)spike=true;
   if(current.c/first.c-1>=target){{crossing=end;break}}
  }}
  if(crossing<0||crossing-start<visual.minSessions||spike)continue;
  const days=calendarSpan(first.d,bars[crossing].d);if(days<visual.minCalendar||days>visual.maxCalendar)continue;
  const a=bars[crossing+1],b=bars[crossing+2];
  if(a.mi!==bars[crossing].mi+1||b.mi!==a.mi+1||a.c/first.c-1<target/2||b.c/first.c-1<target/2)continue;
  candidates.push({{start_index:start,end_index:crossing,sessions:crossing-start,calendar_days:days,start_date:first.d,end_date:bars[crossing].d,start_close:first.c,end_close:bars[crossing].c,return:bars[crossing].c/first.c-1,quality}});
 }}
 const chosen=[];candidates.sort((a,b)=>b.return-a.return||a.start_index-b.start_index);
 for(const candidate of candidates)if(chosen.every(old=>candidate.end_index<old.start_index||candidate.start_index>old.end_index))chosen.push(candidate);
 return chosen.sort((a,b)=>a.start_index-b.start_index)
}}
function recalculateGrowth(){{for(const item of Object.values(symbols))item.growth_windows=findGrowthWindows(item)}}
let visible=[];
let pinnedTickers=new Set();
const stockRows=document.getElementById('stocks');
function safe(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function rank(x){{return x.test_opportunities>=20&&x.test_opportunity_mean_net!=null?x.test_opportunity_mean_net:-100}}
function growthRank(x){{return Math.max(0,...symbols[x.ticker].growth_windows.map(w=>w.return))}}
function growthDetails(w){{const q=w.quality;return `<div class="growth-window"><b>${{safe(w.start_date)}} → ${{safe(w.end_date)}} · ${{w.calendar_days}} 自然日 / ${{w.sessions}} 个交易间隔 · ${{pct(w.return)}}</b><small>收盘 $${{nice(w.start_close)}} → $${{nice(w.end_close)}} · 起点已披露年营收 ${{nice(q.revenue/1e9)}}B、经营现金流 ${{nice(q.operating_cashflow/1e9)}}B、净利润 ${{nice(q.net_income/1e9)}}B（财报 ${{safe(q.latest_filed)}}）</small></div>`}}
function stockCard(x){{
 const item=symbols[x.ticker],bars=item.candles,windows=visual.showGrowth?item.growth_windows:[];
 const change=bars.length>1?bars[bars.length-1].c/bars[0].c-1:null,isPinned=pinnedTickers.has(x.ticker),isComposer=Boolean(x.composer_candidate),isComposerOnly=Boolean(x.composer_only),lastDate=bars.length?bars[bars.length-1].d:'—';
 return `<article class="stock" data-ticker="${{safe(x.ticker)}}"><div class="stock-head"><div class="stock-title"><div class="stock-title-line"><span class="ticker">${{safe(x.ticker)}}</span><button class="pin-button ${{isPinned?'pinned':''}}" type="button" data-pin="${{safe(x.ticker)}}" aria-pressed="${{isPinned}}">${{isPinned?'★ 已置顶':'☆ 置顶'}}</button></div><span class="name">${{safe(item.name)}}</span></div><b class="stock-return ${{x.test_opportunity_mean_net==null?'':x.test_opportunity_mean_net>=0?'good':'bad'}}">${{isComposerOnly?'—':pct(x.test_opportunity_mean_net)}}<small>${{isComposerOnly?'尚未回测':'测试期逐日机会均值'}}</small></b></div><span class="stock-meta">${{x.funds.length?'<span class="tag">'+safe(x.funds.join(' · '))+'</span>':''}}${{isComposer?'<span class="tag">Composer AI 候选</span>':''}}<span class="tag ${{x.quality_pass?'':'warn'}}">${{isComposerOnly?'尚未回测':x.quality_pass?'当前业务筛选通过':'当前未通过筛选'}}</span>${{windows.length?`<span class="tag growth">历史 ≥${{visual.threshold}}% 区间 · ${{windows.length}} 段</span>`:''}}<span class="tag">建议：${{safe(x.action)}}</span></span><span class="stock-chart-label"><span>近 ${{bars.length}} 个交易日日 K 线</span><b class="${{change==null?'':change>=0?'good':'bad'}}">区间 ${{pct(change)}}</b></span><canvas class="stock-chart" data-ticker="${{safe(x.ticker)}}" aria-label="${{safe(x.ticker)}} 日K线、历史增长区间及模拟买点" role="img"></canvas><div class="hover" data-hover="${{safe(x.ticker)}}"></div><span class="stats">${{isComposerOnly?'尚未回测 · 最近收盘 '+lastDate+' '+(x.close==null?'—':'$'+nice(x.close)):'测试期 '+x.test_opportunities+' 个逐日机会 · +20% 先达率 '+rate(x.test_opportunity_hit_rate)+' · '+x.portfolio_test_buys+' 笔组合买入 · 现价 '+(x.close==null?'—':'$'+nice(x.close))}}</span><span class="stats">${{safe(x.reason)}}${{isComposerOnly?'':'；该股组合实际交易 '+item.trades.length+' 笔。'}}</span></article>`
}}
async function togglePin(ticker){{const next=new Set(pinnedTickers);if(next.has(ticker))next.delete(ticker);else next.add(ticker);document.querySelectorAll('[data-pin]').forEach(button=>button.disabled=true);try{{const response=await fetch('/api/pinned-stocks',{{method:'PUT',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{tickers:[...next]}})}});if(!response.ok)throw new Error(`HTTP ${{response.status}}`);pinnedTickers=next;updateList()}}catch(error){{alert('置顶状态未保存到 pinned_stocks.json。请确认本机策略服务正在运行后重试。');document.querySelectorAll('[data-pin]').forEach(button=>button.disabled=false)}}}}
function updateList(){{let q=document.getElementById('search').value.trim().toLowerCase(), fund=document.getElementById('fund').value, group=document.getElementById('group').value;
 visible=result.current.filter(x=>(!fund||x.funds.includes(fund))&&(group!=='pinned'||pinnedTickers.has(x.ticker))&&(group!=='composer_ai_2026'||composerTickers.has(x.ticker))&&(group!=='growth'||x.quality_pass&&symbols[x.ticker].growth_windows.length>0)&&(!q||x.ticker.toLowerCase().includes(q)||symbols[x.ticker].name.toLowerCase().includes(q))).sort((a,b)=>Number(pinnedTickers.has(b.ticker))-Number(pinnedTickers.has(a.ticker))||(q?(b.ticker.toLowerCase()===q)-(a.ticker.toLowerCase()===q):0)||(group==='growth'?growthRank(b)-growthRank(a):rank(b)-rank(a))||a.ticker.localeCompare(b.ticker));
 const matching=result.current.filter(x=>x.quality_pass&&symbols[x.ticker].growth_windows.length);
 document.querySelector('#group option[value="growth"]').textContent=`历史 ≥${{visual.threshold}}% 区间（${{matching.length}}）`;
 document.getElementById('growth-summary').textContent=`当前条件：${{matching.length}} 只股票 · ${{matching.reduce((sum,x)=>sum+symbols[x.ticker].growth_windows.length,0)}} 段区间`;
 document.getElementById('growth-legend-label').textContent=`历史 ≥${{visual.threshold}}% 区间`;
 document.getElementById('growth-legend').hidden=!visual.showGrowth;
 document.getElementById('trade-legend-test').hidden=!visual.showTrades;
 document.getElementById('trade-legend-validation').hidden=!visual.showTrades;
 document.getElementById('count').textContent=`${{visible.length}} / ${{result.current.length}}`;
 const rowSize=window.matchMedia('(max-width:520px)').matches?1:2;
 const rows=[];for(let i=0;i<visible.length;i+=rowSize)rows.push(`<div class="stock-row">${{visible.slice(i,i+rowSize).map(stockCard).join('')}}</div>`);
 stockRows.innerHTML=rows.join('');
 document.querySelectorAll('[data-pin]').forEach(button=>button.addEventListener('click',()=>togglePin(button.dataset.pin)));
 document.querySelectorAll('.stock-chart').forEach(canvas=>drawChart(canvas.dataset.ticker,canvas,document.querySelector(`[data-hover="${{canvas.dataset.ticker}}"]`)));
}}
let hoverIndexes={{}};function drawChart(ticker,canvas,hoverNode){{let hoverIndex=hoverIndexes[ticker]??-1;const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=rect.width,h=rect.height;
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const bars=symbols[ticker].candles, marks=symbols[ticker].trades;if(!bars.length){{ctx.fillStyle='#96a9bf';ctx.fillText('缺少连续行情',20,40);return}}
 const windows=visual.showGrowth?symbols[ticker].growth_windows:[];
 const L=49,R=14,T=13,B=38,V=visual.showVolume?60:0,priceH=h-T-B-V-13,plotW=w-L-R,step=plotW/bars.length;
 let lo=Math.min(...bars.map(b=>b.l)),hi=Math.max(...bars.map(b=>b.h)),pad=(hi-lo)*.07||1;lo-=pad;hi+=pad;const yy=p=>T+(hi-p)/(hi-lo)*priceH;const volMax=Math.max(...bars.map(b=>b.v));
 ctx.font='11px -apple-system, sans-serif';ctx.textAlign='right';for(let k=0;k<=4;k++){{const y=T+k*priceH/4,p=hi-(hi-lo)*k/4;ctx.strokeStyle='#263749';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke();ctx.fillStyle='#879eb1';ctx.fillText(p.toFixed(0),L-7,y+4)}}
 windows.forEach(win=>{{const x1=L+win.start_index*step,x2=L+(win.end_index+1)*step;ctx.fillStyle='rgba(85,223,193,.14)';ctx.fillRect(x1,T,x2-x1,priceH);ctx.strokeStyle='#55dfc1';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x1,T);ctx.lineTo(x1,T+priceH);ctx.moveTo(x2,T);ctx.lineTo(x2,T+priceH);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#a0ffe3';ctx.font='bold 10px -apple-system, sans-serif';ctx.textAlign='center';ctx.fillText(pct(win.return),Math.min(w-R-22,Math.max(L+22,(x1+x2)/2)),T+11)}});
 bars.forEach((b,i)=>{{const x=L+i*step+step/2,up=b.c>=b.o,color=up?'#ff8589':'#70cba0';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,yy(b.h));ctx.lineTo(x,yy(b.l));ctx.stroke();let y1=yy(Math.max(b.o,b.c)),y2=yy(Math.min(b.o,b.c));ctx.fillRect(x-Math.max(1,step*.32),y1,Math.max(2,step*.64),Math.max(1.2,y2-y1));if(visual.showVolume){{ctx.globalAlpha=.5;ctx.fillRect(x-Math.max(1,step*.32),h-B-V+(1-b.v/volMax)*V,Math.max(2,step*.64),b.v/volMax*V);ctx.globalAlpha=1}}}});
 const ix=Object.fromEntries(bars.map((b,i)=>[b.d,i]));if(visual.showTrades)marks.forEach((m,j)=>{{let i=ix[m.entry_date];if(i==null)return;let x=L+i*step+step/2,y=yy(bars[i].l)+10+(j%2)*7;ctx.fillStyle=m.period==='test'?'#54c3f5':'#9b95ef';ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-4,y+8);ctx.lineTo(x+4,y+8);ctx.closePath();ctx.fill()}});
 ctx.textAlign='center';ctx.fillStyle='#879eb1';[0,.25,.5,.75,1].forEach(frac=>{{let i=Math.min(bars.length-1,Math.round(frac*(bars.length-1))),x=L+i*step+step/2;ctx.fillText(bars[i].d.slice(5),x,h-10)}});
 if(hoverIndex>=0&&hoverIndex<bars.length){{let x=L+hoverIndex*step+step/2;ctx.strokeStyle='#a2b7ca';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,h-B);ctx.stroke();ctx.setLineDash([]);let b=bars[hoverIndex],dayMarks=visual.showTrades?marks.filter(m=>m.entry_date===b.d):[],dayWindows=windows.filter(win=>win.start_index<=hoverIndex&&hoverIndex<=win.end_index);let markText=dayMarks.length?' · 模拟买入 '+dayMarks.map(m=>(m.period==='test'?'测试':'验证')+' '+pct(m.net_return)).join('、'):'';let priceText=`${{b.d}}  开 ${{nice(b.o)}} / 高 ${{nice(b.h)}} / 低 ${{nice(b.l)}} / 收 ${{nice(b.c)}} · 成交量 ${{nice(b.v)}}${{markText}}`;hoverNode.innerHTML=`<div class="hover-price">${{safe(priceText)}}</div>${{dayWindows.map(growthDetails).join('')}}`}}else{{hoverNode.textContent=`${{visual.showGrowth?'绿色阴影是当前条件的历史增长区间。':''}}${{visual.showTrades?'蓝/紫三角是旧策略的模拟买入点。':''}}将鼠标移到 K 线上查看价格。`}}
 canvas.onmousemove=e=>{{let r=canvas.getBoundingClientRect();hoverIndexes[ticker]=Math.max(0,Math.min(bars.length-1,Math.floor((e.clientX-r.left-L)/step)));drawChart(ticker,canvas,hoverNode)}};canvas.onmouseleave=()=>{{hoverIndexes[ticker]=-1;drawChart(ticker,canvas,hoverNode)}};
}}
function commitVisual(recalculate){{normalizeVisual();syncVisualControls();if(recalculate)recalculateGrowth();try{{localStorage.setItem('preopen-growth-visual-v1',JSON.stringify(visual))}}catch(error){{console.warn('无法保存本地可视化选项',error)}}updateList()}}
let visualTimer=null;function scheduleVisualRecalculation(){{clearTimeout(visualTimer);visualTimer=setTimeout(()=>commitVisual(true),120)}}
normalizeVisual();syncVisualControls();recalculateGrowth();
document.getElementById('threshold-slider').addEventListener('input',event=>{{visual.threshold=Number(event.target.value);document.getElementById('growth-threshold').value=visual.threshold;document.getElementById('threshold-value').textContent=visual.threshold+'%';scheduleVisualRecalculation()}});
document.getElementById('growth-threshold').addEventListener('change',event=>{{visual.threshold=Number(event.target.value);commitVisual(true)}});
for(const [id,key] of [['min-calendar','minCalendar'],['max-calendar','maxCalendar'],['min-sessions','minSessions']])document.getElementById(id).addEventListener('change',event=>{{visual[key]=Number(event.target.value);commitVisual(true)}});
for(const [id,key] of [['show-growth','showGrowth'],['show-trades','showTrades'],['show-volume','showVolume']])document.getElementById(id).addEventListener('change',event=>{{visual[key]=event.target.checked;commitVisual(false)}});
document.getElementById('reset-growth').addEventListener('click',()=>{{visual={{...visualDefaults}};commitVisual(true)}});
document.getElementById('search').addEventListener('input',updateList);document.getElementById('fund').addEventListener('change',updateList);document.getElementById('group').addEventListener('change',()=>{{if(document.getElementById('group').value==='composer_ai_2026')document.getElementById('fund').value='';updateList()}});window.addEventListener('resize',updateList);fetch('/api/pinned-stocks').then(response=>{{if(!response.ok)throw new Error(`HTTP ${{response.status}}`);return response.json()}}).then(data=>{{pinnedTickers=new Set(Array.isArray(data.tickers)?data.tickers:[]);updateList()}}).catch(()=>{{updateList();console.warn('置顶记录文件读取失败；页面仍可查看股票，但置顶不能持久化。')}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / "analysis/preopen_three_week/results.json"
    result = json.loads(path.read_text())
    payload = build_payload(root, result)
    labels = growth_labels(payload)
    labels_path = path.with_name("growth_windows.json")
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n")
    output = path.with_name("dashboard.html")
    output.write_text(render(payload))
    print(f"{output} ({output.stat().st_size:,} bytes)")
    print(f"{labels_path} ({labels['stock_count']} stocks, {labels['window_count']} windows)")


if __name__ == "__main__":
    main()
