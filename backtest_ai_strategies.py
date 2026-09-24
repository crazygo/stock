#!/usr/bin/env python3
"""Point-in-time prototype backtest for seven AI-stock research hypotheses.

Run from the repository root after build_ai_sec_facts.py. All entry decisions
use the close of the signal session; execution uses the next session's open.
The seven strategies intentionally keep separate labels and denominators.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

COST_PER_SIDE = 0.001
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME_20 = 10_000_000
MIN_ANNUAL_REVENUE = 1_000_000_000
SEC_FRESHNESS_DAYS = 550
EARLIEST_TEST = date(2026, 5, 1)
# The archive stores independently fetched adjusted=true daily responses. APH's
# 2026-09-02 2:1 split is not reflected in its earlier archived bars, producing
# a false -51% one-day move. Align earlier OHLC and volume to the later share
# basis. Source: https://investor.amphenol.com/news-and-events/news-details/2026/Amphenol-Announces-Two-for-One-Stock-Split-and-Third-Quarter-2026-Dividend/default.aspx
ARCHIVE_SPLIT_REPAIRS = {"APH": [(date(2026, 9, 2), 2.0)]}


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def pct(xs: list[float], quantile: float) -> float | None:
    if not xs:
        return None
    arr = sorted(xs)
    pos = (len(arr) - 1) * quantile
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)


def bootstrap_precision(rows: list[dict], n: int = 500) -> list[float] | None:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_day[row["date"]].append(row)
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = random.Random(20260923)
    values = []
    for _ in range(n):
        sampled = [row for _ in days for row in by_day[rng.choice(days)]]
        signals = [row for row in sampled if row["signal"]]
        if signals:
            values.append(sum(row["hit"] for row in signals) / len(signals))
    if not values:
        return None
    return [pct(values, 0.025), pct(values, 0.975)]


def bootstrap_mean_delta(rows: list[dict], field: str, n: int = 500) -> list[float] | None:
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_day[row["date"]].append(row[field])
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = random.Random(20260923)
    values = []
    for _ in range(n):
        sample = [value for _ in days for value in by_day[rng.choice(days)]]
        values.append(statistics.fmean(sample))
    return [pct(values, 0.025), pct(values, 0.975)]


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def render_report(result: dict) -> str:
    data, strategies = result["data"], result["strategies"]
    names = {"1": "AI 外因洼地", "2": "四日洼地初涨", "3": "历史增长能力筛选", "5": "七日 +8% 动量"}
    lines = [
        "# 七个 AI 股票策略：基础版回测", "",
        "## 先看结论", "",
        "这是一轮历史研究结果，不是对未来交易的胜率承诺。所谓“命中”，指在指定期限内日内最高价碰到目标；它不等于实际赚到相同金额。交易模拟在信号日收盘后决定、下一交易日开盘买入，按每次买卖各 0.1% 扣成本。", "",
        f"- 本地行情：{data['archive_start']} 至 {data['archive_end']}，{data['archived_sessions']} 个交易日；评估信号从 {data['test_start']} 开始。",
        f"- AI 业务候选池 {data['candidate_count']} 只；SEC 财务数据覆盖 {data['sec_covered_count']} 只；评估起点实际通过财务、价格和流动性门槛 **{data['quality_at_test_start_count']} 只**。",
        "- 财务门槛：当时已披露最近年报营收 ≥10 亿美元、经营现金流和净利润均为正；股价 ≥5 美元，20 日平均成交额 ≥1000 万美元。",
        f"- 30 自然日目标可完整评估至 {strategies['1']['opportunity_end']}；5 交易日目标可完整评估至 {strategies['2']['opportunity_end']}。", "",
        "## 四个入场信号与筛选器", "",
        "| 策略 | 有效信号数 | 目标命中率 | 同日同业务组基准 | 去重后次数 / 命中率 | 平均净收益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("1", "2", "3", "5"):
        m = strategies[key]
        lines.append(
            f"| {key}. {names[key]} | {m['signals']} | {percent(m['hit_rate'])} | "
            f"{percent(m['same_date_category_base_hit_rate'])} | {m['nonoverlap_signals']} / "
            f"{percent(m['nonoverlap_hit_rate'])} | {percent(m['mean_net_return'])} |"
        )
    lines.extend([
        "", "策略 1、3 的目标是 **30 个自然日内先达到 +20%、且此前未先触及 -12%**；策略 2、5 的目标是 **未来 5 个交易日内日内最高价达到 +8%**。因此两类命中率不能直接横向比较。策略 3 是股票筛选器，连续几天选中同一只股票会重复计入原始信号数；“去重后”按股票限制 30 天内只计一次，其他短期信号限制 7 天内一次。", "",
        "基础版使用的可复算规则：", "",
        "- **1**：20 日高点回撤 10–35%，市场或业务组基准近 10 日跌逾 2%，个股近 5 日相对业务组不落后超过 5%，当天出现首个回升，历史 +20% 能力不低于池内中位数。它只能近似识别外部压力，不能确认公司没有结构性损害。",
        "- **2**：至少连续两日下跌、局部跌幅 ≥3%，随后首个上涨日，同时相对 QQQ 转强并有基本成交量确认。",
        "- **3**：只使用信号日前已走完 30 天的历史样本，统计过去最多 150 个交易日的达标频率，每日取池内前 20%。它是先筛股票，不是买点。",
        "- **5**：QQQ 站上 20 日线，业务组过半股票近五日上涨，个股近三日加速、近五日跑赢 QQQ、量能放大且不远离均线。它尚未加入 VIX 和实时消息。", "",
        "同日同业务组对照比全池均值更贴近每个信号面对的市场环境。按交易日成组重抽样得到的命中率差 95% 区间：", "",
    ])
    for key in ("1", "2", "3", "5"):
        m = strategies[key]
        ci = m["bootstrap_95pct_nonoverlap_excess_hit_rate"]
        lines.append(f"- 策略 {key} 去重后：{percent(m['nonoverlap_hit_rate'] - m['nonoverlap_same_date_category_base_hit_rate'])}；区间 {percent(ci[0]) if ci else '—'} 至 {percent(ci[1]) if ci else '—'}。")
    m6, m7 = strategies["6"], strategies["7"]
    ci6, ci7 = m6["bootstrap_95pct_incremental_return"], m7["bootstrap_95pct_active_incremental_return"]
    lines.extend([
        "", "## 持仓规则", "",
        f"- **6. 30 日 +20% 持有纪律**：在策略 1/2 的 {m6['entries']} 个同股至少间隔 21 个交易日的入场中，持至 +20% 或 30 自然日的平均净收益 {percent(m6['hold_mean_net_return'])}；第 5 个交易日卖出并持现金的对照为 {percent(m6['early_5d_mean_net_return'])}。平均差 {percent(m6['mean_incremental_return'])}，95% 区间 {percent(ci6[0]) if ci6 else '—'} 至 {percent(ci6[1]) if ci6 else '—'}。这只检验“拿住”规则，没有盈利预期、论点破坏或换股模型。",
        f"- **7. 五日持有/卖出/回补**：在第 5 日仍亏损的 {m7['underwater_positions']} 个持仓中，固定量价规则建议卖出等待 {m7['active_sell_wait']} 次，真正按预设触发条件回补 {m7['rebought']} 次。这 {m7['active_sell_wait']} 次相对一直持有的平均差 {percent(m7['active_mean_incremental_return'])}，95% 区间 {percent(ci7[0]) if ci7 else '—'} 至 {percent(ci7[1]) if ci7 else '—'}；跑赢持有的比例 {percent(m7['active_beats_hold_rate'])}。尚未构建前瞻路径分布模型。",
        "", "## 暂无可报准确度的策略", "",
        "**4. 势能传导：暂无可信命中率。** 历史新闻、订单、盈利预期修订及反证需要完整的首次公开时间与当时可见版本。富途新闻搜索虽返回 `publish_time`，但文档只给关键词和最多 100 条，没有按历史时点完整回放的参数。代码已提供严格的事件快照输入模式；只提供今日可搜到的新闻会带来前视和遗漏偏差。", "",
        "## 如何复算与看待结果", "",
        "```bash", "python3 backtest_ai_strategies.py", "```", "",
        "`metrics.json` 保存所有分母、命中率、净收益和区间；`signal_opportunities.csv` 保存每次入场机会；`qualified_universe_2026-05-01.csv` 保存 82 只股票的业务组、年报指标、提交日期和 SEC accession。", "",
        "这些门槛只证明一个可量化的基本业务底座，不证明客户需求、竞争优势和订单未恶化。AI 主题候选名单是在今天人工整理，再回看历史，存在选择偏差；历史股票目录也未完整恢复退市和并购标的。行情按拆股调整、不含股息；本地逐日档案在 APH 2026-09-02 拆股处出现拼接断点，代码按公司公告统一了此前价格和成交量。交易模拟没有税、冲击成本及盘中先后顺序，若同一天触及目标与风险线，按风险先发生处理。数据只有约 14 个月，结果不能当成长期稳健性证据。", "",
        "## 四日洼地的目标敏感性", "",
        "| 持有交易日 | 目标 | 信号次数 | 命中率 | 全池基准 |", "|---:|---:|---:|---:|---:|",
    ])
    for horizon in (3, 5, 7, 10):
        for target in (5, 8, 10):
            item = result["strategy_2_horizon_sensitivity"][f"{horizon}trading_days_{target}pct"]
            lines.append(f"| {horizon} | +{target}% | {item['signals']} | {percent(item['hit_rate'])} | {percent(item['base_hit_rate'])} |")
    lines.extend(["", "数据来源：[SEC EDGAR companyfacts](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)、[Massive 日线说明](https://www.massive.com/docs/rest/stocks/aggregates/custom-bars)、[Amphenol 拆股公告](https://investor.amphenol.com/news-and-events/news-details/2026/Amphenol-Announces-Two-for-One-Stock-Split-and-Third-Quarter-2026-Dividend/default.aspx)、[富途新闻搜索文档](https://openapi.futunn.com/futu-api-doc/quote/get-search-news.html)。", ""])
    return "\n".join(lines)


class Backtest:
    def __init__(self, archive_dir: Path, candidates_path: Path, sec_path: Path,
                 extra_tickers: set[str] | None = None, extra_snapshot: Path | None = None,
                 minimum_history: int = 150):
        manifest = json.loads((archive_dir / "manifest.json").read_text())
        self.days = sorted(date.fromisoformat(d) for d in manifest["archives"])
        snapshot = json.loads(extra_snapshot.read_text()) if extra_snapshot else None
        if snapshot and date.fromisoformat(snapshot["session"]) not in self.days:
            self.days.append(date.fromisoformat(snapshot["session"]))
            self.days.sort()
        self.minimum_history = minimum_history
        self.candidates = json.loads(candidates_path.read_text())
        self.sec = json.loads(sec_path.read_text())["companies"]
        self.category = {}
        for group, details in self.candidates["groups"].items():
            for symbol in details["tickers"]:
                self.category.setdefault(symbol, group)
        for symbol in extra_tickers or set():
            self.category.setdefault(symbol, "etf_discovery")
        self.tickers = sorted(self.category)
        self.benchmarks = {"QQQ", "SOXX", "IGV", "XLU"}
        self.bars: dict[str, list[dict | None]] = {
            symbol: [None] * len(self.days) for symbol in self.tickers + sorted(self.benchmarks)
        }
        for i, day in enumerate(self.days):
            if day.isoformat() in manifest["archives"]:
                path = archive_dir / manifest["archives"][day.isoformat()]["path"]
                with gzip.open(path, "rt") as handle:
                    payload = json.load(handle)
                if payload.get("adjusted") is not True or len(payload.get("results", [])) < 1000:
                    raise ValueError(f"invalid archive: {day}")
                results = payload["results"]
            elif snapshot and day.isoformat() == snapshot["session"]:
                results = list(snapshot["bars"].values())
            else:
                raise ValueError(f"missing market session: {day}")
            for row in results:
                symbol = row.get("T")
                if symbol in self.bars:
                    self.bars[symbol][i] = row
        for symbol, splits in ARCHIVE_SPLIT_REPAIRS.items():
            for split_day, ratio in splits:
                for i, day in enumerate(self.days):
                    bar = self.bars[symbol][i]
                    if bar and day < split_day:
                        bar = dict(bar)
                        for field in ("o", "h", "l", "c", "vw"):
                            if field in bar:
                                bar[field] = float(bar[field]) / ratio
                        bar["v"] = float(bar["v"]) * ratio
                        self.bars[symbol][i] = bar
        self.outcome_cache: dict[tuple[str, int, str], dict | None] = {}
        self.capacity_cache: dict[tuple[str, int], tuple[float, float, int] | None] = {}
        self.quality_cache: dict[tuple[str, int], dict | None] = {}

    def bar(self, symbol: str, i: int) -> dict | None:
        return self.bars.get(symbol, [])[i] if 0 <= i < len(self.days) else None

    def close(self, symbol: str, i: int) -> float | None:
        bar = self.bar(symbol, i)
        return float(bar["c"]) if bar and float(bar.get("c", 0)) > 0 else None

    def ret(self, symbol: str, i: int, n: int) -> float | None:
        a, b = self.close(symbol, i), self.close(symbol, i - n)
        return a / b - 1 if a and b else None

    def closes(self, symbol: str, start: int, end: int) -> list[float] | None:
        values = [self.close(symbol, i) for i in range(start, end + 1)]
        return values if values and all(v is not None for v in values) else None

    def ma(self, symbol: str, i: int, n: int) -> float | None:
        xs = self.closes(symbol, i - n + 1, i)
        return mean(xs) if xs else None

    def benchmark(self, symbol: str) -> str:
        group = self.category[symbol]
        if group in {"semiconductors_and_equipment", "compute_and_network_hardware"}:
            return "SOXX"
        if group == "enterprise_ai_software":
            return "IGV"
        if group == "data_center_power_and_construction":
            return "XLU"
        return "QQQ"

    def quality(self, symbol: str, i: int) -> dict | None:
        key = (symbol, i)
        if key in self.quality_cache:
            return self.quality_cache[key]
        company = self.sec.get(symbol)
        if not company:
            self.quality_cache[key] = None
            return None
        asof = self.days[i].isoformat()
        by_end: dict[str, dict[str, dict]] = defaultdict(dict)
        for metric, facts in company["facts"].items():
            for fact in facts:
                if fact["filed"] > asof or fact["end"] > asof:
                    continue
                current = by_end[fact["end"]].get(metric)
                if current is None or (fact["tag_priority"], -int(fact["filed"].replace("-", ""))) < (
                    current["tag_priority"], -int(current["filed"].replace("-", ""))
                ):
                    by_end[fact["end"]][metric] = fact
        for end in sorted(by_end, reverse=True):
            values = by_end[end]
            if not all(k in values for k in ("revenue", "operating_cashflow", "net_income")):
                continue
            if (self.days[i] - date.fromisoformat(end)).days > SEC_FRESHNESS_DAYS:
                break
            result = {
                "fiscal_end": end,
                "revenue": values["revenue"]["value"],
                "operating_cashflow": values["operating_cashflow"]["value"],
                "net_income": values["net_income"]["value"],
                "latest_filed": max(v["filed"] for v in values.values()),
                "accessions": {metric: values[metric]["accession"] for metric in ("revenue", "operating_cashflow", "net_income")},
            }
            self.quality_cache[key] = result
            return result
        self.quality_cache[key] = None
        return None

    def eligible(self, symbol: str, i: int, require_next_bar: bool = True) -> bool:
        if i < self.minimum_history or (require_next_bar and i + 1 >= len(self.days)):
            return False
        quality = self.quality(symbol, i)
        if not quality or not (
            quality["revenue"] >= MIN_ANNUAL_REVENUE
            and quality["operating_cashflow"] > 0
            and quality["net_income"] > 0
        ):
            return False
        bars = [self.bar(symbol, j) for j in range(i - 19, i + 1)]
        if any(bar is None for bar in bars):
            return False
        return (
            float(bars[-1]["c"]) >= MIN_PRICE
            and mean([float(bar["c"]) * float(bar["v"]) for bar in bars]) >= MIN_DOLLAR_VOLUME_20
        )

    def outcome(self, symbol: str, i: int, kind: str) -> dict | None:
        key = (symbol, i, kind)
        if key in self.outcome_cache:
            return self.outcome_cache[key]
        if kind == "swing_30d":
            indices = [j for j in range(i + 1, len(self.days)) if self.days[j] <= self.days[i] + timedelta(days=30)]
            if self.days[-1] < self.days[i] + timedelta(days=30):
                indices = []
            target, risk = 0.20, -0.12
        elif kind == "short_5d":
            indices = list(range(i + 1, i + 6)) if i + 5 < len(self.days) else []
            target, risk = 0.08, -0.05
        else:
            raise ValueError(kind)
        entry_bar = self.bar(symbol, i + 1)
        if not indices or not entry_bar or any(self.bar(symbol, j) is None for j in indices):
            self.outcome_cache[key] = None
            return None
        entry = float(entry_bar["o"])
        if entry <= 0:
            self.outcome_cache[key] = None
            return None
        first_hit = None
        first_risk = None
        max_gain = -1.0
        min_return = 1.0
        for offset, j in enumerate(indices, 1):
            bar = self.bar(symbol, j)
            gain = float(bar["h"]) / entry - 1
            loss = float(bar["l"]) / entry - 1
            max_gain, min_return = max(max_gain, gain), min(min_return, loss)
            if first_hit is None and gain >= target:
                first_hit = offset
            if first_risk is None and loss <= risk:
                first_risk = offset
        hit = first_hit is not None
        safe_hit = hit and (first_risk is None or first_hit < first_risk)
        exit_price = entry * (1 + target) if hit else float(self.bar(symbol, indices[-1])["c"])
        result = {
            "hit": hit, "safe_hit": safe_hit, "first_hit": first_hit,
            "first_risk": first_risk, "max_gain": max_gain, "min_return": min_return,
            "net_return": exit_price / entry - 1 - 2 * COST_PER_SIDE,
            "terminal_return": float(self.bar(symbol, indices[-1])["c"]) / entry - 1 - 2 * COST_PER_SIDE,
            "entry": entry, "entry_date": self.days[i + 1].isoformat(),
            "exit_date": self.days[indices[-1]].isoformat(),
        }
        self.outcome_cache[key] = result
        return result

    def capacity(self, symbol: str, i: int) -> tuple[float, float, int] | None:
        key = (symbol, i)
        if key in self.capacity_cache:
            return self.capacity_cache[key]
        past = []
        for j in range(max(30, i - 150), i):
            if self.days[j] + timedelta(days=30) >= self.days[i]:
                continue
            outcome = self.outcome(symbol, j, "swing_30d")
            if outcome:
                past.append(outcome)
        result = (
            (sum(x["safe_hit"] for x in past) / len(past), median([x["max_gain"] for x in past]), len(past))
            if len(past) >= 60 else None
        )
        self.capacity_cache[key] = result
        return result

    def s1(self, symbol: str, i: int, capacity_cut: float) -> bool:
        cap = self.capacity(symbol, i)
        c = self.closes(symbol, i - 20, i)
        if not cap or not c or cap[0] < capacity_cut:
            return False
        drawdown = c[-1] / max(c) - 1
        sector = self.benchmark(symbol)
        stock5, sector5, qqq10, sector10 = (
            self.ret(symbol, i, 5), self.ret(sector, i, 5),
            self.ret("QQQ", i, 10), self.ret(sector, i, 10),
        )
        if None in (stock5, sector5, qqq10, sector10):
            return False
        return (
            -0.35 <= drawdown <= -0.10
            and min(qqq10, sector10) <= -0.02
            and stock5 - sector5 >= -0.05
            and c[-1] > c[-2] and c[-2] <= c[-3]
        )

    def s2(self, symbol: str, i: int) -> bool:
        c = self.closes(symbol, i - 5, i)
        if not c or self.bar(symbol, i) is None or self.bar(symbol, i - 1) is None:
            return False
        volumes = [float(self.bar(symbol, j)["v"]) for j in range(i - 5, i)]
        qqq1 = self.ret("QQQ", i, 1)
        return bool(
            c[-4] > c[-3] >= c[-2]
            and c[-2] / c[-4] - 1 <= -0.03
            and c[-1] > c[-2]
            and float(self.bar(symbol, i)["l"]) >= float(self.bar(symbol, i - 1)["l"]) * 0.99
            and c[-1] > (float(self.bar(symbol, i - 1)["h"]) + float(self.bar(symbol, i - 1)["l"])) / 2
            and self.ret(symbol, i, 1) > (qqq1 if qqq1 is not None else 0)
            and float(self.bar(symbol, i)["v"]) >= median(volumes) * 0.8
        )

    def s5(self, symbol: str, i: int, breadth: dict[str, float]) -> bool:
        qqq_ma20 = self.ma("QQQ", i, 20)
        c = self.close(symbol, i)
        ma10, ma20 = self.ma(symbol, i, 10), self.ma(symbol, i, 20)
        stock3, prior3, stock5, qqq5 = (
            self.ret(symbol, i, 3), self.ret(symbol, i - 3, 3),
            self.ret(symbol, i, 5), self.ret("QQQ", i, 5),
        )
        if None in (qqq_ma20, c, ma10, ma20, stock3, prior3, stock5, qqq5):
            return False
        group = self.category[symbol]
        peers = breadth.get(group, 0)
        volumes = [float(self.bar(symbol, j)["v"]) for j in range(i - 19, i)]
        volume_ratio = float(self.bar(symbol, i)["v"]) / median(volumes)
        return bool(
            self.close("QQQ", i) > qqq_ma20
            and peers > 0.55
            and stock3 > 0.02 and stock3 > prior3
            and c > ma10 and stock5 > qqq5 + 0.01
            and volume_ratio > 1.1
            and c / ma20 - 1 < 0.15
        )

    def breadth(self, eligible: list[str], i: int) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for symbol in eligible:
            change = self.ret(symbol, i, 5)
            if change is not None:
                grouped[self.category[symbol]].append(change)
        return {group: sum(x > 0 for x in values) / len(values) for group, values in grouped.items()}

    def event_signals(self, feed_path: Path | None) -> tuple[dict, list[dict]]:
        if feed_path is None:
            return {
                "status": "not_estimable",
                "reason": "No complete point-in-time event, order and estimate-revision snapshots; price-only proxies would test another hypothesis.",
            }, []
        feed = json.loads(feed_path.read_text())
        if feed.get("complete_coverage") is not True or not isinstance(feed.get("events"), list):
            raise ValueError("event feed must declare complete_coverage=true and contain events[]")
        signal_events: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for event in feed["events"]:
            required = {"ticker", "published_at_utc", "source_url", "evidence", "transmission",
                        "persistence", "priced_in", "contradiction"}
            if not required.issubset(event):
                raise ValueError(f"event missing required fields: {sorted(required - set(event))}")
            symbol = event["ticker"]
            if symbol not in self.category:
                continue
            published = datetime.fromisoformat(event["published_at_utc"].replace("Z", "+00:00"))
            if published.tzinfo is None:
                raise ValueError("published_at_utc needs a timezone")
            if published.astimezone(ZoneInfo("America/New_York")).date() < EARLIEST_TEST:
                continue
            for score in ("evidence", "transmission", "persistence", "priced_in", "contradiction"):
                if not 0 <= event[score] <= 1:
                    raise ValueError(f"event {score} must be in [0,1]")
            for i, day in enumerate(self.days):
                close_utc = datetime.combine(day, time(16), ZoneInfo("America/New_York")).astimezone(timezone.utc)
                if close_utc >= published and day >= EARLIEST_TEST:
                    signal_events[(i, symbol)].append(event)
                    break
        rows = []
        for (i, symbol), events in sorted(signal_events.items()):
            if not self.eligible(symbol, i):
                continue
            outcome = self.outcome(symbol, i, "swing_30d")
            if outcome is None:
                continue
            score = max(e["evidence"] * e["transmission"] * e["persistence"]
                        * (1 - e["priced_in"]) * (1 - e["contradiction"]) for e in events)
            rows.append({"date": self.days[i].isoformat(), "ticker": symbol, "score": score,
                         "signal": score >= 0.25, "hit": outcome["safe_hit"],
                         "net_return": outcome["net_return"], "source_urls": " | ".join(e["source_url"] for e in events)})
        selected = [r for r in rows if r["signal"]]
        return {
            "status": "measured_from_supplied_complete_feed",
            "event_stock_days": len(rows), "signals": len(selected),
            "hit_rate": mean([float(r["hit"]) for r in selected]),
            "mean_net_return": mean([r["net_return"] for r in selected]),
        }, rows

    def test_signals(self) -> tuple[dict, list[dict], dict[int, dict[str, set[str]]]]:
        rows = []
        daily_signals = {}
        start_i = next(i for i, day in enumerate(self.days) if day >= EARLIEST_TEST)
        for i in range(start_i, len(self.days)):
            eligible = [symbol for symbol in self.tickers if self.eligible(symbol, i)]
            capacities = {s: self.capacity(s, i) for s in eligible}
            ranked = sorted(
                (s for s in eligible if capacities[s]),
                key=lambda s: (-capacities[s][0], -capacities[s][1], s),
            )
            s3 = set(ranked[:math.ceil(len(ranked) * 0.20)])
            capacity_cut = median([capacities[s][0] for s in ranked]) if ranked else 1.0
            breadth = self.breadth(eligible, i)
            signals = {"1": set(), "2": set(), "3": s3, "5": set()}
            for symbol in eligible:
                if self.s1(symbol, i, capacity_cut):
                    signals["1"].add(symbol)
                if self.s2(symbol, i):
                    signals["2"].add(symbol)
                if self.s5(symbol, i, breadth):
                    signals["5"].add(symbol)
                for kind, strategy, label in [
                    ("swing_30d", "1", "safe_hit"),
                    ("short_5d", "2", "hit"),
                    ("swing_30d", "3", "safe_hit"),
                    ("short_5d", "5", "hit"),
                ]:
                    outcome = self.outcome(symbol, i, kind)
                    if outcome is not None:
                        rows.append({
                            "strategy": strategy, "date": self.days[i].isoformat(),
                            "ticker": symbol, "category": self.category[symbol],
                            "signal": symbol in signals[strategy], "hit": bool(outcome[label]),
                            "target_hit": bool(outcome["hit"]),
                            "risk_first": bool(outcome["first_risk"] is not None and (
                                outcome["first_hit"] is None or outcome["first_risk"] <= outcome["first_hit"]
                            )),
                            "max_gain": outcome["max_gain"], "min_return": outcome["min_return"],
                            "net_return": outcome["net_return"], "first_hit": outcome["first_hit"],
                        })
            daily_signals[i] = signals
        metrics = {}
        for strategy in ("1", "2", "3", "5"):
            all_rows = [r for r in rows if r["strategy"] == strategy]
            selected = [r for r in all_rows if r["signal"]]
            date_base: dict[str, float] = {}
            category_date_base: dict[tuple[str, str], float] = {}
            dates = defaultdict(list)
            category_dates = defaultdict(list)
            for r in all_rows:
                dates[r["date"]].append(float(r["hit"]))
                category_dates[(r["date"], r["category"])].append(float(r["hit"]))
            date_base = {key: statistics.fmean(xs) for key, xs in dates.items()}
            category_date_base = {key: statistics.fmean(xs) for key, xs in category_dates.items()}
            independent = []
            last_selected: dict[str, date] = {}
            cooldown = 30 if strategy in {"1", "3"} else 7
            for r in selected:
                current = date.fromisoformat(r["date"])
                if (current - last_selected.get(r["ticker"], date(1900, 1, 1))).days >= cooldown:
                    independent.append(r)
                    last_selected[r["ticker"]] = current
            precision = mean([float(r["hit"]) for r in selected])
            base = mean([float(r["hit"]) for r in all_rows])
            matched = mean([category_date_base[(r["date"], r["category"])] for r in selected])
            metrics[strategy] = {
                "opportunities": len(all_rows), "signals": len(selected),
                "opportunity_start": all_rows[0]["date"] if all_rows else None,
                "opportunity_end": all_rows[-1]["date"] if all_rows else None,
                "distinct_stocks": len({r["ticker"] for r in selected}),
                "signal_days": len({r["date"] for r in selected}),
                "hit_rate": precision, "base_hit_rate": base,
                "lift": precision / base if precision is not None and base else None,
                "same_date_base_hit_rate": mean([date_base[r["date"]] for r in selected]),
                "same_date_category_base_hit_rate": matched,
                "same_date_category_lift": precision / matched if precision is not None and matched else None,
                "bootstrap_95pct_excess_hit_rate_vs_matched": bootstrap_mean_delta(
                    [{"date": r["date"], "excess": float(r["hit"]) - category_date_base[(r["date"], r["category"])]}
                     for r in selected], "excess"
                ),
                "nonoverlap_signals": len(independent),
                "nonoverlap_hit_rate": mean([float(r["hit"]) for r in independent]),
                "nonoverlap_same_date_category_base_hit_rate": mean([
                    category_date_base[(r["date"], r["category"])] for r in independent
                ]),
                "bootstrap_95pct_nonoverlap_excess_hit_rate": bootstrap_mean_delta(
                    [{"date": r["date"], "excess": float(r["hit"]) - category_date_base[(r["date"], r["category"])]}
                     for r in independent], "excess"
                ),
                "nonoverlap_mean_net_return": mean([r["net_return"] for r in independent]),
                "recall": sum(r["hit"] for r in selected) / sum(r["hit"] for r in all_rows) if sum(r["hit"] for r in all_rows) else None,
                "bootstrap_95pct_hit_rate": bootstrap_precision(all_rows),
                "mean_net_return": mean([r["net_return"] for r in selected]),
                "median_net_return": median([r["net_return"] for r in selected]),
                "mean_first_hit_sessions": mean([r["first_hit"] for r in selected if r["first_hit"] is not None]),
                "risk_first_rate": mean([float(r["risk_first"]) for r in selected]),
                "miss_mean_net_return": mean([r["net_return"] for r in selected if not r["target_hit"]]),
            }
        return metrics, rows, daily_signals

    def four_day_sensitivity(self, signal_rows: list[dict]) -> dict[str, dict]:
        index = {day.isoformat(): i for i, day in enumerate(self.days)}
        opportunities = [r for r in signal_rows if r["strategy"] == "2"]
        result = {}
        for horizon in (3, 5, 7, 10):
            for target in (0.05, 0.08, 0.10):
                all_hits, signal_hits = [], []
                for row in opportunities:
                    i = index[row["date"]]
                    if i + horizon >= len(self.days):
                        continue
                    bars = [self.bar(row["ticker"], j) for j in range(i + 1, i + horizon + 1)]
                    if any(bar is None for bar in bars):
                        continue
                    entry = float(bars[0]["o"])
                    hit = any(float(bar["h"]) >= entry * (1 + target) for bar in bars)
                    all_hits.append(float(hit))
                    if row["signal"]:
                        signal_hits.append(float(hit))
                result[f"{horizon}trading_days_{round(target * 100)}pct"] = {
                    "signals": len(signal_hits), "hit_rate": mean(signal_hits),
                    "base_hit_rate": mean(all_hits),
                }
        return result

    def hold_discipline(self, daily_signals: dict[int, dict[str, set[str]]]) -> tuple[dict, list[dict]]:
        rows = []
        last_entry = {}
        for i, signals in daily_signals.items():
            for symbol in sorted(signals["1"] | signals["2"]):
                if i - last_entry.get(symbol, -1000) < 21:
                    continue
                outcome = self.outcome(symbol, i, "swing_30d")
                if outcome is None:
                    continue
                last_entry[symbol] = i
                entry = outcome["entry"]
                early_exit = self.close(symbol, i + 5)
                if early_exit is None:
                    continue
                early = early_exit / entry - 1 - 2 * COST_PER_SIDE
                hold = outcome["net_return"]
                rows.append({"date": self.days[i].isoformat(), "ticker": symbol,
                             "entry_index": i, "hold_net": hold, "early_5d_net": early,
                             "delta": hold - early, "target_hit": outcome["hit"]})
        return {
            "entries": len(rows), "distinct_stocks": len({r["ticker"] for r in rows}),
            "target_hit_rate": mean([float(r["target_hit"]) for r in rows]),
            "hold_mean_net_return": mean([r["hold_net"] for r in rows]),
            "early_5d_mean_net_return": mean([r["early_5d_net"] for r in rows]),
            "mean_incremental_return": mean([r["delta"] for r in rows]),
            "bootstrap_95pct_incremental_return": bootstrap_mean_delta(rows, "delta"),
            "hold_beats_early_rate": mean([float(r["delta"] > 0) for r in rows]),
        }, rows

    def five_day_decisions(self, hold_rows: list[dict]) -> tuple[dict, list[dict]]:
        decisions = []
        for position in hold_rows:
            symbol, entry_i = position["ticker"], position["entry_index"]
            decision_i = entry_i + 5
            if decision_i + 5 >= len(self.days):
                continue
            entry_bar = self.bar(symbol, entry_i + 1)
            decision_close = self.close(symbol, decision_i)
            if not entry_bar or not decision_close or decision_close >= float(entry_bar["o"]):
                continue
            bars = [self.bar(symbol, j) for j in range(decision_i + 1, decision_i + 6)]
            if any(b is None for b in bars):
                continue
            next_open = float(bars[0]["o"])
            terminal = float(bars[-1]["c"])
            if next_open <= 0:
                continue
            # The original buy is sunk at this decision point. Continuing to
            # hold incurs only the eventual sale cost; selling and reentering
            # incurs three future sides (sell, buy, final sell).
            hold_net = terminal / next_open - 1 - COST_PER_SIDE
            stock5 = self.ret(symbol, decision_i, 5)
            sector5 = self.ret(self.benchmark(symbol), decision_i, 5)
            if stock5 is None or sector5 is None:
                continue
            action = "SELL_WAIT_REBUY" if stock5 <= -0.05 and stock5 - sector5 <= -0.02 else "HOLD"
            rebuy_price = None
            rebuy_date = None
            if action == "SELL_WAIT_REBUY":
                for j in range(decision_i + 1, decision_i + 5):
                    bar = self.bar(symbol, j)
                    previous = self.close(symbol, j - 1)
                    sector_day = self.ret(self.benchmark(symbol), j, 1)
                    stock_day = self.ret(symbol, j, 1)
                    if (float(bar["l"]) <= next_open * 0.95 and float(bar["c"]) > previous
                            and stock_day is not None and sector_day is not None and stock_day > sector_day):
                        rebuy_price = float(self.bar(symbol, j + 1)["o"])
                        rebuy_date = self.days[j + 1].isoformat()
                        break
            if action == "HOLD":
                chosen_net = hold_net
            elif rebuy_price is None:
                chosen_net = -COST_PER_SIDE
            else:
                chosen_net = terminal / rebuy_price - 1 - 3 * COST_PER_SIDE
            decisions.append({
                "date": self.days[decision_i].isoformat(), "ticker": symbol, "action": action,
                "chosen_net": chosen_net, "hold_net": hold_net,
                "incremental": chosen_net - hold_net, "rebuy_date": rebuy_date,
                "down_5pct": min(float(b["l"]) for b in bars) / next_open - 1 <= -0.05,
                "up_5pct": max(float(b["h"]) for b in bars) / next_open - 1 >= 0.05,
            })
        active = [r for r in decisions if r["action"] == "SELL_WAIT_REBUY"]
        return {
            "underwater_positions": len(decisions), "active_sell_wait": len(active),
            "rebought": sum(r["rebuy_date"] is not None for r in active),
            "active_beats_hold_rate": mean([float(r["incremental"] > 0) for r in active]),
            "active_mean_incremental_return": mean([r["incremental"] for r in active]),
            "bootstrap_95pct_active_incremental_return": bootstrap_mean_delta(active, "incremental"),
            "chosen_mean_net_return": mean([r["chosen_net"] for r in decisions]),
            "always_hold_mean_net_return": mean([r["hold_net"] for r in decisions]),
            "future_down_5pct_rate": mean([float(r["down_5pct"]) for r in decisions]),
            "future_up_5pct_rate": mean([float(r["up_5pct"]) for r in decisions]),
        }, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("market_data/us"))
    parser.add_argument("--candidates", type=Path, default=Path("ai_universe_candidates.json"))
    parser.add_argument("--sec-facts", type=Path, default=Path("data/ai_sec_annual_facts.json"))
    parser.add_argument("--event-feed", type=Path, help="Complete publication-dated historical event feed; see event_feed.schema.md")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/ai_strategies_baseline"))
    args = parser.parse_args()
    bt = Backtest(args.archive_dir, args.candidates, args.sec_facts)
    signal_metrics, signal_rows, daily_signals = bt.test_signals()
    s2_sensitivity = bt.four_day_sensitivity(signal_rows)
    event_metrics, event_rows = bt.event_signals(args.event_feed)
    hold_metrics, hold_rows = bt.hold_discipline(daily_signals)
    decisions_metrics, decision_rows = bt.five_day_decisions(hold_rows)
    start_i = next(i for i, d in enumerate(bt.days) if d >= EARLIEST_TEST)
    quality_start = [s for s in bt.tickers if bt.eligible(s, start_i)]
    quality_latest = [s for s in bt.tickers if bt.eligible(s, len(bt.days) - 2)]
    result = {
        "data": {
            "archive_start": bt.days[0].isoformat(), "archive_end": bt.days[-1].isoformat(),
            "archived_sessions": len(bt.days), "test_start": EARLIEST_TEST.isoformat(),
            "signal_at": "session close", "execution_at": "next session open",
            "cost_per_side": COST_PER_SIDE, "quality_rule": "latest filed annual USD revenue >=1B, operating cashflow >0, net income >0; price >=5; average daily dollar volume 20d >=10M",
            "candidate_count": len(bt.tickers), "sec_covered_count": len(bt.sec),
            "quality_at_test_start_count": len(quality_start), "quality_at_test_start": quality_start,
            "quality_near_archive_end_count": len(quality_latest), "quality_near_archive_end": quality_latest,
        },
        "strategies": {
            **signal_metrics,
            "4": event_metrics,
            "6": hold_metrics,
            "7": decisions_metrics,
        },
        "strategy_2_horizon_sensitivity": s2_sensitivity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    with (args.output_dir / "qualified_universe_2026-05-01.csv").open("w", newline="") as handle:
        fields = ["ticker", "company", "category", "ai_link", "fiscal_end", "latest_filed",
                  "annual_revenue_usd", "annual_operating_cashflow_usd", "annual_net_income_usd",
                  "revenue_accession", "operating_cashflow_accession", "net_income_accession", "sec_companyfacts_url"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for symbol in quality_start:
            quality = bt.quality(symbol, start_i)
            category = bt.category[symbol]
            company = bt.sec[symbol]
            writer.writerow({
                "ticker": symbol, "company": company["name"], "category": category,
                "ai_link": bt.candidates["groups"][category]["ai_link"],
                "fiscal_end": quality["fiscal_end"], "latest_filed": quality["latest_filed"],
                "annual_revenue_usd": quality["revenue"],
                "annual_operating_cashflow_usd": quality["operating_cashflow"],
                "annual_net_income_usd": quality["net_income"],
                "revenue_accession": quality["accessions"]["revenue"],
                "operating_cashflow_accession": quality["accessions"]["operating_cashflow"],
                "net_income_accession": quality["accessions"]["net_income"],
                "sec_companyfacts_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{company['cik']:010d}.json",
            })
    for name, rows in (("signal_opportunities.csv", signal_rows), ("event_signals.csv", event_rows),
                       ("hold_comparison.csv", hold_rows), ("five_day_decisions.csv", decision_rows)):
        if rows:
            with (args.output_dir / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
