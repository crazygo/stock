#!/usr/bin/env python3
"""Pre-open, 15-session +20% AI-stock research model and action audit.

Uses only the prior completed session and ETF holdings available by then.
Fits once on an early interval, picks decision thresholds on a disjoint
validation interval, and reports the later test interval without tuning it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from backtest_ai_strategies import (Backtest, COST_PER_SIDE, MIN_ANNUAL_REVENUE,
                                    MIN_DOLLAR_VOLUME_20, MIN_PRICE, mean, median)

HORIZON = 15
TARGET = 0.20
STOP = -0.10
MAX_OPEN_GAP = 0.03
MAX_BUYS_PER_DAY = 3
CAPITAL_SLOTS = 10
FEATURES = (
    "return_1", "return_3", "return_5", "return_10", "return_20", "return_60",
    "drawdown_20", "drawdown_60", "volatility_20", "volume_ratio_5_20",
    "relative_5_qqq", "relative_20_qqq", "qqq_return_5", "qqq_over_ma20",
    "breadth_5", "fund_count", "capacity_15", "revenue_log", "cashflow_margin", "net_margin",
)
TRAIN_START, TRAIN_END = date(2025, 11, 3), date(2026, 3, 6)
VALID_START, VALID_END = date(2026, 4, 1), date(2026, 5, 15)
TEST_START, TEST_END = date(2026, 6, 15), date(2026, 8, 31)


class FundMembership:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text())
        self.funds = sorted(payload["funds"])
        self.snapshots = {fund: sorted(
            (snapshot for snapshot in payload["snapshots"] if snapshot["fund"] == fund),
            key=lambda snapshot: snapshot["as_of"]
        ) for fund in self.funds}
        self.all_tickers = {h["ticker"] for snapshots in self.snapshots.values()
                            for snapshot in snapshots for h in snapshot["holdings"]}
        self._cache: dict[str, dict[str, list[str]]] = {}

    def members(self, asof: date) -> dict[str, list[str]]:
        key = asof.isoformat()
        if key not in self._cache:
            found: dict[str, list[str]] = defaultdict(list)
            for fund, snapshots in self.snapshots.items():
                eligible = [s for s in snapshots if s["as_of"] <= key]
                if eligible:
                    for holding in eligible[-1]["holdings"]:
                        found[holding["ticker"]].append(fund)
            self._cache[key] = dict(found)
        return self._cache[key]


class ThreeWeekStudy:
    def __init__(self, market: Backtest, membership: FundMembership):
        self.market = market
        self.membership = membership
        self._outcomes: dict[tuple[str, int], dict | None] = {}
        self._capacity: dict[tuple[str, int], float] = {}

    def outcome(self, symbol: str, i: int) -> dict | None:
        key = symbol, i
        if key in self._outcomes:
            return self._outcomes[key]
        market = self.market
        if i + HORIZON >= len(market.days):
            self._outcomes[key] = None
            return None
        future = [market.bar(symbol, j) for j in range(i + 1, i + HORIZON + 1)]
        if any(bar is None for bar in future):
            self._outcomes[key] = None
            return None
        entry = float(future[0]["o"])
        prior_close = market.close(symbol, i)
        if entry <= 0 or not prior_close:
            self._outcomes[key] = None
            return None
        first_target = None
        first_stop = None
        exit_index = i + HORIZON
        exit_price = float(future[-1]["c"])
        worst_low = 1.0
        for step, bar in enumerate(future, 1):
            open_price = float(bar["o"])
            worst_low = min(worst_low, float(bar["l"]) / entry - 1)
            hit_stop = float(bar["l"]) <= entry * (1 + STOP)
            hit_target = float(bar["h"]) >= entry * (1 + TARGET)
            if hit_target and first_target is None:
                first_target = step
            if hit_stop and first_stop is None:
                first_stop = step
            if hit_stop or hit_target:
                # A day touching both levels is conservatively a stop loss.
                if hit_stop:
                    exit_price = min(open_price, entry * (1 + STOP)) if open_price < entry * (1 + STOP) else entry * (1 + STOP)
                    exit_reason = "stop"
                else:
                    exit_price = entry * (1 + TARGET)
                    exit_reason = "target"
                exit_index = i + step
                break
        else:
            exit_reason = "time"
        result = {
            "entry_index": i + 1, "entry_date": market.days[i + 1].isoformat(),
            "entry": entry, "open_gap": entry / prior_close - 1,
            "exit_index": exit_index, "exit_date": market.days[exit_index].isoformat(),
            "exit_price": exit_price, "exit_reason": exit_reason,
            "target_touched": first_target is not None,
            "target_before_stop": exit_reason == "target",
            "stop_touched": first_stop is not None,
            "max_adverse": worst_low,
            "net_return": exit_price / entry - 1 - 2 * COST_PER_SIDE,
            "terminal_return": float(future[-1]["c"]) / entry - 1 - 2 * COST_PER_SIDE,
        }
        self._outcomes[key] = result
        return result

    def capacity(self, symbol: str, i: int) -> float:
        key = symbol, i
        if key not in self._capacity:
            past = []
            for j in range(max(60, i - 120), i - HORIZON + 1):
                if j + HORIZON > i:
                    continue
                out = self.outcome(symbol, j)
                if out:
                    past.append(float(out["target_before_stop"]))
            # Shrink sparse stock histories toward a neutral 15% prior.
            self._capacity[key] = (sum(past) + 3.0) / (len(past) + 20.0)
        return self._capacity[key]

    def features(self, symbol: str, i: int, members: dict[str, list[str]], breadth: float) -> dict | None:
        market = self.market
        closes = market.closes(symbol, i - 60, i)
        qqq = market.closes("QQQ", i - 20, i)
        bars = [market.bar(symbol, j) for j in range(i - 19, i + 1)]
        quality = market.quality(symbol, i)
        if not closes or not qqq or any(bar is None for bar in bars) or not quality:
            return None
        returns = [closes[j] / closes[j - 1] - 1 for j in range(1, len(closes))]
        volumes = [float(bar["v"]) for bar in bars]
        earlier_vol = mean(volumes[:15])
        stock5, stock20 = market.ret(symbol, i, 5), market.ret(symbol, i, 20)
        qqq5, qqq20 = market.ret("QQQ", i, 5), market.ret("QQQ", i, 20)
        if None in (stock5, stock20, qqq5, qqq20) or not earlier_vol:
            return None
        revenue = quality["revenue"]
        values = {
            "return_1": market.ret(symbol, i, 1),
            "return_3": market.ret(symbol, i, 3),
            "return_5": stock5,
            "return_10": market.ret(symbol, i, 10),
            "return_20": stock20,
            "return_60": market.ret(symbol, i, 60),
            "drawdown_20": closes[-1] / max(closes[-20:]) - 1,
            "drawdown_60": closes[-1] / max(closes[-60:]) - 1,
            "volatility_20": statistics.pstdev(returns[-20:]),
            "volume_ratio_5_20": mean(volumes[-5:]) / earlier_vol,
            "relative_5_qqq": stock5 - qqq5,
            "relative_20_qqq": stock20 - qqq20,
            "qqq_return_5": qqq5,
            "qqq_over_ma20": qqq[-1] / mean(qqq[-20:]) - 1,
            "breadth_5": breadth,
            "fund_count": len(members[symbol]),
            "capacity_15": self.capacity(symbol, i),
            "revenue_log": math.log10(revenue),
            "cashflow_margin": quality["operating_cashflow"] / revenue,
            "net_margin": quality["net_income"] / revenue,
        }
        if any(value is None or not math.isfinite(value) for value in values.values()):
            return None
        return {key: float(max(-10, min(10, value))) for key, value in values.items()}

    def rows(self) -> tuple[list[dict], dict]:
        market = self.market
        rows = []
        counts = Counter()
        for i in range(61, len(market.days)):
            day = market.days[i]
            members = self.membership.members(day)
            if not members:
                continue
            eligible = [s for s in members if market.eligible(s, i, require_next_bar=False)]
            if day == market.days[-1]:
                counts["latest_members"] = len(members)
                counts["latest_quality_pass"] = len(eligible)
            five_day = [market.ret(s, i, 5) for s in eligible]
            valid_returns = [r for r in five_day if r is not None]
            breadth = sum(r > 0 for r in valid_returns) / len(valid_returns) if valid_returns else 0.5
            for symbol in eligible:
                features = self.features(symbol, i, members, breadth)
                if not features:
                    counts["feature_missing"] += 1
                    continue
                outcome = self.outcome(symbol, i)
                rows.append({"asof_index": i, "asof": day.isoformat(),
                             "ticker": symbol, "funds": members[symbol],
                             "quality": market.quality(symbol, i), "close": market.close(symbol, i),
                             "features": features, "outcome": outcome})
        counts["feature_rows"] = len(rows)
        return rows, dict(counts)


class LinearModel:
    def fit(self, rows: list[dict]) -> None:
        x = np.asarray([[row["features"][feature] for feature in FEATURES] for row in rows], dtype=float)
        self.mu = x.mean(axis=0)
        self.sd = np.maximum(x.std(axis=0), 1e-5)
        z = np.clip((x - self.mu) / self.sd, -5, 5)
        z = np.column_stack((np.ones(len(z)), z))
        self.weights = {}
        for label, values in {
            "hit": [row["outcome"]["target_before_stop"] for row in rows],
            "down": [row["outcome"]["stop_touched"] for row in rows],
        }.items():
            y = np.asarray(values, dtype=float)
            weights = np.zeros(z.shape[1])
            weights[0] = math.log(max(1e-4, min(1 - 1e-4, y.mean())) / max(1e-4, 1 - y.mean()))
            for _ in range(500):
                p = 1 / (1 + np.exp(-np.clip(z @ weights, -20, 20)))
                gradient = z.T @ (p - y) / len(y)
                gradient[1:] += 0.01 * weights[1:]
                weights -= 0.15 * gradient
            self.weights[label] = weights
        y_net = np.asarray([max(-0.40, min(0.30, row["outcome"]["net_return"])) for row in rows])
        penalty = np.eye(z.shape[1]) * len(z) * 0.15
        penalty[0, 0] = 0
        self.weights["net"] = np.linalg.solve(z.T @ z + penalty, z.T @ y_net)
        self.train_rate = float(np.mean([row["outcome"]["target_before_stop"] for row in rows]))

    def predict(self, rows: list[dict]) -> None:
        if not rows:
            return
        x = np.asarray([[row["features"][feature] for feature in FEATURES] for row in rows], dtype=float)
        z = np.column_stack((np.ones(len(x)), np.clip((x - self.mu) / self.sd, -5, 5)))
        for label in ("hit", "down"):
            p = 1 / (1 + np.exp(-np.clip(z @ self.weights[label], -20, 20)))
            for row, value in zip(rows, p):
                row["p_" + label] = float(value)
        expected = z @ self.weights["net"]
        for row, value in zip(rows, expected):
            row["expected_net"] = float(value)


def period_rows(rows: list[dict], start: date, end: date) -> list[dict]:
    return [row for row in rows if start <= date.fromisoformat(row["asof"]) <= end and row["outcome"]]


def simulate_buys(rows: list[dict], p_cut: float, net_cut: float, market: Backtest) -> dict:
    by_index: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_index[row["asof_index"]].append(row)
    selected = []
    active: dict[int, dict] = {}
    slots = [1.0] * CAPITAL_SLOTS
    daily_values = []
    buy_days = 0
    for i in range(min(by_index), max(by_index) + HORIZON + 1):
        for slot, trade in list(active.items()):
            if trade["outcome"]["exit_index"] <= i:
                slots[slot] *= 1 + trade["outcome"]["net_return"]
                del active[slot]
        candidates = sorted(
            (row for row in by_index.get(i, [])
             if row["p_hit"] >= p_cut and row["expected_net"] >= net_cut
             and row["outcome"]["open_gap"] <= MAX_OPEN_GAP),
            key=lambda row: (row["expected_net"], row["p_hit"]), reverse=True,
        )
        buys_today = 0
        for row in candidates:
            if buys_today >= MAX_BUYS_PER_DAY or any(x["ticker"] == row["ticker"] for x in active.values()):
                continue
            free = next((slot for slot in range(CAPITAL_SLOTS) if slot not in active), None)
            if free is None:
                break
            active[free] = row
            selected.append(row)
            buys_today += 1
        if buys_today:
            buy_days += 1
        marked = slots.copy()
        for slot, trade in active.items():
            if i < trade["outcome"]["entry_index"]:
                continue
            close = market.close(trade["ticker"], i)
            if close:
                marked[slot] = slots[slot] * (close / trade["outcome"]["entry"] - 1 - 2 * COST_PER_SIDE + 1)
        daily_values.append(sum(marked) / CAPITAL_SLOTS)
    peak, max_drawdown = 1.0, 0.0
    for value in daily_values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1)
    return {"trades": len(selected), "buy_days": buy_days,
            "target_rate": mean([float(r["outcome"]["target_before_stop"]) for r in selected]),
            "mean_net_return": mean([r["outcome"]["net_return"] for r in selected]),
            "median_net_return": median([r["outcome"]["net_return"] for r in selected]),
            "loss_trade_rate": mean([float(r["outcome"]["net_return"] < 0) for r in selected]),
            "stop_rate": mean([float(r["outcome"]["exit_reason"] == "stop") for r in selected]),
            "portfolio_return": sum(slots) / CAPITAL_SLOTS - 1,
            "marked_max_drawdown": max_drawdown,
            "selected": selected}


def choose_threshold(validation: list[dict], market: Backtest) -> tuple[float, float, dict, list[dict]]:
    candidates = []
    for p_cut in (0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12):
        for net_cut in (-0.10, -0.08, -0.06, -0.04, -0.02):
            result = simulate_buys(validation, p_cut, net_cut, market)
            if result["trades"] >= 15 and result["buy_days"] >= 8:
                # The activity floor prevents a zero-trade policy from winning.
                utility = result["portfolio_return"] + 0.5 * result["marked_max_drawdown"]
                candidates.append({"p_cut": p_cut, "net_cut": net_cut,
                                   "utility": utility, "trades": result["trades"],
                                   "buy_days": result["buy_days"],
                                   "portfolio_return": result["portfolio_return"],
                                   "marked_max_drawdown": result["marked_max_drawdown"]})
    if not candidates:
        raise ValueError("No validation threshold produced >=15 trades on >=8 days; strategy lacks coverage")
    best = max(candidates, key=lambda row: (row["utility"], row["trades"]))
    result = simulate_buys(validation, best["p_cut"], best["net_cut"], market)
    return best["p_cut"], best["net_cut"], result, candidates


def summarize_test(test: list[dict], buys: dict, market: Backtest) -> dict:
    selected = buys["selected"]
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in test:
        by_day[row["asof"]].append(row)
    same_day_base = mean([mean([x["outcome"]["net_return"] for x in by_day[row["asof"]]]) for row in selected])
    winner_days = {day for day, day_rows in by_day.items() if any(x["outcome"]["target_before_stop"] for x in day_rows)}
    captured_days = {row["asof"] for row in selected if row["outcome"]["target_before_stop"]}
    qqq_start = market.close("QQQ", min(row["asof_index"] for row in test) + 1)
    qqq_end = market.close("QQQ", max(row["asof_index"] for row in test) + HORIZON)
    y = np.asarray([row["outcome"]["target_before_stop"] for row in test], dtype=float)
    p = np.asarray([row["p_hit"] for row in test], dtype=float)
    return {
        **{k: v for k, v in buys.items() if k != "selected"},
        "test_stock_days": len(test), "test_days": len(by_day),
        "test_base_target_rate": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "constant_base_brier": float(np.mean((y.mean() - y) ** 2)),
        "matched_day_mean_net_return": same_day_base,
        "winner_day_capture_rate": len(captured_days) / len(winner_days) if winner_days else None,
        "days_with_buy_rate": buys["buy_days"] / len(by_day),
        "qqq_buy_hold_return": qqq_end / qqq_start - 1 - 2 * COST_PER_SIDE if qqq_start and qqq_end else None,
    }


def quality_gate(test_summary: dict) -> dict:
    """Freeze the release decision before looking at current model picks."""
    checks = {
        "至少30笔组合交易": test_summary["trades"] >= 30,
        "至少8个买入日": test_summary["buy_days"] >= 8,
        "平均交易净收益为正": test_summary["mean_net_return"] > 0,
        "平均交易净收益高于同日候选基准": (
            test_summary["mean_net_return"] > test_summary["matched_day_mean_net_return"]
        ),
        "10槽位组合收益为正": test_summary["portfolio_return"] > 0,
        "组合最大回撤不低于负20%": test_summary["marked_max_drawdown"] >= -0.20,
        "目标概率Brier优于常数基准": test_summary["brier"] <= test_summary["constant_base_brier"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def ineligible_reason(market: Backtest, symbol: str, i: int) -> str:
    if symbol not in market.sec:
        return "未匹配到可核查的 SEC 年报事实"
    quality = market.quality(symbol, i)
    if quality is None:
        return "缺少当时已披露且口径完整的营收、经营现金流或净利润年报"
    failed = []
    if quality["revenue"] < MIN_ANNUAL_REVENUE:
        failed.append("年营收不足 10 亿美元")
    if quality["operating_cashflow"] <= 0:
        failed.append("经营现金流非正")
    if quality["net_income"] <= 0:
        failed.append("净利润非正")
    if failed:
        return "、".join(failed)
    bars = [market.bar(symbol, j) for j in range(i - 19, i + 1)]
    if any(bar is None for bar in bars):
        return "缺少连续 20 个交易日行情"
    if float(bars[-1]["c"]) < MIN_PRICE:
        return "股价低于 5 美元"
    if mean([float(bar["c"]) * float(bar["v"]) for bar in bars]) < MIN_DOLLAR_VOLUME_20:
        return "过去 20 日平均成交额不足 1000 万美元"
    return "缺少完整 60 日特征或历史样本"


def current_actions(current: list[dict], p_cut: float, net_cut: float, market: Backtest,
                    test: list[dict], selected_test: list[dict], gate: dict) -> list[dict]:
    by_symbol = defaultdict(list)
    for row in test:
        by_symbol[row["ticker"]].append(row)
    selected_by_symbol = defaultdict(list)
    for row in selected_test:
        selected_by_symbol[row["ticker"]].append(row)
    ranked = sorted(current, key=lambda row: (row["expected_net"], row["p_hit"]), reverse=True)
    buys = 0
    result = []
    for row in ranked:
        if row["p_hit"] >= p_cut and row["expected_net"] >= net_cut and buys < MAX_BUYS_PER_DAY:
            research_action = "买入"
            buys += 1
            research_reason = "原始模型分数通过验证阈值，按分数进入每日前三"
        elif row["expected_net"] <= -0.02 and row["p_down"] >= 0.35:
            research_action = "卖出"
            research_reason = "原始模型预估三周收益偏负，且跌破风险线分数较高"
        else:
            research_action = "不动"
            research_reason = "原始模型未达到买入或卖出条件"
        history = by_symbol[row["ticker"]]
        historical_buys = [x for x in history if x["p_hit"] >= p_cut and x["expected_net"] >= net_cut]
        portfolio_buys = selected_by_symbol[row["ticker"]]
        action = research_action if gate["passed"] else "无法判断"
        reason = research_reason if gate["passed"] else "样本外组合亏损且质量门槛未通过；原始模型信号仅供研究"
        result.append({
            "ticker": row["ticker"], "action": action, "reason": reason,
            "quality_pass": True,
            "research_action": research_action, "research_reason": research_reason,
            "funds": row["funds"], "asof": row["asof"], "close": row["close"],
            "raw_p_target": row["p_hit"], "raw_p_stop": row["p_down"],
            "raw_expected_net": row["expected_net"],
            "historical_test_signals": len(historical_buys),
            "historical_test_hit_rate": mean([float(x["outcome"]["target_before_stop"]) for x in historical_buys]),
            "historical_test_mean_net": mean([x["outcome"]["net_return"] for x in historical_buys]),
            "test_opportunities": len(history),
            "test_opportunity_hit_rate": mean([float(x["outcome"]["target_before_stop"]) for x in history]),
            "test_opportunity_mean_net": mean([x["outcome"]["net_return"] for x in history]),
            "portfolio_test_buys": len(portfolio_buys),
            "portfolio_test_hit_rate": mean([float(x["outcome"]["target_before_stop"]) for x in portfolio_buys]),
            "portfolio_test_mean_net": mean([x["outcome"]["net_return"] for x in portfolio_buys]),
            "quality": row["quality"],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("market_data/us"))
    parser.add_argument("--candidates", type=Path, default=Path("ai_universe_candidates.json"))
    parser.add_argument("--fund-holdings", type=Path, default=Path("data/ai_fund_holdings.json"))
    parser.add_argument("--sec-facts", type=Path, default=Path("data/ai_sec_annual_facts.json"))
    parser.add_argument("--latest-snapshot", type=Path, default=Path("data/futu_snapshot_2026-09-22.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/preopen_three_week"))
    args = parser.parse_args()
    membership = FundMembership(args.fund_holdings)
    market = Backtest(args.archive_dir, args.candidates, args.sec_facts,
                      extra_tickers=membership.all_tickers,
                      extra_snapshot=args.latest_snapshot, minimum_history=60)
    study = ThreeWeekStudy(market, membership)
    rows, coverage = study.rows()
    training = period_rows(rows, TRAIN_START, TRAIN_END)
    validation = period_rows(rows, VALID_START, VALID_END)
    test = period_rows(rows, TEST_START, TEST_END)
    latest = [row for row in rows if row["asof"] == market.days[-1].isoformat()]
    if min(map(len, (training, validation, test, latest))) == 0:
        raise ValueError(f"insufficient rows train={len(training)} validation={len(validation)} test={len(test)} current={len(latest)}")
    model = LinearModel()
    model.fit(training)
    for batch in (validation, test, latest):
        model.predict(batch)
    p_cut, net_cut, valid_buys, threshold_grid = choose_threshold(validation, market)
    test_buys = simulate_buys(test, p_cut, net_cut, market)
    test_summary = summarize_test(test, test_buys, market)
    gate = quality_gate(test_summary)
    current = current_actions(latest, p_cut, net_cut, market, test, test_buys["selected"], gate)
    current_members = membership.members(market.days[-1])
    current_valid = {row["ticker"] for row in latest}
    for symbol, funds in current_members.items():
        if symbol not in current_valid:
            reason = ineligible_reason(market, symbol, len(market.days) - 1)
            current.append({"ticker": symbol, "funds": funds, "action": "无法判断",
                            "quality_pass": False,
                            "reason": reason, "asof": market.days[-1].isoformat(),
                            "close": market.close(symbol, len(market.days) - 1),
                            "raw_p_target": None, "raw_p_stop": None, "raw_expected_net": None,
                            "research_action": "无法判断", "research_reason": reason,
                            "historical_test_signals": 0, "historical_test_hit_rate": None,
                            "historical_test_mean_net": None, "test_opportunities": 0,
                            "test_opportunity_hit_rate": None, "test_opportunity_mean_net": None,
                            "portfolio_test_buys": 0, "portfolio_test_hit_rate": None,
                            "portfolio_test_mean_net": None,
                            "quality": market.quality(symbol, len(market.days) - 1)})
    action_counts = dict(Counter(row["action"] for row in current))
    result = {
        "model": "preopen_prior_close_v0", "asof_session": market.days[-1].isoformat(),
        "next_decision_session": "next U.S. session pre-open; conditional entry at market open",
        "periods": {"train": [str(TRAIN_START), str(TRAIN_END)],
                    "validation": [str(VALID_START), str(VALID_END)],
                    "test": [str(TEST_START), str(TEST_END)]},
        "rules": {"horizon_sessions": HORIZON, "target": TARGET, "stop": STOP,
                  "round_trip_cost": 2 * COST_PER_SIDE, "max_open_gap": MAX_OPEN_GAP,
                  "max_buys_per_day": MAX_BUYS_PER_DAY, "capital_slots": CAPITAL_SLOTS,
                  "p_buy_threshold": p_cut, "expected_net_buy_threshold": net_cut},
        "coverage": {**coverage, "train_rows": len(training), "validation_rows": len(validation),
                     "test_rows": len(test), "current_actions": action_counts},
        "train_base_hit_rate": model.train_rate,
        "validation": {k: v for k, v in valid_buys.items() if k != "selected"},
        "test": test_summary,
        "quality_gate": gate,
        "threshold_grid": threshold_grid,
        "current": current,
        "test_trades": [{"asof": row["asof"], "ticker": row["ticker"],
                         "entry_date": row["outcome"]["entry_date"],
                         "exit_date": row["outcome"]["exit_date"],
                         "exit_reason": row["outcome"]["exit_reason"],
                         "net_return": row["outcome"]["net_return"],
                         "p_target": row["p_hit"], "expected_net": row["expected_net"]}
                        for row in test_buys["selected"]],
        "validation_trades": [{"asof": row["asof"], "ticker": row["ticker"],
                               "entry_date": row["outcome"]["entry_date"],
                               "exit_date": row["outcome"]["exit_date"],
                               "exit_reason": row["outcome"]["exit_reason"],
                               "net_return": row["outcome"]["net_return"]}
                              for row in valid_buys["selected"]],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir / "results.json"),
                      "coverage": result["coverage"], "validation": result["validation"],
                      "test": result["test"], "current_actions": action_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
