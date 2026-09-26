"""Deterministic long-only cash simulation. OHLC fills are explicitly proxies."""
from __future__ import annotations
import numpy as np
import pandas as pd

def simulate(signals, rows, labels, paths, grid, cfg, start_date=None, end_date=None):
    start_date = start_date or min(f["outer_start"] for f in cfg["folds"])
    end_date = end_date or cfg["source_end"]
    settings = cfg["portfolio"]
    initial, cost = settings["initial_cash"], settings["cost_bps"]/10000
    cash, positions, trades, rejected = float(initial), {}, [], []
    labels = labels.set_index(["row_id", "expectation_id"])
    signal = signals.merge(rows[["row_id", "symbol", "date", "pos"]], on="row_id", validate="many_to_one")
    signal = signal[signal.p >= cfg["threshold"]].copy()
    if "priority" not in signal:
        signal["priority"] = 0
    signal = signal.sort_values(["date", "priority", "p", "symbol", "binding_id"], ascending=[True, True, False, True, True])
    start = int(np.flatnonzero(grid.date.to_numpy() >= start_date)[0])
    finish = int(np.flatnonzero(grid.date.to_numpy() <= end_date)[-1])
    cash_delta = np.zeros(len(grid))
    asset_value = np.zeros(len(grid))
    for day, candidates in signal.groupby("date", sort=True):
        pos = int(candidates.pos.iloc[0])
        for sym, t in list(positions.items()):
            if t["exit_pos"] < pos:
                cash += t["exit_proceeds"]
                del positions[sym]
        equity = cash
        for sym, t in positions.items():
            equity += t["shares"] * paths[sym][pos-1, 3]
        selected = candidates[~candidates.symbol.isin(positions)].drop_duplicates("symbol").head(settings["top_k"])
        for s in selected.itertuples():
            if len(positions) >= settings["max_positions"]:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "position_limit"})
                continue
            if cash <= 0:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "cash_limit"})
                continue
            # This check occurs AFTER the signal selected a stock. Unavailable outcomes
            # invalidate completeness; they never improve ranking or replace a selection.
            label = labels.loc[(s.row_id, s.expectation_id)]
            if label.status != "mature":
                rejected.append({"date": day, "symbol": s.symbol, "reason": "unresolved_selected_path", "detail": str(label.reason)})
                continue
            entry = float(paths[s.symbol][pos, 0])
            budget = min(cash, equity*settings["max_weight"])
            shares = int(budget/(entry*(1+cost)))
            if shares <= 0:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "insufficient_cash_for_one_share"})
                continue
            exp = next(e for e in cfg["expectations"] if e["id"] == s.expectation_id)
            bars = int(label.hit_minutes/5) if label.hit else exp["days"]*78
            exit_pos = pos+bars-1
            exit_price = entry*(1+exp["target"]) if label.hit else float(paths[s.symbol][exit_pos, 3])
            if exit_pos > finish or not paths[s.symbol][pos:exit_pos+1, 4].all():
                rejected.append({"date": day, "symbol": s.symbol, "reason": "unresolved_selected_path", "detail": "invalid_mark_path"})
                continue
            debit, credit = shares*entry*(1+cost), shares*exit_price*(1-cost)
            fees = shares*(entry+exit_price)*cost
            cash -= debit
            if cash < -1e-7:
                raise AssertionError("Borrowing in cash portfolio")
            t = {"symbol": s.symbol, "row_id": int(s.row_id), "binding_id": s.binding_id,
                 "expectation_id": s.expectation_id, "p": float(s.p), "entry_pos": pos, "exit_pos": exit_pos,
                 "entry_at": grid.start.iloc[pos].isoformat(), "exit_at": grid.end.iloc[exit_pos].isoformat(),
                 "entry_price": entry, "exit_price": exit_price, "shares": shares, "entry_debit": debit,
                 "exit_proceeds": credit, "cost": fees, "pnl": credit-debit, "hit": bool(label.hit),
                 "exit_reason": "target_high_proxy" if label.hit else "horizon_close", "holding_days": bars/78}
            trades.append(t)
            positions[s.symbol] = t
            cash_delta[pos] -= debit
            cash_delta[exit_pos] += credit
            # Exit at the end of the bar; mark all earlier bars at their observed close.
            asset_value[pos:exit_pos] += shares*paths[s.symbol][pos:exit_pos, 3]
    cash_curve = initial+np.cumsum(cash_delta)
    equity_curve = cash_curve+asset_value
    values = equity_curve[start:finish+1]
    peak = np.maximum.accumulate(np.r_[initial, values])[1:]
    drawdown = values/peak-1
    end = float(values[-1])
    pnl = sum(t["pnl"] for t in trades)
    if abs(end-initial-pnl) > 1e-6:
        raise AssertionError("Portfolio cash/P&L reconciliation failed")
    daily = []
    for day, ix in grid.iloc[start:finish+1].groupby("date", sort=True).groups.items():
        i = int(max(ix))
        daily.append({"date": day, "equity": round(float(equity_curve[i]), 2), "cash": round(float(cash_curve[i]), 2)})
    unresolved = [x for x in rejected if x["reason"] == "unresolved_selected_path"]
    complete = not unresolved
    result = {"backtest_complete": complete, "initial_cash": initial,
              "end_balance": end if complete else None, "net_return": end/initial-1 if complete else None,
              "max_drawdown": float(drawdown.min()) if complete else None,
              "cost": sum(t["cost"] for t in trades), "trade_n": len(trades),
              "trade_hit_n": sum(t["hit"] for t in trades),
              "trade_hit_rate": sum(t["hit"] for t in trades)/len(trades) if trades else None,
              "win_rate": sum(t["pnl"] > 0 for t in trades)/len(trades) if trades else None,
              "mean_holding_days": np.mean([t["holding_days"] for t in trades]) if trades else None,
              "unresolved_n": len(unresolved), "open_positions": 0, "unrealized_pnl": 0.,
              "decision_start": start_date, "decision_end": max(f["outer_end"] for f in cfg["folds"]), "valuation_end": end_date,
              "simulation_status": "complete" if complete else "incomplete_selected_path"}
    return result, {"trades": trades, "daily_equity": daily, "rejections": rejected}
