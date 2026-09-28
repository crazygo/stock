"""Observed-price replay with open positions; labels never enter order selection."""
from __future__ import annotations
import numpy as np
import pandas as pd

def simulate(signals, rows, paths, grid, cfg, start_date, decision_end, valuation_end, actions=None):
    settings = cfg["portfolio"]
    initial, cost = settings["initial_cash"], settings["cost_bps"]/10000
    cash, positions, trades, rejected = float(initial), {}, [], []
    signal = signals.merge(rows[["row_id", "symbol", "date", "pos"]], on="row_id", validate="many_to_one")
    signal = signal[(signal.p >= cfg["threshold"]) & signal.date.between(start_date, decision_end)].copy()
    signal = signal.sort_values(["date", "p", "symbol", "binding_id"], ascending=[True, False, True, True])
    start = int(np.flatnonzero(grid.date.to_numpy() >= start_date)[0])
    finish = int(np.flatnonzero(grid.date.to_numpy() <= valuation_end)[-1])
    cash_delta, asset_value = np.zeros(len(grid)), np.zeros(len(grid))
    for day, candidates in signal.groupby("date", sort=True):
        pos = int(candidates.pos.iloc[0])
        for sym, t in list(positions.items()):
            if t["closed"] and t["exit_pos"] < pos:
                cash += t["exit_proceeds"]
                del positions[sym]
        equity = cash + sum(t["shares"]*paths[sym][pos-1, 3] for sym, t in positions.items())
        selected = candidates[~candidates.symbol.isin(positions)].drop_duplicates("symbol").head(settings["top_k"])
        for s in selected.itertuples():
            if len(positions) >= settings["max_positions"]:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "position_limit"})
                continue
            if cash <= 0:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "cash_limit"})
                continue
            path = paths[s.symbol]
            exp = next(e for e in cfg["expectations"] if e["id"] == s.expectation_id)
            horizon = pos+exp["days"]*78-1
            last = min(horizon, finish)
            entry = float(path[pos, 0])
            if last < pos or not path[pos, 5] or not np.isfinite(entry) or entry <= 0:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "unresolved_selected_path", "detail": "entry_unavailable"})
                continue
            observed = path[pos:last+1]
            # Only examine price records through the valuation cutoff. An early hit
            # may close a trade even when its full future label is still pending.
            hits = np.flatnonzero((observed[:, 1] >= entry*(1+exp["target"])) & (observed[:, 4] > 0))
            hit = bool(len(hits))
            end_pos = pos+int(hits[0]) if hit else last
            affected = [d for d in (actions or {}).get(s.symbol, []) if day <= d <= grid.date.iloc[end_pos]]
            if not path[pos:end_pos+1, 4].all() or affected:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "unresolved_selected_path", "detail": "missing_bars_or_corporate_action"})
                continue
            shares = int(min(cash, equity*settings["max_weight"])/(entry*(1+cost)))
            if shares <= 0:
                rejected.append({"date": day, "symbol": s.symbol, "reason": "insufficient_cash_for_one_share"})
                continue
            closed = hit or horizon <= finish
            end_price = entry*(1+exp["target"]) if hit else float(path[end_pos, 3])
            debit = shares*entry*(1+cost)
            credit = shares*end_price*(1-cost) if closed else 0.
            fee = shares*entry*cost + (shares*end_price*cost if closed else 0.)
            pnl = credit-debit if closed else shares*end_price-debit
            cash -= debit
            if cash < -1e-7:
                raise AssertionError("Borrowing in cash portfolio")
            t = {"symbol": s.symbol, "row_id": int(s.row_id), "binding_id": s.binding_id,
                 "expectation_id": s.expectation_id, "p": float(s.p), "entry_pos": pos,
                 "exit_pos": end_pos if closed else None, "mark_pos": end_pos,
                 "entry_at": grid.start.iloc[pos].isoformat(), "exit_at": grid.end.iloc[end_pos].isoformat() if closed else None,
                 "mark_at": grid.end.iloc[end_pos].isoformat(), "entry_price": entry, "exit_price": end_price if closed else None,
                 "mark_price": end_price, "shares": shares, "entry_debit": debit, "exit_proceeds": credit,
                 "cost": fee, "pnl": pnl if closed else None, "unrealized_pnl": 0. if closed else pnl,
                 "hit": hit, "closed": closed, "holding_days": (end_pos-pos+1)/78,
                 "exit_reason": "target_high_proxy" if hit else "horizon_close" if closed else "open_mark_to_market"}
            trades.append(t)
            positions[s.symbol] = t
            cash_delta[pos] -= debit
            if closed:
                cash_delta[end_pos] += credit
                asset_value[pos:end_pos] += shares*path[pos:end_pos, 3]
            else:
                asset_value[pos:end_pos+1] += shares*path[pos:end_pos+1, 3]
    cash_curve = initial+np.cumsum(cash_delta)
    equity_curve = cash_curve+asset_value
    values = equity_curve[start:finish+1]
    drawdown = values/np.maximum.accumulate(np.r_[initial, values])[1:]-1
    end = float(values[-1])
    realized = sum(t["pnl"] for t in trades if t["closed"])
    unrealized = sum(t["unrealized_pnl"] for t in trades)
    if abs(end-initial-realized-unrealized) > 1e-6:
        raise AssertionError("Cash, realized and unrealized P&L do not reconcile")
    daily = [{"date": day, "equity": round(float(equity_curve[max(ix)]), 2),
              "cash": round(float(cash_curve[max(ix)]), 2)}
             for day, ix in grid.iloc[start:finish+1].groupby("date", sort=True).groups.items()]
    unresolved = sum(x["reason"] == "unresolved_selected_path" for x in rejected)
    closed = [t for t in trades if t["closed"]]
    result = {"backtest_complete": not unresolved, "initial_cash": initial,
              "end_balance": end if not unresolved else None, "net_return": end/initial-1 if not unresolved else None,
              "max_drawdown": float(drawdown.min()) if not unresolved else None,
              "realized_pnl": realized, "unrealized_pnl": unrealized,
              "cost": sum(t["cost"] for t in trades), "trade_n": len(trades), "closed_trade_n": len(closed),
              "trade_hit_n": sum(t["hit"] for t in closed),
              "trade_hit_rate": sum(t["hit"] for t in closed)/len(closed) if closed else None,
              "win_rate": sum(t["pnl"] > 0 for t in closed)/len(closed) if closed else None,
              "mean_holding_days": float(np.mean([t["holding_days"] for t in closed])) if closed else None,
              "unresolved_n": unresolved, "open_positions": len(trades)-len(closed),
              "decision_start": start_date, "decision_end": decision_end, "valuation_end": valuation_end,
              "simulation_status": "complete" if not unresolved else "incomplete_selected_path"}
    return result, {"trades": trades, "daily_equity": daily, "rejections": rejected}
