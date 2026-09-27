"""Versioned, read-only manual-action replay over immutable outer predictions.

Prices are 5-minute regular-session OHLC proxies. No broker connection exists here.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import isfinite
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
POLICY_VERSION = "manual_three_cutoffs_v1"
BAR = timedelta(minutes=5)


def instant(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("timestamps must include an offset")
    return result.astimezone(UTC)


@dataclass(frozen=True)
class PolicyConfig:
    threshold: float = 0.35  # Illustrative development threshold; never fitted on outer scores.
    cutoffs_et: tuple[str, ...] = ("11:30", "13:30", "14:30")
    signal_latency_seconds: int = 30
    manual_delay_seconds: int = 120
    entry_max_wait_minutes: int = 5
    horizon_regular_minutes: int = 1170
    target_return: float = 0.05
    fee_bps_per_side: float = 1.0
    slippage_bps_per_side: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.cutoffs_et != ("11:30", "13:30", "14:30"):
            raise ValueError("v1 operation replay has exactly three cutoffs")
        if self.horizon_regular_minutes != 1170 or self.target_return != 0.05:
            raise ValueError("v1 must preserve the original label target and horizon")
        if self.horizon_regular_minutes % 5 or self.entry_max_wait_minutes < 0:
            raise ValueError("invalid regular-bar duration")
        if min(self.fee_bps_per_side, self.slippage_bps_per_side) < 0:
            raise ValueError("costs must be nonnegative")

    def record(self) -> dict:
        return {"version": POLICY_VERSION, **asdict(self),
                "cutoffs_et": list(self.cutoffs_et), "probability_kind": "raw_model_output"}


def _valid_bar(bar: dict | None) -> bool:
    if not bar or bar.get("session_type") != "regular" or bar.get("price_basis") != "NONE":
        return False
    try:
        op, hi, lo, cl = (float(bar[k]) for k in ("open", "high", "low", "close"))
    except (KeyError, TypeError, ValueError):
        return False
    return all(map(isfinite, (op, hi, lo, cl))) and 0 < lo <= min(op, cl) <= max(op, cl) <= hi


def _net_return(entry: float, exit_price: float, config: PolicyConfig) -> float:
    cost = config.fee_bps_per_side / 10000
    slip = config.slippage_bps_per_side / 10000
    return exit_price * (1 - cost - slip) / (entry * (1 + cost + slip)) - 1


def simulate_entry(decision: dict, bars: dict[datetime, dict],
                   regular_starts: list[datetime], config: PolicyConfig,
                   as_of: datetime) -> dict:
    """Replay one proposed entry; fail closed on missing bars and uncertain ordering."""
    decision_at = instant(decision["decision_at"])
    cutoff_at = instant(decision["cutoff_at"])
    required_decision = cutoff_at + timedelta(seconds=config.signal_latency_seconds)
    if decision_at != required_decision:
        return {"status": "decision_time_mismatch", "entry_at": None, "exit_at": None}
    earliest = decision_at + timedelta(seconds=config.manual_delay_seconds)
    i = bisect_left(regular_starts, earliest)
    if i >= len(regular_starts) or regular_starts[i] > earliest + timedelta(minutes=config.entry_max_wait_minutes):
        return {"status": "entry_wait_expired", "entry_at": None, "exit_at": None}
    entry_at = regular_starts[i]
    if entry_at.astimezone(ET).date() != cutoff_at.astimezone(ET).date():
        return {"status": "entry_wait_expired", "entry_at": None, "exit_at": None}
    entry_bar = bars.get(entry_at)
    if not _valid_bar(entry_bar):
        return {"status": "entry_bar_missing_or_invalid", "entry_at": entry_at.isoformat(), "exit_at": None}
    entry = float(entry_bar["open"])
    end_index = i + config.horizon_regular_minutes // 5
    if end_index > len(regular_starts):
        return {"status": "pending_calendar", "entry_at": entry_at.isoformat(),
                "entry_price": entry, "exit_at": None}
    expected_end = regular_starts[end_index - 1] + BAR
    if instant(as_of) < expected_end:
        return {"status": "pending_window", "entry_at": entry_at.isoformat(),
                "entry_price": entry, "expected_end_at": expected_end.isoformat(), "exit_at": None}
    target = entry * (1 + config.target_return)
    lows_before: list[float] = []
    highs_before: list[float] = []
    for j in range(i, end_index):
        at = regular_starts[j]
        bar = bars.get(at)
        if not _valid_bar(bar):
            return {"status": "indeterminate_missing_path", "entry_at": entry_at.isoformat(),
                    "entry_price": entry, "missing_at": at.isoformat(), "exit_at": None}
        low, high, close = (float(bar[k]) for k in ("low", "high", "close"))
        if high >= target:
            # The target touch is known within this bar; its low may be after exit.
            worst = min([entry, *lows_before, low]) / entry - 1
            best = min([entry, *lows_before]) / entry - 1
            exit_at = at + BAR  # Position remains open until bar completion is observable.
            return {"status": "target_touch_proxy", "entry_at": entry_at.isoformat(),
                    "entry_price": entry, "exit_at": exit_at.isoformat(),
                    "exit_price_proxy": target, "target_hit": True,
                    "holding_regular_minutes": (j - i + 1) * 5,
                    "hit_time_interval_minutes": [5 * (j - i), 5 * (j - i + 1)],
                    "gross_return": config.target_return,
                    "net_return": _net_return(entry, target, config),
                    "mae_lower_bound": worst, "mae_upper_bound": best,
                    "mae_order_uncertain": worst != best,
                    "mfe_before_exit_lower_bound": config.target_return,
                    "expected_end_at": expected_end.isoformat()}
        lows_before.append(low)
        highs_before.append(high)
    last = bars[regular_starts[end_index - 1]]
    exit_price = float(last["close"])
    return {"status": "window_end_proxy", "entry_at": entry_at.isoformat(),
            "entry_price": entry, "exit_at": expected_end.isoformat(),
            "exit_price_proxy": exit_price, "target_hit": False,
            "holding_regular_minutes": config.horizon_regular_minutes,
            "hit_time_interval_minutes": None,
            "gross_return": exit_price / entry - 1,
            "net_return": _net_return(entry, exit_price, config),
            "mae_lower_bound": min([entry, *lows_before]) / entry - 1,
            "mae_upper_bound": min([entry, *lows_before]) / entry - 1,
            "mae_order_uncertain": False,
            "mfe_before_exit_lower_bound": max([entry, *highs_before]) / entry - 1,
            "expected_end_at": expected_end.isoformat()}


def replay(decisions: list[dict], bars_by_symbol: dict[str, dict[datetime, dict]],
           regular_starts: list[datetime], config: PolicyConfig,
           as_of: datetime) -> tuple[list[dict], list[dict]]:
    """Return decision ledger and distinct attempted-entry/trade ledger."""
    if regular_starts != sorted(set(regular_starts)):
        raise ValueError("calendar regular bars must be sorted and unique")
    seen: set[str] = set()
    ledger: list[dict] = []
    trades: list[dict] = []
    active: dict[str, dict] = {}
    for row in sorted(decisions, key=lambda r: (instant(r["decision_at"]), r["symbol"])):
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate decision {sample_id}")
        seen.add(sample_id)
        symbol = str(row["symbol"])
        at = instant(row["decision_at"])
        previous = active.get(symbol)
        if previous and previous.get("exit_at") and instant(previous["exit_at"]) <= at:
            active.pop(symbol)
            previous = None
        entry = {"sample_id": sample_id, "symbol": symbol,
                 "session_date": row["session_date"], "cutoff_et": row["cutoff_et"],
                 "probability": row["probability"], "target": row.get("target")}
        if row["cutoff_et"] not in config.cutoffs_et:
            entry["action"] = "scored_only"
        elif previous:
            entry["action"] = "observe_held"
            entry["held_trade_id"] = previous["trade_id"]
        elif row["probability"] is None or not isfinite(float(row["probability"])):
            entry["action"] = "unavailable_probability"
        elif float(row["probability"]) < config.threshold:
            entry["action"] = "below_threshold"
        else:
            entry["action"] = "recommend_entry"
            trade = {**entry, "trade_id": f"{POLICY_VERSION}|{sample_id}"}
            trade.update(simulate_entry(row, bars_by_symbol.get(symbol, {}), regular_starts,
                                        config, as_of))
            trades.append(trade)
            entry["trade_id"] = trade["trade_id"]
            if trade.get("entry_price") is not None:
                active[symbol] = trade
        ledger.append(entry)
    return ledger, trades


def summarize(decisions: list[dict], ledger: list[dict], trades: list[dict]) -> dict:
    """Keep prediction, recommendation and resolved-operation denominators separate."""
    actions = {r["sample_id"]: r["action"] for r in ledger}
    operational = [r for r in decisions if actions[r["sample_id"]] != "scored_only"]
    mature = [r for r in operational if r.get("target") in (0, 1)]
    recommended = [r for r in operational if actions[r["sample_id"]] == "recommend_entry"]
    recommended_mature = [r for r in recommended if r.get("target") in (0, 1)]
    resolved = [t for t in trades if t["status"] in ("target_touch_proxy", "window_end_proxy")]
    return {"predictions": len(operational), "mature": len(mature),
            "pending_or_missing_target": len(operational) - len(mature),
            "pool_hit": sum(r["target"] == 1 for r in mature),
            "recommendations": len(recommended), "recommended_mature": len(recommended_mature),
            "recommended_hit": sum(r["target"] == 1 for r in recommended_mature),
            "entry_proxies": sum(t.get("entry_price") is not None for t in trades),
            "resolved_operations": len(resolved),
            "operation_hits": sum(t["target_hit"] for t in resolved),
            "unresolved_operations": len(trades) - len(resolved)}
