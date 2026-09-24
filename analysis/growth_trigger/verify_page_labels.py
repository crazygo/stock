"""Reproduce the dashboard's current growth-window count with the 8% page settings.

The dashboard recomputes windows in JavaScript with: target = visual.threshold (8%),
>= 3 trading close intervals, no upper bound on sessions, calendar span 3..14 inclusive,
single-session close gain <= 12%, and >= half the target retained at both of the next
two closes.  identify_growth_windows.py defaults differ (20%, 4-9 sessions, strict
3 < days < 14), so this script re-implements the page rule set and checks that the
frozen 8% labels agree with what the page shows.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backtest_ai_strategies import Backtest
from preopen_three_week import FundMembership

PAGE = {"target": 0.08, "min_sessions": 3, "max_sessions": None,
        "calendar_inclusive": (3, 14), "max_one_day_gain": 0.12,
        "follow_through": 2, "min_retained": 0.04}


def page_rule_module() -> types.ModuleType:
    src = (ROOT / "identify_growth_windows.py").read_text()
    mod = types.ModuleType("igw_page")
    mod.__dict__["__name__"] = "igw_page"
    exec(compile(src.replace("if not 3 < calendar_days < 14:",
                             "if not 3 <= calendar_days <= 14:"), "igw_page", "exec"),
         mod.__dict__)
    mod.TARGET_RETURN = PAGE["target"]
    mod.MIN_SESSIONS = PAGE["min_sessions"]
    mod.MAX_SESSIONS = PAGE["max_sessions"] or 10 ** 6
    mod.MIN_RETAINED_GAIN = PAGE["min_retained"]
    return mod


def main() -> None:
    root = ROOT
    result = json.loads((root / "analysis/preopen_three_week/results.json").read_text())
    import render_preopen_dashboard as R
    mod = page_rule_module()
    R.find_growth_windows = mod.find_growth_windows
    payload = R.build_payload(root, result)
    symbols = payload["symbols"]
    matching = [r for r in payload["result"]["current"]
                if r.get("quality_pass") and symbols[r["ticker"]]["growth_windows"]]
    windows = sum(len(symbols[r["ticker"]]["growth_windows"]) for r in matching)
    print(f"page settings (8%, >=3 sessions, calendar 3..14 inclusive): "
          f"{len(matching)} stocks / {windows} windows")
    print("dashboard sidebar currently reports: 60 stocks / 402 windows")
    print("MATCH" if (len(matching), windows) == (60, 402) else "MISMATCH")

    # the frozen label snapshot must use the identical rule set
    label_path = root / "analysis/growth_trigger/labels_8pct_v1.json"
    if not label_path.exists():
        print("frozen label snapshot not built yet - run build_sample.py first")
        return
    labels = json.loads(label_path.read_text())
    pinned = json.loads(
        (root / "analysis/preopen_three_week/pinned_stocks.json").read_text())["tickers"]
    same_rule = (labels["params"]["target_return"] == PAGE["target"]
                 and labels["params"]["min_trading_close_intervals"] == PAGE["min_sessions"]
                 and labels["params"]["calendar_days_inclusive"] == list(PAGE["calendar_inclusive"])
                 and labels["params"]["max_single_session_close_gain"] == PAGE["max_one_day_gain"]
                 and labels["params"]["follow_through_sessions"] == PAGE["follow_through"]
                 and labels["params"]["min_gain_retained_vs_start"] == PAGE["min_retained"])
    print(f"frozen label params identical to page rule set: {same_rule}")
    print(f"frozen labels: {labels['stock_count']} stocks / {labels['window_count']} windows "
          f"over {len(pinned)} pinned tickers")
    per_ticker = {s["ticker"]: len(s["windows"]) for s in labels["stocks"]}
    print("pinned tickers with zero windows:",
          [t for t in pinned if per_ticker.get(t, 0) == 0])


if __name__ == "__main__":
    main()
