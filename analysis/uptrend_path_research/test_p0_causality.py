"""P0 causality test (spec_v1.md).

Appending future data must never change what the pipeline would have produced at an
earlier moment.  This is written as an automated test, not a promise.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from unittest import TestCase, main

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "analysis" / "growth_trigger"))

from gt_common import load_hourly
from observations import (HORIZON_SESSIONS, SESSION_MINUTES, bar_open, build_observations,
                          causal_features, horizon_end, regular_bars, sessions)
from segments import Bar

TICKERS = ["NOW", "NVDA", "MU"]
CUTS = [200, 400, 700, 1000]


def bars_of(ticker: str) -> list[Bar]:
    s = load_hourly(ticker)
    return [Bar(s.ts[i], s.o[i], s.h[i], s.l[i], s.c[i]) for i in range(len(s.ts))]


class CausalityTests(TestCase):
    def test_features_identical_when_future_data_is_appended(self):
        """P0: appending future data must not change any past feature."""
        for tk in TICKERS:
            bars = bars_of(tk)
            reg = regular_bars(bars)
            stamps = [b.ts for b in reg]
            full = {t: causal_features(bars, t) for t in stamps}
            for cut in CUTS:
                if cut >= len(stamps):
                    continue
                truncated = bars[:[i for i, b in enumerate(bars)
                                   if b.ts == stamps[cut]][0] + 1]
                for t in stamps[:cut]:
                    self.assertEqual(full[t], causal_features(truncated, t),
                                     f"{tk}: feature at {t} changed when data was "
                                     f"truncated at bar {cut}")

    def test_causal_state_is_prefix_stable(self):
        from segments import causal_state, trailing_atr_series
        import math
        for tk in TICKERS:
            bars = regular_bars(bars_of(tk))
            y = [math.log(b.c) for b in bars]
            atr = trailing_atr_series(bars, 20)
            previous = []
            for cut in range(3, len(bars)):
                from segments import zigzag_pivots_causal
                now = zigzag_pivots_causal(y[:cut], atr[:cut], 2.45)
                self.assertEqual(now[:len(previous)], previous,
                                 f"{tk}: confirmed pivots rewritten at bar {cut}")
                previous = now

    def test_outcomes_grow_as_data_is_appended(self):
        """Guards against a vacuous causality test: outcomes must depend on the future.

        Observations judged in both runs have identical horizons, so their labels must
        agree exactly; what changes is that previously unjudgeable observations become
        judgeable once the 3-trading-day horizon is inside the archive.
        """
        bars = bars_of("NOW")
        short = build_observations("NOW", bars[:400])
        long_ = build_observations("NOW", bars)
        short_map = {(o.signal_ts, o.entry_ts): o for o in short if o.judged}
        for o in long_:
            k = (o.signal_ts, o.entry_ts)
            if k in short_map:
                self.assertEqual(short_map[k].labels, o.labels,
                                 f"label for {k} changed after appending future data")
        self.assertGreater(sum(1 for o in long_ if o.judged),
                           sum(1 for o in short if o.judged))

    def test_horizon_is_three_trading_days_not_a_bar_count(self):
        bars = regular_bars(bars_of("NOW"))
        sess = sessions(bars)
        # start right at the open of a session: 3 sessions of market time later must be
        # the same clock time three sessions on, never "21 bars later"
        first_open = sess[0][1]
        end = horizon_end(first_open, sess)
        # 3 x 390 minutes from the OPEN of session 0 lands on the CLOSE of session 2
        self.assertEqual(end, sess[HORIZON_SESSIONS - 1][2])
        self.assertEqual((end - first_open).days, 4)      # Fri -> Tue, weekend skipped
        # a mid-session start must land mid-session three sessions later
        mid = sess[5][1] + timedelta(hours=2)
        self.assertEqual(horizon_end(mid, sess), sess[8][1] + timedelta(hours=2))
        # total consumed market minutes is exactly 3 x 390
        self.assertEqual(HORIZON_SESSIONS * SESSION_MINUTES, 1170)

    def test_entry_is_next_bar_open(self):
        bars = regular_bars(bars_of("NOW"))
        obs = build_observations("NOW", bars)
        by_signal = {o.signal_ts: o for o in obs}
        for i in range(len(bars) - 1):
            o = by_signal.get(bars[i].ts.isoformat(sep=" "))
            if o is None:
                continue
            self.assertEqual(o.entry_ts, bars[i + 1].ts.isoformat(sep=" "))
            self.assertEqual(o.entry_price, bars[i + 1].o)
            break

    def test_regular_session_only(self):
        bars = bars_of("NOW")
        reg = regular_bars(bars)
        self.assertLess(len(reg), len(bars))
        self.assertTrue(all(b.ts.strftime("%H:%M") in
                            ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00")
                            for b in reg))

    def test_five_labels_are_monotone(self):
        bars = bars_of("MU")
        for o in build_observations("MU", bars):
            if not o.judged:
                continue
            vals = [o.labels[k] for k in ("4%", "5%", "6%", "7%", "8%")]
            self.assertEqual(vals, sorted(vals, reverse=True))
            break


if __name__ == "__main__":
    main()
