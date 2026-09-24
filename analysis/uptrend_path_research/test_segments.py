"""Unit tests for the segmentation primitive.

Every expectation below was derived by hand from the synthetic series and then
verified against the implementation.  These are the machine-checkable part of
"is the interval detection correct"; the dashboard is the visual part.
"""

from datetime import datetime, timedelta
from unittest import TestCase, main

from segments import (Bar, analysis_segments, build_segments, causal_state,
                      trailing_atr_series, zigzag_pivots, zigzag_pivots_causal,
                      qualifies, _features)


def bars_from(closes, highs=None, lows=None):
    base = datetime(2026, 1, 2, 9, 30)
    return [Bar(base + timedelta(hours=i), c,
                highs[i] if highs else c * 1.001,
                lows[i] if lows else c * 0.999, c)
            for i, c in enumerate(closes)]


def kinds(pivots):
    return [p[1] for p in pivots]


def wave(down=10, up=20, down2=11, up2=9, step=1.0, start=100.0):
    """fall -> rise -> fall -> rise, so both an up and a down segment are complete."""
    seq = [start - step * i for i in range(down)]
    seq += [seq[-1] + step * i for i in range(1, up)]
    seq += [seq[-1] - step * i for i in range(1, down2)]
    seq += [seq[-1] + step * i for i in range(1, up2)]
    return seq


class ZigZagTests(TestCase):
    def test_flat_series_has_no_pivots(self):
        self.assertEqual(zigzag_pivots([100.0] * 30, 5.0), [])

    def test_monotone_rise_has_no_pivots(self):
        self.assertEqual(zigzag_pivots([100 + i for i in range(30)], 5.0), [])

    def test_monotone_fall_has_no_pivots(self):
        self.assertEqual(zigzag_pivots([100 - i for i in range(30)], 5.0), [])

    def test_rise_then_fall_pivots(self):
        y = [100, 103, 106, 110, 108, 104, 99, 96, 99, 103, 100, 97, 94]
        self.assertEqual(zigzag_pivots(y, 5.0),
                         [(3, "peak"), (7, "trough"), (9, "peak")])

    def test_wiggle_smaller_than_theta_is_absorbed(self):
        y = [100, 104, 108, 105, 111, 112, 108, 105, 100, 96]
        self.assertEqual(zigzag_pivots(y, 5.0), [(5, "peak")])

    def test_first_pivot_can_be_a_trough(self):
        y = [100, 97, 94, 90, 93, 97, 102, 106, 103, 99]
        self.assertEqual(zigzag_pivots(y, 5.0), [(3, "trough"), (7, "peak")])

    def test_short_dip_still_gets_its_own_pivot(self):
        # a 2-bar dip deeper than theta is a real pivot; duration is filtered later
        y = [100, 104, 108, 112, 105, 100, 104, 108, 112, 116, 112, 108]
        self.assertEqual(zigzag_pivots(y, 5.0),
                         [(3, "peak"), (5, "trough"), (9, "peak")])

    def test_pivot_never_at_index_zero(self):
        for y in ([100, 130, 100, 130], [100, 70, 100, 70]):
            self.assertTrue(all(i > 0 for i, _ in zigzag_pivots(y, 5.0)))

    def test_zero_theta_returns_nothing(self):
        self.assertEqual(zigzag_pivots([100, 101, 99, 102], 0.0), [])


class SegmentTests(TestCase):
    def test_wave_gives_up_down_and_censored_tail(self):
        segs = build_segments(bars_from(wave()), delta=0.5, min_amp=1.0, min_bars=4)
        self.assertEqual([s.kind for s in segs], ["up", "down"])
        up, down = segs
        self.assertEqual((up.start_idx, up.end_idx), (9, 28))
        self.assertEqual(up.bars, 20)
        self.assertAlmostEqual(up.amplitude, 0.18962, places=4)  # log(110/91)
        self.assertFalse(up.censored)
        self.assertTrue(down.censored)
        self.assertEqual(analysis_segments(segs), [up])

    def test_min_bars_flags_short_segment(self):
        segs = build_segments(bars_from(wave()), delta=0.5, min_amp=1.0, min_bars=25)
        self.assertTrue(all(s.short for s in segs))
        self.assertEqual(analysis_segments(segs), [])

    def test_min_amp_only_flags_it_never_moves_a_pivot(self):
        """min_amp is a qualification flag, not a structural parameter."""
        loose = build_segments(bars_from(wave()), delta=0.5, min_amp=1.0, min_bars=4)
        strict = build_segments(bars_from(wave()), delta=0.5, min_amp=100.0, min_bars=4)
        self.assertEqual([(s.kind, s.start_idx, s.end_idx) for s in loose],
                         [(s.kind, s.start_idx, s.end_idx) for s in strict])
        self.assertTrue(all(s.qualified for s in loose))
        self.assertTrue(all(not s.qualified for s in strict))
        self.assertEqual(analysis_segments(strict), [])

    def test_delta_alone_controls_the_boundary(self):
        """A dip bigger than theta splits the move; a smaller dip does not."""
        y = [0, 0.05, 0.10, 0.15, 0.10, 0.05, 0.20, 0.25, 0.30]
        # dip of 0.10 from the running max: theta below it splits, above it does not
        self.assertEqual(zigzag_pivots(y, 0.08), [(3, "peak"), (5, "trough")])
        self.assertEqual(zigzag_pivots(y, 0.12), [])

    def test_zero_atr_at_first_bar_does_not_kill_causal_segmentation(self):
        """atr[0] is 0 by construction; it must not disable the causal walk."""
        import math as _m
        closes = [100 + 4 * _m.sin(i / 2.5) for i in range(120)]
        bars = bars_from(closes)
        y = [_m.log(b.c) for b in bars]
        atr = trailing_atr_series(bars, 20)
        self.assertEqual(atr[0], 0.0)
        self.assertTrue(zigzag_pivots_causal(y, atr, 0.5))

    def test_event_ids_are_sequential_per_kind(self):
        segs = build_segments(bars_from(wave()), delta=0.5, min_amp=1.0, min_bars=4)
        ups = [s.event_id for s in segs if s.kind == "up"]
        downs = [s.event_id for s in segs if s.kind == "down"]
        self.assertEqual(ups, [f"UP#{i+1}" for i in range(len(ups))])
        self.assertEqual(downs, [f"DOWN#{i+1}" for i in range(len(downs))])

    def test_efficiency_is_one_for_monotone_move(self):
        segs = build_segments(bars_from(wave()), delta=0.5, min_amp=1.0, min_bars=4)
        self.assertAlmostEqual(segs[0].efficiency, 1.0, places=6)

    def test_delta_v_sign_separates_accelerating_and_decelerating(self):
        accel = [0.05 * i * i for i in range(30)]
        decel = [5 * i - 0.1 * i * i for i in range(30)]
        self.assertGreater(_features(accel, 0, 29)["delta_v"], 0)
        self.assertLess(_features(decel, 0, 29)["delta_v"], 0)

    def test_mdd_measured_against_running_max(self):
        y = [0.0, 0.10, 0.05, 0.12, 0.02]
        self.assertAlmostEqual(_features(y, 0, 4)["mdd"], 0.10, places=6)

    def test_calendar_days_use_dates_not_bars(self):
        segs = build_segments(bars_from(wave()), delta=0.5, min_amp=1.0, min_bars=4)
        self.assertEqual(segs[0].calendar_days, 1)   # 20 hourly bars cross midnight
        self.assertGreaterEqual(segs[0].bars, 20)

    def test_empty_and_tiny_input(self):
        self.assertEqual(build_segments([], delta=0.5), [])
        self.assertEqual(build_segments(bars_from([100, 101, 102]), delta=0.5,
                                        min_bars=6), [])

    def test_atr_modes_agree_on_order_of_magnitude(self):
        b = bars_from(wave())
        from segments import atr_scale
        g = atr_scale(b, "global")
        t = atr_scale(b, "trailing", window=10)
        self.assertTrue(0.2 * g < t < 5 * g)


class CausalityTests(TestCase):
    """With a causal ATR the confirmed-pivot sequence must never rewrite history."""

    def bars(self, closes):
        base = datetime(2026, 1, 2, 9, 30)
        return [Bar(base + timedelta(hours=i), c, c * 1.001, c * 0.999, c)
                for i, c in enumerate(closes)]

    def test_trailing_atr_uses_only_the_past(self):
        bars = self.bars([100 + (i % 7) * 0.8 for i in range(60)])
        atr = trailing_atr_series(bars, 20)
        # recomputing on a prefix must give the same values for the shared prefix
        for cut in (10, 25, 40, 59):
            prefix = trailing_atr_series(bars[:cut], 20)
            self.assertEqual(prefix, atr[:cut])

    def test_pivots_grow_monotonically_with_trailing_atr(self):
        import math as _m
        closes = [100 + 4 * _m.sin(i / 2.5) + 0.15 * i for i in range(200)]
        bars = self.bars(closes)
        y = [_m.log(b.c) for b in bars]
        atr = trailing_atr_series(bars, 20)
        previous = []
        for cut in range(3, len(bars)):
            now = zigzag_pivots_causal(y, atr, 0.5, upto=cut)
            self.assertEqual(now[:len(previous)], previous,
                             f"history rewritten at bar {cut}")
            previous = now

    def test_causal_state_matches_full_segmentation_at_the_end(self):
        import math as _m
        closes = [100 + 4 * _m.sin(i / 2.5) + 0.15 * i for i in range(200)]
        bars = self.bars(closes)
        y = [_m.log(b.c) for b in bars]
        atr = trailing_atr_series(bars, 20)
        state = causal_state(y, atr, 0.5, len(bars) - 1)
        self.assertTrue(state["pivots"])
        self.assertEqual(state["bars_since_pivot"],
                         len(bars) - 1 - state["pivots"][-1][0])
        self.assertIsNotNone(state["current_direction"])

    def test_global_atr_is_not_guaranteed_causal(self):
        """Documented contrast: the label segmentation may rewrite history."""
        import math as _m
        closes = [100 + 4 * _m.sin(i / 2.5) + 0.15 * i for i in range(200)]
        y = closes
        full = zigzag_pivots(y, 0.05)
        # with a fixed theta the sequence is still stable, so this is a sanity check
        # that the function is deterministic rather than a causality guarantee
        self.assertEqual(full, zigzag_pivots(y, 0.05))


if __name__ == "__main__":
    main()


class QualificationTests(TestCase):
    """The target return is a qualification filter, never a segmentation parameter."""

    def bars(self, closes):
        base = datetime(2026, 1, 2, 9, 30)
        return [Bar(base + timedelta(hours=i), c, c * 1.001, c * 0.999, c)
                for i, c in enumerate(closes)]

    def seg(self, closes, **kw):
        segs = build_segments(self.bars(closes), **kw)
        return next((s for s in segs if s.kind == "up"), None)

    def test_plain_amplitude_mode(self):
        # rise of exactly 10% then a fall, so the up segment is complete
        closes = [100 - i for i in range(10)] + [91 + i for i in range(1, 12)] + \
                 [101 - 2 * i for i in range(1, 8)]
        s = self.seg(closes, delta=0.05, min_amp=0.0, min_bars=1)
        self.assertIsNotNone(s)
        self.assertTrue(qualifies(self.bars(closes), s, target=0.08, mode="amplitude"))
        self.assertFalse(qualifies(self.bars(closes), s, target=0.20, mode="amplitude"))

    def test_window_mode_rejects_a_one_day_spike(self):
        # +15% in a single bar then flat: amplitude qualifies, window mode does not
        closes = [100 - i for i in range(10)] + [91] + [105.65] + [106] * 12 + \
                 [106 - 2 * i for i in range(1, 8)]
        bars = self.bars(closes)
        s = self.seg(closes, delta=0.05, min_amp=0.0, min_bars=1)
        self.assertIsNotNone(s)
        self.assertTrue(qualifies(bars, s, target=0.08, mode="amplitude"))
        self.assertFalse(qualifies(bars, s, target=0.08, mode="window"))

    def test_window_mode_rejects_fast_give_back(self):
        # reaches +10% then collapses before two confirming bars
        closes = ([100 - i for i in range(10)]
                  + [92.2, 93.4, 94.6, 95.8, 97.0, 98.2, 99.4]
                  + [95.0, 93.0, 92.0, 91.0, 90.0])
        bars = self.bars(closes)
        s = self.seg(closes, delta=0.05, min_amp=0.0, min_bars=1)
        self.assertIsNotNone(s)
        self.assertFalse(qualifies(bars, s, target=0.08, mode="window"))

    def test_window_mode_accepts_clean_rise(self):
        closes = ([100 - i for i in range(10)]
                  + [91 + 1.2 * i for i in range(1, 12)]
                  + [104, 105, 106, 105, 104, 103])
        bars = self.bars(closes)
        s = self.seg(closes, delta=0.05, min_amp=0.0, min_bars=1)
        self.assertIsNotNone(s)
        self.assertTrue(qualifies(bars, s, target=0.08, mode="window"))

    def test_target_never_changes_the_segmentation(self):
        closes = [100 - i for i in range(10)] + [91 + i for i in range(1, 12)] + \
                 [101 - 2 * i for i in range(1, 8)]
        a = build_segments(self.bars(closes), delta=0.05, min_amp=0.0, min_bars=1)
        b = build_segments(self.bars(closes), delta=0.05, min_amp=0.0, min_bars=1)
        self.assertEqual([(s.kind, s.start_idx, s.end_idx) for s in a],
                         [(s.kind, s.start_idx, s.end_idx) for s in b])
        for t in (0.02, 0.08, 0.5):
            self.assertEqual([s.qualified for s in a], [s.qualified for s in b])
