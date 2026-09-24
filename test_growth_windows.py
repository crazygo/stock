"""Boundary checks for the historical growth labels."""

from datetime import date, timedelta
from unittest import TestCase, main

from identify_growth_windows import find_growth_windows


class GrowthWindowTests(TestCase):
    def bars(self, closes, offsets=None):
        offsets = offsets or range(len(closes))
        return [{"d": (date(2026, 1, 1) + timedelta(days=offset)).isoformat(),
                 "c": close} for close, offset in zip(closes, offsets)]

    def test_completed_four_interval_gain_is_labeled(self):
        found = find_growth_windows(self.bars([100, 104, 109, 115, 121, 118, 116]),
                                    lambda _: True)
        self.assertEqual([(w["start_index"], w["end_index"]) for w in found], [(0, 4)])
        self.assertEqual(found[0]["calendar_days"], 4)

    def test_early_crossing_cannot_be_stretched_into_valid_window(self):
        self.assertEqual(find_growth_windows(
            self.bars([100, 107, 114, 121, 119, 121, 120, 119]), lambda _: True), [])

    def test_one_day_spike_and_fast_reversal_are_rejected(self):
        for closes in ([100, 101, 102, 103, 125, 122, 120],
                       [100, 105, 110, 115, 121, 109, 108]):
            with self.subTest(closes=closes):
                self.assertEqual(find_growth_windows(self.bars(closes), lambda _: True), [])

    def test_strict_calendar_boundary_and_start_quality(self):
        closes = [100, 104, 109, 115, 121, 118, 116]
        days = [0, 3, 6, 10, 14, 15, 16]
        self.assertEqual(find_growth_windows(self.bars(closes, days), lambda _: True), [])
        self.assertEqual(find_growth_windows(self.bars(closes), lambda _: False), [])


if __name__ == "__main__":
    main()
