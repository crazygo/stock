"""The seven cells are frozen, and the early-stopping week stays out of the fit."""
from __future__ import annotations

import unittest

import pandas as pd

from research.after_open_3d5pct.models.premarket_tail_v1.opt_grid import (
    CELLS,
    choose_bound,
    fit_and_stop,
    grouped,
    tree_kwargs,
)


def _weeks() -> list[list[str]]:
    start = pd.Timestamp("2024-01-08")
    weeks = []
    for offset in range(10):
        block = [(start + pd.Timedelta(days=7 * offset + day)).strftime("%Y-%m-%d") for day in range(5)]
        weeks.append(block)
    return weeks


class GridTests(unittest.TestCase):
    def test_the_seven_cells_are_the_nonempty_combinations(self):
        flags = {(cell["reg"], cell["early"], cell["rank"]) for cell in CELLS}
        self.assertEqual(len(CELLS), 7)
        self.assertEqual(len(flags), 7)
        self.assertNotIn((False, False, False), flags)
        self.assertEqual([cell["id"] for cell in CELLS], ["R", "E", "K", "RE", "RK", "EK", "REK"])

    def test_regularization_is_the_focus_v8_recipe(self):
        self.assertEqual(tree_kwargs(True)["min_child_samples"], 150)
        self.assertEqual(tree_kwargs(True)["reg_lambda"], 30.0)
        self.assertEqual(tree_kwargs(False)["min_child_samples"], 10)
        self.assertEqual(tree_kwargs(False)["reg_lambda"], 10.0)
        self.assertEqual(tree_kwargs(True)["n_estimators"], 40)
        self.assertEqual(tree_kwargs(True)["num_leaves"], 7)

    def test_the_last_train_week_is_the_stopping_week_and_unmature_days_drop(self):
        weeks = _weeks()
        days = [day for block in weeks[:8] for day in block]
        before = pd.Timestamp(f"{weeks[8][0]} 09:30")
        frame = pd.DataFrame({
            "session_date": days,
            "label_end": [pd.Timestamp(f"{day} 16:00") for day in days],
            "y_3d_5pct": [1.0] * len(days),
            "symbol": ["AAA"] * len(days),
        })
        late = weeks[7][3:]
        frame.loc[frame["session_date"].isin(late), "label_end"] = pd.Timestamp(f"{weeks[8][0]} 16:00")
        fit, stop = fit_and_stop(frame, weeks, 8, before, True)
        self.assertFalse(set(fit["session_date"]) & set(stop["session_date"]))
        self.assertEqual(list(stop["session_date"]), weeks[7][:3])
        self.assertTrue(set(late).isdisjoint(fit["session_date"]))
        self.assertLess(fit["session_date"].max(), weeks[8][0])

    def test_groups_follow_symbol_order(self):
        frame = pd.DataFrame({
            "symbol": ["B", "A", "B", "A"],
            "session_date": ["2024-01-09", "2024-01-08", "2024-01-08", "2024-01-09"],
            "y_3d_5pct": [0, 1, 1, 0],
        })
        ordered, counts = grouped(frame)
        self.assertEqual(list(ordered["symbol"]), ["A", "A", "B", "B"])
        self.assertEqual(list(counts), [2, 2])

    def test_bound_requires_both_weeks_and_uses_the_lower_week(self):
        datum_test = {"precision": 0.48, "buys": 100}
        datum_validation = {"precision": 0.50, "buys": 100}
        cells = [
            {"id": "R", "comparable": True, "test": {"precision": 0.49, "buys": 100}, "validation": {"precision": 0.51, "buys": 100}},
            {"id": "E", "comparable": True, "test": {"precision": 0.47, "buys": 200}, "validation": {"precision": 0.60, "buys": 200}},
            {"id": "K", "comparable": False, "test": {"precision": 0.90, "buys": 100}, "validation": {"precision": 0.90, "buys": 100}},
        ]
        self.assertEqual(choose_bound(cells, datum_test, datum_validation), "R")
        self.assertEqual(choose_bound([], datum_test, datum_validation), "datum")


if __name__ == "__main__":
    unittest.main()
