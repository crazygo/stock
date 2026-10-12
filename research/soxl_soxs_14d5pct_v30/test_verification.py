"""Regression for timestamp wire types without weakening numerical comparisons."""
import unittest
import pandas as pd
from verify_overnight_v2 import compare_records


class WireComparison(unittest.TestCase):
    def test_saved_timestamp_and_in_memory_timestamp_match(self):
        compare_records({'dates':['2026-02-03 00:00:00'],'value':.123},
                        {'dates':[pd.Timestamp('2026-02-03')],'value':.123})

    def test_numerical_disagreement_is_rejected(self):
        with self.assertRaises(AssertionError):
            compare_records({'date':'2026-02-03 00:00:00','value':.12},
                            {'date':pd.Timestamp('2026-02-03'),'value':.13})


if __name__=='__main__':unittest.main()
