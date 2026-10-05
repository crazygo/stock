import unittest
import pandas as pd
from history_prepare import mature_features

class MatureAnchor(unittest.TestCase):
    def test_today_and_future_labels_cannot_change_features(self):
        rows=pd.DataFrame([dict(day='2025-01-02',y=1.,label_end='2025-01-08 16:00:00'),dict(day='2025-01-03',y=0.,label_end='2025-01-10 16:00:00'),dict(day='2025-01-06',y=1.,label_end='2025-01-13 16:00:00')])
        initial=mature_features(rows,['2025-01-10'])
        rows.loc[rows.label_end>='2025-01-10','y']=1-rows.loc[rows.label_end>='2025-01-10','y']
        pd.testing.assert_frame_equal(initial,mature_features(rows,['2025-01-10']))
        self.assertEqual(initial.iloc[0].m5_count_20,1)
        self.assertEqual(initial.iloc[0].m5_rate_20,1)
    def test_unknown_anchor_remains_missing(self):
        f=mature_features(pd.DataFrame([dict(day='2025-01-02',y=float('nan'),label_end='2025-01-08 16:00:00')]),['2025-01-10'])
        self.assertEqual(f.iloc[0].m5_count_20,0)
        self.assertEqual(f.iloc[0].m5_missing,1)
        self.assertTrue(pd.isna(f.iloc[0].m5_rate_20))

if __name__=='__main__':unittest.main()
