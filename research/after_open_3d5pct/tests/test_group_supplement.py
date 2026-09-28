import unittest
from unittest.mock import patch

import numpy as np

from research.after_open_3d5pct.supplement_group_report import metrics, rank_cells, window_features
from research.after_open_3d5pct.v6_data import SymbolData


class SupplementTests(unittest.TestCase):
    def test_unknown_is_not_failure_and_no_positive_precision_is_undefined(self):
        result=metrics([0,1,np.nan],[.1,.2,.9])
        self.assertEqual(result['n'],2)
        self.assertEqual(result['accuracy50'],.5)
        self.assertIsNone(result['precision50'])
        self.assertEqual(result['fn'],1)

    def test_ranking_excludes_reference_and_uses_declared_ties(self):
        def cell(g,a,n):return {'group_id':g,'metrics':{'accuracy50':a,'n':n}}
        chosen=rank_cells([cell('all:all',1,1000),cell('b',.9,20),cell('a',.9,20),cell('c',.9,30),cell('empty',None,0)])
        self.assertEqual([c['group_id'] for c in chosen],['c','a','b'])

    def test_brier_ranks_low_first(self):
        values=[{'group_id':g,'metrics':{'brier':v,'n':20}} for g,v in [('a',.2),('b',.1)]]
        self.assertEqual(rank_cells(values,'brier')[0]['group_id'],'b')

    def test_feature_prefix_does_not_depend_on_later_bars_or_labels(self):
        dates=['2026-09-14','2026-09-15','2026-09-16']
        sessions=[{'open_at':d+'T13:30:00+00:00'} for d in dates]
        v=SymbolData(np.ones((3,192,6)),np.full((3,192),np.datetime64('2026-01-01','ns')),
                     np.ones((3,192,14)),np.ones((3,17,14)),np.ones((3,14)),np.ones(3,bool),
                     np.zeros(3),np.ones(3),'fixture',0)
        with patch('research.after_open_3d5pct.v6_data._group_state',return_value=np.zeros((6,10))):
            first=window_features('TEST',1,dates,sessions,{'TEST':v},{},{})
            v.seq5[1,90:]=999;v.seq5[2]=999
            v.seq60[1,8:]=999;v.seq60[2]=999
            v.seqday[1:]=999
            second=window_features('TEST',1,dates,sessions,{'TEST':v},{},{})
        for key in first:np.testing.assert_array_equal(first[key],second[key])


if __name__=='__main__':unittest.main()
