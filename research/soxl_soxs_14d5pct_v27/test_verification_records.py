import unittest
from verify_clock_reliability_v2 import compare_reencoded_labels


class RecordTests(unittest.TestCase):
    def pending(self):
        return [{'exit_at':None,'reason':'pending','complete_window':False,'net_return':float('nan'),'exit_fill_economic':float('nan')}]

    def test_only_pending_nan_fields_may_become_null(self):
        old=self.pending();new=[{**old[0],'net_return':None,'exit_fill_economic':None}]
        self.assertEqual(compare_reencoded_labels(new,old),2)

    def test_unknown_cannot_become_zero(self):
        old=self.pending();new=[{**old[0],'net_return':0.,'exit_fill_economic':None}]
        with self.assertRaises(AssertionError):compare_reencoded_labels(new,old)

    def test_known_result_change_is_rejected(self):
        old=[{'exit_at':'known','reason':'target','complete_window':True,'net_return':.05,'exit_fill_economic':105.}]
        with self.assertRaises(AssertionError):compare_reencoded_labels([{**old[0],'net_return':.06}],old)

    def test_known_exit_with_nan_is_not_accepted(self):
        old=self.pending();old[0]['exit_at']='known'
        with self.assertRaises(AssertionError):compare_reencoded_labels(old,old)


if __name__=='__main__':unittest.main()
