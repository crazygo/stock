import unittest
from audit_transfer_data import cash_matches


class CashPrecisionTests(unittest.TestCase):
    def test_observed_rounding_preserves_original_precision(self):
        self.assertTrue(cash_matches(.07162, '0.071616'))
        self.assertTrue(cash_matches(.95863, '0.958634'))
        self.assertTrue(cash_matches(.81676, '0.816760'))

    def test_wrong_amount_is_not_hidden_by_tolerance(self):
        self.assertFalse(cash_matches(.07163, '0.071616'))
        self.assertFalse(cash_matches(.071616, '0.071616'))
        self.assertTrue(cash_matches(.07162, '0.071615'))


if __name__ == '__main__':
    unittest.main()
