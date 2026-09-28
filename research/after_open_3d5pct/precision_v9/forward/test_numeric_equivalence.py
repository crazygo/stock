"""Small counterexamples for the isolated R03M numeric evidence verifier."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

import numpy as np

from . import numeric_equivalence as audit
from .ledger import canonical


class NumericEquivalenceTest(unittest.TestCase):
    def test_registered_inventory_exact_fifteen_and_duplicate_rejected(self):
        bundles = audit.registered_bundles()
        self.assertEqual(len(bundles), 15)
        audit.assert_complete_inventory(bundles)
        damaged = deepcopy(bundles)
        damaged[-1] = deepcopy(damaged[0])
        with self.assertRaisesRegex(ValueError, 'complete adopted'):
            audit.assert_complete_inventory(damaged)

    def test_diff_is_json_serializable_and_nonfinite_probabilities_fail(self):
        value = audit._diff(np.zeros((1, 9)), np.zeros((1, 9)), ['key'])
        json.loads(canonical(value))
        self.assertEqual(value['worst_index'], [0, 0])
        nan = np.full((1, 9), np.nan)
        self.assertTrue(audit._diff(nan, nan, ['key'])['pass'])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            audit._probability_array(nan, 1)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            audit._probability_array(np.full((1, 9), 1.1), 1)

    def test_disabled_c_slots_are_empty_at_model_call(self):
        audit.torch.set_num_threads(1)
        data = {'x5': np.zeros((2, 8, 192, 14), np.float32),
                'x60': np.zeros((2, 31, 17, 14), np.float32),
                'xday': np.ones((2, 126, 14), np.float32),
                'group_seq': np.ones((2, 6, 6, 10), np.float32)}
        prior = np.full((2, 9), .5, np.float32)
        for route, daily, group in (('C_no_daily', False, True),
                                    ('C_no_group', True, False)):
            class Spy:
                def __init__(self):
                    self.daily, self.group = daily, group
                    self.calls = 0

                def eval(self):
                    return self

                def __call__(self, *inputs):
                    self.calls += 1
                    if not self.daily:
                        self_test.assertEqual(inputs[2].numel(), 0)
                    if not self.group:
                        self_test.assertEqual(inputs[3].numel(), 0)
                    return audit.torch.zeros((len(inputs[0]), 9))

            self_test = self
            model = Spy()
            arrays = audit._c_input_arrays(route, data, prior)
            result = audit.original.predict_c(model, arrays, np.arange(2))
            self.assertEqual(result.shape, (2, 9))
            self.assertGreater(model.calls, 0)

    def test_selected_c_group_eval_uses_original_eval_batch_partition(self):
        audit.torch.set_num_threads(1)
        bundles = audit.registered_bundles()
        bundle = next(b for b in bundles if b['route'] == 'C_group' and
                      b['fold'] == 'selected_3566')
        rows, data = audit._data(audit.V8/'dataset')
        transformed = audit._transforms(data, rows)
        keys = rows.sample_id.astype(str).tolist()
        raw, processed = audit._predict_offline(bundle, data, transformed)
        reference = audit._bundle_reference(bundle, rows, keys, data,
                                            transformed, raw, processed)
        self.assertEqual(reference['eval_rows'], 1836)
        self.assertTrue(reference['raw']['pass'])
        self.assertTrue(reference['processed']['pass'])
        self.assertLessEqual(reference['processed']['max_abs_error'], 1e-7)
        # The separately reported full-population batch slice is diagnostic;
        # frozen eval was originally predicted in its own split batches.
        self.assertIn('full_batch_eval_slice_processed', reference)


if __name__ == '__main__':
    unittest.main()
