import unittest
from unittest.mock import patch
import copy
import selection


class SelectionTests(unittest.TestCase):
    def test_actual_development_selection_does_not_promote_v18_primary(self):
        result=selection.validate_selection()
        self.assertTrue(result['gate']['passed']);self.assertTrue(result['v18_primary_remains_failed'])
        self.assertEqual(result['selected_route'],'state_hard')
        self.assertFalse(result['new_v19_case_results_used_for_selection'])

    def test_failed_candidate_gate_is_rejected(self):
        with patch.object(selection,'development_gate',return_value={'passed':False}):
            with self.assertRaisesRegex(ValueError,'development screen'):selection.validate_selection()

    def test_case_driven_reselection_cannot_replace_registered_route(self):
        real_read=selection.read
        def altered(path):
            value=copy.deepcopy(real_read(path))
            if str(path).endswith('/protocol.json'):value['primary']='source_direction'
            return value
        with patch.object(selection,'read',side_effect=altered):
            with self.assertRaises(AssertionError):selection.validate_selection()

    def test_changed_development_results_are_rejected(self):
        real_sha=selection.sha
        with patch.object(selection,'sha',side_effect=lambda path:'0'*64 if str(path).endswith('/results.json') else real_sha(path)):
            with self.assertRaises(AssertionError):selection.validate_selection()


if __name__=='__main__':unittest.main()
