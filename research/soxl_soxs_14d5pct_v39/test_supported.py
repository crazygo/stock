"""Test primary selection, stress identity and consumer authorization boundaries."""
import unittest,tempfile
from unittest.mock import patch
from supported import *

class Tests(unittest.TestCase):
    def test_stress_follows_new_primary(self):
        old={k:{'action':k,'allocation':1.} for k in parent.ROUTES};new=selected_events(old)
        self.assertEqual(new['double_cost'],old['all_supported']);self.assertNotEqual(new['double_cost'],old['double_cost'])
        self.assertEqual(set(new),set(ROUTES));new['double_cost']['allocation']=0;self.assertEqual(old['all_supported']['allocation'],1)
    def test_diagnostics_cannot_authorize(self):
        p=read(SOURCE/'protocol.json')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dump(root/'registration.json',{'phase':'development','protocol_sha256':sha(SOURCE/'protocol.json')})
            dump(root/'results.json',{'primary_name':'eligible_supported','primary':{},'routes':{'all_supported':{}}})
            dump(root/'verification.json',{'passed':True})
            with patch('supported.load_frozen',return_value=(p,{})):
                with self.assertRaises(AssertionError):authorize(root)
    def test_false_receipt_cannot_authorize(self):
        p=read(SOURCE/'protocol.json')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dump(root/'registration.json',{'phase':'development','protocol_sha256':sha(SOURCE/'protocol.json')})
            dump(root/'results.json',{'primary_name':'all_supported','primary':{},'routes':{'all_supported':{}}})
            dump(root/'verification.json',{'passed':False,'results_sha256':sha(root/'results.json')})
            with patch('supported.load_frozen',return_value=(p,{})):
                with self.assertRaises(AssertionError):authorize(root)
    def test_source_only_consumer(self):
        s=(SOURCE/'extend_financial_source.py').read_text();self.assertIn("consumer=consumer_authorize(case_after)",s)
        self.assertIn("'upstream_policy_qualified':False",s)
    def test_policy_unchanged(self):
        p=read(SOURCE/'protocol.json');old=read(SOURCE.parent/'soxl_soxs_14d5pct_v38/protocol.json')
        for key in ['fit','history','warmup','weights','support','bootstrap','capital','contract','acceptance']:self.assertEqual(p[key],old[key])

if __name__=='__main__':unittest.main()
