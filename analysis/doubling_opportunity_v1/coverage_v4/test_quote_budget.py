"""Exercise a real stalled child and a failing child; no OpenD or network needed."""
import importlib.util,sys,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('v4_quote_budget_under_test',Path(__file__).with_name('acquire_quotes.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
supervise=module.supervise

class BudgetTest(unittest.TestCase):
    def test_stalled_worker_is_terminated(self):
        out=supervise(.2,[sys.executable,'-c','import time; time.sleep(60)'])
        self.assertTrue(out['timed_out']);self.assertIsNotNone(out['exit_code'])
        self.assertLess(out['elapsed_seconds'],2)
    def test_failed_worker_is_explicit(self):
        out=supervise(2,[sys.executable,'-c','raise SystemExit(7)'])
        self.assertFalse(out['timed_out']);self.assertEqual(out['exit_code'],7)

if __name__=='__main__':unittest.main()
