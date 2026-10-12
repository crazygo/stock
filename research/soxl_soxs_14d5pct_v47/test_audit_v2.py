"""Exercise independent raw-window auditing on mixed metadata/numeric panels."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_options import raw_fixture
from option_learning import build_features, dump, pd
from option_features import option_columns
from audit_option_features_v2 import audit_option_features


class AuditTests(unittest.TestCase):
    def execute(self, tamper=False):
        raw,cal,f=raw_fixture()
        added,_,receipts=build_features(f,raw,cal['sessions'])
        panel=f.join(added)
        panel['clock']='09:25:30'
        panel['feature_ready']=panel.option_ready
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);source=run/'source';source.mkdir()
            panel[['feature_ready']].to_parquet(source/'financial_panel.parquet')
            if tamper:
                at=panel[panel.option_ready].index[-1]
                panel.loc[at,option_columns(2)[0]]+=.01
            panel.to_parquet(run/'option_panel.parquet')
            added.to_parquet(run/'option_features.parquet')
            dump(run/'option_receipts.json',receipts)
            dump(run/'input_manifest.json',{'source':str(source),'option_calendar':cal,'option_source':{}})
            with patch('audit_option_features_v2.load_raw',return_value=(raw,{})):
                return audit_option_features(run)

    def test_rebuilds_all_lagged_values_with_metadata_columns(self):
        result=self.execute()
        self.assertTrue(result['passed'])
        self.assertGreater(result['feature_equations'],400)

    def test_corrupted_panel_feature_is_rejected(self):
        with self.assertRaises(AssertionError):self.execute(tamper=True)


if __name__=='__main__':unittest.main()
