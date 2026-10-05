import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
spec=importlib.util.spec_from_file_location('financial_reuse_under_test',Path(__file__).with_name('reuse_financial_events.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReuseTest(unittest.TestCase):
    def fixture(self,root):
        parent=root/'parent';cache=root/'cache';back=parent/'raw_daily_v3/backtests/example'
        back.mkdir(parents=True);(parent/'joint_v2').mkdir();(cache/'companyfacts').mkdir(parents=True)
        parser=parent/'joint_v2/financials.py';parser.write_text('same frozen parser')
        raw=cache/'companyfacts/CIK0000000001.json';raw.write_text(json.dumps({'cik':1,'facts':{}}))
        event=root/'CIK0000000001_example.parquet'
        pd.DataFrame({'cik':[1],'available_at':pd.to_datetime(['2026-08-01']),'fin_revenue_yoy':[.25]}).to_parquet(event,index=False)
        j={'training_key':'frozen','SEC_financial_event_manifest':{'start':'2023-01-01','end':'2026-10-02',
            'parser_sha256':m.sha(parser),'issuers':[{'cik':1,'path':str(event),'raw_sha256':m.sha(raw)}]},
            'financial_input_hashes':{str(event.resolve()):m.sha(event)}}
        (back/'lineage.json').write_text(json.dumps(j));(parent/'raw_daily_v3/latest_backtest.json').write_text(json.dumps({'path':str(back)}))
        return parent,cache,raw,event
    def test_exact_sources_preserve_real_financial_values(self):
        with tempfile.TemporaryDirectory() as td:
            parent,cache,raw,event=self.fixture(Path(td));before=m.sha(event)
            with patch.object(m,'CACHE',cache),patch.object(m,'PARENT',parent):
                r=m.reuse('2023-01-01','2026-10-02');self.assertEqual(r['copied'],1)
            copied=cache/'financial_events_v2'/event.name
            self.assertEqual(m.sha(copied),before);self.assertEqual(m.sha(event),before)
            self.assertEqual(pd.read_parquet(copied).fin_revenue_yoy.iloc[0],.25)
    def test_changed_raw_source_must_be_parsed_normally(self):
        with tempfile.TemporaryDirectory() as td:
            parent,cache,raw,event=self.fixture(Path(td));raw.write_text(json.dumps({'cik':1,'facts':{'new':{}}}))
            with patch.object(m,'CACHE',cache),patch.object(m,'PARENT',parent):
                self.assertEqual(m.reuse('2023-01-01','2026-10-02')['validated'],0)
            self.assertFalse((cache/'financial_events_v2'/event.name).exists())
    def test_changed_period_does_not_reuse_old_events(self):
        with tempfile.TemporaryDirectory() as td:
            parent,cache,raw,event=self.fixture(Path(td))
            with patch.object(m,'CACHE',cache),patch.object(m,'PARENT',parent):
                self.assertEqual(m.reuse('2023-01-01','2026-10-05')['copied'],0)
            self.assertFalse((cache/'financial_events_v2'/event.name).exists())

if __name__=='__main__':unittest.main()
