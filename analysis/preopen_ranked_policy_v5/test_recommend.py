import argparse,json,tempfile,unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from common import OUT,ET,ROUTES
from portable import NativeModel,digest
from cloud_data import MarketData,normalize_archive
from recommend import run,schedule

def args(**kwargs):
 d=dict(data_dir=None,offline=True,futu_host=None,futu_port=11111,as_of=None,top=3,save_features=None,output=None,record=None);d.update(kwargs);return argparse.Namespace(**d)

class Recommendation(unittest.TestCase):
 def test_native_file_hash_enforced(self):
  with self.assertRaises(ValueError):NativeModel(OUT/'portable_models/hazard_linear_2026-10.json.gz','wrong')
 def test_missing_inputs_use_reference_and_report_every_model(self):
  with tempfile.TemporaryDirectory() as td:
   result=run(args(data_dir=td),datetime(2026,10,4,12,tzinfo=ET));self.assertFalse(result['current_probability']);self.assertFalse(result['options_are_issued_signals']);self.assertEqual(result['data_source'],'bundled_reference_snapshot');self.assertEqual([r['route'] for r in result['routes']],ROUTES)
   for route in result['routes']:
    self.assertEqual(len(route['options']),3);self.assertIsNone(route['threshold']);self.assertTrue(all(not o['issued_signal'] and not o['current_qualified_candidate'] for o in route['options']));p=[o['probability'] for o in route['options']];self.assertEqual(p,sorted(p,reverse=True))
 def test_future_reference_not_used_before_its_cutoff(self):
  with tempfile.TemporaryDirectory() as td:
   r=run(args(data_dir=td,as_of='2026-10-02T08:00:30-04:00'),datetime(2026,10,2,8,0,30,tzinfo=ET));self.assertFalse(r['current_probability']);self.assertTrue(all(not m['options'] for m in r['routes']))
 def test_target_is_delayed_entry_formula_not_reference_fixed_price(self):
  with tempfile.TemporaryDirectory() as td:
   r=run(args(data_dir=td),datetime(2026,10,4,12,tzinfo=ET));o=r['routes'][0]['options'][0];self.assertEqual(o['target_formula'],'actual_next_5m_open * 1.001 * 1.03');self.assertAlmostEqual(o['indicative_target_price'],o['reference_price']*1.001*1.03);self.assertIn('15:35',o['evaluation_entry_at']);self.assertIn('16:00',o['window_end'])
 def test_decision_offset_and_short_window(self):
  self.assertEqual(schedule(datetime(2026,10,5,9,40,29,tzinfo=ET))['cut'],575);self.assertEqual(schedule(datetime(2026,10,5,9,40,30,tzinfo=ET))['cut'],580);self.assertFalse(schedule(datetime(2026,10,5,15,31,tzinfo=ET))['in_window']);self.assertFalse(schedule(datetime(2026,10,4,12,tzinfo=ET))['in_window'])
 def test_custom_empty_directory_does_not_read_other_local_wip(self):
  with tempfile.TemporaryDirectory() as td:
   m=MarketData(td)
   with patch('cloud_data.read_raw',side_effect=AssertionError('must not read other directory')):self.assertTrue(m.raw('AAOI').empty)
 def test_unknown_adjustment_basis_rejected(self):
  with self.assertRaises(ValueError):normalize_archive(pd.DataFrame(dict(time_key=['2026-10-02 09:35'],open=[100],high=[101],low=[99],close=[100],volume=[1])))
 def test_r2_cache_rechecks_etag_and_preserves_actual_receipt(self):
  class Client:
   etag='one';downloads=0
   def list_objects(self,prefix):return [dict(key='us_5m/AAOI/2026.parquet',etag=self.etag)] if prefix=='us_5m/' else []
   def get_object(self,key,path):
    self.downloads+=1;pd.DataFrame(dict(start_at_et=['2026-09-25T09:30:00-04:00'],end_at_et=['2026-09-25T09:35:00-04:00'],price_basis=['NONE'],open=[100.],high=[102.],low=[98.],close=[100.+(self.etag=='two')],volume=[10.])).to_parquet(path)
  client=Client()
  with tempfile.TemporaryDirectory() as td,patch('scripts.r2_client.R2Client',return_value=client),patch('cloud_data.socket.create_connection',side_effect=OSError('unreachable')):
   m=MarketData(td);m.refresh(['AAOI'],'2026-10-05',245);self.assertEqual(client.downloads,1);stamp=Path(td)/'r2_cache/us_5m/AAOI/2026.etag.json';metadata=json.loads(stamp.read_text());metadata['received_at']='2026-10-01T00:00:00+00:00';stamp.write_text(json.dumps(metadata));m.refresh(['AAOI'],'2026-10-05',245);self.assertEqual(client.downloads,1);self.assertEqual(m.audit[-1]['received_at'],'2026-10-01T00:00:00+00:00');client.etag='two';m.refresh(['AAOI'],'2026-10-05',245);self.assertEqual(client.downloads,2);self.assertEqual(float(m.raw('AAOI').close.iloc[0]),101.)
 def test_portable_reference_matches_committed_probabilities(self):
  manifest=json.loads((OUT/'portable_models/manifest.json').read_text());snapshot=json.loads((OUT/'portable_models/reference_snapshot.json').read_text());f=pd.DataFrame(snapshot['features']);expected=json.loads((OUT/'reference_recommendation.json').read_text());self.assertEqual(snapshot['model_manifest_sha256'],digest(OUT/'portable_models/manifest.json'))
  for m in manifest['all_routes']:
   a=NativeModel(OUT/'portable_models'/m['file'],m['sha256']);g=f[f.symbol.isin(a.a['registered'])].copy();g['stock_id']=g.symbol.map(a.a['stock_map'])
   for c in a.a['features']:
    if c!='symbol':g[c]=g[c].astype('float32')
   g['p']=a.predict(g);g=g.sort_values(['p','symbol'],ascending=[False,True]).head(3);e=next(r for r in expected['routes'] if r['route']==m['route']);self.assertEqual(list(g.symbol),[o['symbol'] for o in e['options']]);np.testing.assert_allclose(g.p,[o['probability'] for o in e['options']],rtol=0,atol=1e-10)
if __name__=='__main__':unittest.main()
