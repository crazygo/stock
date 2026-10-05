import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.r2_client import R2Client,load_r2_config

CONFIG=dict(endpoint='https://example.invalid',bucket='test',region='auto',access_key_id='test-id',secret_access_key='test-secret')

class CloudAccess(unittest.TestCase):
 def test_request_uses_remaining_budget_and_stops_before_network_after_deadline(self):
  client=R2Client(config=CONFIG,timeout=5,deadline=time.monotonic()+.2)
  with patch('urllib.request.urlopen') as opener:
   client.request('GET');self.assertGreater(opener.call_args.kwargs['timeout'],0);self.assertLessEqual(opener.call_args.kwargs['timeout'],.2)
   client.deadline=time.monotonic()-1
   with self.assertRaises(TimeoutError):client.request('GET')
   self.assertEqual(opener.call_count,1)
 def test_malformed_config_has_actionable_error_without_exposing_contents(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'bad.json';p.write_text('{"secret_access_key":"must-not-print"')
   with self.assertRaises(ValueError) as error:load_r2_config(p)
   self.assertIn(str(p),str(error.exception));self.assertNotIn('must-not-print',str(error.exception))
 def test_expired_download_preserves_existing_file_and_removes_partial(self):
  client=R2Client(config=CONFIG,deadline=time.monotonic()-1)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'data.parquet';p.write_bytes(b'existing')
   with patch('urllib.request.urlopen') as opener:
    with self.assertRaises(TimeoutError):client.get_object('archive',p)
    opener.assert_not_called()
   self.assertEqual(p.read_bytes(),b'existing');self.assertFalse(p.with_suffix('.parquet.r2download').exists())

if __name__=='__main__':unittest.main()
