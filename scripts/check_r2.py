#!/usr/bin/env python3
"""Check committed R2 configuration and one read request; never print credentials."""
import argparse,json,time,urllib.error,xml.etree.ElementTree as ET,subprocess,sys
from r2_client import R2Client,DEFAULT_CONFIG_PATH,load_r2_config

def check(timeout=5):
 started=time.monotonic();result={'config_file':str(DEFAULT_CONFIG_PATH),'config_tracked_path':'config/r2_storage.json','no_writes':True}
 try:
  client=R2Client(timeout=timeout,deadline=started+timeout);result['configured']=True
  query=client._canonical_query({'list-type':'2','max-keys':1,'prefix':'us_5m/'})
  with client.request('GET',query=query) as response:
   root=ET.fromstring(response.read());result.update(read_access='passed',http_status=response.status,sample_available=bool(root.findall('{http://s3.amazonaws.com/doc/2006-03-01/}Contents')))
 except Exception as error:
  result.setdefault('configured',False);result.update(read_access='failed',error=type(error).__name__)
  if isinstance(error,urllib.error.HTTPError):result['http_status']=error.code
 result['elapsed_seconds']=round(time.monotonic()-started,2)
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--timeout',type=float,default=5);parser.add_argument('--probe-worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
 if not 0<args.timeout<=30:parser.error('--timeout must be between 0 and 30 seconds')
 if args.probe_worker:result=check(args.timeout)
 else:
  # DNS, proxies and response reads can outlast socket timeouts; bound the process too.
  started=time.monotonic()
  try:
   completed=subprocess.run([sys.executable,str(__file__),'--probe-worker','--timeout',str(args.timeout)],capture_output=True,text=True,timeout=args.timeout)
   result=json.loads(completed.stdout)
  except (subprocess.TimeoutExpired,json.JSONDecodeError) as error:
   try:load_r2_config();configured=True
   except Exception:configured=False
   result=dict(config_file=str(DEFAULT_CONFIG_PATH),config_tracked_path='config/r2_storage.json',configured=configured,read_access='failed',error='TimeoutError' if isinstance(error,subprocess.TimeoutExpired) else 'probe_worker_failed',elapsed_seconds=round(time.monotonic()-started,2),no_writes=True)
 print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['read_access']=='passed' else 1)

if __name__=='__main__':main()
