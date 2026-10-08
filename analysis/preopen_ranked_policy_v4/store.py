"""Idempotent local SQLite ledger; prediction, decision and future outcome separate."""
from __future__ import annotations
import json,sqlite3
from common import *

def connect(path=None):
 con=sqlite3.connect(path or OUT/'research.sqlite',timeout=30);con.row_factory=sqlite3.Row;con.execute('PRAGMA journal_mode=WAL')
 con.executescript('''
 CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,mode TEXT,route TEXT,period TEXT,created_at TEXT,model_hash TEXT,threshold REAL,metadata TEXT);
 CREATE TABLE IF NOT EXISTS candidates(run_id TEXT,day TEXT,minute INTEGER,symbol TEXT,rank INTEGER,score REAL,baseline REAL,reference REAL,qualified INTEGER,feature_available TEXT,PRIMARY KEY(run_id,day,minute,symbol));
 CREATE TABLE IF NOT EXISTS decisions(run_id TEXT,day TEXT,minute INTEGER,action TEXT,reason TEXT,top_symbol TEXT,top_score REAL,qualified INTEGER,payload TEXT,PRIMARY KEY(run_id,day,minute));
 CREATE TABLE IF NOT EXISTS trades(run_id TEXT,id INTEGER,day TEXT,symbol TEXT,decision_minute INTEGER,entry_minute INTEGER,entry REAL,target REAL,qty INTEGER,score REAL,payload TEXT,PRIMARY KEY(run_id,id));
 CREATE TABLE IF NOT EXISTS outcomes(run_id TEXT,trade_id INTEGER,exit_minute INTEGER,exit REAL,hit INTEGER,net REAL,mae REAL,known_at TEXT,PRIMARY KEY(run_id,trade_id));
 CREATE TABLE IF NOT EXISTS equity(run_id TEXT,day TEXT,minute INTEGER,value REAL,cash REAL,unsettled REAL,position TEXT,PRIMARY KEY(run_id,day,minute));
 CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY,observed_at TEXT,decision_key TEXT UNIQUE,market_state TEXT,action TEXT,reason TEXT,payload TEXT);
 CREATE TABLE IF NOT EXISTS live_fills(observation_id INTEGER UNIQUE,symbol TEXT,entry_at TEXT,entry REAL,target REAL,qty INTEGER,exit_at TEXT,exit REAL,hit INTEGER,net REAL,payload TEXT);
 CREATE INDEX IF NOT EXISTS candidate_lookup ON candidates(run_id,day,minute,rank);
 ''');return con

def dump(x):return json.dumps(clean(x),ensure_ascii=False,separators=(',',':'),allow_nan=False)
def persist(run_id,route,month,frame,replay,meta):
 con=connect()
 old=con.execute('SELECT model_hash FROM runs WHERE id=?',(run_id,)).fetchone()
 if old:
  con.close()
  if old['model_hash']!=meta['artifact_sha256']:raise RuntimeError('Immutable run id has a different model hash: '+run_id)
  return
 with con:
  con.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)',(run_id,'historical_forward',route,month,datetime.now(ET).isoformat(),meta['artifact_sha256'],meta['threshold'],dump(meta)))
  f=frame.sort_values(['day','minute','score','symbol'],ascending=[True,True,False,True]);f['rank']=f.groupby(['day','minute'],observed=True).cumcount()+1
  threshold=meta['threshold'];rows=[(run_id,str(r.day),int(r.minute),str(r.symbol),int(r.rank),float(getattr(r,'score_original',r.score)),float(r.baseline),float(r.reference),int(threshold is not None and r.score>=threshold and r.ex_action==0),r.feature_available) for r in f.itertuples()]
  con.executemany('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?)',rows)
  con.executemany('INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?)',[(run_id,d['day'],d['minute'],d['action'],d['reason'],d['top_symbol'],d['top_score'],d['qualified'],dump(d)) for d in replay['decisions']])
  for t in replay['trades']:
   forecast={k:v for k,v in t.items() if k not in ['exit','exit_minute','y','net','mae']}
   con.execute('INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?)',(run_id,t['id'],t['day'],t['symbol'],t['minute'],t['entry_minute'],t['entry'],t['target'],t['qty'],t['score'],dump(forecast)))
   con.execute('INSERT INTO outcomes VALUES(?,?,?,?,?,?,?,?)',(run_id,t['id'],t['exit_minute'],t['exit'],t['y'],t['net'],t['mae'],t['day']+'T'+f"{t['exit_minute']//60:02d}:{t['exit_minute']%60:02d}:01-04:00"))
  con.executemany('INSERT INTO equity VALUES(?,?,?,?,?,?,?)',[(run_id,e['day'],e['minute'],e['value'],e['cash'],e['unsettled'],e['position']) for e in replay['equity']])
 con.close()
