"""Prospective validity is fixed at issue time, independently of paper actions."""
from common import *
from store import connect
from signals import select,put_run,put_tick,put_event,summary
from data import LABELS
from phase import phase_name
import json

def deployment():return json.loads((OUT/'signals_deployment.json').read_text())

def evaluate(rows,day,minute,route,active,dep=None):
 dep=dep or deployment();meta=next(m for m in dep['all_routes'] if m['route']==route);phase=phase_name(minute);threshold=meta['thresholds'][phase]
 run=f'signal_v1:{route}:2026-10_observation';con=connect();old=con.execute('SELECT payload FROM signal_ticks WHERE run_id=? AND day=? AND minute=?',(run,day,minute)).fetchone()
 if old and active:
  tick=json.loads(old['payload']);con.close();return tick
 issued={r[0] for r in con.execute('SELECT symbol FROM signal_events WHERE run_id=? AND day=?',(run,day))}
 if not active:top=None;reason='outside_signal_window';eligible=[];ranked=[]
 elif not dep['admitted'] or dep['route']!=route:top=None;reason='signal_model_not_admitted';eligible=[];ranked=[]
 else:
  candidates=[dict(r['feature_row'],score=r['score'],baseline=r['baseline'],reference=r['reference'],feature_available=r['feature_available'],day=day,minute=minute,symbol=r['symbol']) for r in rows]
  top,reason,eligible,ranked=select(candidates,threshold,issued)
 tick=dict(day=day,minute=minute,symbol=top['symbol'] if top else None,reason=reason,eligible=len(eligible),top20=ranked,threshold=threshold,route=route,valid=bool(top),run_id=run,observed_at=datetime.now(ET).isoformat())
 if active:
  with con:
   put_run(con,run,route,'prospective_signal',dict(frozen_at=dep['frozen_at'],configuration=meta,protocol_sha256=dep['protocol_sha256']))
   put_tick(con,run,tick)
   if top:
    features={k:clean(v) for k,v in top.items() if k not in LABELS+['score','baseline','reference','feature_available']}
    e=dict(day=day,symbol=top['symbol'],minute=minute,phase=phase,score=top['score'],threshold=threshold,baseline=top['baseline'],reference=top['reference'],feature_available=top['feature_available'],feature_row=features,issued_at=tick['observed_at']);put_event(con,run,e)
 con.close();return tick

def observation_summary(con):
 dep=deployment();route=dep['route'] or 'recovery_forest';run=f'signal_v1:{route}:2026-10_observation'
 return dict(route=route,run_id=run,metrics=summary(con,run))
