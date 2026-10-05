"""One command: each available research arm's top three, with explicit abstention."""
import argparse,json,pickle,subprocess,sys
import numpy as np
import pandas as pd
from common import OUT,OLD,DATES,SESSIONS,write,clean,now,sha
from research import apply_cal,raw_predict,baseline,ARMS,NAMES,ALG
from portable import load as load_portable,load_upstream,probabilities,raw_probability,baseline_table

def source_snapshot(dest,name):
    path=dest/'cache'/name
    return path if path.exists() else OUT/'source_data'/dest.name/name

def unavailable_report(reason):
    report=dict(version='v6_research_reference',calculated_at=now(),current_probability=False,issued_signal=False,orders_sent=False,target='5d5pct',abstention=reason,routes=[])
    for rid,base in [('R00',OUT),('R01',OUT/'R01'),('R02',OUT/'R02'),('R03',OUT/'R03'),('R04',OUT/'R04')]:
        path=base/'results.json'
        if path.exists():
            for arm in json.loads(path.read_text())['arms']:
                report['routes'].append(dict(round=rid,arm=arm['arm'],name=arm['name'],algorithm=arm['algorithm'],options=[],reason=reason))
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--top',type=int,default=3);ap.add_argument('--refresh',action='store_true');ap.add_argument('--round',choices=['all','R00','R01','R02','R04'],default='all');ap.add_argument('--format',choices=['json','markdown'],default='markdown');a=ap.parse_args()
    if not 1<=a.top<=20:ap.error('top must be 1–20')
    path=OUT/'cache/latest_features.parquet'
    if a.refresh or not path.exists():
        result=subprocess.run([sys.executable,str(OUT/'latest_features.py')],stdout=sys.stderr,stderr=subprocess.PIPE,text=True)
        if result.returncode:
            report=unavailable_report('latest causal minute coverage unavailable; run free data preparation; abstained')
            write(OUT/'reference_latest.json',report)
            print(json.dumps(clean(report),ensure_ascii=False,indent=2) if a.format=='json' else '当前有效信号：弃权。缺少最新因果分钟行情，请先运行免费数据准备。')
            return
    frame=pd.read_parquet(path);meta=json.loads((OUT/'latest_features_metadata.json').read_text());day=meta['day'];minute=meta['minute'];phase='pre' if minute<=570 else 'regular'
    endday=DATES[DATES.index(day)+4];deadline=endday+' '+('13:00' if SESSIONS[endday]['duration_minutes']==210 else '16:00')+' America/New_York'
    report=dict(version='v6_research_reference',calculated_at=now(),feature_cutoff=f'{day} {minute//60:02d}:{minute%60:02d} America/New_York',current_probability=False,issued_signal=False,orders_sent=False,
        target='5d5pct',window_end=deadline,abstention='research models not independently admitted; past session is not a current signal',routes=[])
    rounds=['R00','R01','R02','R04'] if a.round=='all' else [a.round]
    for round_id in rounds:
      dest=OUT if round_id=='R00' else OUT/round_id
      if not (dest/'results.json').exists():continue
      registered_arms=json.loads((dest/'results.json').read_text())['arms']
      roundframe=frame.copy();upstream_meta=None
      if round_id=='R01':
        native=load_upstream()
        if native:
            upstream,item=native;roundframe=roundframe[roundframe.symbol.isin(upstream['registered'])].copy();roundframe['opp_oof_raw']=raw_probability(upstream,roundframe)
            upstream_meta=dict(training=upstream['training'],max_training_label_end=upstream['max_training_label_end'],model_sha256=item['sha256'],score_is_calibrated=False)
        else:
            blocks=json.loads((dest/'upstream_provenance.json').read_text())['blocks'];last=next(b for b in reversed(blocks) if b['status']=='fitted')
            upstream_path=dest/'models'/f"daily_opportunity_block_{last['block']:03d}.pkl";assert sha(upstream_path)==last['model_sha256']
            upstream=pickle.loads(upstream_path.read_bytes());roundframe=roundframe[roundframe.symbol.isin(last['registered'])].copy();roundframe['opp_oof_raw']=upstream.predict_proba(roundframe[last['features']])[:,1]
            upstream_meta=dict(training=last['training'],max_training_label_end=last['max_training_label_end'],model_sha256=last['model_sha256'],score_is_calibrated=False)
      if round_id in ['R02','R04']:
        from research_extensions import peers_exact,event_features,quarter_features
        prefix=pd.read_parquet(OUT/'cache/latest_base.parquet')
        roundframe,_=event_features(peers_exact(prefix),latency_seconds=300 if round_id=='R04' else 60,events_path=source_snapshot(dest,'sec_events_source.json'),audit_path=source_snapshot(dest,'sec_audit_source.json'))
        if round_id=='R04':roundframe,_=quarter_features(roundframe,source_path=source_snapshot(dest,'sec_facts_source.json'))
        roundframe=roundframe[roundframe.minute==minute].copy()
      for arm_meta in registered_arms:
        arm=arm_meta['arm'];name=arm_meta['name'];algorithm=arm_meta['algorithm']
        native=load_portable(dest,arm);models=sorted((dest/'models').glob(f'{arm}_*.pkl'));options=[]
        if not native and not models:report['routes'].append(dict(round=round_id,arm=arm,name=name,algorithm=algorithm,options=[],reason='free event PIT coverage unavailable; no trained model'));continue
        if native:model,item=native;model_sha=item['sha256']
        else:p=models[-1];model=pickle.loads(p.read_bytes());model_sha=sha(p)
        f=roundframe[roundframe.symbol.isin(model['registered'])].copy()
        if f.empty:
            report['routes'].append(dict(round=round_id,arm=arm,name=name,algorithm=algorithm,options=[],reason='registered current feature coverage unavailable'));continue
        if native:f['probability']=probabilities(model,f,phase);f['baseline']=baseline(baseline_table(model),f)
        else:
            raw=raw_predict(model['model'],model['features'],f);f['probability']=apply_cal(model['top_calibrators'][phase],apply_cal(model['candidate_calibrators'][phase],raw));f['baseline']=baseline(model['baseline'],f)
        f=f.sort_values(['probability','symbol'],ascending=[False,True]);historical=model['variants']['top_one'][phase]
        for r in f.head(a.top).to_dict('records'):
            actual=pd.read_parquet(OLD/'raw'/f"{r['symbol']}.parquet",columns=['day','minute','open']);actual=actual[(actual.day==day)&(actual.minute==minute+5)]
            price=float(actual.open.iloc[0])*1.001 if len(actual)==1 else float(r['reference'])*1.001
            options.append(dict(symbol=r['symbol'],probability=r['probability'],matched_baseline=r['baseline'],effective_threshold=None,historical_threshold=historical,
                 evaluation_price=price,target_price=price*1.05,price_status='historical_next_Open_known' if len(actual)==1 else 'indicative_close_only',
                 evaluation_entry=f'{day} {(minute+5)//60:02d}:{(minute+5)%60:02d} America/New_York',window_end=deadline,issued_signal=False,
                 reason='no current frozen admission; probabilities are research estimates, not proven precision'))
        report['routes'].append(dict(round=round_id,arm=arm,name=name,algorithm=algorithm,model_month=model['month'],model_sha256=model_sha,artifact_format='data_only_native' if native else 'local_training_pickle',upstream=upstream_meta,options=options,scored=len(f)))
    write(OUT/'reference_latest.json',report)
    if a.format=='json':print(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False))
    else:
        print(f"五日+5%研究参考；行情截止 {report['feature_cutoff']}；截止 {deadline}。当前有效信号：弃权。")
        for route in report['routes']:
            print('\n'+route['round']+' '+route['arm']+' '+route['name']+' / '+route['algorithm'])
            for r in route['options']:print(f"{r['symbol']}: 概率估计{r['probability']:.2%}，目标${r['target_price']:.4f}，有效门槛未获准，未发信号。")
            if not route['options']:print(route['reason'])
if __name__=='__main__':main()
