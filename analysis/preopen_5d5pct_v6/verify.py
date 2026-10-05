"""Independent raw-price label checks and model-load equivalence for R00 artifacts."""
import argparse,json,pickle
import numpy as np
import pandas as pd
from common import OUT,OLD,DATES,SESSIONS,write,sha,now
from research import apply_cal,raw_predict

def expected_event(e,raw,actions):
    start=pd.Timestamp(e['day'])+pd.Timedelta(minutes=int(e['minute'])+5)
    row=raw[raw.start==start]
    if len(row)!=1:return None,'missing_entry',None,None
    entry=float(row.open.iloc[0])*1.001;target=entry*1.05
    days=DATES[DATES.index(e['day']):DATES.index(e['day'])+5]
    if days[-1]>'2026-09-30':return None,'pending_window',entry,target
    if set(days)&actions:return None,'corporate_action_window',entry,target
    peak=-np.inf
    for i,day in enumerate(days):
        close=570+SESSIONS[day]['duration_minutes'];begin=max(570,int(e['minute'])+5) if i==0 else 570
        g=raw[(raw.day==day)&(raw.minute>=begin)&(raw.minute<close)]
        if not np.array_equal(g.minute.to_numpy(),np.arange(begin,close,5)):return None,'missing_future_bars',entry,target
        prices=g[['open','high','low','close','volume']].to_numpy(float)
        if not np.isfinite(prices).all() or not (prices[:,:4]>0).all() or not (g.high>=g[['open','close']].max(axis=1)).all() or not (g.low<=g[['open','close']].min(axis=1)).all() or not (g.volume>=0).all():return None,'missing_future_bars',entry,target
        peak=max(peak,float(g.high.max()))
    return int(peak>=target),'mature',entry,target

def main():
    global OUT
    base_out=OUT
    ap=argparse.ArgumentParser();ap.add_argument('--round',choices=['R00','R01','R02','R03','R04','R05','R06'],default='R00');arguments=ap.parse_args()
    if arguments.round!='R00':OUT=OUT/arguments.round
    errors=[];checked=0;price_rounding=0;cache={};actions={};predictions=[];rank_ticks=0;purges=[]
    coverage=json.loads((OUT/'coverage.json').read_text());assert sha(OUT/'cache/panel.parquet')==coverage['panel_sha256']
    for path in sorted((OUT/'cache/runs').glob('*/meta.json')):
        meta=json.loads(path.read_text())
        if meta['status']!='completed_development':continue
        artifact_path=OUT/'models'/f"{meta['arm']}_{meta['month']}.pkl";assert sha(artifact_path)==meta['model_sha256']
        model=pickle.loads(artifact_path.read_bytes())
        ranges=[meta['training'],meta['candidate_calibration'],meta['top_calibration'],meta['selection'],[meta['month']+'-01',meta['month']+'-28']]
        for left,right in zip(ranges,ranges[1:]):
            gap=sum(left[-1]<d<right[0] for d in DATES)
            if gap<10:errors.append(dict(kind='purge',arm=meta['arm'],month=meta['month'],gap=gap))
        if set(meta['features'])&{'y','entry','target','label_end','future_status','_row_id'}:errors.append(dict(kind='future_feature',arm=meta['arm']))
        purges.append(dict(arm=meta['arm'],month=meta['month'],ranges=ranges))
        panel_path=OUT/'cache'/meta.get('panel_file','panel.parquet')
        assert sha(panel_path)==meta['panel_sha256']
        features=pd.read_parquet(panel_path,columns=['symbol','day','minute']+meta['features'],filters=[('day','>=',meta['month']+'-01'),('day','<',meta['month']+'-32')]).head(250)
        features['symbol']=features.symbol.astype(str);features=features[features.symbol.isin(model['registered'])]
        saved=pd.read_parquet(path.parent/'top_one.parquet').merge(features[['symbol','day','minute']],on=['symbol','day','minute'],validate='one_to_one')
        merged=features.merge(saved[['symbol','day','minute','score']],on=['symbol','day','minute'],validate='one_to_one')
        p=raw_predict(model['model'],model['features'],merged)
        for phase in ['pre','regular']:
            ix=(merged.minute<=570).to_numpy() if phase=='pre' else (merged.minute>570).to_numpy()
            p[ix]=apply_cal(model['top_calibrators'][phase],apply_cal(model['candidate_calibrators'][phase],p[ix]))
        err=float(np.max(np.abs(p-merged.score.to_numpy()))) if len(p) else 0.
        predictions.append(dict(arm=meta['arm'],month=meta['month'],rows=len(p),max_error=err))
        if err>1e-10:errors.append(dict(kind='prediction',arm=meta['arm'],month=meta['month'],max_error=err))
        for variant in ['candidate_only','top_one']:
            sim=json.loads((path.parent/f'{variant}_replay.json').read_text());keys=set()
            scored=pd.read_parquet(path.parent/f'{variant}.parquet',columns=['symbol','day','minute','score'])
            actual={(e['day'],int(e['minute'])):e['symbol'] for e in sim['events']}
            issued=set();previous=None
            for (day,minute),group in scored.groupby(['day','minute'],sort=True,observed=True):
                if day!=previous:issued=set();previous=day
                phase='pre' if minute<=570 else 'regular';threshold=model['variants'][variant][phase]
                eligible=group[group.score.notna() & ~group.symbol.isin(issued)] if threshold is not None else group.iloc[:0]
                if threshold is not None:eligible=eligible[eligible.score>=threshold]
                expected=None
                if len(eligible):
                    expected=min(eligible.itertuples(),key=lambda r:(-r.score,r.symbol)).symbol;issued.add(expected)
                rank_ticks+=1
                if expected!=actual.get((day,int(minute))):errors.append(dict(kind='first_rank',arm=meta['arm'],variant=variant,day=day,minute=int(minute),expected=expected,actual=actual.get((day,int(minute)))))
            for e in sim['events']:
                key=(e['day'],e['symbol'])
                if key in keys:errors.append(dict(kind='duplicate',key=key))
                keys.add(key)
                if not e['score']>=e['threshold']:errors.append(dict(kind='threshold',key=key))
                symbol=e['symbol']
                if symbol not in cache:
                    source=base_out/'cache/backfill'/f'{symbol}.parquet' if arguments.round in ['R03','R05','R06'] else OLD/'raw'/f'{symbol}.parquet'
                    if not source.exists():source=OLD/'raw'/f'{symbol}.parquet'
                    cache[symbol]=pd.read_parquet(source)
                    ad=set()
                    for base in [OLD.parents[1]/'market_data/model_training_history_v1/corporate_actions',OLD.parents[1]/'market_data/corporate_actions',OLD.parents[1]/'analysis/preopen_intraday_v2/cache/corporate_actions']:
                        pth=base/f'{symbol}.parquet'
                        if pth.exists():action_data=pd.read_parquet(pth);ad.update(action_data.ex_div_date.astype(str).str[:10])
                    if OUT.name in ['R03','R04','R05','R06']:
                        pth=base_out/'cache/corporate_actions'/f'{symbol}.parquet'
                        if pth.exists():action_data=pd.read_parquet(pth);ad.update(action_data.ex_div_date.astype(str).str[:10])
                    actions[symbol]=ad
                y,status,entry,target=expected_event(e,cache[symbol],actions[symbol]);checked+=1
                if y!=e['y'] or status!=e['future_status']:errors.append(dict(kind='label',symbol=symbol,day=e['day'],expected=[y,status],saved=[e['y'],e['future_status']]))
                if entry is not None and (abs(entry-e['entry'])>1e-8 or abs(target-e['target'])>1e-8):price_rounding+=1
    upstream=[]
    provenance=OUT/'upstream_provenance.json'
    if provenance.exists():
        blocks=json.loads(provenance.read_text())['blocks'];scores=pd.read_parquet(OUT/'cache/opportunity_scores.parquet')
        dcols=next(b['features'] for b in blocks if b['status']=='fitted')
        inputs=pd.read_parquet(OUT/'cache/panel.parquet',columns=['symbol','day']+dcols).drop_duplicates(['symbol','day'])
        inputs['symbol']=inputs.symbol.astype(str);inputs['day']=inputs.day.astype(str)
        for b in blocks:
            if b['status']!='fitted':continue
            if not b['max_training_label_end']<b['start']:errors.append(dict(kind='OOF_maturity',block=b['block']))
            p=OUT/'models'/f"daily_opportunity_block_{b['block']:03d}.pkl"
            if sha(p)!=b['model_sha256']:errors.append(dict(kind='OOF_model_hash',block=b['block']))
            rows=scores[(scores.day>=b['start'])&(scores.day<=b['end'])].head(150).merge(inputs,on=['symbol','day'],validate='one_to_one')
            fitted=pickle.loads(p.read_bytes());pred=fitted.predict_proba(rows[dcols])[:,1]
            err=float(np.max(np.abs(pred-rows.opp_oof_raw))) if len(rows) else 0
            upstream.append(dict(block=b['block'],rows=len(rows),max_error=err))
            if err>1e-10:errors.append(dict(kind='OOF_probability',block=b['block'],max_error=err))
    if price_rounding:errors.append(dict(kind='reporting_price',rows=price_rounding))
    write(OUT/'verification.json',dict(at=now(),status='passed' if not errors else 'failed',events_checked=checked,ranking_ticks_checked=rank_ticks,predictions=predictions,upstream=upstream,purges=purges,errors=errors,
       reporting_float32_price_rows=price_rounding,reporting_price_note='Exact original float64 label evaluation prices; reporting repair does not change emissions or outcomes.'))
    print(json.dumps(dict(events=checked,errors=len(errors),float32_reporting_rows=price_rounding)),flush=True)
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
