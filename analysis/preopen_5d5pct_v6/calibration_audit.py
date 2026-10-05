"""Probability evidence on candidates, policy firsts and identical held cohorts.

Post-fit diagnostics only: never changes a model, threshold, or signal.
"""
import argparse,json
import numpy as np
import pandas as pd
from common import OUT,write,now
from research import replay,probability_report,weights

KEY=['symbol','day','minute']

def date_ci(f,values,w,iterations=1000):
    x=pd.DataFrame({'day':f.day.to_numpy(),'sum':values*w,'weight':w})
    x['week']=(pd.to_datetime(x.day)-pd.to_timedelta(pd.to_datetime(x.day).dt.weekday,unit='D')).dt.strftime('%Y-%m-%d')
    sums=x.groupby('week')[['sum','weight']].sum().to_numpy()
    if not len(sums) or not sums[:,1].sum():return [None,None]
    rng=np.random.default_rng(1005);v=[];n=len(sums)
    for _ in range(iterations):
        starts=rng.integers(n,size=(n+1)//2);ix=np.array([[i,(i+1)%n] for i in starts]).ravel()[:n];s=sums[ix].sum(0)
        if s[1]:v.append(s[0]/s[1])
    return np.quantile(v,[.025,.975]).tolist()

def report(f):
    known=f[f.y.notna()].copy()
    if known.empty:return dict(stock_days=0,brier=None,baseline_brier=None,bins=[],reliable_high_probability=False)
    r=probability_report(known);w=weights(known)
    r.update(dates=known.day.nunique(),rows=len(known),pending_rows=int(f.y.isna().sum()),
        brier_delta_ci=date_ci(known,((known.score-known.y)**2-(known.baseline-known.y)**2).to_numpy(),w),
        overestimate_ci=date_ci(known,(known.score-known.y).to_numpy(),w))
    high=known[known.score>=.8].copy()
    h=probability_report(high);h.update(dates=high.day.nunique())
    if len(high):
        h['overestimate']=h['predicted']-h['observed'];h['overestimate_ci']=date_ci(high,(high.score-high.y).to_numpy(),weights(high))
    r['high_probability']=h
    r['reliable_high_probability']=bool(h['stock_days']>=40 and h['dates']>=20 and h.get('overestimate',1)<=.05 and r['brier']<=r['baseline_brier'])
    r['monthly_high_probability']=[]
    for month,g in known.groupby(known.day.str[:7]):
        z=g[g.score>=.8];item=dict(month=month,stock_days=z[['symbol','day']].drop_duplicates().shape[0],dates=z.day.nunique())
        if len(z):
            wz=weights(z);item.update(predicted=float(np.average(z.score,weights=wz)),observed=float(np.average(z.y,weights=wz)),overestimate_ci=date_ci(z,(z.score-z.y).to_numpy(),wz))
        r['monthly_high_probability'].append(item)
    return r

def load(dest,arm,variant):
    files=sorted((dest/'cache/runs').glob(f'{arm}_*/{variant}.parquet'))
    if not files:return pd.DataFrame()
    f=pd.concat([pd.read_parquet(p,columns=KEY+['score','baseline','y']) for p in files],ignore_index=True)
    f['symbol']=f.symbol.astype(str);f['day']=f.day.astype(str)
    assert not f.duplicated(KEY).any()
    return f

def firsts(f):
    # First-candidate diagnostic follows stock/day dedup across both phases.
    issued=set();events=[];prev=None
    for (day,minute),g in f.sort_values(['day','minute','score','symbol'],ascending=[True,True,False,True]).groupby(['day','minute'],sort=True):
        if day!=prev:issued=set();prev=day
        eligible=g[g.score.notna()&~g.symbol.isin(issued)]
        if len(eligible):e=eligible.iloc[0];events.append(e.to_dict());issued.add(e.symbol)
    return pd.DataFrame(events)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round',choices=['R00','R01','R02','R03','R04','R05'],default='R00');a=ap.parse_args();dest=OUT if a.round=='R00' else OUT/a.round
    results=json.loads((dest/'results.json').read_text());audits=[]
    for arm in results['arms']:
        frames={v:load(dest,arm['arm'],v) for v in ['candidate_only','top_one']}
        if frames['candidate_only'].empty:continue
        for v,f in frames.items():
            first=firsts(f);signals=json.loads((dest/f"{arm['arm']}_{v}_signals.json").read_text())['events'];signal=pd.DataFrame(signals)
            issued=report(signal) if len(signal) else dict(stock_days=0,reliable_high_probability=False)
            audits.append(dict(arm=arm['arm'],variant=v,all_candidates=report(f),first_candidates=report(first),issued_signals=issued))
            result=next(x for x in arm['variants'] if x['variant']==v)
            result.update(all_candidates=audits[-1]['all_candidates'],first_candidates=audits[-1]['first_candidates'],signal_probability=issued)
            favorites=result['favorite_five_passed'];core=result.pop('historical_constraints_met',result.get('historical_core_constraints_met',False))
            result['historical_core_constraints_met']=core
            result['historical_all_constraints_met']=bool(core and len(favorites)>=5 and issued['reliable_high_probability'])
        # Same held first-candidate rows: calibration change is measured without
        # replacing the cohort with a newly selected set of outcomes.
        first=firsts(frames['candidate_only']);merged=first[KEY+['y','baseline','score']].merge(frames['top_one'][KEY+['score']],on=KEY,suffixes=('_candidate','_top'),validate='one_to_one')
        valid=merged[merged.y.notna()].copy();w=weights(valid)
        delta=((valid.score_top-valid.y)**2-(valid.score_candidate-valid.y)**2).to_numpy()
        audits.append(dict(arm=arm['arm'],comparison='top_calibration_on_identical_candidate_first_cohort',stock_days=valid[['symbol','day']].drop_duplicates().shape[0],
            brier_candidate=float(np.average((valid.score_candidate-valid.y)**2,weights=w)),brier_top=float(np.average((valid.score_top-valid.y)**2,weights=w)),
            brier_delta=float(np.average(delta,weights=w)),brier_delta_ci=date_ci(valid,delta,w)))
        print(json.dumps(dict(arm=arm['arm'],calibration_audited=True)),flush=True)
    comparisons={'R00':[('A1','A0'),('B1','A0'),('B2','B1'),('ALG_LR','A0'),('ALG_ET','A0')],'R01':[('S1','S0'),('S2','S0'),('S3','S1')],'R02':[('P1','P0'),('E1','E0')],'R03':[('H1','H0'),('H2','H1'),('H3','H2')],'R04':[('F1','F0')],'R05':[('D1','D0')]}[a.round]
    for newer,reference in comparisons:
        f=load(dest,newer,'candidate_only');g=load(dest,reference,'candidate_only')
        if f.empty or g.empty:continue
        pair=f.merge(g[KEY+['score','y']],on=KEY,suffixes=('_new','_reference'),validate='one_to_one')
        assert np.array_equal(pair.y_new.fillna(-1),pair.y_reference.fillna(-1))
        pair['y']=pair.y_new;pair=pair[pair.y.notna()];w=weights(pair);delta=((pair.score_new-pair.y)**2-(pair.score_reference-pair.y)**2).to_numpy()
        audits.append(dict(comparison='input_on_identical_all_candidate_rows',newer=newer,reference=reference,rows=len(pair),newer_rows=len(f),reference_rows=len(g),
            stock_days=pair[['symbol','day']].drop_duplicates().shape[0],dates=pair.day.nunique(),brier_delta=float(np.average(delta,weights=w)),brier_delta_ci=date_ci(pair,delta,w)))
    write(dest/'calibration_audit.json',dict(at=now(),status='exposed_development_diagnostic',weighting='each stock-day equal within reported cohort; two-week moving date blocks',
        caveat='Brier combines discrimination and calibration; high bins and overestimation are also reported. Different issued cohorts do not prove a calibration increment.',audits=audits))
    write(dest/'results.json',results)
    lines=['# 概率与输入增量审计','','所有候选、第一名及有效信号分别报告；分钟重复不增加股票日权重。置信区间按两周日期块。高概率区间须满足样本与日期要求。结果仅为暴露历史开发诊断。','', '| 相同群体比较 | Brier变化（新减旧，负值更好） | 95%日期块区间 |','|---|---:|---|']
    for r in audits:
        if 'comparison' in r:lines.append(f"| {r.get('newer',r.get('arm'))} / {r.get('reference','候选校准')} | {r['brier_delta']:.5f} | {r['brier_delta_ci']} |")
    lines+=['','完整分箱、逐月高概率偏差、三个群体Brier及待确认数量见 calibration_audit.json。Brier较低不单独证明概率校准。', '', '方法定义参考 [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html)。']
    (dest/'CALIBRATION.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
