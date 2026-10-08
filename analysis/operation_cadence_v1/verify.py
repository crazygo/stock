"""Independent label/metric math and causal-boundary checks on actual captures."""
from pathlib import Path
import hashlib,json,math,subprocess
from collections import defaultdict
import numpy as np
import pandas as pd
from model import feature,predict,calendar,TARGETS

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert abs(a-b)<1e-10,(a,b)
def changes(prices):
    state=0;minimum=maximum=prices[0];n=0
    for value in prices[1:]:
        if state==0:
            minimum=min(minimum,value);maximum=max(maximum,value)
            if value/minimum>=1.05:state=1;maximum=value;n+=1
            elif value/maximum<=.95:state=-1;minimum=value;n+=1
        elif state==1:
            maximum=max(maximum,value)
            if value/maximum<=.95:state=-1;minimum=value;n+=1
        else:
            minimum=min(minimum,value)
            if value/minimum>=1.05:state=1;maximum=value;n+=1
    return n
def five(c):
    r=np.diff(np.log(c))
    def four_at(a):
        v=a[-60:];e=(v-v.mean())**2
        return np.array([np.mean(a[-120:])*252,np.std(v,ddof=1)*np.sqrt(252),np.corrcoef(v[:-1],v[1:])[0,1],np.sort(e)[-6:].sum()/e.sum()])
    a=four_at(r);seq=[]
    for length in range(len(r)-20,len(r)+1):
        v=four_at(r[:length]);seq.append([np.tanh(v[0]/.5),np.tanh((v[1]-.35)/.25),v[2],np.tanh((v[3]-.55)/.2)])
    return np.r_[a,np.exp(-10*np.abs(np.diff(seq,axis=0)).mean()/2)]

def main():
    r=json.loads((OUT/'results.json').read_text());e=json.loads((OUT/'evaluation.json').read_text());p=json.loads((OUT/'prices.json').read_text());m=json.loads((OUT/'models.json').read_text());s=calendar()
    assert '2025-01-09' not in s and '2026-07-03' not in s and '2026-10-07' in s
    for item in r['inputs']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    assert sha(OUT/'models.json')==r['model_sha256']
    assert len({x['code'] for x in r['records']})==len(r['records'])==55
    features_checked=0;labels_checked=0;probs_checked=0;price_by_code=p['records']
    for rec in r['records']:
        b=price_by_code[rec['code']]
        if not rec['in_domain']:assert not rec['p'];continue
        if rec['latest_features'] is None:assert not rec['p'];continue
        c=np.array([a[4] for a in b]);np.testing.assert_allclose(rec['latest_features'][:5],five(c[-253:]),atol=1e-10,rtol=1e-10)
        idx=len(b)-1;x,status=feature(b,idx,s);assert status=='scored'
        # Appending an arbitrary future bar cannot alter the historical observation.
        future=[s[s.index(b[-1][0])+1],999.,1001.,997.,999.,1.]
        xp,_=feature(b+[future],idx,s);np.testing.assert_equal(x,xp)
        # An internal missing session must be detected despite intact endpoints.
        missing=b[:-20]+b[-19:];assert feature(missing,len(missing)-1,s)[0] is None
        features_checked+=1
        if rec['year']:
            close(rec['year']['best20'],max(c[i]/c[i-20]-1 for i in range(len(c)-233,len(c))))
    by_code={c:{b[0]:i for i,b in enumerate(bs)} for c,bs in price_by_code.items()}
    for row in e['rows']:
        bs=price_by_code[row['code']];idx=by_code[row['code']][row['day']];si=s.index(row['day'])
        assert row['day']>='2026-06-01'
        x,status=feature(bs,idx,s);np.testing.assert_equal(x,row['x'])
        for key,config in TARGETS.items():
            ls=row['labels'][key];h=config['horizon'];maturity=s[si+h]
            assert ls['maturity']==maturity
            if maturity>r['as_of']:assert ls['status']=='pending' and ls['y'] is None
            if ls['status']=='mature':
                fb=bs[idx+1:idx+h+1];assert [b[0] for b in fb]==s[si+1:si+h+1]
                entry=fb[0][1];prices=np.r_[entry,[b[4] for b in fb]];net=prices[1:]/entry*.998-1
                draw=max(1-v/max(prices[:i+1]) for i,v in enumerate(prices))
                close(entry,ls['entry_open']);close(net[-1],ls['end_net']);close(draw,ls['max_drawdown'])
                if key=='cycle':actual=int(changes(prices)>=4)
                elif key=='quick':
                    actual=0
                    for v in net:
                        if v>=.05:actual=1;break
                        if v<=-.05:break
                elif key=='wave':actual=int(net[-1]>=.05 and draw<=.10)
                elif key=='hold':actual=int(net[-1]>=.10 and draw<=.15)
                else:actual=int(min(net)<=-.10)
                assert actual==ls['y'],(row['code'],row['day'],key);labels_checked+=1
            close(predict(m['models'][key],row['x'])[0],row['p'][key]);probs_checked+=1
    for key in TARGETS:
        rows=[x for x in e['rows'] if x['labels'][key]['status']=='mature'];daily=defaultdict(list)
        for x in rows:daily[x['day']].append(x)
        losses=[];baselines=[];globals_=[]
        for day,rs in daily.items():
            losses.append(np.mean([(x['p'][key]-x['labels'][key]['y'])**2 for x in rs]));baselines.append(np.mean([(x['baseline'][key]-x['labels'][key]['y'])**2 for x in rs]));globals_.append(np.mean([(x['global_baseline'][key]-x['labels'][key]['y'])**2 for x in rs]))
        metric=e['metrics'][key];assert metric['n']==len(rows) and metric['dates']==len(daily)
        close(np.mean(losses),metric['brier']);close(np.mean(baselines),metric['stock_baseline_brier']);close(np.mean(globals_),metric['global_baseline_brier'])
        close(1-np.mean(losses)/np.mean(baselines),metric['brier_skill'])
        assert sum(z['n'] for z in metric['reliability'])==len(rows),'Reliability bins partition all mature predictions'
        d=m['fit_details'][key];assert d['max_training_label_end']<'2026-02-02' and d['max_calibration_label_end']<'2026-06-01'
    # Retained old v2 inputs are checked against their preexisting published hashes.
    old=json.loads((ROOT/'analysis/ai_trend_quadrant_v2/results.json').read_text());old_checked=0
    for item in old['provenance']:assert sha(ROOT/item['path'])==item['sha256'];old_checked+=1
    hourly=json.loads((OUT/'intraday.json').read_text());alignments=0;discrepancies=[]
    for code,bars in hourly['records'].items():
        for b in bars:
            if b[0][11:16]=='09:30':assert b[7]=='盘前'
            if b[0][11:16]=='16:00':assert b[7]=='常规盘'
            if b[0][11:16]=='20:00':assert b[7]=='盘后'
            assert b[8]<b[9]
        for b in bars:
            if b[0][11:16]!='10:30' or b[6] not in by_code[code]:continue
            db=price_by_code[code][by_code[code][b[6]]];delta=abs(b[1]/db[1]-1)
            if delta>1e-4:discrepancies.append({'code':code,'day':b[6],'relative_difference':delta})
            alignments+=1
    assert not discrepancies,discrepancies[:5]
    probe=json.loads((OUT/'time_contract_probe.json').read_text())
    for code,rows in probe['records'].items():
        one=next(x for x in rows if x['time_key'].endswith('09:31:00'));db=price_by_code[code][-1];close(one['open'],db[1])
    # Parquet captures must remain ignored by Git.
    new=list((ROOT/'market_data/operation_cadence_v1').rglob('*.parquet'))
    result=subprocess.run(['git','check-ignore','--stdin'],input='\n'.join(str(x.relative_to(ROOT)) for x in new)+'\n',text=True,capture_output=True,cwd=ROOT,check=True)
    assert len(result.stdout.splitlines())==len(new)
    report={'status':'passed','current_feature_vectors':features_checked,'independent_future_labels':labels_checked,
        'portable_prediction_checks':probs_checked,'old_v2_input_hashes_unchanged':old_checked,'hourly_daily_open_alignments':alignments,
        'minute_open_alignment_stocks':len(probe['records']),'parquet_ignored':len(new),
        'checks':['independent G/V/M/J/S','append-future isolation','internal feature gap detection','next-open and full prefix','unknown preservation',
                  'purged train/calibration','date-equal Brier and both baselines','reliability partition','hourly completion-time/session contract'],
        'results_sha256':sha(OUT/'results.json'),'evaluation_sha256':sha(OUT/'evaluation.json')}
    (OUT/'model_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
