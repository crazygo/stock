"""Read-only independent arithmetic for diagnostic denominators and sharing."""
import argparse
from collections import defaultdict
from shared_learning import *
from verify_continuation import near


def csv(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def audit(run):
    run=Path(run);d=read(run/'diagnostics.json');r=read(run/'results.json');assert d['results_sha256']==sha(run/'results.json') and d['post_result_only'] and not d['rules_changed']
    v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    settings=read(run/'registration.json')['preregistration']['protocol'][r['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));counts=defaultdict(lambda:[0,0,0]);unknown=[];change={k:0 for k in ['independent','shared','v56']};fb=[];checks=0
    for f in sorted((run/'training').glob('*.json')):
        rec=read(f);date=rec['date'][:10]
        if not settings['evaluation_start']<=date<=settings['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);row=panel.loc[at];known=pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D)
        if not known:unknown.append(str(at))
        raws={**rec['raw_details'],**{k:rec['baselines'][k]['fit']['raw'] for k in ['v56','v55']}}
        for k in change:change[k]+=int(bool(raws['partial'] and raws[k]) and any(abs(raws['partial'][s]-raws[k][s])>1e-12 for s in core.ETFS))
        base=read(run/'base_daily'/(date+'.json'));meta=base['prepared'];c=CLOCKS.index(rec['clock'])
        fb.append({'at':str(at),'date':str(row.date),'clock':rec['clock'],'joint_parent_dates':meta['N'],'joint_clock_rows':meta['clock_rows'][c],'old_v56_clock_rows':len(rec['baselines']['v56']['fit']['fit']['members']),'expired_unresolved_prefix':base['feedback']['expired_unresolved_prefix']})
        for k,raw in raws.items():
            if not raw:continue
            counts[k][0]+=1;out=values[k][date]
            if not known:continue
            counts[k][1]+=1;cerror=(raw['C']-float(row.C))**2;derror=(raw['D']-float(row.D))**2
            out['raw_C_mse'].append(cerror);out['raw_D_mse'].append(derror);out['raw_pair_mse'].append(cerror+derror)
            out['magnitude'].append(abs(float(row.D)));out['brier_numerator'].append(abs(float(row.D))*(raw['q']-(1. if row.D>0 else 0.))**2)
            estimate=rec['estimates'][k] if k in VARIANTS else rec['baselines'][k]['estimate']
            if estimate['ready']:
                a,b=estimate['point'];out['calibrated_pair_mse'].append(((a-float(row.C+row.D))**2+(b-float(row.C-row.D))**2)/2);counts[k][2]+=1
    for k,days in values.items():
        saved=d['metrics'][k];assert [saved['queries'],saved['mature_raw_queries'],saved['mature_calibrated_queries']]==counts[k] and saved['dates']==len(days)
        estimates={}
        for field in ['raw_C_mse','raw_D_mse','raw_pair_mse','magnitude','brier_numerator','calibrated_pair_mse']:
            perday=[sum(v[field])/len(v[field]) for v in days.values() if v[field]];estimates[field]=sum(perday)/len(perday);near(saved['date_equal'][field],estimates[field]);checks+=1
        near(saved['date_equal']['magnitude_weighted_brier'],estimates['brier_numerator']/estimates['magnitude']);checks+=1
    assert list(pd.to_datetime(d['unknown_actual14_queries_retained'],utc=True))==list(pd.to_datetime(unknown,utc=True))
    assert d['feedback_summary']['changed_raw_queries_partial_vs']==change and d['feedback_rows']==fb
    weights=pd.DataFrame(fb).groupby('date').mean(numeric_only=True)
    for field,value in weights.mean().items():near(d['feedback_summary']['date_equal_mean'][field],value);checks+=1
    # Pairwise weighted differences equal weighted deviations from the mean.
    dispersion=defaultdict(list)
    for f in sorted((run/'base_daily').glob('*.json')):
        date=f.stem
        if not settings['evaluation_start']<=date<=settings['evaluation_end']:continue
        base=read(f);meta=base['prepared']
        if not meta['ready']:continue
        fields=meta['columns'];names=list(dict.fromkeys(v for row in fields for v in row))
        for k in VARIANTS:
            offset=0;ids=[]
            for row in fields:
                ids.append([names.index(x) for x in row] if k=='shared' else list(range(offset,offset+len(row))));offset+=len(row)
            for head in ['C','M','q']:
                coef=base['models'][k][head]['coefficients'];total=0.
                for name in names:
                    active=[c for c in range(len(fields)) if name in fields[c]];mass=sum(meta['eta'][c] for c in active)
                    for i,c in enumerate(active):
                        for other in active[i+1:]:
                            a=coef[ids[c][fields[c].index(name)]];b=coef[ids[other][fields[other].index(name)]]
                            total+=meta['eta'][c]*meta['eta'][other]*(a-b)**2/mass
                dispersion[k,head].append(total)
    for (k,head),items in dispersion.items():near(d['mean_slope_dispersion'][k][head],sum(items)/len(items));checks+=1
    for name,es in read(run/'decisions_frozen.json')['routes'].items():
        positive=[e for e in es if e['action']!='CASH'];trades=csv(run/(name+'_trades.csv'));row=d['routes'][name]
        assert row['proposals']==len(positive) and row['proposal_dates']==len(set(e['date'] for e in positive)) and row['actual_trades']==len(trades);checks+=1
    primary=csv(run/'partial_band_trades.csv');entry=lambda t:{(x.decision_at,x.symbol) for x in t.itertuples()};a=entry(primary)
    for name,row in d['increments'].items():
        control=csv(run/(name+'_trades.csv'));b=entry(control)
        assert {(x['at'],x['symbol']) for x in row['added_actual_buys']}==a-b
        assert {(x['at'],x['symbol']) for x in row['removed_actual_buys']}==b-a and row['common_actual_buys']==len(a&b)
        near(row['primary_minus_control_monthly_mean'],r['primary']['primary_average']-r['routes'][name]['primary_average']);assert row['paired_capital_interval']==r['paired_increments'][name];checks+=1
    receipt={'passed':True,'diagnostic_equations':checks,'unknown_queries_retained':len(unknown),'metrics_denominators':dict(counts),'new_rules':False,'source_sha256':sha(Path(__file__)),'diagnostics_sha256':sha(run/'diagnostics.json'),'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics_audit.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
