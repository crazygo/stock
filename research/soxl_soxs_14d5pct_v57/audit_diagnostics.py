"""Post-result independent aggregation; fixes only the interpretation of age units."""
import argparse
from collections import defaultdict
from datetime import date
from short_learning import *
from verify_continuation import near


def audit(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');d=read(run/'diagnostics.json')
    assert v['passed'] and v['results_sha256']==d['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'][r['phase']]
    original=pd.read_parquet(run/'activity_panel.parquet');short=pd.read_parquet(run/'short_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    values=defaultdict(lambda:defaultdict(list));feedback=defaultdict(list);counts=defaultdict(int);ages=defaultdict(list)
    for f in sorted((run/'training').glob('*.json')):
        rec=read(f);day=rec['date'][:10]
        if not p['evaluation_start']<=day<=p['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);a=original.loc[at];b=short.loc[at]
        known14=pd.notna(a.terminal_release) and a.terminal_release<cutoff;known_short=pd.notna(b.terminal_release) and b.terminal_release<cutoff
        mf=rec['fits']['fast']['fit']['members'];ml=rec['fits']['lagged']['fit']['members'];feedback[day].append(len(mf)-len(ml))
        for k,m in [('fast',mf),('lagged',ml)]:
            if m:ages[k].append((date.fromisoformat(day)-pd.Timestamp(m[-1]).tz_convert(core.ET).date()).days)
        for k in VARIANTS+['v56','v55']:
            raw=rec['fits'][k]['raw'] if k in VARIANTS else rec['baselines'][k]['fit']['raw']
            est=rec['estimates'][k] if k in VARIANTS else rec['baselines'][k]['estimate']
            if not raw:continue
            if k in VARIANTS and known_short:
                truth=[b.C+b.D,b.C-b.D];pred=[raw[s] for s in core.ETFS]
                for name,x in [('short_raw_pair_mse',sum((pred[j]-truth[j])**2 for j in range(2))/2),('short_direction_mse',(raw['D']-b.D)**2),('short_magnitude',abs(b.D)),('short_brier_numerator',abs(b.D)*(raw['q']-int(b.D>0))**2)]:values[k,name][day].append(float(x))
                counts[k,'short']+=1
            if known14 and est['ready']:
                truth=[a.C+a.D,a.C-a.D];x=sum((est['point'][j]-truth[j])**2 for j in range(2))/2
                values[k,'calibrated_actual14_pair_mse'][day].append(float(x));counts[k,'actual14']+=1
    checked=0;support={}
    for (k,name),days in values.items():
        value=sum(sum(xs)/len(xs) for xs in days.values())/len(days)
        near(value,d['metrics'][k]['date_equal'][name]);checked+=1
        support[k,name]={'queries':sum(len(xs) for xs in days.values()),'dates':len(days),'value':value}
    for k in VARIANTS:
        expected=support[k,'short_brier_numerator']['value']/support[k,'short_magnitude']['value']
        near(expected,d['metrics'][k]['date_equal']['short_magnitude_weighted_brier']);checked+=1
    extra=sum(sum(xs)/len(xs) for xs in feedback.values())/len(feedback);near(extra,d['feedback_summary']['extra_recent_members_date_equal_mean']);checked+=1
    out={'passed':True,'post_result_only':True,'rules_changed':False,'metric_equations':checked,'support':{k+':'+name:v for (k,name),v in support.items()},
         'fast_latest_source_age_ET_dates':sum(ages['fast'])/len(ages['fast']),
         'lagged_latest_source_age_ET_dates':sum(ages['lagged'])/len(ages['lagged']),
         'age_unit_clarification':'Frozen diagnostics fields containing age_calendar_days are elapsed timestamp seconds divided by86400. This audit separately reports ET calendar-date differences; DST can differ by an hour. Neither quantity is a trading rule.',
         'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics_audit.json',out);print({k:v for k,v in out.items() if k!='support'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
