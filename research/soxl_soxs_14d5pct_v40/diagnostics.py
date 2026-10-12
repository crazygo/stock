"""Post-result direction/regret diagnostics with date-equal scoring and unknowns."""
import argparse,gzip,pickle
from collections import Counter
from nonlinear_direction import *

def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];start,end=settings['evaluation_start'],settings['evaluation_end']
    root=Path(read(run/'input_manifest.json')['source']);panel=pd.read_parquet(root/'financial_panel.parquet');part=panel[panel.date.between(start,end)]
    cutoff=pd.Timestamp(read(root/'input_manifest.json')['observation_cutoff_exclusive']);rows={k:[] for k in VARIANTS};unknown=0;split_counts={k:Counter() for k in ['tree','stump']}
    for at,row in part.iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));known=bool(pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D))
        unknown+=int(not known)
        for k in VARIANTS:
            raw=rec['raws'][k];est=rec['estimates'][k]
            if raw is not None and known and est['ready']:
                truth=np.array([row.C+row.D,row.C-row.D]);entry={'at':at,'date':row.date,'magnitude':abs(row.D),'direction_label':float(row.D>0),'q':rec['fits'][k]['raw']['q']}
                for field,pred in [('raw',raw),('calibrated',est['point'])]:
                    pred=np.array(pred);j=int(pred[1]>pred[0]);entry[field+'_mse']=float(np.mean((pred-truth)**2));entry[field+'_chosen_logreturn']=float(truth[j]);entry[field+'_regret']=float(np.max(truth)-truth[j]);entry[field+'_positive']=float(truth[j]>0)
                entry['weighted_direction_brier_numerator']=entry['magnitude']*(entry['q']-entry['direction_label'])**2;rows[k].append(entry)
            if k in split_counts:
                f=run/'models'/(k+'_'+tag(at)+'.pkl.gz')
                if f.exists():
                    model=pickle.load(gzip.open(f,'rb'))
                    if 'q' in model['heads']:
                        for step in model['heads']['q']._predictors:
                            for node in step[0].nodes:
                                if not node['is_leaf']:split_counts[k][model['columns'][int(node['feature_idx'])]]+=1
    stats={}
    for k,records in rows.items():
        frame=pd.DataFrame(records);daily=frame.drop(columns='at').groupby('date').mean(numeric_only=True)
        means=daily.mean().to_dict();means['weighted_direction_brier']=means['weighted_direction_brier_numerator']/means['magnitude'] if means['magnitude']>0 else None
        stats[k]={'scored_queries':len(frame),'scored_dates':len(daily),'date_equal_metrics':means}
    predictions=read(run/'decisions_frozen.json')['routes'];counts={k:dict(Counter(e['action'] for e in es)) for k,es in predictions.items()}
    receipt={'post_result_only':True,'rules_changed':False,'phase':r['phase'],'paired_unknown_queries_retained':unknown,'evaluation_queries':len(part),
        'paired_maturity_cutoff':cutoff,'metrics':stats,'candidate_counts':counts,'split_field_counts':{k:dict(v.most_common()) for k,v in split_counts.items()},
        'split_counts_are_not_importance_or_independent_evidence':True,'overlapping_hypothetical_payoffs_are_not_capital_returns':True,
        'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',receipt);print({k:v for k,v in receipt.items() if k!='split_field_counts'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
