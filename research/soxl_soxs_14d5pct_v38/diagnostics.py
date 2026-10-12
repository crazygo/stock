"""Post-result qualification/support decomposition without changing policy."""
import argparse
from collections import Counter
from eligible_risk import core,np,pd,Path,read,dump,sha,tag


def diagnose(run):
    run=Path(run);p=pd.read_parquet(run/'selected_panel.parquet');pred=read(run/'predictions.json');dates=[];queries=[]
    for f in sorted((run/'daily').glob('*.json')):
        d=read(f)
        if not ('2026-01-01'<=d['date'][:10]<='2026-04-30'):continue
        dates.append({'date':d['date'][:10],'ready':d['ready'],'all_dates':len(d['parents']),'eligible_dates':len(d['subsets']['eligible']),
            'eligible_rows':sum(len(x['members']) for x in d['subsets']['eligible']),
            'empty_bootstrap_models':sum(m is None for b in d['bootstrap'] for m in b.values())})
    for at,row in p[p.date.between('2026-01-01','2026-04-30')].iterrows():
        r=read(run/'training'/(tag(at)+'.json'))
        if r['events']['source_v35']['action']=='CASH':continue
        q={'at':at,'date':row.date,'clock':row.clock,'actual_netlog':row.selected_y,'known':bool(np.isfinite(row.selected_y))}
        for k,v in r['estimates'].items():q[k]={f:v.get(f) for f in ['source_fallback','quality','lower_quality','outside_columns','bootstrap_complete']}
        q['actions']={k:{f:e.get(f) for f in ['action','reason','quality_used']} for k,e in r['events'].items()};queries.append(q)
    old=pd.read_csv(run/'source_v35_trades.csv');key=lambda f:set(zip(f.decision_at,f.symbol));okey=key(old);changes={}
    for n in pred:
        try:now=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:now=pd.DataFrame(columns=old.columns)
        added=key(now)-okey;removed=okey-key(now)
        fields=['decision_at','symbol','reason','net_return','capital']
        changes[n]={'added':now[[x in added for x in zip(now.decision_at,now.symbol)]][fields].to_dict('records') if len(now) else [],
            'removed':old[[x in removed for x in zip(old.decision_at,old.symbol)]][fields].to_dict('records')}
    counts={n:dict(Counter(e['action'] for e in es if '2026-01-01'<=e['date'][:10]<='2026-04-30')) for n,es in pred.items()}
    reasons={n:dict(Counter(q['actions'][n]['reason'] for q in queries)) for n in ['eligible_supported','eligible_gate','all_supported','all_gate','eligible_scores','eligible_band']}
    # Exact primary/source replication checks the no-increment conclusion directly.
    for suffix in ['equity','trades','decisions','orders']:
        pd.testing.assert_frame_equal(pd.read_csv(run/('eligible_supported_'+suffix+'.csv')),pd.read_csv(run/('source_v35_'+suffix+'.csv')))
    out={'post_result_only':True,'rules_changed':False,'first_meta_ready_date':next(d['date'] for d in dates if d['ready']),
        'ready_dates':sum(d['ready'] for d in dates),'daily_membership':dates,'source_eligible_queries':queries,'candidate_counts':counts,
        'eligible_reason_counts':reasons,'actual_trade_changes':changes,'primary_exactly_matches_source_v35':True,
        'diagnostic_all_supported_meets_development_numbers':read(run/'results.json')['routes']['all_supported']['primary_pass'],
        'diagnostic_is_not_registered_primary':True,'no_case_replay_authorized_for_v38':True,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',out);print({k:v for k,v in out.items() if k not in ['source_eligible_queries','actual_trade_changes','daily_membership']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
