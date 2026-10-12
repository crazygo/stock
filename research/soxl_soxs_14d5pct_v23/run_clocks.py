"""Freeze every clock's forecast before replaying any new capital route."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import pickle
import pandas as pd
from threadpoolctl import threadpool_limits
from clock_training import *
from verify_continuation import read
from allocation_ledger import replay
from study import save_replay
from run_intraday import score
from run_breadth import paired_uncertainty
SOURCE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tag(at): return pd.Timestamp(at).tz_convert(core.ET).strftime('%Y-%m-%d_%H%M')


def authorize_cases(path,p):
    run=Path(path); reg=read(run/'registration.json'); result=read(run/'results.json'); v=read(run/'verification.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    gate=development_gate(result['primary'],result['coverage']['complete'],p)
    if not gate['passed']: raise ValueError('Development primary failed; new case replay is forbidden')
    return {'run':str(run.resolve()),'results_sha256':sha(run/'results.json'),'gate':gate}


def run(output,case_after=None):
    p=read(SOURCE/'protocol.json'); prereg=read(SOURCE/'evidence/preregistration.json')
    assert sha(SOURCE/'protocol.json')==prereg['protocol_sha256']
    for f,h in prereg['code_sha256'].items(): assert sha(core.ROOT/f)==h
    authorization=authorize_cases(case_after,p) if case_after else None
    phase='cases' if case_after else 'development'; settings=p[phase]
    start,end=settings['evaluation_start'],settings['evaluation_end']; cutoff=pd.Timestamp(end,tz=core.ET)+pd.Timedelta(days=1)
    output=Path(output)
    if output.exists(): raise ValueError('Use a new output directory; preserve every attempt')
    for sub in ['','training','models','executed_sources']: (output/sub).mkdir(parents=True,exist_ok=True)
    core.dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),
        'preregistration':prereg,'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':authorization,'source_sha256':prereg['code_sha256']})
    for f in prereg['code_sha256']:
        path=core.ROOT/f; dest=output/'executed_sources'/path.parent.name/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes())
    source=core.ROOT/p['feature_source']; manifest=read(source/'input_manifest.json')
    assert sha(source/'features.parquet')==manifest['feature_sha256']
    assert read(source/'verification.json')['passed']
    original=core.load_market(); assert original.manifest==manifest['market']; market=restrict_market(original,cutoff)
    leaders=load_panel(market.calendar,PANEL_ROOT); assert leaders[2]==manifest['panel']
    features=pd.read_parquet(source/'features.parquet'); features=features[features.index<cutoff].copy()
    features.to_parquet(output/'features.parquet',compression='zstd',compression_level=7)
    core.dump(output/'input_manifest.json',{'source_run':str(source.resolve()),'source_files_sha256':
        {n:sha(source/n) for n in ['features.parquet','input_manifest.json','verification.json','input_verification.json','numerical_audit.json']},
        'full_market':original.manifest,'panel':leaders[2],'observation_cutoff_exclusive':cutoff})
    base=baseline_labels(market,features); stress=baseline_labels(market,features,2)
    core.dump(output/'baseline_labels.json',list(base.values()));core.dump(output/'stress_labels.json',list(stress.values()))
    panels={}; receipts=[]; volume_audits=[]
    for clock in CLOCKS:
        f=features[features.clock==clock]
        volumes,audits=clock_volumes(leaders[0],market.sessions,market.calendar,cutoff,clock)
        code=clock[:5].replace(':',''); volumes.to_parquet(output/('volumes_'+code+'.parquet'))
        panel=prepare(f,base,joint_releases(f,base),factors(f,volumes))
        panels[clock],r=terminal_panel(market,panel,base);receipts.extend(r);volume_audits.extend(audits)
    panel=pd.concat(panels.values()).sort_index(); panel.to_parquet(output/'panel.parquet',compression='zstd',compression_level=7)
    core.dump(output/'terminal_receipts.json',receipts);core.dump(output/'volume_audit.json',volume_audits)
    print('prepared',len(features),'ticks',features.date.nunique(),'dates',flush=True)
    histories={(c,v):{'early':[],'full':[]} for c in CLOCKS for v in ['base','context']}
    anchors={(c,v):[] for c in CLOCKS for v in ['base','context']}; predictions={n:[] for n in ROUTES}
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            clock=row.clock; panel=panels[clock]; dc=day_cutoff(at)
            record={'at':at,'date':row.date,'clock':clock,'cutoff':dc,'variants':{}}; models={}; capital={}; fs={}
            for variant in ['base','context']:
                h=histories[clock,variant]; model,r=forecasts(panel,at,clock,variant,h)
                full=r['forecasts']['full']['event']; anchors[clock,variant].append({'at':at,'date':row.date,**full})
                riskbank=bank(panel,anchors[clock,variant]) if any(e['ready'] for e in anchors[clock,variant]) else pd.DataFrame(index=pd.MultiIndex.from_tuples([],names=['at','symbol']))
                cap={}
                for mode in ['early','full','raw']:
                    forecast=r['forecasts'][mode]['event']; choice=capital_decision(riskbank,dc,at,forecast)
                    event={'at':at,'date':row.date,**{k:v for k,v in choice.items() if k!='fit'},'adjusted':forecast['adjusted']}
                    cap[mode]={'risk':choice['fit'],'event':event}
                for mode in ['early','full']:
                    h[mode].append({'at':at,'date':row.date,'raw':r['forecasts'][mode]['event']['raw']})
                r['capital']=cap;record['variants'][variant]=r;models[variant]=model;capital[variant]=cap;fs[variant]=r['forecasts']
            for n,e in route_events(at,row.date,clock,capital,fs).items(): predictions[n].append(e)
            core.dump(output/'training'/(tag(at)+'.json'),record)
            with gzip.open(output/'models'/(tag(at)+'.pkl.gz'),'wb') as stream:pickle.dump(models,stream)
            if (i+1)%200==0:print('trained through',at,flush=True)
    for (clock,variant),events in anchors.items():
        bank(panels[clock],events).to_parquet(output/('bank_'+clock[:5].replace(':','')+'_'+variant+'.parquet'))
    core.dump(output/'anchors.json',{c:{v:anchors[c,v] for v in ['base','context']} for c in CLOCKS})
    core.dump(output/'predictions.json',predictions)
    frozen={n:[e for e in h if start<=str(e['date'].date())<=end] for n,h in predictions.items()}
    core.dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'phase':phase,
        'new_v23_capital_outcomes_seen':False,'older_history_exposed':True,'predictions_sha256':sha(output/'predictions.json'),'routes':frozen})
    test=features[(features.date>=start)&(features.date<=end)]; months=settings.get('months',settings.get('regression_months'))
    books={}; result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for n,h in frozen.items():
        m=2 if n=='double_cost' else 1
        book=replay(market,test,list((stress if m==2 else base).values()),{e['at']:e['action'] for e in h},start,end,
                    cost_multiple=m,allocations={e['at']:e['allocation'] for e in h})
        books[n]=book;save_replay(output,n,book);result['routes'][n]=score(book,months)
    required=[]
    for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)]:
        for c in CLOCKS:
            t=pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC')
            if t<pd.Timestamp(market.calendar[d]['close_at']):required.append(t)
    complete=list(test.index)==required
    result['coverage']={'complete':complete,'expected_ticks':len(required),'observed_ticks':len(test),'dates':test.date.nunique(),
                        'same_day_ticks_are_not_independent_dates':True}
    result['primary']=result['routes'][p['primary']]
    result['paired_increments']={n:paired_uncertainty(books[p['primary']][0],books[n][0]) for n in ROUTES if n not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],complete,p);result['development_gate']=gate
        result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['case_acceptance']={'passed':bool(complete and result['primary']['primary_pass'])}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    core.dump(output/'results.json',result)
    print({n:{'mean':s['primary_average'],'total':s['continuous_return']} for n,s in result['routes'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-after',type=Path)
    a=parser.parse_args();run(a.output,a.case_after)
