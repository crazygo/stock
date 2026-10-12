"""Freeze clock-specific calibrated actions before a single sequential ledger."""
import argparse,gzip,pickle
from datetime import datetime,timezone
from pathlib import Path
from threadpoolctl import threadpool_limits
from censored_learning import *
from clock_reliability import issue as old_issue
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay
SOURCE=Path(__file__).resolve().parent


def authorize_cases(path,p):
    run=Path(path);reg=read(run/'registration.json');r=read(run/'results.json');v=read(run/'verification.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    gate=development_gate(r['primary'],r['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Development primary failed; new case replay is forbidden')
    return {'run':str(run.resolve()),'results_sha256':sha(run/'results.json'),'gate':gate}


def run(output,case_after=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json')
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(core.ROOT/f)==h
    auth=authorize_cases(case_after,p) if case_after else None;phase='cases' if auth else 'development'
    output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve every attempt')
    for sub in ['','training','models','executed_sources']:(output/sub).mkdir(parents=True,exist_ok=True)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,
        'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':auth,'source_sha256':prereg['code_sha256']})
    for f in prereg['code_sha256']:
        src=core.ROOT/f;dest=output/'executed_sources'/src.parent.name/src.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(src.read_bytes())
    market,features,panel,base,stress,manifest=prepare_inputs(p,phase,output);dump(output/'input_manifest.json',manifest)
    histories={c:{} for c in CLOCKS};predictions={n:[] for n in ROUTES}
    tapes={(at,s):tape(market,s,at) for at in features.index for s in core.ETFS}
    oldroot=core.ROOT/p['control_source'];delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v27/evidence/delivery_manifest.json')
    assert sha(oldroot/'predictions.json')==delivery['cache_outputs_sha256']['predictions.json']
    old={pd.Timestamp(e['at']):e for e in read(oldroot/'predictions.json')['score_sequential']}
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            raw,origin,model=base_forecast(panel,at,manifest);c=row.clock
            record={'at':at,'date':row.date,'clock':c,'cutoff':day_cutoff(at),'raw':raw,'origin':origin}
            estimates={k:estimate(histories[c],tapes,at,raw,terminal_only=(k=='terminal')) for k in ['censored','terminal']}
            record['estimates']=estimates
            reference=old[at] if at in old else old_issue(panel,histories[c],at,raw)['band']
            issued=events(at,row.date,c,estimates,reference);record['events']=issued
            for n,e in issued.items():predictions[n].append(e)
            dump(output/'training'/(tag(at)+'.json'),record);histories[c][at]=raw
            if model is not None:
                with gzip.open(output/'models'/(tag(at)+'.pkl.gz'),'wb') as stream:pickle.dump(model,stream)
            if (i+1)%200==0:print('trained through',at,flush=True)
    dump(output/'predictions.json',predictions);settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end']
    frozen={n:[e for e in events if start<=str(e['date'].date())<=end] for n,events in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_v28_capital_outcomes_seen':False,
        'prior_versions_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':frozen})
    test=features[features.date.between(start,end)];months=settings.get('months',settings.get('regression_months'));books={}
    result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for n,h in frozen.items():
        multiple=2 if n=='double_cost' else 1
        book=replay(market,test,list((stress if multiple==2 else base).values()),{e['at']:e['action'] for e in h},start,end,
            cost_multiple=multiple,allocations={e['at']:e['allocation'] for e in h},require_full_cash=False)
        books[n]=book;save_replay(output,n,book);result['routes'][n]=score(book,months)
    expected=[]
    for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)]:
        for c in CLOCKS:
            t=pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC')
            if t<pd.Timestamp(market.calendar[d]['close_at']):expected.append(t)
    covered=list(test.index)==expected
    result['coverage']={'complete':covered,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique(),
        'same_day_ticks_are_not_independent_dates':True};result['primary']=result['routes'][p['primary']]
    result['paired_increments']={n:paired_uncertainty(books[p['primary']][0],books[n][0]) for n in ROUTES if n not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],covered,p);result['development_gate']=gate
        result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['case_acceptance']={'passed':bool(covered and result['primary']['primary_pass'])}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    dump(output/'results.json',result);print({n:{'mean':s['primary_average'],'total':s['continuous_return']} for n,s in result['routes'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case-after',type=Path)
    a=p.parse_args();run(a.output,a.case_after)
