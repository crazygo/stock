"""Generate causal daily state probability models, freeze every action, then replay capital."""
import argparse,shutil
from datetime import datetime,timezone
from state_calibration import *
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay


def run(output):
    output=Path(output);assert not output.exists();p=read(SOURCE/'protocol.json');pre=read(SOURCE/'evidence/preregistration.json')
    assert pre['protocol_sha256']==sha(SOURCE/'protocol.json') and pre['parent_delivery_sha256']==verify_parent()
    assert pre['calendar_sha256']==sha(CATALOG) and pre['calendar_audit_sha256']==sha(SOURCE/'evidence/calendar_verification.json')
    assert pre['price_manifest_sha256']==sha(SOURCE/'evidence/price_inputs.json')
    assert pre['reaction_audit_sha256']==sha(SOURCE/'evidence/reaction_verification.json')
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert read(SOURCE/'evidence/model_preflight.json')['passed'] and pre['model_preflight_sha256']==sha(SOURCE/'evidence/model_preflight.json')
    market=extended_market();ledger=load_market();panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet');reg=registry(panel,market);panel=prepare_panel(panel,market)
    for f in ['','daily','training']:(output/f).mkdir(parents=True,exist_ok=True)
    for name in ['semiconductor_panel.parquet','features.parquet','baseline_labels.json','stress_labels.json']:
        assert pre['input_files_sha256'][name]==sha(PARENT/name);shutil.copyfile(PARENT/name,output/name)
    prepare_panel(pd.read_parquet(PARENT/'semiconductor_panel.parquet'),market,output/'event_features.csv')
    dump(output/'registration.json',{'phase':'development','created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(SOURCE/'protocol.json'),'preregistration':pre,'parent':str(PARENT)})
    history={};predictions={k:[] for k in ROUTES};previous=None
    for i,(at,row) in enumerate(panel.iterrows()):
        oldpath=PARENT/'training'/(tag(at)+'.json');old=read(oldpath)
        if row.date!=previous:
            parents,fb=mature_prefix(reg,at);daily=daily_models(panel,parents,fb,history,at)
            dump(output/'daily'/(str(row.date.date())+'.json'),daily);previous=row.date
        basepath=BASE/'training'/(tag(at)+'.json');waitpath=WAIT/'training'/(tag(at)+'.json')
        base=read(basepath);wait=read(waitpath);assert old['base_source_sha256']==sha(basepath)
        issued=issue(at,row,daily,old,base,wait)
        dump(output/'training'/(tag(at)+'.json'),{'at':at,'cutoff':day_cutoff(at),'events':issued,'parent_source_sha256':sha(oldpath),'base_source_sha256':sha(basepath),'wait_source_sha256':sha(waitpath)})
        for k,e in issued.items():predictions[k].append(e)
        history[at]=old
        if (i+1)%800==0:print('State probability forecasts',i+1,'/',len(panel),at,flush=True)
    dump(output/'predictions.json',predictions);settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    actions={k:[e for e in es if start<=str(pd.Timestamp(e['date']).date())<=end] for k,es in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_capital_outcomes_seen':False,
        'prior_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':actions})
    features=pd.read_parquet(output/'features.parquet');test=features[features.date.between(start,end)]
    baseline=labels(output/'baseline_labels.json');stress=labels(output/'stress_labels.json');books={};result={'phase':'development','primary_name':p['primary'],'routes':{}}
    for name,es in actions.items():
        multiple=2 if name=='double_cost' else 1
        book=replay(ledger,test,list((stress if multiple==2 else baseline).values()),{pd.Timestamp(e['at']):e['action'] for e in es},start,end,
            cost_multiple=multiple,allocations={pd.Timestamp(e['at']):e['allocation'] for e in es},require_full_cash=False)
        books[name]=book;save_replay(output,name,book);result['routes'][name]=score(book,months)
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in ledger.sessions[(ledger.sessions>=start)&(ledger.sessions<=end)]
        for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(ledger.calendar[d]['close_at'])]
    complete=list(test.index)==expected
    result['coverage']={'complete':complete,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique(),'same_day_ticks_are_not_independent_dates':True}
    result['primary']=result['routes'][p['primary']];result['development_gate']=development_gate(result['primary'],complete,p)
    result['case_regression']={'executed':False,'eligible_after_verification':result['development_gate']['passed']}
    result['paired_increments']={k:paired_uncertainty(books[p['primary']][0],books[k][0]) for k in ROUTES if k not in [p['primary'],'cash','double_cost']}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False);dump(output/'results.json',result)
    print({k:{'mean':s['primary_average'],'months':[m['return'] for m in s['monthly']]} for k,s in result['routes'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
