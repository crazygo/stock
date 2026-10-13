"""Freeze daily categorical forecasts and mixtures before unchanged capital replay."""
import argparse,shutil
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from class_learning import *
from run_transfer import write_models
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay


def events(at,row,estimates,feedback,old):
    cash={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None};out={}
    for name,info in estimates.items():
        e=choose(info)
        if feedback['expired_unresolved_prefix']:e={'ready':False,'action':'CASH','allocation':0.,'values':None}
        out[name]={'at':at,'date':row.date,'clock':row.clock,**e}
    for name,source in CONTROL_MAP.items():out[name]=old['events'][source]
    out['cash']=cash;out['double_cost']=dict(out['gap_blend'])
    out['premarket_only']=dict(out['gap_blend']) if row.clock==CLOCKS[0] else dict(cash)
    return out


def run(inputs,output):
    inputs=Path(inputs);output=Path(output);assert not output.exists();p=read(SOURCE/'protocol.json');pre=read(SOURCE/'evidence/preregistration.json')
    assert pre['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert pre['parent_delivery_sha256']==verify_parent()
    for f in ['input_manifest','input_audit']:assert pre[f+'_sha256']==sha(inputs/(f+'.json'))
    assert read(inputs/'input_audit.json')['passed'] and read(SOURCE/'evidence/model_preflight.json')['passed']
    assert pre['model_preflight_sha256']==sha(SOURCE/'evidence/model_preflight.json')
    manifest=read(inputs/'input_manifest.json');market=extended_market();ledger=load_market();original=read(PARENT/'input_manifest.json')
    assert market.manifest==original['expanded_market_manifest'] and ledger.manifest==original['ledger_market_manifest']
    for f in ['','models','daily','training']:(output/f).mkdir(parents=True,exist_ok=True)
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h;shutil.copyfile(inputs/f,output/f)
    for f in ['input_manifest.json','input_audit.json']:shutil.copyfile(inputs/f,output/f)
    dump(output/'registration.json',{'phase':'development','created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(SOURCE/'protocol.json'),'preregistration':pre,'parent':str(PARENT),'inputs':str(inputs)})
    panels={k:pd.read_parquet(output/(k+'_panel.parquet')) for k in PAIRS};panel=panels['semiconductor']
    reg=matched_registry(panels,market);own=registry(panel,market);hist={k:{} for k in MODES};predictions={k:[] for k in ROUTES};previous=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            if row.date!=previous:
                parents,feedback=mature_prefix(reg,at);daily={k:fit_mixture(panel,hist[k],own,at,k) for k in MODES};models={};members={}
                for clock in CLOCKS:models[clock],members[clock]=fit_clock(panels,parents,clock)
                filename=str(row.date.date());dump(output/'daily'/(filename+'.json'),{'parents':parents,'feedback':feedback,'mixture':daily,'members':members})
                write_models(output/'models'/(filename+'.json.gz'),models);previous=row.date
            raw={k:predict_distribution(models[row.clock].get(k),row) for k in MODES}
            estimates={k+'_'+s:issue_distribution(raw[k],daily[k],s) for k in MODES for s in ['blend','raw','prior','equal']}
            oldpath=PARENT/'training'/(tag(at)+'.json');old=read(oldpath);issued=events(at,row,estimates,feedback,old)
            dump(output/'training'/(tag(at)+'.json'),{'at':at,'date':row.date,'clock':row.clock,'cutoff':day_cutoff(at),
                'distributions':raw,'estimates':estimates,'events':issued,'parent_source_sha256':sha(oldpath)})
            for k in MODES:hist[k][at]=raw[k]
            for k,e in issued.items():predictions[k].append(e)
            if (i+1)%400==0:print('Class forecasts',i+1,'/',len(panel),at,flush=True)
    dump(output/'predictions.json',predictions);settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    actions={k:[e for e in es if start<=str(pd.Timestamp(e['date']).date())<=end] for k,es in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_capital_outcomes_seen':False,
        'prior_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':actions})
    features=pd.read_parquet(output/'features.parquet');test=features[features.date.between(start,end)]
    base=labels(output/'baseline_labels.json');stress=labels(output/'stress_labels.json');books={};result={'phase':'development','primary_name':p['primary'],'routes':{}}
    for name,es in actions.items():
        multiple=2 if name=='double_cost' else 1
        book=replay(ledger,test,list((stress if multiple==2 else base).values()),{pd.Timestamp(e['at']):e['action'] for e in es},start,end,
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
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.inputs,a.output)
