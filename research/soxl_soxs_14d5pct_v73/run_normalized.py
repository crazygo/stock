"""Immutable daily size-normalized transfer forecasts, then unchanged capital replay."""
import argparse
import shutil
from datetime import datetime, timezone
from threadpoolctl import threadpool_limits
from normalized_learning import *
from run_transfer import write_models
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay


def events(at,row,estimates,daily,feedback,old):
    result={}
    for kind in VARIANTS:
        for suffix in ['band','point']:
            e=decide(estimates[kind],daily[kind]['feedback'],suffix=='band')
            if feedback['expired_unresolved_prefix']:e={'ready':False,'action':'CASH','allocation':0.,'values':None}
            result['normalized_'+kind+'_'+suffix]={'at':at,'date':row.date,'clock':row.clock,**e}
    result['old_quadratic_band']=old['events']['quadratic_partial_band'];result['old_quadratic_point']=old['events']['quadratic_partial_point']
    result['linear_band']=old['events']['linear_band'];result['linear_point']=old['events']['linear_point']
    result['premarket_only']=dict(result['normalized_partial_band']) if row.clock==CLOCKS[0] else {'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None}
    result['cash']={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None}
    result['double_cost']=dict(result['normalized_partial_band'])
    return result


def run(output):
    output=Path(output);assert not output.exists();p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json')
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert prereg['parent_delivery_sha256']==verify_parent()
    assert read(SOURCE/'evidence/model_preflight.json')['passed']
    assert prereg['model_preflight_sha256']==sha(SOURCE/'evidence/model_preflight.json')
    manifest=read(PARENT/'input_manifest.json');market=load_market();assert market.manifest==manifest['market_manifest']
    for name in ['','daily','training','models']:(output/name).mkdir(parents=True,exist_ok=True)
    for f in [*manifest['derived_files_sha256'],'input_manifest.json','input_audit.json']:shutil.copyfile(PARENT/f,output/f)
    dump(output/'registration.json',{'phase':'development','created_at':datetime.now(timezone.utc).isoformat(),
         'protocol_sha256':sha(SOURCE/'protocol.json'),'preregistration':prereg,'parent':str(PARENT)})
    panels={k:pd.read_parquet(output/(k+'_panel.parquet')) for k in PAIRS};panel=panels['semiconductor']
    reg=matched_registry(panels,market);ownreg=registry(panel,market);histories={k:{} for k in VARIANTS};predictions={k:[] for k in ROUTES};previous=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            if previous!=row.date:
                parents,feedback=mature_prefix(reg,at);daily={k:fit_day(panel,histories[k],ownreg,at) for k in VARIANTS};models={};members={}
                for c in CLOCKS:models[c],members[c]=fit_clock(panels,parents,c)
                filename=str(row.date.date());dump(output/'daily'/(filename+'.json'),{'calibration':daily,'feedback':feedback,'parents':parents,'members':members})
                write_models(output/'models'/(filename+'.json.gz'),models);previous=row.date
            raw={k:raw_prediction(models[row.clock],row,k) for k in VARIANTS};estimates={k:issue(panel,at,raw[k],daily[k]) for k in VARIANTS}
            oldpath=PARENT/'training'/(tag(at)+'.json');old=read(oldpath);issued=events(at,row,estimates,daily,feedback,old)
            dump(output/'training'/(tag(at)+'.json'),{'at':at,'date':row.date,'clock':row.clock,'cutoff':day_cutoff(at),
                 'raws':raw,'estimates':estimates,'events':issued,'parent_source_sha256':sha(oldpath),
                 'old_quadratic_raw':old['raws']['partial'],'old_quadratic_estimate':old['estimates']['partial'],
                 'linear_raw':old['linear_raw'],'linear_estimate':old['linear_estimate']})
            for k,e in issued.items():predictions[k].append(e)
            for k in VARIANTS:histories[k][at]=raw[k]
            if (i+1)%400==0:print('Normalized forecasts',i+1,'/',len(panel),at,flush=True)
    dump(output/'predictions.json',predictions);settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    actions={k:[e for e in es if start<=str(pd.Timestamp(e['date']).date())<=end] for k,es in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_capital_outcomes_seen':False,
         'prior_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':actions})
    features=pd.read_parquet(output/'features.parquet');test=features[features.date.between(start,end)]
    base=labels(output/'baseline_labels.json');stress=labels(output/'stress_labels.json');books={};result={'phase':'development','primary_name':p['primary'],'routes':{}}
    for name,es in actions.items():
        multiple=2 if name=='double_cost' else 1
        book=replay(market,test,list((stress if multiple==2 else base).values()),{pd.Timestamp(e['at']):e['action'] for e in es},start,end,
             cost_multiple=multiple,allocations={pd.Timestamp(e['at']):e['allocation'] for e in es},require_full_cash=False)
        books[name]=book;save_replay(output,name,book);result['routes'][name]=score(book,months)
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)]
              for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    complete=list(test.index)==expected
    result['coverage']={'complete':complete,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique(),'same_day_ticks_are_not_independent_dates':True}
    result['primary']=result['routes'][p['primary']];result['development_gate']=development_gate(result['primary'],complete,p)
    result['case_regression']={'executed':False,'eligible_after_verification':result['development_gate']['passed']}
    result['paired_increments']={k:paired_uncertainty(books[p['primary']][0],books[k][0]) for k in ROUTES if k not in [p['primary'],'cash','double_cost']}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False);dump(output/'results.json',result)
    print({k:{'mean':s['primary_average'],'months':[m['return'] for m in s['monthly']]} for k,s in result['routes'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.output)
