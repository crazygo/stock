"""Freeze daily option direction forecasts before continuous capital replay."""
import argparse,gzip,pickle
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from option_learning import *
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay


def authorize_cases(path,p):
    root=Path(path);reg=read(root/'registration.json');r=read(root/'results.json');v=read(root/'verification.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert r['primary_name']==p['primary'] and r['primary']==r['routes'][p['primary']]
    assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    gate=development_gate(r['primary'],r['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Development primary failed; new case replay is forbidden')
    return {'run':str(root.resolve()),'results_sha256':sha(root/'results.json'),'verification_sha256':sha(root/'verification.json'),'gate':gate}


def run(output,case_after=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json');assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(core.ROOT/f)==h
    auth=authorize_cases(case_after,p) if case_after else None;phase='cases' if auth else 'development';output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve every attempt')
    root,source_hashes=verified_source(phase,p);manifest=read(root/'input_manifest.json');market=core.load_market()
    assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    panel=pd.read_parquet(root/'financial_panel.parquet');features=pd.read_parquet(root/'features.parquet')
    raw,option_manifest=load_raw();cal=calendar(*p['option_range'])
    assert option_manifest==read(SOURCE/'evidence/option_input_manifest.json')['files']
    added,daily_inputs,feature_receipts=build_features(features,raw,cal['sessions']);panel=panel.join(added)
    assert panel.loc[panel.feature_ready,'option_ready'].all()
    panel['feature_ready']=panel.feature_ready & panel.option_ready
    old={pd.Timestamp(e['at']):e for e in read(root/'predictions.json')['full_band']}
    for sub in ['','models','daily','training','executed_sources']:(output/sub).mkdir(parents=True,exist_ok=True)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,
        'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':auth,'source_sha256':prereg['code_sha256']})
    dump(output/'input_manifest.json',{'source':str(root),'source_files_sha256':source_hashes,'market':manifest,'option_source':option_manifest,'option_calendar':cal})
    panel.to_parquet(output/'option_panel.parquet');added.to_parquet(output/'option_features.parquet');daily_inputs.to_parquet(output/'option_daily.parquet');dump(output/'option_receipts.json',feature_receipts)
    for f in prereg['code_sha256']:
        src=core.ROOT/f;dest=output/'executed_sources'/src.parent.name/src.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(src.read_bytes())
    histories={kind:{} for kind in VARIANTS};reg=registry(panel,market);predictions={n:[] for n in ROUTES};previous=None;daily=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            if row.date!=previous:
                daily=fit_calibration(panel,histories,reg,at);dump(output/'daily'/(str(row.date.date())+'.json'),daily);previous=row.date
            models,fits,raws=fit_bases(panel,at);estimates=issue(panel,at,raws,daily);issued=events(at,row.date,row.clock,estimates,daily['feedback'],old[at])
            dump(output/'training'/(tag(at)+'.json'),{'at':at,'date':row.date,'clock':row.clock,'cutoff':day_cutoff(at),'fits':fits,
                'raws':raws,'estimates':estimates,'events':issued,'source_record_sha256':sha(root/'training'/(tag(at)+'.json'))})
            for n,e in issued.items():predictions[n].append(e)
            for kind in VARIANTS:
                histories[kind][at]=raws[kind]
                if models[kind] is not None:
                    with gzip.open(output/'models'/(kind+'_'+tag(at)+'.pkl.gz'),'wb') as stream:pickle.dump(models[kind],stream)
            if (i+1)%200==0:print('option trained through',at,flush=True)
    dump(output/'predictions.json',predictions);settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end']
    actions={n:[e for e in es if start<=str(e['date'].date())<=end] for n,es in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_v47_capital_outcomes_seen':False,
        'prior_versions_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':actions})
    test=features[features.date.between(start,end)];months=settings.get('months',settings.get('regression_months'));base=labels(root/'baseline_labels.json');stress=labels(root/'stress_labels.json');books={}
    result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for n,es in actions.items():
        multiple=2 if n=='double_cost' else 1
        book=replay(market,test,list((stress if multiple==2 else base).values()),{e['at']:e['action'] for e in es},start,end,
            cost_multiple=multiple,allocations={e['at']:e['allocation'] for e in es},require_full_cash=False)
        books[n]=book;save_replay(output,n,book);result['routes'][n]=score(book,months)
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)]
        for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    complete=list(test.index)==expected;result['coverage']={'complete':complete,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique()}
    result['primary']=result['routes'][p['primary']];result['paired_increments']={n:paired_uncertainty(books[p['primary']][0],books[n][0]) for n in ROUTES if n not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],complete,p);result['development_gate']=gate;result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['case_acceptance']={'passed':bool(complete and result['primary']['primary_pass'])}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    dump(output/'results.json',result);print({n:{'mean':s['primary_average'],'total':s['continuous_return']} for n,s in result['routes'].items()},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-after',type=Path)
    a=parser.parse_args();run(a.output,a.case_after)
