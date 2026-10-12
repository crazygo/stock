"""Independent new features, model equations, calibration and continuous ledger audit."""
import argparse,gzip,pickle
from threadpoolctl import threadpool_limits
from conditional_options import *
from audit_conditional_models import weighted_equations
from audit_conditional_calibration import daily_check,input_rows,predict as reference_predict
from verify_reliability import close
from verify_financial import check_action
from verify_continuation import near
from audit_books import book_check
from audit_statistics import audit_statistics
from audit_option_features import audit_option_features


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    controls=verified_controls(p);assert controls==manifest['factorial_controls']
    root,want=verified_source(reg['phase'],p);assert root==Path(manifest['source']) and want==manifest['source_files_sha256']
    market=core.load_market();assert market.manifest==manifest['market']['full_market'];market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    features=pd.read_parquet(root/'features.parquet');panel=pd.read_parquet(run/'option_panel.parquet');audit_option_features(run);old=read(root/'predictions.json')['full_band']
    histories={k:{} for k in VARIANTS};predictions=read(run/'predictions.json');prev=None;models=None;daily=None;checks=calchecks=actions=daychecks=anchors=0
    anchor_days={'2025-09-02','2026-01-05','2026-04-01','2026-06-01','2026-09-01'}
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            rec=read(run/'training'/(tag(at)+'.json'));source=read(root/'training'/(tag(at)+'.json'))
            assert rec['source_record_sha256']==sha(root/'training'/(tag(at)+'.json'))
            assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['date'])==row.date and rec['clock']==row.clock and pd.Timestamp(rec['cutoff'])==day_cutoff(at)
            same=panel[panel.clock==row.clock];prior=list(same[same.feature_ready&(same.date<row.date)].index);ids=[]
            for t in prior:
                if pd.isna(same.loc[t,'terminal_release']) or same.loc[t,'terminal_release']>=day_cutoff(at):break
                ids.append(t)
            fresh=fit_bases(panel,at) if str(row.date.date()) in anchor_days else None
            for kind in VARIANTS:
                info=rec['fits'][kind]['fit'];assert list(pd.to_datetime(info['members'],utc=True))==ids and info['columns']==columns(row.clock,kind)
                if kind!='baseline':assert info['q_columns']==q_columns(row.clock,kind)
                path=run/'models'/(kind+'_'+tag(at)+'.pkl.gz');model=pickle.load(gzip.open(path,'rb')) if path.exists() else None
                w=2.**(-np.arange(len(ids))[::-1]/126.)
                if len(w):w*=len(w)/w.sum()
                close(info['parent_weights'],w);near(info['weight_sum'],w.sum());near(info['parent_effective_n'],w.sum()**2/(w@w) if len(w) else 0.)
                values,n=weighted_equations(same,ids,at,model,info,columns(row.clock,kind),w,kind);checks+=n
                expected_raw=rec['fits'][kind]['raw']
                if values:
                    for k,x in values.items():near(expected_raw[k],x)
                    close(rec['raws'][kind],[values[s] for s in core.ETFS])
                else:assert expected_raw=={} and rec['raws'][kind] is None
                fb=rec['fits'][kind]['feedback'];assert fb['selected_early_dates']==len(ids)
                expired=len(ids)<len(prior) and same.loc[prior[len(ids)],'date']+pd.Timedelta(days=14)<=row.date
                assert fb['expired_unresolved_prefix']==bool(expired)
                if fresh is not None and model is not None:
                    close(rec['raws'][kind],fresh[2][kind]);anchors+=1
                    if kind in ['tree','delayed','old_tree'] and 'q' in model['heads']:
                        for aa,bb in zip(model['heads']['q']._predictors,fresh[0][kind]['heads']['q']._predictors):
                            np.testing.assert_array_equal(aa[0].nodes,bb[0].nodes)
            for kind in ['tree','delayed','old_tree','options']:
                for h in ['C','M']:
                    if rec['fits'][kind]['raw']:
                        near(rec['fits'][kind]['raw'][h],source['fits']['full']['raw'][h])
                        assert rec['fits'][kind]['fit']['heads'][h]==source['fits']['full']['fit']['heads'][h]
            for kind,version,source_kind in [('old_tree','40','tree'),('options','47','options')]:
                old_file=Path(controls[version]['path'])/'training'/(tag(at)+'.json')
                if old_file.exists():
                    control=read(old_file)
                    assert rec['raws'][kind]==control['raws'][source_kind]
                    assert rec['fits'][kind]['raw']==control['fits'][source_kind]['raw']
                    assert rec['estimates'][kind]==control['estimates'][source_kind]
                else:assert reg['phase']=='cases' and row.date>=pd.Timestamp('2026-05-01')
            # Identical source means/errors and learner are reconstructed, not substituted.
            assert rec['fits']['baseline']==source['fits']['full'] and rec['raws']['baseline']==source['raws']['full']
            if row.date!=prev:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));models,n=daily_check(panel,market,histories,at,daily);calchecks+=n;daychecks+=1;prev=row.date
                source_day=read(root/'daily'/(str(row.date.date())+'.json'))
                for field in ['parents','feedback','members','ready']:assert daily[field]==source_day[field]
                if daily['ready']:assert daily['variants']['baseline']==source_day['variants']['full']
            for kind in VARIANTS:
                raw=rec['raws'][kind];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];info=rec['estimates'][kind]
                assert info['ready']==bool(ready)
                if ready:
                    X=input_rows(panel,{at:raw},[at]);m=models[kind];point=np.array(raw)+reference_predict(m['point_model'],X)[0]
                    samples=np.array([np.array(raw)+reference_predict(z,X)[0] for z in m['bootstrap_models']]);radius=max(0.,sorted(np.max(samples-point,axis=1))[179])
                    close(info['point'],point);close(info['bootstrap_predictions'],samples);near(info['radius'],radius);close(info['lower'],point-radius)
                else:assert info['point'] is None and info['lower'] is None
            assert rec['estimates']['baseline']==source['estimates']['full']
            for name,es in predictions.items():
                e=es[i];assert e==rec['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name=='cash' or name=='premarket_only' and row.clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:
                    kind={'delayed_band':'delayed','options_band':'options','old_tree_band':'old_tree','baseline_band':'baseline'}.get(name,'tree')
                    check_action(rec['estimates'][kind],daily['feedback']['expired_unresolved_prefix'],name!='tree_point',e)
                actions+=1
            for kind in VARIANTS:histories[kind][at]=rec['raws'][kind]
            if (i+1)%200==0:print('option verified through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))};books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for name in ['baseline_band']:
        for suffix in ['equity','trades','orders','decisions']:pd.testing.assert_frame_equal(pd.read_csv(run/(name+'_'+suffix+'.csv')),pd.read_csv(root/('full_band_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    if reg['phase']=='development':
        for name,version,old_name in [('options_band','47','options_band'),('old_tree_band','40','tree_band')]:
            for suffix in ['equity','trades','orders','decisions']:
                pd.testing.assert_frame_equal(pd.read_csv(run/(name+'_'+suffix+'.csv')),pd.read_csv(Path(controls[version]['path'])/(old_name+'_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);old_predictions=read(dev/'predictions.json')
        for name,es in predictions.items():assert es[:len(old_predictions[name])]==old_predictions[name]
        for folder in ['training','daily']:
            for f in (dev/folder).glob('*.json'):assert read(f)==read(run/folder/f.name)
    stats=audit_statistics(run);assert result['primary_name']=='tree_band' and result['primary']==result['routes']['tree_band']
    if reg['phase']=='development':assert result['development_gate']==development_gate(result['primary'],result['coverage']['complete'],p)
    else:assert result['case_acceptance']['passed']==bool(result['coverage']['complete'] and result['primary']['primary_pass'])
    receipt={'passed':True,'audit_revision':1,'audit_sources_sha256':{str(f.relative_to(core.ROOT)):sha(f) for f in [Path(__file__),SOURCE/'audit_conditional_models.py',SOURCE/'audit_option_features.py']},'phase':reg['phase'],'clock_ticks':len(features),'daily_fits':daychecks,'base_and_tree_leaf_equations':checks,
        'independent_calibration_equations':calchecks,'fresh_anchor_models':anchors,'action_checks':actions,'capital_books':books,
        'baseline_raw_calibration_actions_and_ledgers_exact':True,'v40_and_v47_development_controls_exact':True,'independent_statistic_assertions':stats,'case_prefix_exact':reg['phase']=='cases',
        'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
