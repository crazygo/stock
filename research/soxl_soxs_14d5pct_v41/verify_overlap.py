"""Independent interval geometry, weighting, model equations and continuous ledger audit."""
import argparse,gzip,pickle
from threadpoolctl import threadpool_limits
from overlap_learning import *
from audit_overlap import base_check
from audit_overlap import daily_check,input_rows,predict as reference_predict
from verify_reliability import close
from verify_financial import check_action
from verify_continuation import near
from audit_books import book_check
from audit_statistics import audit_statistics


def check_spans(panel,market,labelmap,saved):
    from verify_continuation import key
    from datetime import timedelta
    grid=[]
    for d in market.sessions:
        t=pd.Timestamp(market.calendar[d]['open_at'])+timedelta(minutes=5);end=pd.Timestamp(market.calendar[d]['close_at'])
        while t<=end:grid.append(t);t+=timedelta(minutes=5)
    assert list(pd.to_datetime(saved['grid'],utc=True))==grid
    positions={t:i for i,t in enumerate(grid)};out=panel.copy();out['span_start']=-1;out['span_end']=-1
    rows={pd.Timestamp(r['at']):r for r in saved['rows']};known=list(panel[panel.terminal_release.notna()].index);assert list(rows)==known
    for at in known:
        pair=[labelmap[key(at,s)] for s in core.ETFS];entry=pd.Timestamp(pair[0]['entry_at']);ends=[pd.Timestamp(z['exit_at']) for z in pair]
        assert entry==pd.Timestamp(pair[1]['entry_at']) and entry>=at and panel.loc[at,'terminal_release']>max(ends)
        # Explicit filtering independent of production searchsorted/indexer.
        eligible=[t for t in grid if entry<t<=max(ends)];assert eligible
        a=positions[eligible[0]];b=positions[eligible[-1]];r=rows[at]
        assert r['start']==a and r['end']==b and pd.Timestamp(r['entry_at'])==entry
        assert [pd.Timestamp(x) for x in r['exits']]==ends
        assert pd.Timestamp(r['first_bar_end'])==grid[a] and pd.Timestamp(r['last_bar_end'])==max(ends)==grid[b]
        out.loc[at,['span_start','span_end']]=[a,b]
    return out


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    root,want=verified_source(reg['phase'],p);assert root==Path(manifest['source']) and want==manifest['source_files_sha256']
    market=core.load_market();assert market.manifest==manifest['market']['full_market'];market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    features=pd.read_parquet(root/'features.parquet');panel=pd.read_parquet(root/'financial_panel.parquet');old=read(root/'predictions.json')['full_band'];panel=check_spans(panel,market,labels(root/'baseline_labels.json'),read(run/'span_registry.json'))
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
                info=rec['fits'][kind]['fit'];assert list(pd.to_datetime(info['members'],utc=True))==ids and info['columns']==columns(row.clock)
                path=run/'models'/(kind+'_'+tag(at)+'.pkl.gz');model=pickle.load(gzip.open(path,'rb')) if path.exists() else None
                values,n=base_check(same,ids,at,model,rec['fits'][kind],columns(row.clock),kind);checks+=n
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
            # Identical source means/errors and learner are reconstructed, not substituted.
            assert rec['fits']['baseline']==source['fits']['full'] and rec['raws']['baseline']==source['raws']['full']
            assert rec['fits']['cal_only']==rec['fits']['baseline'] and rec['fits']['unique']==rec['fits']['base_only']
            if row.date!=prev:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));models,n=daily_check(panel,market,histories,at,daily);calchecks+=n;daychecks+=1;prev=row.date
                source_day=read(root/'daily'/(str(row.date.date())+'.json'))
                for field in ['parents','feedback','members','ready']:assert daily[field]==source_day[field]
                if daily['ready']:
                    for field in ['point_model','bootstrap_models']:assert daily['variants']['baseline'][field]==source_day['variants']['full'][field]
                    assert daily['variants']['baseline']['day_weights']==source_day['day_weights'] and daily['variants']['baseline']['row_weights']==source_day['row_weights']
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
                if name=='source_v35':assert e==old[i]
                elif name=='cash' or name=='premarket_only' and row.clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:
                    kind=name[:-5] if name in ['base_only_band','cal_only_band','baseline_band'] else 'unique'
                    check_action(rec['estimates'][kind],daily['feedback']['expired_unresolved_prefix'],name!='unique_point',e)
                actions+=1
            for kind in VARIANTS:histories[kind][at]=rec['raws'][kind]
            if (i+1)%200==0:print('overlap verified through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))};books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for name in ['baseline_band','source_v35']:
        for suffix in ['equity','trades','orders','decisions']:pd.testing.assert_frame_equal(pd.read_csv(run/(name+'_'+suffix+'.csv')),pd.read_csv(root/('full_band_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);old_predictions=read(dev/'predictions.json')
        for name,es in predictions.items():assert es[:len(old_predictions[name])]==old_predictions[name]
        for folder in ['training','daily']:
            for f in (dev/folder).glob('*.json'):assert read(f)==read(run/folder/f.name)
    stats=audit_statistics(run);assert result['primary_name']=='unique_band' and result['primary']==result['routes']['unique_band']
    if reg['phase']=='development':assert result['development_gate']==development_gate(result['primary'],result['coverage']['complete'],p)
    else:assert result['case_acceptance']['passed']==bool(result['coverage']['complete'] and result['primary']['primary_pass'])
    receipt={'passed':True,'audit_revision':1,'audit_sources_sha256':{str(f.relative_to(core.ROOT)):sha(f) for f in [Path(__file__),SOURCE/'audit_overlap.py']},'phase':reg['phase'],'clock_ticks':len(features),'daily_fits':daychecks,'base_equations':checks,
        'independent_calibration_equations':calchecks,'fresh_anchor_models':anchors,'action_checks':actions,'capital_books':books,
        'baseline_raw_calibration_actions_and_ledgers_exact':True,'independent_statistic_assertions':stats,'case_prefix_exact':reg['phase']=='cases',
        'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
