"""Independent raw reference bar, release time and current-history reconciliation."""
import argparse
from path_learning import *
from audit_state_paths import state_equations


def reference_status(context,bank,members,at,clock):
    if context is None or at.value not in context['positions']:return {'ready':False,'reason':'missing_reference_context'}
    i=context['positions'][at.value];q=context['queries'];r=context['reference'];cut=day_cutoff(at).value
    if not q['ready'][i] or q['history_release_ns'][i]==pd.NaT.value or not q['history_release_ns'][i]<cut or q['quote_release_ns'][i]==pd.NaT.value or not q['quote_release_ns'][i]<=at.value:return {'ready':False,'reason':'incomplete_current_history_or_quote'}
    ids=sorted(set(bank['blocks'][members].ravel().tolist()));first=bank['blocks'][members,0];c=CLOCKS.index(clock)
    values=[q['history'][i],q['quotes'][i],r['regular_close'][ids],r['previous_close'][ids],r['quotes'][first,c]]
    if any(not np.isfinite(x).all() or np.any(x<=0) for x in values) or np.any(r['release_ns'][ids]==pd.NaT.value) or np.any(r['release_ns'][ids]>=cut):return {'ready':False,'reason':'reference_support_missing'}
    return {'ready':True,'query_index':i,'source_days':ids,'history_release_ns':int(q['history_release_ns'][i]),'quote_release_ns':int(q['quote_release_ns'][i]),'latest_reference_release_ns':int(max(r['release_ns'][ids]))}


def audit(inputs):
    from state_paths import load_context,input_status
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');market=core.load_market();assert market.manifest==m['full_market'];cutoff=pd.Timestamp(m['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    leaders=load_panel(market.calendar,PANEL_ROOT);origin=core.ROOT/'.cache/soxl_soxs_14d5pct_v55/runs/development_20261013/input_manifest.json';assert leaders[2]==read(origin)['panel']
    symbols=['SOXX','QQQ',*read(SOURCE/'protocol.json')['panel']];bars={s:(market.bars[s] if s in ['SOXX','QQQ'] else leaders[0][s]) for s in symbols};daily={};parts={};quotes={}
    for symbol,f in bars.items():
        f=f[(f.end_at<cutoff)&(f.available_at<cutoff)];daily[symbol]={};parts[symbol]={};quotes[symbol]={}
        for day,g in f.groupby('date'):
            cal=market.calendar.get(day)
            if cal is None:continue
            expected=pd.date_range(cal['open_at'],cal['close_at'],freq='5min',inclusive='right');reg=g[g.index.isin(expected)]
            good=reg.index.equals(expected) and reg.available_at.notna().all() and reg.available_at.ge(reg.end_at).all() and reg.start_at.eq(reg.end_at-pd.Timedelta(minutes=5)).all() and reg.econ_close.gt(0).all() and np.isfinite(reg.econ_close).all()
            if good:daily[symbol][day]=(float(reg.econ_close.iloc[-1]),reg.available_at.max());parts[symbol][day]=reg
            for clock in CLOCKS:
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
                if end not in g.index:continue
                z=g.loc[end]
                if pd.notna(z.available_at) and end<=z.available_at<=at and z.start_at==end-pd.Timedelta(minutes=5) and np.isfinite(z.econ_close) and z.econ_close>0:quotes[symbol][at]=(float(z.econ_close),z.available_at)
    panel=pd.read_parquet(inputs/'activity_panel.parquet');bank=dict(np.load(inputs/'path_bank.npz'));context=load_context(inputs);ref=context['reference'];query=context['queries'];sessions=list(market.sessions);positions={d:i for i,d in enumerate(sessions)};rawchecks=sourcechecks=statechecks=missingstates=0
    assert query['history'].shape==(len(panel),21,10) and query['quotes'].shape==(len(panel),10);np.testing.assert_array_equal(query['at_ns'],panel.index.asi8)
    assert ref['regular_close'].shape==(len(bank['date_ns']),78,10)
    for i,ns in enumerate(bank['date_ns']):
        day=pd.Timestamp(ns);j=positions[day];prev=sessions[j-1] if j else None;cal=market.calendar[day];length=int((pd.Timestamp(cal['close_at'])-pd.Timestamp(cal['open_at'])).total_seconds()/300)
        ready=length==78 and all(day in daily[s] and prev in daily[s] for s in symbols)
        if not ready:assert ref['release_ns'][i]==pd.NaT.value;continue
        releases=[]
        for k,s in enumerate(symbols):
            np.testing.assert_array_equal(ref['regular_close'][i,:,k],parts[s][day].econ_close.to_numpy(float));assert ref['previous_close'][i,k]==daily[s][prev][0];rawchecks+=78;releases.extend([daily[s][day][1],daily[s][prev][1]])
            for c,clock in enumerate(CLOCKS):
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC')
                if at in quotes[s]:assert ref['quotes'][i,c,k]==quotes[s][at][0];releases.append(quotes[s][at][1])
                else:assert np.isnan(ref['quotes'][i,c,k])
        assert ref['release_ns'][i]==max(releases).value;sourcechecks+=1
    rows=[];settings=read(SOURCE/'protocol.json')[m['phase']];currentmissing=0
    for i,(at,row) in enumerate(panel.iterrows()):
        j=positions[row.date];prior=sessions[max(0,j-21):j];cut=day_cutoff(at);ready=len(prior)==21 and all(d in daily[s] and daily[s][d][1]<cut for s in symbols for d in prior) and all(at in quotes[s] for s in symbols)
        assert query['ready'][i]==ready
        if ready:
            h=np.array([[daily[s][d][0] for s in symbols] for d in prior]);np.testing.assert_array_equal(query['history'][i],h);np.testing.assert_array_equal(query['quotes'][i],[quotes[s][at][0] for s in symbols])
            assert query['history_release_ns'][i]==max(daily[s][d][1] for s in symbols for d in prior).value;assert query['quote_release_ns'][i]==max(quotes[s][at][1] for s in symbols).value
            wanted=state_equations(h[None,:])[0];saved=row[PRIOR].to_numpy(float);finite=np.isfinite(saved);np.testing.assert_allclose(saved[finite],wanted[finite],rtol=1e-10,atol=1e-12);statechecks+=int(finite.sum());missingstates+=int((~finite).sum())
            if row.feature_ready:assert finite.all()
        else:
            currentmissing+=1;assert np.isnan(query['history'][i]).all() and np.isnan(query['quotes'][i]).all()
        members=price_known_pool(bank,at,row.clock)
        if len(members):
            expected=reference_status(context,bank,members,at,row.clock);assert input_status(context,bank,members,at,row.clock)==expected
            if settings['evaluation_start']<=str(row.date.date())<=settings['evaluation_end']:rows.append({'at':at,'date':row.date,'members':len(members),'reference_inputs_ready':expected['ready']})
    out={'passed':True,'phase':m['phase'],'source_reference_days':sourcechecks,'raw_reference_close_checks':rawchecks,'current_queries':len(panel),'current_missing':currentmissing,'initial_state_formula_checks':statechecks,'original_missing_state_fields_preserved':missingstates,'evaluation_queries':len(rows),'evaluation_all_references_ready':all(x['reference_inputs_ready'] for x in rows),'evaluation_missing':sum(not x['reference_inputs_ready'] for x in rows),'rows':rows,'reference_source_manifest_sha256':sha(origin),'reference_bank_sha256':sha(inputs/'reference_bank.npz'),'reference_queries_sha256':sha(inputs/'reference_queries.npz'),'input_manifest_sha256':sha(inputs/'input_manifest.json'),'source_sha256':sha(Path(__file__)),'capital_outcomes_read':False}
    dump(inputs/'state_input_audit.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
