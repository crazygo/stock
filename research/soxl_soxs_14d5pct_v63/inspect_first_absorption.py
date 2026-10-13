"""Post-result exact early absorption audit; no new estimator or capital replay."""
import argparse,gzip,pickle
from path_learning import *
from audit_paths import reference_values


def inspect(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    panel=pd.read_parquet(run/'activity_panel.parquet');bank=dict(np.load(run/'path_bank.npz'));p=read(SOURCE/'protocol.json')[r['phase']];records=[];contracts=matched=0
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));a,b=rec['fits']['conditional'],rec['fits']['fixed'];assert a['ready']==b['ready']
        if not a['ready']:continue
        assert a['members']==b['members'];members=np.array(a['members']);plan=dict(a['plan']);assert len(plan['dates'])>2
        plan['dates']=plan['dates'][:2];plan['bar_counts']=plan['bar_counts'][:2];plan['expiry_at']=None
        raw=np.array([x['raw_close'] for x in a['query_quote']]);selected=members[:,None]
        short=simulate(bank,selected,row.clock,raw,plan);ref=reference_values(bank,selected,row.clock,raw,plan)
        for field in short:np.testing.assert_allclose(short[field],ref[field],rtol=1e-10,atol=1e-11)
        resolved=(short['reason']!=3).all(axis=1);contracts+=2*len(members)
        weights=[np.array(z['kernels']['first']['probability']) for z in [a,b]];models=[]
        for k in ['conditional','fixed']:
            with gzip.open(run/'models'/(k+'_'+tag(at)+'.pkl.gz'),'rb') as file:model=pickle.load(file)
            local=np.searchsorted(members,model['selected_blocks'][:,0]);closed=resolved[local]
            np.testing.assert_array_equal(closed,(model['result']['exit_day']<2).all(axis=1))
            np.testing.assert_allclose(model['result']['log_returns'][closed],short['log_returns'][local[closed]],rtol=1e-12,atol=1e-12)
            matched+=int(closed.sum())*2;models.append(model)
        common_mass=float(np.minimum(*weights).sum());tv=float(np.abs(weights[0]-weights[1]).sum()/2)
        assert abs(common_mass-(1-tv))<1e-12
        records.append({'at':at,'date':row.date,'source_blocks':len(members),'resolved_source_pairs':int(resolved.sum()),
            'primary_exact_resolved_probability':float(weights[0]@resolved),'fixed_exact_resolved_probability':float(weights[1]@resolved),
            'maximum_equal_first_index_coupling':common_mass,'observed_equal_first_index_fraction':float(np.mean(models[0]['selected_blocks'][:,0]==models[1]['selected_blocks'][:,0]))})
    frame=pd.DataFrame(records);cols=['primary_exact_resolved_probability','fixed_exact_resolved_probability','maximum_equal_first_index_coupling','observed_equal_first_index_fraction']
    out={'passed':True,'post_result_diagnostic':True,'new_estimator_built':False,'new_capital_replay':False,'queries':len(frame),'dates':frame.date.nunique(),
        'exact_two_session_contracts_independently_valued':contracts,'full_horizon_resolved_contracts_matched':matched,
        'date_equal_averages':frame.groupby('date')[cols].mean().mean().to_dict(),'primary_resolved_mass_min_max':[float(frame.primary_exact_resolved_probability.min()),float(frame.primary_exact_resolved_probability.max())],
        'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),'records':records,
        'interpretation':'Only first-block BOTH-ETF barriers qualify as fully resolved; forced two-session expiry is not an exit-rule change. Maximum equality is a coupling bound, not a win probability. No numerical-variance reduction or capital benefit established.'}
    dump(run/'first_absorption_audit.json',out);print({k:x for k,x in out.items() if k!='records'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);inspect(p.parse_args().run)
