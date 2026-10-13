"""Read-only attribution after full verification; no parameter or outcome filtering."""
import argparse
from path_learning import *


def audit(run):
    run=Path(run);v=read(run/'verification.json');result=read(run/'results.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(SOURCE/'protocol.json')[result['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');rows=[];unavailable=[]
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));a,b=rec['fits']['dynamic'],rec['fits']['fresh'];assert a['members']==b['members']
        if not a['ready']:
            unavailable.append({'at':at,'fresh_ready':b['ready'],'state_inputs':a.get('state_inputs'),'state_transition':a.get('state_transition')});continue
        assert b['ready'] and a['kernels']['first']==b['kernels']['first']
        for key in ['resolved_members','unresolved_members','unresolved_probability','exact_moments']:assert a['partition'][key]==b['partition'][key]
        mass=a['partition']['unresolved_probability']
        if mass==0:assert a['raw']==b['raw']
        ca,cb=rec['events']['dynamic_point'],rec['events']['fresh_point'];trace=a['state_transition']
        rows.append({'at':at,'date':row.date,'unresolved_probability':mass,'absolute_raw_difference':float(np.abs(np.array(rec['raws']['dynamic'])-rec['raws']['fresh']).mean()),'raw_preferred_side_changed':int(np.argmax(rec['raws']['dynamic'])!=np.argmax(rec['raws']['fresh'])),'proposal_changed':int(ca['action']!=cb['action'] or ca['allocation']!=cb['allocation']),'minimum_effective_dates':trace['minimum_effective_dates'],'reference_relative_change':trace['mean_absolute_state_change'][0],'reference_breadth_change':trace['mean_absolute_state_change'][1],'reference_logvol_change':trace['mean_absolute_state_change'][2]})
    frame=pd.DataFrame(rows);columns=['unresolved_probability','absolute_raw_difference','raw_preferred_side_changed','proposal_changed','reference_relative_change','reference_breadth_change','reference_logvol_change'];average=frame.groupby('date')[columns].mean().mean().to_dict() if len(frame) else {}
    entries={}
    for name in ['dynamic_point','fresh_point']:
        try:t=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        entries[name]=set(zip(t.decision_at,t.symbol)) if len(t) else set()
    common=entries['dynamic_point']&entries['fresh_point'];out={'passed':True,'post_result_diagnostic':True,'rules_changed':False,'phase':result['phase'],'ready_queries':len(rows),'unavailable_queries':unavailable,'evaluation_dates':int(frame.date.nunique()) if len(frame) else 0,'all_resolved_queries':int(frame.unresolved_probability.eq(0).sum()) if len(frame) else 0,'same_first_kernel_and_exact_component':True,'date_equal_averages':average,'minimum_dynamic_effective_dates':float(frame.minimum_effective_dates.min()) if len(frame) and frame.minimum_effective_dates.notna().any() else None,'changed_proposal_queries':int(frame.proposal_changed.sum()) if len(frame) else 0,'changed_proposal_dates':int(frame.loc[frame.proposal_changed.eq(1),'date'].nunique()) if len(frame) else 0,'primary_entries':len(entries['dynamic_point']),'fixed_entries':len(entries['fresh_point']),'shared_entry_events':len(common),'only_primary_entries':len(entries['dynamic_point']-common),'only_fixed_entries':len(entries['fresh_point']-common),'rows':rows,'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'source_sha256':sha(Path(__file__)),'scope':'State and causal proposal attribution only. Does not read individual future labels, select rules or qualify a failed primary. Repeated intraday queries and simulated path counts are not independent market samples.'}
    dump(run/'transition_effect_audit.json',out);print({k:v for k,v in out.items() if k not in ['rows','unavailable_queries']},flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
