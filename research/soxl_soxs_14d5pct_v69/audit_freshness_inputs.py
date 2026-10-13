"""Independently enumerate each permitted known-price block and its added mass."""
import argparse
from path_learning import *
from audit_paths import reference_kernel
from audit_matched_control import matched_bank_source as matched_source


def audit(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json');phase=manifest['phase'];bank=dict(np.load(inputs/'path_bank.npz'));panel=pd.read_parquet(inputs/'activity_panel.parquet')
    old=matched_source(phase);prior=dict(np.load(old/'path_bank.npz'));assert set(bank)==set(prior)
    for key in bank:np.testing.assert_array_equal(bank[key],prior[key])
    rows=[];new_counts=[];total_added=0;new_readiness=0;settings=read(SOURCE/'protocol.json')[phase]
    for at,row in panel.iterrows():
        c=CLOCKS.index(row.clock);cut=day_cutoff(at).value;want=[];legacy=[]
        for i,(a,b) in enumerate(bank['blocks']):
            if any(bank['release_ns'][j]==pd.NaT.value or bank['release_ns'][j]>=cut for j in [a,b]):continue
            if bank['date_ns'][b]>=row.date.value:continue
            if not (np.isfinite(bank['quotes'][a,c]).all() and np.isfinite(bank['states'][a,c]).all() and np.isfinite(bank['states'][a,0,:3]).all() and np.isfinite(bank['absolute_states'][a,c]) and np.isfinite(bank['activity_states'][a,c])):continue
            want.append(i)
            if np.isfinite(bank['successor_states'][i]).all() and bank['successor_at_ns'][i]!=pd.NaT.value and bank['successor_at_ns'][i]<cut:legacy.append(i)
        assert price_known_pool(bank,at,row.clock).tolist()==want and source_pool(bank,at,row.clock).tolist()==legacy
        assert set(legacy)<=set(want);added=sorted(set(want)-set(legacy));total_added+=len(added);new_counts.append(len(added))
        for i in added:
            a,b=bank['blocks'][i];assert max(bank['release_ns'][a],bank['release_ns'][b])<cut
            assert not (np.isfinite(bank['successor_states'][i]).all() and bank['successor_at_ns'][i]!=pd.NaT.value and bank['successor_at_ns'][i]<cut)
        if len(want)>=120 and row.feature_ready and np.isfinite(row[FIELDS+[ABSOLUTE,ACTIVITY]].to_numpy(float)).all():
            ids=bank['blocks'][want,0];features=np.column_stack((bank['states'][ids,c],bank['absolute_states'][ids,c],bank['activity_states'][ids,c]));k=reference_kernel(features,row[FIELDS+[ABSOLUTE,ACTIVITY]].to_numpy(float),True)
            old,absolute,fresh=first_kernels(bank,ids,row.clock,row);later=reference_kernel(bank['states'][ids,0,:3],row[PRIOR].to_numpy(float),True)
            np.testing.assert_allclose(fresh['probability'],k['probability'],rtol=1e-11,atol=1e-13)
            ready=all(z['ready'] for z in [old,absolute,fresh,later]);new_readiness+=int(ready and len(legacy)<120)
            mass=sum(k['probability'][j] for j,v in enumerate(want) if v in added)
            if settings['evaluation_start']<=str(row.date.date())<=settings['evaluation_end']:
                last=lambda members:pd.Timestamp(max(bank['date_ns'][bank['blocks'][i,1]] for i in members))
                rows.append({'at':at,'date':row.date,'cutoff':day_cutoff(at),'legacy_members':len(legacy),'fresh_members':len(want),'added_members':added,'added_probability':mass,'legacy_latest_end':last(legacy),'fresh_latest_end':last(want),'legacy_age_days':int((row.date-last(legacy)).days),'fresh_age_days':int((row.date-last(want)).days),'fresh_first_ess':k['effective_dates'],'all_actual_kernels_ready':ready})
    frame=pd.DataFrame(rows);columns=['legacy_members','fresh_members','added_probability','legacy_age_days','fresh_age_days','fresh_first_ess']
    result={'passed':True,'phase':phase,'queries':len(panel),'evaluation_queries':len(rows),'evaluation_dates':int(frame.date.nunique()),'all_13_bank_arrays_exact_v66':True,'total_added_members_across_all_queries':total_added,'added_members_range':[min(new_counts),max(new_counts)],'newly_minimum_sample_ready_queries':new_readiness,'evaluation_queries_with_newer_history':int(frame.fresh_age_days.lt(frame.legacy_age_days).sum()),'date_equal_averages':frame.groupby('date')[columns].mean().mean().to_dict(),'rows':rows,'capital_outcomes_read_by_this_audit':False,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'all_added_prices_known_before_cutoff':True,'successor_dependency_only':True}
    dump(inputs/'freshness_audit.json',result);print({k:v for k,v in result.items() if k!='rows'},flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
