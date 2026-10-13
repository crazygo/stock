"""Read-only causal input availability diagnosis; no new strategy or capital replay."""
import argparse
from path_learning import *


def inspect(inputs):
    inputs=Path(inputs);reg=read(SOURCE/'evidence/successor_gate_inspection_registration.json');assert reg['source_sha256']==sha(Path(__file__))
    assert reg['input_manifest_sha256']==sha(inputs/'input_manifest.json');m=read(inputs/'input_manifest.json')
    for f,h in m['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    assert read(inputs/'bank_audit.json')['passed'] and read(inputs/'activity_audit.json')['passed']
    bank=dict(np.load(inputs/'path_bank.npz'));panel=pd.read_parquet(inputs/'activity_panel.parquet');rows=[];p=read(SOURCE/'protocol.json')['development'];b=bank['blocks'];first,last=b.T
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        c=CLOCKS.index(row.clock);cut=day_cutoff(at);good=(bank['release_ns'][first]<cut.value)&(bank['release_ns'][last]<cut.value)&(bank['date_ns'][last]<row.date.value)
        good&=np.isfinite(bank['quotes'][first,c]).all(axis=1)&np.isfinite(bank['states'][first,c]).all(axis=1)&np.isfinite(bank['states'][first,0,:3]).all(axis=1)
        good&=np.isfinite(bank['absolute_states'][first,c])&np.isfinite(bank['activity_states'][first,c]);price_known=np.flatnonzero(good)
        legacy=source_pool(bank,at,row.clock);assert set(legacy)<=set(price_known)
        added=np.array(sorted(set(price_known)-set(legacy)),int)
        unavailable=(~np.isfinite(bank['successor_states'][added]).all(axis=1))|(bank['successor_at_ns'][added]==pd.NaT.value)|(bank['successor_at_ns'][added]>=cut.value);assert unavailable.all()
        # Independently inspect each added pair's original source receipts.
        for i in added:
            a,z=b[i];assert bank['release_ns'][a]<cut.value and bank['release_ns'][z]<cut.value and bank['date_ns'][z]<row.date.value
            assert np.isfinite(bank['ohlc'][[a,z]]).all() and np.isfinite(bank['quotes'][a,c]).all()
        ids=first[price_known];x=np.column_stack((bank['states'][ids,c],bank['absolute_states'][ids,c],bank['activity_states'][ids,c]))
        hypothetical=kernel(x,row[FIELDS+[ABSOLUTE,ACTIVITY]].to_numpy(float),'conditional');w=np.array(hypothetical['probability']);added_mass=float(sum(w[np.isin(price_known,added)]))
        last_day=lambda members:pd.Timestamp(max(bank['date_ns'][last[members]]))
        oldlast,newlast=last_day(legacy),last_day(price_known)
        rows.append({'at':at,'date':row.date,'legacy_source_blocks':len(legacy),'price_known_source_blocks':len(price_known),'added_blocks':len(added),'added_members':added.tolist(),
            'added_probability_under_unchanged_six_field_formula':added_mass,'legacy_newest_source_last_day':oldlast,'price_known_newest_source_last_day':newlast,
            'legacy_newest_age_calendar_days':int((row.date-oldlast).days),'price_known_newest_age_calendar_days':int((row.date-newlast).days),'hypothetical_first_ess':hypothetical['effective_dates']})
    frame=pd.DataFrame(rows);fields=['legacy_source_blocks','price_known_source_blocks','added_blocks','added_probability_under_unchanged_six_field_formula','legacy_newest_age_calendar_days','price_known_newest_age_calendar_days','hypothetical_first_ess']
    out={'passed':True,'queries':len(rows),'dates':int(frame.date.nunique()),'all_added_price_paths_known_before_daily_cutoff':True,'date_equal_averages':frame.groupby('date')[fields].mean().mean().to_dict(),
        'added_blocks_min_max':[int(frame.added_blocks.min()),int(frame.added_blocks.max())],'queries_newest_history_becomes_newer':int((frame.price_known_newest_age_calendar_days<frame.legacy_newest_age_calendar_days).sum()),
        'future_return_labels_used':False,'new_capital_replay_executed':False,'rules_changed':False,'registration_sha256':sha(SOURCE/'evidence/successor_gate_inspection_registration.json'),'source_sha256':sha(Path(__file__)),'rows':rows,
        'scope':'Removing an unused successor gate is an unimplemented candidate. Available historical price paths alone do not establish prediction or capital improvement. Hypothetical weights are not issued forecasts.'}
    dump(SOURCE/'evidence/successor_gate_input_audit.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);inspect(p.parse_args().inputs)
