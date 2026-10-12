"""Audit actual daily model availability and mature training cutoffs after results."""
import argparse
from fixed_candidate import *


def audit(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'macro_panel.parquet');part=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])];root=Path(read(run/'input_manifest.json')['source'])
    events=read(run/'decisions_frozen.json')['routes']['strong_band'];out=[];base_checks=0
    for day,group in part.groupby('date'):
        d=read(run/'daily'/(str(day.date())+'.json'));cutoff=pd.Timestamp(d['cutoff']);assert cutoff==pd.Timestamp(day,tz=core.ET).tz_convert('UTC')
        parents=d['parents'];assert all(pd.Timestamp(x['date'])<day and pd.Timestamp(x['release'])<cutoff for x in parents);base=[]
        for at,row in group.iterrows():
            rec=read(run/'training'/(tag(at)+'.json'));s=read(root/'training'/(tag(at)+'.json'));assert rec['raw']==s['raws']['full'];ids=list(pd.to_datetime(s['fits']['full']['fit']['members'],utc=True))
            assert all(panel.loc[t,'date']<day and pd.notna(panel.loc[t,'terminal_release']) and panel.loc[t,'terminal_release']<cutoff for t in ids)
            base_checks+=len(ids);base.append({'clock':row.clock,'n_training_rows':len(ids),'last_training_date':panel.loc[ids[-1],'date'] if ids else None,'latest_release':panel.loc[ids,'terminal_release'].max() if ids else None,'forecast_ready':rec['raw'] is not None})
        out.append({'date':day,'cutoff':cutoff,'calibration_ready':d['ready'],'calibration_dates':len(parents),'latest_calibration_date':parents[-1]['date'] if parents else None,'latest_calibration_release':max(x['release'] for x in parents) if parents else None,'expired_history_block':d['feedback']['expired_unresolved_prefix'],'base_clocks':base})
    months={}
    for month,g in part.groupby(part.date.dt.to_period('M')):
        e=[x for x in events if x['date'].startswith(str(month))];ds=[x for x in out if str(x['date'].date()).startswith(str(month))]
        months[str(month)]={'dates':len(ds),'query_ticks':len(g),'all_calibration_ready':all(d['calibration_ready'] for d in ds),'all_base_forecasts_ready':all(c['forecast_ready'] for d in ds for c in d['base_clocks']),
            'expired_history_block_dates':sum(d['expired_history_block'] for d in ds),'positive_advantage_proposals':sum(x['action']!='CASH' for x in e),'maximum_buy_advantage':max(x['buy_advantage'] for x in e if x['ready']),'first_day':ds[0]}
    receipt={'passed':True,'post_result_only':True,'phase':r['phase'],'base_training_member_time_checks':base_checks,'monthly':months,'all_days':out,'results_sha256':sha(run/'results.json'),'code_sha256':sha(Path(__file__))}
    dump(run/'training_schedule_audit.json',receipt);print({'passed':True,'base_training_member_time_checks':base_checks,'monthly':months},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
