"""Recompute every descriptive loss, comparison and actual-entry observation."""
import argparse
from block_learning import *


def audit(run):
    run=Path(run);out=read(run/'diagnostics.json');p=pd.read_parquet(run/'semiconductor_panel.parquet');p=p[p.date.between('2026-01-01','2026-04-30')]
    table=pd.read_csv(run/'diagnostic_rows.csv');assert list(pd.to_datetime(table['at'],utc=True))==list(p.index)
    checks=0;series={k:{s:{} for s in ['raw','cal']} for k in VARIANTS+['uniform','old_quadratic','linear']};available={k:{s:0 for s in ['raw','cal']} for k in series}
    for i,(at,row) in enumerate(p.iterrows()):
        rec=read(run/'training'/(tag(at)+'.json'));truth=np.array([row.C+row.D,row.C-row.D]);mature=pd.notna(row.terminal_release);saved=table.iloc[i]
        assert saved.mature==mature
        for k in series:
            raw=rec[k+'_raw'] if k in ['uniform','linear','old_quadratic'] else rec['raws'][k];info=rec[k+'_estimate'] if k in ['uniform','linear','old_quadratic'] else rec['estimates'][k]
            for stage,pred in [('raw',raw),('cal',info['point'])]:
                available[k][stage]+=pred is not None;assert saved[k+'_'+stage+'_available']==(pred is not None)
                if mature and pred is not None:
                    error=sum((np.asarray(pred)-truth)**2)/2;j=0 if pred[0]>=pred[1] else 1;regret=max(truth)-truth[j]
                    np.testing.assert_allclose([saved[k+'_'+stage+'_mse'],saved[k+'_'+stage+'_top_regret']],[error,regret],atol=1e-12,rtol=1e-10)
                    series[k][stage].setdefault(str(row.date.date()),[]).append([error,regret]);checks+=2
            prefix=k if k in ['uniform','linear','old_quadratic'] else 'block_'+k
            for mode in ['band','point']:assert saved[k+'_'+mode+'_action']==rec['events'][prefix+'_'+mode]['action']
    for k in series:
        for stage,days in series[k].items():
            v=out['errors'][k][stage];means=np.array([np.mean(x,axis=0) for x in days.values()]);np.testing.assert_allclose([v['date_equal_mse'],v['date_equal_top_regret']],means.mean(axis=0),atol=1e-12,rtol=1e-12)
            assert v['dates']==len(days) and v['ticks']==sum(map(len,days.values())) and v['available_ticks']==available[k][stage]
    for k,record in out['primary_minus_controls'].items():
        for stage in ['raw','cal']:
            dates=sorted(set(series['partial'][stage]) & set(series[k][stage]));d=np.array([np.mean(series['partial'][stage][x],axis=0)[0]-np.mean(series[k][stage][x],axis=0)[0] for x in dates])
            assert record[stage]['matched_dates']==len(dates) and record[stage]['dates_partial_better']==int((d<0).sum())
            np.testing.assert_allclose(record[stage]['mean_mse_difference'],d.mean(),atol=1e-12)
        for mode in ['band','point']:assert record[mode+'_different_proposal_ticks']==int(table['partial_'+mode+'_action'].ne(table[k+'_'+mode+'_action']).sum())
    try:t=pd.read_csv(run/'block_partial_band_trades.csv');a=pd.read_csv(run/'actual_entry_forecasts.csv')
    except pd.errors.EmptyDataError:t=pd.DataFrame();a=pd.DataFrame()
    assert len(a)==len(t)==out['primary_actual_entry_forecasts']['entries']
    for i,row in a.iterrows():
        rec=read(run/'training'/(tag(pd.Timestamp(row.decision_at))+'.json'));j=core.ETFS.index(row.symbol);info=rec['estimates']['partial']
        assert row.decision_at==t.iloc[i].decision_at
        np.testing.assert_allclose([row.raw_selected,row.point_selected,row.lower_selected,row.actual_log],[rec['raws']['partial'][j],info['point'][j],info['lower'][j],np.log1p(t.iloc[i].net_return)],atol=1e-12,rtol=1e-12,equal_nan=True)
    receipt={'passed':True,'diagnostics_sha256':sha(run/'diagnostics.json'),'loss_and_regret_rows_recomputed':checks,
             'actual_entry_forecasts':len(a),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics_audit.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();audit(a.run)
