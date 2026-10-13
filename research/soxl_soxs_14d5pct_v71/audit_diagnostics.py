"""Verify descriptive numbers and exact original semiconductor label lineage."""
import argparse
from transfer_learning import *


def audit(run):
    run=Path(run);out=read(run/'diagnostics.json');manifest=read(run/'input_manifest.json');old=read(BASE/'input_manifest.json')
    assert all(manifest['market_manifest'][s]==m for s,m in old['full_market'].items())
    for f in ['features.parquet','baseline_labels.json','stress_labels.json']:
        assert sha(run/f)==sha(BASE/f)==old['derived_files_sha256'][f]
    legacy=ROOT/'research/soxl_soxs_14d5pct_v69/evidence';delivery=read(legacy/'delivery_manifest.json')
    for f,h in delivery['delivery_files_sha256'].items():assert sha(legacy.parent/f)==h
    assert read(legacy/'verification.json')['passed']
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');table=pd.read_csv(run/'diagnostic_rows.csv')
    checks=0;daily={k:{s:{} for s in ['raw','cal']} for k in VARIANTS};counts={k:[0,0] for k in VARIANTS}
    wanted=panel[panel.date.between('2026-01-01','2026-04-30')]
    assert list(pd.to_datetime(table['at'],utc=True))==list(wanted.index)
    for i,(at,row) in enumerate(wanted.iterrows()):
        rec=read(run/'training'/(tag(at)+'.json'));truth=np.array([row.C+row.D,row.C-row.D]);saved=table.iloc[i]
        assert saved.mature==pd.notna(row.terminal_release)
        for k in VARIANTS:
            for j,stage in enumerate(['raw','cal']):
                pred=rec['raws'][k] if stage=='raw' else rec['estimates'][k]['point']
                available=pred is not None;assert saved[k+'_'+stage+'_available']==available;counts[k][j]+=available
                if available and pd.notna(row.terminal_release):
                    err=sum((truth-np.asarray(pred))**2)/2
                    np.testing.assert_allclose(saved[k+'_'+stage+'_mse'],err,rtol=1e-10,atol=1e-12)
                    daily[k][stage].setdefault(str(row.date.date()),[]).append(float(err));checks+=1
    for k in VARIANTS:
        for j,stage in enumerate(['raw','cal']):
            got=out['errors'][k][stage];days=daily[k][stage]
            np.testing.assert_allclose(got['date_equal_mse'],np.mean([np.mean(v) for v in days.values()]),rtol=1e-12,atol=1e-12)
            assert got['dates']==len(days) and got['ticks']==sum(map(len,days.values()))
            assert out['errors'][k][stage+'_available']==counts[k][j]
    rows=pd.read_csv(run/'actual_entry_forecasts.csv');trades=pd.read_csv(run/'partial_band_trades.csv')
    assert list(rows.decision_at)==list(trades.decision_at)
    for i,r in rows.iterrows():
        rec=read(run/'training'/(tag(pd.Timestamp(r.decision_at))+'.json'));j=core.ETFS.index(r.symbol);info=rec['estimates']['partial']
        np.testing.assert_allclose([r.raw_selected,r.point_selected,r.lower_selected,r.radius,r.actual_log],
             [rec['raws']['partial'][j],info['point'][j],info['lower'][j],info['radius'],np.log1p(trades.iloc[i].net_return)],atol=1e-12,rtol=1e-12)
    receipt={'passed':True,'diagnostics_sha256':sha(run/'diagnostics.json'),'mse_rows_recomputed':checks,
             'actual_entry_forecasts':len(rows),'original_semiconductor_inputs_exact':True,
             'original_v69_delivery_sha256':sha(legacy/'delivery_manifest.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics_audit.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();audit(a.run)
