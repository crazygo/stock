"""Source coverage, scalar daily equations and all historical reaction queries."""
from state_calibration import *
from reaction_inputs import load_prices,PANEL
from reference_reaction import raw_sources,day_return,economic_volume,verify_reaction_table


def audit():
    price_hash=verify_prices();manifest=read(SOURCE/'evidence/price_inputs.json');tables,sessions=load_prices()
    raw,actions,_=raw_sources();dates=[x['session_date'] for x in sessions if x['session_date']<='2026-04-30'];checks=0
    for s in PANEL+['SPY']:
        for i,day in enumerate(dates):
            r=tables[s].loc[day];assert r.valid
            np.testing.assert_allclose(r.economic_volume,economic_volume(s,day,raw,actions),rtol=1e-12,atol=1e-7);checks+=1
            if i:
                assert r.return_valid
                np.testing.assert_allclose(r.economic_log_return,day_return(s,day,dates[i-1],raw,actions),rtol=1e-11,atol=1e-13);checks+=1
    market=extended_market();base=pd.read_parquet(PARENT/'semiconductor_panel.parquet');temp=CACHE/'reaction_preflight.csv'
    panel=prepare_panel(base,market,temp);table=pd.read_csv(temp);queries=verify_reaction_table(table,panel)
    # Comparison is an audit; no minute-volume replacement or input selection.
    comparison={};minute_hashes={}
    for s in PANEL+['SPY']:
        prefix='soxl_soxs_14d5pct_v7' if s!='SPY' else 'soxl_soxs_14d5pct_v70'
        path=ROOT/'.cache'/prefix/'data/bars'/f'{s}.parquet';minute_hashes[str(path.relative_to(ROOT))]=sha(path)
        f=pd.read_parquet(path);f=f[f.session_date.between('2024-10-14','2026-04-30')].copy()
        for k in ['start_at','end_at']:f[k]=pd.to_datetime(f[k],utc=True)
        calendar={x['session_date']:x for x in sessions}
        start=pd.to_datetime(f.session_date.map({d:r['open_at'] for d,r in calendar.items()}),utc=True)
        end=pd.to_datetime(f.session_date.map({d:r['close_at'] for d,r in calendar.items()}),utc=True)
        regular=f[(f.start_at>=start)&(f.end_at<=end)]
        daily=regular.groupby('session_date').agg(close=('close','last'),volume=('volume','sum'))
        reference=tables[s].loc[daily.index]
        assert (reference.close.to_numpy()==daily.close.to_numpy()).all(),s
        ratio=daily.volume.to_numpy()/reference.volume.to_numpy()
        comparison[s]=dict(dates=len(daily),closes_exact=True,minute_to_daily_volume_median=float(np.median(ratio)),
            minute_to_daily_volume_minimum=float(min(ratio)),minute_to_daily_volume_maximum=float(max(ratio)))
    # Inputs must add degrees of freedom beyond the old nine state coordinates.
    columns=['trend5','current','relative5','log_vol20','event_future_count','event_future_proximity','event_recent_count','event_recent_proximity']
    X=panel[columns].to_numpy(float);X=np.c_[np.ones(len(X)),X,panel.trend5.to_numpy()*panel.current.to_numpy()]
    finite=np.isfinite(X).all(axis=1);A=X[finite];A=A/np.maximum(np.linalg.norm(A,axis=0),1e-12)
    Y=panel.loc[finite,['reaction_count','issuer_reaction','peer_reaction','volume_reaction']].to_numpy(float)
    residual=Y-A@np.linalg.lstsq(A,Y,rcond=None)[0]
    Yscaled=Y/np.maximum(np.linalg.norm(Y,axis=0),1e-12)
    rank_before=int(np.linalg.matrix_rank(A));rank_after=int(np.linalg.matrix_rank(np.c_[A,Yscaled]));assert rank_after>rank_before
    out=dict(passed=True,price_manifest_sha256=price_hash,source_sha256=sha(Path(__file__)),daily_scalar_checks=checks,
        query_verification=queries,minute_comparison=comparison,minute_comparison_files_sha256=minute_hashes,
        information_check=dict(rows=int(finite.sum()),old_coordinate_rank=rank_before,combined_rank=rank_after,
            residual_rms=np.sqrt(np.mean(residual**2,axis=0)).tolist(),descriptive_not_predictive_validation=True),
        observed_query_rows=int((table.reaction_count>0).sum()),pending_query_rows=int((table.reaction_pending_count>0).sum()),
        new_capital_outcomes_seen=False,goal_complete=False)
    dump(SOURCE/'evidence/reaction_verification.json',out);print(out,flush=True)


if __name__=='__main__':audit()
