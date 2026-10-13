"""Reconstruct each feature from raw counts using explicit historical windows."""
from state_reliability import read, dump, sha, core, pd, np
from option_features import load_raw, SYMBOLS, PANEL, option_columns, NAMES
from verify_reliability import close


def reference_day(raw, dates, k):
    if k < 20:
        return None
    vals = {}
    for s in SYMBOLS:
        o = raw[s]['options'].set_index('source_day')
        v = raw[s]['volatility'].set_index('source_day')
        days = [str(d.date()) for d in dates[k-20:k+1]]
        if not all(d in o.index for d in days) or not all(d in v.index for d in [days[-1],days[-6]]):
            return None
        rows = o.loc[days]
        counts = rows[['call_volume','put_volume','option_volume']].to_numpy(float)
        if not np.isfinite(counts).all() or (counts < 0).any():
            return None
        total = counts[:,0]+counts[:,1]
        if (total<=0).any() or not np.array_equal(total,counts[:,2]):
            return None
        vol = v.loc[[days[-1],days[-6]], ['iv','hv']].to_numpy(float)
        if not np.isfinite(vol).all() or (vol<=0).any():
            return None
        balance = (counts[:,0]-counts[:,1])/total
        activity = np.log(total[-1]/sum(total[:-1])*20)
        vals[s] = [balance[-1],balance[-1]-balance[-6],activity,float(activity>0),
                   np.log(vol[0,0]/vol[0,1]),np.log(vol[0,0]/vol[1,0])]
    x = [sum(vals[s][i] for s in PANEL)/8 for i in range(6)]
    x.extend([vals['SOXX'][0]-vals['QQQ'][0],vals['SOXX'][4]-vals['QQQ'][4]])
    return x


def audit_option_features(run):
    raw, manifest = load_raw()
    saved = read(run/'input_manifest.json')
    assert saved['option_source'] == manifest
    dates = pd.DatetimeIndex([s['session_date'] for s in saved['option_calendar']['sessions']])
    panel = pd.read_parquet(run/'option_panel.parquet')
    added = pd.read_parquet(run/'option_features.parquet')
    receipts = {d['date']:d for d in read(run/'option_receipts.json')['days']}
    equations = time_checks = 0
    for day, group in panel.groupby('date'):
        j = dates.get_loc(day)
        ready = True
        for lag in [2,5]:
            k = j-lag
            x = reference_day(raw,dates,k)
            r = receipts[str(day.date())]['lags'][str(lag)]
            assert r['latest_source_day'] == (str(dates[k].date()) if k>=0 else None)
            assert r['assumed_available_day'] == str(day.date())
            assert r['ready'] == (x is not None)
            ready &= x is not None
            if x is not None:
                assert r['source_days_21'] == [str(d.date()) for d in dates[k-20:k+1]]
                assert dates[k] < day and dates[k+lag] == day
                close([r['values'][n] for n in NAMES],x)
                for at in group.index:
                    close(panel.loc[at,option_columns(lag)].to_numpy(dtype=float),x)
                    close(added.loc[at,option_columns(lag)].to_numpy(dtype=float),x)
                    equations += len(x)
                time_checks += 1
        assert group.option_ready.eq(ready).all()
    src = pd.read_parquet(core.ROOT/saved['source']/'financial_panel.parquet')
    assert panel.feature_ready.equals(src.feature_ready)
    receipt = {'passed':True,'raw_files':len(manifest),'feature_equations':equations,
               'availability_checks':time_checks,'historical_publication_certified':False,
               'matched_original_training_readiness':True,
               'panel_sha256':sha(run/'option_panel.parquet')}
    dump(run/'option_feature_audit.json',receipt)
    return receipt
