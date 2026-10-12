"""Independently verify date-weighted errors and local derivative diagnostics."""
from shared_error import *
from audit_shared import predict as ref_predict
from verify_reliability import close
from verify_continuation import near


def audit(run):
    run=Path(run);d=read(run/'diagnostics.json');p=pd.read_parquet(run/'panel.parquet');obs=[];ready=0;derivatives=0
    slopes={pd.Timestamp(r['at']):r for r in d['state_dependent_score_slopes']}
    for at,r in p[p.date.between('2026-01-01','2026-04-30')].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));info=rec['estimates']['interaction']
        if not info['ready']:continue
        ready+=1;daily=read(run/'daily'/rec['daily_record']);m=daily['variants']['interaction']['point_model']
        x=np.array([*rec['raw'],*[r[c] for c in STATE_COLUMNS],*[float(r.clock==c) for c in CLOCKS[1:]]]);ref=np.eye(2);center=np.eye(2)
        for k in range(2):
            a=x.copy();b=x.copy();a[k]+=1e-6;b[k]-=1e-6
            ref[:,k]+=(ref_predict(m,a)[0]-ref_predict(m,b)[0])/(2e-6)
            a[2:8]=m['input_center'][2:8];b[2:8]=m['input_center'][2:8]
            center[:,k]+=(ref_predict(m,a)[0]-ref_predict(m,b)[0])/(2e-6);derivatives+=4
        close(slopes[at]['observed_gain'],ref);close(slopes[at]['training_center_gain'],center)
        near(slopes[at]['state_slope_change_max'],abs(ref-center).max())
        if pd.isna(r.terminal_release):continue
        truth=np.array([r.C+r.D,r.C-r.D]);row={'date':r.date}
        for name in ['raw',*VARIANTS]:
            values=np.array(rec['raw'] if name=='raw' else rec['estimates'][name]['point']);row[name]=sum((truth-values)**2)/2
        obs.append(row)
    f=pd.DataFrame(obs)
    for name in ['raw',*VARIANTS]:near(d['date_equal_mse'][name],f.groupby('date')[name].mean().mean())
    assert d['mature_error_rows']==len(obs) and d['mature_error_dates']==f.date.nunique()
    sf=pd.DataFrame(d['source_v27_local_slopes']);cross=sf.groupby('date').spread_gain.agg(['min','max'])
    summary={c:{'min':float(g.spread_gain.min()),'max':float(g.spread_gain.max()),'positive':int(g.spread_gain.gt(0).sum()),'dates':len(g)} for c,g in sf.groupby('clock')}
    receipt={'passed':True,'ready_ticks':ready,'mature_error_ticks':len(obs),'unknown_pair_error_ticks_retained':ready-len(obs),
        'independent_finite_difference_derivatives':derivatives,'source_clock_spread_gain':summary,
        'opposite_source_gain_sign_dates':int(((cross['min']<0)&(cross['max']>0)).sum()),'source_gain_dates':len(cross),
        'local_derivative_not_causal_market_effect':True,'diagnostics_sha256':sha(run/'diagnostics.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostic_audit.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
