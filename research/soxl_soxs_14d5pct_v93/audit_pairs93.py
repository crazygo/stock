"""Audit paired population scores separately from policy-selected outcomes."""
import math
from paired93 import *


def audit_pairs(run,panel,records,report,end):
    table=pd.read_csv(run/'paired_candidate_rows.csv');models=['fundamental','reaction','raw_fundamental','identity'];checks=0;buckets={}
    assert len(table)==len(records)*8 and not table.duplicated(['at','model','symbol']).any()
    for r in table.itertuples():
        at=pd.Timestamp(r.at);truth=panel.loc[at];est=records[at]['pair_estimates'];side=['SOXL','SOXS'].index(r.symbol)
        assert r.model in models and r.ready==(r.model in est)
        mature=bool(pd.notna(truth.terminal_release) and truth.terminal_release<end);assert r.mature==mature
        assert pd.Timestamp(r.date)==truth.date and r.clock==truth.clock
        if r.ready:
            e=est[r.model];q=e['probabilities'][side];value=e['values'][side];assert r.outside==bool(e['outside'][side])
            np.testing.assert_allclose(r.value,value,atol=1e-13)
            for k in range(4):np.testing.assert_allclose(getattr(r,f'q{k}'),q[k],atol=1e-13);checks+=1
        else:assert pd.isna(r.value) and not r.outside
        if r.ready and r.mature:
            prefix='L' if side==0 else 'S';actual=float(truth[prefix+'_log']);y=int(truth[prefix+'_gap']);assert r.truth==y
            expected=dict(value=value,actual=actual,squared_error=(value-actual)**2,brier=sum((q[k]-int(y==k))**2 for k in range(4)),log_loss=-math.log(max(q[y],1e-8)))
            expected.update({f'q{k}':q[k] for k in range(4)});expected.update({f'y{k}':int(y==k) for k in range(4)})
            for k,v in expected.items():np.testing.assert_allclose(getattr(r,k),v,atol=1e-13,rtol=1e-10);checks+=1
            for pop in ['both',r.symbol]:buckets.setdefault((r.model,pop),{}).setdefault(r.date,[]).append(expected)
        else:assert pd.isna(r.truth) and pd.isna(r.actual) and pd.isna(r.brier)
    for model,pops in report['paired_candidate_scores'].items():
        for pop,s in pops.items():
            g=table[table.model==model];g=g if pop=='both' else g[g.symbol==pop];known=g[g.ready & g.mature];days=buckets.get((model,pop),{})
            assert s['rows']==len(g) and s['ready_rows']==g.ready.sum() and s['known_rows']==len(known) and s['known_dates']==len(days)
            assert s['unknown_rows']==(~g.mature).sum() and s['outside_rows']==g.outside.sum()
            if not days:assert s['date_equal_means']=={}
            for k,v in s['date_equal_means'].items():np.testing.assert_allclose(v,sum(sum(x[k] for x in a)/len(a) for a in days.values())/len(days),atol=1e-13,rtol=1e-10)
    return checks
