"""Post-run mathematical attribution only; no new fit, action, or capital result."""
import argparse
from normalized_learning import *
from audit_quadratic import design
from threadpoolctl import threadpool_limits


def audit(run):
    run=Path(run);result_sha=sha(run/'results.json');prediction_sha=sha(run/'predictions.json')
    panels={k:pd.read_parquet(run/(k+'_panel.parquet')) for k in PAIRS};rows=[]
    with threadpool_limits(limits=1):
        for day in ['2025-06-03','2025-11-28','2026-01-02','2026-04-30']:
            ids=pd.to_datetime(read(run/'daily'/(day+'.json'))['members'][CLOCKS[0]],utc=True);n=len(ids)
            X=np.concatenate([p.loc[ids,FIELDS].to_numpy(float) for p in panels.values()])
            Y=np.concatenate([p.loc[ids,['C','D']].to_numpy(float) for p in panels.values()])
            names=np.repeat(list(panels),n);w=np.tile(2.**(-np.arange(n)[::-1]/126),3)
            meta,Z,groups=design(X,Y,names,w,n);mass=meta['pair_mass'];j=meta['names'].index('semiconductor');d=Z.shape[1]
            for name,multiplier in [('old_quadratic',1.),('normalized',10.5)]:
                between=multiplier*(300*np.diag(mass)-200*np.outer(mass,mass))
                full=np.linalg.inv(np.kron(between,np.eye(d)));own=full[j*d:(j+1)*d,j*d:(j+1)*d]
                state=Z[groups==j];weights=meta['weights'][groups==j]
                values={}
                for block,a,b in [('original_fields',0,18),('quadratic_terms',18,189)]:
                    direct=np.average(np.einsum('ij,jk,ik->i',state[:,a:b],own[a:b,a:b],state[:,a:b]),weights=weights)
                    # Independent factorization: group inverse diagonal times standardized feature count.
                    expected=np.linalg.inv(between)[j,j]*(b-a)
                    np.testing.assert_allclose(direct,expected,atol=1e-12,rtol=1e-12);values[block]=float(direct)
                rows.append({'day':day,'variant':name,'mean_inverse_penalty_by_block':values,
                             'original_share':values['original_fields']/sum(values.values())})
    assert sha(run/'results.json')==result_sha and sha(run/'predictions.json')==prediction_sha
    receipt={'passed':True,'stage':'post_run_mathematical_attribution_only','rows':rows,'independent_block_checks':16,
             'results_sha256':result_sha,'predictions_sha256':prediction_sha,'source_sha256':sha(Path(__file__)),
             'interpretation':'Uniform width scaling also multiplies penalties on original18 fields by10.5. Under this representation their share of the semiconductor training-state inverse-penalty scale is18/189. This is not causal proof of trading loss or evidence that a different penalty would pass.',
             'new_strategy_evaluated':False}
    dump(run/'penalty_blocks.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();audit(a.run)
