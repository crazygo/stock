"""Post-run fixed-fixture effective slope degrees of freedom; no new strategy."""
import argparse
import gzip
import json
from quadratic_learning import *
from audit_quadratic import design
from threadpoolctl import threadpool_limits


def diagnostic(run):
    run=Path(run);panels={k:pd.read_parquet(run/(k+'_panel.parquet')) for k in PAIRS};rows=[]
    with threadpool_limits(limits=1):
        for day in ['2025-06-03','2025-11-28','2026-01-02','2026-04-30']:
            daily=read(run/'daily'/(day+'.json'));ids=pd.to_datetime(daily['members'][CLOCKS[0]],utc=True);n=len(ids)
            X=np.concatenate([p.loc[ids,FIELDS].to_numpy(float) for p in panels.values()]);Y=np.concatenate([p.loc[ids,['C','D']].to_numpy(float) for p in panels.values()])
            names=np.repeat(list(panels),n);weights=np.tile(2.**(-np.arange(n)[::-1]/126),3)
            for degree in [1,2]:
                meta,Z,group=design(X,Y,names,weights,n,degree);k=len(meta['names']);d=Z.shape[1];mass=meta['pair_mass']
                A=(np.eye(k)[group,:,None]*Z[:,None,:]).reshape(len(Z),k*d);G=A.T@(meta['weights'][:,None]*A)
                P=np.kron(300*np.diag(mass)-200*np.outer(mass,mass),np.eye(d))
                direct=float(np.trace(np.linalg.solve(G+P,G)))
                # Independent spectral identity: df=sum(lambda/(1+lambda)) after whitening by penalty.
                val,vec=np.linalg.eigh(P);invhalf=(vec/np.sqrt(val))@vec.T
                eigen=np.linalg.eigvalsh(invhalf@G@invhalf);spectral=float(np.sum(eigen/(1+eigen)))
                np.testing.assert_allclose(direct,spectral,atol=1e-9,rtol=1e-10)
                pinv=np.linalg.inv(P);j=meta['names'].index('semiconductor');z=Z[group==j];v=meta['weights'][group==j]
                precision_scale=float(np.average(np.einsum('ij,jk,ik->i',z,pinv[j*d:(j+1)*d,j*d:(j+1)*d],z),weights=v))
                rows.append({'day':day,'clock':CLOCKS[0],'degree':degree,'features':d,'real_dates':n,
                             'slope_effective_df':direct,'plus_unpenalized_pair_intercepts':k,
                             'mean_semiconductor_quadratic_form_inverse_penalty':precision_scale})
    out={'scope':'post-run fixed mathematical capacity fixtures, no strategy or parameter change',
         'rows':rows,'trace_and_spectral_checks':len(rows),'results_sha256':sha(run/'results.json'),
         'source_sha256':sha(Path(__file__)),'passed':True,'next_candidate_not_executed':
         'Consider a separately registered penalty normalization for enlarged representation; first verify feature replication invariance. No alpha selection from these or target-month returns.'}
    dump(run/'capacity_diagnostic.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();diagnostic(a.run)
