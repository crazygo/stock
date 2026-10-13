"""Fixed actual dates: null model equals the frozen parent own intercept."""
import gzip,json
from mean_learning import *


def preflight():
    parent_sha=verify_parent();panels={k:pd.read_parquet(PARENT/(k+'_panel.parquet')) for k in PAIRS}
    market=load_market();reg=matched_registry(panels,market);fixtures=[]
    for day in ['2025-06-03','2025-11-28','2026-01-02','2026-04-30']:
        at=pd.Timestamp(day+' 09:25:30',tz=core.ET).tz_convert('UTC');parents,fb=mature_prefix(reg,at)
        with gzip.open(PARENT/'models'/(day+'.json.gz'),'rt') as f:old=json.load(f)
        for clock in CLOCKS:
            model,ids=fit_clock(panels,parents,clock);y=panels['semiconductor'].loc[ids,['C','D']].to_numpy();n=len(ids)
            w=np.array([2**((i-n+1)/126) for i in range(n)]);expected=sum(w[i]*y[i] for i in range(n))/sum(w)
            np.testing.assert_allclose(model['mean_cd'],expected,atol=1e-12,rtol=1e-12)
            j=old[clock]['names'].index('semiconductor');np.testing.assert_allclose(model['mean_cd'],old[clock]['intercepts'][j],atol=1e-12,rtol=1e-12)
            fixtures.append({'day':day,'clock':clock,'real_dates':n,'parent_intercept_equal':True})
    receipt={'passed':True,'models':len(fixtures),'fixtures':fixtures,'parent_delivery_sha256':parent_sha,
             'strategy_outcomes_seen':False,'source_sha256':sha(Path(__file__))}
    dump(SOURCE/'evidence/model_preflight.json',receipt);print({'passed':True,'models':len(fixtures)},flush=True)


if __name__=='__main__':preflight()
