"""Fixed real-model independent Newton checks before strategy capital results."""
import argparse
from threadpoolctl import threadpool_limits
from class_learning import *
from reference_class import arrays,setup,evaluate,newton
from verify_transfer import reference_parents,canonical


def preflight(inputs,output):
    inputs=Path(inputs);market=extended_market();panels={k:pd.read_parquet(inputs/(k+'_panel.parquet')) for k in PAIRS}
    reg=matched_registry(panels,market);records=[]
    with threadpool_limits(limits=1):
        for date in ['2024-01-02','2025-06-03','2026-04-30']:
            at=pd.Timestamp(date+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC');parents,fb=mature_prefix(reg,at)
            rp,rf=reference_parents(panels,market,at);assert canonical(parents)==canonical(rp) and canonical(fb)==canonical(rf)
            for c in CLOCKS:
                models,ids=fit_clock(panels,parents,c)
                for mode,model in models.items():
                    X,y,values,groups,dw,ess=arrays(panels,ids,FIELDS,mode);ref=setup(X,y,values,groups,dw,CLASSES[mode])
                    for k in ['center','scale','weights','prior','payoff_means','pooled_prior','pooled_payoff_sum','own_class_mass','own_payoff_sum']:
                        np.testing.assert_allclose(model[k],ref[k],atol=1e-10,rtol=1e-10)
                    B,iterations=newton(ref);np.testing.assert_allclose(model['coefficients'],B,atol=1e-7,rtol=1e-6)
                    value,g=evaluate(model['coefficients'],ref);assert np.max(abs(g))<=1e-6
                    records.append({'date':date,'clock':c,'mode':mode,'dates':len(ids),'date_ess':ess,
                        'newton_iterations':iterations,'gradient_inf':float(np.max(abs(g))),'coefficient_max_difference':float(np.max(abs(model['coefficients']-B)))})
            print('Real preflight',date,flush=True)
    out={'passed':True,'model_checks':len(records),'records':records,'input_manifest_sha256':sha(inputs/'input_manifest.json'),
        'source_sha256':sha(Path(__file__)),'strategy_capital_outcomes_seen':False}
    dump(output,out);print({'passed':True,'model_checks':len(records)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();preflight(a.inputs,a.output)
