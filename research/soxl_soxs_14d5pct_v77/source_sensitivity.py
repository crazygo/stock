"""Descriptive disputed-low sensitivity; never correct raw prices or primary models."""
import argparse
from history_learning import *
from audit_inputs import reference_features


def check(inputs,output):
    inputs=Path(inputs);market=extended_market()
    original={s:pd.read_parquet(inputs/(s+'_prefix.parquet')) for s in sorted({s for p in PAIRS.values() for s in p})}
    altered={s:p.copy() for s,p in original.items()};at=pd.Timestamp('2023-10-30 10:30',tz=core.ET).tz_convert('UTC')
    bar=market.bars['SPY'].loc[at];assert bar.low==408.91
    # 412.22 is the disagreeing daily-source low, not an asserted true intraday print.
    delta=(412.22-float(bar.low))*float(bar.volume)/3
    same=(altered['SPY'].date==pd.Timestamp('2023-10-30'))&(altered['SPY'].index>=at+pd.Timedelta(seconds=30))
    altered['SPY'].loc[same,'amount']+=delta
    dates=market.sessions[market.sessions>=pd.Timestamp('2023-10-30')][:31]
    ticks=original['SPY'].index[original['SPY'].date.isin(dates)];changed=[];checks=0
    keep=[i for i in range(18) if i not in range(11,16)]
    for pair in PAIRS:
        for t in ticks:
            a=reference_features(market,original,pair,t);b=reference_features(market,altered,pair,t)
            assert (a is None)==(b is None)
            if a is None:continue
            a=np.array(a);b=np.array(b);checks+=1
            np.testing.assert_array_equal(a[keep],b[keep])
            difference=np.abs(a-b);ids=np.flatnonzero(difference>0)
            assert set(ids)<=set(range(11,16))
            if len(ids):changed.append({'at':t,'pair':pair,'fields':[FIELDS[i] for i in ids],
                'max_absolute_feature_difference':float(difference.max())})
    result={'passed':True,'stage':'counterfactual_input_sensitivity_only','source_low':408.91,'alternative_daily_low':412.22,
        'alternative_is_not_verified_true_intraday_low':True,'changed_prefixes':int(same.sum()),'feature_rows_checked':checks,
        'changed_full_feature_rows':len(changed),'changed_rows':changed,'price_only_coordinates_exactly_invariant':True,
        'readiness_preserved':True,'changes_prices_labels_models_primary_or_gate':False,'source_sha256':sha(Path(__file__)),
        'input_manifest_sha256':sha(inputs/'input_manifest.json')}
    dump(output,result);print({k:v for k,v in result.items() if k!='changed_rows'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();check(a.inputs,a.output)
