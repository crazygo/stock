"""Verify every new-clock feature and raw exit before any capital run."""
import argparse
from dense_learning import *
from audit_dense_inputs import input_check,raw_price_check
from audit_features import audit_features
from verify_clock_reliability_v2 import compare_reencoded_labels
from verify_feedback_v2 import terminal_audit
from verify_continuation import raw_exit,key


def pair_targets(pair):
    # JSON null remains unknown, matching the Parquet floating missing value.
    l,s=np.log1p(np.asarray([x.get('net_return',np.nan) for x in pair],dtype=float))
    return np.array([(l+s)/2,(l-s)/2,abs(l-s)/2])


def audit(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json');market=core.load_market();assert market.manifest==manifest['full_market'];cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h
    source=Path(manifest['source_run']);leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==manifest['panel']
    features=pd.read_parquet(inputs/'features.parquet');panel=pd.read_parquet(inputs/'panel.parquet');fp=pd.read_parquet(inputs/'financial_panel.parquet')
    original=pd.read_parquet(source/'features.parquet');pd.testing.assert_frame_equal(features.loc[original.index,original.columns],original)
    oldpanel=pd.read_parquet(source/'financial_panel.parquet');pd.testing.assert_frame_equal(fp.loc[oldpanel.index,oldpanel.columns],oldpanel)
    raw=raw_price_check(market,leaders[0],features);print('new clock price inputs',raw,flush=True)
    factors_check=input_check(inputs,market,leaders[0],features,panel,cutoff);print('factor/prefix check',factors_check,flush=True)
    financial,financial_manifest=load_sources();assert financial_manifest==manifest['financial_panel_source'];new=pd.read_parquet(inputs/'financial_features.parquet')
    fin=audit_features(features,new,read(inputs/'feature_receipts.json'),financial,market.sessions)
    pd.testing.assert_frame_equal(fp[new.columns],new);assert fp.feature_ready.equals(panel.feature_ready&new.financial_ready)
    pd.testing.assert_frame_equal(fp[panel.columns].drop(columns='feature_ready'),panel.drop(columns='feature_ready'))
    checks=terminal_checks=0;terminals={}
    for name,multiple in [('baseline',1),('stress',2)]:
        rows=read(inputs/(name+'_labels.json'));old_at=set(original.index)
        compare_reencoded_labels([x for x in rows if pd.Timestamp(x['decision_at']) in old_at],read(source/(name+'_labels.json')))
        lab=labels(inputs/(name+'_labels.json'));assert len(lab)==2*len(features)
        for at,row in features.iterrows():
            pair=[lab[key(at,s)] for s in core.ETFS]
            if row.clock not in OLD_CLOCKS:
                for item in pair:
                    if 'entry_fill_economic' in item:raw_exit(market,item,None,multiple);checks+=1
            if name=='baseline':
                np.testing.assert_allclose(panel.loc[at,['C','D','M']].to_numpy(float),pair_targets(pair),equal_nan=True)
                if panel.loc[at,'feature_ready']:
                    t=[terminal_audit(market,x) for x in pair]
                    if all(x is not None for x in t):assert panel.loc[at,'terminal_release']==max(t);terminals[at]=max(t)
                    else:assert pd.isna(panel.loc[at,'terminal_release'])
                    terminal_checks+=1
    result={'passed':True,'new_price_inputs':raw,'matched_factors':factors_check,'financial_features':fin,'old_clock_features_panels_labels_exact':True,
        'new_clock_raw_exits':checks,'terminal_pairs':terminal_checks,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'new_capital_outcomes_seen':False}
    dump(inputs/'input_audit.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
