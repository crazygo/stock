"""Reconstruct visible sector state from immutable source features and volumes."""
from curve_reliability import core,np,pd,read,dump,sha,CLOCKS,STATE_INPUTS
from breadth import PANEL
from verify_continuation import near


def audit_state(run,features):
    from pathlib import Path
    manifest=read(run/'input_manifest.json');source=Path(manifest['source'])
    p=pd.read_parquet(run/'option_panel.parquet');original=pd.read_parquet(source/'financial_panel.parquet')
    pd.testing.assert_frame_equal(p[original.columns],original)
    sm=read(source/'input_manifest.json')
    if sm['new_inputs'] is None:
        volumes=Path(sm['source_run']);hashes=read(core.ROOT/'research/soxl_soxs_14d5pct_v23/evidence/delivery_manifest.json')['cache_outputs_sha256']
    else:
        volumes=source;hashes=read(source/'extension_receipt.json')['files_sha256']
    checks=0;volume_hashes={}
    for clock in CLOCKS:
        f='volumes_'+clock[:5].replace(':','')+'.parquet'
        assert sha(volumes/f)==hashes[f];volume_hashes[f]=hashes[f]
        v=pd.read_parquet(volumes/f)
        for at,row in features[features.clock==clock].iterrows():
            x=p.loc[at,STATE_INPUTS].to_numpy(float);ready=bool(np.isfinite(x).all())
            assert bool(p.loc[at,'state_ready'])==ready
            if not ready:assert pd.isna(p.loc[at,'state_heat']);continue
            past=v[v.index<row.date].iloc[-20:]
            assert len(past)==20 and past.notna().all().all() and v.loc[row.date].notna().all()
            ratio=v.loc[row.date].to_numpy(float)/past.mean().to_numpy(float)
            today=np.array([row['leader_'+s+'_today'] for s in PANEL])
            r5=min(5,max(-5,row.sector_relative_5/(max(row.SOXX_vol20,.0001)*np.sqrt(5))))
            breadth=2*row.breadth_ret5_positive_fraction-1
            flow=sum((ratio>1)&(today>0))/len(PANEL)-sum((ratio>1)&(today<0))/len(PANEL)
            for actual,expected in zip(x,[r5,breadth,flow]):near(actual,expected)
            # expit equivalent of tanh, independent of the producer transform.
            strength=2/(1+np.exp(-2*r5))-1
            expected=(1+(strength+breadth+flow)/3)/2
            near(p.loc[at,'state_heat'],expected);assert 0<=expected<=1;checks+=1
    assert p.loc[original.feature_ready,'state_ready'].all()
    result={'passed':True,'state_rows_reconstructed':checks,'component_equations':checks*3,
        'matched_volume_files_sha256':volume_hashes,'original_eligible_rows_removed':0,
        'uses_future_labels':False,'hot_regime_probability_claimed':False,'panel_sha256':sha(run/'option_panel.parquet')}
    dump(run/'state_feature_audit.json',result);return result
