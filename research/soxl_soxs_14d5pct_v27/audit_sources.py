"""Bind reused v23 raw forecasts to independently rechecked source equations."""
import gzip,pickle
from pathlib import Path
from clock_reliability import core,pd,np,read,sha,dump,CLOCKS,columns,day_cutoff,tag
from audit_models import model_equations
from verify_continuation import near


def audit():
    source=core.ROOT/'.cache/soxl_soxs_14d5pct_v23/runs/clocks_20261012'
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v23/evidence/delivery_manifest.json')
    for f in ['panel.parquet','features.parquet','verification.json','terminal_receipts.json','registration.json']:
        assert sha(source/f)==delivery['cache_outputs_sha256'][f]
    reg=read(source/'registration.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    panel=pd.read_parquet(source/'panel.parquet');features=pd.read_parquet(source/'features.parquet')
    records={};models={};equations=0
    for i,(at,row) in enumerate(features.iterrows()):
        filename=tag(at)+'.json';path=source/'training'/filename;r=read(path)['variants']['context']
        part=panel.loc[features[features.clock==row.clock].index];dc=day_cutoff(at)
        past=part[part.feature_ready&(part.date<row.date)&(part.index<dc)];ids=[]
        for t,x in past.iterrows():
            if pd.isna(x.terminal_release) or x.terminal_release>=dc:break
            ids.append(t)
        info=r['fits']['early'];assert list(pd.to_datetime(info['members'],utc=True))==ids
        modelpath=source/'models'/(tag(at)+'.pkl.gz')
        with gzip.open(modelpath,'rb') as f:model=pickle.load(f)['context']['early']
        raw,count=model_equations(part,ids,at,model,info,columns(row.clock,'context'));equations+=count
        stored=r['forecasts']['early']['event']['raw'];assert set(raw)==set(stored)
        for k in raw:near(raw[k],stored[k])
        records[filename]=sha(path);models[modelpath.name]=sha(modelpath)
    out={'passed':True,'source_run':str(source),'rows':len(records),'independent_raw_model_equations':equations,
        'training_records_sha256':records,'models_sha256':models,'new_v27_outcomes_seen':False,
        'source_input_hashes':{f:delivery['cache_outputs_sha256'][f] for f in ['panel.parquet','features.parquet','verification.json','terminal_receipts.json','registration.json']}}
    dump(Path(__file__).parent/'evidence/source_forecasts_manifest.json',out)
    print({'source_forecasts':len(records),'independent_raw_model_equations':equations,'new_outcomes_seen':False},flush=True)


if __name__=='__main__':audit()
