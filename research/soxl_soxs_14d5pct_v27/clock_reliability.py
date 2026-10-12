"""Separate reliability histories by clock; first eligible opportunity wins."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v26'))
from reliability import core,rows,estimate,decide,day_cutoff,restrict_market,development_gate,np,pd
from run_planning import read,labels,dump,sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v23'))
from clock_training import (CLOCKS,columns,fit_clock,predict,training_dates,baseline_labels,load_panel,PANEL_ROOT,
    clock_volumes,prepare,joint_releases,factors,terminal_panel)
from run_clocks import tag

ROUTES=['score_sequential','score_after_open','premarket_only','score_1005','score_1035','score_1305',
        'point_sequential','cash','double_cost']


def vector(raw):
    return np.array([raw[s] for s in core.ETFS]) if raw else None


def issue(panel,history,at,raw):
    clock=panel.loc[at,'clock'];same=panel[panel.clock==clock]
    # A mixed history is accepted only to make the separation explicit/testable.
    old={t:v for t,v in history.items() if t in same.index}
    ids,feedback=rows(same,old,at);info=estimate(same,old,at,raw,ids,'score')
    return {'feedback':feedback,'estimate':info,'band':decide(info,feedback,True),'point':decide(info,feedback,False)}


def routes(at,day,clock,decision):
    cash={'at':at,'date':day,'ready':True,'action':'CASH','allocation':0.}
    out={}
    only={'premarket_only':CLOCKS[0],'score_1005':CLOCKS[1],'score_1035':CLOCKS[2],'score_1305':CLOCKS[3]}
    for name in ROUTES:
        active=name!='cash' and not(name=='score_after_open' and clock==CLOCKS[0]) and (name not in only or clock==only[name])
        out[name]={'at':at,'date':day,**decision['point' if name=='point_sequential' else 'band']} if active else cash.copy()
    out['double_cost']=out['score_sequential'];return out


def verified_source(p):
    source=core.ROOT/p['development_raw_source'];delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v23/evidence/delivery_manifest.json')
    names=['features.parquet','panel.parquet','baseline_labels.json','stress_labels.json','terminal_receipts.json','verification.json','input_manifest.json']
    for f in names:assert sha(source/f)==delivery['cache_outputs_sha256'][f]
    assert read(source/'verification.json')['passed']
    audit=read(Path(__file__).parent/'evidence/source_forecasts_manifest.json')
    assert audit['passed'] and audit['source_run']==str(source)
    delivery={**delivery,'training_records_sha256':audit['training_records_sha256']}
    return source,delivery,{f:sha(source/f) for f in names}


def prepare_inputs(p,phase,output):
    source,delivery,hashes=verified_source(p);sm=read(source/'input_manifest.json')
    cutoff=pd.Timestamp(p[phase]['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1)
    original=core.load_market();assert original.manifest==sm['full_market'];market=restrict_market(original,cutoff)
    if phase=='development':
        features=pd.read_parquet(source/'features.parquet');panel=pd.read_parquet(source/'panel.parquet')
        base=labels(source/'baseline_labels.json');stress=labels(source/'stress_labels.json');receipts=read(source/'terminal_receipts.json')
        new_inputs=None
    else:
        fs=core.ROOT/p['feature_source'];fm=read(fs/'input_manifest.json')
        assert sha(fs/'features.parquet')==fm['feature_sha256'] and fm['market']==original.manifest
        assert read(fs/'verification.json')['passed']
        leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==fm['panel']==sm['panel']
        features=pd.read_parquet(fs/'features.parquet');features=features[features.index<cutoff].copy()
        base=baseline_labels(market,features);stress=baseline_labels(market,features,2);panels=[];receipts=[]
        for c in CLOCKS:
            f=features[features.clock==c];vol,audits=clock_volumes(leaders[0],market.sessions,market.calendar,cutoff,c)
            vol.to_parquet(output/('volumes_'+c[:5].replace(':','')+'.parquet'))
            part=prepare(f,base,joint_releases(f,base),factors(f,vol));part,r=terminal_panel(market,part,base)
            panels.append(part);receipts.extend(r)
        panel=pd.concat(panels).sort_index()
        new_inputs={'source':str(fs),'features_sha256':sha(fs/'features.parquet'),'panel':leaders[2]}
    # prepare/terminal_panel does not carry the clock; join it explicitly.
    panel=panel.copy();panel['clock']=features.clock
    features.to_parquet(output/'features.parquet');panel.to_parquet(output/'panel.parquet')
    dump(output/'baseline_labels.json',list(base.values()));dump(output/'stress_labels.json',list(stress.values()))
    dump(output/'terminal_receipts.json',receipts)
    manifest={'source_run':str(source),'source_files_sha256':hashes,'source_training_sha256':delivery['training_records_sha256'],
        'observation_cutoff_exclusive':cutoff,'full_market':original.manifest,'new_inputs':new_inputs}
    return market,features,panel,base,stress,manifest


def base_forecast(panel,at,manifest):
    day=panel.loc[at,'date'];clock=panel.loc[at,'clock'];old_source=Path(manifest['source_run'])
    if day<pd.Timestamp('2026-05-01'):
        filename=tag(at)+'.json';path=old_source/'training'/filename
        assert sha(path)==manifest['source_training_sha256'][filename]
        rec=read(path);raw=rec['variants']['context']['forecasts']['early']['event']['raw']
        return vector(raw),{'kind':'frozen_v23','source_record':str(path),'sha256':sha(path),'raw':raw},None
    # An extension of the same raw learner for this new experiment. It does not
    # run or grant acceptance to the failed v23 capital policy.
    same=panel[panel.clock==clock];dates,fb=training_dates(same,day_cutoff(at))
    model,info=fit_clock(same,dates['early'],columns(clock,'context'));raw=predict(model,same.loc[[at]])
    return vector(raw),{'kind':'new_daily_raw','raw':raw,'fit':info,'feedback':fb},model
