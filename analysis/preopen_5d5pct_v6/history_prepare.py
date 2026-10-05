"""R03 longer-history input comparison; only fully mature prior anchor labels."""
import json
import numpy as np
import pandas as pd
from common import OUT,OLD,DATES,sha,save,write,now
from prepare import daily_features
from research_extensions import repair_action_labels

def mature_features(anchors,days):
    a=anchors[anchors.y.notna()].sort_values(['label_end','day']).copy()
    rows=[]
    for day in sorted(days):
        prior=a[a.label_end.astype(str)<day]
        item={'day':day}
        for n in [20,60]:
            p=prior.tail(n);item[f'm5_rate_{n}']=float(p.y.mean()) if len(p) else np.nan
            item[f'm5_count_{n}']=float(len(p))
        item['m5_stale_sessions']=float(DATES.index(day)-DATES.index(str(prior.iloc[-1].label_end)[:10])) if len(prior) else np.nan
        item['m5_max_label_end']=str(prior.iloc[-1].label_end) if len(prior) else None
        item['m5_missing']=float(prior.empty);rows.append(item)
    return pd.DataFrame(rows)

def main():
    dest=OUT/'R03';paths=sorted((dest/'cache/parts').glob('*.parquet'))
    expected=sorted(p.stem for p in (OLD/'raw').glob('*.parquet'))
    assert sorted(p.stem for p in paths)==expected,'all fixed-registry parts must finish first'
    parts=[];sources=[];mature=[]
    base_meta=json.loads((OUT/'cache/runs/A0_2026-09/meta.json').read_text());base_cols=base_meta['features']
    for i,p in enumerate(paths,1):
        f=pd.read_parquet(p);f['symbol']=f.symbol.astype(str);f['day']=f.day.astype(str)
        raw_path=OUT/'cache/backfill'/p.name
        if not raw_path.exists():raw_path=OLD/'raw'/p.name
        raw=pd.read_parquet(raw_path);d=daily_features(raw);d['symbol']=p.stem
        f=f.drop(columns=[c for c in f if c.startswith('d_')],errors='ignore').merge(d,on=['symbol','day'],how='left',validate='many_to_one')
        anchors=f[f.minute==575][['day','y','label_end']].copy();assert not anchors.day.duplicated().any()
        m=mature_features(anchors,f.day.unique());m['symbol']=p.stem
        f=f.merge(m,on=['symbol','day'],how='left',validate='many_to_one')
        assert set(base_cols)<=set(f.columns)
        parts.append(f);mature.append(anchors.assign(symbol=p.stem))
        sources.append(dict(symbol=p.stem,raw_path=str(raw_path.relative_to(OUT)) if raw_path.is_relative_to(OUT) else str(raw_path),raw_sha256=sha(raw_path),part_sha256=sha(p),rows=len(f),first=f.day.min(),last=f.day.max()))
        if i%20==0:print(json.dumps(dict(stage='R03_join',symbols=i)),flush=True)
    panel=pd.concat(parts,ignore_index=True);panel,actions=repair_action_labels(panel)
    for c in panel.select_dtypes('float'):
        if c not in ['entry','target']:panel[c]=panel[c].astype('float32')
    panel.symbol=panel.symbol.astype('category');panel.day=panel.day.astype('category');save(panel,dest/'cache/panel.parquet')
    old=pd.read_parquet(OUT/'cache/panel.parquet');old,old_actions=repair_action_labels(old);save(old,dest/'cache/H0_panel.parquet')
    save(pd.concat(mature,ignore_index=True),dest/'cache/anchors.parquet')
    write(dest/'coverage.json',dict(at=now(),requested=['2024-10-04','2026-09-30'],actual_features=[min(panel.day.astype(str)),max(panel.day.astype(str))],rows=len(panel),symbols=panel.symbol.nunique(),panel_sha256=sha(dest/'cache/panel.parquet'),H0_panel_sha256=sha(dest/'cache/H0_panel.parquet'),base_features=base_cols,actions=actions,H0_actions=old_actions,sources=sources,protocol_sha256=sha(dest/'PROTOCOL.md'),membership='current_snapshot_retrospective_not_PIT',mature_rule='fixed 09:35 cutoff, 09:40 entry; known label_end strictly earlier than signal day'))
    print(json.dumps(dict(stage='R03_prepared',rows=len(panel))),flush=True)
if __name__=='__main__':main()
