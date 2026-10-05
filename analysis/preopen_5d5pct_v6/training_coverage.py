"""Describe R03 history/registration changes without mutating fitted metadata."""
import json
import pandas as pd
from common import OUT,write,sha,now

def main():
    dest=OUT/'R03';panels={};rows=[]
    for path in sorted((dest/'cache/runs').glob('*/meta.json')):
        meta=json.loads(path.read_text());name=meta['panel_file']
        if name not in panels:
            panel=pd.read_parquet(dest/'cache'/name,columns=['symbol','day','minute','y'])
            panel.symbol=panel.symbol.astype(str);panel.day=panel.day.astype(str);panels[name]=panel
        panel=panels[name];start,end=meta['training']
        sample=panel[(panel.day>=start)&(panel.day<=end)&panel.y.notna()&(panel.minute%15==5)&panel.symbol.isin(meta['registered'])]
        rows.append(dict(arm=meta['arm'],month=meta['month'],panel_file=name,panel_sha256=meta['panel_sha256'],
            training_window=meta['training'],requested_training_days=200,available_panel_days=meta['train_days'],
            mature_training_dates=sample.day.nunique(),sampled_training_rows=len(sample),mature_training_stock_days=sample[['symbol','day']].drop_duplicates().shape[0],
            registered_count=len(meta['registered']),registered=meta['registered']))
    comparisons=[]
    for month in sorted({r['month'] for r in rows}):
        old=next(r for r in rows if r['month']==month and r['arm']=='H0');new=next(r for r in rows if r['month']==month and r['arm']=='H1')
        a=set(old['registered']);b=set(new['registered'])
        comparisons.append(dict(month=month,old_registered=len(a),extended_registered=len(b),common_registered=len(a&b),newly_registered=sorted(b-a),no_longer_registered=sorted(a-b),
            old_mature_stock_days=old['mature_training_stock_days'],extended_mature_stock_days=new['mature_training_stock_days']))
    write(dest/'training_coverage.json',dict(at=now(),status='descriptive_not_effect_claim',models=rows,comparisons=comparisons,
        note='H0 and H1 share protocol repairs and new action audit. Acquisition, mature training coverage and stock registration differ; evaluate probability changes on identical candidate rows, not differing issued cohorts.'))
    print(json.dumps(dict(models=len(rows),comparisons=comparisons)))
if __name__=='__main__':main()
