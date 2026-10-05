"""Produce genuine time-forward daily opportunity scores, then an isolated R01 panel."""
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from common import OUT,OLD,DATES,write,save,sha,now

def main():
    dest=OUT/'R01';p=OUT/'cache/panel.parquet';source=pd.read_parquet(p)
    source['symbol']=source.symbol.astype(str);source['day']=source.day.astype(str)
    labels=pd.read_parquet(OLD/'horizon_v1/labels_5d5pct.parquet',columns=['entry','target'])
    assert np.array_equal(source._row_id,np.arange(len(source)))
    source['entry']=labels.entry.to_numpy();source['target']=labels.target.to_numpy()
    cols=[c for c in source if c.startswith('d_')];daily=source[source.minute==575][['symbol','day','label_end','y']+cols].copy()
    # Prediction eligibility cannot depend on the future existence of a 09:35 bar.
    inputs=source[['symbol','day']+cols].drop_duplicates(['symbol','day'])
    dates=[d for d in DATES if source.day.min()<=d<=source.day.max()];scores=[];blocks=[]
    for block,start in enumerate(range(0,len(dates),20)):
        blockdays=dates[start:start+20];first=blockdays[0]
        pastdays=dates[max(0,start-210):start]
        train=daily[daily.day.isin(pastdays)&daily.y.notna()&(daily.label_end<first)].copy()
        trainingdays=sorted(train.day.unique())[-200:];train=train[train.day.isin(trainingdays)]
        counts=train.groupby('symbol').day.nunique();registered=sorted(counts[counts>=60].index);train=train[train.symbol.isin(registered)]
        test=inputs[inputs.day.isin(blockdays)&inputs.symbol.isin(registered)].copy()
        if train.empty or test.empty or train.y.nunique()<2:
            blocks.append(dict(block=block,start=first,end=blockdays[-1],status='insufficient_prior_history',registered=len(registered)));continue
        assert train.label_end.max()<first
        model=LGBMClassifier(n_estimators=180,max_depth=4,num_leaves=15,min_child_samples=80,reg_lambda=10,learning_rate=.04,random_state=1005,n_jobs=1,verbosity=-1)
        model.fit(train[cols],train.y)
        test['opp_oof_raw']=model.predict_proba(test[cols])[:,1]
        path=dest/'models'/f'daily_opportunity_block_{block:03d}.pkl';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(pickle.dumps(model))
        scores.append(test[['symbol','day','opp_oof_raw']]);blocks.append(dict(block=block,start=first,end=blockdays[-1],status='fitted',training=[trainingdays[0],trainingdays[-1]],train_days=len(trainingdays),max_training_label_end=train.label_end.max(),registered=registered,model_sha256=sha(path),features=cols,rows=len(test)))
        print(json.dumps(dict(block=block,start=first,training=len(train),predictions=len(test))),flush=True)
    score=pd.concat(scores,ignore_index=True);assert not score.duplicated(['symbol','day']).any()
    panel=source.merge(score,on=['symbol','day'],how='inner',validate='many_to_one',sort=False)
    panel['_row_id']=np.arange(len(panel));panel.symbol=panel.symbol.astype('category');panel.day=panel.day.astype('category')
    save(panel,dest/'cache/panel.parquet');save(score,dest/'cache/opportunity_scores.parquet')
    write(dest/'upstream_provenance.json',dict(at=now(),blocks=blocks,labels='09:40 Open*1.001; 5d5pct',score_is_calibrated=False,source_sha256=sha(p),protocol_sha256=sha(dest/'PROTOCOL.md')))
    write(dest/'coverage.json',dict(at=now(),rows=len(panel),source_rows=len(source),stock_days=score.shape[0],symbols=panel.symbol.nunique(),actual_features=[min(panel.day.astype(str)),max(panel.day.astype(str))],
        panel_sha256=sha(dest/'cache/panel.parquet'),source_sha256=sha(p),protocol_sha256=sha(dest/'PROTOCOL.md'),exclusions='upstream requires 60 mature training stock-days; identical downstream samples for all four arms',membership='current_snapshot_retrospective_not_PIT'))
    print(json.dumps(dict(phase='staged_prepared',rows=len(panel),stock_days=len(score))),flush=True)
if __name__=='__main__':main()
