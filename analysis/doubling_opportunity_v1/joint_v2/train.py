#!/usr/bin/env python3
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).parent;PARENT=HERE.parent
sys.path.insert(0,str(HERE))
from model import forward_backtest
sys.path.insert(0,str(PARENT))
from data import CACHE
import pandas as pd
cfg=json.loads((PARENT/'config.json').read_text());d=pd.read_parquet(CACHE/'dataset.parquet')
key=json.loads((CACHE/'dataset_key.json').read_text())['key']
out=HERE/'backtests'/key[:16]
pred,audits,summary=forward_backtest(d,cfg,out)
(out/'lineage.json').write_text(json.dumps({'dataset_key':key,'config':cfg,'scope':'joint first-touch price control; does not establish good-company or business-catalyst joint effect','protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
(HERE/'latest_backtest.json').write_text(json.dumps({'path':str(out.resolve()),'dataset_key':key}))
print(json.dumps([{k:r[k] for k in ['horizon','algorithm','signals','tp','fp','unknown','precision','brier','historical_numeric_gate']} for r in summary],indent=2),flush=True)
