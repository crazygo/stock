from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar
CALENDAR=calendar('2024-08-01','2026-12-31')
SESSIONS={x['session_date']:x for x in CALENDAR['sessions']}
ET=ZoneInfo('America/New_York')
ROUTES=['hazard_linear','relative_flow','recovery_forest']
NAMES=dict(hazard_linear='剩余时间与单股基础风险',relative_flow='相关股分钟量价',recovery_forest='回撤恢复与量价效率')
ALGORITHMS=dict(hazard_linear='LogisticRegression',relative_flow='LightGBM',recovery_forest='ExtraTrees')
THRESHOLDS=[.60,.65,.70,.75,.80,.85,.90]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,(pd.Timestamp,datetime)):return x.isoformat()
 return x
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def save(f,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.writing.parquet');f.to_parquet(t,index=False,compression='zstd',compression_level=7);t.replace(p)
def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/2/n)/d;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/d
 return [float(c-h),float(c+h)]
