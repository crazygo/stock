"""Build an isolated R02 panel; never mutates R00 feature rows or labels."""
import json
import numpy as np
import pandas as pd
from common import OUT,OLD,save,write,sha,now
from research_extensions import peers_exact,event_features

def main():
 dest=OUT/'R02';source=OUT/'cache/panel.parquet';panel=pd.read_parquet(source);panel.symbol=panel.symbol.astype(str);panel.day=panel.day.astype(str)
 labels=pd.read_parquet(OLD/'horizon_v1/labels_5d5pct.parquet',columns=['entry','target']);assert np.array_equal(panel._row_id,np.arange(len(panel)))
 panel['entry']=labels.entry.to_numpy();panel['target']=labels.target.to_numpy()
 panel=peers_exact(panel);panel,events=event_features(panel);panel=panel.sort_values('_row_id').reset_index(drop=True)
 assert np.array_equal(panel._row_id,np.arange(len(panel)))
 panel.symbol=panel.symbol.astype('category');panel.day=panel.day.astype('category')
 save(panel,dest/'cache/panel.parquet');write(dest/'coverage.json',dict(at=now(),requested=['2024-10-04','2026-09-30'],actual_features=[min(panel.day.astype(str)),max(panel.day.astype(str))],rows=len(panel),symbols=panel.symbol.nunique(),events=events,
  panel_sha256=sha(dest/'cache/panel.parquet'),source_sha256=sha(source),sec_events_sha256=sha(OUT/'cache/sec/events.json'),protocol_sha256=sha(dest/'PROTOCOL.md'),membership='current_snapshot_retrospective_not_PIT',price_basis='NONE'))
 print(json.dumps(dict(phase='R02_prepared',events=events,rows=len(panel))),flush=True)
if __name__=='__main__':main()
