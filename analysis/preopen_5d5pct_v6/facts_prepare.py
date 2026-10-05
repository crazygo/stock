import json
import numpy as np
import pandas as pd
from common import OUT,OLD,write,save,sha,now
from research_extensions import event_features,quarter_features,repair_action_labels

def main():
 dest=OUT/'R04';source=OUT/'R02/cache/panel.parquet';panel=pd.read_parquet(source);panel.symbol=panel.symbol.astype(str);panel.day=panel.day.astype(str)
 panel,actions=repair_action_labels(panel);panel,events=event_features(panel,latency_seconds=300);panel,facts=quarter_features(panel,latency_seconds=300)
 for p,newname in [(OUT/'cache/sec/events.json','sec_events_source.json'),(OUT/'sec_audit.json','sec_audit_source.json'),(OUT/'cache/sec_facts/events.json','sec_facts_source.json')]:
  (dest/'cache'/newname).write_bytes(p.read_bytes())
 panel.symbol=panel.symbol.astype('category');panel.day=panel.day.astype('category');save(panel,dest/'cache/panel.parquet')
 write(dest/'coverage.json',dict(at=now(),requested=['2024-10-04','2026-09-30'],actual_features=[min(panel.day.astype(str)),max(panel.day.astype(str))],rows=len(panel),symbols=panel.symbol.nunique(),events=events,facts=facts,actions=actions,
  panel_sha256=sha(dest/'cache/panel.parquet'),source_sha256=sha(source),sec_events_sha256=sha(dest/'cache/sec_events_source.json'),sec_facts_sha256=sha(dest/'cache/sec_facts_source.json'),protocol_sha256=sha(dest/'PROTOCOL.md'),membership='current_snapshot_retrospective_not_PIT'))
 print(json.dumps(dict(phase='R04_prepared',events=events,facts=facts,actions=actions)),flush=True)
if __name__=='__main__':main()
