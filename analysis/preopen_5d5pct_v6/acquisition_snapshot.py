"""Summarize completed per-symbol downloads without querying any provider."""
import json
from collections import Counter
from common import OUT,write,now
from acquire_history import universe

def main():
    members=universe();records=[]
    for member in members:
        path=OUT/'cache/acquired'/f"{member['symbol']}.json"
        row=json.loads(path.read_text()) if path.exists() else dict(**member,status='queued_free_history')
        if row.get('first'):
            row['requested_start_covered']=row['first'][:10]<='2024-10-04'
            row['requested_end_covered']=row['last'][:10]>='2026-10-02'
            row['prefix_status']='bound_covered_not_quality_approved' if row['requested_start_covered'] else 'shorter_actual_history_or_IPO_not_automatically_complete'
        records.append(row)
    write(OUT/'history_acquisition.json',dict(at=now(),requested=['2024-10-04','2026-10-02'],records=records,
        complete=all(r['status'] in ['local_available','r2_available','futu_available'] for r in records),
        note='Completed atomic symbol metadata only; queued/unavailable/partial retained. Date bounds do not prove full-session or scoreable coverage.'))
    print(json.dumps(dict(symbols=len(records),counts=dict(Counter(r['status'] for r in records)))))
if __name__=='__main__':main()
