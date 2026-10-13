"""Independent raw-announcement reconstruction and exhaustive historical checks."""
from datetime import datetime, timedelta, time
from functools import lru_cache
import importlib.util
import hashlib
import json
from state_calibration import *


@lru_cache(maxsize=4)
def raw_inputs(serialized):
    cat=json.loads(serialized)
    return read(ROOT/cat['facts_path'])['announcements'],read(ROOT/cat['coverage_path'])['issuers']


def reference_raw(announcements,coverage,at,expiry):
    at=pd.Timestamp(at).tz_convert(core.ET);expiry=pd.Timestamp(expiry).tz_convert(core.ET)
    symbols=['NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC']
    assert at<=expiry and (expiry.date()-at.date()).days<=13
    missing=[]
    for symbol in symbols:
        c=coverage.get(symbol,{})
        if not (c.get('body_catalogue_verified') is True and c.get('pending_announcements_at_start_resolved') is True
                and c.get('publication_from','9999')<=str(at.date())<=c.get('publication_through','0000')):
            missing.append(symbol)
    latest={}
    for a in announcements:
        if a['published_at']:
            day=pd.Timestamp(a['published_at']).tz_convert(core.ET).date()+timedelta(days=1)
        else:
            day=datetime.strptime(a['published_date'],'%Y-%m-%d').date()+timedelta(days=2)
        available=pd.Timestamp(str(day),tz=core.ET)
        if available>at:continue
        for e in a['events']:
            key=a['activity_key']+'_'+e['kind']
            if key not in latest or latest[key][0]<available:
                latest[key]=(available,a,e)
    future=set();recent=set();ids=[]
    for _,a,e in latest.values():
        ids.append(hashlib.sha256((a['source_url']+'#'+e['kind']).encode()).hexdigest())
        if e['kind'] not in ['earnings_release','earnings_call']:continue
        day=datetime.strptime(e['event_date'],'%Y-%m-%d').date()
        if at.date()<=day<=expiry.date():future.add((a['symbol'],e['event_date']))
        if 1<=(at.date()-day).days<=14:recent.add((a['symbol'],e['event_date']))
    fdist=[(datetime.strptime(d,'%Y-%m-%d').date()-at.date()).days for _,d in future]
    rdist=[(at.date()-datetime.strptime(d,'%Y-%m-%d').date()).days for _,d in recent]
    values=None if missing else [len(future),1-min(fdist)/14 if fdist else 0.,len(recent),1-min(rdist)/15 if rdist else 0.]
    return dict(known=not missing,values=values,missing_coverage=missing,
        future_days=[list(x) for x in sorted(future)],previous_days=[list(x) for x in sorted(recent)],
        visible_record_ids=sorted(ids),actual_event_occurrence_verified=False,exact_intraday_event_window=False)


def reference(catalog,at,expiry):
    return reference_raw(*raw_inputs(json.dumps(catalog,sort_keys=True)),at,expiry)


def audit():
    catalog=read(CATALOG)
    for name,digest in catalog['source_files_sha256'].items():assert sha(ROOT/name)==digest,name
    source=ROOT/'research/soxl_soxs_14d5pct_v86'
    manifest=read(source/'delivery_manifest.json')
    for item in manifest['files']:
        if item['path'].startswith('research/soxl_soxs_14d5pct_v86/'):
            assert sha(ROOT/item['path'])==item['sha256'],item['path']
    spec=importlib.util.spec_from_file_location('v86_source_reaudit',source/'audit_calendar.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    sourceaudit=mod.audit()
    assert sourceaudit==read(source/'evidence/source_audit.json')==read(SOURCE/'evidence/source_audit.json')
    panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet');market=extended_market();prepared=prepare_panel(panel,market)
    rows=[]
    for at,r in prepared.iterrows():
        dates=[d for d in market.sessions if r.date<=d<=r.date+pd.Timedelta(days=13)]
        expiry=pd.Timestamp(market.calendar[max(dates)]['close_at'])
        expected=reference(catalog,at,expiry);actual=event_features(catalog,at.isoformat(),expiry.isoformat())
        assert expected==actual and expected['known'],at
        np.testing.assert_allclose([r.event_future_count,r.event_future_proximity,r.event_recent_count,r.event_recent_proximity],expected['values'],atol=1e-14,rtol=0)
        rows.append(dict(at=at,date=r.date,future=expected['values'][0],recent=expected['values'][2]))
    table=pd.DataFrame(rows);dev=table[table.date.between('2026-01-01','2026-04-30')]
    result=dict(passed=True,calendar_sha256=sha(CATALOG),source_audit_sha256=sha(SOURCE/'evidence/source_audit.json'),
        announcements=sourceaudit['announcements'],source_fact_versions_checked=sourceaudit['event_versions'],
        source_hashes_checked=len(sourceaudit['source_hashes']),query_rows=len(rows),dates=panel.date.nunique(),
        first_cutoff=str(panel.index.min()),last_cutoff=str(panel.index.max()),missing_queries=0,
        future_positive=int((table.future>0).sum()),recent_positive=int((table.recent>0).sum()),
        development=dict(rows=len(dev),future_positive=int((dev.future>0).sum()),recent_positive=int((dev.recent>0).sum())),
        source_sha256=sha(Path(__file__)),new_capital_outcomes_seen=False,
        scope='Reviewed fixed-panel quarterly plans; no results content, actual occurrence, complete historical receipt or all revisions claim.')
    dump(SOURCE/'evidence/calendar_verification.json',result);print(result,flush=True)


if __name__=='__main__':audit()
