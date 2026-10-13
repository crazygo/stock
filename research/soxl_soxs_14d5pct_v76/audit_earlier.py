"""Independent raw lineage, causal clocks, overlaps and cross-provider diagnostics.

No predictions, strategy results or model fitting are used here.
"""
from pathlib import Path
from decimal import Decimal
import datetime, hashlib, json, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
CACHE = ROOT/'.cache/soxl_soxs_14d5pct_v76'
sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar
P = json.loads((SOURCE/'DATA_PROTOCOL.json').read_text())
COLS = ['open','high','low','close','volume','turnover']
CLOCKS = ['09:25:30','10:05:30','10:35:30','11:05:30','12:05:30','13:05:30','14:05:30','15:05:30']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def action_audit(symbol):
    original = CACHE/'actions'/f'{symbol}.parquet'
    frame = pd.read_parquet(original)
    frame = frame[frame.ex_div_date.between(P['extension_start'],P['overlap_end'])]
    other = read(CACHE/'ibkr'/f'{symbol}.json')['corp_actions']
    lookup = {}
    for a in other:
        day = pd.Timestamp(a['date']).strftime('%Y-%m-%d')
        if P['extension_start'] <= day <= P['overlap_end']:
            assert (day,a['type']) not in lookup
            lookup[(day,a['type'])] = a
    checks = []; observed=set()
    for row in frame.itertuples():
        for field in ['spin_off_ratio','allotment_ratio','stk_spo_ratio','per_share_trans_ratio','per_share_div_ratio']:
            value=float(getattr(row,field));assert not np.isfinite(value) or value==0, (symbol,field)
        for field,kind in [('split_ratio','Splits'),('per_cash_div','CashDividends')]:
            x=float(getattr(row,field))
            if kind=='CashDividends':
                special=float(row.special_dividend)
                x=(x if np.isfinite(x) else 0)+(special if np.isfinite(special) else 0)
            if not np.isfinite(x) or x == (1 if kind=='Splits' else 0):continue
            key=(row.ex_div_date,kind);observed.add(key); y=lookup.get(key)
            ok=y is not None
            if ok and kind=='Splits': ok=abs(x-1/float(y['value']))<1e-12
            if ok and kind=='CashDividends':
                ok=abs(Decimal(str(x))-Decimal(y['value']))<=Decimal('.000005')
                ok=ok and bool(y.get('payment_date')) and pd.Timestamp(y['payment_date'])>=pd.Timestamp(row.ex_div_date)
            checks.append(dict(date=row.ex_div_date,kind=kind,passed=bool(ok),futu=x,ibkr=y))
    assert observed == set(lookup), (symbol,'unmatched action dates')
    return dict(symbol=symbol,passed=all(x['passed'] for x in checks),events=checks,
                futu_sha256=sha(original),ibkr_sha256=sha(CACHE/'ibkr'/f'{symbol}.json'))


def raw_lineage(symbol, bars):
    data=CACHE/'minutes'/symbol; parts=[];sources=[]
    if symbol=='QQQ':
        for item in read(data/'local_parts.json'):
            path=ROOT/item['capture'];assert sha(path)==item['sha256'];parts.append(pd.read_parquet(path))
            sources.append(dict(path=item['capture'],sha256=sha(path)))
        # Independently compare legacy monthly files to their actual original pages.
        legacy=[]
        for year in ['2023','2024']:
            folders=list((ROOT/'.cache/model_history_20260927/requests/QQQ').glob(year+'*/attempt_*'))
            assert len(folders)==1
            for path in sorted(folders[0].glob('raw_*.parquet')):
                legacy.append(pd.read_parquet(path));sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        raw=pd.concat(legacy).drop_duplicates('time_key',keep='last').set_index('time_key')
        local=pd.concat(parts).drop_duplicates('time_key',keep='first').set_index('time_key')
        common=local.index.intersection(raw.index)
        assert len(common)==len(local)
        np.testing.assert_array_equal(raw.loc[common,COLS].to_numpy(float),local.loc[common,COLS].to_numpy(float))
    else:
        for line in (data/'acquisition.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['source']=='r2': assert 'error' not in r and not r['found']
            if r['source']=='OpenD' and 'sha256' in r:
                path=data/'captured/OpenD'/symbol/f"{r['start']}_{r['end']}_{r['page']:04d}.parquet"
                assert sha(path)==r['sha256']; f=pd.read_parquet(path)
                assert f.code.eq('US.'+symbol).all();parts.append(f);sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    raw=pd.concat(parts,ignore_index=True)
    duplicate=raw[raw.time_key.duplicated(keep=False)]
    if len(duplicate): assert duplicate.groupby('time_key')[COLS].nunique().max().max()==1
    raw=raw.drop_duplicates('time_key',keep='last').copy()
    t=pd.to_datetime(raw.time_key)
    dates=t.dt.normalize()+pd.to_timedelta((t.dt.hour*60+t.dt.minute>1200).astype(int),unit='D')
    raw=raw[dates.between(P['extension_start'],P['overlap_end'])].set_index('time_key').sort_index()
    f=bars.set_index('time_key').sort_index();assert raw.index.equals(f.index)
    np.testing.assert_array_equal(raw[COLS].to_numpy(float),f[COLS].to_numpy(float))
    return dict(raw_values_verified=len(raw)*len(COLS),sources=sources,duplicate_rows=len(duplicate))


def audit_symbol(symbol):
    path=CACHE/'minutes'/symbol/'bars'/f'{symbol}.parquet'
    bars=pd.read_parquet(path);lineage=raw_lineage(symbol,bars)
    original_end=pd.to_datetime(bars.time_key).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    for col,expected in [('end_at',original_end),('start_at',original_end-pd.Timedelta(minutes=5)),('available_at',original_end+pd.Timedelta(seconds=1))]:
        assert (pd.to_datetime(bars[col],utc=True)==expected).all()
    assert bars.price_basis.eq('NONE').all() and bars.end_at.is_monotonic_increasing and bars.end_at.is_unique
    assert np.isfinite(bars[COLS].to_numpy(float)).all()
    assert (bars[['open','high','low','close']]>0).all().all() and (bars[['volume','turnover']]>=0).all().all()
    assert (bars.high >= bars[['open','low','close']].max(axis=1)).all()
    assert (bars.low <= bars[['open','high','close']].min(axis=1)).all()
    cal=calendar(P['extension_start'],P['overlap_end'])['sessions']
    # Match the official calendar before counting half-day sessions.
    opens={r['session_date']:pd.Timestamp(r['open_at']) for r in cal}
    closes={r['session_date']:pd.Timestamp(r['close_at']) for r in cal}
    session_types=bars.session_type.copy()
    op=pd.to_datetime(bars.session_date.map(opens),utc=True)
    cl=pd.to_datetime(bars.session_date.map(closes),utc=True)
    session_types.loc[bars.start_at.ge(op)&bars.end_at.le(cl)]='regular'
    session_types.loc[bars.start_at.ge(cl)&session_types.ne('overnight')]='post_market'
    idx=pd.DatetimeIndex(bars.end_at); by=bars.set_index('end_at'); missing=[];prefix_missing=[];daily=[];ticks=0
    for row in cal:
        day=row['session_date'];expected=pd.date_range(row['open_at'],row['close_at'],freq='5min',inclusive='right')
        if not expected.isin(idx).all():missing.append(dict(date=day,missing=int((~expected.isin(idx)).sum())));continue
        g=by.loc[expected];daily.append(dict(date=day,open=float(g.open.iloc[0]),high=float(g.high.max()),low=float(g.low.min()),close=float(g.close.iloc[-1]),volume=float(g.volume.sum())))
        for clock in CLOCKS:
            at=pd.Timestamp(day+' '+clock,tz='America/New_York').tz_convert('UTC')
            if at>=pd.Timestamp(row['close_at']):continue
            ticks+=1
            exp=pd.date_range(pd.Timestamp(day+' 04:05',tz='America/New_York').tz_convert('UTC'),at.floor('5min'),freq='5min')
            if not exp.isin(idx).all():prefix_missing.append(dict(date=day,clock=clock,missing=int((~exp.isin(idx)).sum())))
    oldroot='soxl_soxs_14d5pct_v2' if symbol in ['SOXL','SOXS','SOXX','QQQ'] else 'soxl_soxs_14d5pct_v70'
    oldpath=ROOT/'.cache'/oldroot/'data/bars'/f'{symbol}.parquet';old=pd.read_parquet(oldpath)
    cut=lambda d:d[d.session_date.between('2024-10-14',P['overlap_end'])].set_index('time_key')
    a,b=cut(bars),cut(old);common=a.index.intersection(b.index)
    overlap_diff=(a.loc[common,COLS]!=b.loc[common,COLS]).any(axis=1)
    new_only=a.index.difference(b.index)
    extra_before_initial_boundary=bool(len(new_only)==0 or (new_only < old.time_key.min()).all())
    assert len(common)>0
    external=read(CACHE/'ibkr'/f'{symbol}.json');other=pd.DataFrame(external['rows']).set_index('date')
    price_rows=[]
    for r in daily:
        day=r['date']; share=1.
        for event in external['corp_actions']:
            if event['type']=='Splits' and pd.Timestamp(event['date'])>pd.Timestamp(day):share*=float(event['value'])
        if day not in other.index:raise ValueError((symbol,'IBKR date missing',day))
        for field in ['open','high','low','close']:
            # IBKR history uses current-share split basis; undo only known split ratios for price audit.
            value=float(other.loc[day,field])*share;diff=value-r[field]
            price_rows.append(dict(symbol=symbol,date=day,field=field,futu=r[field],ibkr_unadjusted=value,difference=diff,relative_difference=diff/r[field],share_conversion=share))
    diffs=pd.DataFrame(price_rows);diffs.to_csv(CACHE/f'{symbol}_price_comparison.csv',index=False)
    residual=(diffs.difference.abs()>.011)
    return dict(symbol=symbol,path=str(path.relative_to(ROOT)),sha256=sha(path),rows=len(bars),
        full_sessions=len(daily),requested_sessions=len(cal),extension_sessions=sum(r['session_date']<=P['extension_end'] for r in cal),
        missing_regular=missing,clock_queries=ticks,prefix_missing=prefix_missing,
        half_days=[r['session_date'] for r in cal if r['is_early_close']],
        overlap=dict(old_sha256=sha(oldpath),common_rows=len(common),new_only=len(new_only),old_only=len(b.index.difference(a.index)),changed_rows=int(overlap_diff.sum()),
          extra_before_old_first_timestamp=extra_before_initial_boundary,
          extra_first=str(new_only.min()) if len(new_only) else None,extra_last=str(new_only.max()) if len(new_only) else None,
          merge_rule='Use extension session_date <=2024-10-11; reuse frozen parent from 2024-10-14, including its original initial-boundary missingness.'),
        raw_lineage=lineage,actions=action_audit(symbol),
        cross_provider=dict(values=len(diffs),within_cent=int((~residual).sum()),max_absolute=float(diffs.difference.abs().max()),max_relative=float(diffs.relative_difference.abs().max()),
          residuals=diffs[residual].to_dict('records')),
        session_counts=session_types.value_counts().to_dict(),
        raw_session_field_requires_official_half_day_correction=True,
        all_prefixes_complete=not prefix_missing,
        structure_passed=not missing and not overlap_diff.any() and len(b.index.difference(a.index))==0 and extra_before_initial_boundary)


def main():
    from scripts.model_history_calendar import CLOSED, EARLY
    reference=read(SOURCE/'evidence/calendar_reference.json')
    assert {d for d in CLOSED if d[:4] in ['2023','2024']}==set(reference['holidays_2023']+reference['holidays_2024'])
    assert {d for d in EARLY if d[:4] in ['2023','2024']}==set(reference['early_close_13_et'])
    rows=[]
    for symbol in P['symbols']:
        row=audit_symbol(symbol);rows.append(row)
        print({k:row[k] for k in ['symbol','rows','full_sessions','clock_queries','structure_passed']},flush=True)
    result=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),symbols=rows,
        structure_passed=all(r['structure_passed'] for r in rows),actions_passed=all(r['actions']['passed'] for r in rows),
        all_prefixes_complete=all(r['all_prefixes_complete'] for r in rows),
        cross_provider_residual_review_required=True,training_dataset_accepted=False,
        new_model_trained=False,goal_complete=False)
    (CACHE/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['structure_passed'] and result['actions_passed']


if __name__=='__main__':main()
