"""Read-only full-market acquisition and causal daily feature/label construction."""
from __future__ import annotations
import csv, gzip, hashlib, io, json, os, sys, time, subprocess, re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from functools import lru_cache
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.model_history_calendar import calendar

CACHE = ROOT / '.cache/doubling_opportunity_v1'
FEATURES = ['ret5','ret20','ret60','vol20','downvol20','range20','volume_ratio',
            'peak_ratio','trough_ratio','position183','ma20_ratio','ma60_ratio',
            'dollar_volume20','price','coverage183','rs20','rs60']


def get_json(url, path, refresh=False):
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    contact=subprocess.run(['git','config','user.email'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    ua=os.environ.get('SEC_USER_AGENT') or f'StockDoublingResearch/1.0 {contact}'
    with urlopen(Request(url, headers={'User-Agent':ua}), timeout=40) as res:
        raw = res.read()
    j = json.loads(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.tmp').write_bytes(raw)
    path.with_suffix('.tmp').replace(path)
    return j


def universe(refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / 'nasdaqtraded.txt'
    fetched = CACHE / 'universe_received_at.txt'
    if refresh or not p.exists():
        with urlopen('https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt', timeout=40) as r:
            p.write_bytes(r.read())
        fetched.write_text(datetime.now(timezone.utc).isoformat())
    raw = p.read_text()
    rows = list(csv.DictReader(io.StringIO(raw), delimiter='|'))
    accepted, exclusions = {}, []
    for r in rows:
        ticker, name = r.get('Symbol',''), r.get('Security Name','')
        if not ticker or ticker.startswith('File Creation'): continue
        reason = None
        lower = name.lower()
        if r.get('Test Issue') == 'Y': reason = 'test_issue'
        elif r.get('ETF') == 'Y' or r.get('NextShares') == 'Y': reason = 'fund_etf'
        elif any(x in lower for x in ['warrant','preferred','depositary shares','debenture',' notes',' bond',' units','rights','interest in','beneficial interest']): reason = 'non_common_or_uncertain'
        elif 'fund' in lower and not any(x in lower for x in ['common stock','ordinary shares']): reason = 'fund_or_uncertain'
        elif not any(x in lower for x in ['common stock','ordinary share','common share','american depositary','american depository','adr','class a stock','class b stock']): reason = 'unclassified_requires_review'
        if reason:
            exclusions.append({'ticker':ticker,'name':name,'reason':reason})
        else:
            accepted[ticker] = {'name':name,'exchange':r.get('Listing Exchange'),
                                'financial_status':r.get('Financial Status'), 'membership_mode':'current_retrospective'}
    try:
        sec = get_json('https://www.sec.gov/files/company_tickers.json', CACHE / 'sec_tickers.json', refresh)
        sec_status='available'
    except Exception as exc:
        sec={}
        sec_status=f'unavailable:{type(exc).__name__}'
    for item in sec.values():
        t = item['ticker'].replace('-', '.')
        if t in accepted: accepted[t]['cik'] = int(item['cik_str'])
    if not sec:
        # Official SEC frame data can identify an issuer by its exact normalized legal name.
        # Ambiguous matches are never guessed; this is current metadata, not PIT membership.
        def issuer_key(name):
            name=name.split(' - ')[0]
            name=re.sub(r'(?i)\s+(?:class [a-z] )?(?:common stock|ordinary shares|common shares|american depositary.*)$','',name)
            return re.sub(r'[^a-z0-9]','',name.lower())
        mapping={};sources=[]
        for tag,period in [('Assets','CY2026Q2I'),('RevenueFromContractWithCustomerExcludingAssessedTax','CY2026Q2'),('Revenues','CY2026Q2')]:
            url=f'https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json'
            try:
                frame=get_json(url,CACHE/'sec_frames'/f'{tag}_{period}.json',refresh)
                sources.append(url)
                for r in frame.get('data',[]):
                    k=issuer_key(r['entityName']);mapping.setdefault(k,set()).add(int(r['cik']))
            except Exception:continue
        for t,item in accepted.items():
            matches=mapping.get(issuer_key(item['name']),set())
            if len(matches)==1:
                item['cik']=next(iter(matches));item['cik_mapping_source']='SEC frames exact normalized issuer name'
        sec_status=f'frames_exact_name_fallback:{sum("cik" in r for r in accepted.values())}'
        (CACHE/'sec_frame_mapping_sources.json').write_text(json.dumps(sources))
    meta = {'received_at':fetched.read_text(), 'nasdaq_footer':raw.splitlines()[-1],
            'directory_count':len(rows)-1, 'accepted_count':len(accepted), 'excluded_count':len(exclusions),
            'sources':['https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt',
                       'https://www.sec.gov/files/company_tickers.json'],
            'sec_mapping_status':sec_status, 'membership_mode':'current_retrospective', 'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (CACHE / 'universe.json').write_text(json.dumps({'metadata':meta,'stocks':accepted,'exclusions':exclusions},ensure_ascii=False))
    return accepted, meta, exclusions


@lru_cache(maxsize=64)
def sessions(start, end):
    return pd.DatetimeIndex([x['session_date'] for x in calendar(str(start),str(end))['sessions']])


def refresh_archive():
    """Restore only missing objects; never overwrite or upload shared raw archives."""
    from scripts.r2_client import R2Client
    r = R2Client()
    objects = r.list_objects('market_data/us/')
    downloaded = 0
    for o in objects:
        if not o['key'].endswith('/grouped.json.gz'): continue
        p = ROOT / o['key']
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            r.get_object(o['key'], p)
            downloaded += 1
    # The provider is the grouped archive source, not the hourly OpenD namespace.
    from screener import MassiveClient, load_env_file
    load_env_file(ROOT / '.env')
    key = os.environ.get('MASSIVE_API_KEY')
    latest = max(p.parent.name for p in (ROOT/'market_data/us').glob('*/grouped.json.gz'))
    if not key:
        return {'r2_downloads':downloaded,'provider_refresh':'missing_MASSIVE_API_KEY','latest_session':latest}
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('America/New_York'))
    end = now.date() - timedelta(days=1)
    client = MassiveClient(key, ROOT/'.cache/massive', request_delay=12.2)
    got = []
    for day in sessions(latest,end.isoformat()):
        ds = day.date().isoformat(); p = ROOT/'market_data/us'/ds/'grouped.json.gz'
        if p.exists(): continue
        bars = client.fetch_grouped_day(day.date())
        if not bars: continue
        payload = json.loads((ROOT/'.cache/massive/bars'/f'{ds}.json').read_text())
        if payload.get('adjusted') is not True: raise ValueError('Grouped day must be split adjusted')
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps(payload).encode()))
        got.append(ds)
    return {'r2_downloads':downloaded,'provider_refresh':'ok','downloaded_sessions':got}


def load_panel(stocks):
    files = sorted((ROOT / 'market_data/us').glob('*/grouped.json.gz'))
    if not files: raise ValueError('No full-market archive')
    lineage = [(p.parent.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    key = hashlib.sha256(json.dumps([lineage, sorted(stocks)]).encode()).hexdigest()
    cp = CACHE/'panel.parquet'; mp=CACHE/'panel_metadata.json'
    if cp.exists() and mp.exists():
        m=json.loads(mp.read_text())
        if m.get('cache_key')==key: return pd.read_parquet(cp),m
    rows=[]; audit=[]
    for p in files:
        j=json.loads(gzip.decompress(p.read_bytes()))
        if j.get('adjusted') is not True: raise ValueError(f'Unknown split adjustment {p.parent.name}')
        ds=p.parent.name
        audit.append({'session':ds,'count':len(j.get('results',[])),
                      'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        for r in j.get('results',[]):
            t=r.get('T')
            if t not in stocks and t!='SPY':continue
            if not all(isinstance(r.get(k),(int,float)) and r[k]>0 for k in ('o','h','l','c')):continue
            if r['h']<max(r['o'],r['c']) or r['l']>min(r['o'],r['c']):continue
            rows.append((t,ds,r['o'],r['h'],r['l'],r['c'],r.get('v',0)))
    df=pd.DataFrame(rows,columns=['ticker','date','open','high','low','close','volume'])
    df['date']=pd.to_datetime(df['date'])
    df=df.drop_duplicates(['ticker','date']).sort_values(['ticker','date'])
    m={'cache_key':key,'first_session':files[0].parent.name,'last_session':files[-1].parent.name,
       'archive_days':len(files),'rows':len(df),'stock_count':int(df.ticker.nunique()),
       'adjustment':'Massive split-adjusted; archive vintages mixed, corporate-action reconciliation outstanding',
       'lineage':audit}
    df.to_parquet(cp,compression='zstd',compression_level=7,index=False)
    mp.write_text(json.dumps(m))
    return df,m


def build_dataset(panel, cfg):
    cp=CACHE/'dataset.parquet'; mp=CACHE/'dataset_key.json'
    pmeta=json.loads((CACHE/'panel_metadata.json').read_text())
    key=hashlib.sha256(json.dumps([pmeta['cache_key'],cfg,Path(__file__).read_text()],sort_keys=True).encode()).hexdigest()
    if cp.exists() and mp.exists() and json.loads(mp.read_text()).get('key')==key:
        return pd.read_parquet(cp)
    start,end=panel.date.min().date().isoformat(),panel.date.max().date().isoformat()
    idx=sessions(start,(panel.date.max()+pd.Timedelta(days=65)).date().isoformat())
    spy=panel[panel.ticker=='SPY'].set_index('date').reindex(idx).close
    spyret={n:spy/spy.shift(n)-1 for n in (20,60)}
    observed_end=panel.date.max()
    out=[]
    for number,(t,g) in enumerate(panel.groupby('ticker',sort=True)):
        if t=='SPY':continue
        q=g.set_index('date').reindex(idx).drop(columns='ticker')
        c=q.close; r=c.pct_change(fill_method=None)
        f=pd.DataFrame(index=idx)
        for n in (5,20,60):f[f'ret{n}']=c/c.shift(n)-1
        f['vol20']=r.rolling(20,min_periods=20).std()
        f['downvol20']=r.clip(upper=0).rolling(20,min_periods=20).std()
        f['range20']=q.high.rolling(20).max()/q.low.rolling(20).min()-1
        f['volume_ratio']=q.volume.rolling(5).mean()/q.volume.rolling(20).mean()
        peak=q.high.rolling('183D',min_periods=cfg['minimum_sessions']).max()
        trough=q.low.rolling('183D',min_periods=cfg['minimum_sessions']).min()
        f['peak_ratio']=c/peak;f['trough_ratio']=c/trough
        f['position183']=(c-trough)/(peak-trough)
        f['ma20_ratio']=c/c.rolling(20).mean();f['ma60_ratio']=c/c.rolling(60).mean()
        f['dollar_volume20']=np.log1p((c*q.volume).rolling(20).mean())
        f['price']=np.log(c)
        counts=c.rolling('183D').count()
        expected_idx=sessions((pd.Timestamp(start)-pd.Timedelta(days=183)).date().isoformat(),idx[-1].date().isoformat())
        expected=pd.Series(1.,index=expected_idx).rolling('183D').count().reindex(idx)
        f['coverage183']=counts/expected
        for n in (20,60):f[f'rs{n}']=f[f'ret{n}']-spyret[n]
        f['close']=c;f['peak183']=peak;f['trough183']=trough
        f['dollar_volume_actual']=np.expm1(f.dollar_volume20)
        f['eligible']=(c>=1)&(f.dollar_volume_actual>=500_000)&(counts>=cfg['minimum_sessions'])&f[FEATURES].notna().all(axis=1)
        # Retain suspicious split/discontinuous archive cases as unknown, rather than labeling jumps as wins.
        discontinuity=((q.open/c.shift(1)>3)|(q.open/c.shift(1)<1/3))
        high_array=q.high.to_numpy(); close_array=c.to_numpy()
        missing_prefix=np.r_[0,np.cumsum(np.isnan(high_array))]
        discontinuity_prefix=np.r_[0,np.cumsum(discontinuity.to_numpy())]
        for h in cfg['horizons']:
            expiry=idx+pd.Timedelta(days=h)
            stop=idx.searchsorted(expiry,side='right')
            ys=np.full(len(idx),np.nan);ends=np.full(len(idx),np.nan)
            firsts=[None]*len(idx);reasons=[]
            maturities=[idx[j-1] if j>i+1 else pd.NaT for i,j in enumerate(stop)]
            for i,j in enumerate(stop):
                if np.isnan(close_array[i]):reason='missing_reference'
                elif expiry[i]>observed_end:reason='immature'
                elif j<=i+1 or missing_prefix[j]>missing_prefix[i+1]:reason='missing_future_session'
                elif discontinuity_prefix[j]>discontinuity_prefix[i+1]:reason='corporate_action_or_discontinuity_unreconciled'
                else:
                    reason='mature'
                    target=close_array[i]*(1+cfg['reference_cost'])*2
                    hits=np.flatnonzero(high_array[i+1:j]>=target)
                    ys[i]=int(len(hits)>0)
                    ends[i]=int(close_array[j-1]>=target)
                    if len(hits):firsts[i]=idx[i+1+hits[0]].date().isoformat()
                reasons.append(reason)
            f[f'y{h}']=ys;f[f'mature_at{h}']=maturities;f[f'first_touch{h}']=firsts
            f[f'end_close_double{h}']=ends;f[f'label_status{h}']=reasons
        f['ticker']=t;f['date']=idx
        # Store only real reference prices. Missing decisions are represented in separate coverage, not fake rows.
        out.append(f[c.notna()].reset_index(drop=True))
        if number%500==0:print(f'features {number} stocks',flush=True)
    dataset=pd.concat(out,ignore_index=True)
    dataset.to_parquet(cp,compression='zstd',compression_level=7,index=False)
    mp.write_text(json.dumps({'key':key}))
    return dataset


def peak_and_stage(panel,ticker,asof):
    cutoff=pd.Timestamp(asof)
    q=panel[(panel.ticker==ticker)&(panel.date<=cutoff)&(panel.date>cutoff-pd.Timedelta(days=183))].sort_values('date')
    if q.empty:return {}
    latest=float(q.iloc[-1].close)
    highs=q.high.to_numpy();lows=q.low.to_numpy(); peaks=[]
    for i in range(5,len(q)-5):
        if highs[i]>=max(highs[i-5:i+6]) and min(lows[i+1:i+6])<=highs[i]*.95:peaks.append(i)
    last=peaks[-1] if peaks else None
    peakrow=q.iloc[last] if last is not None else None
    highest=q.iloc[int(np.argmax(highs))]
    c=q.close;ret20=float(c.iloc[-1]/c.iloc[-21]-1) if len(c)>20 else None
    ma20=float(c.tail(20).mean());ma60=float(c.tail(60).mean())
    peakratio=latest/float(highest.high)
    if latest>=float(q.high.iloc[:-1].max())*.98:stage='高位突破 / 强势延续'
    elif peakratio<.6 and latest<ma20:stage='深度回撤，尚未恢复'
    elif latest>ma20>ma60 and ret20 is not None and ret20>.1:stage='回升趋势'
    elif len(c)>=20 and max(c.tail(20))/min(c.tail(20))<1.15:stage='横盘整理'
    elif latest<ma20<ma60:stage='下降趋势'
    else:stage='震荡过渡'
    return {'previous_peak_date':peakrow.date.date().isoformat() if peakrow is not None else None,
            'previous_peak_price':float(peakrow.high) if peakrow is not None else None,
            'current_to_previous_peak':latest/float(peakrow.high) if peakrow is not None else None,
            'six_month_peak_date':highest.date.date().isoformat(),'six_month_peak_price':float(highest.high),
            'current_to_six_month_peak':peakratio,'stage':stage,'actual_price_date':q.iloc[-1].date.date().isoformat()}
