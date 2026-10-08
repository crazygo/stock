"""Read-only OpenD watchlists, issuer holdings and explicit scope gaps."""
from __future__ import annotations
import csv, io, json, re, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=25) as r:
        return r.read()

def main():
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    q = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    try:
        ret, groups = q.get_user_security_group()
        if ret != ft.RET_OK: raise RuntimeError(str(groups))
        names = set(groups.group_name)
        favorite_name = '特别关注' if '特别关注' in names else 'Favorites'
        ret, fav = q.get_user_security(favorite_name)
        if ret != ft.RET_OK: raise RuntimeError(str(fav))
        ret, etf = q.get_user_security('ETF')
        if ret != ft.RET_OK: raise RuntimeError(str(etf))
        ret, all_watch = q.get_user_security('全部')
        if ret != ft.RET_OK: raise RuntimeError(str(all_watch))
    finally:
        q.close()
    snap = {'observed_at': datetime.now(timezone.utc).isoformat(),
            'membership_mode': 'current_snapshot_retrospective',
            'favorite_group': favorite_name, 'favorites': fav.to_dict('records'),
            'etf_watchlist': etf.to_dict('records'), 'all_watchlist':all_watch.to_dict('records'), 'funds': {}, 'members': {}}
    def add(symbol, group, name=None):
        if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', symbol): return
        row = snap['members'].setdefault(symbol, {'symbol': symbol, 'groups': [], 'name': name or symbol})
        if group not in row['groups']: row['groups'].append(group)
    for r in snap['favorites']:
        if r['code'].startswith('US.') and r['stock_type'] in ('STOCK', 'ETF'):
            add(r['code'][3:], '特别关注', r['name'])
    for r in snap['all_watchlist']:
        if r['code'].startswith('US.') and r['stock_type'] in ('STOCK','ETF'):
            add(r['code'][3:], 'futud自选', r['name'])
            snap['members'][r['code'][3:]]['stock_type']=r['stock_type']
    existing = json.loads((ROOT/'analysis/qqq_constituents.json').read_text())
    for s in existing['tickers']:
        if s not in ('SPY','DIA'): add(s.removeprefix('US.'), 'QQQ成分(仓库快照)')
    add('QQQ', '市场基准')
    # Parse full issuer CSV files only; never silently use the top-ten table.
    funds = {r['code'][3:] for r in snap['etf_watchlist'] if r['code'].startswith('US.') and r['stock_type']=='ETF'}
    funds |= {r['code'][3:] for r in snap['all_watchlist'] if r['code'].startswith('US.') and r['stock_type']=='ETF'}
    funds |= {r['code'][3:] for r in snap['favorites'] if r['code'].startswith('US.') and r['stock_type']=='ETF'}
    for s in funds: add(s, '自选ETF')
    ish = {'SOXX':239705,'IGV':239771,'IBB':239699,'EWH':239657,'EWY':239681,
           'IYE':239507,'IXC':239741,'VLUE':251616,'SLVP':239656,'IYM':239505,
           'IWO':239709,'QUAL':256101,'ITA':239502}
    slugs={'SOXX':'ishares-phlx-semiconductor-etf','IGV':'ishares-north-american-techsoftware-etf',
           'IBB':'ishares-nasdaq-biotechnology-etf','EWH':'ishares-msci-hong-kong-etf','EWY':'ishares-msci-south-korea-capped-etf',
           'IYE':'ishares-us-energy-etf','IXC':'ishares-global-energy-etf','VLUE':'ishares-msci-usa-value-factor-etf',
           'SLVP':'ishares-msci-global-silver-miners-etf','IYM':'ishares-us-basic-materials-etf',
           'IWO':'ishares-russell-2000-growth-etf','QUAL':'ishares-msci-usa-quality-factor-etf','ITA':'ishares-us-aerospace-defense-etf'}
    from concurrent.futures import ThreadPoolExecutor
    def holdings(fund):
        urls=[]
        if fund in ish:
            page=f'https://www.ishares.com/us/products/{ish[fund]}/{slugs[fund]}'
            try:
                page_html=fetch(page).decode('utf8','replace')
                urls=list(dict.fromkeys(re.findall(r'https://www\.ishares\.com/[^"\s<>]+latest-holdings\.csv',page_html)))[:1]
            except Exception:urls=[]
            if not urls:urls=[page+'/latest-holdings.csv']
        elif fund in ('BOTZ','FINX','PAVE','COPX'):
            urls=[f'https://assets.globalxetfs.com/funds/holdings/{fund.lower()}_full-holdings_20261002.csv']
        elif fund in ('CHAT','LYTE','DRAM'):
            urls=['https://www.roundhillinvestments.com/assets/data/FilepointRoundhill.40RU.RU_Holdings_20261002.csv']
        elif fund=='LAZR':
            page_html=fetch('https://temaetfs.com/lazr').decode('utf8','replace')
            urls=re.findall(r'https://temaetfs\.com/[^"\s<>]+\.csv[^"\s<>]*',page_html)[:1]
        elif fund=='ARKG':
            urls=['https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv']
        elif fund in ('SMH','BBH'):
            urls=[f'https://www.vaneck.com/us/en/investments/{"semiconductor-etf-smh" if fund=="SMH" else "biotech-etf-bbh"}/holdings/']
        members=[]; errors=[]
        for url in urls:
            try:
                raw=fetch(url); text=raw.decode('utf-8-sig',errors='replace')
                if '<html' in text[:1000].lower() or '<!doctype' in text[:1000].lower():
                    errors.append('HTML response; full holdings not parsed'); continue
                lines=text.splitlines(); header=next((i for i,l in enumerate(lines) if 'Ticker' in l or 'ticker' in l),None)
                if header is None: raise ValueError('No full-holdings ticker header')
                rows=list(csv.DictReader(io.StringIO('\n'.join(lines[header:]))))
                if fund in ('CHAT','LYTE','DRAM'):
                    rows=[r for r in rows if any(str(v).strip().upper()==fund for v in list(r.values())[:3])]
                for r in rows:
                    ticker=(r.get('Ticker') or r.get('ticker') or '').strip().upper()
                    asset=(r.get('Asset Class') or '').strip()
                    if asset and asset!='Equity': continue
                    location=r.get('Location','')
                    exchange=r.get('Exchange','')
                    if location and location!='United States' and not any(k in exchange for k in ('NASDAQ','NYSE','Cboe')): continue
                    if re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}',ticker) and ticker not in ('USD','CASH'):
                        members.append(ticker)
                if len(members)>=1 or (rows and fund in ish):
                    return fund, {'status':'full_issuer_file_us_equities', 'source_url':url,
                                  'source_header':lines[:header], 'raw_sha256':__import__('hashlib').sha256(raw).hexdigest(),
                                  'members':sorted(set(members))}
                errors.append('Fewer than five stock rows; not treated as complete')
            except Exception as e: errors.append(type(e).__name__+': '+str(e)[:120])
        return fund, {'status':'unresolved_full_constituents','attempted_urls':urls,'errors':errors,
                      'members':[], 'reason':'No verified complete US stock membership source in this capture'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fund, result in pool.map(holdings, sorted(funds)):
            snap['funds'][fund]=result
            for s in result['members']: add(s,'ETF成分:'+fund)
            print(json.dumps({'fund':fund,'status':result['status'],'members':len(result['members'])}),flush=True)
    # Retain official dated BOTZ snapshot as explicitly stale fallback.
    prior=ROOT/'data/ai_fund_holdings.json'
    if prior.exists():
        j=json.loads(prior.read_text())
        for fund in funds:
            if snap['funds'][fund]['members']: continue
            eligible=[s for s in j['snapshots'] if s['fund']==fund]
            if eligible:
                s=max(eligible,key=lambda x:x['as_of'])
                snap['funds'][fund]['dated_fallback']=s
                for h in s['holdings']: add(h['ticker'],'ETF成分(旧快照):'+fund)
    (OUT/'universe.json').write_text(json.dumps(snap,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({'total':len(snap['members']),'favorite_us':sum('特别关注' in m['groups'] for m in snap['members'].values()),
                      'resolved_funds':sum(bool(s['members']) for s in snap['funds'].values()),'funds':len(funds)}),flush=True)

if __name__=='__main__': main()
