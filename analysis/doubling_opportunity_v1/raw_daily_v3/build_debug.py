#!/usr/bin/env python3
"""All registered v3 stocks and covered daily paths; source basis stays explicit."""
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(HERE));from run import clean
from minute_paths import export as export_minute_paths
sys.path.insert(0,str(PARENT));from data import CACHE
from scripts.model_history_calendar import calendar

def main():
    run=Path(json.loads((HERE/'latest_run.json').read_text())['path']);report=json.loads((run/'report.json').read_text())
    pointer=json.loads((HERE/'latest_backtest.json').read_text());back=Path(pointer['path'])
    lineage=json.loads((back/'lineage.json').read_text());stocks=pd.read_csv(run/'all_stocks.csv',keep_default_na=False,na_values=['']).replace({np.nan:None}).to_dict('records')
    signals=[]
    for p in sorted(back.glob('signals_*.csv')):
        signals+=pd.read_csv(p,keep_default_na=False,na_values=['']).replace({np.nan:None}).to_dict('records')
    for s in signals:s['id']=f"{s['algorithm']}:{s['horizon']}:{s['ticker']}:{s['date'][:10]}"
    embed={s['ticker'] for s in signals}|{r['ticker'] for r in report['healthy_and_market_heat_stocks']+report['exploratory_price_stocks']+report['case_stocks']}
    mini_end=report['metadata']['expected_latest_session'];mini_start=(pd.Timestamp(mini_end)-pd.DateOffset(years=2)).date().isoformat()
    panel=pd.read_parquet(pointer['panel_path']);daily={};detail=[];data_dir=HERE/'debug_data';data_dir.mkdir(exist_ok=True)
    for t,q in panel[panel.ticker!='SPY'].groupby('ticker'):
        records=clean([[r.date.date().isoformat(),r.open,r.high,r.low,r.close,r.volume,r.split_basis_multiplier] for r in q.itertuples()])
        (data_dir/(t+'.json')).write_text(json.dumps({'ticker':t,'daily':records,'basis':'native NONE with frozen native/primary split reconciliation; unresolved actions retained unknown','asof':pointer['asof']},separators=(',',':'),allow_nan=False))
        detail.append(t)
        if t in embed:daily[t]=records
    minute_metadata,minute_export=export_minute_paths(lineage,data_dir)
    data=pd.read_parquet(pointer['dataset_path']);features={}
    for s in signals:
        q=data[(data.ticker==s['ticker'])&(data.date==pd.Timestamp(s['date']))]
        features[s['id']]={k:q.iloc[0][k] for k in lineage['features']} if len(q) else {}
    pred=pd.read_parquet(pointer['prediction_path'],columns=['ticker','label','horizon','algorithm','selected_algorithm','signal_eligible'])
    pred=pred[pred.signal_eligible];baselines={}
    for h in (30,60):
        for a in ('logistic','hist_gbdt','monthly_selected'):
            pool=pred[(pred.horizon==h)&(pred.selected_algorithm if a=='monthly_selected' else pred.algorithm==a)]
            baselines[f'{a}:{h}']=pool[pool.label.notna()].groupby('ticker').label.mean().to_dict()
    raw_first=min(v['raw_first'][:10] for v in lineage['native_sources'].values() if v.get('raw_rows',0))
    payload=clean({'experiment':'native-v3','report':report,'stocks':stocks,'signals':signals,'daily':daily,'hourly':{},'minute5':{},'minute_metadata':minute_metadata,'minute_export':minute_export,
        'features':features,'stock_baselines':baselines,'detail_tickers':detail,'source_available_equities':len(detail),
        'raw_first_date':raw_first,'calendar':calendar(mini_start,min('2026-12-31',(pd.Timestamp(mini_end)+pd.Timedelta(days=65)).date().isoformat())),
        'mini_start':mini_start,'mini_end':mini_end,'split_audit':json.loads((back/'split_audit.json').read_text())})
    text=(HERE/'debug_template.html').read_text().replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    output=HERE/'debug.html';output.write_text(text)
    manifest={'path':str(output.resolve()),'generated_at':datetime.now(timezone.utc).isoformat(),'bytes':output.stat().st_size,'stock_rows':len(stocks),'all_source_detail_tickers':len(detail),
        'embedded_detail_tickers':len(daily),'signal_rows':len(signals),'source_dataset_key':pointer['dataset_key'],
        'report_path':str(run.resolve()),'report_run_at':report['metadata']['run_at'],
        'report_sha256':hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),
        'html_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'template_sha256':hashlib.sha256((HERE/'debug_template.html').read_bytes()).hexdigest(),
        'minute_paths':minute_export,'external_daily_details':'generated local static JSON under ignored debug_data/'}
    (HERE/'debug_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='minute_paths'},indent=2))

if __name__=='__main__':main()
