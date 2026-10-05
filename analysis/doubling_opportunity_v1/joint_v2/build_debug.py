#!/usr/bin/env python3
"""Build one offline grayscale research page from actual retained reports and paths."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT));from data import CACHE,ROOT
from scripts.model_history_calendar import calendar


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,pd.Timestamp):return x.date().isoformat()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--experiment',choices=['price-control','financial'],default='price-control');args=ap.parse_args()
    run=Path(json.loads((HERE/'latest_run.json').read_text())['path']);report=json.loads((run/'report.json').read_text())
    pointer=json.loads((HERE/('latest_financial_backtest.json' if args.experiment=='financial' else 'latest_backtest.json')).read_text())
    back=Path(pointer['path'])
    if args.experiment=='financial':report['backtest']=report['financial_backtest']
    stocks=pd.read_csv(run/'all_stocks.csv',keep_default_na=False,na_values=['']).replace({np.nan:None}).to_dict('records')
    if args.experiment=='price-control':
        for r in stocks:r['p30']=r.get('price_control_p30');r['p60']=r.get('price_control_p60')
    signals=[]
    for p in sorted(back.glob('signals_*.csv')):
        q=pd.read_csv(p,keep_default_na=False,na_values=['']).replace({np.nan:None});signals+=q.to_dict('records')
    for s in signals:s['id']=f"{s['algorithm']}:{s['horizon']}:{s['ticker']}:{s['date'][:10]}"
    detail={s['ticker'] for s in signals}|{r['ticker'] for r in report['healthy_and_market_heat_stocks']+report['exploratory_price_stocks']+report['case_stocks']}
    panel=pd.read_parquet(CACHE/'panel.parquet');d=pd.read_parquet(pointer['dataset_path'] if args.experiment=='financial' else CACHE/'dataset.parquet')
    daily={t:[[r.date.date().isoformat(),r.open,r.high,r.low,r.close,r.volume] for r in q.itertuples()] for t,q in panel[panel.ticker.isin(detail)].groupby('ticker')}
    fields=['ret5','ret20','ret60','vol20','downvol20','range20','volume_ratio','peak_ratio','trough_ratio','position183','ma20_ratio','ma60_ratio','dollar_volume_actual','price','coverage183','rs20','rs60']
    if args.experiment=='financial':
        from prepare_joint import FEATURES as JOINT_FEATURES
        fields=list(dict.fromkeys(fields+JOINT_FEATURES))
    features={}
    for s in signals:
        row=d[(d.ticker==s['ticker'])&(d.date==pd.Timestamp(s['date']))]
        features[s['id']]={k:row.iloc[0][k] for k in fields} if len(row) else {}
    name='joint_v2_financial_market_heat_forward_predictions.parquet' if args.experiment=='financial' else 'joint_v2_forward_predictions.parquet'
    prediction_cols=['ticker','label','horizon','algorithm','selected_algorithm']+(['signal_eligible'] if args.experiment=='financial' else [])
    predictions=pd.read_parquet(CACHE/name,columns=prediction_cols)
    if args.experiment=='financial':predictions=predictions[predictions.signal_eligible]
    stock_baselines={}
    for h in (30,60):
        for algo in ('logistic','hist_gbdt','monthly_selected'):
            pool=predictions[(predictions.horizon==h)&(predictions.selected_algorithm if algo=='monthly_selected' else predictions.algorithm==algo)]
            stock_baselines[f'{algo}:{h}']=pool[pool.label.notna()].groupby('ticker').label.mean().to_dict()
    hourly={}
    for ticker in detail:
        p=ROOT/f'market_data/us_60m/{ticker}/2026.parquet'
        if p.exists():
            q=pd.read_parquet(p)
            hourly[ticker]=[[str(r.time_key),r.open,r.high,r.low,r.close,r.volume] for r in q.itertuples()]
    mini_end=report['metadata']['expected_latest_session'];mini_start=(pd.Timestamp(mini_end)-pd.DateOffset(years=2)).date().isoformat()
    cal=calendar(mini_start,(pd.Timestamp(mini_end)+pd.Timedelta(days=65)).date().isoformat())
    payload=clean({'experiment':args.experiment,'report':report,'stocks':stocks,'signals':signals,'daily':daily,'hourly':hourly,'features':features,'stock_baselines':stock_baselines,
                   'calendar':cal,'mini_start':mini_start,'mini_end':mini_end,'split_audit':json.loads((back/'split_audit.json').read_text())})
    template=(HERE/'debug_template.html').read_text()
    text=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    output=HERE/('financial_debug.html' if args.experiment=='financial' else 'debug.html');output.write_text(text)
    print(json.dumps({'path':str(output.resolve()),'bytes':output.stat().st_size,'stock_rows':len(stocks),'detail_tickers':len(daily),'hourly_tickers':len(hourly),'signal_rows':len(signals),'missing_minute_paths_explicit':True},indent=2))

if __name__=='__main__':main()
