#!/usr/bin/env python3
"""Repeatable broad-market EOD doubling-opportunity report with evidence gates."""
from __future__ import annotations
import argparse,json,hashlib,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib,numpy as np,pandas as pd
from data import CACHE,FEATURES,universe,refresh_archive,load_panel,build_dataset,peak_and_stage,sessions,ROOT
from model import fit_month,predict,ALGORITHMS
from quality import inspect_company

HERE=Path(__file__).parent

def fmt(v,percent=False):
    if v is None or pd.isna(v):return '待证 / 不可评分'
    return f'{100*v:.2f}%' if percent else f'${v:.2f}'


def mxl_case(panel):
    early=panel[(panel.ticker=='MXL')&(panel.date==pd.Timestamp('2026-09-02'))]
    x={'case_mode':'user_exposed_example_not_validation','sep2_close':float(early.iloc[0].close) if len(early) else None}
    p=ROOT/'market_data/us_60m/MXL/2026.parquet'
    if p.exists():
        q=pd.read_parquet(p);q=q[q.time_key.astype(str).str[11:19]=='16:00:00'].sort_values('time_key')
        if len(q):
            x.update(latest_rth_date=str(q.iloc[-1].time_key)[:10],latest_rth_close=float(q.iloc[-1].close),source='Futu OpenD QFQ hourly 16:00 bar; separate from Massive archive')
            if x['sep2_close']:x['return_from_sep2']=x['latest_rth_close']/x['sep2_close']-1
    return x


def render(rows,meta,summary):
    lines=['# 30 / 60 日翻倍机会 · 单次运行报告','',
      f"运行时间：{meta['received_at']}。目录 {meta['directory_count']:,} 只证券，普通股 / ADR 候选 {meta['accepted_count']:,} 只。",
      f"全市场行情截止：**{meta['data_asof']}**；最近应有完整交易日：**{meta['expected_latest_session']}**。当前时效状态：**{meta['freshness']}**。",
      f"本次可评分 {meta['scored_count']:,} 只；未评分 {meta['unscored_count']:,} 只，原因保留在全表。其中 {meta['incoherent_horizon_count']} 只出现原始 30 日概率大于 60 日概率，等待联合校准，原始估计保留但不可用。",
      '**通过“60 日翻倍概率 >80%”完整证据门禁的股票：0。** 当前排序供研究，未证明达到用户的概率目标。','',
      '两目标都是日历日；当前标签为提供商日聚合 High 触及参考收盘价 × 1.002 × 2，常规盘 scope 尚未验证（见 SOURCE_AUDIT.md）。收盘仍翻倍是单独标签；不把触及当成交保证。概率针对行情截止时的 EOD 条件，不是盘中实时概率。',
      '算法：LogisticRegression 与 HistGradientBoostingClassifier；分月训练，先做时间隔离，再作 isotonic 概率校准，模型只以评价月前选择块 Brier 择优。',
      '当前目录回溯有幸存者偏差；原始日线跨抓取批次的公司行动复权尚待统一；未来独立观察及公司质量 × 热点联合回测尚未完成。','',
      '## 探索排序','',
      '|股票|现价 / 日期|上一确认峰值 / 日期|现价占上峰|阶段|30 日概率|60 日概率|公司质量|业务热点|',
      '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"|{r['ticker']}|{fmt(r['current_close'])} / {r['actual_price_date']}|{fmt(r.get('previous_peak_price'))} / {r.get('previous_peak_date') or '未确认'}|{fmt(r.get('current_to_previous_peak'),True)}|{r.get('stage','未知')}|{fmt(r.get('p30'),True)}|{fmt(r.get('p60'),True)}|{r.get('company_quality_status','未核验')}|{r.get('catalyst_status','待核验')}|")
    lines+=['','## 历史前向开发回测','',
      '|窗口|算法 / 策略|>80% 信号|成熟 TP / FP|未知|precision|Wilson 95%|股票 / 日期数|门禁|',
      '|---|---|---|---|---|---|---|---|---|']
    for b in summary:
        ci=b['wilson95'];interval=f"{fmt(ci[0],True)} – {fmt(ci[1],True)}" if ci[0] is not None else '不可评分'
        lines.append(f"|{b['horizon']} 日|{b['algorithm_name']}|{b['signals']}|{b['tp']} / {b['fp']}|{b['unknown']}|{fmt(b['precision'],True)}|{interval}|{b['stock_count']} / {b['decision_dates']}|{'数值通过，证据仍不足' if b['historical_numeric_gate'] else '未通过'}|")
    mx=meta['mxl_case'];lines+=['','## MXL 事实核对','',
      f"9 月 2 日常规盘收盘 {fmt(mx.get('sep2_close'))}；补齐后最新常规盘收盘 {fmt(mx.get('latest_rth_close'))}（{mx.get('latest_rth_date','未知')}），涨幅 {fmt(mx.get('return_from_sep2'),True)}。这是已暴露案例，不用它挑模型，也不称作事前命中。",
      'MXL 发行人 Q2 公告：收入同比 +55%，基础设施业务同比 +145%，涉及 AI 数据中心光互连；GAAP 经营利润率仍为 -2.5%。这些证明业务动量，不能直接推导未来翻倍概率。 [发行人公告](https://investors.maxlinear.com/press-releases/detail/617/maxlinear-inc-announces-second-quarter-2026-financial)','',
      '全量候选、未评分与排除记录见 all_stocks.csv、universe_exclusions.csv；完整概率、财报字段与出处见 report.json。回测的未知、失败、分块和可靠性分箱全部保留在 backtest_v1。','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--refresh',action='store_true');ap.add_argument('--top',type=int,default=30)
    ap.add_argument('--quality-top',type=int,default=30);ap.add_argument('--output-dir',type=Path);ap.add_argument('--backtest',action='store_true')
    args=ap.parse_args();cfg=json.loads((HERE/'config.json').read_text());refresh=refresh_archive() if args.refresh else {'refresh':'skipped'}
    u,umeta,exclusions=universe(args.refresh);panel,lineage=load_panel(u);d=build_dataset(panel,cfg)
    # --backtest is explicit and preserves previous output runs by dataset fingerprint.
    backdir=HERE/'backtest_v1';summary=json.loads((backdir/'backtest_summary.json').read_text()) if (backdir/'backtest_summary.json').exists() else []
    if args.backtest:
        from model import forward_backtest,evaluate
        pred,_=forward_backtest(d,cfg,backdir);summary=evaluate(pred,cfg,backdir)
    asof=lineage['last_session'];month=asof[:7]
    latest=d[d.date==pd.Timestamp(asof)].copy();score=latest[latest.eligible].copy()
    estimates={};model_audits=[]
    data_key=json.loads((CACHE/'dataset_key.json').read_text())['key']
    modelhash=hashlib.sha256((Path(__file__).with_name('model.py').read_text()+data_key).encode()).hexdigest()[:16]
    for h in cfg['horizons']:
        p=CACHE/'models'/f'{month}_{h}_{modelhash}.joblib';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():models,audit=joblib.load(p)
        else:
            print('current model',month,h,flush=True);models,audit=fit_month(d,month,h,cfg);joblib.dump((models,audit),p)
        model_audits.append(audit)
        if models:
            pr=predict(models,score);estimates[h]=dict(zip(score.ticker,(float(v) for v in pr[audit['champion']])))
    now=datetime.now(ZoneInfo('America/New_York'))
    expected=[x for x in sessions((now.date()-timedelta(days=14)).isoformat(),now.date().isoformat()) if x.date()<now.date() or (x.date()==now.date() and now.hour>=17)]
    latest_expected=expected[-1].date().isoformat()
    rows=[];last_by=panel.groupby('ticker').tail(1).set_index('ticker')
    for t,m in u.items():
        if t not in last_by.index:
            rows.append({'ticker':t,'name':m['name'],'current_close':None,'actual_price_date':None,'p30':None,'p60':None,'score_status':'no_archive_history'});continue
        r=last_by.loc[t];q=latest[latest.ticker==t]
        status='scored' if len(q) and bool(q.iloc[0].eligible) else 'insufficient_history_missing_or_low_liquidity'
        if r.date.date().isoformat()!=asof:status='missing_latest_session'
        x={'ticker':t,'name':m['name'],'current_close':float(r.close),'p30':estimates.get(30,{}).get(t),
           'p60':estimates.get(60,{}).get(t),'score_status':status,'qualified':False,
           'company_quality_status':'未核验','catalyst_status':'待核验',**peak_and_stage(panel,t,asof)}
        if len(q):
            x.update(coverage183=float(q.iloc[0].coverage183),ret20=float(q.iloc[0].ret20) if pd.notna(q.iloc[0].ret20) else None,
                 dollar_volume20=float(q.iloc[0].dollar_volume_actual) if pd.notna(q.iloc[0].dollar_volume_actual) else None,
                 volume_ratio=float(q.iloc[0].volume_ratio) if pd.notna(q.iloc[0].volume_ratio) else None)
        x['probability_order_consistent']=x['p30'] is None or x['p60'] is None or bool(x['p30']<=x['p60'])
        x['raw_model_p30']=x['p30'];x['raw_model_p60']=x['p60']
        if not x['probability_order_consistent']:
            x['p30']=None;x['p60']=None;x['score_status']='incoherent_horizon_predictions_require_joint_calibration'
        rows.append(x)
    rows.sort(key=lambda r:(-(r.get('p60') if r.get('p60') is not None else -1),-(r.get('p30') if r.get('p30') is not None else -1),r['ticker']))
    review=list(dict.fromkeys([r['ticker'] for r in rows[:max(args.quality_top,args.top)]]+['MXL']))
    qualities={};catalysts=json.loads((HERE/'catalysts.json').read_text())
    for t in review[:args.quality_top]+(['MXL'] if 'MXL' not in review[:args.quality_top] else []):
        print('quality',t,flush=True);qualities[t]=inspect_company(t,u.get(t,{}),asof,args.refresh);time.sleep(.15)
    for r in rows:
        t=r['ticker']
        if t in qualities:r['company_quality_status']=qualities[t]['status']
        cat=catalysts.get(t)
        if cat and cat['published_at']<asof and (pd.Timestamp(asof)-pd.Timestamp(cat['published_at'])).days<=cat['expires_after_days']:
            r['catalyst_status']=cat['review_status']
    meta={**umeta,'received_at':datetime.now(timezone.utc).isoformat(),'data_asof':asof,'expected_latest_session':latest_expected,
         'freshness':'current_complete_EOD' if asof==latest_expected else 'stale_not_current',
         'scored_count':sum(r['score_status']=='scored' and r['p60'] is not None for r in rows),
         'unscored_count':sum(r['score_status']!='scored' or r['p60'] is None for r in rows),
         'qualified_count':0,'incoherent_horizon_count':sum(r.get('probability_order_consistent') is False for r in rows),'refresh':refresh,'mxl_case':mxl_case(panel),'lineage':{k:v for k,v in lineage.items() if k!='lineage'},
         'model_audits':model_audits,'dataset_key':data_key,'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
         'completion':'incomplete: no verified >80% candidate; PIT, freshness, corporate actions, joint quality and independent validation pending'}
    out=args.output_dir or HERE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True,exist_ok=True)
    top=[r for r in rows if r['score_status']=='scored' and r['p60'] is not None][:args.top]
    payload={'metadata':meta,'qualified_stocks':[],'exploratory_stocks':top,'company_reviews':qualities,'case_stocks':[r for r in rows if r['ticker']=='MXL'],'catalysts':catalysts,'backtest':summary}
    (out/'report.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    (out/'report.md').write_text(render(top+([r for r in rows if r['ticker']=='MXL'] if not any(r['ticker']=='MXL' for r in top) else []),meta,summary))
    pd.DataFrame(rows).to_csv(out/'all_stocks.csv',index=False);pd.DataFrame(exclusions).to_csv(out/'universe_exclusions.csv',index=False)
    (HERE/'latest_run.json').write_text(json.dumps({'path':str(out.resolve()),'data_asof':asof,'run_at':meta['received_at']}))
    print(json.dumps({'report':str(out.resolve()),'qualified':0,'scored':meta['scored_count'],'top':top[:5]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
