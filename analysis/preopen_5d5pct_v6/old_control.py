"""Fixed old +3% emissions, new five-day outcomes and target-matched controls."""
import json
import numpy as np
import pandas as pd
from common import OUT,OLD,DATES,sha,write,now
from research import baseline_table,baseline,metrics
from research_extensions import repair_action_labels
from verify import expected_event

def main():
    panel=pd.read_parquet(OUT/'cache/panel.parquet');panel.symbol=panel.symbol.astype(str);panel.day=panel.day.astype(str)
    panel,_=repair_action_labels(panel);raw_cache={};rows=[]
    for route in ['hazard_linear','relative_flow','recovery_forest']:
        source=OLD/'weekly_v1'/f'{route}_replay.json';sim=json.loads(source.read_text());events=[];tables={}
        for old in sim['events']:
            day=old['day'];symbol=old['symbol'];month=day[:7]
            if symbol not in raw_cache:
                raw=pd.read_parquet(OLD/'raw'/f'{symbol}.parquet');actions=set()
                for folder in [OUT/'cache/corporate_actions',OLD.parents[1]/'market_data/model_training_history_v1/corporate_actions',OLD.parents[1]/'market_data/corporate_actions',OLD.parents[1]/'analysis/preopen_intraday_v2/cache/corporate_actions']:
                    p=folder/f'{symbol}.parquet'
                    if p.exists():actions.update(pd.read_parquet(p).ex_div_date.astype(str).str[:10])
                raw_cache[symbol]=(raw,actions)
            e=dict(old);e['score']=e.get('score',e.get('probability',np.nan));y,status,entry,target=expected_event(e,*raw_cache[symbol])
            match=panel[(panel.symbol==symbol)&(panel.day==day)&(panel.minute==e['minute'])];assert len(match)==1
            if month not in tables:
                cursor=next(i for i,d in enumerate(DATES) if d>=month+'-01')-10;dates=DATES[max(0,cursor-200):cursor]
                history=panel[panel.day.isin(dates)&panel.y.notna()&panel.label_end.str[:10].lt(month+'-01')]
                tables[month]=baseline_table(history)
            e.update(y=y,future_status=status,entry=entry,target=target,label_end=match.label_end.iloc[0],baseline=float(baseline(tables[month],match)[0]),h_rv_30=float(match.h_rv_30.iloc[0]))
            events.append(e)
        m=metrics(events,sim['ticks']);m.pop('brier');m.pop('baseline_brier');m.pop('mean_signal_probability')
        stocks=[dict(symbol=s,**{k:v for k,v in metrics([e for e in events if e['symbol']==s],sim['ticks']).items() if k not in ['brier','baseline_brier','mean_signal_probability']}) for s in sorted({e['symbol'] for e in events})]
        last={};non=[]
        for e in sorted(events,key=lambda x:(x['day'],x['minute'])):
            pos=DATES.index(e['day'])
            if pos-last.get(e['symbol'],-100)>=5:non.append(e);last[e['symbol']]=pos
        nm={k:v for k,v in metrics(non,sim['ticks']).items() if k not in ['brier','baseline_brier','mean_signal_probability']}
        rows.append(dict(route=route,source_sha256=sha(source),metrics=m,stocks=stocks,nonoverlap=nm,events=events,probability='original +3% only; five-day probability not calibrated'))
    write(OUT/'old_control.json',dict(at=now(),status='exposed_fixed_legacy_control',independent_pass=False,protocol_sha256=sha(OUT/'OLD_CONTROL_PROTOCOL.md'),routes=rows))
    lines=['# 固定旧当天 +3% 信号的五日后续','','旧信号不重新筛选；旧概率尚未针对五日+5%校准。本表是后续结果对照，不能与新目标模型不同样本直接比较胜负。','', '| 原路线 | 五日TP/成熟 | 全部 | 日期 | 每周信号 | 匹配增量 | 两周块95% |','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        m=r['metrics'];lines.append(f"| {r['route']} | {m['tp']}/{m['mature']}={m['precision']:.2%} | {m['signals']} | {m['date_count']} | {m['signals_per_week']:.2f} | {m['lift']:.2%} | {m['block_ci']} |")
    lines+=['','全部原信号、逐股集中与不重叠敏感性见old_control.json。三路线不能重复累加成独立样本；每条路线都少于100成熟信号，未满足新验收。']
    (OUT/'OLD_CONTROL.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
