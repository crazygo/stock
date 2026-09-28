"""Separate recent 30/60-session exploration plus fixed-rule migration audit."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

import search_rules as search_module
from search_rules import mask_rule, search_group
from evaluate import evaluate_rule, json_safe, pct

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(os.environ['TOUCH_RESEARCH_RUN'])
SOURCE = ROOT/'research/group_expectation_matrix/outputs/20260925_v5'


def fast_describe(frame,mask):
    """Cache unchanged reference data; same arithmetic as search_rules.describe."""
    cached=frame.attrs.get('_touch_fast_stats')
    if cached is None:
        daily=frame.groupby('date').target.mean()
        cached=(frame.target.to_numpy(),pd.factorize(frame.date)[0],pd.factorize(frame.symbol)[0],
                frame.date.map(daily).to_numpy())
        frame.attrs['_touch_fast_stats']=cached
    y,dates,stocks,base=cached
    idx=np.flatnonzero(mask);n=len(idx)
    if not n:return {'n':0,'hits':0,'rate':None,'dates':0,'stocks':0,'lcb':0.,'baseline':None,'lift':None}
    h=int(y[idx].sum());b=float(base[idx].mean())
    return {'n':n,'hits':h,'rate':h/n,'dates':len(np.unique(dates[idx])),'stocks':len(np.unique(stocks[idx])),
            'lcb':search_module.wilson(h,n),'baseline':b,'lift':h/n-b}


def migration(panel,item,dates):
    spans={'recent60':dates[-60:],'recent30':dates[-30:],'previous30':dates[-60:-30]}
    metrics={name:evaluate_rule(panel[panel.date.isin(ds)],item,'all') for name,ds in spans.items()}
    old,new=metrics['previous30'],metrics['recent30']
    delta = new['rate']-old['rate'] if old['rate'] is not None and new['rate'] is not None else None
    total_delta = new['total_rate']-old['total_rate'] if old['total_rate'] is not None and new['total_rate'] is not None else None
    ci=None
    data=panel[panel.date.isin(dates[-60:])&panel.checkpoint.eq(item['checkpoint'])&~panel.already_hit
               &panel.group_ids.map(lambda ids:item['group_id'] in ids)]
    selected=data.loc[mask_rule(data,item['rule'])]
    if len(selected) and old['n'] and new['n']:
        daily=selected.groupby('date').target.agg(['sum','size']).reindex(dates[-60:],fill_value=0).to_numpy(float)
        rng=np.random.default_rng(20260927)
        starts=rng.integers(0,60,(3000,6))
        picks=((starts[:,:,None]+np.arange(10))%60).reshape(3000,60)
        h,n=daily[:,0][picks],daily[:,1][picks]
        isnew=picks>=30
        nn=(n*isnew).sum(1);no=(n*~isnew).sum(1)
        good=(nn>0)&(no>0)
        delta_boot=(h*isnew).sum(1)[good]/nn[good]-(h*~isnew).sum(1)[good]/no[good]
        ci=np.quantile(delta_boot,[.025,.975]).tolist()
    oldstocks=set(old['ticker_counts']);newstocks=set(new['ticker_counts'])
    return {'metrics':metrics,'change_recent30_minus_previous30':delta,'total_change_recent30_minus_previous30':total_delta,
            'change_ci95':ci,'triggered_stock_jaccard':len(oldstocks&newstocks)/len(oldstocks|newstocks) if oldstocks|newstocks else None,
            'support_in_both':old['n']>=20 and new['n']>=20 and old['dates']>=8 and new['dates']>=8,
            'interpretation':'同一条固定规则跨窗口比较；近期规则本身在近期选出，迁移对比仍是探索性'}


def main():
    # Process-local optimization; the original discovery run remains unchanged.
    search_module.describe=fast_describe
    panel=pd.read_parquet(HERE/'features.parquet')
    groups=json.loads((SOURCE/'groups.json').read_text())
    strategies={s['strategy_id']:s['name'] for s in json.loads((SOURCE/'strategies.json').read_text())}
    catalogs=json.loads((HERE/'threshold_catalog.json').read_text())
    # Official calendar dates, never inferred from one ticker.
    calendar=json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())
    end=panel.date.max()
    dates=[s['session_date'] for s in calendar['sessions'] if '2026-01-02'<=s['session_date']<=end]
    windows={}
    for days in (30,60):
        recent=panel[panel.date.isin(dates[-days:])]
        records=[]
        coverage=[]
        for checkpoint,cp in recent.groupby('checkpoint'):
            pending=cp[~cp.already_hit]
            rules=[tuple(tuple(atom) for atom in rule) for rule in catalogs[checkpoint]]
            masks=np.column_stack([mask_rule(pending,r) for r in rules])
            members=pending.group_ids.tolist()
            for group in groups:
                membership=np.array([group['group_id'] in g for g in members])
                g=pending.loc[membership]
                if len(g)<20 or g.date.nunique()<8:
                    coverage.append({'group_id':group['group_id'],'checkpoint':checkpoint,'n':len(g),'status':'insufficient_20_samples_8_dates','rules':len(rules)})
                    continue
                found=search_group(g,g,rules,masks[membership],masks[membership],group,checkpoint,min_samples=20,min_dates=8)
                records.extend(found)
                coverage.append({'group_id':group['group_id'],'checkpoint':checkpoint,'n':len(g),'status':'searched','rules':len(rules),'supported_candidates':len(found)})
            print(f'Recent {days}, {checkpoint}: {len(records)} supported candidates so far',flush=True)
        winners=[]
        for group in groups:
            candidates=[r for r in records if r['group_id']==group['group_id'] and r['discovery']['lift']>0]
            if not candidates:continue
            candidates.sort(key=lambda r:(r['discovery']['lcb'],r['discovery']['lift'],-r['terms']),reverse=True)
            best=dict(candidates[0])
            best['strategy_name']=strategies[best['strategy_id']]
            best['window_metrics']=evaluate_rule(recent,best,'all')
            singles=[r for r in candidates if r['checkpoint']==best['checkpoint'] and r['terms']==1]
            chosen=[];used=set()
            for s in singles:
                key=s['rule'][0][0]
                if key not in used:
                    used.add(key)
                    chosen.append({**s,'window_metrics':evaluate_rule(recent,s,'all')})
                if len(chosen)==3:break
            best['top3_single_features']=chosen
            best['migration']=migration(panel,best,dates)
            winners.append(best)
        top=[];sets=[];aliases=[]
        for best in sorted(winners,key=lambda r:(r['window_metrics']['total_rate'],r['window_metrics']['lcb']),reverse=True):
            g=recent[recent.checkpoint.eq(best['checkpoint'])&~recent.already_hit&recent.group_ids.map(lambda ids:best['group_id'] in ids)]
            events=set(g.loc[mask_rule(g,best['rule']),'sample_id'])
            duplicate=next((i for i,s in enumerate(sets) if len(events&s)/len(events|s)>=.9),None)
            if duplicate is not None:
                aliases.append({'group_id':best['group_id'],'representative':top[duplicate]['group_id']})
                continue
            top.append(best);sets.append(events)
            if len(top)==5:break
        windows[str(days)]={'dates':[dates[-days],dates[-1]],'sessions':days,'top5':top,'group_winners':winners,
                            'coverage':coverage,'supported_candidate_count':len(records),'near_duplicate_aliases':aliases,
                            'evidence_status':'window_in_sample_exploration_not_independent_validation'}
        flat=[{'group_id':r['group_id'],'checkpoint':r['checkpoint'],'terms':r['terms'],'text':r['text'],
               'rule_json':json.dumps(r['rule']),**r['discovery']} for r in records]
        pd.DataFrame(flat).to_parquet(HERE/f'recent_{days}_search.parquet',index=False,compression='zstd')
    main_results=json.loads((HERE/'results.json').read_text())
    frozen_migration=[{'group_id':x['group_id'],'group_name':x['group_name'],'strategy_name':x['strategy_name'],
                       'checkpoint':x['checkpoint'],'text':x['text'],'rule':x['rule'],'migration':migration(panel,x,dates)}
                      for x in main_results['top5_by_total_probability']]
    output={'windows':windows,'fixed_primary_rule_migration':frozen_migration,
            'calendar_spans':{'recent30':[dates[-30],dates[-1]],'recent60':[dates[-60],dates[-1]],'previous30':[dates[-60],dates[-31]]},
            'note':'Recent30 is contained in recent60. Changes use non-overlapping previous30 versus recent30. All recent winners are exploratory.'}
    (HERE/'recent_results.json').write_text(json.dumps(json_safe(output),ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# 最近30/60交易日与时间迁移','',
           '**近期排名是窗口内探索结果，不能与先冻结规则再看后段的复核证据混称。** 各群先以未达标样本的Wilson下界选条件，再按原口径总触达率排名；至少20样本、8日期。每个群仍只有一个观察时点，去掉高度重复事件。','',
           f"最近30日起点：{dates[-30]}—{dates[-1]}；最近60日起点：{dates[-60]}—{dates[-1]}；前30日：{dates[-60]}—{dates[-31]}。这些是五日结果已完整成熟的起始日。最近的未成熟起点不算失败，也不凭提前成功纳入。",'']
    for days in ('30','60'):
        w=windows[days]
        lines += [f'## 最近{days}日 Top5','',f"支持条件/组合共{w['supported_candidate_count']:,}条。",'',
                  '| 排名 | 策略 / 群组 | ET时点与同时满足的条件 | 总触达率 | 未达标者后续触达 | 日期/证券 | 同群同日基线 |',
                  '|---:|---|---|---:|---:|---:|---:|']
        for rank,x in enumerate(w['top5'],1):
            m=x['window_metrics']
            lines.append(f"| {rank} | {x['strategy_name']} / {x['group_name']} | {x['checkpoint']}；{x['text']} | {pct(m['total_rate'])}（{m['total_hits']}/{m['total_n']}） | {pct(m['rate'])}（{m['hits']}/{m['n']}） | {m['dates']}/{m['stocks']} | {pct(m['baseline'])} |")
        for x in w['top5']:
            lines += ['',f"### {x['group_name']}：三个单项条件",'', '| 单项条件 | 总触达率 | 未达标者后续触达 |','|---|---:|---:|']
            for s in x['top3_single_features']:
                m=s['window_metrics']
                lines.append(f"| {s['text']} | {pct(m['total_rate'])}（{m['total_hits']}/{m['total_n']}） | {pct(m['rate'])}（{m['hits']}/{m['n']}） |")
            mig=x['migration'];old=mig['metrics']['previous30'];new=mig['metrics']['recent30']
            delta=mig['change_recent30_minus_previous30']
            lines += ['',f"固定这个条件不重调：前30日 {pct(old['rate'])}（{old['hits']}/{old['n']}），后30日 {pct(new['rate'])}（{new['hits']}/{new['n']}）；变化 {delta*100:+.1f}个百分点。" if delta is not None else '固定规则跨窗口样本不足。']
            if not mig['support_in_both']:lines.append('至少一个子窗口不足20样本或8日期，不能把波动视为已确认迁移。')
    lines += ['', '## 固定主要候选：前后30日迁移', '',
              '| 群组 | 固定条件 | 前30日 | 后30日 | 最近60日 | 变化百分点 | 两段样本门槛 |','|---|---|---:|---:|---:|---:|---|']
    for x in frozen_migration:
        m=x['migration'];v=m['metrics'];delta=m['change_recent30_minus_previous30']
        lines.append(f"| {x['group_name']} | {x['checkpoint']}；{x['text']} | {pct(v['previous30']['rate'])}（{v['previous30']['hits']}/{v['previous30']['n']}） | {pct(v['recent30']['rate'])}（{v['recent30']['hits']}/{v['recent30']['n']}） | {pct(v['recent60']['rate'])}（{v['recent60']['hits']}/{v['recent60']['n']}） | {delta*100:+.1f} | {'满足' if m['support_in_both'] else '不足'} |" if delta is not None else f"| {x['group_name']} | 跨段不足 | — | — | — | — | 不足 |")
    lines += ['', '完整58群的逐时点搜索数量、未覆盖原因、群最佳候选、Top3单项、近期条件同规则迁移与日期块区间见 `recent_results.json`。','']
    (HERE/'RECENT_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(json_safe({'spans':output['calendar_spans'],'top5':{k:[{'group':x['group_name'],'time':x['checkpoint'],'condition':x['text'],'metric':x['window_metrics']} for x in v['top5']] for k,v in windows.items()}}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
