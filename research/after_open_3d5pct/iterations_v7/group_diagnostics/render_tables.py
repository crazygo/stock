"""Render inspectable tables from frozen subgroup metrics, without recomputation."""
import argparse
import json
from pathlib import Path


def pct(value): return '—' if value is None else f'{value:.1%}'
def num(value): return '—' if value is None else f'{value:.4f}'


def render(source, output):
    data = json.loads(source.read_text())
    lanes = list(data['main_versions'])
    cells = {(r['split'], r['lane'], r['group_id']): r for r in data['panels']}
    def cell(lane, gid, split='outer'): return cells[split, lane, gid]
    def primary(lane, gid, split='outer'): return cell(lane,gid,split)['targets'].get('3d_5pct', {})
    lines = ['# v7 分群效果全表', '',
             '2026-09-01至09-17，同一1,140行、95个证券代码、12个ET交易日。全部为已暴露开发数据。', '',
             '固定版本：' + '；'.join(f'{k} {v}' for k,v in data['main_versions'].items()) + '，神经网络seed3566。', '',
             'Brier越低越好；skill为相对训练期收缩发生率参照的误差降低比例，负值表示比参照差。不同群不能仅按Brier排名。', '',
             '## 总体排序与同一股票不同日期的排序', '',
             '| 模型 | Brier | 平均预测 | 实际率 | 总体AUC | 同股内配对AUC | 相对历史每股基准skill | Brier差95%日期块区间 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for lane in lanes:
        m=primary(lane,'all:all'); lo,hi=m['delta_stock_reference_5date_ci']
        lines.append(f"| {lane} {data['main_versions'][lane]} | {num(m['brier'])} | {pct(m['predicted'])} | {pct(m['observed'])} | {num(m['auc'])} | {num(m['within_stock']['auc'])} | {pct(m['skill_vs_stock_history'])} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', '同股内AUC只比较同一证券中一条正例和一条负例的概率，按正负样本对数加权；63个证券有两类结果，共1,587对，重叠标签不等于独立事件。', '',
              '## 行业父群：五模型Brier', '', '行业标签为当前快照回溯，只作报告切片；多标签组不可相加。未覆盖行业标签不是一个行业。', '']
    def table(groups, metric='brier'):
        out=['| 群组 | 证券数 | 样本/正例 | 实际率 | '+' | '.join(lanes)+' |',
             '|---|---:|---:|---:|'+'---:|'*len(lanes)]
        for g in groups:
            gid=g['group_id']; c=cell(lanes[0],gid); m=primary(lanes[0],gid)
            values=[primary(lane,gid).get(metric) for lane in lanes]
            formatter=pct if metric.startswith('skill') or metric=='predicted' else num
            out.append(f"| {g['name']} | {len(c['symbols'])} | {c['n']}/{m.get('hits',0)} | {pct(m.get('observed'))} | "+' | '.join(formatter(v) for v in values)+' |')
        return out
    parents=[g for g in data['group_catalog'] if g.get('group_kind')=='parent' or g['group_id']=='industry:unclassified']
    lines += table(parents)
    lines += ['', '## 行业父群：相对历史每股基准的误差改善', '', '以下百分比是误差改善，不是盈利率或触及率。', ''] + table(parents, 'skill_vs_stock_history')
    lines += ['', '## 行业父群：平均预测概率', ''] + table(parents, 'predicted')
    dynamic=[g for g in data['group_catalog'] if g['strategy_id'] in ('trend_15','trend_63','trend_126','volatility','liquidity')]
    lines += ['', '## 动态趋势、波动、流动性：五模型Brier', '', '同一证券随周版本变更所属组，证券数是窗口内去重并集，行数才是该组实际分母。', ''] + table(dynamic)
    market=[g for g in data['group_catalog'] if g['strategy_id'] in ('market_r2','residual_sigma','market_residual')]
    thresholds=data['thresholds_training_tertiles']
    lines += ['', '## 市场联动与残差波动：相对历史每股基准skill', '',
              f"训练期三分位切点：市场R²={thresholds['market_r2'][0]:.4f}/{thresholds['market_r2'][1]:.4f}；残差日波动={thresholds['residual_sigma'][0]:.2%}/{thresholds['residual_sigma'][1]:.2%}。分组只使用决策前20个完整日收益。", ''] + table(market, 'skill_vs_stock_history')
    lines += ['', '## 市场联动与残差波动：AUC', ''] + table(market, 'auc')
    children=[g for g in data['group_catalog'] if g.get('group_kind')=='subgroup']
    lines += ['', '## 全部行业子群：五模型Brier', '', '零样本和单股群保留，不能把其低误差当成独立有效证据。', ''] + table(children)
    lines += ['', '## 各行业及市场组：八月→九月同股基准skill', '', '八月用于早停/选型，九月此前已暴露，两者都不是独立验证。', '',
              '| 群组 | '+' | '.join(lanes)+' |', '|---|'+'---:|'*len(lanes)]
    for g in parents+market:
        vals=[pct(primary(l,g['group_id'],'inner').get('skill_vs_stock_history'))+' → '+pct(primary(l,g['group_id']).get('skill_vs_stock_history')) for l in lanes]
        lines.append('| '+g['name']+' | '+' | '.join(vals)+' |')
    (output/'TABLES.md').write_text('\n'.join(lines)+'\n')
    members=['# 群组成员（九月评估窗口内去重并集）', '',
             '动态组成员会随日期变化，同一代码可以在不同日期进入不同状态；行业父子群也重叠。准确逐日成员见本地运行产物memberships.json。', '',
             '| 群组 | 样本数 | 证券代码 |', '|---|---:|---|']
    for g in data['group_catalog']:
        c=cell(lanes[0],g['group_id']);members.append('| '+g['name']+' | '+str(c['n'])+' | '+(', '.join(c['symbols']) or '无样本')+' |')
    (output/'GROUP_MEMBERS.md').write_text('\n'.join(members)+'\n')
    stocks={(s['symbol'],s['lane']):s for s in data['stocks_primary']}
    lines=['# v7逐股结果：仅12个日期，按代码排列', '',
           '这些是2026-09-01至09-17的历史预测，不是当前实时概率。主目标为延迟入场后3日触及+5%。GOOG/GOOGL为两个证券代码。', '',
           '| 股票 | 样本/触及 | 实际率 | '+' | '.join(l+'均值概率' for l in lanes)+' |',
           '|---|---:|---:|'+'---:|'*len(lanes)]
    for sym in data['universe']['evaluated_symbols']:
        m=stocks[sym,lanes[0]]['metrics']
        lines.append(f"| {sym} | {m['n']}/{m['hits']} | {pct(m['observed'])} | "+' | '.join(pct(stocks[sym,l]['metrics']['predicted']) for l in lanes)+' |')
    lines += ['', '## Brier及每股过去参照', '', '| 股票 | 历史每股基准Brier | '+' | '.join(lanes)+' |','|---|---:|'+'---:|'*len(lanes)]
    for sym in data['universe']['evaluated_symbols']:
        lines.append(f"| {sym} | {num(stocks[sym,lanes[0]]['stock_reference_brier'])} | "+' | '.join(num(stocks[sym,l]['metrics']['brier']) for l in lanes)+' |')
    lines += ['', '候选池中未进入这批评估的代码：'+', '.join(data['universe']['absent_from_outer'])+'。本次没有重新判定其缺席原因。']
    (output/'STOCKS.md').write_text('\n'.join(lines)+'\n')
    return {'tables':3, 'groups':len(data['group_catalog']), 'stocks':len(data['universe']['evaluated_symbols'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(render(args.source,args.output)))
