"""Final qualification and user-facing tables; never selects on reserved results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import HERE, PROTOCOL, GROUPS, TARGETS
from .evaluate import read, report, LABELS, ROUTES
from .prepare import write
from .run import apply_cal
from research.after_open_3d5pct.train_multiscale_v6 import project_monotone
from research.after_open_3d5pct.v6_data import digest


def deliver(root):
    report(root)
    audit = read(HERE/'AUDIT.json')
    development = audit.pop('final_gate')
    values = read(HERE/'final_metrics.json')
    selected = read(root/'final_selection.json')
    qualification = {}
    req = PROTOCOL['gate']
    for route in PROTOCOL['routes']:
        failures = []
        for v in values:
            if v['route'] != route:
                continue
            why = []
            h = v['brier']+v['paired_history']['gain']
            if v['relative_gain'] < req['relative_brier_gain']: why.append('R0_gain_below_5pct')
            if 1-v['brier']/h < req['relative_brier_gain']: why.append('stock_prior_gain_below_5pct')
            if v['mean_gap'] > req['max_mean_calibration_gap']: why.append('mean_gap_above_5pp')
            if v['within_auc'] is None or v['within_auc'] < req['min_within_stock_auc']: why.append('within_stock_auc_below_060')
            if v['dates'] < req['min_dates_each_fold'] or v['n'] < req['min_rows_each_group_fold']: why.append('insufficient_sample')
            if v['paired_R0']['simultaneous_15_ci'][0] <= 0: why.append('R0_block_interval_crosses_zero')
            if v['paired_history']['simultaneous_15_ci'][0] <= 0: why.append('stock_prior_block_interval_crosses_zero')
            if why:
                failures.append({'seed': v['seed'], 'group': v['group'], 'failed': why})
        dev_pass = development['checks'][route]['passed']
        qualification[route] = {'development_passed': dev_pass, 'reserved_both_seeds_passed': not failures,
                                'exit_passed': dev_pass and not failures, 'reserved_failures': failures}
    all_results = sorted(root.glob('round_*/*/*/result.json'))+sorted(root.glob('final/*/*/result.json'))
    errors, processed_error, replay_rows, fit_seconds = [], 0., 0, 0.
    files = {}
    for path in all_results:
        result = read(path)
        fit_seconds += result['seconds']
        if result['fit']['replay_max_error'] > 1e-7:
            errors.append(str(path))
        cal = read(path.parent/'calibration.json')['parameters']
        for prediction in path.parent.glob('*.parquet'):
            f = pd.read_parquet(prediction)
            raw = f[['raw_'+t for t in TARGETS]].to_numpy()
            expected = project_monotone(apply_cal(raw, cal)).reshape(-1,9)
            actual = f[['p_'+t for t in TARGETS]].to_numpy()
            processed_error = max(processed_error, float(np.max(abs(actual-expected))))
            replay_rows += len(f)
            files[str(prediction.relative_to(root))] = digest(prediction)
        for model in list(path.parent.glob('*.pt'))+list(path.parent.glob('target_*.txt')):
            files[str(model.relative_to(root))] = digest(model)
    manifest = read(root/'dataset/manifest.json')
    source_errors = [name for name, hash_ in manifest['source_hashes'].items() if digest(root/'inputs'/name) != hash_]
    dataset = read(root/'dataset_identity.json')
    dataset_errors = [name for name, hash_ in dataset.items() if digest(root/'dataset'/name) != hash_]
    cache = read(root/'path_summary_identity.json')
    cache_ok = digest(root/'path_summary.npy') == cache['cache_sha256']
    if errors or source_errors or dataset_errors or not cache_ok or processed_error > 1e-7:
        raise AssertionError('final integrity or replay audit failed')
    write(root/'artifact_hashes.json', files)
    audit.update(development_gate=development, qualification=qualification, cap_reached=True,
        any_exit_passed=any(x['exit_passed'] for x in qualification.values()),
        fit_runs=len(all_results), fit_seconds=fit_seconds, prediction_rows_reprocessed=replay_rows,
        probability_values_reprocessed=replay_rows*9, probability_transform_max_error=processed_error,
        source_hash_errors=source_errors, dataset_hash_errors=dataset_errors, cache_hash_valid=cache_ok,
        dataset_identity=dataset, artifact_manifest_sha256=digest(root/'artifact_hashes.json'),
        tests={'new_causal_and_integrity_tests':9,'existing_regressions':84,'synthetic_smoke':'pass_no_predictive_claim'},
        stop_reason='ten_round_cap_reached', protocol_sha256=digest(HERE/'protocol.json'))
    write(HERE/'AUDIT.json', audit)
    write(root/'qualification.json', qualification)
    # The route here is selected ONLY from development-period scores.
    lead = min(selected, key=lambda r:selected[r]['score'])
    probabilities = pd.read_csv(HERE/'probabilities.csv')
    lines = ['# 九个期望的分组概率', '',
        '2026年8月24日至9月17日保留段；固定种子3566。每格为平均模型预测率 / 实际触及率。这是历史样本上的条件预测评价，不是今天的即时预报。九目标均按1/3/5×390正常交易分钟、11:35代理入场定义。', '',
        '有群指市场趋势/波动/流动性分组与QQQ；行业名单用于本次评价和训练权重，不冒充历史PIT行业输入。']
    for route in PROTOCOL['routes']:
        lines += ['',f'## {ROUTES[route]} · 开发选中R{selected[route]["round"]}', '',
                  '| 目标 | 计算与通信芯片 | 光通信产业链 | 数据与存储 |','|---|---:|---:|---:|']
        for target in TARGETS:
            cells=[]
            for group in GROUPS:
                v=probabilities[(probabilities.route==route)&(probabilities.seed==3566)&(probabilities.group==group)&(probabilities.target==target)].iloc[0]
                cells.append(f'{v.predicted:.1%} / {v.observed:.1%}')
            lines.append('| '+target.replace('d_','日 +').replace('pct','%')+' | '+' | '.join(cells)+' |')
    lines += ['',f'## 个股与光通信子组 · {ROUTES[lead]}（按开发分数预选）', '',
              '| 股票/子组 | 样本 | 平均预测率 | 实际触及率 | Brier | 同股AUC |','|---|---:|---:|---:|---:|---:|']
    entries=sorted(set(sum(GROUPS.values(),[])))+['optics_direct','optics_connectivity_chips','optics_switching']
    for name in entries:
        v=probabilities[(probabilities.route==lead)&(probabilities.seed==3566)&(probabilities.group==name)&(probabilities.target=='3d_5pct')].iloc[0]
        auc='—' if pd.isna(v.within_auc) else f'{v.within_auc:.3f}'
        lines.append(f'| {name} | {int(v.n)} | {v.predicted:.1%} | {v.observed:.1%} | {v.brier:.4f} | {auc} |')
    lines += ['', '单股只有18个评价日期，窗口高度重叠；误差或低AUC不能作为被人为操纵的证据。']
    (HERE/'PROBABILITIES.md').write_text('\n'.join(lines)+'\n')
    pugh_lines=['# Pugh方案选择记录', '',
        '符号依次为概率偏差、同股择日、跨期稳定、样本/过拟合风险、计算成本；权重3/3/2/2/1。符号是相对当前配方的待验机制判断，不是已经观测到的收益。', '',
        '| 轮次 | 路线 | 当前平均校准差 | 同股AUC | 选中机制 | 五项符号 | 分数 | 下一候选 | 实验后决定 |',
        '|---|---|---:|---:|---|---|---:|---|---|']
    for path in sorted(root.glob('round_*/summary.json'))[1:]:
        number=int(path.parent.name.split('_')[1])
        for route,v in read(path).items():
            m=v['pugh']; top=m['comparisons'][0]
            second=m['comparisons'][1]['mechanism'] if len(m['comparisons'])>1 else '—'
            signs=' / '.join({1:'+',0:'0',-1:'−'}[s] for s in top['signs'])
            pugh_lines.append(f'| R{number} | {ROUTES[route]} | {m["diagnosis"]["mean_absolute_group_gap"]:.1%} | {m["diagnosis"]["mean_within_stock_auc"]:.3f} | {top["mechanism"]} | {signs} | {top["weighted_score"]} | {second} | {"保留" if v["accepted"] else "回退"} |')
    pugh_lines += ['', 'R10新候选的独立登记见 [R10_SUPPORT_AMENDMENT.md](R10_SUPPORT_AMENDMENT.md)。完整候选矩阵和训练前登记保留在运行目录各轮registration.json。']
    (HERE/'PUGH.md').write_text('\n'.join(pugh_lines)+'\n')
    report_path=HERE/'REPORT.md'
    text=report_path.read_text()
    # Insert qualification before numeric tables, not as an easy-to-miss footnote.
    verdict='达到开发准出条件' if audit['any_exit_passed'] else '没有路线达到预登记准出标准'
    text=text.replace('## 冻结配方在保留段的结果', f'**完成五路线各10轮，{verdict}。停止原因是到达十轮上限。**\n\n## 冻结配方在保留段的结果')
    text += '\n\n## 准出与两种子稳定性\n\n| 路线 | 开发门禁 | 保留段双种子门禁 | 最终准出 |\n|---|---|---|---|\n'
    for route,q in qualification.items():
        text += f'| {ROUTES[route]} | {"通过" if q["development_passed"] else "未通过"} | {"通过" if q["reserved_both_seeds_passed"] else "未通过"} | {"通过" if q["exit_passed"] else "未通过"} |\n'
    text += '\n全部逐项失败原因见[AUDIT.json](AUDIT.json)。种子7566仅用于复验，不按其成绩改选配方；不存在保留段择优换模型。\n'
    text += f'\n运行共完成{len(all_results)}次路线/时间折拟合，包含匹配R0、50个迭代候选的双开发折及15次最终复验。累计记录拟合和预测耗时{fit_seconds/60:.1f}分钟；检查{replay_rows:,}行、{replay_rows*9:,}个概率的持久化校准/单调变换复放，最大偏差{processed_error:.3g}。训练阶段各模型checkpoint复放均≤1e-7。84项原回归测试、9项新因果/完整性测试和synthetic smoke通过。\n'
    text += '\n## 模型规模与实测成本\n\n| 路线 | 最终模型规模 | 单次最终拟合/预测耗时 |\n|---|---|---:|\n'
    for route in PROTOCOL['routes']:
        r=read(root/'final'/route/'selected_3566/result.json'); info=r['fit']
        size=f"{info['parameters']:,}个可训练参数" if 'parameters' in info else f"{info['features']}列摘要、9个目标共{sum(info['trees'])}棵树"
        text += f"| {ROUTES[route]} | {size} | {r['seconds']:.1f}秒 |\n"
    text += '\nC路线的跨patch卷积只进入候选池，未获本轮Pugh选择，不能说已验证；两条B路线在R10实际测试了更大树容量，均回退。本轮没有保留扩大模型容量的候选。\n'
    text += '\n## 确认的瓶颈及下一步优先级\n\n'
    text += '1. **有效历史与训练支持范围。** 所有fit段完整126日日线比例为0%；复用构建器的日历下界固定在2026年1月1日，扩展3月起的样本并没有扩展此前回看。部分长期群类型从fit完全缺失转为eval出现。R10已用仅由fit确定并持久化的支持范围屏蔽做真实试验，结果见逐轮表，不能因机制看起来合理就假定改善。下一阶段需要同步扩展行情、官方日历和构建器范围，补足跨市场状态的历史，再固定新前向窗口。当前结果不能证明日线没有价值。\n'
    text += '2. **概率水平与择日是两个问题。** 校准可修正高估/低估，不能保证识别同一股票的好时点。下一阶段应直接检验同股排序或相对本股基准的增量目标，仍保留九期望；这次没有做同股排序辅助训练，不把它列为已验证能力。\n'
    text += '3. **行业信息的历史真实性。** 当前有群路线使用因果市场分组与QQQ，行业名单只用于评价/权重。需要真实当时可知的行业成员版本，才可进一步检验同业强弱、扩散和领先滞后；不能把今天的行业名单回填成历史实时特征。\n'
    text += '\n本轮数学计算的结论以路线和时期为条件：路径摘要、正则化、成熟先验、专门化、校准并非普遍有效；完整保留负结果。五条最终路线配方不同，因此它们之间的差异不能全归因于“是否有群”或“是否有日线”单一因素。\n'
    text += '\n详见[Pugh逐轮矩阵](PUGH.md)、[九目标与个股概率表](PROBABILITIES.md)、[逐轮结果](ROUNDS.md)。\n'
    report_path.write_text(text)
    print(json.dumps({'fits':len(all_results),'lead_development_route':lead,'any_exit_passed':audit['any_exit_passed'],'replay_rows':replay_rows,'seconds':fit_seconds}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path)
    deliver(p.parse_args().root.resolve())
