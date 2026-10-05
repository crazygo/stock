"""Write the final acceptance audit from completed artifacts, not aspirations."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import OUT,write,now,sha
P=OUT/'R06'

def pct(v):return '不可评分' if v is None else f'{v:.2%}'

def persistent_overestimate(audit):
    previous=None
    for m in audit.get('monthly_high_probability',[]):
        ci=m.get('overestimate_ci',[None,None]);month=int(m['month'][:4])*12+int(m['month'][5:7])
        clear=m.get('stock_days',0)>=40 and m.get('dates',0)>=10 and m.get('predicted',0)-m.get('observed',0)>.05 and ci[0] is not None and ci[0]>0
        if clear and previous==month-1:return True
        previous=month if clear else None
    return False

def main():
    result=json.loads((P/'results.json').read_text());rows=[]
    for arm in result['arms']:
        for v in arm['variants']:
            m=v['metrics'];fail=[]
            for condition,name in [(m['mature']>=100,'成熟信号至少100'),(m['mature_date_count']>=40,'成熟信号日期至少40'),((m['precision'] or 0)>=.8,'达成率至少80%'),((m['block_ci'][0] or 0)>=.7,'日期块下限至少70%'),((m['lift'] or 0)>=.05,'匹配基准增量至少5pp'),(m['weeks']>=12,'评价连续至少12周'),(m['signals_per_week']>=3,'平均每周至少3信号'),(len(v['favorite_five_passed'])>=5,'固定五股各n至少20且达成率至少75%'),(v['signal_probability']['reliable_high_probability'],'已发信号概率可信门禁'),(v['first_candidates']['reliable_high_probability'],'第一名概率可信门禁')]:
                if not condition:fail.append(name)
            if any(persistent_overestimate(v[k]) for k in ['first_candidates','signal_probability']):fail.append('高概率区间连续两月明显高估')
            v['historical_all_constraints_met']=not fail
            v['acceptance_failures']=fail;rows.append(dict(arm=arm['arm'],variant=v['variant'],metrics=m,passed=not fail,failures=fail))
    write(P/'results.json',result)
    eligible=[r for r in rows if r['arm'] not in ['T0','TC'] and r['metrics']['mature']>=100]
    best=max(eligible,key=lambda r:r['metrics']['precision'] or 0) if eligible else None
    sources=[dict(file=name,sha256=sha(P/name)) for name in ['coverage.json','calibration_audit.json','verification.json','control_verification.json','portable_verification.json','input_stability_audit.json','api_audit.json','training_coverage.json','source_quality.json','ACCEPTANCE_IMPLEMENTATION.md']]
    write(P/'delivery_manifest.json',dict(at=now(),status='completed_development',models=40,variants=80,historical_passed=sum(r['passed'] for r in rows),independent_pass=False,best_added_information_with_n100=best,acceptance=rows,sources=sources,paid_data_cost=0,current_probability=False,issued_signal=False,orders_sent=False))
    lines=['# R06：新历史信息训练结果','','富途四类历史接口已完成固定124股采集、八组输入×五月至九月共40份实际月模型及80份校准变体，复算与原生导出完成。全部为已暴露历史开发；尚未完成冻结后的独立验证。','',f"完整历史验收通过 {sum(r['passed'] for r in rows)}/{len(rows)}；未来独立通过0。当前有效门槛为空、弃权。",'','| 输入 / LightGBM | 校准 | TP/成熟（未知） | 达成率 | 95%日期块区间 | 成熟日期 | 信号/周 | 匹配基准增量 | 第一名高概率：预测/实际 |','|---|---|---:|---:|---|---:|---:|---:|---|']
    for arm in result['arms']:
        for v in arm['variants']:
            m=v['metrics'];high=v['first_candidates'].get('high_probability',{});ci='—' if m['block_ci'][0] is None else f"{pct(m['block_ci'][0])}–{pct(m['block_ci'][1])}"
            lines.append(f"| {arm['arm']} {arm['name']} | {v['variant']} | {m['tp']}/{m['mature']}（{m['pending']}） | {pct(m['precision'])} | {ci} | {m['mature_date_count']} | {m['signals_per_week']:.2f} | {pct(m['lift'])} | {pct(high.get('predicted'))}/{pct(high.get('observed'))} |")
    if best:
        m=best['metrics'];lines+=['',f"新增信息中、成熟n≥100的描述性最高结果：{best['arm']} / {best['variant']}，{m['tp']}/{m['mature']}={pct(m['precision'])}；日期块区间{m['block_ci']}。这是评价后摘要，不赋予部署资格。未满足：{'；'.join(best['failures']) or '历史门禁已满足，但未来独立验证未完成'}。"]
    lines+=['','TC控制共有覆盖/缺失/陈旧信息；单族与联合输入相对于TC才是新增数值信息对照。T0与旧H3逐行一致，不能算新市场证据。相同候选Brier变化、各月方向、日期块区间及Holm见[INPUT_STABILITY.md](INPUT_STABILITY.md)和[CALIBRATION.md](CALIBRATION.md)；不同已发信号群体的precision变化不能直接当因果增量。','','数据边界：资金流只有最近一年，真实turnover分母覆盖另受归档限制；其他三族主要达到两年。原始历史接收/修订版本不可认证，2/5交易日延迟是预先登记的开发假设。延期敏感性改善也不认证PIT。财报共识、宏观原始惊喜、历史盘口等逐项排除并结案，见[API_AUDIT.md](API_AUDIT.md)。','','完整周、月、逐股、当前全部特别关注、固定五股、弃权、未知及同股五日不重叠结果在results.json；所有候选/第一名/已发信号分箱与逐月高概率偏差在calibration_audit.json。基准沿用训练期同股、分钟、波动分层估计，不能提升为当期实际成交或因果收益。','','本轮零付费已完成数据增量对照，没有证据要求购买相同字段。付费原始预期版本/历史NBBO仍须具体样本证明；本轮没有采购或下单，也没有R2上传。','','统一历史参考命令：`python3 analysis/preopen_5d5pct_v6/recommend.py --round R06 --top 3`。最新可用完整行情为十月2日，明确不是当下概率；原十月配置不改。调试页：只读服务8771的`/futu_data.html`。']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (P/'REVIEW.md').write_text('# R06 验收结案\n\n'+f"40/40模型及80校准变体完成；{sum(r['passed'] for r in rows)}/16路线校准组合通过完整历史门禁，0通过未来独立门禁。"+'\n\n'+ '\n'.join(f"- {r['arm']} / {r['variant']}："+('历史条件满足，独立验证未完成' if r['passed'] else '未达标：'+'、'.join(r['failures'])) for r in rows)+'\n\n协议没有留在建议阶段。接口不可用/时点不可信者已明确排除；可用四族已训练、评分、校准审计、复算并导出。所有原结果保留，未知未删，失败计FP，无后验股票筛选或阈值调整。工程检查不代表模型有效。\n')
    print(json.dumps(dict(status='completed',best=best,passed=sum(r['passed'] for r in rows))),flush=True)

if __name__=='__main__':main()
