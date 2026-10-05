"""Evidence-led delivery snapshot; no selection or threshold changes."""
import argparse,json,subprocess
from collections import Counter
from common import OUT,sha,write,now

def unmet(variant):
    """Explain the registered gate without changing any selection or result."""
    m=variant['metrics'];p=m.get('precision');ci=m.get('block_ci') or [None,None]
    checks={
        'mature_signals_100':m['mature']>=100,
        'precision_80pct':p is not None and p>=.8,
        'mature_signal_dates_40':m.get('mature_date_count',m['date_count'])>=40,
        'evaluation_weeks_12':m['weeks']>=12,
        'average_signals_per_week_3':m['signals_per_week']>=3,
        'date_block_lower_70pct':ci[0] is not None and ci[0]>=.7,
        'matched_lift_5pp':m.get('lift') is not None and m['lift']>=.05,
        'predeclared_focus_five':len(variant.get('favorite_five_passed',[]))>=5,
        'issued_high_probability_audit':variant.get('signal_probability',{}).get('reliable_high_probability',False),
    }
    return [k for k,passed in checks.items() if not passed]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');a=ap.parse_args()
    rounds=[];comparisons=[];checks=[];artifacts=[]
    for rid in ['R00','R01','R02','R03','R04','R05']:
        base=OUT if rid=='R00' else OUT/rid;path=base/'results.json'
        if not path.exists():continue
        result=json.loads(path.read_text());months=sum(sum(s=='completed_development' for s in arm['statuses']) for arm in result['arms'])
        rows=[]
        for arm in result['arms']:
            for variant in arm['variants']:
                rows.append(dict(arm=arm['arm'],name=arm['name'],algorithm=arm['algorithm'],variant=variant['variant'],metrics=variant['metrics'],historical_all_constraints_met=variant.get('historical_all_constraints_met',False),unmet_requirements=unmet(variant),statuses=arm['statuses']))
        rounds.append(dict(round=rid,monthly_models=months,variants=rows,independent_pass=False))
        audit=base/'calibration_audit.json'
        if audit.exists():comparisons.extend(dict(round=rid,**r) for r in json.loads(audit.read_text())['audits'] if r.get('comparison')=='input_on_identical_all_candidate_rows')
        verification=base/'verification.json'
        if verification.exists():checks.append(dict(round=rid,**json.loads(verification.read_text())))
        for p in [path,base/'PROTOCOL.md',base/'coverage.json',verification,audit]:
            if p.exists():artifacts.append(dict(file=str(p.relative_to(OUT)),sha256=sha(p)))
    acquisition=[json.loads(p.read_text()) for p in (OUT/'cache/acquired').glob('*.json')]
    prefix=[json.loads(p.read_text()) for p in (OUT/'cache/backfill').glob('*.json')]
    universe=json.loads((OUT/'universe.json').read_text())
    engineering={}
    for name in ['portable_verification.json','ui_verification.json','service_verification.json','inference_verification.json','tests_verification.json']:
        p=OUT/name
        value=json.loads(p.read_text())
        engineering[name]=dict(status=value['status'],checks=len(value['checks']),sha256=sha(p))
        artifacts.append(dict(file=name,sha256=sha(p)))
    for name in ['input_stability_audit.json','acquisition_pause.json','history_acquisition.json','data_quality.json','runtime_evidence.json','scope_verification.json','BLOCKERS.md']:
        p=OUT/name
        if p.exists():artifacts.append(dict(file=name,sha256=sha(p)))
    quality=json.loads((OUT/'data_quality.json').read_text())
    quality_summary=dict(symbol_files=len(quality['records']),invalid_bars=sum(r.get('invalid_bars',0) for r in quality['records']),quality_errors=sum('error' in r for r in quality['records']))
    pause=json.loads((OUT/'acquisition_pause.json').read_text())
    report=dict(at=now(),window_status='final_snapshot' if a.final else 'ongoing',window_deadline='2026-10-05T12:59:58+08:00',independent_pass=False,
        rounds=rounds,input_comparisons=comparisons,verification=checks,engineering=engineering,source_artifacts=artifacts,
        current_membership_universe=len(universe['members']),historical_acquisition_counts=dict(Counter(r['status'] for r in acquisition)),
        priority_prefix_symbols=len(prefix),prefix_bars_added=sum(r['added_prefix_bars'] for r in prefix),
        paid_data_purchased=False,orders_sent=False,r2_uploaded=False,legacy_modified=False,
        current_intraday_probability_service_deployed=False,all_future_effective_thresholds_empty=True,
        data_quality=quality_summary,acquisition_pause=pause,
        caveat='Different arms/calibration/months overlap; fitted model count and summed engineering events are not independent statistical samples. Final independent evaluation requires frozen new data for at least twelve weeks.')
    write(OUT/'delivery_manifest.json',report)
    lines=['# 五日 +5% 免费数据研究交付','','当前有效操作入口：弃权。已暴露历史开发研究不能赋予未来买入资格。80%验收目标尚未完成独立证明。','',
      f"本次快照 {report['at']}；12小时窗口截止北京时间2026-10-05 12:59:58。只在main新增v6，保留旧模型/结果/十月配置及外部WIP。没有付费、下单或上传R2。",'',
      '| 轮次 | 问题 | 实际月度模型 | 完整历史门禁 |','|---|---|---:|---|']
    names={'R00':'固定算法比较信息；再固定信息比较算法','R01':'真时间前向日线机会OOF与分钟择时','R02':'精确30分钟板块持续性；原申报延续','R03':'更长历史；已成熟五日状态；日线','R04':'原季度实际同比，无历史共识','R05':'前日多日板块广度与领涨持续性'}
    for r in rounds:
        passed=sum(v['historical_all_constraints_met'] for v in r['variants']);lines.append(f"| {r['round']} | {names[r['round']]} | {r['monthly_models']} | {passed}/{len(r['variants'])} 完整通过；独立验证未通过 |")
    lines+=['','拟合数量不是市场证据份数。每月模型、校准变体及股票窗口重叠，不把它们合计成独立信号。周、月、逐股、零信号周、所有特别关注及第一名概率见各轮REPORT/CALIBRATION/results。delivery_manifest逐组合列出未满足的约束；不可评分也不判通过。连续周数是评价范围，零信号周保留，另报信号日期和平均供给。','',
      '| 相同候选比较 | Brier新减旧（负值更好） | 两周日期块95% |','|---|---:|---|']
    for c in comparisons:lines.append(f"| {c['round']} {c['newer']} / {c['reference']} | {c['brier_delta']:.6f} | [{c['brier_delta_ci'][0]:.6f}, {c['brier_delta_ci'][1]:.6f}] |")
    lines+=['','以上只比较相同评价行，不用不同已发群体的precision差值推断信息增量。Brier兼有辨别与校准，概率分箱、第一名及已发高概率偏差另报。','',
      '在成熟信号至少100个的组合中，最高描述性达成率为H1的388/507=76.53%：58个成熟信号日期、日期块95%区间66.67%–85.51%，未到80%且下限未到70%。H3高概率候选平均报83.30%，实际69.45%；额外第一名校准后高概率区间平均84.92%，实际63.90%。概率仍不能称可信。校准第一名群体与最终有门槛已发群体还存在差别，已单独审计。','',
      '日线增量H3/H2在五个月均改善Brier，两周块区间为负，但15次输入/算法比较的Holm校正p=0.6006；未形成稳定显著增量证据。全部失败保留，见INPUT_STABILITY.md。','',
      '免费数据：Futu105周28685财报记录、124股行动；SEC57股原申报及325原季度事件。预登记五股最新季度营收/EPS抽查原文；其他事件不是全部原文核验。共识历史版本、最早财报发布、历史成员PIT、历史BBO/深度尚缺。季度实际值没有明确稳定增量不等于付费预期/新闻无价值；是否值得付费需先拿样本做独立增量对照。','',
      f"当前成员池{report['current_membership_universe']}证券（非历史PIT）；历史下载状态{report['historical_acquisition_counts']}。已有注册池优先前缀完成{len(prefix)}/124证券，新增{report['prefix_bars_added']}条。状态与边界不证明全日或五日可评分；data_quality逐股另列。",'',
      f"下载已在{pause['at']}的完整股票边界暂停；全成员清单保留未完成队列，可断点续传。{quality_summary['symbol_files']}份源文件覆盖终审发现{quality_summary['invalid_bars']}根无效价格占位线，{quality_summary['quality_errors']}质检运行错误；无效源线保留，不画作真实K线。当前成员池尚未全部取得两年可评分历史。",'',
      '旧当天+3%对照：相关股84/95=88.42%，仅16个信号日期，两周块下限约42.3%；虽有后续五日增量线索，旧概率没有针对五日校准。恢复73/90=81.11%，匹配增量约2.95pp。都不足新门禁，不能把两种目标不同样本直接判胜负。详见OLD_CONTROL.md。','',
      '当前盘口五股可查询且临时订阅已取消；三股服务器时间为空，其他时间字符串时区未证实，没有历史盘口版本。触及标签、可成交性与实际收益分开。','',
      '工程复算：'+str(sum(c['events_checked'] for c in checks))+'个路线/校准内信号、'+str(sum(c['ranking_ticks_checked'] for c in checks))+'排名时点；状态'+str({c['round']:c['status'] for c in checks})+'。23项原生模型/上游数值检查、8项浏览器双宽度检查、18项只读API场景、2项推断/缺数据弃权检查和11项因果/分母测试通过。工程通过不证明模型有效。','',
      '机器03:02:27因低电量休眠，10:34:07恢复（pmset系统证据，约7小时32分）；期间没有计算或实时采集。恢复后按已保存符号继续，剩余窗口临时防空闲休眠。R2 503已保留并修复回退。','',
      '后续硬条件：确认未暴露评价范围或冻结后的新数据；连续至少12周和全部样本/供给/概率门禁。免费源不足处保留缺失；只有确需购买且拿到具体报价时才需要预算确认。当前失败可以继续免费研究，不能把失败或概率数值包装为买入建议。','',
      '```bash','python3 analysis/preopen_5d5pct_v6/recommend.py --top 3 --round all','python3 analysis/preopen_5d5pct_v6/server.py --port 8771','```','',
      '调试入口：http://127.0.0.1:8771/ ，三思路各有页面；日期/两年mini/全日分钟/禁缩放/成功失败五日路径可检查。统一命令为研究参考，有效门槛为空；闭市历史截止清楚列出。当前没有部署当下盘前/盘中每五分钟自动刷新概率服务；统一命令用最新已完常规盘历史截止输出各模型前三。后续实时与独立验证须另行冻结协议，见BLOCKERS.md。']
    (OUT/'DELIVERY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rounds=len(rounds),monthly_models=sum(r['monthly_models'] for r in rounds),priority_prefix_symbols=len(prefix),status=report['window_status'])),flush=True)
if __name__=='__main__':main()
