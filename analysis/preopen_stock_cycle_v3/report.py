"""Evidence-led delivery: five distinct stocks, all failures and limits."""
import json
import pandas as pd
import numpy as np
import build as b
OUT=b.OUT
PICKS=[('AAOI','2026-01-26','solo_forest'),('AXTI','2025-10-28','logistic_solo_1'),('COHR','2026-01-26','analog_solo_15'),('IONQ','2026-05-26','macro_related_long'),('LIFE','2026-07-15','related_long')]
def pct(x):return '—' if x is None else f'{x*100:.1f}%'

def main():
 delivery=json.loads((OUT/'delivery_combinations.json').read_text());r=json.loads((OUT/'results.json').read_text());ar=json.loads((OUT/'alignment_results.json').read_text());audit=json.loads((OUT/'audit.json').read_text());ab=json.loads((OUT/'ablation.json').read_text())
 pred=pd.concat([pd.read_parquet(OUT/'enrolled_predictions.parquet'),pd.read_parquet(OUT/'alignment_enrolled_predictions.parquet')],ignore_index=True)
 picks=[next(c for c in delivery['passed'] if (c['symbol'],c['start'],c['selected'])==k) for k in PICKS];diagnostics=[];signals=[]
 for c in delivery['passed']:
  s=c['symbol'];g=pred[pred.symbol.eq(s)&pred.cycle_start.eq(c['start'])&pred.model.eq(c['selected'])&pred.signal];f=pd.read_parquet(OUT/'raw'/f'{s}.parquet');rth=f[(f.minute>=570)&(f.minute<960)].groupby('day');details=[]
  for row in g.itertuples():
   day=rth.get_group(row.day);touch=day[day.high>=row.reference*1.03];stop=touch.end.iloc[0] if len(touch) else day.end.iloc[-1];before=day[day.start<stop]
   details.append(dict(day=row.day,hit=int(row.y),touch_volume=float(touch.volume.iloc[0]) if len(touch) else None,holding_upper_minutes=float((stop-pd.Timestamp(row.day+' 09:30')).total_seconds()/60),mae_before_exit_bound=float(before.low.min()/row.reference-1)))
  assert all(d['touch_volume'] is None or d['touch_volume']>0 for d in details)
  diagnostics.append(dict(symbol=s,start=c['start'],model=c['selected'],mean_holding_upper_minutes=float(np.mean([d['holding_upper_minutes'] for d in details])),worst_preexit_mae_bound=float(min(d['mae_before_exit_bound'] for d in details)),details=details));signals.append(g)
 allsig=pd.concat(signals,ignore_index=True);unique=allsig.drop_duplicates(['symbol','day']);counts=dict(combination_signal_cases=len(allsig),unique_stock_days=len(unique),duplicates=len(allsig)-len(unique),qualified_combinations=len(delivery['passed']),qualified_stocks=len(delivery['distinct_stocks']))
 (OUT/'execution_diagnostics.json').write_text(json.dumps(dict(counts=counts,combinations=diagnostics,interpretation='bar-close holding upper bound; touch-bar low conservative because within-bar order unknown; no execution proof'),ensure_ascii=False,indent=2))
 lines=['# 盘前量价 → 当天触及 +3%：五股 / 30 天周期研究结果','',
  f'已找到 **{len(delivery["passed"])} 个历史合格模型组合，覆盖 {len(delivery["distinct_stocks"])} 只特别关注股票：'+', '.join(delivery['distinct_stocks'])+'**。主标签严格采用用户明确的「当天触及 +3%」：09:25:30 ET 截止输入，09:30 Open 起算，官方常规盘内 High 触及 +3%。这不是三日 +5%，不是收盘上涨。', '',
  '**这满足历史组合的研究门禁，不代表已找到下一周期可直接交易的 70% 策略。** 原固定锚点的唯一择模策略仍只有 '+f'{r["policy"]["tp"]}/{r["policy"]["n"]} = {pct(r["policy"]["precision"])}'+ '；所列赢家周期是事后发现。已暴露历史、多轮候选与两个额外锚点使选择偏差增加，不能称独立验证。', '',
  '[本地研究总览](http://127.0.0.1:8768/analysis/preopen_stock_cycle_v3/index.html) 提供 14 个合格组合页面、6 个独立思路页面，以及所有已登记 / 未通过周期。', '',
  '## 五个不同股票的代表组合','',
  '所有代表：周期开始前登记模型及阈值；前置选择至少 8 个不同日期信号且 precision >=70%、比基准高 >=3pp；外层至少 14 个合格日、12 个不同日期信号、precision >=70%、比同股同周期基准高 >=3pp、20bp 成本代理均值 >0。门禁从首轮冻结后未调低。', '',
  '| 股票 / 调试页 | 30 天周期 | 模型 | 命中 / 信号 | precision | 同股基准 / 增量 | 09:35 敏感性 | Wilson 95% 区间 |','|---|---|---|---:|---:|---:|---:|---:|']
 for c in picks:
  m=c['metrics'];path=f'{c["symbol"]}_{c["start"]}_{c["selected"]}.html';lines.append(f'| [{c["symbol"]}]({path}) | {c["start"]}—{c["end"]} | {r["models"][c["selected"]]} | {m["tp"]}/{m["n"]} | {pct(m["precision"])} | {pct(m["base"])} / +{m["lift"]*100:.1f}pp | {pct(m["delayed_precision"])} | {pct(m["ci"][0])}—{pct(m["ci"][1])} |')
 lines += ['', '五个页面分别覆盖非线性单股、平滑单股、相似日、相关股风险环境、相关股行业共享。LIFE 的窗口属于预先写入第三轮协议的额外固定锚点；其前置模型确实由过去选择，窗口相对首轮不是独立新数据。AAOI 的该模型属于前置合格候选，未被唯一择模器选为当期第一名；不能把“候选有成功模型”混同“择模器能提前选中它”。', '',
  '## 发现了什么','',
  '1. **这个目标首先是振幅问题。** 命中 +3% 不要求收盘上涨，也可能先大跌再反弹。代表周期同股基础触及率已约 62%—82%；不比较基准，很容易把高波动股票的自然触及率当成盘前预测力。当前有意义的组合既过 70%，也过 >=3pp 增量，但选择这些周期仍是后验行为。', '',
  '2. **加更多分钟参数，并没有在所有前置入选周期里证明增量。** 对 v3.1 全部 59 个入选股票周期 / 1207 个逐日预测，完整特征 Brier '+f'{ab["full_brier"]:.6f}，历史 H-only {ab["variants"]["H_only"]["brier"]:.6f}'+ '，差异区间跨零。去分钟反而点估计略好；量和相关股的增量区间也跨零。分钟滑窗值得逐股检查，但“加量价就一定提高准确度”被本轮总体审计否定。', '',
  '3. **过去校准冠军不一定适应下一周期。** 单一择模器原周期只有 58.24%，而全部前置登记候选里有 10 个合格组合；择模本身比找到一个事后表现好的模型更难。宏观候选让 IONQ 的 2026-05-26 周期成为合格组合；这是具体实例，尚不能推出 VIX 对所有股票有效。', '',
  '4. **周期边界会影响入选。** 按原锚点 LIFE 没过；固定额外锚点中的 2026-07-15—08-13 相关股模型 13/16=81.25%。说明应该展示周期对齐敏感性，而不是只展示某一个月份。每个模型的阈值始终来自周期之前；额外锚点的失败窗口也全部保留。', '',
  '5. **触及预测与赚钱之间仍有明显缺口。** LIFE 从 09:30 的 81.25% 降为 09:35 的 62.5%；IONQ 75% 降为 66.7%。全天最大下探及等待时间不能忽略。即使价格触及，也未证明订单能成交、止损能承受或资金可以稳定复用。', '',
  '## 失败样本给出的新假设','',
  '**成交量的含义需要因股而异。** AAOI 代表周期 13 次命中的盘前相对量均值约 1.22 倍，2 次失败却为 1.81 倍；AXTI 则相反，14 次命中约 3.64 倍，唯一失败仅 0.84 倍。全池统一的“放量即确认”会把两种状态混在一起。样本很小，这些是下一轮应冻结验证的假设，不是当前可新增的过滤门禁。', '',
  '**盘前已涨，未必还有从开盘出发的空间。** AAOI 2026-02-03 在 09:25 相对前收已 +4.51%、盘前量 1.82 倍，但开盘后最高只 +2.68%；2026-02-04 盘前仍 +1.41%、末 30m 相对量 3.47 倍，却转为尾窗下跌 0.60%，当天最高仅 +0.02%、收盘 -15.11%。值得检验的量价交互是“事件已提前定价 + 尾段卖出/失速”，不是继续奖励总量。', '',
  '**低基数会制造看似很强的量比。** LIFE 2026-07-15 的末 30m 相对量达 348 倍、末段上涨 1.52%，却只在常规盘触及 +0.90%，收盘 -8.27%；另外两次失败的末 30m 相对量为 0 和 0.28 倍。需要同时检查绝对量、交易年龄和先前同窗基数。完整分钟时间戳与高相对量都不能单独代表可成交深度。页面保留真实零成交柱，没有填造量。', '',
  '## 分钟特征与失败检查','',
  '真实粒度为 5 分钟，不是把 60m 数据插值成分钟。10 / 15 / 30 / 45 / 60 / 90 / 120 / 180 分钟窗口截至 09:25；15 / 30 / 60 分钟窗口在盘前每 5 分钟滑动，提取最大 / 最小涨幅及末段排名。每股每天只计一次预测。窗口中的 OHLCV、振幅、效率、修复幅度、成交相对量、有向量代理、集中度和近似 bar-VWAP 均可复算。缺夜盘和真实零成交分别编码，不再凭夜盘低活跃删掉可用盘前样本。', '',
  '页面用完整常规盘阴影画预测会达标区间，失败用斜线；+3% 目标线及首次触及在事后路径中标注。点击失败日进入 5m 视窗，检查量柱、盘前末 30m 收益、盘前相对量、行业 breadth、此前基础触及率、QQQ/VIX 环境。成功 / 失败特征均值仅用于找新假设，不是因果结论。', '',
  '## 价格路径 / 成本边界','',
  '| 股票 | +3% / 收盘退出 20bp 代理均值 | 09:35 成本代理 | 全天最差下探 | 触及前最低价保守界 | 平均等待上界（分钟） |','|---|---:|---:|---:|---:|---:|']
 for c in picks:
  m=c['metrics'];d=next(x for x in diagnostics if x['symbol']==c['symbol'] and x['start']==c['start'] and x['model']==c['selected']);lines.append(f'| {c["symbol"]} | {pct(m["cost_proxy"])} | {pct(m["delayed_cost_proxy"])} | {pct(m["worst_mae"])} | {pct(d["worst_preexit_mae_bound"])} | {d["mean_holding_upper_minutes"]:.1f} |')
 lines+=['', '成本代理：命中时 +3% 减 20bp；未命中则从参考价持有至常规盘收盘，再减 20bp。没有止损，也没有真实撮合。最低价保守界包含首次触及的整根 5m K，因无法知道柱内先后；等待时间取触及柱结束，否则官方收盘。全部首次触及柱有正成交量，仍不等于可成交证明。', '',
  '## 全部尝试与覆盖','',
  f'- 首轮六模型：608 个股票周期；193/324=59.57%，主策略合格 0。紧凑扩展：329/570=57.72%，主策略合格 1。宏观扩展：350/601=58.24%，主策略合格 3，逐模型登记合格 10。',
  f'- 两个额外起点：{len(ar["cells"])} 个股票周期；564/965=58.45%，主策略合格 2，逐模型合格 4。额外锚点之间相互重叠，其合计是窗口评分次数，不是独立机会次数。',
  f'- 共 {len(r["enrolled_combinations"])+len(ar["enrolled_combinations"])} 个前置登记模型，{counts["qualified_combinations"]} 个通过。通过组合中的 {counts["combination_signal_cases"]} 次模型信号只对应 {counts["unique_stock_days"]} 个不同股票日期，{counts["duplicates"]} 次为重复；不能简单累加。',
  '- 股票池快照是 futud 自选 + QQQ + 自选 ETF 成分，1858 名称；本轮重点 32 只特别关注 STOCK，另用 QQQ、固定相关股、SOXX/IGV。新补 RGTI/QBTS/SOXX/IGV 2024-08 至 2026-10-02 ALL/NONE 行情，Local→R2→OpenD，不自动推送。IPO 前不造数据，低成交与真实缺失分别审计。全 1858 成分尚未验证，当前成员回溯不是历史 PIT。',
  '- Cboe VIX/VVIX/VIX9D 免费日数据到 2026-10-02，只映射前一官方交易日；不把当天收盘当盘前输入。来源：[Cboe 历史波动指数](https://www.cboe.com/tradable_products/vix/vix_historical_data)。',
  '- Futu bar_end 与 end+1s 可用时点为研究假设，未证明真实接收延迟。半日市、假日、ET DST、周日夜盘归属周一按官方日历处理。原始行情 / 模型被忽略，不进入 Git。', '',
  '## 验证与复算','',
  '8 个因果 / 缺失测试通过：未来量价、未来标签完整性、晚到盘前、安静与缺失夜盘、相关股状态、单股日唯一计数、宏观同日收盘无泄漏。396 个模型逐一重放到 1e-12，1410 个来源文件加 3 个宏观来源哈希核对。浏览器验收记录见 ui_verification.json，包含日期快捷项、两年 mini、无鼠标缩放、失败聚焦、平移、粒度、导出和桌面 / 手机溢出。', '',
  '```bash','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/build.py','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/enrich_macro.py','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/evaluate.py','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/evaluate.py --alignment','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/test_causality.py','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/render.py','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/verify_artifacts.py','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/verify_ui.py','/opt/homebrew/bin/python3 analysis/preopen_stock_cycle_v3/report.py','```', '',
  '初始六模型和紧凑版本的结果与预测分别保留为 results_initial / results_compact 及对应 Parquet；宏观快照另存 results_macro。每轮先写协议再运行，不能以最后成功组合覆盖前面的失败。信息量消融冻结于 ABLATION_PROTOCOL.md，针对 v3.1 已登记队列，属于诊断，不冒充新独立验证。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'representatives.json').write_text(json.dumps(picks,ensure_ascii=False,indent=2));(OUT/'README.md').write_text('# 单股 / 相关股 × 30 天周期盘前研究\n\n先读 [结果与边界](REPORT.md)，再打开 [调试总览](http://127.0.0.1:8768/analysis/preopen_stock_cycle_v3/index.html)。14 个历史合格组合覆盖 5 只特别关注股票；21 个固定候选配置归为 6 个思路，各有独立 wireframe。整体择模策略仍未达到 70%，不能作为可直接交易的完成验证。\n\n协议顺序：PROTOCOL → COMPACT_PROTOCOL → MACRO_ENROLLMENT_PROTOCOL → ALIGNMENT_PROTOCOL。STATE 记录起始问题，所有失败保留。审计见 audit / run_manifest / ui_verification / execution_diagnostics。页面单文件自包含，原始行情和模型不进 Git。\n')
 print(json.dumps(dict(stocks=delivery['distinct_stocks'],representatives=len(picks),counts=counts)))
if __name__=='__main__':main()
