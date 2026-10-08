"""Generate the evidence report from the final immutable artifacts."""
from pathlib import Path
import json,hashlib
import pandas as pd
OUT=Path(__file__).resolve().parent
P=lambda x:'—' if x is None else f'{x:.1%}'
PP=lambda x:'—' if x is None else f'{x*100:+.1f}pp'
CI=lambda xs:' / '.join(P(x) for x in xs)
NAMES={'continuation':'波动尺度与动量保留','repair':'下跌后的修复','relative':'相对市场定价'}
def main():
    fixed=json.loads((OUT/'results.json').read_text());monthly=json.loads((OUT/'adaptive_results.json').read_text());audit=json.loads((OUT/'audit.json').read_text());u=json.loads((OUT/'universe.json').read_text());d=json.loads((OUT/'diagnostics.json').read_text());panel=pd.read_parquet(OUT/'panel.parquet')
    assert hashlib.sha256((OUT/'universe.json').read_bytes()).hexdigest()==audit['universe_sha256']
    assert fixed['panel_sha256']==monthly['panel_sha256']==d['panel_sha256']==audit['panel_sha256']
    favorites={s for s,m in u['members'].items() if '特别关注' in m['groups'] and m.get('stock_type')=='STOCK'}
    rows=[]
    for version,r in [('固定年度',fixed),('按月更新',monthly)]:
        for key,name in NAMES.items():
            a=r['routes'][key];s=a['summary'];f=a['favorite_summary']
            rows.append(f"| {version} / [{name}]({key}.html) | {s['tp']}/{s['n']} | {P(s['precision'])} | {s['days']} | {P(s['matched_base'])} | {PP(s['lift'])} | {P(f['precision'])} / {f['n']} | {'通过' if a['pass'] else '未通过'} |")
    tests=monthly['periods']['test'];cal=fixed['periods']['calibration'];train=fixed['periods']['train']
    raw=[]
    for route,name in NAMES.items():
        a=monthly['routes'][route]
        for s in a['symbols']:
            if s['symbol'] in favorites and s['n']>=5 and s['precision']>=.7:
                raw.append(f"| {name} | {s['symbol']} | {s['tp']}/{s['n']} | {P(s['precision'])} | {CI(s['wilson'])} | {s['months']} | {s['signals_enrolled_beforehand']}/{s['n']} | {'通过' if s['pass'] else '未通过'} |")
    evidence=[]
    for route,name in NAMES.items():
        a=monthly['routes'][route];s=a['summary'];b=a['brier_improvement'];am=a['amplitude']
        evidence.append(f"| {name} | {CI(s['wilson'])} | {CI(a['lift_block']['ci'])} | {b['estimate']:.4f} [{b['ci'][0]:.4f}, {b['ci'][1]:.4f}] | {a['holm_p']:.3f} | {P(s['delayed_precision'])} | {P(s['close_up'])} | {P(s['mean_proxy_net'])} |")
    scope=[]
    cov={c['symbol']:c for c in audit['coverage']}
    for symbol in sorted(favorites):
        c=cov[symbol];scope.append(f"| {symbol} | {c.get('first_raw','—')} | {c.get('last_raw','—')} | {c.get('complete_rth',0)} | {c['eligible']} | {c.get('first_eligible','—')} / {c.get('last_eligible','—')} |")
    resolved=sum(f['status'].startswith('full_issuer_') for f in u['funds'].values());missing=[s for s,f in u['funds'].items() if not f['status'].startswith('full_issuer_')]
    qualified=len(set(panel.symbol)&favorites);raw_n=len({s['symbol'] for a in monthly['routes'].values() for s in a['symbols'] if s['symbol'] in favorites and s['n']>=5 and (s['precision'] or 0)>=.7})
    diag=[]
    for key,name in [('pre_direction_high_range','高振幅日，盘前净涨减净跌'),('late_repair','早段下跌后，末段正减负'),('relative_gap','相对QQQ跳空正减负')]:
        x=d[key];diag.append(f"| {name} | {x['n']} | {PP(x['difference'])} | {CI(x['ci'])} |")
    text=f'''# 夜盘 + 盘前 → 当天触及 +3%：修正与研究结果

本次交付三个独立、真实数据驱动的调试页面，以及固定年度/按月更新两版可复算模型。**仍未证明三路线都通过，也未证明五只特别关注股票稳定达到70%。** 高点触及、收盘方向和可执行收益分别报告，不能互换。

生成依据：`results.json`、`adaptive_results.json`、`diagnostics.json`、`audit.json`；同一面板 SHA256 `{audit['panel_sha256']}`。本次外层仍是开发期已暴露的历史，结果不是新独立前向验证。数据补齐后以相同参数复算，保留失败。

## 找到的旧研究及问题

旧实现是 [`research/after_open_3d5pct/models/premarket_tail_v1/`](../../research/after_open_3d5pct/models/premarket_tail_v1/REPORT.md)，其优化登记在同目录 `OPT_PROTOCOL.md` / `OPT_REPORT.md`。旧目标是从09:30盘前结束柱的Close出发、未来1170常规分钟（3交易日）触及+5%，21只芯片/光学/存储股票。第二轮下一周47.9%（2643信号）、再下一周49.9%（2620信号）；“路径推动为正”的方向规则39.3%/39.7%，低于同期全买39.9%/40.1%。7种正则/早停/同股排序组合都未同时胜过对照。

这些负结果本身有价值，问题是它们没有回答**当天开盘后还剩多少上涨空间**，也没有覆盖这次完整关注池。主要修正如下：

- 标签改为当天常规盘High/09:30 Open−1≥3%；09:25截断全部输入，09:30 Open只能进入标签。09:35 Open另做延迟敏感性。
- 拆开当日盘前信息和股票长期波动能力：同股匹配基准、B0同股历史常数和H-only历史振幅模型同时保留。跨股AUC不能证明同股择时；另报同股AUC。
- 概率校准和阈值选择分用两个时间块；不用外层早停、筛股或找最佳目标。固定年度结果完整保留，按月更新只用该月开始前已成熟的数据。
- 广度集合在09:25从当时已知前缀独立构建。当天未来RTH是否缺柱，只影响标签资格，不能反过来改变当时广度。此处已做修改未来/删除未来柱测试。
- 数据采用Futu原始NONE的真实5m；未知价格基础或时间字段的缓存拒绝混用。RTH必须匹配官方全部5m槽；低成交、盘前截止缺失、夜盘不足、公司行动日分别剔除审计。turnover/volume异常，未拿它制造VWAP特征。

## 三个可检验机制

1. **[波动尺度与动量保留](continuation.html)**：区分夜盘、盘前早段和末段；同时看净涨跌、路径效率、回吐与活跃量，检验上涨能否保留。净方向只是一个输入，模型可以学到方向相反的关系。
2. **[下跌后的修复](repair.html)**：看盘前谷底位置、从Low反弹、尾段恢复和相对昨收的位置。它检验修复是否意味着剩余空间，不能预设“反弹就会继续涨”。
3. **[相对市场定价](relative.html)**：个股路径减QQQ同窗路径，加当时可用池广度与分化；观察市场同步运动以外的增量。

每路线各一组固定浅树参数，均有H基准。月更新额外使用此前60个完整常规日+3%触及基准（至少40日）以及历史20日振幅归一化；H同样得到该历史基准。预期MFE的20/50/80分位另训固定模型，不当成实际成交价格。协议见 [v2](PROTOCOL.md)、[v2.1](ADAPTIVE_PROTOCOL.md)。

## 两年数据与时间切分

请求 **{audit['requested_start']}—{audit['requested_end']}**（ET），面板实际 **{audit['first']}—{audit['last']}**，{audit['rows']}合格股票日 / {audit['scored_symbols']}股票。预测比较使用QQQ夜盘和盘前均合格的共同集合，另有 {fixed['counts']['qqq_missing']} 行因QQQ不可用不参加模型比较。

固定训练实际 {train['first']}—{train['last']}：{train['n']}行；校准 {cal['first']}—{cal['last']}：{cal['n']}行。外层 **{tests['first']}—{tests['last']}**：{tests['n']}行、{tests['symbols']}股票、{tests['days']}交易日。两年包含训练与校准；外层评分不是两年全长。

月初重训：最近252个已有交易日，其末64日分32+32日，前块拟合sigmoid，后块检查0.70固定阈值可靠性并事前登记个股；之前的日子训练。每个预测行附fold、训练和校准末日。最低500训练行、每校准块100行，不足则无预测。参数和阈值不随外层成绩修改。

## 同面板前向结果

Precision=预测信号中实际当天触及+3%的比例。统计包括校准门槛失败时的研究预测，**不等于通过门槛后的可执行信号**。

| 版本 / 调试页 | TP/信号 | Precision | 信号日 | 同股匹配基准 | 增量 | 特别关注股票precision / 信号 | 总验收 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

同股基准将各股同期全部合格日触及率，按信号中的股票权重匹配；它是事后评价对照，不能进入当日决策。全部外层的无条件+3%基准 {P(tests['base'])}。仅“盘前净涨”的固定对照 {P(monthly['pre_up_baseline']['precision'])} / {monthly['pre_up_baseline']['n']}信号。

| 按月路线 | Precision Wilson95% | 同股增量日期块95% | 对H的Brier改善 [95%] | 3路线Holm p | 09:35触及 | 收盘上涨 | +3%/−1%成本代理均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(evidence)}

日期块为连续5交易日整股联合重采样1000次，保留同日横截面相关性。Wilson仅辅助展示；块区间未做同时覆盖修正。Holm只校正本版本3个机制，不覆盖此前全部研究与开发搜索；其p值和区间只能作为历史开发诊断。历史优势不能替代后续未见时期。

路线通过另要求：先前校准通过、至少300信号和60信号日、precision≥70%、同股增量CI下界>0、对H的Brier改善CI下界>0、Holm<.05。个股要求事前登记、至少30外层信号、覆盖4个月、precision≥70%且较同股基准高3pp。月更新特别关注验收还检查计入的信号均事前登记。样本和供给门槛是本研究预先增加的证据约束，不是用户给出的额外交易条件。

## 特别关注70%：点估计与证据区别

当前46个特别关注美股证券中，OpenD标为STOCK的股票 {len(favorites)} 只，本次有合格面板行 {qualified} 只。月更新任一路线至少5信号且点估计≥70%的股票去重 {raw_n} 只；下表展示全部这种候选。**这张表按外层成绩列举，属于事后观察，不是事前筛选名单或通过股票。** 零信号不能算100%；一两次全中也不列为70%候选。

| 路线 | 股票 | TP/n | Precision | Wilson95% | 月数 | 信号中事前登记/n | 严格验收 |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(raw) or '| — | 无 ≥5信号候选 | — | — | — | — | — | 未通过 |'}

三路线严格通过股：{'; '.join(name+': '+(', '.join(monthly['routes'][route]['favorite_passed']) or '0') for route,name in NAMES.items())}。这才是“至少五只股票达到有样本约束的70%”验收对应名单。

## 机制 insight：观察支持什么

额外几何诊断在看到初步模型后登记，见 [诊断协议](DIAGNOSTIC_PROTOCOL.md)，是描述性关联，不是追加策略。各股训练期固定25/75分位划分“盘前振幅/历史常规振幅”：外层低段 {d['range_low']['n']} 日，+3%触及 {P(d['range_low']['touch3'])}、收盘上涨 {P(d['range_low']['close_up'])}；高段 {d['range_high']['n']} 日，+3%触及 {P(d['range_high']['touch3'])}、收盘上涨 {P(d['range_high']['close_up'])}。上下组股票/市场状态供给仍有差别，这个差值不能当因果效果。

下表在股票×历史振幅五分位×盘前振幅五分位层内比较两侧（各至少5行），分位边界仅用训练期，按min(n正,n负)加权；显示匹配后覆盖，缺失层不外推。

| 预先固定比较 | 保留股票日 | +3%触及差 | 5日块95% |
|---|---:|---:|---:|
{chr(10).join(diag)}

- **应预测剩余上涨空间，而非把盘前上涨重复算一次。** 相对QQQ正负跳空的匹配结果允许检验追涨直觉；正向优势不能预设。开盘价已吸收盘前定价，是方向与剩余空间可能反向的一种解释，当前数据没有建立因果识别。
- **波动尺度和方向要分开。** +3%触及率、收盘上涨率并不一致；模型主要重要特征中的振幅不能单独证明方向预测。同股匹配、H对照和同股AUC是防止股票波动排序冒充择时的方法。
- **模型更新、概率可靠性和入场价影响结论。** 月更新与固定版用同一面板比较；单看某版更高precision无法说明增量通过。09:35敏感性和成本代理给出70%触及在更现实参考下的脆弱性。

这些insight支持保留三个条件机制并继续前向核验，尚不支持任意上涨延续、低位修复或相对强势就是可交易买点。分位预测的平均绝对误差、实际区间覆盖和交叉数在各路线JSON/页面证据区，不能拿预测中位幅度承诺当天涨幅。

## 覆盖与未解决项

股票池为本次OpenD全部自选 {sum('futud自选' in m['groups'] for m in u['members'].values())} 个美股证券 + 仓库QQQ成分快照 + 发行人ETF成员，去重 {len(u['members'])}。{resolved}/{len(u['funds'])} 个OpenD ETF分类条目取得完整发行人来源并筛美股成员；EWY完整来源没有美股成员，不把韩国当地股票当作美股。OpenD ETF类型还包括部分REIT，未核实者保留原始分类与缺口。

未核实全成分：{', '.join(missing)}。QQQ当前成分来源为仓库快照，不冒充本次发行人新抓取。非美股、现金和衍生品不作为美股股票样本；当前成员回溯存在存活/成员回看偏差。缺少历史池快照、实际received_at、盘口成交与新独立未来样本；没有完成全池两年夜盘覆盖。盘前柱价格可能来自更早的最后成交，页面和CSV列出正成交距09:25的分钟数，以及夜盘实际观测跨度；不把补零成交柱当作实时盘口。

只读补齐 QQQ 2026年及 IONQ、ORCL、VRT、NBIS、PLTR、TER、SMTC、RMBS、TWST、TXG、SDGR、NOK、CRWV、SNPS 的2024-08至2026-10请求，以及CBRS、LIFE的2026年请求（IPO前空数据保留空标记）；另按同一池补取2026-09-24至10-02尾部；未自动上传R2。其它现存有效源保留，多源重叠明确按cache后置优先，逐源SHA在audit。历史available_at=end+1秒是假设，故尚不能证明实盘到达及时。

| 特别关注股票 | 原始首端ET | 原始末端ET | 完整常规日 | 夜盘+盘前合格日 | 可评分首末 |
|---|---|---|---:|---:|---|
{chr(10).join(scope)}

## 调试、复算和验证

打开 [总览](index.html) 及三个路线页。默认全日真实5m聚合小时K，标盘中、盘前、盘后、夜盘、官方非交易日；提供日期起止、30/60/90/180日历天、两年固定日K mini点击/滑杆/方向键。K线区无滚轮、拖拽或双指缩放。日期下钻、当日模型输入、概率与MFE分位、失败路径、同股统计及CSV可以联动；页面切换阈值/目标/价格只评价冻结预测，明确标探索，不能当新模型回测通过。

页面自含真实压缩数据，无外部CDN。股票行情缺失保留空白，训练期不伪造样本外预测。MFE标签使用日内High，并未保证该高点可卖到。+3%/−1%代理在同一5m双触时先计止损，双边成本共0.2%，开盘穿透止损更保守；未模拟真实盘口、交易资金、订单或策略持仓。该代理不作为完整执行回测。

复算从仓库根运行，已存在cache则复用。**保留本次universe.json和本地源才能复现同一结果；重新capture是新股票池版本。**

```sh
python3 analysis/preopen_intraday_v2/capture_universe.py
python3 analysis/preopen_intraday_v2/holdings_extensions.py
python3 analysis/preopen_intraday_v2/acquire.py --symbols QQQ --start 2026-01-01
python3 analysis/preopen_intraday_v2/acquire.py --symbols IONQ ORCL VRT NBIS PLTR TER SMTC RMBS TWST TXG SDGR NOK CRWV --start 2024-08-01
python3 analysis/preopen_intraday_v2/acquire.py --symbols CBRS LIFE --start 2026-01-01
python3 analysis/preopen_intraday_v2/acquire.py --symbols SNPS --start 2024-08-01
python3 analysis/preopen_intraday_v2/acquire.py --favorites --start 2026-09-24
python3 analysis/preopen_intraday_v2/acquire_corporate_actions.py
python3 analysis/preopen_intraday_v2/build.py
python3 analysis/preopen_intraday_v2/evaluate.py
python3 analysis/preopen_intraday_v2/adaptive.py
python3 analysis/preopen_intraday_v2/diagnostics.py
python3 analysis/preopen_intraday_v2/report.py
python3 analysis/preopen_intraday_v2/render.py
python3 analysis/preopen_intraday_v2/test_causality.py
python3 analysis/preopen_intraday_v2/verify_artifacts.py
```

因果边界测试覆盖修改未来价格/量、删除未来常规柱、盘前available_at晚到、晚到中间柱量不得输入、同柱双触保守顺序、零信号、周日夜盘归属和历史触及率必须shift。冻结模型和MFE分位逐月抽样重放，核对TP/FP/信号数、面板和股票池哈希、全部源哈希，证据见run_manifest.json。浏览器交互与响应式检查结果另存ui_verification.json。根AGENTS.md已写入用户要求的交互契约。

行情接口和时段参数据 [Futu官方历史K线文档](https://openapi.futunn.com/futu-api-doc/quote/request-history-kline.html)；假日与半日市据仓库官方会话日历并参考 [Nasdaq日历](https://www.nasdaq.com/market-activity/stock-market-holiday-schedule)。ETF成员完整来源在universe.json逐基金链接，例如 [iShares SOXX](https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf)、[State Street XBI](https://www.ssga.com/us/en/institutional/etfs/state-street-spdr-sp-biotech-etf-xbi)、[First Trust GRID](https://www.ftportfolios.com/Retail/Etf/EtfHoldings.aspx?Ticker=GRID)。
'''
    (OUT/'REPORT.md').write_text(text);print(json.dumps({'report':'REPORT.md','nominal_favorite_candidates':raw_n,'strict_passed_routes':sum(monthly['routes'][r]['pass'] for r in NAMES)}))
if __name__=='__main__':main()
