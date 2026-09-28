"""Self-contained result explorer plus evidence-linked factual report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from research.group_expectation_matrix.build import write_json
from .facts import clean

NAMES={"d3_r5":"3 日 +5%","d5_r8":"5 日 +8%","d21_r10":"21 日 +10%"}
ROUNDS={"R0":"旧权重重放","R1":"覆盖与时效","R2":"路径与同群","R3":"内层训练选择"}
def percent(v): return "—" if v is None else f"{v*100:.2f}%"
def number(v): return "—" if v is None else f"{v:.4f}"

def render(output):
    def read(name): return json.loads((output/name).read_text())
    results,summary,facts = read("results.json"),read("summary.json"),read("facts.json")
    selected = [r for r in results if r["group_id"]=="optics:chain" and r["algorithm"]=="lgbm" and r["granularity"]==5 and r["scope"]=="pooled"]
    losses=[]
    for r in selected:
        if r["round"] != "R3" or r["window"] != "common": continue
        detail=read(f"details/{r['id']}.json")
        tr=pd.DataFrame(detail["trades"])
        tr["net_pct"]=tr.pnl/tr.entry_debit
        gains=tr[tr.hit]; lost=tr[~tr.hit]
        losses.append({"expectation":r["expectation"],"trade_n":len(tr),"wins":len(gains),"losses":len(lost),
                       "mean_win":float(gains.net_pct.mean()),"mean_loss":float(lost.net_pct.mean()),
                       "gain_dollars":float(gains.pnl.sum()),"loss_dollars":float(lost.pnl.sum()),
                       "largest_losses":tr.sort_values("pnl").head(5).to_dict("records")})
    predictions=pd.read_parquet(output/"predictions.parquet")
    focus=[r["symbol"] for r in facts["stocks"] if r["symbol"]!="QQQ"]
    f=predictions[(predictions.algorithm=="lgbm")&(predictions.granularity==5)&(predictions.scope=="pooled")&
                  (predictions.expectation=="d5_r8")&(predictions.date=="2026-09-18")&predictions.symbol.isin(focus)]
    pivot=f.pivot(index="symbol",columns="round",values="p").join(f[f["round"]=="R3"].set_index("symbol")[["hit","mfe","mae","entry_at","label_end_at"]])
    missed=pivot.reset_index().to_dict("records")
    diagnostics={"loss_decomposition":losses,"september_18_case_study":missed,
                 "case_study_is_posthoc":True,"case_study_changes_no_model_or_threshold":True}
    write_json(output/"diagnostics.json",clean(diagnostics))
    notes=["# 光产业链：事实核查、Pugh 矩阵与三轮实测", "",
           "**结论：覆盖和展示问题已修复；三轮没有建立稳定的预测优势。共同成熟窗口内，光产业链整体账户的全部 84 个轮次/算法/粒度/训练范围/期望组合都亏损。不能把 9 月短期盈利或局部准确率提升解释为模型已通过验收。**", "",
           "## 1. 事实与偏差", "",
           "行情截至 2026-09-25。近 20 个完整交易日为 08-28—09-25，以 08-27 收盘为基准；另列近 5 日。从原始 NONE 5m 常规盘汇总，各股完整性和公司行动已核对。", "",
           "| 股票 | 环节 | 5日收益 | 20日收益 | 窗口最高触及 | 峰后回吐 | 事后形态 |", "|---|---|---:|---:|---:|---:|---|"]
    for r in facts["stocks"]:
        notes.append(f"| {r['symbol']} | {r['stage']} | {percent(r.get('return_5'))} | {percent(r.get('return_20'))} | {percent(r.get('peak_return'))} | {percent(r.get('giveback_from_peak'))} | {r.get('retrospective_shape','不可评价')} |")
    notes += ["", "光产业链不是同步上涨的一组同质股票：ALAB、AXTI有阶段性持续上行，CRDO近5日反弹但20日仍下跌，LITE/COHR出现冲高回吐。5日/20日收盘收益和延迟入场后的高点触及标签也不是同一个问题。以上形态只用于事后事实核查，未输入模型。", "",
              "核实的三项工程偏差：", "", "- 原光通信子群只有 LITE；新观察集中的 COHR、CIEN、AAOI、FN、AXTI、CRDO、ANET 不在原可评价股票池。", "- 原评价决策截止 08-25，不能直接回答 9 月的走势。", "- 原摘要采用至少两只证券的例子筛选，遗漏单股光通信群。原周选择器实际在 08-10、08-12、08-19、08-21 交易过 LITE，不能说旧系统完全没有识别。", "",
              "旧58群全部保留，新增12股产业链整体、光学直接环节及5个子环节，共65群。单股群可以展示全池模型分层评价；本轮产业链专属模型的拟合范围统一为12股，子群是评价绑定，不伪称单股专属训练。", "",
              "## 2. Pugh 设计矩阵", "", "R0 为 datum。以下 +/0/− 是工程能力和代价的相对判断，不是预测收益打分。", "",
              "| 准则 | R1 覆盖/时效 | R2 路径/同群 | R3 训练强度 |", "|---|---|---|---|", "| 光产业链覆盖 | + | + | + |", "| 成熟标签更新时效 | + | + | + |", "| 走势顺序与群内相对位置 | 0 | + | + |", "| 训练轮数有内部时间验证依据 | 0 | 0 | + |", "| 计算及审计成本 | − | − | − |", "| 实际概率和资金质量 | 待实测 | 待实测 | 待实测 |", "",
              "方法依据：[MIT 16.842 Concept Selection / Pugh Matrix](https://ocw.mit.edu/courses/16-842-fundamentals-of-systems-engineering-fall-2015/dac225aff5a84125fd1bf9364d748221_MIT16_842F15_Ses_5_Design.pdf)。", "",
              "三轮：R1补覆盖及每月按各目标成熟时间划拟合/校准，保留100树/5epoch预算；R2增加因果路径、同截止QQQ和排除自身的同群信息（静态124→221维）；R3上限500树/30epoch，由内部时间验证选预算后重训。每轮72模型，共216新模型，另36旧权重重放任务，全部成功。", "",
              "R0对新增股票是离线迁移重放，并非旧产品当时覆盖。它在9月沿用旧8月检查点；R1/R2/R3按9月前可知成熟标签更新。R1同时更改覆盖与时效，是工程组合方案，无法把它的差异严格归因到某一单因素。", "",
              "## 3. 三轮共同成熟窗口结果", "", "预先固定的主路线：**LightGBM、5m、全池训练，绑定光产业链12股**。07-01—08-25共39个决策日、每目标468个成熟股票日期，持仓估值到09-25。每个目标初始10万美元，三个账户独立。", "",
              "| 期望 | 轮次 | 分类准确率 | 候选命中率 | 机会召回 | Brier↓ | 扣费收益 | 期末金额 | 最大回撤 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for target in NAMES:
        for round_id in ROUNDS:
            r=next(r for r in selected if r["window"]=="common" and r["expectation"]==target and r["round"]==round_id)
            notes.append(f"| {NAMES[target]} | {round_id} {ROUNDS[round_id]} | {percent(r['accuracy'])} | {percent(r['precision'])} | {percent(r['recall'])} | {number(r['brier'])} | {percent(r['net_return'])} | ${r['end_balance']:,.2f} | {percent(r['max_drawdown'])} |")
    notes += ["", "实测 Pugh（主路线，相对R0；+改善/−退化，不表示显著）：", "", "| 轮次/目标 | Brier误差 | 机会召回 | 扣费收益 | 回撤 |", "|---|---|---|---|---|"]
    for r in selected:
        if r["window"]!="common" or r["round"]=="R0":continue
        base=next(b for b in selected if b["window"]=="common" and b["round"]=="R0" and b["expectation"]==r["expectation"])
        signs=[]
        for key,direction in [("brier",-1),("recall",1),("net_return",1),("max_drawdown",1)]:
            delta=(r[key]-base[key])*direction
            signs.append("0" if abs(delta)<.0001 else "+" if delta>0 else "−")
        notes.append(f"| {r['round']} / {NAMES[r['expectation']]} | "+" | ".join(signs)+" |")
    notes += ["", "全路线主产业链成对概率比较，3/5日期望共有48项可检验，**Holm校正后0项通过**；21日期望的独立日期块不足，未伪造显著性。均为开发期近似检验，历史已接触，未建立独立验证。CSV列出了3960条群/期望/算法/范围/轮次/窗口结果；17个其他群账户因所选路径不完整不可评价，48条无可用观测，未混作零收益。", "",
              "## 4. 近期窗口单列：09-01—09-25", "", "18个决策日；3日期望180个成熟样本，5日期望156个，21日期望0个。资金账户可根据已经发生的目标触及退出，尚未到期持仓计入未实现盈亏。", "",
              "| 期望 | 轮次 | 成熟样本准确率 | 候选命中率 | 扣费净值收益 | 已平/未平 | 未实现盈亏 |", "|---|---|---:|---:|---:|---:|---:|"]
    for target in NAMES:
        for round_id in ROUNDS:
            r=next(r for r in selected if r["window"]=="recent" and r["expectation"]==target and r["round"]==round_id)
            notes.append(f"| {NAMES[target]} | {round_id} | {percent(r['accuracy'])} | {percent(r['precision'])} | {percent(r['net_return'])} | {r['closed_trade_n']}/{r['open_positions']} | ${r['unrealized_pnl']:,.2f} |")
    notes += ["", "R2近期3日/5日期望收益更高，但第三轮没有延续该提升。9月区间短且21日未成熟，不能据此选择上线赢家。", "",
              "## 5. 具体漏报：09-18的5日+8%机会", "", "固定原来的0.5门槛，以下是09-18 11:30 ET概率，结果从11:35入场后计算5×390分钟。这个事后切片用来定位漏报，不改变模型或门槛，也不等同09-18收盘至09-25收盘收益。", "",
              "| 股票 | R0迁移对照 | R1 | R2 | R3 | 实际触及+8% | 期间最高涨幅 |", "|---|---:|---:|---:|---:|---|---:|"]
    for r in missed:
        notes.append(f"| {r['symbol']} | {percent(r['R0'])} | {percent(r['R1'])} | {percent(r['R2'])} | {percent(r['R3'])} | {'是' if r['hit'] else '否'} | {percent(r['mfe'])} |")
    notes += ["", "这一切片12股中7只实际达标，R3全部低于0.5：ALAB 28.36%、CRDO 32.40%、AXTI 49.91%。R1曾把CRDO和AXTI推过门槛，后两轮又降下来。因此，补齐股票池修复了可见性，但目前特征、校准与固定门槛仍没稳定识别这次共同反弹；不能把漏报全部归因到样本外股票缺失。", "",
              "## 6. 为什么较高命中率仍亏钱", "", "R3主路线共同窗口的实际成交，不是全部股票日期候选：", "", "| 期望 | 达标/未达标交易 | 平均达标净收益 | 平均未达标净收益 | 达标合计盈利 | 未达标合计损失 |", "|---|---:|---:|---:|---:|---:|"]
    for r in losses:
        notes.append(f"| {NAMES[r['expectation']]} | {r['wins']}/{r['losses']} | {percent(r['mean_win'])} | {percent(r['mean_loss'])} | ${r['gain_dollars']:,.2f} | ${r['loss_dollars']:,.2f} |")
    notes += ["", "例如21日期望：27次达标盈利约$25,139，被17次未达标损失约$33,576抵消。单纯预测是否触及，不限制未触及时损失、途中回撤、持仓时间和同主题相关暴露；这些是目标与交易选择之间的缺口。每日候选命中率也不能当作实际成交胜率。", "",
              "## 7. 更多训练的证据", ""]
    for algo,r in summary["training_budgets"].items():
        notes.append(f"- {algo.upper()}：原预算{r['original_budget']}；内层选中范围{r['min']}—{r['max']}，中位数{r['median']:g}；{r['model_n']}个模型中{r['above_original']}个高于原预算、{r['below_original']}个低于原预算。")
    notes += ["", "多数模型更早停止，增加训练不是普遍瓶颈；部分TCN需要更长训练，但没有带来跨路线稳定优势。这不能排除更长历史、更优架构或不同市场状态建模的潜力，只能否定本次‘单纯增加训练预算即可解决’的假设。", "",
              "## 8. 下一步优先级与本次决策", "", "1. 保留覆盖修复、单股群可见性、分期评价、pending持仓估值和全量矩阵，三个原期望继续保留。", "2. 优先设计触及概率与下行/未达标损失/达成时间的联合输出，再做资金约束下的选择。收益期望继续作为独立输出，风险信息用于决定是否值得承担该机会。", "3. 保留因果路径与同群信息分支供后续研发，但不根据9月最高收益直接上线R2；要冻结下一协议，在未接触日期收集真正的前向预测、接收时间和校准证据。", "4. 暂停仅增加树数或epoch的方向。本次严格结束于三轮，不根据这些外层结果追加第四轮。", "",
              "## 9. 验收与复现", "", "5项新不变量测试、原工程8项测试；源行情哈希、训练/校准maturity purge、内层验证边界、检查点预测重放、资金与未实现盈亏守恒。见audit.json和README。", "",
              "资金规则：$100,000，每侧6bp，每日最多3新股，最多10持仓，每次最多10%，不加仓重复股票；达到目标或到期卖出，无止损。5m OHLC成交代理，无订单簿、现金股息、结算延迟，回撤仅按5m收盘估值。", "",
              "所有名单为当前快照回溯，有群体选择/幸存者偏差。26只ETF不可评价。原105可评价股票中除本次13个更新标的（12股+QQQ）外，其余行情仍停留在旧数据时点；近期资金结果只提供更新完整的光产业链群。历史available_at为回补假设，不是实盘收到行情的证明。", "",
              "[交互报告](index.html) · [全量结果](results.csv) · [成对对照](comparisons.json) · [逐股票概率](latest.csv) · [漏报与盈亏分解](diagnostics.json) · [审计](audit.json)"]
    (output/"REPORT.md").write_text("\n".join(notes)+"\n")
    slim=[{k:v for k,v in r.items() if k not in ["reliability","trial_statuses"]} for r in results]
    payload={"summary":summary,"facts":facts,"results":slim,"latest":read("latest.json"),
             "curves":read("learning_curves.json"),"groups":read("data/groups.json"),"audit":read("audit.json") if (output/"audit.json").exists() else None,
             "diagnostics":diagnostics}
    encoded=json.dumps(clean(payload),ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    html=Path(__file__).with_name("report.html").read_text().replace("__PAYLOAD__",encoded)
    (output/"index.html").write_text(html)
    print(f"wrote {output/'index.html'} ({len(html):,} chars)")

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    render(p.parse_args().output.resolve())
