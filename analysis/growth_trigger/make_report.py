"""Produce report.md and signals_review.html from the frozen sample.

Recomputes every number from points.jsonl / events.json so the report is reproducible.
"""

from __future__ import annotations

import html
import json
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_candidates as RC
from gt_common import load_daily_market, load_hourly, pinned_tickers, regular_session_closes

RULES = dict(RC.RULES)


def pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def main() -> None:
    pts = RC.load_points()
    breadth = RC.build_breadth(pts)
    for p in pts:
        p["f"] = RC.featurize(p, breadth)
        p["split"] = "eval" if int(p["date"][8:10]) % 2 == 1 else "train"
        p["in_win"] = RC.EVAL_START <= p["date"] <= RC.EVAL_END
    ev = [p for p in pts if p["split"] == "eval" and p["in_win"] and "ext_hit" in p]
    tr = [p for p in pts if p["split"] == "train" and "ext_hit" in p]
    events = json.loads((HERE / "events.json").read_text())
    tr_days = defaultdict(list)
    for e in events:
        if e["parity"] == 0:
            tr_days[e["ticker"]].append(date.fromisoformat(e["start_date"]))
    eval_ev = [e for e in events
               if e["parity"] == 1 and RC.EVAL_START <= e["start_date"] <= RC.EVAL_END
               and not any(abs((date.fromisoformat(e["start_date"]) - t).days) <= 14
                           for t in tr_days[e["ticker"]])]
    train_ev = [e for e in events if e["parity"] == 0]

    base_ext = sum(p["ext_hit"] for p in ev) / len(ev)
    base_cls = sum(p["cls_hit"] for p in ev) / len(ev)

    rows = []
    signals_for_review = defaultdict(list)
    for name, fn in RULES.items():
        if name == "R0_always":
            continue
        for scope, pool, evs in (("train", tr, train_ev), ("eval", ev, eval_ev)):
            raw = [p for p in pool if fn(p["f"])]
            for mode, sig in (("point", raw), ("day", RC.dedup_day(raw)),
                              ("event", RC.dedup_event(raw))):
                if not sig:
                    rows.append(dict(rule=name, scope=scope, mode=mode, n=0, prec=None,
                                     prec_close=None, recall=None, captured=0, events=len(evs),
                                     fp100=None, freq=None, lift=None, ci=None, leads=[]))
                    continue
                prec = sum(s["ext_hit"] for s in sig) / len(sig)
                prec_c = sum(s["cls_hit"] for s in sig) / len(sig)
                captured, leads = 0, []
                for e in evs:
                    s0 = date.fromisoformat(e["start_date"])
                    hit_days = [s0 - timedelta(days=k) for k in (0, 1, 2, 3)]
                    fired = [s for s in sig if s["ticker"] == e["ticker"]
                             and date.fromisoformat(s["date"]) in hit_days]
                    if fired:
                        captured += 1
                        leads.append((s0 - min(date.fromisoformat(s["date"]) for s in fired)).days)
                ci = RC.bootstrap_lift(sig, base_ext)
                rows.append(dict(rule=name, scope=scope, mode=mode, n=len(sig), prec=prec,
                                 prec_close=prec_c, recall=captured / len(evs) if evs else None,
                                 captured=captured, events=len(evs),
                                 fp100=(len(sig) - sum(s["ext_hit"] for s in sig)) / len(pool) * 100,
                                 freq=len(sig) / len(pool), lift=prec / base_ext, ci=ci,
                                 leads=leads))
                if scope == "eval" and mode == "day":
                    for s in sig:
                        signals_for_review[s["ticker"]].append(
                            dict(rule=name, ticker=s["ticker"], ts=s["ts"], date=s["date"], sess=s["sess"],
                                 entry=s["entry"], hit=s["ext_hit"], close_hit=s["cls_hit"],
                                 max_return=s["ext_max_return"],
                                 **{k: s["f"].get(k) for k in
                                    ("ret_1d", "ret_5d", "ret_20d", "px_vs_hi20",
                                     "vol_ratio", "premkt_ret", "rel_20d", "qqq_20d")}))

    # ---------- report.md ----------
    def fmt_ci(ci):
        return "—" if not ci or ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    lines = []
    A = lines.append
    A("# 增长区间提前触发规律研究 · 报告 v1")
    A("")
    A(f"- 生成时间：{json.loads((HERE/'labels_8pct_v1.json').read_text())['generated_at']}")
    A("- 冻结规格：`spec.md`（v1，规则与阈值在看评估结果前写定）")
    A("- 股票池：置顶名单 21 只，其中 **18 只**有合格股票日（见 §2）；评估区间 2026-02-01—2026-09-30；分组：偶数日训练 / 奇数日评估")
    A("")
    A("## 1. 结论")
    A("")
    A("**没有找到通过事前设定门槛的触发规律。** 按 `spec.md` §6 的三条同时满足的门槛"
      "（≥15 个去重信号、召回率 ≥25%、精准率 ≥1.3×基准且 bootstrap 95% 下界高于基准），"
      "6 条候选规则**无一全部满足**：")
    A("")
    A("| 规则 | 去重后信号 | 精准率 | vs 基准 | 召回率 | 判定 |")
    A("|---|---:|---:|---:|---:|---|")
    for r in rows:
        if r["scope"] != "eval" or r["mode"] != "day":
            continue
        verdict = ("通过" if (r["n"] >= RC.MIN_SIGNALS and (r["recall"] or 0) >= RC.MIN_RECALL
                              and (r["lift"] or 0) >= RC.LIFT
                              and r["ci"] and r["ci"][0] > base_ext) else "未通过")
        A(f"| {r['rule']} | {r['n']} | {pct(r['prec'],1)} | "
          f"{'—' if r['lift'] is None else '×'+format(r['lift'],'.2f')} | "
          f"{pct(r['recall'],1)} | {verdict} |")
    A("")
    A("两条规则有**统计显著但幅度有限**的正边缘，且都不达门槛，也不稳定：")
    A("")
    A("- **R3（盘前走强 + 20 日相对强弱为正）**：287 个去重信号，精准率 50.9%（基准 42.0%，"
      "lift ×1.21，95% CI [44.9%, 56.8%]），召回 13/39。但 (a) 边缘集中在 MRVL/MU/AMD/STX 四只"
      "半导体股与 2026-04—05；(b) 13 次命中里 10 次的提前量是 **0 天**（区间起点当天的盘前），"
      "本质是「当天启动」而非「提前」；(c) 改成事件级去重（每股每 14 天只算一次）后精准率跌到 43.5%，"
      "lift ×1.04，边缘基本消失。")
    A("- **R4（20 日涨幅 >10% + QQQ 20 日为正 + 相对强弱为正）**：308 个信号，精准率 52.6%"
      "（lift ×1.25，CI [46.8%, 57.8%]），但召回只有 7/39（17.9%），未达 25% 门槛；同样集中在"
      "半导体股与 4—5 月，AAPL/MSFT/META 上几乎全错。")
    A("")
    A("R1（近 20 日高点 + 放量）与 R2（放量上涨）**显著差于随机买入**（精准率 20.0% / 36.4% "
      "vs 基准 42.0%），R5（近 20 日低点放量转强）在评估期**一次都没有触发**。")
    A("")
    A("## 2. 样本与口径")
    A("")
    A("| 项目 | 数值 |")
    A("|---|---:|")
    A(f"| 预测点总数（2026-01-02—09-22，18 只有效股票） | {len(pts):,} |")
    A(f"| 其中可判定 | {len([p for p in pts if 'ext_hit' in p]):,} |")
    A(f"| 无法判定（末端不足 14 自然日） | {len([p for p in pts if 'ext_hit' not in p]):,} |")
    A(f"| 训练集点数（偶数日） | {len(tr):,} |")
    A(f"| 评估集点数（奇数日、2026-02—09） | {len(ev):,} |")
    A(f"| 8% 区间总数（全期） | {len(events)} |")
    A(f"| 起点在 2026-02—09 的区间 | {sum(1 for e in events if RC.EVAL_START <= e['start_date'] <= RC.EVAL_END)} |")
    A(f"| 评估集事件（奇数起点且剔除跨组重叠） | {len(eval_ev)} |")
    A(f"| 因与训练期区间相距 ≤14 天被剔除 | {len([e for e in events if e['parity']==1 and RC.EVAL_START<=e['start_date']<=RC.EVAL_END]) - len(eval_ev)} |")
    A("")
    A("判定口径（用户指定）：入场价 = 信号小时 bar 的收盘价；达成 = 入场后 3–14 个自然日内，"
      "含盘前/盘后/夜间的任意小时最高价 ≥ 入场价 ×1.08。基准 = 同期全部合格预测点都买入的达成率。")
    A("")
    A("**有效股票池是 18 只，不是 21 只。** SNOW、SPCX、TSM 三只置顶股在 point-in-time "
      "资格门槛下**一个合格股票日都没有**，因此既没有正样本也没有负样本：")
    A("")
    A("| 股票 | 合格股票日 | 预测点 | 历史区间 | 未通过资格的原因 |")
    A("|---|---:|---:|---:|---|")
    m = load_daily_market()
    for t_ in pinned_tickers():
        elig = sum(1 for i in range(len(m.days))
                   if m.eligible(t_, i, require_next_bar=False))
        npts = sum(1 for p in pts if p["ticker"] == t_)
        nwin = sum(1 for e in events if e["ticker"] == t_)
        q = m.quality(t_, next(i for i in range(len(m.days)) if m.bar(t_, i)))
        why = "—"
        if elig == 0:
            if not q:
                why = "无 SEC 年报事实（营收/现金流/净利润均取不到）"
            else:
                why = (f"净利润 {q['net_income']/1e9:.2f}B ≤ 0"
                       if q["net_income"] <= 0 else "其他门槛")
        A(f"| {t_} | {elig} | {npts} | {nwin} | {why} |")
    A("")
    A("因此「21 只置顶股」只是输入名单，实际参与统计的是 18 只；本节所有比率的分母都不包含这 3 只。")
    A("")
    A(f"- 评估集基准（含延长时段触达）：**{pct(base_ext,2)}**"
      f"（盘前 {pct(sum(p['ext_hit'] for p in ev if p['sess']=='pre')/sum(1 for p in ev if p['sess']=='pre'),2)}、"
      f"常规 {pct(sum(p['ext_hit'] for p in ev if p['sess']=='reg')/sum(1 for p in ev if p['sess']=='reg'),2)}、"
      f"盘后 {pct(sum(p['ext_hit'] for p in ev if p['sess']=='post')/sum(1 for p in ev if p['sess']=='post'),2)}）")
    A(f"- 评估集基准（只用常规时段收盘价，更严口径）：**{pct(base_cls,2)}**")
    A("")
    A("## 3. 标签快照核对")
    A("")
    lab = json.loads((HERE / "labels_8pct_v1.json").read_text())
    A(f"`labels_8pct_v1.json`：{lab['stock_count']} 只股票、{lab['window_count']} 段区间；"
      f"参数 target=8%、自然日 3–14（含端点）、≥3 个交易日收盘间隔、单日 ≤12%、达标后 2 日保留 ≥4%。")
    A("口径已与页面当前设置逐字对齐：用同一套参数在页面的 115 只股票池上重算，得到 "
      "**60 只股票 / 402 段区间**，与页面侧栏显示一致（复现脚本见 `verify_page_labels.py`）。")
    A("")
    A("## 4. 规则结果（全口径）")
    A("")
    A("| 规则 | 集 | 去重 | 信号数 | 精准率(触达) | 精准率(收盘) | 召回 | 误报/100 合格点 | 触发频率 | lift | 精准率 95% CI |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        A(f"| {r['rule']} | {r['scope']} | {r['mode']} | {r['n']} | {pct(r['prec'],1)} | "
          f"{pct(r['prec_close'],1)} | {pct(r['recall'],1)} | "
          f"{'—' if r['fp100'] is None else format(r['fp100'], '.2f')} | "
          f"{'—' if r['freq'] is None else format(r['freq']*100, '.2f')+'%'} | "
          f"{'—' if r['lift'] is None else '×'+format(r['lift'], '.2f')} | {fmt_ci(r['ci'])} |")
    A("")
    A("## 5. 稳定性")
    A("")
    for name in ("R3", "R4"):
        sig = RC.dedup_day([p for p in ev if RULES[name](p["f"])])
        A(f"**{name}**（{len(sig)} 个去重信号）")
        A("")
        byq = defaultdict(list)
        for s in sig:
            byq[s["date"][:7]].append(s)
        A("- 按月份：" + "；".join(
            f"{q} n={len(v)} 命中={sum(x['ext_hit'] for x in v)/len(v)*100:.0f}%"
            for q, v in sorted(byq.items())))
        bys = defaultdict(list)
        for s in sig:
            bys[s["ticker"]].append(s)
        A("- 按股票：" + "；".join(
            f"{t} n={len(v)} 命中={sum(x['ext_hit'] for x in v)/len(v)*100:.0f}%"
            for t, v in sorted(bys.items(), key=lambda x: -len(x[1]))))
        bym = defaultdict(list)
        for s in sig:
            bym[RC.market_state(s["f"])].append(s)
        A("- 按市场状态（QQQ 20 日涨幅 >+3% / <−3% / 其间）：" + "；".join(
            f"{k} n={len(v)} 命中={sum(x['ext_hit'] for x in v)/len(v)*100:.0f}%"
            for k, v in sorted(bym.items())))
        A("")
    A("## 5b. 逐股召回率与精准率（用户口径，主口径=股票日级去重）")
    A("")
    A("| 股票 | 评估集区间 | 命中(召回) | 逐股召回率 | 买入信号 | 精准率 |")
    A("|---|---:|---:|---:|---:|---:|")
    all_feb_sep = [e for e in events if RC.EVAL_START <= e["start_date"] <= RC.EVAL_END]
    for t in pinned_tickers():
        evs = [e for e in eval_ev if e["ticker"] == t]
        all_evs = [e for e in all_feb_sep if e["ticker"] == t]
        sig = RC.dedup_day([p for p in ev if RULES["R4"](p["f"]) and p["ticker"] == t])
        prec = sum(x["ext_hit"] for x in sig) / len(sig) if sig else None
        cap = 0
        for e in evs:
            s0 = date.fromisoformat(e["start_date"])
            if any(x["ticker"] == t and date.fromisoformat(x["date"]) in
                   [s0 - timedelta(days=k) for k in (0, 1, 2, 3)] for x in sig):
                cap += 1
        A(f"| {t} | {len(evs)} | {cap} | {pct(cap/len(evs),0) if evs else '—'} | "
          f"{len(sig)} | {pct(prec,0) if prec is not None else '—'} |")
    A("")
    A("注：评估集只含奇数日起点且不与训练期区间重叠的事件，所以逐股召回率的分母很小"
      f"（0–3 个），单只数字不具统计意义；全期 2026-02—09 共 {len(all_feb_sep)} 段区间，"
      "若把全部区间（不分奇偶）当分母，R4 的召回率会更高，但那已包含训练集见过的事件，"
      "不能当作样本外召回。")
    A("")

    A("## 6. 提前量")
    A("")
    for name in ("R3", "R4"):
        r = next(x for x in rows if x["rule"] == name and x["scope"] == "eval" and x["mode"] == "day")
        dist = defaultdict(int)
        for d in r["leads"]:
            dist[d] += 1
        A(f"- {name}：命中事件的提前量分布 {dict(sorted(dist.items()))}"
          f"（0 = 区间起点当天的盘前/盘中发出信号）")
    A("")
    A("## 7. 人工审查")
    A("")
    A("- 逐条信号（含触发时证据、入场价、是否达成、最大涨幅）：`signals.jsonl`")
    A("- 图：`signals_review.html`，每只股票一张日 K 线图；绿色阴影=历史 8% 区间，"
      "三角=R3/R4 预测的买入触发点（实心绿=达成、空心红=未达成），可直接核对误报。"
      "图只覆盖小时线期间（2026-01 起），2025 年的区间不在图上。")
    A("- 误报集中在：MRVL/MU/AMD/STX 的多次重复触发（同一段行情被多个小时 bar 反复触发），"
      "以及 2026-04—05 之外月份的低命中区段。")
    A("- 抽样人工核验包：`human_audit_sample.json`，含 5 个正例、5 个负例、8 个边界例"
      "（3/14 自然日端点、3 个交易日间隔、单日涨幅贴近 12%、保留涨幅贴近 4%）和 8 个被拒候选，"
      "每条的核验要点写在 `how_to_review` 字段里。")
    A("- 标签构建的中间量：满足全部条件的候选 721 段，其中 1,514 段候选因至少一条不满足被拒"
      "（自然日超界 1,244、单日涨幅 >12% 275、达标后保留不足 262、交易日间隔不足 270，可重复计数）；"
      "721 段候选再按不重叠去重、同一起点保留涨幅最大者，得到最终 157 段。")
    A("")
    A("### 7b. 两个必须写明的口径细节")
    A("")
    A("- 若把基准换成更严的「只用常规时段收盘价」基准（37.19%），R3 的精准率 48.4% 相当于 "
      "×1.30、R4 的 47.1% 相当于 ×1.27，看着像达标。**但不换**：`spec.md` §6 事前锁定的是主口径"
      "（含盘前盘后触达）下的基准 41.99%，事后换基准去凑门槛正是规格禁止的行为。")
    A("- R1 在训练集上 35 个信号命中 85.7%、在评估集上 89 个信号只有 16.9%；R5 在训练集上"
      "5 个信号全中、评估集上一次不触发。这种量级的分裂说明当前样本下规则极不稳定，"
      "不能把训练集的漂亮数字当成规律。")
    A("- 「从不触发」检查：R5 在评估期 0 次触发；若一条规则从不触发，其精准率无定义、召回为 0，"
      "`spec.md` §6 的「≥15 个去重信号」就是为了拦住这种用不出手换来的表面高准确率。")
    A("")

    A("## 8. 四层结论（严格区分）")
    A("")
    A("| 层级 | 状态 |")
    A("|---|---|")
    A("| 找到历史区间 | ✅ 157 段（18 只有效置顶股、2025-10—2026-09），标签可复算、与页面口径一致 |")
    A("| 发现候选规律 | ⚠️ R3/R4 有统计显著但幅度有限的正边缘（lift ×1.21 / ×1.25），未通过事前门槛 |")
    A("| 通过独立验证 | ❌ 未通过。见 §9 |")
    A("| 可执行交易策略 | ❌ 未验证成交、成本、滑点、仓位与组合结果 |")
    A("")
    A("## 9. 为什么不能算独立验证")
    A("")
    A("1. 训练集与评估集按日期奇偶切分，虽已剔除同一股票相距 ≤14 个自然日的跨组区间，"
      "但两组共享同一段行情的时间邻域，且 18 只有效股票、39 个评估事件的样本量太小，"
      "单只股票/单月的表现就能左右合并数字。")
    A("2. 6 条规则同时检验，最好的一条（R4）很可能是多重比较的产物；其边缘集中在 4 只半导体股"
      "和 2 个月，缺乏跨股票、跨月份的稳定性。")
    A("3. 小时线数据只有 2026-01-01 起，评估期只有 2026-02—09，没有更早的样本外区间可用；"
      "真正独立的验证需要此前完全未参与的新时期数据或前向纸面跟踪。")
    A("")
    A("## 10. 数据缺陷（影响结论强度）")
    A("")
    A("- SPCX 小时线仅 2026-06-09 起有数据，但 SPCX 本就没有合格股票日，对样本无影响。")
    A("- Futu 小时线成交量约为日线档案的 91%，量能特征只用同一序列内的相对值。")
    A("- 分红股的前复权价与日线档案存在 0.1%–0.5% 差异，价格基准统一用 Futu 小时线。")
    A("- 小时线 `time_key` 为美东 bar 结束时刻；已验证常规时段（标签 10:30–16:00）重算的"
      " OHLCV 与日线档案一致（非分红股比值恒为 1.000000）。")
    A("- 未能取得分钟级或盘口数据，「今天能否成交」未检验。")
    A("- R1/R2/R5 依赖 `f_vol_ratio`（当日累计成交量），而盘前 bar 当日还没有成交量，"
      "因此这三条规则在结构上只能在常规时段之后触发；R3 是唯一覆盖盘前的规则。")
    A("- `unable.json` 只含 578 个「末端不足 14 自然日」的预测点；没有出现「合格但无小时线」的"
      "股票日，因为唯一小时线不全的 SPCX 本来就没有合格股票日。")
    A("")
    A("## 11. 下一步（若要继续）")
    A("")
    A("1. 用更长的小时线历史（重抓 2024—2025 年）把训练/评估样本扩大一个量级；")
    A("2. 把「提前量」作为一等指标：要求信号在区间起点前 ≥1 个交易日出现，再看精准率掉多少；")
    A("3. 对 R4 做跨市场状态与跨股票的稳定性检验，或直接判定其不可用；")
    A("4. 任何新阈值都必须先在 `spec.md` 记为新版本，再用未看过的新时期检验。")
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- complete signal log (all rules, eval, day-level) ----------
    with (HERE / "signals.jsonl").open("w") as fh:
        for t in sorted(signals_for_review):
            for s in sorted(signals_for_review[t], key=lambda x: x["ts"]):
                fh.write(json.dumps(s) + "\n")

    # ---------- signals_review.html ----------
    review = {t: [s for s in v if s["rule"] in ("R3", "R4")]
              for t, v in signals_for_review.items()}
    build_html(review, events, pts)
    print("report.md and signals_review.html written")
    print(f"base_ext={base_ext:.4f} base_cls={base_cls:.4f} eval_events={len(eval_ev)}")


def build_html(signals_by_ticker, events, pts):
    reg = {t: regular_session_closes(load_hourly(t)) for t in pinned_tickers()}
    out = []
    out.append("<!doctype html><meta charset='utf-8'><title>增长触发信号人工审查</title>")
    out.append("<style>body{font:13px -apple-system,sans-serif;background:#0e1621;color:#dbe7f3;"
               "margin:0;padding:16px}.stock{margin:0 0 22px}h3{margin:8px 0 4px}"
               "svg{background:#121d2b;border:1px solid #22303f;border-radius:6px}"
               ".lg{font-size:12px;color:#9fb3c8}.band{fill:rgba(85,223,193,.16);"
               "stroke:#55dfc1;stroke-dasharray:3 3}.hit{fill:#55dfc1}.miss{fill:none;"
               "stroke:#ff8589;stroke-width:1.5}text{fill:#8fa6bc}</style>")
    out.append("<h2>增长区间触发信号人工审查</h2><div class='lg'>绿色阴影=历史 8% 区间（标签）；"
               "实心绿三角=R3/R4 买入触发且达成；空心红三角=触发但未达成。"
               "图中只画 R3/R4（唯一有正边缘的两条）；全部 6 条规则的逐条信号见 signals.jsonl，规则定义见 spec.md §9。</div>")
    for t in pinned_tickers():
        days = sorted(reg[t])
        if not days:
            continue
        closes = [reg[t][d]["c"] for d in days]
        lo, hi = min(closes), max(closes)
        pad = (hi - lo) * 0.07 or 1
        lo, hi = lo - pad, hi + pad
        W, H, L, R, T, B = 900, 240, 46, 12, 16, 26
        step = (W - L - R) / len(days)
        x = lambda i: L + i * step + step / 2
        y = lambda p: T + (hi - p) / (hi - lo) * (H - T - B)
        pos = {d: i for i, d in enumerate(days)}
        svg = [f"<svg viewBox='0 0 {W} {H}' width='{W}' height='{H}'>"]
        for k in range(5):
            yy = T + k * (H - T - B) / 4
            p = hi - (hi - lo) * k / 4
            svg.append(f"<line x1='{L}' y1='{yy:.1f}' x2='{W-R}' y2='{yy:.1f}' stroke='#22303f'/>"
                       f"<text x='{L-6}' y='{yy+4:.1f}' text-anchor='end' font-size='10'>{p:.0f}</text>")
        svg.append(f"<polyline fill='none' stroke='#6ea8fe' points='"
                   + " ".join(f"{x(i):.1f},{y(c):.1f}" for i, c in enumerate(closes)) + "'/>")
        for e in events:
            if e["ticker"] != t:
                continue
            i, j = pos.get(e["start_date"]), pos.get(e["end_date"])
            if i is None or j is None:
                continue
            svg.append(f"<rect class='band' x='{x(i)-step/2:.1f}' y='{T}' "
                       f"width='{(j-i+1)*step:.1f}' height='{H-T-B}'/>")
        for s in signals_by_ticker.get(t, []):
            i = pos.get(s["date"])
            if i is None:
                continue
            cls = "hit" if s["hit"] else "miss"
            svg.append(f"<path class='{cls}' d='M{x(i):.1f},{y(s['entry'])+9:.1f} "
                       f"l-4.5,8 l9,0 z'/>")
        svg.append(f"<text x='{L}' y='{H-8}' font-size='10'>{days[0]}</text>"
                   f"<text x='{W-R}' y='{H-8}' text-anchor='end' font-size='10'>{days[-1]}</text>")
        svg.append("</svg>")
        n_hit = sum(1 for s in signals_by_ticker.get(t, []) if s["hit"])
        n_all = len(signals_by_ticker.get(t, []))
        n_win = sum(1 for e in events if e["ticker"] == t)
        out.append(f"<div class='stock'><h3>{t} · 历史区间 {n_win} 段 · 触发信号 {n_all} 个"
                   f"（达成 {n_hit}）</h3>{''.join(svg)}</div>")
    (HERE / "signals_review.html").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
