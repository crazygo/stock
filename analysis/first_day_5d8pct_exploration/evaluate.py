"""Evaluate a frozen five-candidate shortlist and document all 58 group attempts."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

from search_rules import mask_rule, describe, rule_text, FEATURES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(os.environ.get("TOUCH_RESEARCH_RUN", str(Path(__file__).resolve().parent)))
SOURCE = ROOT / "research/group_expectation_matrix/outputs/20260925_v5"


def json_safe(x):
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def evaluate_rule(panel, item, split, ci=False):
    group = panel[panel.checkpoint.eq(item["checkpoint"]) & panel.group_ids.map(lambda ids: item["group_id"] in ids)].copy()
    if split != "all":
        group = group[group.split.eq(split)]
    total_condition = mask_rule(group, item["rule"])
    total = group.loc[total_condition]
    pending = group[~group.already_hit].copy()
    condition = mask_rule(pending, item["rule"])
    selected = pending.loc[condition]
    result = describe(pending, condition)
    result.update(total_n=len(total), total_hits=int(total.target.sum()),
                  total_rate=float(total.target.mean()) if len(total) else None,
                  already_hit_n=int(total.already_hit.sum()),
                  observed_price_remaining_upside_median=float(selected.remaining_upside.median()) if len(selected) else None,
                  future_mae_median=float(selected.future_mae_from_observation.median()) if len(selected) else None,
                  future_mae_p10=float(selected.future_mae_from_observation.quantile(.1)) if len(selected) else None,
                  terminal_median=float(selected.terminal_from_observation.median()) if len(selected) else None,
                  hit_minutes_median=float(selected.minutes_to_hit_after_observation.median()) if len(selected) else None,
                  ticker_counts=selected.symbol.value_counts().to_dict(),
                  supported=bool(len(selected) >= 20 and selected.date.nunique() >= 8))
    # Diagnostic: does it add beyond merely having already travelled toward +8%?
    if len(selected):
        bins = [-np.inf, -.05, -.02, 0, .01, .02, .03, .04, .05, .06, .07, .08, np.inf]
        pending["progress_bin"] = pd.cut(pending.progress.fillna(0), bins=bins, right=False).astype(str)
        reference = pending.groupby(["date", "progress_bin"]).target.mean()
        selected_bins = pending.loc[condition]
        ref_values = [reference.loc[(r.date, r.progress_bin)] for r in selected_bins.itertuples()]
        result["date_progress_matched_baseline"] = float(np.mean(ref_values))
        result["lift_after_progress_matching"] = result["rate"] - result["date_progress_matched_baseline"]
    if ci and len(selected):
        dates = sorted(panel.loc[panel.split.eq(split), "date"].unique()) if split != "all" else sorted(panel.date.unique())
        daily_base = pending.groupby("date").target.mean()
        triggered = selected.groupby("date").target.agg(["sum", "size"]).reindex(dates, fill_value=0)
        sums = triggered["sum"].to_numpy(float)
        counts = triggered["size"].to_numpy(float)
        expected = counts * daily_base.reindex(dates).fillna(0).to_numpy()
        rng = np.random.default_rng(20260926)
        block = 10
        starts = rng.integers(0, len(dates), (5000, (len(dates) + block - 1) // block))
        picks = ((starts[:, :, None] + np.arange(block)) % len(dates)).reshape(5000, -1)[:, :len(dates)]
        denominator = counts[picks].sum(axis=1)
        good = denominator > 0
        rate_boot = sums[picks].sum(axis=1)[good] / denominator[good]
        lift_boot = (sums - expected)[picks].sum(axis=1)[good] / denominator[good]
        result["rate_ci95"] = np.quantile(rate_boot, [.025, .975]).tolist()
        result["lift_ci95"] = np.quantile(lift_boot, [.025, .975]).tolist()
        result["p_value_one_sided"] = float((1 + np.sum((lift_boot - result["lift"]) >= result["lift"])) / (len(lift_boot) + 1))
        total_daily = total.groupby("date").target.agg(["sum", "size"]).reindex(dates, fill_value=0)
        tn = total_daily["size"].to_numpy()[picks].sum(axis=1)
        th = total_daily["sum"].to_numpy()[picks].sum(axis=1)
        result["total_rate_ci95"] = np.quantile(th[tn > 0] / tn[tn > 0], [.025, .975]).tolist()
    return result


def main():
    frozen_path = HERE / "frozen_candidates.json"
    frozen = json.loads(frozen_path.read_text())
    panel = pd.read_parquet(HERE / "features.parquet")
    assert hashlib.sha256((HERE / "features.parquet").read_bytes()).hexdigest() == frozen["feature_panel_sha256"]
    groups = json.loads((SOURCE / "groups.json").read_text())
    strategies = {x["strategy_id"]: x["name"] for x in json.loads((SOURCE / "strategies.json").read_text())}
    evaluated = []
    for item in frozen["top5"]:
        result = dict(item)
        result["strategy_name"] = strategies[item["strategy_id"]]
        result["confirmation"] = evaluate_rule(panel, item, "confirmation", True)
        result["all_history"] = evaluate_rule(panel, item, "all")
        result["top3_single_features"] = [{**x, "confirmation": evaluate_rule(panel, x, "confirmation", True),
                                            "all_history": evaluate_rule(panel, x, "all")}
                                           for x in item["top3_single_features"]]
        evaluated.append(result)
    order = sorted(range(len(evaluated)), key=lambda i: evaluated[i]["confirmation"].get("p_value_one_sided", 1))
    running = 0.
    for rank, i in enumerate(order):
        raw = evaluated[i]["confirmation"].get("p_value_one_sided", 1.)
        running = max(running, min(1., (len(evaluated) - rank) * raw))
        evaluated[i]["confirmation"]["p_value_holm_5"] = running
        evaluated[i]["confirmation"]["positive_lift_supported_after_holm"] = bool(running < .05 and evaluated[i]["confirmation"]["supported"] and evaluated[i]["confirmation"]["lift"] > 0)

    # Three distinct single features selected only by the already frozen selection scores.
    singles = [x for g in frozen["all_group_winners"].values() if g["status"] == "supported_candidate"
               for x in g["winner"]["top3_single_features"]]
    top3, seen = [], set()
    for item in sorted(singles, key=lambda x: x["selection"]["lcb"], reverse=True):
        feature = item["rule"][0][0]
        if feature in seen:
            continue
        seen.add(feature)
        top3.append({**item, "strategy_name": strategies[item["strategy_id"]],
                     "confirmation": evaluate_rule(panel, item, "confirmation", True),
                     "all_history": evaluate_rule(panel, item, "all")})
        if len(top3) == 3:
            break

    # Audit every group. Groups with no early development history get explicitly
    # labelled full-period single-rule exploration; these cannot enter the shortlist.
    catalogs = json.loads((HERE / "threshold_catalog.json").read_text())
    group_results = []
    for group in groups:
        gid = group["group_id"]
        g = panel[panel.group_ids.map(lambda ids: gid in ids) & ~panel.already_hit]
        original = frozen["all_group_winners"][gid]
        r = {"group_id": gid, "group_name": group["name"], "strategy_id": group["strategy_id"],
             "strategy_name": strategies[group["strategy_id"]], "status": original["status"],
             "unique_samples": g.sample_id.nunique(), "stocks": g.symbol.nunique(),
             "first_date": g.date.min() if len(g) else None, "last_date": g.date.max() if len(g) else None,
             "searched_supported_rules": original["searched_supported_rules"]}
        if original["status"] == "supported_candidate":
            winner = original["winner"]
            r["winner"] = {**winner, "confirmation": evaluate_rule(panel, winner, "confirmation"),
                           "all_history": evaluate_rule(panel, winner, "all")}
            r["reason"] = "有前段支持，已在后段按冻结规则复核；不等于已证明有效"
        elif len(g) == 0:
            r["reason"] = "没有可评价的群成员或完整分钟行情"
        else:
            best, examined = None, 0
            for checkpoint, cp in g.groupby("checkpoint"):
                for rule in catalogs[checkpoint]:
                    examined += 1
                    mask = mask_rule(cp, rule)
                    if mask.sum() < 20:
                        continue
                    metric = describe(cp, mask)
                    if metric["dates"] < 8 or metric["lift"] <= 0:
                        continue
                    if best is None or metric["lcb"] > best["metric"]["lcb"]:
                        best = {"checkpoint": checkpoint, "rule": rule, "text": rule_text(rule), "metric": metric}
            r["full_period_exploration"] = best
            r["full_period_rules_examined"] = examined
            r["reason"] = ("仍已搜索全可用期，但发现/选择段支持不足；全期挑优没有独立复核，不能进入Top5"
                           if best else "已搜索，未找到至少20样本、8日期且相对同日群基线为正的单项条件")
        group_results.append(r)

    top_total = sorted(evaluated, key=lambda x: (x["confirmation"]["total_rate"] is not None, x["confirmation"]["total_rate"] or -1), reverse=True)
    top_pending = sorted(evaluated, key=lambda x: (x["confirmation"]["rate"] is not None, x["confirmation"]["rate"] or -1), reverse=True)
    result = {"frozen_candidates_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
              "confirmation_dates": ["2026-07-20", "2026-09-17"], "confirmation_session_count": 43,
              "top5_by_total_probability": top_total, "top5_by_remaining_probability": top_pending,
              "top3_single_features": top3, "all_groups": group_results,
              "supported_after_holm_count": sum(x["confirmation"]["positive_lift_supported_after_holm"] for x in evaluated),
              "note": "Probabilities are exploratory historical frequencies, not calibrated future guarantees or a trading backtest."}
    (HERE / "results.json").write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
    render_report(result, frozen, panel)
    print(json.dumps(json_safe({"top5": [{"group": x["group_name"], "checkpoint": x["checkpoint"], "text": x["text"],
                                       "confirmation": x["confirmation"], "selection": x["selection"]} for x in top_total],
                                "top3": [{"group": x["group_name"], "text": x["text"], "confirmation": x["confirmation"]} for x in top3],
                                "supported_after_holm_count": result["supported_after_holm_count"]}), ensure_ascii=False, indent=2))


def pct(value):
    return "无法估计" if value is None else f"{value:.1%}"


def render_report(result, frozen, panel):
    def count(m, total=False):
        return f"{m['total_hits']}/{m['total_n']}" if total else f"{m['hits']}/{m['n']}"
    def interval(values):
        return "—" if not values else f"{values[0]:.1%}–{values[1]:.1%}"
    lines = ["# 第一日特征与群组：五日触及+8%的探索结果", "",
             "**先选条件，再看后段。** 发现段实际97个起始日（1月2日—5月21日），选择段28日（6月1日—7月10日）；边界各清除5个标签跨段交易日。最后43个起始日（7月20日—9月17日）只复核冻结规则，没有按结果替换五个候选。",
             "", "本轮仍是同一历史库探索。股票池和行业分类有当前名单回溯偏差，后段并非从未研究过的外部数据；同一天多股及跨日窗口存在关联。", "",
             f"检查58群、6个首日观察时点、{len(FEATURES)}项特征；发现段样本支持的单项及至多三项AND组合共有{frozen['supported_candidates']:,}条。仅105只证券有完整可用数据（101股票+4ETF），其余ETF不算失败。",
             "", "**共同起点与期限：** 每日11:35 ET的5分钟Open；五日=1,950常规交易分钟。下午条件只能在相应观察时刻之后知道。两个涨幅都相对原起点，不是看到条件后再涨8%。原口径总触达含观察时已达标；候选筛选和提升检验均先排除这些已知成功。", "",
             "## Top5：按后段原口径总触达率排序", "",
             "这五项在中段先冻结，再按后段总触达率排序；不是查看所有群后段结果再挑最高的五项。其余所有群另有逐组记录。", "",
             "| 排名 | 策略 | 群组 | 观察时点ET | 条件（同时满足） | 总触达率（命中/样本） | 未达标者后续触达率 | 同群同日基线 |", "|---:|---|---|---|---|---:|---:|---:|"]
    for rank, x in enumerate(result["top5_by_total_probability"], 1):
        m = x["confirmation"]
        lines.append(f"| {rank} | {x['strategy_name']} | {x['group_name']} | {x['checkpoint']} | {x['text']} | {pct(m['total_rate'])}（{count(m,True)}） | {pct(m['rate'])}（{count(m)}） | {pct(m['baseline'])} |")
    lines += ["", "## Top3单项特征及匹配群组", "",
              "按中段排序分数预先选三个不同特征，下面概率均为后段复核。这里的最佳群组指开发段选择的匹配，不宣称其他所有群在未来都更差。", "",
              "| 特征 | 策略 / 群组 | 时点ET | 总触达率 | 未达标者后续触达率 | 同群同日基线 |", "|---|---|---|---:|---:|---:|"]
    for x in result["top3_single_features"]:
        m = x["confirmation"]
        lines.append(f"| {x['text']} | {x['strategy_name']} / {x['group_name']} | {x['checkpoint']} | {pct(m['total_rate'])}（{count(m,True)}） | {pct(m['rate'])}（{count(m)}） | {pct(m['baseline'])} |")
    lines += ["", "## 每个入选群的三个单项条件与组合证据", ""]
    for x in result["top5_by_total_probability"]:
        m = x["confirmation"]
        lines += [f"### {x['strategy_name']} / {x['group_name']} · {x['checkpoint']} ET", "",
                  f"组合：{x['text']}。中段 {pct(x['selection']['rate'])}（{count(x['selection'])}）；后段未达标样本 {pct(m['rate'])}（{count(m)}），覆盖{m['dates']}日期、{m['stocks']}只证券；观察时已达标{m['already_hit_n']}例。", "",
                  f"后段未达标比例的日期块95%探索区间 {interval(m.get('rate_ci95'))}；相对同群同日基线提升 {m['lift']:+.1%}，区间 {interval(m.get('lift_ci95'))}；5项Holm校正p={m['p_value_holm_5']:.4f}。",
                  f"在进一步匹配当天已涨幅后，基线 {pct(m.get('date_progress_matched_baseline'))}，剩余提升 {m.get('lift_after_progress_matching',0):+.1%}（仅描述性诊断）。", "",
                  f"观察时达到原+8%目标仍需上涨的中位数 {pct(m['observed_price_remaining_upside_median'])}；后续低点相对观察价MAE中位数 {pct(m['future_mae_median'])}、10%分位 {pct(m['future_mae_p10'])}；到期收益中位数 {pct(m['terminal_median'])}。这些不是带止损或可成交成本的策略收益。", "",
                  "| 单项条件（不与其他条件同时施加） | 后段总触达率 | 后段未达标者后续触达率 | 中段比例 |", "|---|---:|---:|---:|"]
        for single in x["top3_single_features"]:
            sm = single["confirmation"]
            lines.append(f"| {single['text']} | {pct(sm['total_rate'])}（{count(sm,True)}） | {pct(sm['rate'])}（{count(sm)}） | {pct(single['selection']['rate'])} |")
        lines += ["", "后段样本证券分布：" + "、".join(f"{symbol} {n}" for symbol,n in m['ticker_counts'].items()) + "。", ""]
    lines += ["## 全部群组：尝试结果与不能入选的原因", "",
              "有些行业子群只有单一股票；它们仍参与了搜索，但单股结果不能外推为行业普遍规律。6个月趋势组在本数据中直到7月13日才首次可分组，没有可用发现段；仍做了明确标记的全期探索。", "",
              "| 策略 | 群组 | 可用股票日 / 证券数 | 开发段结果或限制 | 后段冻结条件比例 / 全期探索候选 |", "|---|---|---:|---|---|"]
    for g in result["all_groups"]:
        if "winner" in g:
            w = g["winner"]
            m = w["confirmation"]
            detail = f"{w['checkpoint']}：{w['text']}；{pct(m['rate'])}（{count(m)}）"
            state = "有开发段候选" + ("；后段支持不足" if not m['supported'] else "")
        elif g.get("full_period_exploration"):
            e = g['full_period_exploration']
            detail = f"全期挑优、无独立复核：{e['checkpoint']} {e['text']}；{pct(e['metric']['rate'])}（{count(e['metric'])}）"
            state = "无合格开发段候选；已继续全期搜索"
        else:
            detail = "—"
            state = g['reason']
        lines.append(f"| {g['strategy_name']} | {g['group_name']} | {g['unique_samples']} / {g['stocks']} | {state} | {detail} |")
    lines += ["", "## 验证与文件", "",
              f"冻结五项中，后段相对同群同日基线提升通过日期块检验及五项Holm校正的有{result['supported_after_holm_count']}项。区间是43天后段上的探索估计，不能解释为未来校准概率；Top3单项仅作解释，不另行宣称通过多重检验。", "",
              "`PROTOCOL.md`：预登记口径；`dataset_audit.json`：210份行情/公司行动源指纹与旧版一致、标签逐项对账；`features.parquet`：可复算特征；`threshold_catalog.json`与`development_search.parquet`：全部阈值和支持候选；`frozen_candidates.json`：在后段复核前冻结的候选；`results.json`：全部概率、计数、风险、区间与群组覆盖。", "",
              "复跑顺序：`build_dataset.py` → `search_rules.py` → `evaluate.py`。筛选脚本拒绝覆盖已冻结候选，避免看过后段结果后无记录地重新调参。", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
