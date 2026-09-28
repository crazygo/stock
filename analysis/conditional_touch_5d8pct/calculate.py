#!/usr/bin/env python3
"""Recompute the two requested historical conditional rates from frozen labels.

Run from the repository root with research/after_open_3d5pct/.venv/bin/python.
Reads local data only; does not train, fetch, trade, or overwrite source labels.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "research/group_expectation_matrix/outputs/20260925_v5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    instruments = json.loads((SOURCE / "instruments.json").read_text())
    symbols = sorted(x["symbol"] for x in instruments
                     if x["instrument_type"] == "stock" and x["candidate"])
    labels = pd.read_parquet(SOURCE / "outcomes.parquet")
    five = labels[labels.symbol.isin(symbols) & labels.expectation_id.eq("d5_r5")].copy()
    assert not five.sample_id.duplicated().any()
    samples = five[five.status.eq("mature")].copy().reset_index(drop=True)
    assert samples.hit.eq(samples.hit_minutes.notna().astype(int)).all()
    assert samples.loc[samples.hit.eq(1), "hit_minutes"].between(5, 1950).all()
    assert not ((samples.mfe - .08).abs() < 1e-12).any(), "Audit exact threshold ties"
    samples["hit_8pct_by_5d"] = samples.mfe.ge(.08)
    for days in (2, 3):
        samples[f"hit_5pct_by_{days}d"] = samples.hit.eq(1) & samples.hit_minutes.le(days * 390)

    # Independently stored 3-day labels must agree on every common 5-day sample.
    three = labels[labels.symbol.isin(symbols) & labels.expectation_id.eq("d3_r5")]
    paired = samples.merge(three[["sample_id", "status", "hit", "hit_minutes"]],
                           on="sample_id", suffixes=("", "_3"), validate="one_to_one")
    assert len(paired) == len(samples) and paired.status_3.eq("mature").all()
    assert paired.hit_5pct_by_3d.eq(paired.hit_3.eq(1)).all()
    hit3 = paired.hit_3.eq(1)
    assert paired.loc[hit3, "hit_minutes"].eq(paired.loc[hit3, "hit_minutes_3"]).all()
    assert (~samples.hit_5pct_by_2d | samples.hit_5pct_by_3d).all()

    # Direct raw-bar spot checks, stratified by both condition and outcome.
    checks = []
    for _, group in samples.groupby(["hit_5pct_by_2d", "hit_5pct_by_3d", "hit_8pct_by_5d"]):
        checks.extend(group.sample(min(3, len(group)), random_state=20260926).index)
    checks.extend(samples[samples.hit_minutes.isin([775, 780, 785, 1165, 1170, 1175])].index[:12])
    check_samples = samples.loc[sorted(set(checks))]
    for symbol, group in check_samples.groupby("symbol"):
        raw = pd.read_parquet(ROOT / "market_data/us_5m" / symbol / "2026.parquet")
        raw["start"] = pd.to_datetime(raw.start_at, utc=True)
        raw["end"] = pd.to_datetime(raw.end_at, utc=True)
        for row in group.itertuples():
            window = raw[raw.session_type.eq("regular") & raw.start.ge(pd.Timestamp(row.entry_at))
                         & raw.end.le(pd.Timestamp(row.label_end_at))].sort_values("start")
            assert len(window) == 390 and not window.start.duplicated().any()
            entry = float(window.iloc[0].open)
            assert entry == row.entry_price
            assert bool((window.high >= entry * 1.08).any()) == row.hit_8pct_by_5d
            for days in (2, 3):
                actual = bool((window.iloc[:days * 78].high >= entry * 1.05).any())
                assert actual == getattr(row, f"hit_5pct_by_{days}d")

    results, monthly = [], []
    dates = sorted(samples.date.unique())
    rng = np.random.default_rng(20260926)
    block = 10
    # Same sampled dates for both rates; retain each day's whole cross-section.
    starts = rng.integers(0, len(dates), (5000, (len(dates) + block - 1) // block))
    picks = ((starts[:, :, None] + np.arange(block)) % len(dates)).reshape(5000, -1)[:, :len(dates)]
    for days in (2, 3):
        condition = samples[f"hit_5pct_by_{days}d"]
        success = condition & samples.hit_8pct_by_5d
        daily = pd.DataFrame({"date": samples.date, "den": condition.astype(int), "num": success.astype(int)})
        daily = daily.groupby("date")[["num", "den"]].sum().reindex(dates)
        sums = daily.to_numpy()[picks].sum(axis=1)
        intervals = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975]).tolist()
        results.append({"condition_days": days, "denominator": int(condition.sum()),
                        "numerator": int(success.sum()), "rate": float(success.sum() / condition.sum()),
                        "date_block_bootstrap_95pct": intervals})
        subset = samples[condition].assign(month=samples.loc[condition, "date"].str[:7])
        for month, group in subset.groupby("month"):
            monthly.append({"condition_days": days, "month": month, "denominator": len(group),
                            "numerator": int(group.hit_8pct_by_5d.sum()), "rate": float(group.hit_8pct_by_5d.mean())})

    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_hashes": {name: sha(SOURCE / name) for name in ["outcomes.parquet", "instruments.json", "manifest.json", "config.json"]},
        "script_sha256": sha(Path(__file__)),
        "entry": "Each stock once per session at 11:35 ET, 5m Open; shared reference price",
        "horizons": "2/3/5 times 390 regular-session minutes, excluding extended hours",
        "target": "High touch >= +8% within 5 days, conditional on first +5% touch within 2 or 3 days",
        "candidate_stocks": len(symbols), "effective_stocks": int(samples.symbol.nunique()),
        "eligible_samples": len(samples), "entry_date_range": [min(dates), max(dates)],
        "entry_dates": len(dates), "all_status_counts": {str(k): int(v) for k, v in five.status.value_counts().items()},
        "unconditional_5d8pct": {"numerator": int(samples.hit_8pct_by_5d.sum()), "denominator": len(samples), "rate": float(samples.hit_8pct_by_5d.mean())},
        "results": results, "monthly": monthly,
        "audit": {"three_day_label_comparisons": len(paired), "raw_window_spot_checks": len(check_samples), "passed": True},
        "bootstrap": {"method": "circular moving date blocks with full cross-sections", "block_sessions": block, "replicates": 5000, "seed": 20260926},
        "limitations": ["Historical descriptive rate, not independently validated future probability",
                        "Overlapping stock-date windows are not independent trades",
                        "Current candidate-universe retrospective selection",
                        "Includes samples already at +8% within the 2/3-day condition period",
                        "Both targets refer to original entry; +8% is not an additional +8% after +5%",
                        "Touch is not closing return or an executable fill; no stop or costs included"],
    }
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    samples.to_parquet(HERE / "samples.parquet", index=False, compression="zstd", compression_level=7)
    lines = ["# 两组历史条件概率", "", "按现有101只股票研究池、每日11:35 ET的5分钟Open为共同起点。",
             "天数为常规盘交易分钟：2/3/5天分别为780/1170/1950分钟；仅统计High触及，排除盘前盘后。",
             "只纳入完整5天窗口成熟且行情有效、未跨不支持公司行动的股票日。", "",
             f"有效起始日期：{min(dates)}—{max(dates)}，{len(dates)}个日期、{len(samples):,}个股票日。", "",
             "| 条件 | 五天内+8% / 条件样本数 | 历史比例 | 日期块95%探索区间 |", "|---|---:|---:|---:|"]
    for item in results:
        lo, hi = item["date_block_bootstrap_95pct"]
        lines.append(f"| 前{item['condition_days']}天触及+5% | {item['numerator']:,} / {item['denominator']:,} | {item['rate']:.2%} | {lo:.2%}–{hi:.2%} |")
    base = result["unconditional_5d8pct"]
    lines += ["", f"同批完整样本不加前置条件时，五天内触及+8%的比例为 {base['rate']:.2%}（{base['numerator']:,}/{base['denominator']:,}）。",
              "", "上述两个条件彼此嵌套，前三天包含前两天；涨幅均相对原始起点。包含观察期内已经达到+8%的样本，不能解释为观察期结束后再达到目标的概率。",
              "", "这是当前股票池的历史描述统计，存在名单回溯和窗口重叠；不是实盘胜率或独立验证后的未来概率。",
              "", f"核验：{len(paired):,}个样本与独立保存的三日标签逐条一致；直接重读行情核对{len(check_samples)}个分层/边界样本，均通过。",
              "", "数据、排除数量、输入指纹、逐月统计见 `result.json`；股票日明细见 `samples.parquet`（Git忽略）。",
              "", "复跑：", "", "```bash", "research/after_open_3d5pct/.venv/bin/python analysis/conditional_touch_5d8pct/calculate.py", "```", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"results": results, "audit": result["audit"], "baseline": base}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
