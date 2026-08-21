#!/usr/bin/env python3
"""One-time source migration: make Apr-May pullback clustering fit all matches."""

from pathlib import Path


path = Path("analyze_apr_may_peak.py")
text = path.read_text(encoding="utf-8")

old_fit = '''    fit_records = [row for row in records if row["liquid"]]\n    if len(fit_records) < 30:\n        fit_records = records\n    clustering = learn_drawdown_groups(\n        [float(row["drawdown_from_peak_pct"]) / 100.0 for row in fit_records]\n    )\n'''
new_fit = '''    # The user-defined class is the full matched population. Liquidity is a\n    # presentation/tradability lens only, so it must not determine the natural\n    # drawdown boundaries.\n    fit_records = records\n    clustering = learn_drawdown_groups(\n        [float(row["drawdown_from_peak_pct"]) / 100.0 for row in fit_records]\n    )\n'''
if old_fit not in text:
    raise SystemExit("expected fit_records source block not found")
text = text.replace(old_fit, new_fit, 1)

old_population = '        "clustering_fit_population": "liquid_matches" if len([r for r in records if r["liquid"]]) >= 30 else "all_matches",\n'
new_population = '        "clustering_fit_population": "all_matches",\n'
if old_population not in text:
    raise SystemExit("expected clustering_fit_population source line not found")
text = text.replace(old_population, new_population, 1)

old_method = '            "group_learning": "1D k-means; k=3..5 chosen by silhouette minus small complexity penalty; p95 winsorization for fit only",\n'
new_method = '            "group_learning": "1D k-means fit on all matched stocks; k=3..5 chosen by silhouette minus small complexity penalty; p95 winsorization for fit only",\n'
if old_method not in text:
    raise SystemExit("expected group_learning source line not found")
text = text.replace(old_method, new_method, 1)

path.write_text(text, encoding="utf-8")
print("patched analyze_apr_may_peak.py to fit clustering on all matched stocks")
