"""Post-outcome descriptive partitions; never imported by the frozen policy."""
import argparse,math,statistics
from paired93 import *


def review(run):
    run=Path(run);audit=read(run/'diagnostic_audit.json');assert audit['passed'] and audit['diagnostics_sha256']==sha(run/'diagnostics.json')
    selected=pd.read_csv(run/'diagnostic_rows.csv');selected=selected[selected.route=='paired_fundamental']
    entries=pd.read_csv(run/'actual_entry_forecasts.csv');entries=entries[entries.route=='paired_fundamental'].copy()
    events={str(r.at):r for r in selected.itertuples()}
    entries['source_eligible']=[events[str(t)].source_eligible for t in entries['at']]
    fields=['value','actual','q0','y0'];groups={}
    for name,g in [('accepted',selected[selected.action!='CASH']),('accepted_original_eligible',selected[(selected.action!='CASH') & selected.source_eligible]),
        ('accepted_original_cash',selected[(selected.action!='CASH') & ~selected.source_eligible])]:
        known=g[g.ready & g.mature];bydate=known.groupby('date')[fields].mean();means=bydate.mean().to_dict() if len(bydate) else {}
        for field,v in means.items():
            days={}
            for r in known.itertuples():days.setdefault(r.date,[]).append(float(getattr(r,field)))
            independent=statistics.mean(statistics.mean(values) for values in days.values());assert math.isclose(v,independent,abs_tol=1e-13)
        groups[name]=dict(rows=len(g),known_rows=len(known),unknown_rows=int((~g.mature).sum()),known_dates=len(bydate),date_equal_means=means)
    actual={}
    for name,g in [('all',entries),('original_eligible',entries[entries.source_eligible]),('original_cash',entries[~entries.source_eligible])]+[(symbol,entries[entries.symbol==symbol]) for symbol in core.ETFS]:
        known=g[g.known];actual[name]=dict(entries=len(g),completed=len(known),targets=int((known.truth==0).sum()),
            ordinary_stops=int((known.truth==1).sum()),gap_stops=int((known.truth==2).sum()),
            mean_predicted_target=float(known.q0.mean()) if len(known) else None,
            mean_predicted_log_return=float(known.value.mean()) if len(known) else None,
            mean_actual_log_return=float(known.actual_log_return.mean()) if len(known) else None)
    primary=read(run/'results.json')['routes'];out=dict(stage='post_run_descriptive_only',strategy_changed=False,groups=groups,actual_entries=actual,
        arithmetic_monthly_mean_differences=dict(training_population_fixed_source=primary['paired_fixed_source']['primary_average']-primary['bounded_fundamental']['primary_average'],
        release_original_eligibility_fixed_top=primary['paired_fixed_top']['primary_average']-primary['paired_fixed_source']['primary_average'],
        allow_direction_reselection=primary['paired_fundamental']['primary_average']-primary['paired_fixed_top']['primary_average']),
        inference='Exact historical policy comparisons and descriptive subgroups, not future-causal guarantees or additive trade contributions.',
        inputs_sha256={f:sha(run/f) for f in ['results.json','diagnostic_rows.csv','actual_entry_forecasts.csv','diagnostic_audit.json']},source_sha256=sha(Path(__file__)))
    dump(run/'selection_review.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);review(p.parse_args().run)
