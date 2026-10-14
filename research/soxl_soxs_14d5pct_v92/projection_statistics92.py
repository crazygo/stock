"""Post-verification description of bounded corrections; no policy changes."""
import argparse
from bounded92 import *


def summarize(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['projection_rows_sha256']==sha(run/'projection_queries.csv')
    table=pd.read_csv(run/'projection_queries.csv');p=read(SOURCE/'protocol.json')['development']
    assert len(table)==v['convex_projection_certificates'];out={}
    for name,group in table.groupby('route'):
        selected=group[group.source_eligible & group.date.between(p['evaluation_start'],p['evaluation_end'])]
        out[name]={}
        for scope,g in [('all_ready',group),('development_eligible',selected)]:
            out[name][scope]=dict(queries=len(g),dates=g.date.nunique(),modified_queries=int((g.distance>1e-9).sum()),
                empty_context_queries=int((g.context_rows==0).sum()),changed_actions=int((g.raw_action!=g.bounded_action).sum()),
                maximum_projection_distance=float(g.distance.max()),maximum_optimality_gap=float(g.optimality_gap.max()),
                mean_raw_norm=float(g.raw_norm.mean()),mean_bounded_norm=float(g.bounded_norm.mean()),
                maximum_norm_increase=float((g.bounded_norm-g.raw_norm).max()))
    result=dict(stage='post_run_descriptive_only',rows_sha256=sha(run/'projection_queries.csv'),routes=out,
        source_sha256=sha(Path(__file__)),no_policy_changes=True)
    dump(run/'projection_statistics.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);summarize(p.parse_args().run)
