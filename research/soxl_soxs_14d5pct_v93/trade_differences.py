"""Changed actual entries across the matched selection controls; post-run only."""
import argparse
from paired93 import *


def compare(run):
    run=Path(run);assert read(run/'verification.json')['passed'];result=read(run/'results.json')
    def trades(name):
        try:return {(t.decision_at,t.symbol):t for t in pd.read_csv(run/(name+'_trades.csv')).itertuples()}
        except pd.errors.EmptyDataError:return {}
    def details(source,keys):
        return [dict(decision_at=source[k].decision_at,symbol=source[k].symbol,reason=source[k].reason,net_return=source[k].net_return) for k in sorted(keys)]
    A=trades('paired_fundamental');comparisons={}
    for name in ['bounded_fundamental','paired_fixed_source','paired_fixed_top','paired_identity','paired_reaction','paired_raw']:
        B=trades(name);common=set(A)&set(B)
        same=all(A[k].reason==B[k].reason and A[k].exit_at==B[k].exit_at and abs(A[k].net_return-B[k].net_return)<1e-12 for k in common)
        comparisons[name]=dict(common_entry_keys=len(common),same_returns_and_exits_for_common_entries=same,
            only_primary=details(A,set(A)-set(B)),only_control=details(B,set(B)-set(A)))
    out=dict(stage='post_run_descriptive_only',comparisons=comparisons,not_additive_monthly_attribution=True,results_sha256=sha(run/'results.json'),source_sha256=sha(Path(__file__)),no_policy_changes=True)
    dump(run/'trade_differences.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);compare(p.parse_args().run)
