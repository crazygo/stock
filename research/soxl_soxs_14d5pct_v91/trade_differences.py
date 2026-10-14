"""Describe changed executed entries after verification; do not alter the strategy."""
import argparse
from calibration91 import *


def compare(run):
    run=Path(run);assert read(run/'verification.json')['passed'];result=read(run/'results.json')
    def read_trades(name):
        try:return pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:return pd.DataFrame()
    a=read_trades('eligible_fundamental');b=read_trades('eligible_reaction')
    def index(table):return {(t.decision_at,t.symbol):t for t in table.itertuples()}
    A,B=index(a),index(b);common=set(A)&set(B)
    same=all(A[k].reason==B[k].reason and A[k].exit_at==B[k].exit_at and abs(A[k].net_return-B[k].net_return)<1e-12 for k in common)
    def details(source,keys):
        return [{'decision_at':source[k].decision_at,'symbol':source[k].symbol,'reason':source[k].reason,'net_return':source[k].net_return} for k in sorted(keys)]
    out={'stage':'post_run_descriptive_only','common_entry_keys':len(common),'same_returns_and_exits_for_common_entries':same,
        'only_primary':details(A,set(A)-set(B)),'only_reaction_control':details(B,set(B)-set(A)),
        'not_additive_monthly_attribution':True,'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),'no_policy_changes':True}
    dump(run/'trade_differences.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);compare(p.parse_args().run)
