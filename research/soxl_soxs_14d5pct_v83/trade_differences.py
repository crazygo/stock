"""Describe changed executed entries after verification; do not alter the strategy."""
import argparse
from state_calibration import *


def compare(run):
    run=Path(run);assert read(run/'verification.json')['passed'];result=read(run/'results.json')
    def read_trades(name):
        try:return pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:return pd.DataFrame()
    a=read_trades('eligible_state');b=read_trades('identity_matched')
    def index(table):return {(t.decision_at,t.symbol):t for t in table.itertuples()}
    A,B=index(a),index(b);common=set(A)&set(B)
    for key in common:
        x,y=A[key],B[key];assert x.reason==y.reason and x.exit_at==y.exit_at
        np.testing.assert_allclose(x.net_return,y.net_return,atol=1e-13,rtol=1e-10)
    def details(source,keys):
        return [{'decision_at':source[k].decision_at,'symbol':source[k].symbol,'reason':source[k].reason,'net_return':source[k].net_return} for k in sorted(keys)]
    out={'stage':'post_run_descriptive_only','common_entry_keys':len(common),'same_returns_and_exits_for_common_entries':True,
        'only_primary':details(A,set(A)-set(B)),'only_identity_matched':details(B,set(B)-set(A)),
        'not_additive_monthly_attribution':True,'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),'no_policy_changes':True}
    dump(run/'trade_differences.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);compare(p.parse_args().run)
