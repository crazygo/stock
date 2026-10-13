"""Explain actual primary/stress differences from immutable raw execution bars."""
import argparse
from history_learning import *


def check(run):
    run=Path(run);market=load_market();a=pd.read_csv(run/'long_price_band_trades.csv');b=pd.read_csv(run/'double_cost_trades.csv')
    assert list(zip(a.decision_at,a.symbol))==list(zip(b.decision_at,b.symbol))
    changes=[];gaps=[]
    for x,y in zip(a.itertuples(),b.itertuples()):
        if x.reason!=y.reason:
            bars=market.bars[x.symbol];end=pd.Timestamp(x.exit_at);bar=bars.loc[end]
            assert x.reason=='target' and y.reason=='risk_stop'
            assert float(bar.high)>=x.limit_raw and float(bar.high)<y.limit_raw
            # No split between these fixed dates; raw and economic scales agree within the trade.
            assert x.entry_share_factor==y.entry_share_factor
            prefix=bars[(bars.start_at>=pd.Timestamp(y.entry_at))&(bars.end_at<pd.Timestamp(y.exit_at))&bars.session_type.eq('regular')]
            assert float(prefix.high.max())<y.limit_raw
            changes.append({'decision_at':x.decision_at,'symbol':x.symbol,'original_entry_fill':x.entry_fill_raw,'stress_entry_fill':y.entry_fill_raw,
                'original_target':x.limit_raw,'stress_target':y.limit_raw,'original_exit_bar_high':float(bar.high),'original_exit_at':x.exit_at,
                'stress_exit_at':y.exit_at,'max_high_before_stress_stop':float(prefix.high.max()),'original_return':x.net_return,'stress_return':y.net_return})
        if bool(x.gap_below_stop):
            bar=market.bars[x.symbol].loc[pd.Timestamp(x.exit_at)];raw_stop=x.stop_economic/x.entry_share_factor
            assert bar.open<raw_stop
            gaps.append({'decision_at':x.decision_at,'symbol':x.symbol,'exit_at':x.exit_at,'raw_stop':raw_stop,
                'exit_bar_open':float(bar.open),'actual_net_return':x.net_return})
    out={'passed':True,'stage':'post_run_descriptive_only','same_entry_sequence_under_double_cost':True,
        'reason_changes':changes,'primary_gap_below_stop':gaps,'results_sha256':sha(run/'results.json'),
        'source_sha256':sha(Path(__file__)),'policy_or_data_changed':False}
    dump(run/'execution_sensitivity.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);check(p.parse_args().run)
