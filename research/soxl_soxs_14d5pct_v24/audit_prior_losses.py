"""Describe v23 development losses before registering a new loss model."""
from pathlib import Path
import sys,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v23'))
from clock_training import core,restrict_market
from run_clocks import tag
from verify_continuation import read,labels,key,near


def audit():
    source=core.ROOT/'.cache/soxl_soxs_14d5pct_v23/runs/clocks_20261012'
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v23/evidence/delivery_manifest.json')
    for name in ['results.json','verification.json','predictions.json','context_sequential_trades.csv','baseline_labels.json']:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==delivery['cache_outputs_sha256'][name]
    market=restrict_market(core.load_market(),pd.Timestamp('2026-05-01',tz=core.ET))
    labs=labels(source/'baseline_labels.json');trades=pd.read_csv(source/'context_sequential_trades.csv');out=[]
    for t in trades.itertuples():
        at=pd.Timestamp(t.decision_at);lab=labs[key(at,t.symbol)];rec=read(source/'training'/(tag(at)+'.json'))
        forecast=rec['variants']['context']['forecasts']['early']['event'];raw_top='SOXL' if forecast['raw']['D']>=0 else 'SOXS'
        opposite='SOXS' if t.symbol=='SOXL' else 'SOXL';other=labs[key(at,opposite)]
        frame=market.bars[t.symbol];exitbar=frame.loc[lab['exit_at']];factor=exitbar.share_factor
        stop=np.floor(lab['entry_fill_economic']*.94/factor*100+1e-10)/100*factor
        no_gap_exit=stop*(1-core.SLIP);no_gap_net=(no_gap_exit*(1-core.FEE)+lab['dividend_per_economic_unit'])/(lab['entry_fill_economic']*(1+core.FEE))-1
        path=frame[(frame.start_at>=lab['entry_at'])&(frame.end_at<lab['exit_at'])&frame.session_type.eq('regular')]
        rawalternative=labs[key(at,raw_top)].get('net_return',np.nan)
        out.append({'at':at,'clock':rec['clock'],'symbol':t.symbol,'net_return':t.net_return,'reason':t.reason,
            'raw_symbol':raw_top,'calibration_flipped_direction':raw_top!=t.symbol,'raw_alternative_net_return':rawalternative,
            'opposite_net_return':other.get('net_return',np.nan),'both_etfs_nonpositive':t.net_return<=0 and other.get('net_return',np.nan)<=0,
            'gap_below_stop':lab.get('gap_below_stop',False),'same_bar_both_barriers':lab.get('same_bar_both_barriers',False),
            'gap_excess_return_loss':max(0.,no_gap_net-t.net_return) if t.reason=='risk_stop' else 0.,
            'strict_prior_bar_best_return':path.econ_high.max()/lab['entry_fill_economic']-1 if len(path) else None,
            'dividend':lab['dividend_per_economic_unit']})
    losses=[r for r in out if r['net_return']<0];flipped=[r for r in out if r['calibration_flipped_direction']]
    stats={'completed_trades':len(out),'losses':len(losses),'calibration_direction_flips':len(flipped),
        'flipped_better_than_raw':sum(r['net_return']>r['raw_alternative_net_return'] for r in flipped),
        'losses_where_opposite_profitable':sum(r['opposite_net_return']>0 for r in losses),
        'losses_both_nonpositive':sum(r['both_etfs_nonpositive'] for r in losses),
        'losses_after_calibration_flip':sum(r['calibration_flipped_direction'] for r in losses),
        'losses_with_gap':sum(r['gap_below_stop'] for r in losses),
        'sum_loss_trade_percentages':sum(r['net_return'] for r in losses),
        'sum_gap_excess_trade_percentages':sum(r['gap_excess_return_loss'] for r in losses),
        'ambiguous_loss_bars':sum(r['same_bar_both_barriers'] for r in losses),
        'losses_prior_favorable_2pct':sum(r['strict_prior_bar_best_return'] is not None and r['strict_prior_bar_best_return']>=.02 for r in losses),
        'not_a_capital_monthly_decomposition':True}
    result={'source_run':str(source),'source_results_sha256':delivery['cache_outputs_sha256']['results.json'],
        'prior_version_posthoc_diagnostic':True,'new_strategy_run':False,'statistics':stats,'trades':out}
    core.dump(Path(__file__).parent/'evidence/prior_loss_audit.json',result);print(stats)


if __name__=='__main__':audit()
