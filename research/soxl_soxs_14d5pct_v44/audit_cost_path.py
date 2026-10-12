"""Post-result raw-bar audit of the non-monotone cost scenario; no policy change."""
import argparse,math
from conditional_calibration import *
from verify_continuation import near

def audit(run):
    run=Path(run);frozen=read(run/'decisions_frozen.json')['routes'];assert frozen['macro_context_band']==frozen['double_cost']
    market=core.load_market();manifest=read(run/'input_manifest.json');assert market.manifest==manifest['market']['full_market']
    at=pd.Timestamp('2026-04-24 13:25:30+00:00');proof={}
    for name,multiple in [('macro_context_band',1),('double_cost',2)]:
        df=pd.read_csv(run/(name+'_trades.csv'));trade=df[pd.to_datetime(df.decision_at,utc=True)==at].iloc[0]
        entry=pd.Timestamp(trade.entry_at);bars=market.bars[trade.symbol];path=bars[(bars.start_at>=entry)&(bars.end_at<=pd.Timestamp(trade.expiry_at))&bars.session_type.eq('regular')]
        near(trade.entry_fill_economic,path.iloc[0].econ_open*(1+.0005*multiple));first=None
        for row in path.itertuples():
            stop=math.floor(trade.entry_fill_economic*.94/row.share_factor*100+1e-10)/100*row.share_factor
            target=math.ceil(trade.entry_fill_economic*1.05/row.share_factor*100-1e-10)/100*row.share_factor
            if row.econ_low<=stop:
                first={'end_at':row.end_at,'start_at':row.start_at,'reason':'risk_stop','open':row.econ_open,'low':row.econ_low,'stop':stop,'fill':min(row.econ_open,stop)*(1-.0005*multiple),'gap':bool(row.econ_open<stop)};break
            if row.econ_high>=target:
                first={'end_at':row.end_at,'start_at':row.start_at,'reason':'target','fill':target};break
        assert first and pd.Timestamp(trade.exit_at)==first['end_at'] and trade.reason==first['reason']
        near(trade.exit_fill_economic,first['fill']);near(trade.net_return,(first['fill']*(1-.0005*multiple)+trade.dividend_per_economic_unit)/(trade.entry_fill_economic*(1+.0005*multiple))-1)
        proof[name]={'entry_fill':trade.entry_fill_economic,'net_return':trade.net_return,'first_exit':first}
    assert proof['double_cost']['first_exit']['end_at']<proof['macro_context_band']['first_exit']['end_at']
    stress_bar=market.bars['SOXL'].loc[proof['double_cost']['first_exit']['end_at']]
    assert proof['macro_context_band']['first_exit']['stop']<stress_bar.econ_low<=proof['double_cost']['first_exit']['stop']
    check_at=pd.Timestamp('2026-04-28 14:35:30+00:00');statuses={}
    for name in ['macro_context_band','double_cost']:
        ds=pd.read_csv(run/(name+'_decisions.csv'));statuses[name]=ds.loc[pd.to_datetime(ds.decision_at,utc=True)==check_at,'decision'].item()
    assert statuses=={'macro_context_band':'daily_entry_limit','double_cost':'buy_SOXL'}
    out={'passed':True,'post_result_only':True,'frozen_proposals_identical':True,'diagnostic_entry':at,'path':proof,'new_entry_at':check_at,'new_entry_statuses':statuses,
        'explanation':'Higher entry slippage raises attached6pctstop; April27low hits only stressstop, avoidingApril28gap and freeing settledcash for laterproposal. Costscenario is pathreplay, not simple subtractionfromidenticaltrades. No cost/stop parameters changed.',
        'results_sha256':sha(run/'results.json')}
    dump(run/'cost_path_audit.json',out);print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
