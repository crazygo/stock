"""Raw-OHLC reconstruction of gap labels and exact old payoff/maturity checks."""
import argparse,math
from class_learning import *
from prepare_classes import source_labels


def audit(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json')
    original=SOURCE/'evidence/input_protocol.json'
    assert manifest['protocol_sha256']==sha(original if original.exists() else SOURCE/'protocol.json')
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    assert manifest['parent_delivery_sha256']==verify_parent()
    market=extended_market();labs=source_labels();checks=gap_checks=unknown=0;counts={}
    for pair,(long,short,_) in PAIRS.items():
        p=pd.read_parquet(inputs/(pair+'_panel.parquet'));old=pd.read_parquet(PARENT/(pair+'_panel.parquet'))
        assert p[old.columns].equals(old)
        for at,row in p.iterrows():
            for side,symbol in [('L',long),('S',short)]:
                lab=labs[at.isoformat()+'|'+symbol]
                if pd.isna(row.terminal_release):
                    assert row[side+'_gap']==-1 and row[side+'_merged']==-1 and pd.isna(row[side+'_log']);unknown+=1;continue
                assert lab['complete_window'] and lab['label_available_at'] is not None
                if lab['reason']=='target':gap=0;merged=0
                elif lab['reason']=='expiry':gap=3;merged=2
                else:
                    assert lab['reason']=='risk_stop';entry=pd.Timestamp(lab['entry_at']).tz_convert(core.ET).tz_localize(None).normalize()
                    end=pd.Timestamp(lab['exit_at']);bar=market.bars[symbol].loc[end];last=bar.date
                    ratio=math.prod(float(a['value']) for a in market.actions[symbol] if a['type']=='Splits' and entry<pd.Timestamp(a['date'])<=last)
                    stop=math.floor(float(lab['entry_fill_raw'])*.94/ratio*100+1e-10)/100
                    assert float(bar.low)<=stop
                    isgap=float(bar.open)<stop;assert lab['gap_below_stop']==isgap
                    gap=2 if isgap else 1;merged=1;gap_checks+=1
                assert int(row[side+'_gap'])==gap and int(row[side+'_merged'])==merged
                np.testing.assert_allclose(row[side+'_log'],math.log1p(lab['net_return']),atol=1e-12,rtol=1e-12)
                counts[symbol+'|'+str(gap)]=counts.get(symbol+'|'+str(gap),0)+1;checks+=1
    assert counts==manifest['known_class_counts']
    for f in ['features.parquet','baseline_labels.json','stress_labels.json']:assert sha(inputs/f)==sha(PARENT/f)
    out={'passed':True,'known_class_and_payoff_checks':checks,'raw_stop_gap_checks':gap_checks,'unknown_retained':unknown,
        'all_parent_panel_values_exact':True,'old_ledger_inputs_exact':True,'input_manifest_sha256':sha(inputs/'input_manifest.json'),
        'source_sha256':sha(Path(__file__)),'capital_outcomes_evaluated':False}
    dump(inputs/'input_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);audit(p.parse_args().inputs)
