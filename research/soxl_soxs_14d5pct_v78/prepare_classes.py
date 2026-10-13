"""Register actual contractual exit classes while preserving every parent input."""
import argparse
import shutil
from datetime import datetime,timezone
from class_learning import *


def source_labels():
    a=labels(PARENT/'earlier_labels.json');b=labels(parent.PARENT/'all_labels.json')
    assert not set(a)&set(b)
    return {**a,**b}


def prepare(output):
    output=Path(output);assert not output.exists();parent_sha=verify_parent();output.mkdir(parents=True)
    dump(output/'input_registration.json',{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(SOURCE/'protocol.json'),
        'parent_delivery_sha256':parent_sha,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in SOURCE.glob('*.py')},
        'new_strategy_capital_outcomes_seen':False})
    labs=source_labels();counts={};unknown={}
    for pair,(long,short,_) in PAIRS.items():
        panel=pd.read_parquet(PARENT/(pair+'_panel.parquet'));original=panel.copy();out={k:[] for k in ['L_gap','S_gap','L_merged','S_merged','L_log','S_log']}
        for at,row in original.iterrows():
            for side,symbol in [('L',long),('S',short)]:
                lab=labs[at.isoformat()+'|'+symbol];known=pd.notna(row.terminal_release)
                if known:
                    assert lab['complete_window'] and lab['label_available_at'] is not None
                    reason=lab['reason'];assert reason in ['target','risk_stop','expiry']
                    gap=0 if reason=='target' else 3 if reason=='expiry' else 2 if lab['gap_below_stop'] else 1
                    merged=0 if reason=='target' else 2 if reason=='expiry' else 1
                    value=float(np.log1p(lab['net_return']));counts[symbol+'|'+str(gap)]=counts.get(symbol+'|'+str(gap),0)+1
                else:gap=merged=-1;value=np.nan;unknown[symbol]=unknown.get(symbol,0)+1
                out[side+'_gap'].append(gap);out[side+'_merged'].append(merged);out[side+'_log'].append(value)
        for k,v in out.items():panel[k]=v
        assert panel[original.columns].equals(original)
        known=panel.terminal_release.notna()
        np.testing.assert_allclose(panel.loc[known,'L_log'],panel.loc[known,'C']+panel.loc[known,'D'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(panel.loc[known,'S_log'],panel.loc[known,'C']-panel.loc[known,'D'],atol=1e-12,rtol=1e-12)
        panel.to_parquet(output/(pair+'_panel.parquet'))
    for f in ['features.parquet','baseline_labels.json','stress_labels.json']:shutil.copyfile(PARENT/f,output/f)
    dump(output/'input_manifest.json',{'protocol_sha256':sha(SOURCE/'protocol.json'),'parent_delivery_sha256':parent_sha,
        'parent_input_manifest_sha256':sha(PARENT/'input_manifest.json'),
        'label_sources_sha256':{str(f.relative_to(ROOT)):sha(f) for f in [PARENT/'earlier_labels.json',parent.PARENT/'all_labels.json']},
        'known_class_counts':counts,'unknown_by_symbol':unknown,'derived_files_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    print({'known':sum(counts.values()),'unknown':unknown},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);prepare(p.parse_args().output)
