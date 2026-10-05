"""Matched loss differences and multiplicity, preserving prior diagnostic bytes."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import OUT,sha,write,now
from calibration_audit import load,KEY
from research import weights
from input_stability_audit import exact_block_test
PAIRS=[('TC','T0'),('TF','TC'),('TO','TC'),('TV','TC'),('TS','TC'),('TA','TC'),('TL','TA')]

def main():
    base=OUT/'R06';previous=json.loads((OUT/'input_stability_audit.json').read_text());rows=[]
    for newer,reference in PAIRS:
        f=load(base,newer,'candidate_only');g=load(base,reference,'candidate_only');pair=f.merge(g[KEY+['score','y']],on=KEY,suffixes=('_new','_reference'),validate='one_to_one');assert np.array_equal(pair.y_new.fillna(-1),pair.y_reference.fillna(-1));pair['y']=pair.y_new;pair=pair[pair.y.notna()].copy();w=weights(pair)
        delta=((pair.score_new-pair.y)**2-(pair.score_reference-pair.y)**2).to_numpy();item=dict(round='R06',newer=newer,reference=reference,stock_days=pair[KEY[:2]].drop_duplicates().shape[0],brier_delta=float(np.average(delta,weights=w)),monthly=[],**exact_block_test(pair,delta,w))
        for month,indices in pair.groupby(pair.day.str[:7]).indices.items():
            z=pair.iloc[indices];item['monthly'].append(dict(month=month,stock_days=z[KEY[:2]].drop_duplicates().shape[0],brier_delta=float(np.average(delta[indices],weights=w[indices]))))
        rows.append(item);print(json.dumps(dict(newer=newer,p=item['two_sided_p'])),flush=True)
    for family,field in [(rows,'holm_within_R06'),(previous['comparisons']+rows,'holm_across_all_22')]:
        maximum=0.
        for rank,r in enumerate(sorted(family,key=lambda r:r['two_sided_p'])):
            maximum=max(maximum,min(1.,r['two_sided_p']*(len(family)-rank)));r[field]=maximum
    write(base/'input_stability_audit.json',dict(at=now(),status='completed_post_fit_diagnostic',comparisons=rows,family_size_R06=len(rows),family_size_all=len(previous['comparisons'])+len(rows),previous_source_sha256=sha(OUT/'input_stability_audit.json'),assumption='two-week symmetric paired loss blocks; sensitivity only, no model selection or causal/future pass'))
    lines=['# 新信息的相同候选增量','','全部对照保持股票/日期/分钟相同，按股票日等权。负Brier差更好；95%移动日期块区间在CALIBRATION.md。下表额外给两周非重叠块符号翻转，并对本轮7次、连同旧轮22次比较应用Holm。不会用诊断重选模型。','','| 输入比较 | Brier差 | 改善月份 | 原p | 本轮Holm | 全部22次Holm |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['newer']}/{r['reference']} | {r['brier_delta']:.6f} | {sum(m['brier_delta']<0 for m in r['monthly'])}/5 | {r['two_sided_p']:.4f} | {r['holm_within_R06']:.4f} | {r['holm_across_all_22']:.4f} |")
    lines+=['','低p不认证历史PIT，不证明未来稳定复现，也不直接证明第一名概率可信。TC/T0主要检查缺失/覆盖信息影响；股票覆盖与历史记录修订风险另报。']
    (base/'INPUT_STABILITY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
