"""Post-fit stability and multiplicity sensitivity, never a selection rule.

This diagnostic family combines all fifteen registered input/algorithm pairs.
The exact paired two-week sign-flip test assumes symmetric block differences;
it is supplementary evidence, not a causal test or an independent validation.
"""
import json
import numpy as np
import pandas as pd
from common import OUT,write,now,sha
from calibration_audit import load,KEY
from research import weights

PAIRS={'R00':[('A1','A0'),('B1','A0'),('B2','B1'),('ALG_LR','A0'),('ALG_ET','A0')],
 'R01':[('S1','S0'),('S2','S0'),('S3','S1')], 'R02':[('P1','P0'),('E1','E0')],
 'R03':[('H1','H0'),('H2','H1'),('H3','H2')], 'R04':[('F1','F0')], 'R05':[('D1','D0')]}

def exact_block_test(frame,delta,w):
    week=(pd.to_datetime(frame.day)-pd.to_timedelta(pd.to_datetime(frame.day).dt.weekday,unit='D'))
    block=((week-week.min()).dt.days//14).to_numpy()
    sums=pd.DataFrame({'block':block,'sum':delta*w}).groupby('block')['sum'].sum().to_numpy()
    assert len(sums)<=16,'bounded exact diagnostic requires at most sixteen date blocks'
    combinations=np.arange(2**len(sums),dtype=np.uint32)[:,None]
    signs=2*((combinations>>np.arange(len(sums)))&1).astype(float)-1
    observed=abs(sums.sum());simulated=abs(signs@sums)
    return dict(two_week_blocks=len(sums),two_sided_p=float(np.mean(simulated>=observed-1e-12)),
        permutations=len(simulated),assumption='independent symmetric paired date-block loss differences; sensitivity only')

def main():
    rows=[];sources=[]
    for rid,pairs in PAIRS.items():
        base=OUT if rid=='R00' else OUT/rid
        if not (base/'calibration_audit.json').exists():continue
        sources.append(dict(round=rid,file=str((base/'calibration_audit.json').relative_to(OUT)),sha256=sha(base/'calibration_audit.json')))
        for newer,reference in pairs:
            f=load(base,newer,'candidate_only');g=load(base,reference,'candidate_only')
            pair=f.merge(g[KEY+['score','y']],on=KEY,suffixes=('_new','_reference'),validate='one_to_one')
            assert np.array_equal(pair.y_new.fillna(-1),pair.y_reference.fillna(-1))
            pair['y']=pair.y_new;pair=pair[pair.y.notna()].copy();w=weights(pair)
            delta=((pair.score_new-pair.y)**2-(pair.score_reference-pair.y)**2).to_numpy()
            item=dict(round=rid,newer=newer,reference=reference,stock_days=pair[KEY[:2]].drop_duplicates().shape[0],
                brier_delta=float(np.average(delta,weights=w)),monthly=[],**exact_block_test(pair,delta,w))
            for month,indices in pair.groupby(pair.day.str[:7]).indices.items():
                z=pair.iloc[indices];item['monthly'].append(dict(month=month,stock_days=z[KEY[:2]].drop_duplicates().shape[0],
                    brier_delta=float(np.average(delta[indices],weights=w[indices]))))
            rows.append(item);print(json.dumps(dict(round=rid,newer=newer,p=item['two_sided_p'])),flush=True)
    ordered=sorted(rows,key=lambda r:r['two_sided_p']);maximum=0.
    for rank,item in enumerate(ordered):
        maximum=max(maximum,min(1.,item['two_sided_p']*(len(rows)-rank)));item['holm_adjusted_p']=maximum
        item['holm_significant_improvement']=item['brier_delta']<0 and maximum<.05
    write(OUT/'input_stability_audit.json',dict(at=now(),status='post_fit_exploratory_sensitivity',family_size=len(rows),comparisons=rows,sources=sources,
        caveat='Post-fit sensitivity does not alter registered gates, models, thresholds or signals. Block symmetry and independence are assumptions. A low p-value would not establish future replication or reliable first-candidate probabilities.'))
    lines=['# 相同候选的分月与多次比较敏感性','','这是训练后诊断，不修改预登记门禁或选择模型。相同股票/日期/分钟按股票日等权；两周非重叠日期块做精确双侧符号翻转，假设块差值独立且对称。对全部15个预登记输入与算法比较应用Holm；与原两周移动块区间一起查看。','',
        '| 轮次 / 比较 | Brier新减旧 | 原p | Holm p | 改善月份 / 有覆盖月份 |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['round']} {r['newer']}/{r['reference']} | {r['brier_delta']:.6f} | {r['two_sided_p']:.4f} | {r['holm_adjusted_p']:.4f} | {sum(m['brier_delta']<0 for m in r['monthly'])}/{len(r['monthly'])} |")
    lines+=['','原p与校正p都是上述假设下的诊断，不能升级为因果或未来独立通过。逐月数据、日期块数及源SHA见input_stability_audit.json。已有全部失败不因新增诊断改名通过。']
    (OUT/'INPUT_STABILITY.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
