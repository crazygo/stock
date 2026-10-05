#!/usr/bin/env python3
"""Inspect already-frozen price-eligible estimates; never alter the joint signal policy."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('doubling_frozen_probability_diagnostic',PARENT/'model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def percent(x):return '不可评分' if x is None else f'{x:.2%}'

def main():
    pointer=json.loads((HERE/'latest_backtest.json').read_text());back=Path(pointer['path'])
    lineage=json.loads((back/'lineage.json').read_text());source=Path(pointer['prediction_path'])
    if sha(source)!=lineage['artifact_hashes'][str(source)]:raise ValueError('Frozen prediction source changed')
    protocol=HERE/'PROBABILITY_DIAGNOSTIC_PROTOCOL.md'
    registration={'kind':'pre_registered_v4_diagnostic_of_frozen_predictions','training_key':pointer['training_key'],
        'prediction_path':str(source),'prediction_sha256':sha(source),
        'protocol_sha256':sha(protocol),'diagnostic_code_sha256':sha(Path(__file__)),
        'evaluation_code_sha256':sha(PARENT/'model.py'),'config':lineage['config'],
        'main_joint_signal_policy_changed':False,'new_independent_effect_evidence':False}
    key=hashlib.sha256(json.dumps(registration,sort_keys=True).encode()).hexdigest()
    out=HERE/'diagnostics'/key[:16];out.mkdir(parents=True,exist_ok=True)
    prior=out/'lineage.json'
    if prior.exists():
        old=json.loads(prior.read_text())
        if old['registration']!=registration:raise ValueError('Diagnostic registration differs')
        for file,digest in old['artifacts'].items():
            if sha(file)!=digest:raise ValueError('Existing frozen diagnostic artifact changed: '+file)
        (HERE/'latest_probability_diagnostic.json').write_text(json.dumps({'path':str(out.resolve()),'key':key,'training_key':pointer['training_key']},indent=2))
        print(json.dumps({'path':str(out.resolve()),'reused_frozen_diagnostic':True},indent=2));return
    pred=pd.read_parquet(source)
    stats=model.evaluate(pred,lineage['config'],out)
    for s in stats:
        s['diagnostic_only']=True;s['joint_quality_and_heat_gate_applied']=False
        s['historical_numeric_gate_in_price_diagnostic']=s.pop('historical_numeric_gate')
        s['deployment_qualified']=False
        s['blocking_evidence'].append('price_eligible_diagnostic_not_joint_valid_signal')
    (out/'backtest_summary.json').write_text(json.dumps(stats,indent=2))
    sig=pd.read_csv(out/'backtest_signals.csv',keep_default_na=False,na_values=[''])
    by_stock=[]
    for (h,a,t),q in sig.groupby(['horizon','algorithm','ticker']):
        known=q[q.label.notna()];tp=int(known.label.sum());fp=len(known)-tp
        by_stock.append({'horizon':int(h),'algorithm':a,'ticker':t,'signals':len(q),'tp':tp,'fp':fp,
            'unknown':len(q)-len(known),'precision':tp/len(known) if len(known) else None,
            'mean_estimated_probability':float(q.probability.mean()),'dates':sorted(q.date.str[:10].tolist())})
    pd.DataFrame(by_stock).to_csv(out/'per_stock_diagnostic.csv',index=False)
    evidence={'recorded_at':datetime.now(timezone.utc).isoformat(),'registration':registration,'key':key,
        'prediction_rows_not_independent_samples':len(pred),'joint_signal_eligible_rows':int(pred.signal_eligible.sum()),
        'scope':'all frozen price-eligible rows, without joint company-quality and industry-heat signal gate',
        'summary':stats,'per_stock':by_stock,'no_model_refit':True,'no_threshold_change':True,
        'no_label_change':True,'goal_achieved':False}
    (out/'report.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# 冻结模型概率诊断','',
        '**概率诊断，不是经营质量×热点有效信号，也不是独立验证。**','',
        '使用全部4–9月冻结预测，阈值仍为>80%，同股60日历日冷却；不重训、不改标签或主信号门禁。每日概率与跨算法重复不能累计成独立证据。','',
        '|期限|算法/策略|诊断机会|股票/日期|TP/FP|未知|成熟precision|TP/全部|Wilson95%|同股基准|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for s in stats:
        lines.append(f"|{s['horizon']}日|{s['algorithm']}|{s['signals']}|{s['stock_count']}/{s['decision_dates']}|{s['tp']}/{s['fp']}|{s['unknown']}|{percent(s['precision'])}|{percent(s['conservative_lower_bound'])}|{'–'.join(percent(x) for x in s['wilson95'])}|{percent(s['same_stock_base_rate'])}|")
    lines+=['','固定概率区间的平均估计与真实达成率、股票/周簇区间、逐股样本及未知见report.json与per_stock_diagnostic.csv。原主策略结果仍独立保留；无论诊断数值如何，均不提升部署资格。','']
    (out/'report.md').write_text('\n'.join(lines))
    hashes={str(p.resolve()):sha(p) for p in out.iterdir() if p.is_file() and p.name!='lineage.json'}
    (out/'lineage.json').write_text(json.dumps({'registration':registration,'artifacts':hashes},indent=2))
    (HERE/'latest_probability_diagnostic.json').write_text(json.dumps({'path':str(out.resolve()),'key':key,'training_key':pointer['training_key']},indent=2))
    print(json.dumps({'path':str(out.resolve()),'summary':[{k:s[k] for k in ['horizon','algorithm','signals','tp','fp','unknown','precision']} for s in stats]},indent=2))

if __name__=='__main__':main()
