"""Post-run matched actual-entry distributions; never a selection or training input."""
import argparse, math
from collections import Counter
from class_learning import *


def diagnose(run):
    run=Path(run);result=read(run/'results.json');p=read(run/'registration.json')['preregistration']['protocol']
    primary=result['primary_name'];assert primary=='gap_blend'
    trades=pd.read_csv(run/(primary+'_trades.csv'));predictions=read(run/'predictions.json')
    summary={k:{'probabilities':[],'payoffs':[],'log_loss':[],'brier':[]} for k in ['gap_blend','gap_raw','gap_prior']}
    actual=[];logreturns=[];counts=Counter();rows=[]
    for t in trades.itertuples():
        assert np.isfinite(t.net_return)
        y=0 if t.reason=='target' else (2 if t.gap_below_stop else 1) if t.reason=='risk_stop' else 3
        value=math.log1p(t.net_return);at=pd.Timestamp(t.decision_at);j=core.ETFS.index(t.symbol)
        rec=read(run/'training'/(tag(at)+'.json'));means=np.array(rec['distributions']['gap']['payoff_means'][j])
        actual.append(y);logreturns.append(value);counts[y]+=1
        row={'at':at,'symbol':t.symbol,'actual_class':y,'actual_log_return':value}
        for k,s in summary.items():
            info=rec['estimates'][k];assert info['ready'];q=np.array(info['probabilities'][j]);pred=float(q@means)
            np.testing.assert_allclose(pred,info['values'][j],atol=1e-13,rtol=1e-11)
            s['probabilities'].append(q);s['payoffs'].append(q*means)
            s['log_loss'].append(-math.log(max(q[y],1e-8)));s['brier'].append(sum((q-np.eye(4)[y])**2))
            row[k+'_expected_log_return']=pred
        rows.append(row)
    n=len(trades);observed_frequency=[counts[k]/n for k in range(4)]
    observed_contributions=[sum(v for y,v in zip(actual,logreturns) if y==k)/n for k in range(4)]
    matched={}
    for k,s in summary.items():
        probabilities=np.mean(s['probabilities'],axis=0);payoffs=np.mean(s['payoffs'],axis=0)
        matched[k]={'mean_probabilities':probabilities,'mean_log_payoff_contributions':payoffs,
            'mean_expected_log_return':float(sum(payoffs)),'mean_log_loss':float(np.mean(s['log_loss'])),
            'mean_brier':float(np.mean(s['brier']))}
        # Recompute each mean with scalar arithmetic independently of matrix reductions.
        for c in range(4):
            np.testing.assert_allclose(probabilities[c],sum(float(q[c]) for q in s['probabilities'])/n,atol=1e-13)
            np.testing.assert_allclose(payoffs[c],sum(float(q[c]) for q in s['payoffs'])/n,atol=1e-13)
    diagnostic=read(run/'diagnostics.json')['actual_primary_entries']['means']
    np.testing.assert_allclose([matched[primary]['mean_expected_log_return'],sum(observed_contributions)],
        [diagnostic['predicted_log_return'],diagnostic['actual_log_return']],atol=1e-13)
    win=trades[trades.reason=='target'].net_return;loss=trades[trades.reason!='target'].net_return
    stats={'winning_entries':len(win),'mean_net_win':float(win.mean()),'losing_entries':len(loss),'mean_net_loss':float(loss.mean()),
        'observed_mix_arithmetic_break_even_win_share':float(-loss.mean()/(win.mean()-loss.mean())),
        'observed_mix_is_ex_post_not_a_trading_threshold':True}
    settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end']
    def selected(name):return [e for e in predictions[name] if start<=e['date'][:10]<=end]
    a,b=selected(primary),selected('merged_blend');assert len(a)==len(b)==result['coverage']['expected_ticks']
    differences=[]
    for x,y in zip(a,b):
        assert x['at']==y['at']
        if x['action']!=y['action']:differences.append({'at':x['at'],'gap':x['action'],'merged':y['action']})
    paired=pd.read_csv(run/'merged_blend_trades.csv')
    mainkeys=[(t.decision_at,t.symbol) for t in trades.itertuples()]
    otherkeys=[(t.decision_at,t.symbol) for t in paired.itertuples()]
    decisions=pd.read_csv(run/(primary+'_decisions.csv'))
    output={'stage':'post_run_descriptive_only','actual_entry_count':n,'class_order':['target','ordinary_stop','gap_stop','expiry'],
        'actual_class_counts':[counts[c] for c in range(4)],'actual_class_frequencies':observed_frequency,
        'actual_log_payoff_contributions':observed_contributions,'matched_same_primary_entry_forecasts':matched,
        'trade_payoff_balance':stats,'gap_vs_merged':{'query_action_differences':differences,
            'same_actual_entry_keys':mainkeys==otherkeys,'primary_entry_keys_only':sorted(set(mainkeys)-set(otherkeys)),
            'merged_entry_keys_only':sorted(set(otherkeys)-set(mainkeys))},
        'primary_requested_action_counts':dict(Counter(e['action'] for e in a)),
        'primary_actual_decision_counts':decisions.decision.value_counts().to_dict(),
        'executed_subset_selected_by_primary_not_independent_validation':True,
        'early_actual_exits_not_released_for_model_training':True,'changes_policy_or_gate':False,
        'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__))}
    pd.DataFrame(rows).to_csv(run/'matched_entry_forecasts.csv',index=False)
    dump(run/'entry_diagnostics.json',output);print(read(run/'entry_diagnostics.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);diagnose(p.parse_args().run)
