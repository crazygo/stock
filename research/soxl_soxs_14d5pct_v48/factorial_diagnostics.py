"""Post-result matched factorial contrasts; no action/model selection changes."""
import argparse
import gzip
import pickle
from collections import Counter
from conditional_options import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    d=read(run/'diagnostics.json')
    equity={k:pd.read_csv(run/(route+'_equity.csv'),parse_dates=['date']).set_index('date').equity
            for k,route in [('tree','tree_band'),('options','options_band'),
                            ('old_tree','old_tree_band'),('baseline','baseline_band')]}
    dates=equity['tree'].index
    assert all(x.index.equals(dates) for x in equity.values())
    returns={k:np.diff(np.log(np.r_[1.,x.to_numpy()])) for k,x in equity.items()}
    contrasts={
        'option_input_effect_on_tree':returns['tree']-returns['old_tree'],
        'option_input_effect_on_linear':returns['options']-returns['baseline'],
        'algorithm_effect_with_options':returns['tree']-returns['options'],
        'algorithm_effect_without_options':returns['old_tree']-returns['baseline'],
        'difference_of_differences':returns['tree']-returns['options']-returns['old_tree']+returns['baseline'],
    }
    rng=np.random.default_rng(826);n=len(dates)
    draws=rng.integers(0,n,size=(5000,(n+9)//10))
    samples=(draws[:,:,None]+np.arange(10)[None,None,:]).reshape(5000,-1)[:,:n]%n
    capital={k:{'mean_daily_log':float(x.mean()),
                'block_interval95':np.quantile(x[samples].mean(axis=1),[.025,.975]).tolist(),
                'sum_log':float(x.sum())} for k,x in contrasts.items()}
    comparisons={}
    for name,a,b in [('options_on_tree','tree','old_tree'),('options_on_linear','options','baseline'),
                      ('tree_with_options','tree','options'),('tree_without_options','old_tree','baseline')]:
        aa=d['metrics'][a]['date_equal'];bb=d['metrics'][b]['date_equal']
        comparisons[name]={f:aa[f]-bb[f] for f in ['raw_mse','raw_D_mse','weighted_brier','calibrated_mse']}
    protocol=read(run/'registration.json')['preregistration']['protocol']
    settings=protocol[r['phase']]
    panel=pd.read_parquet(run/'option_panel.parquet')
    panel=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])]
    paths={}
    for kind in ['tree','delayed','old_tree']:
        query_count=tree_count=option_paths=mixed_paths=0;features=Counter()
        for at,row in panel.iterrows():
            path=run/'models'/(kind+'_'+tag(at)+'.pkl.gz')
            if not path.exists() or not row.feature_ready:continue
            model=pickle.load(gzip.open(path,'rb'))
            if 'q' not in model['heads']:continue
            cols=model['q_columns']
            x=(row[cols].to_numpy(float)-model['q_scaler'].mean_)/model['q_scaler'].scale_
            query_count+=1
            for step in model['heads']['q']._predictors:
                nodes=step[0].nodes;j=0;groups=set()
                while not nodes[j]['is_leaf']:
                    node=nodes[j];name=cols[int(node['feature_idx'])]
                    features[name]+=1;groups.add('option' if name.startswith('option_lag') else 'original')
                    j=int(node['left'] if x[int(node['feature_idx'])]<=node['num_threshold'] else node['right'])
                tree_count+=1
                option_paths+=int('option' in groups);mixed_paths+=int(len(groups)==2)
        paths[kind]={'queries':query_count,'evaluated_tree_paths':tree_count,
                     'paths_using_options':option_paths,'paths_combining_option_and_original_inputs':mixed_paths,
                     'feature_split_visits':dict(features.most_common()),
                     'not_independent_samples_or_predictive_importance':True}
    out={'post_result_only':True,'no_selection_or_rule_changes':True,'dates':n,'block_sessions':10,
         'bootstrap_replicates':5000,'seed':826,'capital_log_contrasts':capital,
         'date_equal_prediction_loss_differences':comparisons,
         'actual_prediction_path_input_use':paths,
         'results_sha256':sha(run/'results.json'),'diagnostics_sha256':sha(run/'diagnostics.json'),
         'interpretation':'Descriptive matched recipe contrasts in exposed development history. Different executable trades and overlapping labels remain dependent. Not an independent causal or future-profit guarantee.'}
    dump(run/'factorial_diagnostics.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
