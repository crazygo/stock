"""Post-result diagnostics of frozen v44 calibration; never creates trading rules."""
from pathlib import Path
import sys,argparse,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v44'))
from conditional_calibration import np,pd,read,dump,sha,core,CLOCKS,MACRO,VARIANTS,tag


def design(X,model):
    X=np.atleast_2d(X);p=X.shape[1]-1;c=X[:,-1].astype(int)
    assert np.array_equal(c,X[:,-1]) and np.all((c>=0)&(c<4))
    Z=(X[:,:p]-np.asarray(model['input_center'])[c])/np.asarray(model['input_scale'])[c]
    A=np.zeros((len(X),4+4*p));A[np.arange(len(X)),c]=1
    for i,j in enumerate(c):A[i,4+j*p:4+(j+1)*p]=Z[i]
    return A,Z


def penalty(p):
    P=np.zeros((4+4*p,4+4*p))
    for j in range(p):
        avg=np.zeros(len(P));avg[4+j::p]=.25;P+=20*np.outer(avg,avg)
        for c in range(4):
            v=-avg.copy();v[4+c*p+j]+=1;P+=20*np.outer(v,v)
    return P


def geometry(X,w,model,n_dates):
    w=np.asarray(w,float);w=w*n_dates/w.sum();keep=w>0;X=X[keep];w=w[keep]
    A,_=design(X,model);G=A.T@(w[:,None]*A);K=G+penalty(X.shape[1]-1);Ki=np.linalg.solve(K,np.eye(len(K)))
    df=float(np.trace(Ki@G));train_leverage=np.einsum('ij,jk,ik->i',A,Ki,A)
    assert -1e-10<=df<=len(K)+1e-10
    np.testing.assert_allclose(df,w@train_leverage,atol=1e-10)
    return {'inverse':Ki,'gram':G,'df':df,'train_leverage_median':float(np.median(train_leverage)),
            'train_leverage_max':float(max(train_leverage)),'train_design':A,'weights':w}


def query_terms(x,model,g):
    a,z=design(x,model);a=a[0];z=z[0];c=int(x[-1]);b=np.asarray(model['coefficients']);common=b.mean(0)
    out={'intercept':np.asarray(model['intercepts'])[c]}
    for key,sl in [('raw',slice(0,2)),('macro',slice(2,None))]:
        out[key+'_shared']=z[sl]@common[sl];out[key+'_clock']=z[sl]@(b[c,sl]-common[sl])
    out['correction']=sum(out.values());out['leverage']=float(a@g['inverse']@a)
    # Response sensitivity assuming independent equal-variance standardized errors;
    # a diagnostic only, never a confidence interval for clustered market outcomes.
    out['response_sensitivity']=float(a@g['inverse']@g['gram']@g['inverse']@a)
    return out


def frozen_source(root):
    root=Path(root);manifest=read(core.ROOT/'research/soxl_soxs_14d5pct_v44/evidence/delivery_manifest.json')
    assert root.resolve()==Path(manifest['run'])
    for name,h in manifest['cache_outputs_sha256'].items():assert sha(root/name)==h,name
    for folder in ['daily','training']:
        for name,h in manifest[folder+'_sha256'].items():assert sha(root/folder/name)==h,(folder,name)
    return manifest


def diagnose(root,output):
    root=Path(root);output=Path(output);assert not output.exists(),'Preserve old diagnostic runs';manifest=frozen_source(root)
    panel=pd.read_parquet(root/'macro_panel.parquet');records={pd.Timestamp(r['at']):r for r in [read(p) for p in sorted((root/'training').glob('*.json'))]}
    histories={k:{t:r['raws'][k] for t,r in records.items()} for k in VARIANTS}
    settings=read(root/'registration.json')['preregistration']['protocol']['development'];ev=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])]
    cutoff=pd.Timestamp(read(root/'input_manifest.json')['market']['observation_cutoff_exclusive'])
    decisions={k:pd.read_csv(root/(k+'_band_decisions.csv')) for k in VARIANTS}
    for d in decisions.values():d.index=pd.to_datetime(d.decision_at,utc=True)
    rows=[];dayrows=[];checks=0
    for day,part in ev.groupby('date'):
        daily=read(root/'daily'/(str(day.date())+'.json'));assert daily['ready'];ids=list(pd.to_datetime(daily['members'],utc=True));n=len(daily['parents'])
        assert all(t<pd.Timestamp(daily['cutoff']) for t in ids)
        dw=np.asarray(daily['day_weights']);neff=float(dw.sum()**2/(dw@dw))
        for kind in VARIANTS:
            m=daily['variants'][kind]['point_model'];fields=MACRO if kind.endswith('context') else []
            X=np.array([[*histories[kind][t],*[panel.loc[t,f] for f in fields],CLOCKS.index(panel.loc[t,'clock'])] for t in ids]);g=geometry(X,daily['row_weights'],m,n)
            beta=np.vstack([np.array(m['intercepts']),np.array(m['coefficients']).reshape(-1,2)])
            Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
            np.testing.assert_allclose((g['gram']+penalty(len(fields)+2))@beta,g['train_design'].T@(g['weights'][:,None]*Y),atol=1e-10);checks+=1
            dayrows.append({'date':str(day.date()),'kind':kind,'n_dates':n,'weight_effective_dates':neff,'df':g['df'],'df_over_effective_dates':g['df']/neff})
            for at,row in part.iterrows():
                rec=records[at];raw=np.array(rec['raws'][kind]);est=rec['estimates'][kind];assert est['ready']
                x=np.r_[raw,[row[f] for f in fields],CLOCKS.index(row.clock)];terms=query_terms(x,m,g)
                np.testing.assert_allclose(raw+terms['correction'],est['point'],atol=1e-12);checks+=1
                j=int(est['point'][1]>est['point'][0]);event=rec['events'][kind+'_band'];proposed=event['action']!='CASH'
                if proposed:assert event['action']==core.ETFS[j]
                status=decisions[kind].loc[at,'decision'];actual=status.startswith('buy_');assert not actual or proposed
                known=pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D)
                truth=float(row.C+(row.D if j==0 else -row.D)) if known else None
                item={'at':str(at),'date':str(day.date()),'kind':kind,'symbol':core.ETFS[j],'proposed':proposed,'actual':actual,'status':status,'known':known,'raw':float(raw[j]),'point':est['point'][j],'radius':est['radius'],'truth':truth,
                    'overprediction':float(est['point'][j]-truth) if known else None,'df':g['df'],'weight_effective_dates':neff,'leverage':terms['leverage'],'response_sensitivity':terms['response_sensitivity'],
                    'leverage_over_train_median':terms['leverage']/g['train_leverage_median'],'outside_training_leverage':terms['leverage']>g['train_leverage_max']}
                for name in ['intercept','raw_shared','raw_clock','macro_shared','macro_clock','correction']:item[name]=float(terms[name][j])
                # Identical query/ETF for context-versus-plain correction comparison.
                peer=kind.replace('_context','_plain');item['matched_plain_point']=records[at]['estimates'][peer]['point'][j]
                item['context_minus_matched_plain']=item['point']-item['matched_plain_point']
                rows.append(item)
    frame=pd.DataFrame(rows);daily_frame=pd.DataFrame(dayrows);summary={};daily_summary={}
    fields=['raw','point','truth','overprediction','radius','intercept','raw_shared','raw_clock','macro_shared','macro_clock','correction','df','weight_effective_dates','leverage','response_sensitivity','leverage_over_train_median','outside_training_leverage','context_minus_matched_plain']
    for kind in VARIANTS:
        allrows=frame[frame.kind==kind];groups={'all':allrows,'proposed':allrows[allrows.proposed],'actual':allrows[allrows.actual],'unavailable_proposals':allrows[allrows.proposed&~allrows.actual]};summary[kind]={}
        for name,q in groups.items():
            known=q[q.known];d=known.groupby('date')[fields].mean();summary[kind][name]={'queries':len(q),'known':len(known),'dates':len(d),'unknown':q.loc[~q.known,'at'].tolist(),'date_equal_mean':d.mean().to_dict(),
                'daily_leverage_error_correlation':float(d.leverage.corr(d.overprediction)) if len(d)>2 else None,
                'median_leverage_ratio':float(known.leverage_over_train_median.median()),'max_leverage_ratio':float(known.leverage_over_train_median.max())}
        q=daily_frame[daily_frame.kind==kind];daily_summary[kind]={c:{'min':float(q[c].min()),'mean':float(q[c].mean()),'max':float(q[c].max())} for c in ['n_dates','weight_effective_dates','df','df_over_effective_dates']}
    output.mkdir(parents=True);frame.to_csv(output/'queries.csv',index=False);daily_frame.to_csv(output/'daily_geometry.csv',index=False)
    dump(output/'geometry.json',{'post_result_only':True,'new_actions_generated':False,'new_results_generated':False,'source_run':str(root.resolve()),'source_results_sha256':sha(root/'results.json'),
        'source_delivery_sha256':sha(core.ROOT/'research/soxl_soxs_14d5pct_v44/evidence/delivery_manifest.json'),'diagnostic_code_sha256':sha(Path(__file__)),'model_equation_and_prediction_checks':checks,
        'summary':summary,'daily':daily_summary,'caveats':['Fixed-design conditional smoother df excludes feature preprocessing and all base-model complexity.','Effective date count is weight concentration, not independent market sample size.','Attribution of correlated inputs is algebraic, not a causal effect.','Realized-error association is post-result and cannot be a same-day trading input.']})
    print(json.dumps({'daily':daily_summary,'summary':summary},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();diagnose(a.source,a.output)
