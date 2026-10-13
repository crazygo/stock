"""Independent prior-only volatility, support validity and input-only amplitude audit."""
import argparse
from path_learning import *
from audit_paths import reference_kernel


def reference_support(bank,members,clock,logvol,plan):
    days=sorted(set(int(j) for i in members for j in bank['blocks'][i]));source=np.array([bank['states'][j,0,2] for j in days])
    out={'enabled':True,'ready':False,'current_log_volatility':float(logvol),'source_days':days}
    if not np.isfinite(logvol) or not np.isfinite(source).all():out['reason']='missing_or_nonfinite_prior_volatility';return out
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):ratios=np.exp(logvol-source)
    if not (np.isfinite(ratios).all() and np.all(ratios>0)):out['reason']='invalid_volatility_multiplier';return out
    out['multipliers']=ratios.tolist();extremes=[];c=CLOCKS.index(clock)
    for first in [True,False]:
        ids=np.array([bank['blocks'][i,0] for i in members]) if first else np.array(days)
        lo,hi=(plan['first_offset'],plan['bar_counts'][0]) if first else (0,78)
        anchor=bank['quotes'][ids,c] if first else bank['previous_close'][ids]
        dividend=np.zeros_like(anchor) if first else bank['dividend_ratio'][ids]
        price=bank['ohlc'][ids,lo:hi];vol=np.exp(logvol-bank['states'][ids,0,2]);unit=price/anchor[:,None,:,None]
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):value=np.power(unit+dividend[:,None,:,None],vol[:,None,None,None])-dividend[:,None,:,None]
        value=np.where((vol==1)[:,None,None,None],price*(1/anchor)[:,None,:,None],value)
        if not (np.isfinite(value).all() and np.all(value>0)):out['reason']='nonpositive_or_nonfinite_full_support_price';return out
        extremes.append((float(np.min(value)),float(np.max(value))))
    out.update(ready=True,minimum_price_ratio=min(x[0] for x in extremes),maximum_price_ratio=max(x[1] for x in extremes));return out


def audit(inputs):
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');panel=pd.read_parquet(inputs/'activity_panel.parquet');features=pd.read_parquet(inputs/'features.parquet');bank=dict(np.load(inputs/'path_bank.npz'))
    market=core.load_market();assert market.manifest==m['full_market'];market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    daily={};frame=market.bars['SOXX'];rawchecks=0
    for day,g in frame[frame.session_type.eq('regular')].groupby('date'):
        cal=market.calendar.get(day)
        if cal is None:continue
        expected=pd.date_range(cal['open_at'],cal['close_at'],freq='5min',inclusive='right')
        if not g.index.equals(expected) or g.available_at.isna().any() or not g.available_at.ge(g.end_at).all():continue
        assert np.isfinite(g.econ_close).all() and g.econ_close.gt(0).all()
        np.testing.assert_allclose(g.econ_close,g.close*g.share_factor,rtol=1e-12,atol=1e-12);rawchecks+=len(g)
        daily[day]=(float(g.econ_close.iloc[-1]),g.available_at.max())
    sigma={};date_rows=[]
    for day,group in panel.groupby('date'):
        prior=list(market.sessions[market.sessions<day][-21:]);assert len(prior)==21 and all(d in daily for d in prior)
        values=np.array([daily[d][0] for d in prior]);latest=max(daily[d][1] for d in prior);cutoff=day.tz_localize(core.ET).tz_convert('UTC');assert latest<cutoff
        returns=values[1:]/values[:-1]-1;mean=sum(returns)/20;value=max(float(np.sqrt(sum((returns-mean)**2)/19)),1e-4);sigma[day]=value
        np.testing.assert_allclose(group.log_sector_volatility,np.log(value),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(features.loc[group.index,'SOXX_vol20'],np.std(returns,ddof=1),rtol=1e-12,atol=1e-12)
        date_rows.append({'date':day,'prior_start':prior[0],'prior_end':prior[-1],'latest_prior_price_available_at':latest,'cutoff':cutoff,'sigma':value})
    finite_checks=0
    for i,ns in enumerate(bank['date_ns']):
        for c in range(len(CLOCKS)):
            value=bank['states'][i,c,2]
            if np.isfinite(value):np.testing.assert_allclose(value,np.log(sigma[pd.Timestamp(ns)]),rtol=1e-12,atol=1e-12);finite_checks+=1
    rows=[];settings=read(SOURCE/'protocol.json')[m['phase']];invalid=0;formulas=0
    for at,row in panel.iterrows():
        members=price_known_pool(bank,at,row.clock);plan=horizon(market.calendar,at)
        if len(members)<120 or plan is None or not row.feature_ready:continue
        a=volatility_support(bank,members,row.clock,float(row.log_sector_volatility),plan);b=reference_support(bank,members,row.clock,float(row.log_sector_volatility),plan)
        assert a.keys()==b.keys()
        for key,value in b.items():
            if key in ['enabled','ready','reason','source_days']:assert a[key]==value
            else:np.testing.assert_allclose(a[key],value,rtol=1e-12,atol=1e-12)
        formulas+=1;invalid+=int(not a['ready'])
        if not settings['evaluation_start']<=str(row.date.date())<=settings['evaluation_end']:continue
        ids=bank['blocks'][members,0];c=CLOCKS.index(row.clock);x=np.column_stack((bank['states'][ids,c],bank['absolute_states'][ids,c],bank['activity_states'][ids,c]));kernel=reference_kernel(x,row[FIELDS+[ABSOLUTE,ACTIVITY]].to_numpy(float),True)
        ratio=np.exp(row.log_sector_volatility-bank['states'][ids,0,2]);w=kernel['probability'];dist=np.log(ratio)
        rows.append({'at':at,'date':row.date,'ready':a['ready'],'source_days':len(a['source_days']),'multiplier_min':float(ratio.min()),'multiplier_max':float(ratio.max()),'weighted_multiplier':float(w@ratio),'weighted_abs_log_multiplier':float(w@abs(dist)),'weighted_mass_outside_0_8_to_1_25':float(w@((ratio<.8)|(ratio>1.25))),'minimum_price_ratio':a.get('minimum_price_ratio'),'maximum_price_ratio':a.get('maximum_price_ratio')})
    f=pd.DataFrame(rows);numeric=['weighted_multiplier','weighted_abs_log_multiplier','weighted_mass_outside_0_8_to_1_25'];out={'passed':True,'phase':m['phase'],'prior_volatility_dates':len(date_rows),'raw_economic_bar_checks':rawchecks,'bank_logvol_checks':finite_checks,'support_formula_checks':formulas,'unavailable_support_queries':invalid,'evaluation_queries':len(rows),'evaluation_dates':int(f.date.nunique()),'evaluation_all_support_valid':bool(f.ready.all()),'date_equal_averages':f.groupby('date')[numeric].mean().mean().to_dict(),'multiplier_range':[float(f.multiplier_min.min()),float(f.multiplier_max.max())],'daily_volatility':date_rows,'rows':rows,'capital_outcomes_read_by_this_audit':False,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'source_sha256':sha(Path(__file__))}
    dump(inputs/'volatility_audit.json',out);print({k:v for k,v in out.items() if k not in ['daily_volatility','rows']},flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
