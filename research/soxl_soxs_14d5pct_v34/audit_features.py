"""Independent pairwise dot products and raw-close participation arithmetic."""
from coherence_features import core,np,pd,PANEL,COHERENCE
from verify_continuation import near


def reference_values(prices,current):
    R=np.diff(prices,axis=0)/prices[:-1];corr=[]
    for n in [20,5]:
        values=[]
        for j in range(8):
            for k in range(j+1,8):
                a=R[-n:,j]-R[-n:,j].mean();b=R[-n:,k]-R[-n:,k].mean()
                if (a@a)/n<1e-24 or (b@b)/n<1e-24:return None
                values.append(float((a@b)/np.sqrt((a@a)*(b@b))))
        corr.append(sum(values)/len(values))
    scales=[]
    for j in range(8):scales.append(max(float(np.sqrt(sum((R[:,j]-R[:,j].mean())**2)/19)),1e-4))
    past=np.clip((prices[-1]/prices[-6]-1)/(np.array(scales)*np.sqrt(5)),-5,5)
    now=np.clip(np.asarray(current)/scales,-5,5)
    def part(z):
        avg=sum(z)/8;energy=sum(t*t for t in z)/8
        return float(np.sign(avg)*avg*avg/energy) if energy>0 else 0.
    order=sorted(range(8),key=lambda j:(past[j],j));lag=sum(now[j] for j in order[:4])/4;lead=sum(now[j] for j in order[4:])/4
    return dict(zip(COHERENCE,[part(past),part(now),(lag-lead)/2,corr[0],corr[1]-corr[0],corr[0]*part(now)]))


def audit_features(features,new,receipts,leaders,sessions):
    bars,daily,_=leaders;bytime={pd.Timestamp(r['at']):r for r in receipts};checks=0;ready=0
    assert new.index.equals(features.index) and set(bytime)==set(features.index)
    for at,row in features.iterrows():
        dates=list(sessions[sessions<row.date][-21:]);saved=bytime[at]
        assert list(pd.to_datetime(saved['history_dates']))==dates
        prices=[];current=[];available=True
        for j,s in enumerate(PANEL):
            f=daily[s].reindex(dates)
            if len(dates)!=21 or not all(pd.notna(f.loc[d,'complete']) and bool(f.loc[d,'complete']) and pd.notna(f.loc[d,'close']) and f.loc[d,'close']>0 and pd.notna(f.loc[d,'available_at']) and f.loc[d,'available_at']<at for d in dates):
                available=False;break
            seen=bars[s][(bars[s].end_at<=at)&(bars[s].available_at<=at)&bars[s].date.eq(row.date)]
            if seen.empty or (at-seen.index[-1]).total_seconds()>360 or not np.isfinite(seen.econ_close.iloc[-1]) or seen.econ_close.iloc[-1]<=0:
                available=False;break
            p=f.close.to_numpy(float);prices.append(p);current.append(seen.econ_close.iloc[-1]/p[-1]-1)
            r=saved['stocks'][j];assert r['symbol']==s and pd.Timestamp(r['last_bar_end'])==seen.index[-1] and pd.Timestamp(r['last_available'])==seen.available_at.iloc[-1]
            assert pd.Timestamp(r['history_latest_available'])==f.available_at.max() and pd.Timestamp(r['history_last_date'])==dates[-1]
            near(current[-1],row['leader_'+s+'_today']);near(p[-1]/p[-6]-1,row['leader_'+s+'_ret5']);checks+=2
        vals=reference_values(np.array(prices).T,current) if available else None
        assert bool(new.loc[at,'coherence_ready'])==(vals is not None)==saved['ready']
        if vals is None:assert new.loc[at,COHERENCE].isna().all()
        else:
            assert saved['reason'] is None;ready+=1
            for c,v in vals.items():near(new.loc[at,c],v);checks+=1
    return {'passed':True,'feature_ticks':len(features),'ready_ticks':ready,'independent_feature_equalities':checks,
        'past21_sessions_and_current_available_bars_checked':True,'pairwise_dot_products_not_production_corrcoef':True}
