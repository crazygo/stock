"""Reconstruct matched prefixes and 31 factors without production feature helpers."""
import numpy as np
import pandas as pd
from clock_training import PANEL,BASE_COLUMNS,REGIME_COLUMNS,EXTRA
from dense_learning import core,CLOCKS
from verify_continuation import near


def input_check(run,market,bars,features,panel,cutoff):
    prefix_checks=factor_checks=0
    for clock in CLOCKS:
        v=pd.read_parquet(run/('volumes_'+clock[:5].replace(':','')+'.parquet'))
        expected_days=[]
        for d in market.sessions:
            at=pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC')
            if d.tz_localize(core.ET)<cutoff and at<pd.Timestamp(market.calendar[d]['close_at']):expected_days.append(d)
        assert list(v.index)==expected_days
        for symbol in PANEL:
            groups={d:g for d,g in bars[symbol].groupby('date')}
            for day in v.index:
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC')
                start=(day.tz_localize(core.ET)+pd.Timedelta(hours=4)).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
                g=groups.get(day,bars[symbol].iloc[:0]);g=g[(g.end_at>start)&(g.end_at<=end)&(g.end_at<cutoff)&(g.available_at<cutoff)]
                want=list(pd.date_range(start,end,freq='5min',inclusive='left').tz_convert('UTC'))
                good=(list(g.start_at)==want and list(g.end_at)==[t+pd.Timedelta(minutes=5) for t in want] and
                      g.available_at.notna().all() and g.available_at.ge(g.end_at).all() and g.available_at.le(at).all() and
                      g.econ_volume.ge(0).all() and g.econ_volume.sum()>0)
                if good:near(v.loc[day,symbol],sum(g.econ_volume))
                else:assert pd.isna(v.loc[day,symbol])
                prefix_checks+=1
        f=features[features.clock==clock];breadth=2*f.breadth_ret5_positive_fraction.to_numpy()-1
        for i,(at,r) in enumerate(f.iterrows()):
            sv=max(r.SOXX_vol20,.0001);qv=max(r.QQQ_vol20,.0001)
            vals=[*np.clip([r.SOXX_ret1/sv,r.SOXX_ret5/sv/np.sqrt(5),r.SOXX_ret20/sv/np.sqrt(20),
                  r.QQQ_ret5/qv/np.sqrt(5),r.SOXX_today_return/sv,r.breadth_today_mean/sv],-5,5),np.log(sv)]
            h=v[v.index<r.date].iloc[-20:]
            ready=i>=5 and len(h)==20 and h.notna().all().all() and v.loc[r.date].notna().all()
            if clock!=CLOCKS[0]:ready=ready and r.regular_context_ready==1 and r.breadth_regular_ready==1
            assert bool(panel.loc[at,'feature_ready'])==bool(ready)
            if ready:
                ratio=v.loc[r.date].values/h.mean().values;log=np.clip(np.log(ratio),-3,3)
                today=np.array([r['leader_'+s+'_today'] for s in PANEL]);trend=np.array([r['leader_'+s+'_ret5'] for s in PANEL])
                relative=np.clip(r.sector_relative_5/sv/np.sqrt(5),-5,5);vm=log.mean()
                extra=[relative,np.clip(r.sector_relative_20/sv/np.sqrt(20),-5,5),breadth[i],
                    2*r.breadth_ret20_positive_fraction-1,2*r.breadth_today_positive_fraction-1,
                    np.sign(f.sector_relative_5.iloc[i-4:i+1]).mean(),breadth[i]-breadth[i-5:i].mean(),vm,
                    np.mean((ratio>1)&(today>0))-np.mean((ratio>1)&(today<0)),np.mean(np.sign(trend)*log),relative*vm,breadth[i]*vm]
            else:extra=[np.nan]*12
            np.testing.assert_allclose(panel.loc[at,BASE_COLUMNS+REGIME_COLUMNS].to_numpy(float),vals+extra,atol=1e-11,rtol=1e-11,equal_nan=True)
            intraday=[r.x_session_return/sv,r.x_vwap_distance/sv,r.x_recent15/sv,r.x_previous15/sv,
                r.x_range_high_distance/sv,r.x_range_low_distance/sv,r.q_session_return/qv,r.q_recent15/qv,
                r.breadth_session_return_mean/sv,r.breadth_vwap_distance_mean/sv,2*r.breadth_recent15_positive_fraction-1,r.x_efficiency]
            np.testing.assert_allclose(panel.loc[at,EXTRA].to_numpy(float),np.clip(intraday,-5,5),atol=1e-12)
            factor_checks+=1
    return {'matched_volume_prefixes':prefix_checks,'factor_rows':factor_checks,'half_days_follow_official_close':True}


def raw_price_check(market,leader_bars,features):
    """Rebuild all price/volume inputs used by the new-clock base models."""
    sources={**{s:f for s,f in market.bars.items() if s in ['SOXX','QQQ']},**leader_bars}
    groups={s:{d:g for d,g in f.groupby('date')} for s,f in sources.items()};daily={}
    for s,days in groups.items():
        d={}
        for day,f in days.items():
            op=pd.Timestamp(market.calendar[day]['open_at']);cl=pd.Timestamp(market.calendar[day]['close_at'])
            reg=f[(f.start_at>=op)&(f.end_at<=cl)];expected=list(pd.date_range(op,cl,freq='5min',inclusive='right'))
            if list(reg.end_at)==expected and reg.available_at.ge(reg.end_at).all():
                d[day]={'close':float(reg.econ_close.iloc[-1]),'volume':float(reg.econ_volume.sum()),'available':reg.available_at.max()}
        daily[s]=d
    checks=0;queries=0
    for at,row in features.iterrows():
        if row.clock in ['09:25:30','10:05:30','10:35:30','13:05:30']:continue
        day=row.date;past=market.sessions[market.sessions<day][-21:];want={};aggregate={}
        for s in sources:
            history=[daily[s][d] for d in past];assert len(history)==21 and all(x['available']<=at for x in history)
            h=np.array([x['close'] for x in history]);f=groups[s][day];seen=f[(f.end_at<=at)&(f.available_at<=at)]
            assert len(seen) and (at-seen.end_at.iloc[-1]).total_seconds()<=360
            reg=seen[seen.start_at>=pd.Timestamp(market.calendar[day]['open_at'])];starts=list(pd.date_range(market.calendar[day]['open_at'],reg.end_at.iloc[-1],freq='5min',inclusive='left'))
            assert list(reg.start_at)==starts and len(reg)>=6 and reg.available_at.ge(reg.end_at).all()
            c=reg.econ_close.to_numpy();o=reg.econ_open.iloc[0];last=seen.econ_close.iloc[-1]
            vwap=np.sum(((reg.econ_high+reg.econ_low+reg.econ_close)/3)*reg.econ_volume)/sum(reg.econ_volume)
            fields={f'ret{n}':h[-1]/h[-1-n]-1 for n in [1,5,20]};fields['vol20']=np.std(h[1:]/h[:-1]-1,ddof=1)
            fields.update(today=last/h[-1]-1,session_return=last/o-1,vwap_distance=last/vwap-1,recent15=last/reg.econ_open.iloc[-3]-1,
                efficiency=(last-o)/sum(abs(np.diff(np.r_[o,c]))) if sum(abs(np.diff(np.r_[o,c]))) else 0.)
            if s in PANEL:
                fields['relative_volume']=sum(seen.econ_volume)/np.mean([x['volume'] for x in history[-20:]])
                for key,value in fields.items():want['leader_'+s+'_'+key]=value;aggregate.setdefault(key,[]).append(value)
            else:
                for key in ['ret1','ret5','ret20','vol20']:want[s+'_'+key]=fields[key]
                want[s+'_today_return']=fields['today'];prefix='x_' if s=='SOXX' else 'q_'
                for key in ['session_return','vwap_distance','recent15','efficiency']:want[prefix+key]=fields[key]
                want[prefix+'previous15']=c[-4]/reg.econ_open.iloc[-6]-1
                want[prefix+'range_high_distance']=last/max(reg.econ_high.iloc[:6])-1
                want[prefix+'range_low_distance']=last/min(reg.econ_low.iloc[:6])-1
        for key,values in aggregate.items():
            a=np.array(values);want['breadth_'+key+'_mean']=a.mean();want['breadth_'+key+'_std']=a.std(ddof=0);want['breadth_'+key+'_positive_fraction']=np.mean(a>0)
        for n in [5,20]:want['sector_relative_'+str(n)]=want['SOXX_ret'+str(n)]-want['QQQ_ret'+str(n)]
        for key,value in want.items():np.testing.assert_allclose(row[key],value,atol=1e-12,rtol=1e-10,err_msg=str(at)+':'+key);checks+=1
        queries+=1
    return {'new_clock_queries':queries,'raw_fields':checks,'future_bars_excluded':True}
