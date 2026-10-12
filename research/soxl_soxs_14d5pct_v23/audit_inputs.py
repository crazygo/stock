"""Reconstruct matched prefixes and 31 factors without production feature helpers."""
import numpy as np
import pandas as pd
from clock_training import core,PANEL,CLOCKS,BASE_COLUMNS,REGIME_COLUMNS,EXTRA
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
