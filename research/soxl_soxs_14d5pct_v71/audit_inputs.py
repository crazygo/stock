"""Independent price/volume formulas, raw auxiliary contracts, and maturity."""
import argparse
import math
from transfer_learning import *


def reference_features(market, prefixes, pair, at):
    long, short, bench = PAIRS[pair]; day = at.tz_convert(core.ET).tz_localize(None).normalize()
    clock = at.tz_convert(core.ET).strftime('%H:%M:%S')
    days = [d for d in market.sessions if d < day][-21:]
    olddays = [d for d in market.sessions if d < day and pd.Timestamp(str(d.date())+' '+clock,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])][-20:]
    if len(days)!=21 or len(olddays)!=20:return None
    oldticks = [pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC') for d in olddays]
    st = {}; aa = {}
    for s in set([long,short,bench,'SPY']):
        df=market.daily[s].reindex(days);p=prefixes[s].reindex(oldticks+[at])
        if not df.complete.fillna(False).all() or not df.available_at.lt(day_cutoff(at)).all() or not p.prefix_ready.fillna(False).all():return None
        v=df.close.to_numpy(float)
        if not np.isfinite(v).all() or min(v)<=0:return None
        r=np.array([v[i]/v[i-1]-1 for i in range(1,21)]);sd=max(math.sqrt(sum((r-r.mean())**2)/19),.0001)
        st[s]=[v[-1]/v[-2]-1,v[-1]/v[-6]-1,v[-1]/v[0]-1,sd,
               math.sqrt(sum((r[-5:]-r[-5:].mean())**2)/4),p.quote.iloc[-1]/v[-1]-1,
               p.quote.iloc[-1]/p.quote15.iloc[-1]-1,
               (v[-1]-v[0])/sum(abs(np.diff(v))) if sum(abs(np.diff(v))) else 0.]
        aa[s]=max(-3,min(3,math.log(p.amount.iloc[-1]/(sum(p.amount.iloc[:20])/20))))
    b=st[bench];m=st['SPY'];clip=lambda x:max(-8,min(8,float(x)))
    t=clip(b[1]/b[3]/math.sqrt(5));q=clip(b[5]/b[3]);direction=(aa[long]-aa[short])/2;rel=(aa[bench]-aa['SPY'])/2
    return [clip(b[0]/b[3]),t,clip(b[2]/b[3]/math.sqrt(20)),math.log(b[3]),min(b[4]/b[3],8),
            clip((b[1]-m[1])/b[3]/math.sqrt(5)),q,clip(b[6]/b[3]),clip(m[5]/m[3]),
            clip(m[1]/m[3]/math.sqrt(5)),b[7],(aa[long]+aa[short])/2,direction,rel,
            clip(direction*q),clip(rel*q),clip(t*q),clip((st[long][5]+st[short][5])/(st[long][3]+st[short][3]))]


def raw_contract(market, symbol, at):
    """Scalar raw OHLC loop; no use of label engine or economic price columns."""
    day=at.tz_convert(core.ET).tz_localize(None).normalize();f=market.bars[symbol]
    entry=max((at+pd.Timedelta(seconds=30)).floor('5min')+pd.Timedelta(minutes=5),pd.Timestamp(market.calendar[day]['open_at']))
    end=entry+pd.Timedelta(minutes=5)
    if end not in f.index:return {'reason':'no_entry_bar','release':None}
    a=market.actions[symbol];factor=lambda d:math.prod(float(x['value']) for x in a if x['type']=='Splits' and pd.Timestamp(x['date'])<=d)
    fill=float(f.loc[end,'open'])*1.0005;scale=factor(day);basis=fill*scale
    target=math.ceil(fill*1.05*100-1e-10)/100*scale
    expiry_day=max(d for d in market.sessions if d<=day+pd.Timedelta(days=13));expiry=pd.Timestamp(market.calendar[expiry_day]['close_at'])
    expected=[]
    for d in market.sessions[(market.sessions>=day)&(market.sessions<=expiry_day)]:
        expected.extend(pd.date_range(max(entry,pd.Timestamp(market.calendar[d]['open_at'])),market.calendar[d]['close_at'],freq='5min',inclusive='right'))
    path=f.reindex(expected);complete=path.available_at.notna().all()
    release=path.available_at.max() if complete else None
    event=None
    for bar in path.itertuples():
        if pd.isna(bar.open):break
        sf=factor(bar.date);target_raw=math.ceil(target/sf*100-1e-10)/100;stop_raw=math.floor(basis*.94/sf*100+1e-10)/100
        if bar.low<=stop_raw:event=('risk_stop',bar.end_at,min(bar.open,stop_raw)*.9995*sf);break
        if bar.high>=target_raw:event=('target',bar.end_at,target_raw*sf);break
    if event is None and complete:event=('expiry',expiry,float(path.close.iloc[-1])*.9995*factor(expiry_day))
    if event is None:return {'reason':'pending','release':None,'complete_window':bool(complete)}
    reason,exit_at,value=event;last=exit_at.tz_convert(core.ET).tz_localize(None).normalize()
    cash=sum(float(x['value'])*factor(pd.Timestamp(x['date'])) for x in a if x['type']=='CashDividends' and day<pd.Timestamp(x['date'])<=last)
    return {'reason':reason,'entry_at':entry,'entry_fill_raw':fill,'exit_at':exit_at,
            'exit_fill_economic':value,'net_return':(value*.9995+cash)/(basis*1.0005)-1,
            'dividend_per_economic_unit':cash,'complete_window':bool(complete),'release':release}


def audit(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json')
    assert manifest['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    market=load_market();assert market.manifest==manifest['market_manifest']
    prefixes={s:pd.read_parquet(inputs/(s+'_prefix.parquet')) for s in sorted({s for p in PAIRS.values() for s in p})}
    prefix_count=0
    for s,table in prefixes.items():
        f=market.bars[s]
        for at,row in table.iterrows():
            start=pd.Timestamp(str(row.date.date())+' 04:00',tz=core.ET).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
            ix=pd.date_range(start+pd.Timedelta(minutes=5),end,freq='5min');g=f.reindex(ix)
            good=(len(g)>=4 and g.available_at.notna().all() and g.available_at.le(at).all() and
                  g.available_at.ge(g.end_at).all() and g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all())
            v=g[['open','high','low','close','volume']].to_numpy(float)
            good=bool(good and np.isfinite(v).all() and (v[:,:4]>0).all() and (v[:,4]>=0).all())
            amount=np.sum((v[:,1]+v[:,2]+v[:,3])*v[:,4]/3) if good else np.nan;good=bool(good and amount>0)
            assert row.prefix_ready==good,(s,at)
            if good:
                np.testing.assert_allclose([row.amount,row.quote,row.quote15],[amount,g.econ_close.iloc[-1],g.econ_close.iloc[-4]],rtol=1e-12,atol=1e-9)
                assert row.available_at==g.available_at.max()
            prefix_count+=1
    lab=labels(inputs/'all_labels.json');feature_checks=contract_checks=release_checks=0;outcomes={}
    for pair,(long,short,_) in PAIRS.items():
        p=pd.read_parquet(inputs/(pair+'_panel.parquet'))
        for i,(at,row) in enumerate(p.iterrows()):
            want=reference_features(market,prefixes,pair,at)
            assert row.feature_ready==(want is not None)
            if want is not None:
                np.testing.assert_allclose(row[FIELDS].to_numpy(float),want,rtol=1e-11,atol=1e-11);feature_checks+=len(FIELDS)
            pairlabels=[lab[at.isoformat()+'|'+s] for s in [long,short]]
            for l in pairlabels:
                if pair=='semiconductor':continue
                ref=raw_contract(market,l['symbol'],at)
                assert l['reason']==ref['reason'] and l['complete_window']==ref['complete_window'],(l['symbol'],at)
                assert l['label_available_at']==ref['release']
                for k in ['entry_fill_raw','exit_fill_economic','net_return','dividend_per_economic_unit']:
                    if k in ref:np.testing.assert_allclose(l[k],ref[k],rtol=1e-11,atol=1e-11)
                for k in ['entry_at','exit_at']:
                    if k in ref:assert l[k]==ref[k]
                contract_checks+=1;outcomes[ref['reason']]=outcomes.get(ref['reason'],0)+1
            known=all(l['complete_window'] and l.get('label_available_at') is not None and l.get('net_return') is not None and np.isfinite(l['net_return']) and l['net_return']>-1 for l in pairlabels)
            if known:
                y=np.array([math.log1p(l['net_return']) for l in pairlabels]);np.testing.assert_allclose([row.C,row.D],[sum(y)/2,(y[0]-y[1])/2],rtol=1e-12,atol=1e-12)
                assert row.terminal_release==max(l['label_available_at'] for l in pairlabels)
                assert row.terminal_release>=max(l['expiry_at'] for l in pairlabels)+pd.Timedelta(seconds=1)
            else:assert pd.isna(row.terminal_release) and pd.isna(row.C) and pd.isna(row.D)
            release_checks+=1
            if (i+1)%1000==0:print('Input audit',pair,i+1,flush=True)
    result={'passed':True,'prefixes':prefix_count,'feature_values':feature_checks,'auxiliary_contracts':contract_checks,
            'paired_release_checks':release_checks,'auxiliary_outcomes':outcomes,'input_manifest_sha256':sha(inputs/'input_manifest.json'),
            'source_sha256':sha(Path(__file__)),'strategy_outcomes_evaluated':False}
    dump(inputs/'input_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();audit(a.inputs)
