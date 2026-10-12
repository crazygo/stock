"""Independent vector barrier events compete against scheduled review exits."""
from overnight import *
from verify_continuation import near


def reference_forecasts(market,features):
    allrows={};forecasts={};checks=0
    for symbol in core.ETFS:
        frame=market.bars[symbol];rows=[]
        for i in range(1,len(market.sessions)):
            day=market.sessions[i];previous=market.sessions[i-1]
            mark_time=pd.Timestamp(market.calendar[previous]['close_at'])-pd.Timedelta(minutes=15)
            open_time=pd.Timestamp(market.calendar[day]['open_at']);end=open_time+pd.Timedelta(minutes=5)
            if end>frame.index.max():break
            ready=False;release=None;loss=None
            a=frame[frame.end_at==mark_time];b=frame[frame.start_at==open_time]
            if len(a)==1 and len(b)==1:
                a=a.iloc[0];b=b.iloc[0]
                ready=bool(a.session_type==b.session_type=='regular' and a.start_at==mark_time-pd.Timedelta(minutes=5) and b.end_at==end
                    and pd.notna(a.available_at) and pd.notna(b.available_at) and a.available_at>=mark_time and b.available_at>=end
                    and np.isfinite([a.close,a.share_factor,b.open,b.share_factor]).all() and min(a.close,a.share_factor,b.open,b.share_factor)>0)
                if ready:release=max(a.available_at,b.available_at);loss=max(0.,1-b.open*b.share_factor/(a.close*a.share_factor));checks+=1
            rows.append({'from_date':previous,'date':day,'mark_end':mark_time,'open_at':open_time,'open_bar_end':end,'ready':ready,'release':release,'loss':loss})
        allrows[symbol]=rows
        for day in market.sessions[(market.sessions>=features.date.min())&(market.sessions<=features.date.max())]:
            at=pd.Timestamp(market.calendar[day]['close_at'])-pd.Timedelta(minutes=14,seconds=30)
            past=[r for r in rows if r['date']<=day][-126:]
            valid=len(past)>=60 and past[-1]['date']==day and all(r['ready'] and r['release']<at for r in past)
            size=int(np.ceil(len(past)/10)) if valid else None;tail=None
            if valid:
                values=np.array([r['loss'] for r in past]);tail=float(np.partition(values,len(values)-size)[-size:].mean())
            forecasts[(day,symbol)]={'at':at,'date':day,'members':[r['date'] for r in past],'ready':bool(valid),'tail_count':size,
                'adverse_tail_mean':tail,'latest_release':max((r['release'] for r in past if r['release'] is not None),default=None)}
    return allrows,forecasts,checks


def reference_label(market,base,forecasts,mode,multiple):
    if 'entry_fill_economic' not in base:return None,[]
    expected=[]
    for day in market.sessions[(market.sessions>=base['entry_date'])&(market.sessions<=base['expiry_date'])]:
        op=pd.Timestamp(market.calendar[day]['open_at']);cl=pd.Timestamp(market.calendar[day]['close_at'])
        expected.extend(t for t in pd.date_range(op+pd.Timedelta(minutes=5),cl,freq='5min') if t-pd.Timedelta(minutes=5)>=base['entry_at'])
    grid=pd.DatetimeIndex(expected);full=market.bars[base['symbol']].reindex(grid)
    valid=full.end_at.eq(grid)&full.start_at.eq(grid-pd.Timedelta(minutes=5))&full.session_type.eq('regular')
    valid &= full.available_at.notna()&full.available_at.ge(full.end_at)
    for col in ['open','high','low','close','share_factor']:valid &= np.isfinite(full[col])&full[col].gt(0)
    missing=np.flatnonzero(~valid.to_numpy());n=int(missing[0]) if len(missing) else len(grid);f=full.iloc[:n]
    if not n:return None,[]
    factors=f.share_factor.to_numpy();stop=np.floor(base['entry_fill_economic']*.94/factors*100+1e-10)/100
    target=np.ceil(base['limit_economic']/factors*100-1e-10)/100;fee=slip=.0005*multiple
    stopped=f.low.to_numpy()<=stop;hit=f.high.to_numpy()>=target;indices=np.flatnonzero(stopped|hit)
    fixed=None
    if len(indices):
        j=int(indices[0]);r=f.iloc[j];reason='risk_stop' if stopped[j] else 'target'
        price=min(r.open,stop[j])*factors[j]*(1-slip) if stopped[j] else target[j]*factors[j]
        fixed={'at':r.end_at,'start':r.start_at,'observation_end':r.end_at,'fill':price,'reason':reason,'index':j,
            'gap':bool(stopped[j] and r.open<stop[j]),'both':bool(stopped[j] and hit[j])}
    elif f.end_at.iloc[-1]==base['expiry_at']:
        r=f.iloc[-1];fixed={'at':r.end_at,'start':r.start_at,'observation_end':r.end_at,'fill':r.close*r.share_factor*(1-slip),
            'reason':'expiry','index':n-1,'gap':False,'both':False}
    reviews=[];intent=None;chosen=fixed
    for j,r in enumerate(f.itertuples()):
        mark_time=pd.Timestamp(market.calendar[r.date]['close_at'])-pd.Timedelta(minutes=15);at=mark_time+pd.Timedelta(seconds=30)
        if r.end_at!=mark_time or r.end_at==base['expiry_at'] or (fixed is not None and fixed['at']<=at):continue
        forecast=forecasts[(r.date,base['symbol'])];mark_ready=r.available_at<=at
        margin=None
        if mode=='flat':action='EXIT';reason='scheduled_preclose'
        elif not mark_ready or not forecast['ready']:action='EXIT';reason='risk_data_unavailable'
        else:
            liquidation=r.close*r.share_factor*(1-forecast['adverse_tail_mean'])*(1-slip)*(1-fee)
            margin=liquidation/(base['entry_fill_economic']*(1+fee))-1
            action='HOLD' if margin>=0 else 'EXIT';reason='profit_buffer_covers_tail' if margin>=0 else 'insufficient_profit_buffer'
        reviews.append({'entry_at':base['entry_at'],'decision_at':base['decision_at'],'symbol':base['symbol'],'at':at,
            'mark_end':r.end_at,'mark_available_at':r.available_at,'mark_economic':r.close*r.share_factor if mark_ready else None,
            'mode':mode,'cost_multiple':multiple,'forecast':forecast,'action':action,'reason':reason,'buffer_after_tail':margin})
        if action=='EXIT':
            intent=at;starts=grid-pd.Timedelta(minutes=5);possible=starts[starts>at+pd.Timedelta(seconds=30)]
            execution=possible[0]
            if fixed is not None and fixed['start']<execution:break
            where=np.flatnonzero((f.start_at==execution).to_numpy())
            if not len(where) or (execution-at).total_seconds()>360:chosen=None;break
            k=int(where[0]);bar=f.iloc[k]
            if bar.open<=stop[k]:reason='risk_stop';fill=bar.open*factors[k]*(1-slip)
            elif bar.open>=target[k]:reason='target';fill=target[k]*factors[k]
            else:reason='preclose_exit';fill=bar.open*factors[k]*(1-slip)
            chosen={'at':execution,'start':execution,'observation_end':bar.end_at,'fill':fill,'reason':reason,'index':k,
                'gap':bool(bar.open<stop[k]),'both':False};break
    if chosen is None:return None,reviews
    day=pd.Timestamp(chosen['at'].tz_convert(core.ET).date());divs=[d for d in base['dividends'] if d['ex']<=day];div=sum(d['economic_amount'] for d in divs)
    release=f.iloc[:chosen['index']+1].available_at.max()
    if divs:release=max(release,base['label_available_at']) if base.get('complete_window') and base.get('label_available_at') is not None else None
    chosen.update(net=(chosen['fill']*(1-fee)+div)/(base['entry_fill_economic']*(1+fee))-1,release=release,dividends=divs,dividend=div,intent=intent)
    return chosen,reviews


def compare(label,event):
    if event is None:
        assert label.get('exit_at') is None and label.get('terminal_release') is None
        assert label.get('net_return') is None or not np.isfinite(label['net_return'])
        return
    for field,k in [('exit_at','at'),('exit_bar_start','start'),('exit_observation_end','observation_end'),('reason','reason'),
        ('terminal_release','release'),('exit_intent','intent'),('gap_below_stop','gap'),('same_bar_both_barriers','both')]:assert label[field]==event[k],(field,label[field],event[k])
    for field,k in [('exit_fill_economic','fill'),('net_return','net'),('dividend_per_economic_unit','dividend')]:near(label[field],event[k])
    assert label['dividends']==event['dividends']


def compare_records(actual,expected):
    from run_planning import json_ready
    a=json_ready(actual);b=json_ready(expected)
    if isinstance(b,dict):
        assert set(a)==set(b)
        for k in b:compare_records(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare_records(x,y)
    elif isinstance(b,float):near(a,b)
    else:assert a==b,(a,b)
