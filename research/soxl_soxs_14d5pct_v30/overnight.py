"""Daily preclose profit-buffer review, with causal observed overnight tails."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v27'))
from clock_reliability import (core,np,pd,read,dump,sha,labels,prepare_inputs,base_forecast,day_cutoff,
    restrict_market,development_gate,CLOCKS,tag,columns,fit_clock,predict,training_dates,issue)
from verify_continuation import key

ROUTES=['buffer_band','buffer_point','same_entry_fixed','same_entry_flat','source_v29','source_v29_buffer','cash','double_cost']


def gap_records(market):
    """Review mark at close-15m to next regular open, on consecutive sessions."""
    out={}
    for symbol in core.ETFS:
        f=market.bars[symbol];records=[]
        for prior,day in zip(market.sessions[:-1],market.sessions[1:]):
            end=pd.Timestamp(market.calendar[prior]['close_at'])-pd.Timedelta(minutes=15)
            start=pd.Timestamp(market.calendar[day]['open_at']);opened=start+pd.Timedelta(minutes=5)
            if opened>f.index.max():break
            record={'from_date':prior,'date':day,'mark_end':end,'open_at':start,'open_bar_end':opened,'ready':False,'release':None,'loss':None}
            if end in f.index and opened in f.index:
                a,b=f.loc[end],f.loc[opened]
                valid=all(r.session_type=='regular' and r.start_at==r.end_at-pd.Timedelta(minutes=5)
                    and pd.notna(r.available_at) and r.available_at>=r.end_at for r in [a,b])
                if valid and np.isfinite([a.econ_close,b.econ_open]).all() and min(a.econ_close,b.econ_open)>0:
                    record.update(ready=True,release=max(a.available_at,b.available_at),loss=max(0.,1-b.econ_open/a.econ_close))
            records.append(record)
        out[symbol]=records
    return out


def gap_forecast(records,day,at):
    # Do not silently discard a missing pair inside the last official window.
    recent=[r for r in records if r['date']<=day][-126:]
    valid=len(recent)>=60 and recent[-1]['date']==day and all(r['ready'] and r['release']<at for r in recent)
    tail=None;k=None
    if valid:
        k=math.ceil(.1*len(recent));tail=float(np.mean(sorted(r['loss'] for r in recent)[-k:]))
    return {'at':at,'date':day,'members':[r['date'] for r in recent],'ready':bool(valid),'tail_count':k,'adverse_tail_mean':tail,
        'latest_release':max((r['release'] for r in recent if r['release'] is not None),default=None)}


def make_gap_forecasts(market,features):
    records=gap_records(market);last=features.date.max();forecasts={}
    for day in market.sessions[(market.sessions>=features.date.min())&(market.sessions<=last)]:
        at=pd.Timestamp(market.calendar[day]['close_at'])-pd.Timedelta(minutes=15)+pd.Timedelta(seconds=30)
        for s in core.ETFS:forecasts[(day,s)]=gap_forecast(records[s],day,at)
    return records,forecasts


def review(entry,mark,forecast,multiple,mode,mark_ready=True):
    fee=slip=.0005*multiple
    if mode=='flat':return {'action':'EXIT','reason':'scheduled_preclose','buffer_after_tail':None}
    if not mark_ready or not forecast['ready']:return {'action':'EXIT','reason':'risk_data_unavailable','buffer_after_tail':None}
    value=mark*(1-forecast['adverse_tail_mean'])*(1-slip)*(1-fee)/(entry*(1+fee))-1
    return {'action':'HOLD' if value>=0 else 'EXIT','reason':'profit_buffer_covers_tail' if value>=0 else 'insufficient_profit_buffer',
        'buffer_after_tail':float(value)}


def expected_grid(market,label):
    grid=pd.DatetimeIndex([],tz='UTC')
    for day in market.sessions[(market.sessions>=label['entry_date'])&(market.sessions<=label['expiry_date'])]:
        left=max(label['entry_at'],pd.Timestamp(market.calendar[day]['open_at']));right=pd.Timestamp(market.calendar[day]['close_at'])
        grid=grid.append(pd.date_range(left,right,freq='5min',inclusive='right'))
    return grid


def managed_label(market,base,forecasts,mode,multiple=1):
    if 'entry_fill_economic' not in base:return dict(base),[]
    out=dict(base);out.update(risk_rule='preclose_'+mode,exit_at=None,exit_fill_economic=np.nan,net_return=np.nan,
        reason='pending',target_hit=None,terminal_release=None,exit_intent=None,exit_bar_start=None,exit_observation_end=None,
        gap_below_stop=False,same_bar_both_barriers=False,protection_raised=False)
    out['attached_order']={**base.get('attached_order',{}),'preclose_review':mode,'review_minutes_before_close':15,'broker_submission':False}
    out.pop('stop_economic',None)
    entry=base['entry_fill_economic'];target=base['limit_economic'];stop=entry*.94;fee=slip=.0005*multiple
    f=market.bars[base['symbol']];path=f.reindex(expected_grid(market,base));intent=None;records=[];used=[];event=None
    for bar in path.itertuples():
        if pd.isna(bar.start_at) or pd.isna(bar.available_at) or bar.available_at<bar.end_at or bar.session_type!='regular':break
        if not np.isfinite([bar.open,bar.high,bar.low,bar.close,bar.share_factor]).all() or min(bar.open,bar.high,bar.low,bar.close,bar.share_factor)<=0:break
        if bar.end_at!=bar.Index or bar.start_at!=bar.end_at-pd.Timedelta(minutes=5):break
        used.append(bar.available_at);factor=bar.share_factor
        rs=math.floor(stop/factor*100+1e-10)/100;rt=math.ceil(target/factor*100-1e-10)/100
        if intent is not None and bar.start_at>intent+pd.Timedelta(seconds=30):
            if (bar.start_at-intent).total_seconds()>360:break
            # At the opening instant existing orders precede a discretionary exit.
            if bar.open<=rs:reason='risk_stop';fill=bar.open*factor*(1-slip);out['gap_below_stop']=bool(bar.open<rs)
            elif bar.open>=rt:reason='target';fill=rt*factor
            else:reason='preclose_exit';fill=bar.open*factor*(1-slip)
            event=(bar.start_at,bar.start_at,bar.end_at,fill,reason);break
        if bar.low<=rs:
            out.update(gap_below_stop=bool(bar.open<rs),same_bar_both_barriers=bool(bar.high>=rt))
            event=(bar.end_at,bar.start_at,bar.end_at,min(bar.open,rs)*factor*(1-slip),'risk_stop');break
        if bar.high>=rt:event=(bar.end_at,bar.start_at,bar.end_at,rt*factor,'target');break
        if bar.end_at==base['expiry_at']:
            event=(bar.end_at,bar.start_at,bar.end_at,bar.econ_close*(1-slip),'expiry');break
        close=pd.Timestamp(market.calendar[bar.date]['close_at']);mark_time=close-pd.Timedelta(minutes=15)
        if bar.end_at==mark_time:
            at=mark_time+pd.Timedelta(seconds=30);forecast=forecasts[(bar.date,base['symbol'])]
            mark_ready=bar.available_at<=at
            d=review(entry,float(bar.econ_close) if mark_ready else None,forecast,multiple,mode,mark_ready)
            records.append({'entry_at':base['entry_at'],'decision_at':base['decision_at'],'symbol':base['symbol'],
                'at':at,'mark_end':bar.end_at,'mark_available_at':bar.available_at,'mark_economic':float(bar.econ_close) if mark_ready else None,
                'mode':mode,'cost_multiple':multiple,'forecast':forecast,**d})
            if d['action']=='EXIT':intent=at
    if event is None:
        out.update(dividends=base['dividends'],dividend_per_economic_unit=base['dividend_per_economic_unit'],prefix_complete=False)
        return out,records
    end,start,observation_end,fill,reason=event;end_day=pd.Timestamp(end.tz_convert(core.ET).date())
    dividends=[d for d in base['dividends'] if d['ex']<=end_day];div=sum(d['economic_amount'] for d in dividends)
    release=max(used)
    if dividends:
        release=max(release,base['label_available_at']) if base.get('complete_window') and base.get('label_available_at') is not None else None
    out.update(exit_at=end,exit_bar_start=start,exit_observation_end=observation_end,exit_fill_economic=fill,reason=reason,target_hit=int(reason=='target'),
        net_return=(fill*(1-fee)+div)/(entry*(1+fee))-1,dividends=dividends,dividend_per_economic_unit=div,
        prefix_complete=True,terminal_release=release,exit_intent=intent,stop_economic=rs*factor)
    return out,records


def build_labels(market,base,stress,forecasts):
    sets={};reviews={}
    for name,source,mode,multiple in [('buffer',base,'buffer',1),('buffer_stress',stress,'buffer',2),('flat',base,'flat',1)]:
        lab={};rs=[]
        for k,b in source.items():
            v,records=managed_label(market,b,forecasts,mode,multiple);lab[k]=v;rs.extend(records)
        sets[name]=lab;reviews[name]=rs
    return sets,reviews


def managed_panel(original,labs):
    p=original.copy();p['terminal_release']=pd.Series(pd.NaT,index=p.index,dtype='datetime64[ns, UTC]')
    for at in p.index:
        pair=[labs[key(at,s)] for s in core.ETFS];net=np.array([x.get('net_return') for x in pair],dtype=float)
        l,s=np.log1p(net);p.loc[at,'C']=(l+s)/2;p.loc[at,'D']=(l-s)/2;p.loc[at,'M']=abs((l-s)/2)
        releases=[x.get('terminal_release') for x in pair]
        if all(t is not None for t in releases):p.loc[at,'terminal_release']=max(releases)
    # Only early releases are used; full-window dates are not rewritten.
    return p


def train(panel,history,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at))
    model,fit=fit_clock(same,dates['early'],columns(panel.loc[at,'clock'],'context'));pred=predict(model,same.loc[[at]])
    raw=np.array([pred[s] for s in core.ETFS]) if pred else None
    decision=issue(panel,history,at,raw)
    return model,{'raw':raw,'fit':fit,'base_feedback':fb,**decision}


def route_events(at,date,record,old):
    band=record['band'];point=record['point'];cash={'ready':True,'action':'CASH','allocation':0.}
    events={'buffer_band':band,'buffer_point':point,'same_entry_fixed':band,'same_entry_flat':band,
        'source_v29':old,'source_v29_buffer':old,'cash':cash,'double_cost':band}
    return {n:{**e,'at':at,'date':date} for n,e in events.items()}


def labels_for(route,sets,base):
    if route in ['source_v29','same_entry_fixed']:return base
    return sets['buffer_stress' if route=='double_cost' else 'flat' if route=='same_entry_flat' else 'buffer']


def read_managed(path):
    out=labels(path)
    for value in out.values():
        for field in ['terminal_release','exit_observation_end','exit_intent']:
            if value.get(field) is not None:value[field]=pd.Timestamp(value[field])
    return out
