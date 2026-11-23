"""Late-entry diagnostic: earlier confirmed tranches can trade only on paper."""
from math import isclose
import json
from pathlib import Path

from study_scale import Config, Context, PATHS, run, LEVELS
from engine import LiveLot, process_price, EPS


HERE = Path(__file__).resolve().parent
INITIAL = 1000.
FEE = 2.50
VIRTUAL = Config((100,200,300),(1,1),'q50','step')
PERIODS = ('development_a','development_b','october_check','november','last_four')


def late_account(data, lots, stage, notional, *, initial=INITIAL):
    # At an open use only the contemporaneous paper fill flag & symbol. Paper
    # exit fields and the occurrence of later stages never enter this routing map.
    triggers = {t['entry_i']:(t['symbol'],t['cycle']) for t in lots if t['stage']==stage}
    cash = equity_peak = initial
    held = None
    trades = []
    closed = blocked = 0
    maximum_drawdown = 0.
    daily = {}
    def cross(i, price, at_open):
        nonlocal held, closed, cash
        event = process_price(held, price, at_open)
        if event is None:
            return
        why,px = event
        t = held.record
        gross = t['qty']*(px-t['entry_price'])
        t.update(exit_i=i,end=data['times'][i].isoformat(),exit_price=px,reason=why,
                 gross=gross,net=gross-2*FEE)
        closed += 1
        cash += t['qty']*px - FEE
        held = None

    for i,now in enumerate(data['times']):
        if held is not None:
            cross(i,data['tape'][held.record['symbol']][i][0],True)
        if i in triggers:
            assert held is None, 'same-stage entries from distinct paper cycles must be exclusive'
            s,cycle = triggers[i]
            if cash + EPS < notional+FEE:
                blocked += 1
            else:
                px = data['tape'][s][i][0]
                t = dict(entry_i=i,start=now.isoformat(),symbol=s,paper_cycle=cycle,
                         qty=notional/px,entry_price=px,exit_i=None,reason='open')
                trades.append(t)
                cash -= notional + FEE
                held = LiveLot(t,px,1.08*px)
        if held is not None:
            _,h,l,c = data['tape'][held.record['symbol']][i]
            for px in (l,h,c) if data['path']=='OLHC' else (h,l,c):
                cross(i,px,False)
                if held is None:
                    break
        equity = cash + (held.record['qty']*c if held is not None else 0.)
        equity_peak = max(equity_peak,equity)
        maximum_drawdown = max(maximum_drawdown,1-equity/equity_peak)
        if i==len(data['days'])-1 or data['days'][i+1]!=data['days'][i]:
            daily[data['days'][i]]=dict(end_equity=equity,open=int(held is not None))
    if held is not None:
        t=held.record
        mark=data['tape'][t['symbol']][-1][3]
        gross=t['qty']*(mark-t['entry_price'])
        t.update(mark_price=mark,net=gross-FEE,gross=gross)
    actual_gross = sum(t['gross'] for t in trades)
    fees = FEE*(len(trades)+closed)
    assert isclose(equity-initial,actual_gross-fees,abs_tol=1e-8)
    paper = {t['entry_i']:t for t in lots if t['stage']==stage}
    for t in trades:
        check = paper[t['entry_i']]
        assert t['reason']==check['reason'] and t['exit_i']==check['exit_i']
        if t['reason']!='open':
            assert isclose(t['exit_price'],check['exit_price'],abs_tol=1e-9)
    return dict(triggers=len(triggers),buys=len(trades),sells=closed,open=len(trades)-closed,
                wins=sum(t['reason']!='open' and t['net']>EPS for t in trades),
                losses=sum(t['reason']!='open' and t['net']<-EPS for t in trades),
                targets=sum(t['reason']=='target' for t in trades),cash_blocked=blocked,
                gross=actual_gross,fees=fees,net=actual_gross-fees,end_equity=equity,
                max_drawdown=maximum_drawdown,daily=daily,trades=trades)


def check_prefix(ctx, path, lots):
    # Removing the suffix can change a final unresolved position, but it must
    # not alter which contemporaneous paper fills had already happened.
    d=ctx.data
    cuts=(LEVELS['q75'],LEVELS['q90'])
    for end in (len(d['times'])//2,3*len(d['times'])//4):
        source={s:x[:end] for s,x in d['tape'].items()}
        small=run(source,d['times'][:end],d['days'][:end],ctx.seed[path,'q50'],ctx.confirm,
                  VIRTUAL.amounts,VIRTUAL.wait,cuts,path,FEE,initial=INITIAL)
        key=lambda t:(t['entry_i'],t['symbol'],t['stage'])
        assert [key(t) for t in small['lots']]==[key(t) for t in lots if t['entry_i']<end]


def main():
    out = dict(kind='Post-freeze paper-probe diagnostic; inception without live deployment.',
               initial=INITIAL, fee_each_fill=FEE, virtual_config=VIRTUAL.record(),
               paper_cash_constrained=True,
               common_fill='The paper and actual conditional orders share the same scheduled next open and cancel together if the paper seed exits before fill. The actual order also requires its own available cash. Both use the same modeled open; no extra routing latency modeled.',
               blocks={})
    for b in PERIODS:
        ctx=Context(b)
        block=out['blocks'][b]={}
        for path in PATHS:
            virtual=ctx.execute(VIRTUAL,path,FEE)
            check_prefix(ctx,path,virtual['lots'])
            data=dict(ctx.data,path=path)
            real={}
            for stage in (2,3):
                for amount in (300,600):
                    real[f'stage{stage}_{amount}']=late_account(data,virtual['lots'],stage,amount)
            block[path]=dict(paper=virtual['summary'],real=real)
    (HERE/'paper_probe.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
