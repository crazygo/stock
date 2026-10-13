"""Frozen ETF-activity features and separate prequential learning variants."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v55'))
import dense_learning as prior
from dense_learning import (core,np,pd,read,dump,sha,labels,day_cutoff,restrict_market,development_gate,tag,block_counts,decide,
    fit_clock,predict,training_dates,load_sources,load_panel,PANEL_ROOT,registry,fit_day,issue,CLOCKS,OLD_CLOCKS,fit_map,forecast)
SOURCE=Path(__file__).resolve().parent
SYMBOLS=['SOXL','SOXS','SOXX','QQQ']
PAIR=['pair_activity','pair_imbalance','pair_activity_price','pair_imbalance_price']
SECTOR=['sector_relative_activity','sector_activity_price']
FEATURES=PAIR+SECTOR
VARIANTS=['full','pair','sector']
ROUTES=['activity_band','activity_point','pair_band','sector_band','source_v55','four_only','premarket_only','cash','double_cost']


def verified_source():
    src=core.ROOT/'research/soxl_soxs_14d5pct_v55/evidence/delivery_manifest.json';m=read(src)
    run=core.ROOT/'.cache/soxl_soxs_14d5pct_v55/runs/development_20261013';rec=m['runs'][run.name]
    for key,folder in [('top_level_sha256',''),('training_records_sha256','training'),('daily_records_sha256','daily'),('source_daily_records_sha256','source_daily'),('base_models_sha256','models'),('executed_sources_sha256','executed_sources')]:
        for f,h in rec[key].items():assert sha(run/folder/f)==h
    for f,h in m['delivery_files_sha256'].items():assert sha(src.parents[1]/f)==h
    v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    return run,sha(src)


def prefix_activity(group,day,clock,closing):
    at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC')
    if at>=pd.Timestamp(closing):return None
    start=(day.tz_localize(core.ET)+pd.Timedelta(hours=4)).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
    expected=pd.date_range(start,end,freq='5min',inclusive='right')
    g=group[(group.end_at>start)&(group.end_at<=end)]
    good=(g.index.equals(expected) and g.available_at.notna().all() and g.available_at.between(g.end_at,at).all()
          and g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all() and g.volume.notna().all() and g.volume.ge(0).all())
    value=float((((g.high+g.low+g.close)/3)*g.volume).sum()) if good else np.nan
    good=bool(good and np.isfinite(value) and value>0)
    return {'at':at,'date':day,'clock':clock,'complete':good,'activity':value if good else np.nan,'rows':len(g),'last_available':g.available_at.max() if len(g) else None}


def build_prefixes(market,cutoff):
    records=[]
    for symbol in SYMBOLS:
        bars=market.bars[symbol];groups={d:g for d,g in bars.groupby('date')}
        for day in market.sessions:
            group=groups.get(day,bars.iloc[:0])
            for clock in CLOCKS:
                if pd.Timestamp(str(day.date())+' '+clock,tz=core.ET)>=cutoff:continue
                row=prefix_activity(group,day,clock,market.calendar[day]['close_at'])
                if row is not None:records.append({**row,'symbol':symbol})
    return pd.DataFrame(records).set_index(['symbol','at']).sort_index()


def combine_activity(logs,move):
    l,s,x,q=logs;pair=(l+s)/2;imbalance=(l-s)/2;relative=(x-q)/2
    return np.array([pair,imbalance,pair*move,imbalance*move,relative,relative*move])


def build_features(base,prefixes):
    rows=[];receipts=[]
    for at,row in base.iterrows():
        logs=[];members={};ready=True
        for symbol in SYMBOLS:
            f=prefixes.loc[symbol];current=f.loc[at];past=f[(f.clock==row.clock)&(f.date<row.date)].tail(20)
            good=bool(current.complete and len(past)==20 and past.complete.all())
            average=float(past.activity.mean()) if good else None
            value=float(np.clip(np.log(current.activity/average),-3,3)) if good else np.nan
            logs.append(value);ready=ready and good
            members[symbol]={'current':at,'prior':list(past.index),'complete':good,'prior_mean':average,'log_relative':value}
        values=combine_activity(logs,float(row.sector_premarket)) if ready else np.repeat(np.nan,6)
        rows.append({'at':at,'activity_ready':bool(ready),**dict(zip(FEATURES,values))});receipts.append({'at':at,'members':members})
    return pd.DataFrame(rows).set_index('at'),receipts


def columns(clock,kind):
    return prior.columns(clock,'full')+(FEATURES if kind=='full' else PAIR if kind=='pair' else SECTOR if kind=='sector' else [])


def fit_bases(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at));models={};fits={};raws={}
    for kind in VARIANTS:
        model,info=fit_clock(same,dates['early'],columns(panel.loc[at,'clock'],kind));raw=predict(model,same.loc[[at]])
        models[kind]=model;fits[kind]={'fit':info,'feedback':fb,'raw':raw};raws[kind]=np.array([raw[s] for s in core.ETFS]) if raw else None
    return models,fits,raws


def events(at,date,clock,estimates,daily,baseline):
    primary=decide(estimates['full'],daily['full']['feedback'],True);cash={'ready':True,'action':'CASH','allocation':0.}
    actions={'activity_band':primary,'activity_point':decide(estimates['full'],daily['full']['feedback'],False),
             'pair_band':decide(estimates['pair'],daily['pair']['feedback'],True),'sector_band':decide(estimates['sector'],daily['sector']['feedback'],True),
             'source_v55':baseline,'four_only':primary if clock in OLD_CLOCKS else cash,'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in actions.items()}
