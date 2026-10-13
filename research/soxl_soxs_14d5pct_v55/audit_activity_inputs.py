"""Post-result input feasibility audit only. No labels, forecasts or capital use."""
from dense_learning import *


def inspect():
    source,source_sha=verified_dense_source();manifest=read(source/'input_manifest.json');market=core.load_market();assert market.manifest==manifest['full_market']
    queries=pd.read_parquet(source/'features.parquet',columns=['date','clock'])
    cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    coverage={};checks=0
    for symbol in ['SOXL','SOXS','SOXX','QQQ']:
        bars=market.bars[symbol];days={d:g for d,g in bars.groupby('date')};records=[]
        for day in market.sessions:
            for clock in CLOCKS:
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC')
                if at>=cutoff or at>=pd.Timestamp(market.calendar[day]['close_at']):continue
                start=(day.tz_localize(core.ET)+pd.Timedelta(hours=4)).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
                expected=pd.date_range(start,end,freq='5min',inclusive='right');g=days.get(day,bars.iloc[:0]);g=g[(g.end_at>start)&(g.end_at<=end)]
                good=g.index.equals(expected) and g.available_at.notna().all() and g.available_at.le(at).all() and g.available_at.ge(g.end_at).all() and g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all() and g.volume.notna().all() and g.volume.ge(0).all()
                activity=None
                if good:
                    # Approximate traded-dollar activity, not signed order flow.
                    raw=((g.high+g.low+g.close)/3*g.volume).to_numpy(float)
                    economic=((g.econ_high+g.econ_low+g.econ_close)/3*g.econ_volume).to_numpy(float)
                    np.testing.assert_allclose(raw,economic,rtol=1e-12,atol=1e-8)
                    activity=float(raw.sum());good=activity>0;checks+=1
                records.append({'at':at,'date':day,'clock':clock,'complete':bool(good),'activity':activity if good else None,'zero_bars':int(g.volume.eq(0).sum())})
        frame=pd.DataFrame(records).set_index('at');byclock={};ready=[];failures=[]
        for at,row in queries.iterrows():
            current=frame.loc[at];past=frame[(frame.clock==row.clock)&(frame.date<row.date)].tail(20)
            okay=current.complete and len(past)==20 and past.complete.all()
            ready.append(bool(okay))
            if not okay:failures.append({'at':str(at),'current_complete':bool(current.complete),'prior_dates':len(past),'incomplete_prior_dates':[str(x.date()) for x in past.loc[~past.complete,'date']]})
        for clock in CLOCKS:
            indices=np.flatnonzero(queries.clock.eq(clock).to_numpy());byclock[clock]={'queries':len(indices),'current_and_prior20_complete':int(np.array(ready)[indices].sum())}
        coverage[symbol]={'query_count':len(queries),'current_and_prior20_complete':sum(ready),'by_clock':byclock,'missing_retained':failures,
                          'zero_volume_bars_in_used_history':int(bars.volume.eq(0).sum()),'native_turnover_nonnull_rows':int(bars.turnover.notna().sum()) if 'turnover' in bars else 0}
    used={clock:columns(clock,'full') for clock in CLOCKS}
    assert not any('SOXL' in c or 'SOXS' in c for values in used.values() for c in values)
    out={'passed':True,'post_result_only':True,'new_labels_or_capital_computed':False,'source_manifest_sha256':source_sha,
         'raw_market_sha256':{s:m['sha256'] for s,m in market.manifest.items()},'observation_cutoff_exclusive':str(cutoff),
         'coverage':coverage,'split_invariant_complete_prefixes_checked':checks,'current_model_columns':used,
         'finding':'Current model uses leader-share volume factors, not own SOXL/SOXS activity. Same-clock 04:00-through-completed-bar activity needs every expected bar plus20prior eligible sessions. Zero-volume individual bars are retained; nonpositive aggregate remains unknown.',
         'limits':'OHLC-average price times volume is approximate traded-dollar activity, not actual turnover, net creation/redemption or buy/sell order flow. No predictive value tested; no feature/formula chosen from return labels.',
         'source_sha256':sha(Path(__file__))}
    dump(SOURCE/'evidence/activity_input_audit.json',out)
    print({'passed':True,'coverage':{s:{k:v[k] for k in ['query_count','current_and_prior20_complete']} for s,v in coverage.items()},'split_invariant_prefixes':checks},flush=True)

if __name__=='__main__':inspect()
