"""Independent target timing, raw state definition and date-equal state errors."""
import argparse
from path_learning import *
from verify_reliability import close


def audit(run):
    run=Path(run);saved=read(run/'state_diagnostics.json');results=read(run/'results.json')
    assert saved['results_sha256']==sha(run/'results.json')
    panel=pd.read_parquet(run/'activity_panel.parquet');features=pd.read_parquet(run/'features.parquet')
    market=core.load_market();cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    known={pd.Timestamp(r['at']):r for r in saved['records']};missing={pd.Timestamp(r['at']):r for r in saved['unknown']}
    settings=read(SOURCE/'protocol.json')[saved['phase']];query=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])]
    assert set(known).isdisjoint(missing) and set(known)|set(missing)==set(query.index)
    daily={};checks=0
    for at,row in query.iterrows():
        fit=read(run/'training'/(tag(at)+'.json'))['fits']['conditional']
        if not fit['ready']:assert missing[at]['reason']=='forecast_not_ready';continue
        days=[d for d in sorted(market.calendar) if d>=row.date];future=pd.Timestamp(str(days[2].date())+' 09:25:30',tz=core.ET).tz_convert('UTC')
        complete=future<cutoff and future in panel.index and np.isfinite(panel.loc[future,PRIOR].to_numpy(float)).all()
        if not complete:
            assert missing[at]['reason']=='successor_unknown_at_cutoff' and pd.Timestamp(missing[at]['successor_at'])==future;continue
        z=features.loc[future];sv=max(z.SOXX_vol20,.0001)
        truth=np.array([max(-5,min(5,z.sector_relative_5/(sv*np.sqrt(5)))),2*z.breadth_ret5_positive_fraction-1,np.log(sv)])
        r=known[at];assert pd.Timestamp(r['successor_at'])==future and pd.Timestamp(r['date'])==row.date
        close(r['truth'],truth);close(r['prediction'],fit['transition']['next_state_point']);close(r['persistence'],row[PRIOR].to_numpy(float))
        scale=np.array(fit['kernels']['later']['scale']);errors={}
        for method,field in [('transition','prediction'),('persistence','persistence')]:
            squared=np.square(np.array(r[field])-truth);close(r[method+'_squared_error'],squared)
            normalized=float(sum(squared/scale**2)/3);close(r['normalized_'+method+'_mse'],normalized)
            errors[method]=(squared,normalized)
        daily.setdefault(row.date,[]).append(errors);checks+=1
    assert saved['queries']==checks and saved['dates']==len(daily)
    for method in ['transition','persistence']:
        squares=[];norm=[]
        for group in daily.values():
            squares.append(sum(r[method][0] for r in group)/len(group));norm.append(sum(r[method][1] for r in group)/len(group))
        want=np.mean(squares,axis=0)
        for field,value in zip(PRIOR,want):close(saved['state_mse'][method][field],value)
        close(saved['state_mse'][method]['standardized_date_equal_mse'],np.mean(norm))
    for kind in ['conditional','fixed','unconditional']:
        z=results['routes'][kind+'_band'];assert saved['capital_comparison'][kind]=={'monthly_mean':z['primary_average'],'monthly':z['monthly'],'entries':z['entries'],'all_months_nonnegative':z['all_months_nonnegative']}
    out={'passed':True,'known_targets_checked':checks,'unknowns_preserved':len(missing),'actual_dates':len(daily),'diagnostics_sha256':sha(run/'state_diagnostics.json')}
    dump(run/'state_diagnostics_audit.json',out);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
