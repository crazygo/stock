"""Recompute report measures from as-issued records and actual trade CSVs."""
import argparse
from path_learning import *
from verify_reliability import close


def audit(run):
    run=Path(run);saved=read(run/'diagnostics.json');results=read(run/'results.json');assert saved['results_sha256']==sha(run/'results.json') and saved['verification_sha256']==sha(run/'verification.json')
    panel=pd.read_parquet(run/'activity_panel.parquet');settings=read(SOURCE/'protocol.json')[saved['phase']];panel=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])];cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive']);checks=0
    for k in VARIANTS:
        values=[];group={};unknown=0
        for at,r in panel.iterrows():
            rec=read(run/'training'/(tag(at)+'.json'));f=rec['fits'][k]
            if not f['ready']:continue
            values.append((at,f))
            if pd.isna(r.terminal_release) or r.terminal_release>=cutoff:unknown+=1;continue
            group.setdefault(r.date,[]).append((r,rec))
        stats=saved['variants'][k];assert stats['ready_queries']==len(values) and stats['unknown_queries']==unknown and stats['mature_queries']==sum(map(len,group.values())) and stats['mature_dates']==len(group)
        means=[];points=[];brier=[]
        for records in group.values():
            sq=[];cp=[];weighted=den=0.
            for r,rec in records:
                target=[r.C+r.D,r.C-r.D];raw=rec['raws'][k];sq.append(sum((raw[j]-target[j])**2 for j in range(2))/2)
                if rec['estimates'][k]['ready']:cp.append(sum((rec['estimates'][k]['point'][j]-target[j])**2 for j in range(2))/2)
                weighted+=abs(r.D)*(rec['fits'][k]['raw']['q']-int(r.D>0))**2;den+=abs(r.D)
            means.append(sum(sq)/len(sq))
            if cp:points.append(sum(cp)/len(cp))
            if den>0:brier.append(weighted/den)
        close(stats['raw_log_return_mse_date_equal'],np.mean(means));close(stats['calibrated_log_return_mse_date_equal'],np.mean(points));close(stats['direction_brier_absD_within_date_then_date_equal'],np.mean(brier));assert stats['direction_neutral_brier']==.25
        for phase in ['first','later']:
            v=[f['kernels'][phase]['effective_dates'] for _,f in values];close(stats['effective_'+phase+'_min_median'],[min(v),np.median(v)])
        se=[v for _,f in values for v in f['monte_carlo_se']];close(stats['integration_se_mean_max'],[np.mean(se),max(se)])
        counts=[len(f['members']) for _,f in values];assert stats['source_blocks_min_max']==[min(counts),max(counts)] and stats['real_dates']==len(set(at.tz_convert(core.ET).date() for at,_ in values))
        for i in ['1','2','3']:close(stats['mean_simulated_reason_frequencies'][i],np.mean([f['reason_frequencies'][i] for _,f in values],axis=0))
        checks+=15
    pred=read(run/'decisions_frozen.json')['routes']
    for k in ROUTES:
        v=saved['capital_actions'][k];ds=pd.read_csv(run/(k+'_decisions.csv'));issued=[x for x in pred[k] if x['action']!='CASH'];assert v['proposals']==len(issued) and v['proposal_dates']==len(set(x['date'] for x in issued)) and v['decisions']==ds.decision.value_counts().to_dict()
        try:t=pd.read_csv(run/(k+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        assert v['entries']==len(t) and len(v['trades'])==len(t)
        for a,b in zip(v['trades'],t.to_dict('records')):
            for f in ['decision_at','symbol','reason']:assert a[f]==b[f]
            if a['net_return'] is None:assert pd.isna(b['net_return'])
            else:close(a['net_return'],b['net_return'])
        checks+=6
    close(saved['conditional_minus_unconditional_month_mean'],results['primary']['primary_average']-results['routes']['unconditional_band']['primary_average']);assert saved['primary_passed']==results.get('development_gate',results.get('case_acceptance'))['passed']
    out={'passed':True,'checks':checks+2,'diagnostics_sha256':sha(run/'diagnostics.json')};dump(run/'diagnostics_audit.json',out);print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
