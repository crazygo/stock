"""Register local native daily inputs before any new model capital outcomes."""
import sys
from datetime import datetime,timezone
from reaction_inputs import *
sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar


def prepare():
    target=SOURCE/'evidence/price_inputs.json';assert not target.exists()
    original=ROOT/'.cache/doubling_opportunity_v1/coverage_v4/futu_daily_v4/20261005T070246Z'
    cal=calendar('2023-01-01','2026-05-14');used=[x for x in cal['sessions'] if x['session_date']<='2026-04-30']
    dates=[x['session_date'] for x in used];files={};records={}
    out=CACHE/'prices';out.mkdir(exist_ok=True)
    for s in PANEL+['SPY']:
        mp=original/'metadata'/f'{s}.json';meta=read(mp);dp=original/'daily'/f'{s}.parquet';ap=original/'rehab'/f'{s}.parquet'
        assert meta['price_basis']=='NONE' and meta['ticker']==s and meta['status']=='available'
        assert sha(dp)==meta['daily_sha256'] and sha(ap)==meta['rehab_sha256']
        raw=pd.read_parquet(dp);assert raw.code.eq('US.'+s).all()
        raw=raw[raw.time_key.str[:10].between(dates[0],dates[-1])].copy()
        assert list(raw.time_key.str[:10])==dates,(s,'official session coverage')
        actions=pd.read_parquet(ap);actions=actions[actions.ex_div_date.between(dates[0],dates[-1])].copy()
        norm=normalize(raw,actions,used);assert norm.valid.all(),s
        paths={}
        for kind,data in [('daily',raw),('actions',actions),('normalized',norm)]:
            path=out/f'{s}_{kind}.parquet';assert not path.exists()
            data.to_parquet(path,compression='zstd',compression_level=7,index=kind=='normalized');paths[kind+'_path']=str(path.relative_to(ROOT));files[str(path.relative_to(ROOT))]=sha(path)
        for p in [mp,dp,ap]:files[str(p.relative_to(ROOT))]=sha(p)
        records[s]={**paths,'metadata_path':str(mp.relative_to(ROOT)),'source_daily_path':str(dp.relative_to(ROOT)),
            'source_actions_path':str(ap.relative_to(ROOT)),'price_basis':meta['price_basis'],'source':meta['source'],
            'daily_received_at':meta['daily_received_at'],'actions_received_at':meta['rehab_received_at'],
            'official_days':len(dates),'first':dates[0],'last':dates[-1],'missing_days':0,'actions':len(actions),
            'splits':actions[actions.split_ratio.notna()][['ex_div_date','split_ratio']].to_dict('records')}
    for p in [ROOT/'scripts/model_history_calendar.py',ROOT/'analysis/doubling_opportunity_v1/coverage_v4/acquire_daily.py',ROOT/'analysis/doubling_opportunity_v1/raw_daily_v3/acquire.py']:
        files[str(p.relative_to(ROOT))]=sha(p)
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),price_basis='NONE',coverage_passed=True,
        symbols=records,calendar=cal,files_sha256=files,source_tier='existing_local_SHA_verified',new_market_requests=0,
        historical_availability='Conservative midnight ET after daily session; actual 2026 reception separate. No historical receipt or revision-completeness proof.',
        effective_actions_only=True,new_capital_outcomes_seen=False,raw_market_data_in_git=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print({'symbols':len(records),'official_days_each':len(dates),'registered_files':len(files),'new_market_requests':0})


if __name__=='__main__':prepare()
