"""R05 prior-complete-day peer context; two fixed input arms."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import OUT,DATES,sha,save,write,now
import research
from prepare import GROUP_OF,GROUPS
sys.path.insert(0,str(Path(__file__).resolve().parent))

def peer_daily(daily):
    f=daily.copy();f['_group']=f.symbol.map(GROUP_OF).fillna('unmapped')
    f['gd_expected_peers']=f['_group'].map({g:len(s)-1 for g,s in GROUPS.items()}).astype(float)
    for n in [5,20]:
        key=f'd_return_{n}';valid=f[key].notna();value=f[key].fillna(0.)
        groups=[f.day,f['_group']]
        count=valid.astype(float).groupby(groups).transform('sum')-valid.astype(float)
        total=value.groupby(groups).transform('sum')-value
        positive=((value>0)&valid).astype(float)
        squared=(value*value).groupby(groups).transform('sum')-value*value
        f[f'gd_count_{n}']=count
        f[f'gd_mean_{n}']=(total/count).where(count>=2)
        f[f'gd_breadth_{n}']=((positive.groupby(groups).transform('sum')-positive)/count).where(count>=2)
        f[f'gd_dispersion_{n}']=np.sqrt((squared/count-(total/count)**2).clip(lower=0)).where(count>=2)
        f[f'gd_relative_{n}']=f[key]-f[f'gd_mean_{n}']
    f['gd_coverage_5']=f.gd_count_5/f.gd_expected_peers
    f['gd_breadth_mean_5']=np.nan
    for symbol,indices in f.groupby('symbol',sort=False).groups.items():
        rows=f.loc[indices];series=rows.set_index('day').gd_breadth_5.reindex(DATES).rolling(5,min_periods=3).mean()
        f.loc[indices,'gd_breadth_mean_5']=series.reindex(rows.day).to_numpy()
    leaders=f[f.d_return_5.notna()].sort_values(['_group','day','d_return_5','symbol'],ascending=[True,True,False,True]).drop_duplicates(['_group','day'])
    persistence=[]
    for group,rows in leaders.groupby('_group',sort=False):
        identities=rows.set_index('day').symbol.reindex(DATES);codes=pd.Series(pd.factorize(identities)[0],index=DATES,dtype=float).where(identities.notna())
        roll=codes.rolling(5,min_periods=5)
        values=(roll.max()==roll.min()).astype(float).where(roll.count()==5)
        persistence.append(pd.DataFrame({'day':DATES,'_group':group,'gd_leader_persistence_5':values.to_numpy()}))
    f=f.merge(pd.concat(persistence,ignore_index=True),on=['day','_group'],how='left',validate='many_to_one',sort=False)
    columns=[c for c in f if c.startswith('gd_')];f.loc[f['_group']=='unmapped',columns]=np.nan
    f['gd_group_missing']=(f['_group']=='unmapped').astype(float)
    return f[['symbol','day']+columns+['gd_group_missing']]

def prepare():
    dest=OUT/'R05';source=OUT/'R03/cache/panel.parquet';parent=json.loads((OUT/'R03/coverage.json').read_text());assert sha(source)==parent['panel_sha256']
    panel=pd.read_parquet(source);panel.symbol=panel.symbol.astype(str);panel.day=panel.day.astype(str)
    columns=['symbol','day','d_return_5','d_return_20'];daily=panel[columns].drop_duplicates(['symbol','day'])
    keys=panel[['symbol','day','minute']].to_numpy();added=peer_daily(daily)
    panel=panel.merge(added,on=['symbol','day'],how='left',validate='many_to_one',sort=False);assert np.array_equal(keys,panel[['symbol','day','minute']].to_numpy())
    for c in added:
        if c.startswith('gd_'):panel[c]=panel[c].astype('float32')
    panel.symbol=panel.symbol.astype('category');panel.day=panel.day.astype('category');save(panel,dest/'cache/panel.parquet')
    reference=json.loads((OUT/'R03/cache/runs/H3_2026-09/meta.json').read_text())
    coverage=dict(parent,at=now(),rows=len(panel),panel_sha256=sha(dest/'cache/panel.parquet'),parent_panel_sha256=parent['panel_sha256'],protocol_sha256=sha(dest/'PROTOCOL.md'),base_features=reference['features'],added_features=[c for c in added if c.startswith('gd_')],
        grouping='fixed retrospective GROUP_OF/GROUPS; not PIT; leave-one-out peer statistics',input_time='d_return fields shifted one official session; no current-day completed-close data')
    write(dest/'coverage.json',coverage);print(json.dumps(dict(prepared=len(panel),added=coverage['added_features'])),flush=True)

def configure():
    import research
    research.OUT=OUT/'R05';research.VERSION='preopen_5d5pct_v6_R05';research.REPORT_TITLE='五日 +5% 前日多日板块 R05 结果'
    research.ARMS=['D0','D1'];research.ALG={a:'LightGBM' for a in research.ARMS};research.NAMES={'D0':'完整历史成熟状态与单股日线对照','D1':'加入前日多日板块广度与领涨持续性'}
    research.GLOBAL_TOP_CALIBRATION=True;research.TRAINING_ONLY_VOL_EDGES=True
    def columns(arm):
        coverage=json.loads((research.OUT/'coverage.json').read_text());return coverage['base_features']+(coverage['added_features'] if arm=='D1' else [])
    research.columns=columns
    return research

if __name__ in ['__main__','__mp_main__']:configure()

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:research.main()
