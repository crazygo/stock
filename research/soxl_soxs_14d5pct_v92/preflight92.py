"""Full real-data projection certification before any new capital replay."""
from bounded92 import *
from reference92 import certificate,solve


def preflight():
    hashes=verify_parent();panel=prepare_panel(pd.read_parquet(PARENT/'semiconductor_panel.parquet'),extended_market())
    history={at:read(PARENT/'training'/(tag(at)+'.json')) for at in panel.index};previous_day=None;count=0;anchors=[];maxgap=0.;maxdifference=0.;ranks={};empty=0
    for at,row in panel.iterrows():
        day=str(row.date.date())
        if day!=previous_day:
            daily=read(PREVIOUS/'daily'/(day+'.json'));groups=supports(daily,panel,history);previous_day=day
        record=read(PREVIOUS/'training'/(tag(at)+'.json'))
        for name,parent_name in BMODELS.items():
            e=record['events'][parent_name]
            if not e['ready']:continue
            j=core.ETFS.index(e['top_symbol']);clock=CLOCKS.index(row.clock);group=groups[name][f'{j}:{clock}'];m=daily['models'][parent_name]
            raw=raw_correction(m,state(row,j),j,clock);info=project(raw,group['corrections']);certificate(raw,group['corrections'],info)
            if day in ['2025-07-01','2026-01-02','2026-04-30']:
                other=solve(raw,group['corrections']);difference=float(np.max(abs(other-info['correction'])))
                np.testing.assert_allclose(other,info['correction'],atol=2e-6,rtol=2e-5)
                anchors.append(dict(at=at,route=name,context_rows=len(group['corrections']),rank=info['rank'],difference=difference));maxdifference=max(maxdifference,difference)
            count+=1;maxgap=max(maxgap,info['optimality_gap']);ranks[info['rank']]=ranks.get(info['rank'],0)+1;empty+=int(info['context_rows']==0)
        if (panel.index.get_loc(at)+1)%1600==0:print('Projection preflight',panel.index.get_loc(at)+1,flush=True)
    out=dict(passed=True,projection_checks=count,independent_optimizer_anchors=anchors,maximum_anchor_difference=maxdifference,
        maximum_optimality_gap=maxgap,rank_counts=ranks,no_context_queries=empty,parent_delivery_sha256=hashes,
        new_capital_outcomes_seen=False,source_sha256=sha(Path(__file__)))
    dump(SOURCE/'evidence/model_preflight.json',out);print({k:v for k,v in out.items() if k!='independent_optimizer_anchors'},flush=True)


if __name__=='__main__':preflight()
