"""Whole-history support diagnosis; never reads new strategy capital outcomes."""
from bounded92 import *
from features91 import DATA,snapshot


def audit():
    verify_parent();panel=prepare_panel(pd.read_parquet(PARENT/'semiconductor_panel.parquet'),extended_market())
    data=read(DATA);signatures={};rows=[];previous_day=None;checks=0
    for at in panel.index:
        day=str(at.tz_convert(core.ET).date())
        if day not in signatures:
            state=snapshot(data,at.isoformat());signatures[day]=tuple(x['latest_result_id'] for x in state['issuers'])
    for at,row in panel.iterrows():
        day=str(row.date.date())
        if day!=previous_day:
            daily=read(PREVIOUS/'daily'/(day+'.json'));contexts={}
            if daily['ready']:
                for model_name in ['eligible_fundamental','eligible_reaction','top_fundamental']:
                    m=daily['models'][model_name];items=[]
                    for t,j in m['members']:
                        t=pd.Timestamp(t);r=panel.loc[t];c=CLOCKS.index(r.clock)
                        items.append((t,j,c,raw_correction(m,state_vector(r,j),j,c)))
                    contexts[model_name]=items
            previous_day=day
        record=read(PREVIOUS/'training'/(tag(at)+'.json'));source=read(PARENT/'training'/(tag(at)+'.json'))['events']['recent_ratio']
        for model_name in ['eligible_fundamental','eligible_reaction','top_fundamental']:
            e=record['events'][model_name]
            if not e['ready']:continue
            m=daily['models'][model_name];side=core.ETFS.index(e['top_symbol']);clock=CLOCKS.index(row.clock);x=state_vector(row,side)
            indices=INDICES[m['mode']];normalized=(x-m['center'])/m['scale'];z=[1.,float(side)]+[float(clock==k) for k in range(1,8)]+[float(normalized[k]) for k in indices]
            raw=np.array([sum(z[k]*m['beta'][k][j] for k in range(len(z))) for j in range(4)]);raw-=raw.mean()
            np.testing.assert_allclose(raw,raw_correction(m,x,side,clock),atol=1e-11,rtol=1e-11);checks+=4
            q=e['probabilities'];p=source['probabilities'][side]
            for k in range(4):
                for j in range(k):
                    if p[k]>0 and p[j]>0:
                        np.testing.assert_allclose(np.log(q[k]/q[j])-np.log(p[k]/p[j]),raw[k]-raw[j],atol=1e-10,rtol=1e-10);checks+=1
            all_items=contexts[model_name];group=[i for i in all_items if i[1]==side and i[2]==clock]
            points=np.array([[0.]*4]+[i[3].tolist() for i in group]);contrasts=[raw[k]-raw[j] for k in range(4) for j in range(k)]
            extremes=[]
            for k in range(4):
                for j in range(k):
                    column=points[:,k]-points[:,j];v=raw[k]-raw[j];extremes.append(max(0.,v-column.max(),column.min()-v))
            group_dates={str(i[0].tz_convert(core.ET).date()) for i in group}
            rows.append(dict(at=at,date=day,route=model_name,symbol=e['top_symbol'],clock=row.clock,source_eligible=e['source_eligible'],
                history_dates=m['real_dates'],side_dates=len({i[0].tz_convert(core.ET).date() for i in all_items if i[1]==side}),
                same_context_dates=len(group_dates),same_context_rows=len(group),distinct_result_id_sets=len({signatures[d] for d in group_dates}),
                max_standardized_coordinate=float(max(abs(normalized[k]) for k in indices)),
                min_active_scale=float(min(m['scale'][k] for k in indices)),correction_norm=float(np.linalg.norm(raw)),
                correction_logodds_range=float(max(raw)-min(raw)),max_training_norm=float(np.linalg.norm(points,axis=1).max()),
                max_marginal_logodds_excess=float(max(extremes)),outside_fields=len(e['outside_state_fields'])))
    table=pd.DataFrame(rows);table.to_csv(CACHE/'support_queries.csv',index=False)
    def summary(g):
        return dict(queries=len(g),dates=g.date.nunique(),no_context_queries=int((g.same_context_rows==0).sum()),
            median_context_dates=float(g.same_context_dates.median()),median_disclosure_sets=float(g.distinct_result_id_sets.median()),
            logodds_outside_queries=int((g.max_marginal_logodds_excess>1e-9).sum()),
            standardized_coordinate_max=float(g.max_standardized_coordinate.max()),correction_norm_max=float(g.correction_norm.max()),
            correction_norm_median=float(g.correction_norm.median()),logodds_excess_max=float(g.max_marginal_logodds_excess.max()))
    summaries={name:{scope:{symbol:summary(g) for symbol,g in subset.groupby('symbol')} for scope,subset in [('all_ready',group),('source_eligible',group[group.source_eligible]),('development_eligible',group[group.source_eligible & group.date.between('2026-01-01','2026-04-30')])]} for name,group in table.groupby('route')}
    out=dict(passed=True,stage='existing_v91_support_diagnosis_before_new_capital',queries=len(table),numerical_checks=checks,summaries=summaries,
        rows_sha256=sha(CACHE/'support_queries.csv'),source_sha256=sha(Path(__file__)),new_capital_outcomes_seen=False)
    dump(SOURCE/'evidence/support_audit.json',out);print(out,flush=True)


# Keep the imported model coordinate function separate from issuer snapshots.
state_vector=state
if __name__=='__main__':audit()
