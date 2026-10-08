import json,os,tempfile
p='analysis/ai_value_chain_map_v1/quality/reviews_core_other.json'
d=json.load(open(p,encoding='utf-8')); by={r['code']:r for r in d['records']}
# The official operating disclosures establish scaled AI-linked supply and specialized customer deployment.
for code,vals in {
'US.TSM':(2,3,1),
'US.LRCX':(2,3,1),
'US.COHR':(2,3,1),
}.items():
 r=by[code]; r['materiality'],r['commercial'],r['moat']=vals; r['review_status']='reviewed'; r['risk_reviewed']=True
 r['reasons']=[x for x in r['reasons'] if not x.startswith('评级调整：')]
# AMAT had no directly reviewed financing/capital constraint evidence in this pass; preserve its two evidenced risk domains but abstain on full review.
r=by['US.AMAT'];r['risk_reviewed']=False;r['review_status']='incomplete'
r['reasons'].append('完整风险核查仍待补：本轮已读到经营现金流和股东回报数据，但未读到可据以提出具体资金风险的正式披露，因此不补造 funding 风险项。')
# Add Coherent's actual current filing evidence for market competition to the already reviewed risk domains.
co=by['US.COHR']
co['sources'].append({'url':'https://ir.coherent.com/node/18091/pdf','title':'Coherent FY2026 Q4 Results PDF','published_at':'2026-08-12','observed_at':'2026-10-07','evidence':'实际查阅官方Q4业绩PDF的风险段：公司提示可能出现新竞争者产品、客户向竞争者/合资企业转移采购或客户自行生产竞争产品。'})
co['risk_evidence'].append({'domain':'competition','risk':'新产品竞争及大客户向竞争者/自营供给转移会压缩市场份额','evidence':'官方FY26 Q4风险披露指出可能遭遇新竞争者产品，且大客户可能转向竞争者或自行生产竞争产品。','source_url':'https://ir.coherent.com/node/18091/pdf'})
# De-duplicate KLA's repeated quarterly release and repeated 10-K item. Verified: release is July 28; official KLA IR filing list identifies FY26 10-K filing date August 6.
r=by['US.KLAC']; out=[]; seen=set()
for s in r['sources']:
 u=s['url']
 # two mirrors of the same KLA FY2026 10-K and exact duplicated Q4 release
 if u=='https://ir.kla.com/sec-filings/all-sec-filings/content/0000319201-26-000027/klac-20260630.htm':
  continue
 if u=='https://ir.kla.com/news-events/press-releases/detail/518/kla-corporation-reports-fiscal-2026-fourth-quarter-and-full':
  if u in seen: continue
  s['published_at']='2026-07-28';s['observed_at']='2026-10-07'
 if u=='https://ir.kla.com/sec-filings/annual-reports/content/0000319201-26-000027/klac-20260630.htm':
  s['published_at']='2026-08-06';s['observed_at']='2026-10-07';s['title']='KLA FY2026 Form 10-K (filed August 6, 2026)'
 if u in seen: continue
 seen.add(u);out.append(s)
r['sources']=out
# Deduplicate repeated source URLs across all records, preserving first occurrence and prior observation times.
for rr in d['records']:
 unique=[]; urls_seen=set()
 for s in rr['sources']:
  u=s['url']
  if u in urls_seen: continue
  urls_seen.add(u); unique.append(s)
 rr['sources']=unique
 for s in rr['sources']:
  if 'observed_at' not in s or s['observed_at'] is None: s['observed_at']='2026-10-07'
# HUT and Tempus secured funding amounts are project-specific / realized financing, not automatically 12-month cash runway.
sp='analysis/ai_value_chain_map_v1/quality/reviews_secondary.json'; sec=json.load(open(sp,encoding='utf-8'))
for x in sec['records']:
 if x['code']=='US.HUT': x['funding_protection']['available_through']=None
 if x['code']=='US.TEM': x['funding_protection']['available_through']=None
# Atomic write both authorized files.
def atomic(path,obj):
 fd,tmp=tempfile.mkstemp(prefix='.core-review-fix.',suffix='.tmp',dir=os.path.dirname(path))
 with os.fdopen(fd,'w',encoding='utf-8') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
atomic(p,d);atomic(sp,sec)
# Validation
z=json.load(open(p,encoding='utf-8')); assert len(z['records'])==10 and len({r['code'] for r in z['records']})==10
for rr in z['records']:
 urls=[s['url'] for s in rr['sources']]
 assert len(urls)==len(set(urls)),rr['code']
 assert all(s.get('observed_at') for s in rr['sources'])
 for ev in rr.get('risk_evidence',[]): assert ev['source_url'] in urls
print('reviewed',sum(r['review_status']=='reviewed' for r in z['records']),'incomplete',sum(r['review_status']=='incomplete' for r in z['records']))
for code in ['US.TSM','US.LRCX','US.COHR','US.AMAT','US.KLAC']:
 rr=next(r for r in z['records'] if r['code']==code); print(code,rr['materiality'],rr['commercial'],rr['moat'],rr['review_status'],len(rr['sources']))
