import json, os, tempfile

def atomic(path, obj):
 fd,tmp=tempfile.mkstemp(prefix='.risk-evidence.',suffix='.tmp',dir=os.path.dirname(path))
 with os.fdopen(fd,'w',encoding='utf-8') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
core='analysis/ai_value_chain_map_v1/quality/reviews_core_other.json'
d=json.load(open(core,encoding='utf-8')); by={r['code']:r for r in d['records']}
risks={
'US.MU':[],
'US.TSM':[
 {'domain':'customer','risk':'销售/采购集中度及先进制程客户需求波动','evidence':'官方风险管理披露将 sales and purchasing concentration 列为运营风险，并将需求与产能扩张纳入战略风险范围。','source_url':'https://investor.tsmc.com/english/risk-management'},
 {'domain':'competition','risk':'技术发展、行业变化与竞争可能削弱先进制程优势','evidence':'官方风险范围明确列出 technology development and competition，并指出技术/产业变化属于战略风险。','source_url':'https://investor.tsmc.com/english/risk-management'},
 {'domain':'funding','risk':'重资本支出与水电、材料、土地资源约束可能限制扩产','evidence':'公司披露资源稀缺影响未来产能且带来重资本支出；资源成本上升和产能约束也被列为影响。','source_url':'https://investor.tsmc.com/english/risk-management'}],
'US.AMAT':[
 {'domain':'customer','risk':'客户集中及客户需求/产能规划变化会影响设备需求','evidence':'Q3正式披露将客户基础集中、客户技术和产能需求及半导体需求列为业绩风险因素。','source_url':'https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-third-quarter-2026-results'},
 {'domain':'competition','risk':'新技术迭代、市场接受度和市场份额竞争','evidence':'官方风险因素包括开发/支持新产品能力、市场接受度、扩大市场份额以及保护关键技术知识产权。','source_url':'https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-third-quarter-2026-results'}],
'US.LRCX':[
 {'domain':'customer','risk':'客户行动与半导体行业/整体经济变化可能导致设备订单偏离预期','evidence':'季度稿明确列出客户行动可能与预期不符、消费电子/半导体及整体经济条件变化风险。','source_url':'https://investor.lamresearch.com/2026-07-29-Lam-Research-Corporation-Reports-Financial-Results-for-the-Quarter-Ended-June-28%2C-2026'},
 {'domain':'competition','risk':'竞争者行为、贸易限制及技术变化会影响销售与竞争表现','evidence':'季度稿指出客户和竞争者行为不确定，出口管制/关税可能阻碍销售。','source_url':'https://investor.lamresearch.com/2026-07-29-Lam-Research-Corporation-Reports-Financial-Results-for-the-Quarter-Ended-June-28%2C-2026'},
 {'domain':'funding','risk':'供应链成本、关税和制造能力约束可能挤压利润或限制交付','evidence':'季度稿具体提示供应链成本/通胀、出口控制和制造产能限制可能影响利润率与销售。','source_url':'https://investor.lamresearch.com/2026-07-29-Lam-Research-Corporation-Reports-Financial-Results-for-the-Quarter-Ended-June-28%2C-2026'}],
'US.KLAC':[
 {'domain':'customer','risk':'高客户集中使单一客户业务/技术变化影响放大','evidence':'FY26 10-K指出客户集中，并称客户或foundry/logic行业技术变化会增加对业务、财务与经营结果的影响。','source_url':'https://ir.kla.com/sec-filings/annual-reports/content/0000319201-26-000027/klac-20260630.htm'},
 {'domain':'competition','risk':'行业技术变迁、AI发展和竞争可能削弱技术优势','evidence':'FY26业绩稿列出AI/制造工艺变化、客户投资模式、维持技术优势和竞争能力等风险。','source_url':'https://ir.kla.com/news-events/press-releases/detail/518/kla-corporation-reports-fiscal-2026-fourth-quarter-and-full'},
 {'domain':'funding','risk':'债务/杠杆结构及关键供应商、组件约束影响资本承受与交付','evidence':'FY26业绩稿风险条款明确列出debt and leveraged capital structure及关键/有限来源供应商风险。','source_url':'https://ir.kla.com/news-events/press-releases/detail/518/kla-corporation-reports-fiscal-2026-fourth-quarter-and-full'}],
'US.ORCL':[
 {'domain':'customer','risk':'AI云客户需求超过供给，合同交付依赖持续扩容执行','evidence':'Q1披露AI训练/推理服务需求增长快于供给，新增AI合同30B美元且交付30万余GPU，反映容量兑现风险。','source_url':'https://investor.oracle.com/investor-news/news-details/2026/Oracle-Announces-Q1-Results-Driven-by-Triple-Digit-Growth-in-Cloud-Infrastructure-Revenues/default.aspx'},
 {'domain':'competition','risk':'企业AI平台面临成熟市场参与者竞争','evidence':'Oracle同一业绩稿称Palantir率先为客户构建企业本体，并介绍Oracle自动化方案；可见具体竞品存在，但未量化胜负。','source_url':'https://investor.oracle.com/investor-news/news-details/2026/Oracle-Announces-Q1-Results-Driven-by-Triple-Digit-Growth-in-Cloud-Infrastructure-Revenues/default.aspx'},
 {'domain':'funding','risk':'数据中心投入使自由现金流为负并伴随股权融资/稀释','evidence':'Q1经营现金流23B美元但自由现金流-5B美元，公司通过ATM出售20B美元普通股用于投资计划。','source_url':'https://investor.oracle.com/investor-news/news-details/2026/Oracle-Announces-Q1-Results-Driven-by-Triple-Digit-Growth-in-Cloud-Infrastructure-Revenues/default.aspx'}],
'US.DELL':[
 {'domain':'customer','risk':'AI服务器需求、客户/产品组合和订单转收入存在波动风险','evidence':'FY27 Q2披露AI服务器95B美元backlog，并将未来经营风险明确关联AI解决方案需求及产品/客户组合。','source_url':'https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-second-quarter-fiscal-2027-financial'},
 {'domain':'competition','risk':'竞争压力及产品过渡/交付质量影响方案竞争力','evidence':'FY27 Q2风险因素列出竞争压力、产品/服务转型及交付高质量产品和方案的能力。','source_url':'https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-second-quarter-fiscal-2027-financial'},
 {'domain':'funding','risk':'供应商集中、资本市场准入及较高负债可能影响大规模部署','evidence':'FY27 Q2风险因素列出单一/有限来源供应商、公司/客户资本市场准入与Dell负债水平。','source_url':'https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-second-quarter-fiscal-2027-financial'}],
'US.HPE':[
 {'domain':'customer','risk':'订单/积压和客户需求预测具有不确定性','evidence':'FY26 Q3稿把订单/backlog、产品需求和客户合同执行列入预测不确定性；管理层同时称当前订单积压创纪录。','source_url':'https://investors.hpe.com/?releaseid=909541'},
 {'domain':'competition','risk':'竞争与技术趋势、新产品开发和Juniper整合可能影响竞争表现','evidence':'FY26 Q3稿明确提到竞争压力、AI发展、产品技术趋势及Juniper并购整合兑现风险。','source_url':'https://investors.hpe.com/?releaseid=909541'},
 {'domain':'funding','risk':'供应与贸易冲击、流动性/资本资源及债务偿付影响扩张和现金分配','evidence':'FY26 Q3稿列出组件短缺、贸易限制、流动性与资本资源、债务偿还；同时管理层承诺返还Q4至少75%自由现金流。','source_url':'https://investors.hpe.com/?releaseid=909541'}],
'US.COHR':[
 {'domain':'customer','risk':'客户集中度较高，少数客户收入变化会显著影响业绩','evidence':'FY26年报XBRL披露第一、第二大客户分别占FY26收入20%与12%，主要来自Datacenter & Communications。','source_url':'https://ir.coherent.com/node/18116/xbrl-viewer'},
 {'domain':'funding','risk':'光互连需求扩张依赖制造扩产投入且公司存在长期债务偿付','evidence':'Q4 FY26稿称优先投资扩充制造能力；FY26年报XBRL列出未来年度长期债务本金偿还安排。','source_url':'https://ir.coherent.com/node/18116/xbrl-viewer'}],
'US.CIEN':[
 {'domain':'customer','risk':'客户集中度较高，两家客户贡献显著收入','evidence':'官方Q3 2026结果披露两名客户各超过收入10%，合计占总收入41.7%。','source_url':'https://investor.ciena.com/news/news-details/2026/Ciena-Reports-Fiscal-Third-Quarter-2026-Financial-Results/default.aspx'},
 {'domain':'competition','risk':'竞争技术和AI对网络投资的影响会改变订单需求','evidence':'官方Q3风险条款列出AI对整体网络技术支出的影响、新竞争技术、竞争压力及客户订单规模/时点。','source_url':'https://investor.ciena.com/news/news-details/2026/Ciena-Reports-Fiscal-Third-Quarter-2026-Financial-Results/default.aspx'},
 {'domain':'funding','risk':'季度再融资交易涉及可转债发行与定期贷款偿付','evidence':'Q3财务披露指出偿还经再融资的2030定期贷款并发行可转债，产生债务清偿成本及利率掉期终止事项。','source_url':'https://investor.ciena.com/news/news-details/2026/Ciena-Reports-Fiscal-Third-Quarter-2026-Financial-Results/default.aspx'}]
}
for c,risk in risks.items(): by[c]['risk_evidence']=risk
atomic(core,d)
# already-reviewed secondary risk evidence from actual issuer material
sec='analysis/ai_value_chain_map_v1/quality/reviews_secondary.json'; s=json.load(open(sec,encoding='utf-8')); sb={r['code']:r for r in s['records']}
sec_risks={
'US.HUT':[
 {'domain':'funding','risk':'AI数据中心建设需项目融资且合同收入有交付时间差','evidence':'Q2披露项目融资已落实7.5B美元，但设施仍处建设推进阶段，合同容量和预计多年合同价值不等于已确认收入，项目交付仍是资本部署风险。','source_url':'https://www.hut8.com/news-insights/press-releases/hut-8-reports-second-quarter-2026-results'}],
'US.RXRX':[
 {'domain':'customer','risk':'合作里程碑及合作收入波动，研发/商业化高度依赖伙伴推进','evidence':'Q2收入主要来自合作协议，仅770万美元；公司披露Genentech靶点进入早期发现阶段，主要后续里程碑仍属潜在。','source_url':'https://ir.recursion.com/news-releases/news-release-details/recursion-reports-second-quarter-financial-results-genentech'},
 {'domain':'competition','risk':'临床候选药失败或监管路径不确定，AI发现尚未等同获批药品','evidence':'官方披露候选药处于早期发现、临床1/2期或临床2期，公司仍需与FDA讨论注册路径；研发临床阶段存在转化不确定性。','source_url':'https://ir.recursion.com/news-releases/news-release-details/recursion-reports-second-quarter-financial-results-genentech'},
 {'domain':'funding','risk':'持续研发消耗现金，跑道依赖经营假设及合作收入','evidence':'截至2026-06-30现金及受限现金556.8M美元，Q2经营现金流净流出105.9M美元；管理层预计无额外融资时现金可延续至2028年初。','source_url':'https://ir.recursion.com/news-releases/news-release-details/recursion-reports-second-quarter-financial-results-genentech'}],
'US.TEM':[
 {'domain':'customer','risk':'数据授权、诊断和药企合作依赖客户获取/续约及数据许可','evidence':'Q2稿将吸引和留住客户及合作伙伴列为风险；数据/应用许可是独立业务收入来源，新增许可约200M美元。','source_url':'https://investors.tempus.com/news-releases/news-release-details/tempus-reports-second-quarter-2026-results'},
 {'domain':'competition','risk':'医疗AI竞争、新市场进入者及监管变化影响产品商业化','evidence':'Q2稿明确列出竞争、新市场进入者、AI监管演化和保护知识产权等风险。','source_url':'https://investors.tempus.com/news-releases/news-release-details/tempus-reports-second-quarter-2026-results'},
 {'domain':'funding','risk':'研发和收购投入、可转债债务及再融资能力影响资本需求','evidence':'Q2稿列出偿还/再融资债务和额外融资能力风险；截至6月30日余额含1.173B美元可转债，H1发行可转债净收443.1M美元。','source_url':'https://investors.tempus.com/news-releases/news-release-details/tempus-reports-second-quarter-2026-results'}]
}
for c, ev in sec_risks.items(): sb[c]['risk_evidence']=ev
# documented financing protection: secured project finance and executed convertible note proceeds
sb['US.HUT']['funding_protection']={'source_type':'committed_facility','committed':True,'amount_value':7.5,'amount_unit':'USD billion','source_url':'https://www.hut8.com/news-insights/press-releases/hut-8-reports-second-quarter-2026-results','evidence':'发行人披露已取得7.5B美元投资级项目融资；项目融资已secured to date。','available_through':None}
sb['US.TEM']['funding_protection']={'source_type':'cash_raise','committed':True,'amount_value':443.132,'amount_unit':'USD million','source_url':'https://investors.tempus.com/news-releases/news-release-details/tempus-reports-second-quarter-2026-results','evidence':'截至2026-06-30的半年现金流表记录可转债发行净收443.132M美元，属于已完成到账融资；不把余额等同于未来融资承诺。','available_through':None}
atomic(sec,s)
# Validate precise source URL links and minimal shape
for path in (core,sec):
 z=json.load(open(path,encoding='utf-8'))
 for r in z['records']:
  for e in r.get('risk_evidence',[]):
   assert e['domain'] in ('customer','competition','funding')
   assert e['source_url'] in {s['url'] for s in r['sources']}, (r['code'],e['source_url'])
  if r.get('funding_protection'):
   assert r['funding_protection']['source_url'] in {s['url'] for s in r['sources']}
print('core_other risk evidence',sum(len(x.get('risk_evidence',[])) for x in d['records']))
print('secondary HUT/RXRX/TEM risk evidence',sum(len(sb[c].get('risk_evidence',[])) for c in sec_risks), 'funding protections', [c for c in ('US.HUT','US.TEM') if sb[c].get('funding_protection')])
