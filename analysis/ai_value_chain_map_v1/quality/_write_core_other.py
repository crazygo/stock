import json, os, tempfile
from datetime import date
base_path='analysis/ai_value_chain_map_v1/quality/reviews_core.json'
out_path='analysis/ai_value_chain_map_v1/quality/reviews_core_other.json'
base=json.load(open(base_path, encoding='utf-8'))['records']
want=['US.MU','US.TSM','US.AMAT','US.LRCX','US.KLAC','US.ORCL','US.DELL','US.HPE','US.COHR','US.CIEN']
by={r['code']:r for r in base}
meta={
'US.MU':{
 'risk_reviewed':False,'review_status':'incomplete',
 'reasons':['AI业务：官方FY2026 Q4披露公司称AI驱动需求、战略客户协议支撑FY2027预期；HBM产品是AI加速计算的重要内存配套，主营仍是周期性DRAM/NAND。','优势：规模化内存制造与HBM产品路线有针对性；本次披露未读到具体客户名单或相对竞争数据。','风险/资金：本批仅读到公司提示应参阅10-K/10-Q的标准风险声明；客户集中、竞争与扩产资金风险需继续读最新申报，故保留未完成。'],
 'risk_flags':['待查最新申报：客户集中、行业周期/供需、竞争、制造扩产资本支出'],
 'sources':[{'url':'https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Fiscal-Fourth-Quarter-and-Full-Year-2026-Results/default.aspx','title':'Micron Reports Record Fiscal Fourth-Quarter and Full-Year 2026 Results','published_at':'2026-09-30','evidence':'官方披露FY2026营收133.19B美元、Q4营收54.23B美元及经营现金流；CEO称AI驱动需求、战略客户协议带来财务可见性并提高技术/产品/制造投资。风险段落指向最新10-K/10-Q，未在本轮展开。'}, {'url':'https://investors.micron.com/risk-factor','title':'Micron Investor Relations Risk Factors','published_at':None,'evidence':'实际打开官方风险因素入口，但页面未呈现风险条目，仅提示查看最新10-K/10-Q，故不能视作风险因素已完成核查。'}]
},
'US.TSM':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：Q2 2026季度披露营收40.20B美元，管理层称先进制程需求强劲；同期电话会称AI需求强、CSP客户展望稳健，属于为AI芯片制造提供的先进制程与封装供给。','优势：专注晶圆代工的先进工艺与封装/制造规模构成关键供给能力；资料支持客户需求而非单一AI专属收入。','风险/资金：官方风险管理资料列出资源（水、电、材料、土地）稀缺、重资本支出、地缘与贸易限制、供给链中断及销售/采购集中。'],
 'risk_flags':['高资本开支与水电/土地等资源约束','地缘/贸易限制及供应链中断','销售与采购集中'],
 'sources':[{'url':'https://investor.tsmc.com/english/quarterly-results/2026/q2','title':'TSMC 2026 Q2 Quarterly Results','published_at':'2026-07-16','evidence':'官方季度结果列示Q2营收40.20B美元，Q3营收指引44.6–45.8B美元，毛利率67.7%。'}, {'url':'https://investor.tsmc.com/english/risk-management','title':'TSMC Risk Management','published_at':None,'evidence':'实际查阅官方风险披露：关键资源短缺可限制产能扩张并带来重资本投入；也列出AI治理/第三方依赖、地缘冲突、出口管制/关税与销售采购集中风险。'}]
},
'US.AMAT':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：FY2026 Q3营收9.12B美元，同比增长25%；CEO明确称AI采用推动材料工程解决方案需求，并上调2026年半导体系统业务预期。','优势：公司称材料工程产品与技术对先进芯片制造有支撑；Q3宣布EPIC Center与11项芯片厂/高校/创新伙伴研发合作。','风险/资金：官方披露列出客户需求、客户集中、出口许可/关税、供货能力、新产品被市场采用及IP保护等风险。'],
 'risk_flags':['需求及半导体资本开支周期','客户集中','出口管制/关税与供应商供给','技术迭代、竞争与IP保护'],
 'sources':[{'url':'https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-third-quarter-2026-results','title':'Applied Materials Announces Third Quarter 2026 Results','published_at':'2026-08-13','evidence':'实际阅读官方业绩稿：Q3营收9.12B美元、同比+25%；CEO称AI全球采用带来材料工程需求，调高2026半导体系统收入预期；风险段落具体列出客户集中、出口管制/关税、需求与供应能力等。'}]
},
'US.LRCX':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：官方Q4 FY2026（6月季度）营收6.72B美元，CEO称AI驱动需求正在重塑半导体产业；主营为沉积、刻蚀、清洗等晶圆制造设备。','优势：公司把技术领先与制造复杂度提高联系起来；本轮未取得独立客户份额/竞争份额量化。','风险/资金：季度披露明确指出客户与竞争者行为、出口管制/关税、供应链成本与制造能力限制，以及半导体周期/经济恶化风险。'],
 'risk_flags':['半导体需求周期及客户资本开支','客户/竞争行为变化','出口管制、关税及地缘风险','供应链成本/设备制造产能约束'],
 'sources':[{'url':'https://investor.lamresearch.com/2026-07-29-Lam-Research-Corporation-Reports-Financial-Results-for-the-Quarter-Ended-June-28%2C-2026','title':'Lam Research Reports June 2026 Quarter Results','published_at':'2026-07-29','evidence':'实际阅读官方季度稿：营收6.72B美元、环比+15.1%，CEO称AI需求改变行业并提及技术领先；前瞻风险条款明确列客户/竞争、贸易限制、供应链成本、出口控制及制造容量。'}]
},
'US.KLAC':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：FY2026营收13.58B美元；管理层称AI基础设施扩张提高先进逻辑、存储与先进封装的过程控制需求。','优势：公司披露其过程控制产品处于先进制造关键环节；FY2026 10-K披露客户集中且AI/HPC相关投资支撑增长。','风险/资金：已读FY2026正式业绩稿与10-K中的客户集中、周期、出口管制/关税、关键供应商、技术/竞争迭代、债务及杠杆风险。'],
 'risk_flags':['客户集中和客户资本开支周期','中国出口限制/关税','AI与制程演进带来的竞争及技术风险','关键组件供应、债务/杠杆'],
 'sources':[{'url':'https://ir.kla.com/news-events/press-releases/detail/518/kla-corporation-reports-fiscal-2026-fourth-quarter-and-full','title':'KLA FY2026 Fourth Quarter and Full Year Results','published_at':'2026-07-28','evidence':'官方披露FY26收入13.58B美元，并将AI基础设施扩张、先进设计/存储复杂度与过程控制及先进封装需求联系。风险条款包括周期、AI变化、集中客户、竞争、出口管制、供应商和债务杠杆。'}, {'url':'https://ir.kla.com/sec-filings/annual-reports/content/0000319201-26-000027/klac-20260630.htm','title':'KLA FY2026 Form 10-K','published_at':'2026-08-07','evidence':'实际查阅官方年报页面：FY26增长由AI/HPC支持的先进逻辑、存储及封装投资推动；说明客户集中会放大客户/技术变化影响，并提及贸易限制与债务/杠杆风险。'}]
},
'US.ORCL':{
 'risk_reviewed':False,'review_status':'incomplete',
 'reasons':['AI业务/商业化：Q1 FY2027披露Cloud Infrastructure营收7.4B美元（+121%），AI云客户需求超供给、季度新增AI云合同30B美元，季度内交付超300,000 GPU；其AI Data Platform等产品也作正式发布。','优势：庞大数据库/企业客户基础与自有云基础设施构成部署能力；本次官方来源未披露客户集中度或相对竞争份额。','风险/资金：本期经营现金流23B美元但自由现金流为-5B美元，Q1通过ATM发行20B美元普通股筹资以支持投资计划；竞争和资金可持续性仍需核查最新10-Q/现金流规划。'],
 'risk_flags':['大规模数据中心投资、负自由现金流及外部股权融资','AI云需求高于供应，交付/扩容执行风险','竞争与客户集中度未完成核查'],
 'sources':[{'url':'https://investor.oracle.com/investor-news/news-details/2026/Oracle-Announces-Q1-Results-Driven-by-Triple-Digit-Growth-in-Cloud-Infrastructure-Revenues/default.aspx','title':'Oracle Q1 FY2027 Results','published_at':'2026-09-10','evidence':'实际阅读官方季度稿：IaaS收入7.4B美元同比+121%；新增AI云合同30B美元、RPO 664B美元，交付逾300,000 GPU；Q1自由现金流-5B美元，发行20B美元普通股支持资本投资计划。'}]
},
'US.DELL':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：FY2027 Q2 AI服务器订单60.9B美元、收入16.4B美元，AI服务器积压订单95B美元；ISG营收同比+89%。','优势：广泛的服务器/存储/网络产品、渠道与交付规模支持企业AI部署；尚无独立证据证明长期技术壁垒。','风险/资金：公司列明AI解决方案需求、单一/有限来源供应商、竞争、客户与产品组合、资本市场可得性、负债水平及交付质量风险。'],
 'risk_flags':['AI服务器需求及订单兑现/积压转换','单一/有限来源部件供给与客户/产品组合','竞争、交付质量、债务与资本市场'],
 'sources':[{'url':'https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-second-quarter-fiscal-2027-financial','title':'Dell Technologies Q2 FY2027 Results','published_at':'2026-09-01','evidence':'实际阅读官方结果：Q2 AI服务器订单60.9B美元、AI服务器收入16.4B美元、积压95B美元；风险披露点名AI需求、有限来源供应商、竞争、资本市场及负债。'}]
},
'US.HPE':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：FY2026 Q3 Cloud & AI收入9.0B美元（+25.4%），其中服务器6.8B美元（+35.3%）；另披露Vultr下单1.2B美元AMD Helios AI机架系统。','优势：AI服务器、网络与企业部署能力形成组合供给；本轮未取得可独立验证的长期专利/份额壁垒。','风险/资金：官方当期披露指出组件可得性、贸易政策、需求、利率/流动性/资本资源、Juniper收购整合与竞争风险；管理层同时承诺Q4至少返还75%自由现金流。'],
 'risk_flags':['组件供应与全球贸易限制','Juniper并购整合及竞争','AI需求和交付预测风险','现金流资本配置/债务偿还'],
 'sources':[{'url':'https://investors.hpe.com/?releaseid=909541','title':'HPE Reports Fiscal 2026 Third Quarter Results','published_at':'2026-09-02','evidence':'实际阅读官方Q3业绩稿：Cloud & AI营收9.0B美元、同比+25.4%，服务器6.8B美元、+35.3%；发布稿列出组件可得性、贸易、需求、流动性/资本资源、Juniper整合及竞争风险，并给出现金返还承诺。'}]
},
'US.COHR':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：FY2026 Datacenter & Communications收入5.275B美元，同比增长约40%；Q4营收2.05B美元、同比+34%。管理层指出数据中心架构由铜转光及产能扩张机会。','优势：光子器件/制造规模与数据中心互连组合是具体供给优势；新增长平台正在爬坡而不是已完全成熟。','风险/资金：FY2026年报列出客户集中（两个客户分别约占20%、12%收入）及长期债务；公司称优先资本开支扩充制造能力，故需关注集中客户与投资回报。'],
 'risk_flags':['客户集中：两大客户占收入约20%和12%','扩产资本开支与长期债务偿付','AI光互连需求/代际转型风险'],
 'sources':[{'url':'https://ir.coherent.com/news-releases/news-release-details/coherent-corp-reports-fourth-quarter-and-full-year-fiscal-2026','title':'Coherent Q4 and FY2026 Results','published_at':'2026-08-12','evidence':'实际阅读官方业绩稿：FY26营收7.118B美元，Datacenter & Communications为5.275B美元；管理层称AI数据中心转向光连接、扩大产能以满足需求。'}, {'url':'https://ir.coherent.com/node/18116/xbrl-viewer','title':'Coherent FY2026 Form 10-K XBRL Viewer','published_at':'2026-08-14','evidence':'实际阅读官方申报页面：FY26两名主要客户约占收入20%与12%，收入主要来自Datacenter & Communications；页面列出长期债务本金偿还安排。'}]
},
'US.CIEN':{
 'risk_reviewed':True,'review_status':'reviewed',
 'reasons':['AI业务/商业化：FY2026 Q3营收1.671B美元（+37%）；CEO将增长与AI驱动的高速网络投资联系，公司的光网络系统/互连用于带宽扩容。','优势：公司披露其光系统、互连和网络自动化组合及长期客户部署经验；管理层称其高容量光系统专业能力形成竞争优势。','风险/资金：Q3两位客户各占总收入10%以上，合计占41.7%；公司列出AI改变网络支出、供应链、竞争、订单时点/履约及收入确认等风险。'],
 'risk_flags':['客户集中：两大客户合计41.7%收入','AI资本开支变化及光网络竞争','供应链/交期/订单履约与收入确认'],
 'sources':[{'url':'https://investor.ciena.com/news/news-details/2026/Ciena-Reports-Fiscal-Third-Quarter-2026-Financial-Results/default.aspx','title':'Ciena Fiscal Q3 2026 Financial Results','published_at':'2026-09-03','evidence':'实际阅读官方Q3业绩稿：收入1.671B美元、同比+37%；管理层联系AI与网络投资及高速连接方案。风险明确列出AI对网络开支影响、供应链、竞争、订单规模/时点及履约；两客户合计占收入41.7%。'}]
}}
records=[]
for code in want:
 r=dict(by[code]); m=meta[code]
 r['risk_reviewed']=m['risk_reviewed']; r['review_status']=m['review_status']; r['risk_flags']=m['risk_flags']
 r['reasons']=m['reasons']
 r['sources']=list(r.get('sources',[]))+m['sources']
 r['observed_at']='2026-10-07'
 r['agent']='codex / gpt-6-luna'
 r['review_priority']=2
 records.append(r)
obj={'records':records}
fd,tmp=tempfile.mkstemp(prefix='.reviews_core_other.',suffix='.tmp',dir=os.path.dirname(out_path))
with os.fdopen(fd,'w',encoding='utf-8') as f:
 json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out_path)
# validate
loaded=json.load(open(out_path,encoding='utf-8'))
assert len(loaded['records'])==10 and len({r['code'] for r in loaded['records']})==10
assert all(r.get('review_priority')==2 and r.get('agent')=='codex / gpt-6-luna' for r in loaded['records'])
assert all(all(s['published_at'] is None or s['published_at'] <= '2026-10-07' for s in r['sources']) for r in loaded['records'])
print(out_path, len(loaded['records']), 'reviewed', sum(r['review_status']=='reviewed' for r in loaded['records']), 'incomplete',sum(r['review_status']=='incomplete' for r in loaded['records']))
