"""Create cumulative official-source review records from this run."""
import json
from pathlib import Path

OUT = Path(__file__).with_name("review_luna.json")
OBSERVED = "2026-10-06"

def source(url, title, evidence):
    return {"url": url, "title": title, "observed_at": OBSERVED,
            "status": "luna_official_review", "evidence": evidence}

def rec(code, name, industry, primary, groups, tier, reasons, sources, status="official_review", confidence="medium", caveat=""):
    return {"code": "US." + code, "name": name, "industry": industry,
            "primary": primary, "groups": groups, "tier": tier,
            "reasons": reasons, "sources": sources, "review_status": status,
            "confidence": confidence, "caveat": caveat}

records = [
 rec("ACLS","Axcelis Technologies","半导体设备","design",["design"],1,
     ["官网确认 Purion 离子注入设备服务先进半导体制造工艺；AI 芯片扩产可经晶圆厂设备投入间接传导，未发现公司披露 AI 专用产品或 AI 收入。"],
     [source("https://www.axcelis.com/products/purion-ion-implantation-equipment/","Purion 离子注入设备","Purion 平台面向 10nm 及以下晶圆制造工艺，属于半导体制造设备。")],"official_review","medium","正分只表示有具体的先进制程设备链路；官网该资料未将设备与 AI 客户/收入绑定。"),
 rec("ACN","Accenture plc","IT 咨询与企业服务","data",["data"],2,
     ["公司提供生成式 AI 转型、企业数据基础与落地服务，属于帮助客户部署 AI 的服务商；不能据此推断服务收入或利润增量。"],
     [source("https://www.accenture.com/en/services/ai-data/generative-ai","Generative AI Technology Services","Accenture 明确提供企业生成式 AI 转型服务，覆盖数据基础、架构、用例与组织实施。")],"official_review","medium","服务业务广泛，官网介绍不能证明 AI 对整体营收的贡献。"),
 rec("AEHR","Aehr Test Systems","半导体测试设备","design",["design"],3,
     ["公司明确销售面向 AI 加速器、GPU 与 HPC 处理器的高功率测试/老化设备，属于 AI 半导体专用供给。"],
     [source("https://www.aehr.com/about-us/","Aehr Test Systems","公司介绍将 Sonoma 系列描述为服务 AI 加速器、GPU、HPC 处理器的高功率测试和可靠性解决方案。")],"official_review","high","产品关联明确，但资料本身不证明订单规模、持续性或 AI 收入占比。"),
 rec("AMBA","Ambarella","边缘 AI 半导体","compute",["compute","edge"],3,
     ["CVflow AI 加速器、CV7 视觉 SoC 与 X7 独立 AI 加速器面向边缘视觉、机器人、工业自动化等具体场景。"],
     [source("https://www.ambarella.com/news/ambarella-launches-x7-its-first-standalone-ai-accelerator-to-add-physical-ai-to-any-host-processor/","Ambarella Launches X7, Its First Standalone AI Accelerator","公司宣布 X7 为专用边缘 AI 协处理器，并列举实体安防、车辆安全、无人机与机器人等边缘 AI 用途。"),
      source("https://www.ambarella.com/products/aiot-industrial-robotics/","AIoT, Industrial & Robotics","官网说明 CVflow 实时执行神经网络，用于工业机器人、生产线缺陷识别、库存跟踪等。")],"official_review","high","专用产品证据充分；不等同于全部芯片收入来自 AI。"),
 rec("COHU","Cohu, Inc.","半导体测试与检测设备","design",["design"],3,
     ["公司 Eclipse 测试平台面向 AI 数据中心处理器及 GPU、定制 AI 加速器；PAICe 提供半导体制造 AI 检测/分析软件。"],
     [source("https://www.cohu.com/eclipse","Eclipse PnP Handler","产品页列明支持 AI 数据中心设备，包括 GPU、CPU、定制 AI 加速器和 ASIC。"),
      source("https://www.cohu.com/paice/inspection","PAICe Inspection","PAICe Inspection 将机器学习和深度学习用于半导体制造检测流程。")],"official_review","high","同时包含 AI 计算芯片测试需求和公司内部/制造软件产品；未量化相关收入。"),
 rec("DIOD","Diodes Incorporated","模拟与电源半导体","power",["power","connect"],2,
     ["公司列出面向 AI 数据中心、AI 服务器与网络设备的电源管理、宽禁带功率器件和高速连接产品，属于部署配套。"],
     [source("https://www.diodes.com/solutions/data-center-solutions","Data Center Solutions","官网称其电源管理器件和高速链路产品服务云与 AI 平台的服务器、网络和存储应用。"),
      source("https://www.diodes.com/about/company/company-profile","Company Profile","公司将 AI 数据中心、AI 服务器、存储和边缘 AI 列为计算市场终端应用。")],"official_review","medium","公司产品组合覆盖大量非 AI 终端，AI 终端列示不能直接代表实质收入比例。"),
 rec("ESTC","Elastic N.V.","企业搜索与数据软件","data",["data"],3,
     ["Elastic Search AI Platform 提供向量检索、语义搜索、RAG 与生成式 AI 应用构建能力，属于明确的 AI 数据软件产品。"],
     [source("https://www.elastic.co/products","The Search AI Platform","官方产品页明确列出向量数据库、RAG、生成式 AI 应用构建与 AI 搜索能力。")],"official_review","high","产品能力明确；仍需区分产品采用与付费收入/净增量。"),
 rec("FLEX","Flex Ltd.","电子制造与数据中心基础设施","systems",["systems","power"],3,
     ["公司公布 AI 基础设施平台，集成机柜/计算系统、电力分配和液冷，面向 AI/HPC 数据中心部署。"],
     [source("https://flex.com/resources/flex-ai-infrastructure-platform","Flex AI infrastructure platform","官方方案将电力、冷却、计算集成于面向 AI/HPC 的预制模块化数据中心设计。"),
      source("https://flex.com/industries/data-center","Data Center Power, Infrastructure, and Cooling Solutions","官网列出 AI 数据中心电源、机柜、计算制造与液冷产品组合。")],"official_review","high","产品供给路径直接，但平台发布不证明收入规模、利润率或后续分拆后的证券暴露。"),
 rec("GNRC","Generac Holdings","发电、储能与备用电源","energy",["energy","power"],2,
     ["公司与 EPC Power 宣布面向数据中心部署电池储能、控制器和逆变器解决方案，资料明确提及 AI 数据中心基础设施。"],
     [source("https://www.generac.com/about/news/generac-and-epc-power-to-deploy-fully-integrated-energy-solutions-for-data-center-applications/","Generac and EPC Power to Deploy Fully Integrated Energy Solutions","公司公告合作提供能源系统，使用电池系统、控制器、逆变器服务数据中心市场，明确面向 AI 数据中心开发者/运营方。")],"official_review","medium","属于能源与电力配套；公告称 engagement/deployment，不足以推导规模化订单或收入占比。"),
 rec("HUBB","Hubbell Incorporated","电气设备与电网基础设施","power",["power","energy"],1,
     ["公司官网将数据中心与电力公用事业列为服务市场，可建立建筑配电/电网基础设施的间接路径；未找到 AI 专用产品或 AI 客户证据。"],
     [source("https://www.hubbell.com/hubbell/en/solutions","Solutions","官网将 Data Center 和 Power & Utility 列为其解决方案覆盖的市场。")],"official_review","low","这是一般数据中心及电网供给路径，不是 AI 专属需求证据；收入暴露未核实。"),
]
records += [
 rec("JBL","Jabil Inc.","电子制造与数据中心系统集成","systems",["systems","power","optics"],3,
     ["公司明确提供 AI/HPC 服务器、机柜级系统集成、液冷、电力分配与数据中心光子互连制造服务。"],
     [source("https://jabil.com/industries/data-center/data-center-infrastructure.html","Data Center Infrastructure","Jabil 明确列出 AI/HPC 服务器设计制造、机柜系统、液冷与数据中心供电集成。"), source("https://investors.jabil.com/news/news-details/2025/Jabil-Launches-J-422G-Servers-for-Scalable-AI-and-Data-Center-Performance/default.aspx","Jabil Launches J-422G Servers","Jabil 宣布面向 AI、机器学习、LLM、HPC 负载的服务器产品。")],"official_review","high","业务同时包含大规模通用制造，产品关联不代表 AI 业务收入占比。"),
 rec("KLIC","Kulicke & Soffa Industries","半导体封装设备","design",["design"],2,
     ["公司公告先进封装设备获得支持 AI 长期机会的订单；具体订单匿名且 AI 收入规模未量化，故定位为封装设备配套。"],
     [source("https://investor.kns.com/2023-11-15-Kulicke-Soffa-Announces-Multiple-Advanced-Packaging-Orders-Supporting-Long-Term-AI-Opportunities?asPDF=1","Multiple Advanced Packaging Orders Supporting Long-Term AI Opportunities","发行人公告称先进封装订单支持长期 AI 机会，客户包含领先晶圆代工、IDM 与 OSAT。")],"official_review","medium","订单公告较早且客户/产品收入未披露，不能证明当前 AI 收入规模。"),
 rec("ON","onsemi","功率与传感半导体","power",["power","compute"],3,
     ["公司提供覆盖电网至 GPU 的 AI 数据中心电源器件，并在 NVIDIA MGX 生态提供 AI 数据中心先进电源系统。"],
     [source("https://www.onsemi.com/solutions/computing/data-center","Data Center","onsemi 列出面向 AI 数据中心的 SiC、GaN、MOSFET 与服务器电源链路产品。"), source("https://www.onsemi.com/company/newsroom/news-and-insights/how-onsemi-is-powering-the-next-generation-of-ai-factories","How onsemi Is Powering the Next Generation of AI Factories","公司称参与 NVIDIA MGX 生态，为新一代 AI 数据中心和加速计算平台提供电源技术。")],"official_review","high","面向AI场景产品/生态证据明确，仍需另行验证收入贡献。"),
 rec("POWI","Power Integrations","高压功率半导体","power",["power"],3,
     ["PowiGaN 高压功率器件与 InnoMux2-EP 参考设计明确适配 800V AI 数据中心架构，属于专用供电芯片配套。"],
     [source("https://www.power.com/resources/green-room/blog/1250-v-1750-v-gan-solution-addresses-need-800-v-bus-architectures-power-hungry-ai-data-centers?language=zh-hant","GaN Solution for 800 V AI Data Centers","公司说明 1250V/1700V PowiGaN 器件和 InnoMux2-EP 用于 800V AI 数据中心电源转换。")],"official_review","high","应用适配明确，但参考设计/技术能力不等于商业部署或收入规模。"),
 rec("ROK","Rockwell Automation","工业自动化与控制软件","edge",["edge","data"],3,
     ["公司工业 AI 产品组合包括生成式 AI PLC 编程助手、机器视觉缺陷检测和机器学习过程优化软件，明确用于工厂流程。"],
     [source("https://www.rockwellautomation.com/en-us/future-trends-industrial-operations/industrial-ai.html","Industrial AI","Rockwell 列明 FactoryTalk Design Studio Copilot、LogixAI 和 VisionAI 等工业 AI 产品及具体制造用途。")],"official_review","high","AI产品证据充分；未证明独立 AI 收入或盈利占比。"),
 rec("RXRX","Recursion Pharmaceuticals","AI 药物发现与生物科技","health",["health"],3,
     ["公司以 Recursion OS 和自动化实验/机器学习平台进行药物发现，官网展示由平台推进的临床与临床前药物管线。"],
     [source("https://recursion.com/","Pioneering AI Drug Discovery","Recursion 明确说明用细胞图像训练 AI，平台连接自动化实验、数据与机器学习以发现药物。"), source("https://www.recursion.com/pipeline","AI-driven drug discovery pipeline","官网列出由 Recursion OS 支持的候选药物及临床开发阶段。")],"official_review","high","AI平台与药物管线真实存在；研发成功、获批与商业化仍是独立风险。"),
 rec("QBTS","D-Wave Quantum","量子计算","adjacent",["adjacent"],None,
     ["已核查官方检索结果，但当前未获得可核实的公司资料证明其量子计算产品形成 AI 价值链的直接供给或具规模的 AI 商业路径。"],[],"unresolved","low","主题相邻不构成 AI 关联；保留未知，未据此给零或正分。"),
 rec("RGTI","Rigetti Computing","量子计算","adjacent",["adjacent"],None,
     ["已核查官方检索结果，但当前未找到可核实的发行人资料建立 AI 产品/收入与量子计算业务的具体链路。"],[],"unresolved","low","量子与机器学习概念相邻不足以证明本公司 AI 暴露；保留未知。"),
 rec("NOVT","Novanta Inc.","精密运动与医疗设备技术","edge",["edge","health"],None,
     ["已核查官方检索结果；尚未找到发行人资料将其精密运动/医疗技术具体关联到 AI 产品、AI 专属客户或收入。"],[],"unresolved","low","机器人或医疗设备可能使用 AI 属于通用推测，不作评级。"),
 rec("PLAB","Photronics, Inc.","半导体光掩模","design",["design"],None,
     ["已核查官方检索结果；尚未找到发行人资料将光掩模业务与 AI 芯片客户或专用需求直接关联。"],[],"unresolved","low","先进芯片制造可能需要光掩模属于行业推演；缺少公司特定 AI 证据。"),
]
records += [
 rec("IBM","International Business Machines","企业软件、AI平台与IT服务","data",["data","cloud"],3,
     ["IBM watsonx 提供企业级生成式/预测式 AI 模型开发、部署、数据和治理产品，且有可购买的云服务。"],
     [source("https://www.ibm.com/products/watsonx","IBM watsonx","IBM 官方产品页介绍 watsonx.ai、watsonx.data 与 AI 生命周期工具。"), source("https://www.ibm.com/products/watsonx-ai","IBM watsonx.ai","产品页明确支持开发、运行与部署企业 AI 应用和模型。")],"official_review","high","AI 是产品组合之一；不能推导 AI 对 IBM 总收入的占比。"),
 rec("SMCI","Super Micro Computer","服务器与 AI 数据中心系统","systems",["systems","power"],3,
     ["公司提供生成式 AI 超级集群、GPU 服务器、整柜集成和液冷方案，直接面向 AI 训练与推理部署。"],
     [source("https://www.supermicro.com/en/solutions/ai-supercluster","Generative AI SuperCluster","官方方案列明端到端 AI 数据中心、GPU集群、机柜集成与液冷产品。"), source("https://www.supermicro.com/en/accelerators/nvidia/pcie-gpu","Supermicro NVIDIA PCIe GPU Systems","官网列出面向企业 AI 工作负载的 GPU 系统与部署方案。")],"official_review","high","产品直接匹配 AI 部署；需单独评估客户集中、盈利与营运资本，本文不作推断。"),
 rec("SYM","Symbotic Inc.","仓储机器人与自动化","edge",["edge"],3,
     ["Symbotic 的仓储机器人和软件系统使用模式识别及 AI 增强软件进行库存与物流自动化，属于具体机器人应用。"],
     [source("https://www.symbotic.com/solutions/robots/","Symbotic Warehouse Robotics","官网描述机器人仓储系统与 AI 支持的模式识别、持续改进功能。"), source("https://www.symbotic.com/","Warehouse Automation","公司将其端到端仓储自动化解决方案描述为由 AI 增强软件驱动。")],"official_review","high","AI是自动化系统组成；需以客户项目及合同兑现评估商业规模。"),
 rec("SYNA","Synaptics Incorporated","边缘 AI 芯片","compute",["compute","edge"],3,
     ["Astra 系列含专用 NPU 和多模态边缘 AI 处理器，面向 IoT、工业视觉、机器人和智能设备。"],
     [source("https://www.synaptics.com/products/embedded-processors","Embedded Processors","官方产品页列出 AI-native Astra 处理器、集成 NPU 与工业/IoT边缘推理用途。"), source("https://www.synaptics.com/company/news/synaptics-launches-next-generation-astra-multimodal-genai-processors-to-power-future-intelligent-iot-edge","Astra Multimodal GenAI Processors","公司公告说明 SL2600 系列为多模态 Edge AI 处理器，应用包含工厂自动化、医疗设备、零售和机器人。")],"official_review","high","明确硬件产品关联；实际收入和产品采用程度未在这些资料中量化。"),
 rec("STM","STMicroelectronics","半导体与边缘 AI 芯片","compute",["compute","edge"],3,
     ["STM32N6 集成 Neural-ART NPU，并有边缘 AI 开发工具，用于设备端视觉和音频推理。"],
     [source("https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html","STM32N6 series","ST 官方产品资料说明 Neural-ART NPU 面向低功耗边缘 AI，支持实时视觉/音频神经网络推理。"), source("https://www.st.com/en/development-tools/stedgeai-dc.html","ST Edge AI Developer Cloud","发行人开发平台用于优化、编译并部署 AI 模型至其 MCU、MPU 和传感器。")],"official_review","high","产品证据直接，但 ST 广泛服务汽车、工业和消费市场。"),
 rec("SLAB","Silicon Laboratories","无线物联网与边缘 AI 芯片","edge",["edge","compute"],3,
     ["公司无线 SoC 集成 AI/ML 硬件加速器及 IoT 推理工具，列举预测性维护、语音/声学识别与智能设备应用。"],
     [source("https://www.silabs.com/applications/artificial-intelligence-machine-learning?tab=software","Machine Learning in IoT","官方资料列出 SoC 内置矩阵向量处理器及边缘推理，适用于智能家居和工业自动化等 IoT 设备。")],"official_review","high","面向低功耗 IoT ML 的具体芯片路径；不等于通用训练算力或主要收入来源。"),
 rec("TSEM","Tower Semiconductor","半导体特色工艺代工","design",["design"],None,
     ["已核查官方检索结果；尚未找到发行人资料将其特色工艺/代工业务与 AI 专用产品、AI 客户或 AI 收入直接关联。"],[],"unresolved","low","仅凭芯片制造环节可能服务 AI 属于普遍行业推断，缺公司级证据，保留未知。"),
]
records += [
 rec("DT","Dynatrace, Inc.","企业软件与智能运维","data",["data"],3,
     ["Davis AI 将预测、因果与生成式 AI 用于可观测性、安全、根因分析和自动化运维，是明确嵌入产品的平台能力。"],
     [source("https://www.dynatrace.com/news/press-release/dynatrace-advances-aiops-with-preventive-operations/","Dynatrace Advances AIOps with Preventive Operations","公司公告介绍 Davis AI 预测并预防 IT 事件，扩展自动化问题调查与修复能力。"), source("https://www.dynatrace.com/news/blog/hypermodal-ai-dynatrace-expands-davis-ai-with-davis-copilot/","Davis AI and Davis CoPilot","官网说明平台融合预测、因果和生成式 AI，用于可观测性与安全。")],"official_review","high","产品AI能力明确；未据此推断AI收入占比或用户付费增量。"),
 rec("ISRG","Intuitive Surgical","手术机器人与数字医疗","health",["health","edge"],2,
     ["da Vinci 5 的 Case Insights 已使用 AI 分析手术系统数据、运动学和视频，提供术者技能与流程洞察；公司同时公开后续 AI 开发路线。"],
     [source("https://www.intuitive.com/en-gb/products-and-services/da-vinci/5","da Vinci 5","产品页称 Case Insights 使用 AI 评估系统数据、运动学与视频，为手术团队提供客观洞察。"), source("https://isrg.intuitive.com/news-releases/news-release-details/intuitive-shares-vision-ai-enabled-future-healthcare","AI-Enabled Future of Healthcare","发行人介绍基于逾两千万例手术数据的 AI 开发框架和术中辅助路线。")],"official_review","medium","产品中AI分析功能已有明确证据，更多智能手术能力属于发展路线；手术由医生控制。"),
]
records = [r for r in records if r["code"] not in {"US.DT", "US.ISRG"}]
OUT.write_text(json.dumps({"agent":"codex / gpt-6-luna", "records":records},ensure_ascii=False,indent=2)+"\n")
print(f"wrote {len(records)} records to {OUT}")
