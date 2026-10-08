"""Versioned, overlapping business labels; ordinal relevance is analyst inference.

3 = supplies AI computing / AI is a named product mechanism; 2 = infrastructure
or software enabling deployment; 1 = indirect demand/efficiency hypothesis;
0 = reviewed main business has no established AI transmission in this version.
Unreviewed is None, never zero. Presence does not measure AI revenue share.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [
 ("design", "设计、制造与封装", 0, "设计 IP / EDA → 晶圆设备 → 制造与封装", "AI 芯片复杂度和产能扩张，传导到设计与制造投入；订单兑现有周期。"),
 ("compute", "计算芯片与加速器", 1, "GPU / CPU / 定制 ASIC", "训练与推理算力需求 → 芯片采购；价格、竞争与客户自研改变价值分配。"),
 ("memory", "存储与数据保存", 1, "HBM / DRAM / NAND / HDD", "模型和数据规模扩大 → 内存带宽与数据存储需求。HBM 与硬盘应分别评价。"),
 ("connect", "电互连与交换网络", 1, "AEC / PCIe / Ethernet / 交换芯片", "多芯片与多机柜计算 → 更高带宽、更低延迟的连接。"),
 ("optics", "光通信与光子器件", 1, "光模块 / 激光器 / 光纤 / 光交换", "集群内外流量扩大 → 光连接升级；技术路线和客户集中影响传导。"),
 ("systems", "服务器与机柜集成", 2, "服务器 / 系统制造 / 机柜", "芯片采购转化为可部署的计算系统；收入增长还要看利润和营运资金。"),
 ("power", "供配电、散热与备电", 2, "电源 / 配电 / 液冷 / UPS", "机柜功率密度升高 → 电力和热管理需求。"),
 ("energy", "发电、电网与材料", 2, "电源供给 / 电网工程 / 铜 / 核燃料", "数据中心用电扩张 → 电网和能源投入。传导更远，不能把所有能源股视为 AI 股。"),
 ("cloud", "云平台与算力服务", 3, "云平台 / GPU 云 / 推理服务", "算力投入 → 云服务收入；利用率、定价和折旧决定能否创造价值。"),
 ("data", "企业软件、数据与安全", 3, "数据平台 / 业务软件 / 安全 / 可观测性", "AI 落地需要数据与业务流程；新增产品、付费和留存比产品口号更重要。"),
 ("applications", "广告、内容与商业应用", 4, "推荐 / 广告 / 创作 / 商业平台", "AI 改善效果、供给或效率 → 收入和利润；同时存在被替代和获客成本风险。"),
 ("health", "医疗、生物与 AI 赋能", 4, "诊断 / 药物设计 / 生命科学工具", "数据与模型 → 研发或诊断效率；科学验证、监管与商业化是独立门槛。"),
 ("edge", "端侧、机器人与自动驾驶", 4, "端侧推理 / 感知 / 自动化", "模型能力 → 设备、自动化和服务；量产、成本与实际采用决定兑现。"),
 ("adjacent", "邻近主题与间接叙事", 5, "量子 / 加密资产 / 太空等", "主题相邻不自动产生 AI 订单；需要逐公司证实传导路径。"),
 ("other", "当前缺少明确 AI 路径", 5, "主要业务已识别，AI 增量未证实", "保留在完整池中；本版未发现明确链路不代表公司完全不使用 AI。"),
 ("pending", "业务与 AI 暴露待核实", 5, "未完成逐公司证据整理", "待核实不能记成零关联；ETF 来源只提示核查方向。"),
]


def labels():
    prior = json.loads((ROOT / "research/group_expectation_matrix/config/industry_tags_v3.json").read_text())
    old_sources = {}
    for g in prior["groups"]:
        for s in g.get("sources", []):
            for symbol in s["supports"]:
                old_sources.setdefault(symbol, []).append(dict(url=s["url"], title=s["title"],
                       observed_at=s["observed_at"], status="prior_official_snapshot"))
    result = {}

    def add(symbols, group, tier, reason, urls=None, confidence="business_label"):
        for symbol in symbols.split():
            sources = old_sources.get(symbol, [])[:2]
            if urls:
                sources = [dict(url=u, title=symbol + " 官方业务资料", observed_at="2026-10-06",
                                status="reviewed_this_run") for u in urls]
            row = result.setdefault(symbol, dict(primary=group, groups=[], tier=tier, reasons=[],
                                                  sources=[], confidence=confidence))
            if row["confidence"] == "proposed":
                row["reasons"] = []
                row["confidence"] = confidence
            if group not in row["groups"]:
                row["groups"].append(group)
            row["tier"] = max(v for v in [row["tier"], tier] if v is not None)
            row["reasons"].append(reason)
            for source in sources:
                if source["url"] not in [s["url"] for s in row["sources"]]:
                    row["sources"].append(source)

    add("NVDA AMD AVGO MRVL", "compute", 3, "有面向加速计算、定制计算或 AI 数据中心的芯片业务。")
    add("ARM", "compute", 2, "计算架构与 IP，覆盖服务器与端侧；不是所有授权收入都来自 AI。")
    add("INTC QCOM", "compute", 2, "通用计算和端侧推理可承接 AI 需求，传统业务仍占重要位置。")
    add("CBRS", "compute", 3, "晶圆级 AI 加速计算与推理服务。", ["https://www.cerebras.ai/"])
    add("MU", "memory", 3, "HBM 直接用于 AI 加速计算；DRAM/NAND 还有其他需求。", ["https://www.micron.com/products/memory/hbm"])
    add("SNDK STX WDC", "memory", 2, "数据保存与容量扩张，和 HBM 的价值环节不同。")
    add("ALAB", "connect", 3, "面向机柜级 AI 的连接产品。", ["https://www.asteralabs.com/"])
    add("CRDO", "connect", 3, "高速、低功耗连接产品，服务数据基础设施。", ["https://credosemi.com/"])
    add("AVGO MRVL NVDA", "connect", 3, "除计算外还涉及网络连接；多标签不重复计算证券总数。")
    add("ANET", "connect", 3, "AI 集群以太网与网络系统。", ["https://www.arista.com/en/solutions/ai-networking"])
    add("CSCO", "connect", 2, "数据中心和企业网络，受 AI 部署需求影响。")
    add("COHR", "optics", 3, "AI 数据中心的器件、光模块与光互连组合。", ["https://www.coherent.com/communications/datacom/datacenter"])
    add("LITE", "optics", 3, "AI 基础设施用激光器、外置光源与光交换。", ["https://www.lumentum.com/en/solutions/ai-infrastructure"])
    add("CIEN", "optics", 3, "数据中心之间及新型 AI 架构的光连接；不等同于光模块厂。", ["https://www.ciena.com/about/newsroom/press-releases/lumen-selects-ciena-optical-tech-to-support-ai-workloads"])
    add("POET", "optics", 3, "光子集成平台面向 AI 光网络；产品关联不证明规模化收入。", ["https://www.poet-technologies.com/technology"])
    add("SMTC", "optics", 3, "高速光接收技术服务 AI 网络。", ["https://www.poet-technologies.com/news/poet-technologies-and-semtech-launch-1-6t-optical-receivers-for-ai-networks?hl=en-CA"])
    add("FN", "optics", 2, "光学封装与精密制造，位于光互连制造链。", ["https://investor.fabrinet.com/node/13636/pdf"])
    add("HK.06088", "optics", 2, "高速光收发模块与光子器件合作，仍需检验订单规模。", ["https://www.poet-technologies.com/news/2024-may-14"])
    # Additional named businesses are research hypotheses until their evidence is
    # reviewed. Their ordinal level is withheld by the build, not invented.
    proposed = {
      "optics": "AAOI AXTI MTSI GLW NOK MXL", "memory": "SIMO RMBS SKHY",
      "design": "AMAT ASML KLAC LRCX TER ENTG ONTO CAMT ACLS FORM AEHR TSM ASX AMKR TSEM SNPS CDNS JP.4186 JP.4063 HK.00522 HK.00981",
      "systems": "DELL SMCI HPE CLS JBL FLEX", "power": "MPWR NVTS AOSL POWI VICR ENS MOD NVT HUBB",
      "energy": "CEG AEP EXC XEL VST NRG FLNC NXT CCJ FCX FRVO PWR GNRC",
      "cloud": "ORCL HUT WYFI", "data": "CRM NOW SNOW MDB CFLT ESTC AI RBRK OKTA DT ACN IBM",
      "applications": "NFLX RBLX SPOT RDDT TTD WMT", "health": "TEM SDGR RXRX TWST TXG VCYT LLY MRK",
      "edge": "AAPL TSLA ISRG OUST PONY SYM HON ROK ABB ADI TXN NXPI MCHP ON STM",
      "adjacent": "IONQ RGTI QBTS MSTR RKLB UMAC SPCX",
    }
    for group, symbols in proposed.items():
        for symbol in symbols.split():
            if symbol not in result:
                result[symbol] = dict(primary=group, groups=[group], tier=None,
                    reasons=["已提出业务分类，AI 关联等级尚未完成逐公司核查。"],
                    sources=old_sources.get(symbol, [])[:2], confidence="proposed")
    # Old official sources verify these business locations, but only an indirect
    # demand path is inferred. No future-performance label was consulted.
    for symbol in "AMAT ASML KLAC LRCX TER SNPS CDNS".split():
        result[symbol]["tier"] = 2
        result[symbol]["confidence"] = "business_label"
        result[symbol]["reasons"] = ["官方资料支持芯片设计/制造/测试环节；AI 需求经客户投资间接传导。"]
    for symbol in "ADI TXN NXPI MCHP MPWR".split():
        result[symbol]["tier"] = 1
        result[symbol]["confidence"] = "indirect_hypothesis"
        result[symbol]["reasons"] = ["通用模拟、嵌入式或电源器件，AI 增量只是具体应用中的一部分。"]
    add("VRT", "power", 3, "AI / HPC 数据中心的供电与热管理。", ["https://www.vertiv.com/en-us/solutions/ai-hub/"])
    add("ETN", "power", 2, "数据中心配电与电力管理，有其他工业与商业终端。", ["https://www.eaton.com/us/en-us/markets/data-centers.html"])
    add("NVT", "power", 2, "数据中心基础设施与冷却方案。", ["https://www.nvent.com/en-us/data-solutions"])
    add("AEIS", "power", 2, "数据中心电源，AI 机柜功率升级是需求路径。", ["https://ir.advancedenergy.com/media/document/c218595c-37f2-4921-94f2-b229d641c157/assets/2026-06_AEIS_IR_Presentation.pdf?disposition=inline"])
    add("ENS", "power", 1, "储能和备电支持数字基础设施；AI 增量收入仍待证实。", ["https://www.enersys.com/en/"])
    add("GEV", "energy", 1, "发电与电网设备，数据中心只是潜在需求来源之一。", ["https://www.gevernova.com/"])
    add("BE", "energy", 2, "现场供电可支持数据中心建设与能源保障。", ["https://www.bloomenergy.com/"])
    add("PWR", "energy", 1, "电网建设承接更广泛电力投资；AI 传导路径较远。", ["https://www.quantaservices.com/"])
    add("HPE", "systems", 3, "AI 计算、网络与部署系统。", ["https://www.hpe.com/us/en/solutions/ai-artificial-intelligence.html"])
    add("CLS", "systems", 2, "通信和数据中心系统制造与工程。", ["https://www.celestica.com/our-expertise/markets/communications"])
    add("MSFT AMZN GOOG GOOGL", "cloud", 3, "云与 AI 平台，同时有其他大型业务。")
    add("CRWV NBIS", "cloud", 3, "面向 AI 的云计算平台。")
    add("PLTR", "data", 3, "AIP 将生成式 AI 连接到企业业务操作。", ["https://www.palantir.com/platforms/aip"])
    add("CRWD PANW FTNT DDOG", "data", 2, "安全与可观测性是数字系统部署基础；不等于 AI 专属收入。")
    add("ADBE ADSK INTU WDAY SHOP", "data", 2, "专业或企业软件是 AI 赋能落地的载体；增量付费须另外验证。")
    add("APP", "applications", 3, "Axon 是其广告产品的 AI 机制；关联不等于股价涨幅预测。", ["https://legal.applovin.com/about-applovins-axon-ai/"])
    add("META GOOG GOOGL AMZN", "applications", 2, "广告与内容商业化可通过 AI 效率获得传导，同时承担模型与基础设施成本。")
    add("SHOP MELI", "applications", 1, "商业平台可获得效率改善，需证明 AI 带来的额外收入或利润。")
    add("TEM", "health", 3, "AI 与多模态数据支持精准医疗和研发。", ["https://www.tempus.com/life-sciences/"])
    add("AAOI", "optics", 2, "数据中心光收发模块与激光器，AI 是潜在需求来源而非全部业务。", ["https://ao-inc.com/"])
    add("GLW", "optics", 2, "光纤与连接系统服务数据中心和通信网络。", ["https://www.corning.com/optical-communications/worldwide/en/home.html"])
    add("MTSI", "optics", 2, "高速通信与光网络半导体，覆盖多类终端。", ["https://www.macom.com/"])
    add("RMBS", "memory", 2, "内存接口和高速接口 IP，AI 传导经内存带宽需求。", ["https://www.rambus.com/"])
    add("TSM", "design", 2, "半导体制造承接计算芯片需求，仍需区分 AI 与其他终端。", ["https://www.tsmc.com/english"])
    add("ORCL", "cloud", 3, "AI 基础设施、GPU 云与 AI 数据及应用服务。", ["https://www.oracle.com/artificial-intelligence/"])
    add("ORCL", "data", 3, "数据库、数据平台和企业应用同时承载 AI 部署。", ["https://www.oracle.com/artificial-intelligence/"])
    add("SDGR", "health", 2, "分子设计和计算软件；AI 与物理方法共同服务研发。", ["https://www.schrodinger.com/"])
    add("EQIX", "cloud", 2, "数据中心、托管与互联支持 AI 部署；OpenD 的 ETF 类型已单独纠正。", ["https://www.equinix.com/"])
    add("CCI", "other", 0, "当前官网强调通信塔与无线基础设施，未建立明确的 AI 增量链路。", ["https://www.crowncastle.com/"])
    add("TWST", "health", 1, "DNA 合成工具可支撑研发，AI 增量收入尚未证实。", ["https://www.twistbioscience.com/"])
    add("TXG", "health", 1, "生命科学数据与实验工具可支撑 AI 研发，增量收入尚未证实。", ["https://www.10xgenomics.com/"])
    add("IONQ", "adjacent", 1, "量子计算是独立技术路径，AI 合作或算力叙事不等于 GPU 供给链。", ["https://ionq.com/"])
    add("CRM", "data", 3, "AI 产品把客户数据、工作流与业务操作连接起来。", ["https://www.salesforce.com/artificial-intelligence/"])
    add("SNOW", "data", 3, "Cortex AI 与企业 AI / ML 数据服务。", ["https://www.snowflake.com/en/product/ai/"])
    add("AI", "data", 3, "企业 AI 应用与平台是明确的产品方向。", ["https://c3.ai/"])
    add("NET", "data", 2, "Workers AI 提供边缘推理，同时还有广泛网络与安全业务。", ["https://www.cloudflare.com/products/workers-ai/"])
    add("NXT", "energy", 1, "公用事业规模太阳能系统；AI 经用电扩张间接传导，不能视为 AI 专属供给。", ["https://nextpower.com/"])
    add("MXL", "optics", 2, "通信及连接半导体，面向数据通信与其他终端。", ["https://www.maxlinear.com/"])
    add("NVTS", "power", 3, "GaN / SiC 面向 AI 数据中心电源及 800V 架构；合作不等于收入兑现。", ["https://ir.navitassemi.com/node/10511/pdf"])
    add("NOW", "data", 3, "Now Assist 将生成式 AI 连接到企业工作流和服务。", ["https://www.servicenow.com/docs/r/intelligent-experiences/platform-now-assist-landing.html?contentId=Jbm~HZMzOwC6m64gv1BGzQ"])
    add("HK.00100", "applications", 3, "多模态基础模型与 AI 产品服务。", ["https://www.minimax.cn/"])
    # Broad QQQ consumer businesses remain reviewed as a separate category;
    # this means no documented AI incremental path in this limited taxonomy.
    for symbol in "COST PEP MNST MDLZ KDP SBUX CCEP ROST ORLY CSX ODFL MAR BKNG ABNB CTAS CPRT FAST PCAR FANG".split():
        result[symbol] = dict(primary="other", groups=["other"], tier=None,
          reasons=["当前主要业务不是 AI 供给链；未核查 AI 效率增量，暂不设零分。"], sources=[], confidence="proposed")
    return result
