# 行业与价值链多标签研究记录

本次观察时间：`2026-09-25T14:42:56.483361Z`。分类版本：`industry_value_chain_v1`。

本版是依据官方公司、产品、投资者资料制作的研究标签，不是标准行业分类。覆盖候选池 101 个证券中的 45 个，形成 11 个可重叠群、61 条成员关系；其余 56 个明确列为 `unclassified_symbols`。标签建立过程没有使用目标达成结果。

## 群定义与覆盖

| 群 | 价值链位置 | 证券 |
|---|---|---|
| 计算与通信芯片 | 计算与连接器件 | ALAB, AMD, ARM, AVGO, INTC, MRVL, NVDA, QCOM |
| 模拟、嵌入式与电源芯片 | 信号感知、控制与供电器件 | ADI, MCHP, MPWR, NXPI, TXN |
| 半导体设备与测试 | 芯片制造的设备与测试环节 | AMAT, ASML, KLAC, LRCX, TER |
| EDA 与半导体设计 IP | 芯片设计工具与知识产权 | ARM, CDNS, SNPS |
| 存储器与数据存储 | 数据保存器件与系统 | MU, SNDK, STX, WDC |
| 网络与高速互连 | 计算节点之间的连接 | ALAB, AVGO, CSCO, LITE, MRVL, NVDA |
| 云基础设施平台 | 云平台与算力服务 | AMZN, CRWV, GOOG, GOOGL, MSFT, NBIS |
| 网络与云安全 | 数字系统的安全保障 | AVGO, CRWD, CSCO, DDOG, FTNT, MSFT, PANW |
| 企业应用与开发软件 | 企业和专业应用层 | ADBE, ADSK, AVGO, DDOG, INTU, MSFT, PLTR, SHOP, WDAY |
| 数字商业平台 | 交易与商家应用层 | AMZN, MELI, SHOP |
| 数字广告平台 | 流量分发与商业变现 | AMZN, APP, GOOG, GOOGL, META |

## 从业务结构得到的洞察

1. **一个行业标签不足以表示共同暴露。** 例如 Broadcom 同时涉及半导体、网络、安全与基础软件；Amazon 同时涉及云、商业与广告；Arm 同时涉及计算平台与设计 IP。保留多标签比强制唯一行业归属更接近业务事实。具体依据分别来自 [Broadcom 公司资料](https://www.broadcom.com/company/about-us)、[Amazon 公司资料](https://www.aboutamazon.com/about-us) 与 [Arm 计算平台](https://www.arm.com/company/arm-compute-platform)。
2. **价值链相邻并不意味着股价同向。** 云平台的采购支出可能是芯片、设备、存储和互连的需求来源，但也可能同时压低平台自身的自由现金流。该推论只是可检验的研究假设，本次资料核验没有证明采购关系的强度、领先滞后或价格因果性。
3. **宽群内部仍会混入不同机制。** 存储器群混合 DRAM、NAND 与硬盘，企业软件群混合订阅应用、专业工具与平台。建议把本版作为上层标签，后续在样本足够时追加子群，并比较子群的增量解释力。不能因为本版群内仍有差异就否定行业分组。
4. **重叠群不能当作独立策略样本相加。** 同一只股票、同一天可以进入多个格子；行业群之间的达成率差异有共享观测。比较时宜采用相同日期窗口和发行人聚类，检验时按日期块抽样，并同时报告成员重叠。
5. **群的共性与目标适配需要分开验证。** 本版只证明可审核的业务共性，不证明对任何收益/期限/回撤目标有预测增量。后续应比较群自身的历史基准率、加入波动与趋势后的增量，以及在新时间段中的稳定性。

## 时间与版本边界

- `taxonomy_version` 标识分类定义；每周的运行版本标识当周生成的成员快照，两者不等价。
- 官方网站多为持续更新页面，本次 `observed_at` 与 `source_available_at` 只表示本次取得资料的时间。没有给出可可靠确认的历史发布日期时，不推断其早于任何研究样本。
- 历史矩阵必须保留 `historical_membership=current_snapshot_retrospective`。这可以用于描述性回看和提出假设，不能作为已经通过历史时点可知性审查的训练特征。
- 周运行若只是复用同一来源文件，应保留原始 `observed_at`，并声明来源未重新核验。要建立严格历史时点版本，需要保存当时的披露文件、公告日期、可获取日期和逐次成员变更。
- 业务标签有意采用二元存在关系；本版没有收入占比、供应商权重、利润弹性和事件暴露权重。不能由标签数量推断投资权重。
- GOOG 与 GOOGL 同属 Alphabet，按证券计数与按发行人计数应分别披露。候选池来源本身也采用当前成分回溯，无法消除幸存者和成分选择偏差。
- Sandisk 已在 2025-02-24 从 Western Digital 分拆；本版分别记录 SNDK 和 WDC。历史身份不能简单合并回填。依据：[Sandisk 官方分拆说明](https://www.sandisk.com/sandisk-separation-faqs)。

## 可审核官方来源

以下每条在 JSON 中都有 `supports` 证券列表和本次 `observed_at`。来源足以支持所列业务存在；不支持本报告未声明的财务规模、供应链边权或价格传导结论。

- **ALAB**：[ Astera Labs About: semiconductor connectivity for rack-scale AI ](https://www.asteralabs.com/about/)
- **AMD**：[ AMD Products: processors, graphics, AI accelerators, networking ](https://www.amd.com/en/products.html)
- **ARM**：[ Arm Compute Platform: architecture, subsystems, IP and interconnects ](https://www.arm.com/company/arm-compute-platform)
- **AVGO**：[ Broadcom About Us: semiconductors and infrastructure software ](https://www.broadcom.com/company/about-us)
- **INTC**：[ Intel Company Overview: processors and foundry ](https://www.intel.com/content/www/us/en/company-overview/company-overview.html)
- **MRVL**：[ Marvell Data Center: custom compute, networking and storage silicon ](https://www.marvell.com/solutions/data-center.html)
- **NVDA**：[ NVIDIA Data Centers: GPU, CPU, and networking architectures ](https://www.nvidia.com/en-us/data-center/)
- **QCOM**：[ Qualcomm Products: mobile and compute chipsets ](https://www.qualcomm.com/products)
- **ADI**：[ Analog Devices: analog, mixed-signal and power management ](https://www.analog.com/en/who-we-are.html)
- **MCHP**：[ Microchip: analog and embedded control semiconductors ](https://www.microchip.com/en-us/about/corporate-overview/what-we-do)
- **MPWR**：[ MPS: power management and analog products ](https://www.monolithicpower.com/en/products.html)
- **NXPI**：[ NXP Processors and Microcontrollers: embedded control, analog and communications IP ](https://www.nxp.com/products/processors-and-microcontrollers%3AMICROCONTROLLERS-AND-PROCESSORS)
- **TXN**：[ Texas Instruments: analog and embedded processing chips ](https://www.ti.com/about-ti.html)
- **AMAT**：[ Applied Materials: semiconductor manufacturing products ](https://www.appliedmaterials.com/us/en/semiconductor/products.html)
- **ASML**：[ ASML Products: lithography, metrology and inspection ](https://www.asml.com/en/products)
- **KLAC**：[ KLA FY2026 Form 10-K: process control and process equipment ](https://ir.kla.com/sec-filings/all-sec-filings/content/0000319201-26-000027/klac-20260630.htm)
- **LRCX**：[ Lam Research: deposition, etch, strip and wafer cleaning ](https://www.lamresearch.com/products/products-overview/)
- **TER**：[ Teradyne Semiconductor Test ](https://www.teradyne.com/semiconductor-testing/)
- **CDNS**：[ Cadence Factsheet: EDA and semiconductor IP ](https://www.cadence.com/en_US/home/resources/factsheets/cadence-design-systems-inc-fs.html)
- **SNPS**：[ Synopsys: EDA and silicon IP ](https://www.synopsys.com/silicon-design.html)
- **MU**：[ Micron Corporate Profile: DRAM, NAND and NOR ](https://www.micron.com/about/company/corporate-profile)
- **SNDK**：[ Sandisk: flash storage, SSD and memory card products ](https://www.sandisk.com/products.aspx)
- **STX**：[ Seagate Corporate Overview: hard drive data storage ](https://www.seagate.com/gb/en/stories/corporate-overview/)
- **WDC**：[ Western Digital: hard drive product portfolio ](https://www.westerndigital.com/products/product-portfolio)
- **CSCO**：[ Cisco Networking Products and Solutions ](https://www.cisco.com/site/us/en/products/networking/index.html)
- **LITE**：[ Lumentum: optical and photonic networking technologies ](https://www.lumentum.com/en/solutions)
- **AMZN**：[ Amazon About Us: online shopping, cloud computing and advertising ](https://www.aboutamazon.com/about-us)
- **CRWV**：[ CoreWeave About Us: cloud for AI workloads ](https://www.coreweave.com/about-us)
- **GOOG, GOOGL**：[ Alphabet Q2 2025 Results: Google advertising and Google Cloud ](https://abc.xyz/assets/cc/27/3ada14014efbadd7a58472f1f3f4/2025q2-alphabet-earnings-release.pdf)
- **MSFT**：[ Microsoft 2025 Annual Report: Azure, server and productivity software ](https://www.microsoft.com/investor/reports/ar25/)
- **NBIS**：[ Nebius About: AI cloud platform ](https://nebius.com/about)
- **CRWD**：[ CrowdStrike Falcon Cloud Security ](https://www.crowdstrike.com/en-us/platform/cloud-security/)
- **CSCO**：[ Cisco Security: cloud, endpoint and network protection ](https://www.cisco.com/site/us/en/products/security/index.html)
- **DDOG**：[ Datadog: observability, monitoring and security platform ](https://www.datadoghq.com/product/)
- **FTNT**：[ Fortinet About: cybersecurity and networking ](https://www.fortinet.com/corporate/about-us/about-us)
- **PANW**：[ Palo Alto Networks: network, cloud and security operations ](https://www.paloaltonetworks.com/about-us)
- **ADBE**：[ Adobe About: creative, document and experience software ](https://www.adobe.com/about-adobe.html)
- **ADSK**：[ Autodesk: Design and Make platform ](https://www.autodesk.com/company)
- **INTU**：[ Intuit: TurboTax, QuickBooks, Mailchimp and enterprise software ](https://www.intuit.com/company/)
- **PLTR**：[ Palantir Architecture: AIP, Foundry and Apollo ](https://www.palantir.com/docs/foundry/architecture-center/overview)
- **SHOP**：[ Shopify: commerce platform for online and physical stores ](https://www.shopify.com/about)
- **WDAY**：[ Workday: finance, HR and planning software ](https://www.workday.com/en-au/company.html)
- **MELI**：[ Mercado Libre: commerce and fintech ecosystem ](https://investor.mercadolibre.com/about-meli)
- **APP**：[ AppLovin About: performance advertising platform ](https://www.applovin.com/en/about)
- **META**：[ Meta FY2025 Results: advertising revenue ](https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-Fourth-Quarter-and-Full-Year-2025-Results/)

## 尚未分类的候选证券

AAPL, ABNB, ADP, AEP, ALNY, AMGN, AXON, BKNG, BKR, CCEP, CEG, CMCSA, COST, CPRT, CSX, CTAS, DASH, DXCM, EXC, FANG, FAST, FER, GEHC, GILD, HON, HONA, IDXX, ISRG, KDP, LIN, MAR, MDLZ, MNST, MSTR, NFLX, ODFL, ORLY, PAYX, PCAR, PDD, PEP, PYPL, REGN, RKLB, ROP, ROST, SBUX, SPCX, TMUS, TRI, TSLA, TTWO, VRTX, WBD, WMT, XEL

本版未穷尽每家公司全部业务暴露；未进入某群不等于完全没有该项业务。未分类表示本次小范围证据整理未覆盖，不能把这些证券解释为同一行业或同一风险群。全部证券仍可参与另行定义的波动、趋势和期限状态分组。
