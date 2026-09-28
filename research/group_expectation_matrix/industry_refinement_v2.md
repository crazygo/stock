# 行业细分 v2：30 个平级研究群

分类版本：`industry_value_chain_v2`。生成时间：`2026-09-25T15:22:13.009138Z`。

按要求把原 11 个行业与价值链宽群分别细分为 2–3 个群，最终保留 30 个平级研究行。名称全部采用“原行业 - 子细分”。原宽群不再作为矩阵行，`derived_from_group_id` 和顶层 `refinement` 只记录谱系、用于程序校验，不要求页面呈现树形层级。

45 个证券的分类覆盖保持不变，原有 56 个未分类证券仍未分类。原 61 条宽类成员关系全部得到覆盖；细分后因真实多业务重叠产生 70 条成员关系，其中新增的 9 条是同一宽类内部的重复业务归属，并不是新增股票。

## 全部原群覆盖校验

| 原行业 | 细分群 | 原成员数 | 细分成员去重数 | 细分成员边 | 校验 |
|---|---|---:|---:|---:|---|
| 计算与通信芯片 | 处理器与计算加速（4）；架构与计算 IP（1）；连接与定制芯片（3） | 8 | 8 | 8 | 完整、未扩池 |
| 模拟、嵌入式与电源芯片 | 模拟信号与转换（2）；嵌入式控制（3）；电源管理（4） | 5 | 5 | 9 | 完整、未扩池 |
| 半导体设备与测试 | 晶圆工艺设备（2）；光刻设备（1）；检测与测试（2） | 5 | 5 | 5 | 完整、未扩池 |
| EDA 与半导体设计 IP | 设计自动化工具（2）；可复用设计 IP（3） | 3 | 3 | 5 | 完整、未扩池 |
| 存储器与数据存储 | DRAM 存储器（1）；闪存与固态存储（2）；硬盘存储（2） | 4 | 4 | 5 | 完整、未扩池 |
| 网络与高速互连 | 互连与交换芯片（3）；网络系统与平台（2）；光通信器件（1） | 6 | 6 | 6 | 完整、未扩池 |
| 云基础设施平台 | 通用公有云（4）；AI 算力云（2） | 6 | 6 | 6 | 完整、未扩池 |
| 网络与云安全 | 网络与接入安全（4）；终端保护与安全运营（2）；云与应用安全（3） | 7 | 7 | 9 | 完整、未扩池 |
| 企业应用与开发软件 | 创作与工程设计（2）；财务与企业经营（3）；数据与基础平台（4） | 9 | 9 | 9 | 完整、未扩池 |
| 数字商业平台 | 交易市场平台（2）；商家经营工具（1） | 3 | 3 | 3 | 完整、未扩池 |
| 数字广告平台 | 搜索与零售媒体（3）；社交与短视频广告（1）；应用内广告（1） | 5 | 5 | 5 | 完整、未扩池 |

## 30 个平级群与成员

| 群 | 成员 | 证券/发行人数 |
|---|---|---|
| 计算与通信芯片 - 处理器与计算加速 | AMD, INTC, NVDA, QCOM | 4/4 |
| 计算与通信芯片 - 架构与计算 IP | ARM | 1/1；小群 |
| 计算与通信芯片 - 连接与定制芯片 | ALAB, AVGO, MRVL | 3/3 |
| 模拟、嵌入式与电源芯片 - 模拟信号与转换 | ADI, TXN | 2/2；小群 |
| 模拟、嵌入式与电源芯片 - 嵌入式控制 | MCHP, NXPI, TXN | 3/3 |
| 模拟、嵌入式与电源芯片 - 电源管理 | ADI, MCHP, MPWR, TXN | 4/4 |
| 半导体设备与测试 - 晶圆工艺设备 | AMAT, LRCX | 2/2；小群 |
| 半导体设备与测试 - 光刻设备 | ASML | 1/1；小群 |
| 半导体设备与测试 - 检测与测试 | KLAC, TER | 2/2；小群 |
| EDA 与半导体设计 IP - 设计自动化工具 | CDNS, SNPS | 2/2；小群 |
| EDA 与半导体设计 IP - 可复用设计 IP | ARM, CDNS, SNPS | 3/3 |
| 存储器与数据存储 - DRAM 存储器 | MU | 1/1；小群 |
| 存储器与数据存储 - 闪存与固态存储 | MU, SNDK | 2/2；小群 |
| 存储器与数据存储 - 硬盘存储 | STX, WDC | 2/2；小群 |
| 网络与高速互连 - 互连与交换芯片 | ALAB, AVGO, MRVL | 3/3 |
| 网络与高速互连 - 网络系统与平台 | CSCO, NVDA | 2/2；小群 |
| 网络与高速互连 - 光通信器件 | LITE | 1/1；小群 |
| 云基础设施平台 - 通用公有云 | AMZN, GOOG, GOOGL, MSFT | 4/3 |
| 云基础设施平台 - AI 算力云 | CRWV, NBIS | 2/2；小群 |
| 网络与云安全 - 网络与接入安全 | AVGO, CSCO, FTNT, PANW | 4/4 |
| 网络与云安全 - 终端保护与安全运营 | CRWD, MSFT | 2/2；小群 |
| 网络与云安全 - 云与应用安全 | CRWD, DDOG, PANW | 3/3 |
| 企业应用与开发软件 - 创作与工程设计 | ADBE, ADSK | 2/2；小群 |
| 企业应用与开发软件 - 财务与企业经营 | INTU, SHOP, WDAY | 3/3 |
| 企业应用与开发软件 - 数据与基础平台 | AVGO, DDOG, MSFT, PLTR | 4/4 |
| 数字商业平台 - 交易市场平台 | AMZN, MELI | 2/2；小群 |
| 数字商业平台 - 商家经营工具 | SHOP | 1/1；小群 |
| 数字广告平台 - 搜索与零售媒体 | AMZN, GOOG, GOOGL | 3/2；发行人较少 |
| 数字广告平台 - 社交与短视频广告 | META | 1/1；小群 |
| 数字广告平台 - 应用内广告 | APP | 1/1；小群 |

## 细分口径与解释边界

- **并非互斥分区。** ADI/TXN 的信号转换与电源、MCHP/TXN 的控制与电源、CDNS/SNPS 的 EDA 与 IP、MU 的 DRAM 与闪存、CRWD/PANW 的跨安全业务，都可以重复归属。
- **按产品或业务角色定义。** 标签不声明该业务是公司主要收入来源，也没有收入权重。计算芯片中的“连接与定制芯片”是按连接、交换或定制数据基础设施器件共同角色定义，并不表示每位成员都同时开展定制芯片。
- **保留真实小群。** 30 群中 18 群只有 1–2 个证券，按已知发行人去重后共有 19 群只有 1–2 个发行人。单股群的历史表现不能当作已发现可推广行业共性；不能为了样本数把无证据的股票补入群。
- **同成员群保留并披露。** “计算与通信芯片 - 连接与定制芯片”与“网络与高速互连 - 互连与交换芯片”在本候选池中都恰好是 ALAB、AVGO、MRVL。两行来自不同研究定义，但目标达成统计会相同，不能视作两份独立发现。JSON 的 `equivalent_membership_groups` 记录这一对。
- **细分仍有边界。** KLA 的制程量检测与 Teradyne 的电性/系统测试合在“检测与测试”；企业“数据与基础平台”仍混合若干商业模式。这是本版 2–3 档限制内的透明定义，不能夸大为完全均质。
- **云和广告标签不是排他声明。** 通用云也提供 AI 算力；Google/Meta/Amazon 的广告不止当前子群所列渠道。该版是研究覆盖，不是所有业务的完整枚举。
- **证券与发行人分别看。** GOOG 与 GOOGL 都属于 Alphabet，“搜索与零售媒体”有 3 个证券但仅 2 个发行人。
- **没有按达成结果挑分类。** 本次只依业务事实细分，没有读取或优化各目标的达成率；细分是否有样本外增量，由后续时间验证回答。

## 来源与时间

v2 使用 59 个去重官方来源 URL，其中 16 个是相对 v1 新增的细分证据。v1 中已足够支持细分的来源保留其原 `observed_at`，本次补充或重查的来源记录本次 `observed_at`。每个子群成员至少有一条来源的 `supports` 明确对应。

`historical_membership` 保持 `current_snapshot_retrospective`。分类、分组周版本与行情样本日期是三个不同概念：本次定义可以对历史数据作描述性回看，但不能把现在查到的业务身份当成历史上当时已知的分类。旧 v1 文件保持原样；v2 的生成不表示旧标签被追溯改写。

新增细分证据：

- **ADI**：[Analog Devices Data Converters: ADC and DAC products](https://www.analog.com/en/product-category/data-converters.html)
- **TXN**：[TI Data Converters: ADC, DAC and integrated data converters](https://www.ti.com/product-category/data-converters/overview.html)
- **ADI**：[Analog Devices Power Management Integrated Circuits](https://www.analog.com/en/product-category/pmic.html)
- **MCHP**：[Microchip Voltage/Current Regulators, Converters and Controllers](https://www.microchip.com/en-us/products/power-management/voltage-current-power-conversion)
- **TXN**：[TI Power for Embedded Systems: power management solutions](https://www.ti.com/product-category/power-management/power-embedded-systems.html)
- **CDNS**：[Cadence Interface IP: silicon-proven Design IP portfolio](https://www.cadence.com/en_US/home/tools/silicon-solutions/design-ip/interface-ip.html)
- **SNPS**：[Synopsys Library IP: reusable infrastructure, bus and microcontroller IP](https://www.synopsys.com/designware-ip/soc-infrastructure-ip/designware-library.html)
- **NVDA**：[NVIDIA Networking: Spectrum-X Ethernet, Quantum InfiniBand and BlueField systems](https://www.nvidia.com/en-us/networking/)
- **AVGO**：[Broadcom Symantec Edge Secure Web Gateway: appliance and cloud network access protection](https://www.broadcom.com/products/cybersecurity/network/web-protection/proxy-sg-and-advanced-secure-gateway)
- **CRWD**：[CrowdStrike Falcon Endpoint Protection: antivirus, EDR and threat hunting](https://www.crowdstrike.com/wp-content/uploads/2024/10/Falcon-Endpoint-Protection-Enterprise-Datasheet.pdf)
- **MSFT**：[Microsoft Defender for Business: device, server and endpoint protection](https://www.microsoft.com/en-us/security/business/endpoint-security/microsoft-defender-business)
- **DDOG**：[Datadog Security: application, host, container and cloud infrastructure protection](https://docs.datadoghq.com/security/)
- **PANW**：[Palo Alto Networks Prisma Cloud: code-to-cloud application security](https://www.paloaltonetworks.com/cloud-security/prisma-public-cloud)
- **AMZN**：[Amazon Sponsored Products: ads within shopping results and product pages](https://advertising.amazon.com/en-ca/solutions/products/sponsored-products)
- **GOOG, GOOGL**：[Google Ads: advertising to people searching for products and services](https://ads.google.com/intl/en_na/home/)
- **META**：[Meta for Business: Facebook and Instagram Reels ads](https://www.facebook.com/business/ads/facebook-instagram-reels-ads)

完整来源、支持股票和观察时间见 [industry_tags_v2.json](config/industry_tags_v2.json) 各群的 `sources`。

## 结构与验证结果

- `schema_version=1.0`；`taxonomy_version=industry_value_chain_v2`。
- 顶层 `refinement` 恰好 11 项，每项包含 `source_group_id`、`source_name`、`source_symbols`、`subgroup_ids`。
- 每个群包含 `derived_from_group_id` 与字符串 `segment`；`name` 严格等于 `source_name + " - " + segment`。
- 30 个新 `group_id` 全部唯一，与 11 个旧宽群 ID 无交集；每群恰好归入一个 refinement 项。
- 每个原群的 2–3 个细分群并集严格等于原成员集；没有漏成员，也没有从别的原群或候选池外补入成员。
- 每个成员均有来源 `supports`；每条来源有 HTTPS URL 与可解析的 UTC `observed_at`。
- 原始分类文件的 SHA-256 记录在 `derived_from_taxonomy_sha256`，用于核验本次细分的确定基线。
