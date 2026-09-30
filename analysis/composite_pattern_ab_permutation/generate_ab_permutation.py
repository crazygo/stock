#!/usr/bin/env python3
"""Generate Full Permutation and Combination Analysis for 8 Micro-Patterns (A 【间隔 N】 B).

Scope:
- 138 US stocks and ETFs (QQQ constituents + User Favorites + 0086 Holdings + Sector ETFs)
- Past 3 months: 2026-07-01 to 2026-09-29 (5-minute regular session bars)
- 8 Micro-Patterns:
    4 上涨形态: ▲ 单边拉升, V型深弹, 冲高回落, 阶梯中继
    4 下跌形态: ▼ 单边下杀, 倒V冲顶, 探底回抽, 破位阴跌
- Full 8 x 8 = 64 Permutation Matrix
- 13 Interval Constraints N (12 specific structural/temporal intervals + ALL baseline):
    T1_IMM (紧邻连贯 ≤30m)
    T2_SHORT (短时蓄势 30m~2h)
    T3_INTRADAY (日内跨段 2h~1d)
    T4_NEXT_DAY (次日承接 1d~2d)
    T5_SWING (波段中程 2d~5d)
    R1_SHALLOW (极浅回踩 ≤2.0%)
    R2_MODERATE (中度良性回踩 2.0%~5.0%)
    R3_DEEP (深幅砸坑洗盘 >5.0%)
    V1_SQUEEZE (极致收敛 ≤2.5% VCP)
    V2_VOLATILE (宽幅剧烈震荡 >5.0%)
    S1_HIGHER (台阶抬高 Pb ≥ Pa)
    S2_LOWER (重心下移 Pb < Pa)
    ALL (全量综合基准)
- Transition Scopes:
    1. consecutive: 直接紧邻承接 (Immediate subsequent pattern)
    2. all_5d: 5日衍生全排列 (Any subsequent pattern within 5 days)
- Forward Metrics from end of B:
    后续 1天 +3%
    后续 1天 +5%
    后续 3天 +5% ★ (Core Alpha)
    后续 3天 +10%
    后续 5天 +10%

Output:
- analysis/composite_pattern_ab_permutation/ab_permutation_data.json
- analysis/composite_pattern_ab_permutation/index.html
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import time

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
QQQ_FILE = ROOT / "analysis" / "qqq_constituents.json"
M5_DIR = ROOT / "market_data" / "us_5m"
JSON_OUT = ANALYSIS_DIR / "ab_permutation_data.json"
HTML_OUT = ANALYSIS_DIR / "index.html"

# User favorites from Futu OpenD
FAVORITES_TICKERS = {
    'WDC', 'STX', 'TER', 'PLTR', 'NBIS', 'AVGO', 'AAOZ', 'CRWV', 'AMAT', 'LITX',
    'CBRS', 'RMBS', 'IONL', 'LRNZ', 'AAOI', 'VRT', 'AIPO', 'CIEN', 'SOXS', 'MU',
    'CRDO', 'AMD', 'SMTC', 'ARM', 'TXG', 'TWST', 'SDGR', 'QCOM', 'COHR', 'ALAB',
    'LITE', 'MRVL', 'LIFE', 'NOK', 'KOD'
}

# User 0086 holdings from Futu OpenD
HOLDINGS_0086 = {
    'SNXX', 'RKLB', 'NXT', 'NVTS', 'INTC', 'GOOG', 'FN', 'COHR', 'AVGO', 'AMZN',
    'AIPO', 'CRDO', 'MRVL', 'AMAT', 'HLTH', 'CIEN', 'VRT', 'LITX', 'SEDG', 'BBC', 'AXTI'
}

# Core ETFs
ETF_TICKERS = {
    'QQQ', 'SPY', 'DIA', 'SOXX', 'SMH', 'SOXL', 'SOXS', 'IGV', 'SPCX', 'XLU'
}

# Semiconductor / AI Hardware ecosystem
SEMI_HARDWARE = {
    'NVDA', 'AVGO', 'AMD', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'MU', 'INTC', 'ASML',
    'KLAC', 'MRVL', 'ADI', 'NXPI', 'MCHP', 'MPWR', 'TER', 'RMBS', 'SMTC', 'ALAB',
    'SNDK', 'COHR', 'LITE', 'CRDO', 'VRT', 'NVTS', 'AXTI', 'AAOI', 'AAOZ', 'CBRS',
    'WDC', 'STX', 'SOXX', 'SMH', 'SOXL', 'SOXS', 'FN', 'NXT'
}

# 8 Micro-Patterns Metadata
PATTERNS = [
    '▲ 单边拉升',
    'V型深弹',
    '冲高回落',
    '阶梯中继',
    '▼ 单边下杀',
    '倒V冲顶',
    '探底回抽',
    '破位阴跌'
]

PATTERN_META = {
    '▲ 单边拉升': {
        'type': 'SURGE',
        'code': 'MONOTONIC_RALLY',
        'color': '#10b981',
        'bg': 'rgba(16, 185, 129, 0.16)',
        'border': '#10b981',
        'desc': '无明显回撤直线上攻，多头单边加速赶顶或爆发',
        'role': '多头强攻'
    },
    'V型深弹': {
        'type': 'SURGE',
        'code': 'V_REBOUND',
        'color': '#06b6d4',
        'bg': 'rgba(6, 182, 212, 0.16)',
        'border': '#06b6d4',
        'desc': '前半程深探砸出黄金坑，尾盘放量暴力反扑创新高',
        'role': '极致反抽'
    },
    '冲高回落': {
        'type': 'SURGE',
        'code': 'SPIKE_PULLBACK',
        'color': '#f59e0b',
        'bg': 'rgba(245, 158, 11, 0.16)',
        'border': '#f59e0b',
        'desc': '脉冲式急拉遇阻回踩消化浮筹，整体涨幅仍超+5%',
        'role': '高位试盘'
    },
    '阶梯中继': {
        'type': 'SURGE',
        'code': 'STEP_CONTINUATION',
        'color': '#84cc16',
        'bg': 'rgba(132, 204, 22, 0.16)',
        'border': '#84cc16',
        'desc': '窄幅横盘蓄势平台后二次放量起爆，阶梯式推进',
        'role': '蓄势中继'
    },
    '▼ 单边下杀': {
        'type': 'DROP',
        'code': 'MONOTONIC_PLUNGE',
        'color': '#e11d48',
        'bg': 'rgba(225, 29, 72, 0.16)',
        'border': '#e11d48',
        'desc': '无反抽直线下跌，空头动能单边宣泄砸盘',
        'role': '空头暴击'
    },
    '倒V冲顶': {
        'type': 'DROP',
        'code': 'INVERTED_V_TOP',
        'color': '#8b5cf6',
        'bg': 'rgba(139, 92, 246, 0.16)',
        'border': '#8b5cf6',
        'desc': '冲高试盘后极速破位倒栽葱跳水，诱多套牢',
        'role': '诱多杀跌'
    },
    '探底回抽': {
        'type': 'DROP',
        'code': 'BOTTOM_DRAWDOWM',
        'color': '#f97316',
        'bg': 'rgba(249, 115, 22, 0.16)',
        'border': '#f97316',
        'desc': '前半程深跌超跌，尾盘出现抵抗性回抽但仍处跌幅区',
        'role': '超跌回抽'
    },
    '破位阴跌': {
        'type': 'DROP',
        'code': 'BREAKDOWN_DRIFT',
        'color': '#dc2626',
        'bg': 'rgba(220, 38, 38, 0.16)',
        'border': '#dc2626',
        'desc': '平台震荡破位后持续阴跌下沉，多头抵抗无力',
        'role': '破位下沉'
    }
}

INTERVAL_CONFIGS = {
    'ALL': {
        'name': '全量基准 (ALL)',
        'cat': '综合基准',
        'desc': '5个交易日内任意间隔 (0 ~ 390根 5m K线)',
        'badge': 'ALL 基准',
        'rule': '无额外间隔过滤'
    },
    'T1_IMM': {
        'name': '紧邻连贯 (Gap ≤ 30m)',
        'cat': '时间间隔',
        'desc': 'A结束与B启动间隔 ≤ 30分钟 (0~6根 5m K线)',
        'badge': 'T1 紧邻',
        'rule': 'gap ≤ 6 根K线 (≤ 30m)'
    },
    'T2_SHORT': {
        'name': '短时蓄势 (30m ~ 2h)',
        'cat': '时间间隔',
        'desc': 'A结束与B启动间隔 30分钟 ~ 2小时 (7~24根 5m K线)',
        'badge': 'T2 短憩',
        'rule': '7 ≤ gap ≤ 24 根K线 (30m~2h)'
    },
    'T3_INTRADAY': {
        'name': '日内跨段 (2h ~ 1d)',
        'cat': '时间间隔',
        'desc': 'A结束与B启动间隔 2小时 ~ 1交易日 (25~78根 5m K线)',
        'badge': 'T3 日内',
        'rule': '25 ≤ gap ≤ 78 根K线 (2h~1d)'
    },
    'T4_NEXT_DAY': {
        'name': '次日承接 (1d ~ 2d)',
        'cat': '时间间隔',
        'desc': 'A结束与B启动间隔 1交易日 ~ 2交易日 (79~156根 5m K线)',
        'badge': 'T4 次日',
        'rule': '79 ≤ gap ≤ 156 根K线 (1d~2d)'
    },
    'T5_SWING': {
        'name': '波段中程 (2d ~ 5d)',
        'cat': '时间间隔',
        'desc': 'A结束与B启动间隔 2交易日 ~ 5交易日 (157~390根 5m K线)',
        'badge': 'T5 波段',
        'rule': '157 ≤ gap ≤ 390 根K线 (2d~5d)'
    },
    'R1_SHALLOW': {
        'name': '极浅回踩 (回撤 ≤ 2.0%)',
        'cat': '回撤约束',
        'desc': '间隔期最低点相对A终点回撤不超过 2.0% (极强抗跌支撑)',
        'badge': 'R1 浅踩',
        'rule': 'dd ≥ -2.0% (抗跌)'
    },
    'R2_MODERATE': {
        'name': '中度良性回撤 (2.0% ~ 5.0%)',
        'cat': '回撤约束',
        'desc': '间隔期回撤在 2.0% ~ 5.0% 之间 (良性洗盘释放浮筹)',
        'badge': 'R2 中撤',
        'rule': '-5.0% ≤ dd < -2.0%'
    },
    'R3_DEEP': {
        'name': '深幅砸坑洗盘 (回撤 > 5.0%)',
        'cat': '回撤约束',
        'desc': '间隔期最低点相对A终点回撤超过 5.0% (深度洗盘挖坑或破位下探)',
        'badge': 'R3 深坑',
        'rule': 'dd < -5.0% (深探)'
    },
    'V1_SQUEEZE': {
        'name': '极致收敛 (振幅 ≤ 2.5%)',
        'cat': '波动率约束',
        'desc': '间隔期极窄箱体震荡，振幅 ≤ 2.5% (VCP 波动率收缩)',
        'badge': 'V1 收敛',
        'rule': 'amp ≤ 2.5% (VCP压缩)'
    },
    'V2_VOLATILE': {
        'name': '宽幅剧烈震荡 (振幅 > 5.0%)',
        'cat': '波动率约束',
        'desc': '间隔期高低振幅超过 5.0% (多空剧烈博弈与高换手)',
        'badge': 'V2 宽幅',
        'rule': 'amp > 5.0% (剧烈洗盘)'
    },
    'S1_HIGHER': {
        'name': '台阶抬高 (Pb ≥ Pa)',
        'cat': '形态重心',
        'desc': 'B的起步价高于或平于A的终点价 (结构台阶向上抬升突破)',
        'badge': 'S1 抬高',
        'rule': 'P_start(B) ≥ P_end(A)'
    },
    'S2_LOWER': {
        'name': '重心下移 (Pb < Pa)',
        'cat': '形态重心',
        'desc': 'B的起步价明显低于A的终点价 (重心向下漂移或低开)',
        'badge': 'S2 下移',
        'rule': 'P_start(B) < P_end(A)'
    }
}

# Complete metadata for all 138 tickers
TICKER_META = {
    'NVDA': {'name': '英伟达 (NVIDIA)', 'sector': '半导体/AI算力芯片'},
    'AAPL': {'name': '苹果 (Apple)', 'sector': '消费电子/移动生态'},
    'MSFT': {'name': '微软 (Microsoft)', 'sector': '云计算/企业软件/AI'},
    'AMZN': {'name': '亚马逊 (Amazon)', 'sector': '电商/AWS云计算'},
    'GOOGL': {'name': '谷歌 A (Alphabet)', 'sector': '互联网搜索/AI'},
    'GOOG': {'name': '谷歌 C (Alphabet)', 'sector': '互联网搜索/AI'},
    'META': {'name': 'Meta (Facebook)', 'sector': '社交媒体/数字广告/AI'},
    'TSLA': {'name': '特斯拉 (Tesla)', 'sector': '新能源智能汽车/机器人'},
    'AVGO': {'name': '博通 (Broadcom)', 'sector': '半导体/网络与定制ASIC'},
    'AMD': {'name': '超微半导体 (AMD)', 'sector': '半导体/GPU/CPU'},
    'INTC': {'name': '英特尔 (Intel)', 'sector': '半导体/晶圆代工制造'},
    'QCOM': {'name': '高通 (Qualcomm)', 'sector': '移动SOC芯片/无线通信基带'},
    'TXN': {'name': '德州仪器 (TI)', 'sector': '模拟芯片/嵌入式处理器'},
    'MU': {'name': '美光科技 (Micron)', 'sector': '存储半导体/HBM/DRAM'},
    'ASML': {'name': '阿斯麦 (ASML)', 'sector': '半导体前道光刻机设备'},
    'LRCX': {'name': '泛林集团 (Lam Research)', 'sector': '半导体刻蚀设备'},
    'AMAT': {'name': '应用材料 (Applied Materials)', 'sector': '半导体成膜/薄膜设备'},
    'KLAC': {'name': '科磊半导体 (KLA Corp)', 'sector': '半导体量测与缺陷检测'},
    'ARM': {'name': '安谋科技 (ARM Holdings)', 'sector': '半导体芯片架构与IP'},
    'ADI': {'name': '亚德诺半导体 (Analog Devices)', 'sector': '工业/汽车高性能模拟芯片'},
    'NXPI': {'name': '恩智浦 (NXP Semiconductors)', 'sector': '汽车电子与边缘计算半导体'},
    'MCHP': {'name': '微芯科技 (Microchip)', 'sector': '微控制器MCU与模拟半导体'},
    'MPWR': {'name': '芯源系统 (Monolithic Power)', 'sector': '高压大功率电源管理芯片'},
    'MRVL': {'name': '迈威尔科技 (Marvell)', 'sector': '数据中心网络芯片/光通信'},
    'TER': {'name': '泰瑞达 (Teradyne)', 'sector': '半导体自动化测试/工业机器人'},
    'ALAB': {'name': 'Astera Labs', 'sector': '半导体/数据中心PCIe连接芯片'},
    'LITE': {'name': '朗umentum (Lumentum)', 'sector': 'AI光通信光模块/光子芯片'},
    'WDC': {'name': '西部数据 (Western Digital)', 'sector': '数据存储/闪存SSD/机械硬盘'},
    'STX': {'name': '希捷科技 (Seagate)', 'sector': '大容量机械硬盘HDD/云存储'},
    'SNDK': {'name': '闪迪 (SanDisk/Storage)', 'sector': '闪存存储设备/NAND'},
    'PLTR': {'name': 'Palantir', 'sector': '大数据/AI国防分析'},
    'CSCO': {'name': '思科 (Cisco)', 'sector': '企业网络通信硬件'},
    'PANW': {'name': '派拓网络 (Palo Alto)', 'sector': '网络安全/云防火墙'},
    'CRWD': {'name': 'CrowdStrike', 'sector': '端点安全/威胁情报SaaS'},
    'FTNT': {'name': '飞塔 (Fortinet)', 'sector': '统一威胁管理/网络安全'},
    'DDOG': {'name': 'Datadog', 'sector': '云基础设施可观测性监控平台'},
    'ADBE': {'name': '奥多比 (Adobe)', 'sector': '创意设计与数字媒体SaaS'},
    'INTU': {'name': '财捷 (Intuit)', 'sector': '中小微企业财务记账与TurboTax'},
    'SNPS': {'name': '新思科技 (Synopsys)', 'sector': 'EDA设计工具与半导体IP'},
    'CDNS': {'name': '铿腾电子 (Cadence)', 'sector': 'EDA电子设计自动化工业软件'},
    'WDAY': {'name': 'Workday', 'sector': '企业云原生HCM人事与财务ERP'},
    'ADSK': {'name': '欧特克 (Autodesk)', 'sector': 'AutoCAD/BIM工业设计软件'},
    'MSTR': {'name': '微策略 (MicroStrategy)', 'sector': '比特币财务储备与企业BI软件'},
    'APP': {'name': 'AppLovin', 'sector': 'AI移动应用广告营销引擎'},
    'SHOP': {'name': 'Shopify', 'sector': '独立站电商SaaS生态'},
    'PDD': {'name': '拼多多 (PDD Holdings)', 'sector': '电商零售/跨境电商Temu'},
    'MELI': {'name': '自由市场 (MercadoLibre)', 'sector': '拉美电商龙头与金融科技'},
    'NFLX': {'name': '奈飞 (Netflix)', 'sector': '流媒体视频娱乐'},
    'BKNG': {'name': 'Booking Holdings', 'sector': '全球在线商旅预订OTA'},
    'ABNB': {'name': '爱彼迎 (Airbnb)', 'sector': '全球民宿与特色短租平台'},
    'DASH': {'name': 'DoorDash', 'sector': '本地即时生活配送与外卖'},
    'PYPL': {'name': '贝宝 (PayPal)', 'sector': '全球跨境在线数字支付网络'},
    'ADP': {'name': '安德普翰 (ADP)', 'sector': '企业人力资源管理/薪资SaaS'},
    'PAYX': {'name': '沛齐 (Paychex)', 'sector': '中小企业薪酬核算与福利外包'},
    'CTAS': {'name': '信达思 (Cintas)', 'sector': '企业制服与工作场所设施服务'},
    'FAST': {'name': '快扣 (Fastenal)', 'sector': '工业零配件紧固件智能分销'},
    'PCAR': {'name': '帕卡卡车 (PACCAR)', 'sector': '肯沃斯/彼得比尔特重型商用卡车'},
    'CSX': {'name': 'CSX铁路 (CSX Corp)', 'sector': '北美干线铁路联运物流'},
    'ODFL': {'name': 'Old Dominion Freight', 'sector': '北美高品质零担卡车货运'},
    'HON': {'name': '霍尼韦尔 (Honeywell)', 'sector': '航空航天与智能工业互联'},
    'HONA': {'name': 'HONA Holdings', 'sector': '精密工业制造装备与自动化'},
    'ROP': {'name': '罗珀科技 (Roper)', 'sector': '医疗科技与高壁垒工业测量软件'},
    'AXON': {'name': 'Axon Enterprise', 'sector': '执法部门电击枪与视讯证据云平台'},
    'CRWV': {'name': 'Crown Crafts', 'sector': '高品质纺织品与消费品制造'},
    'RKLB': {'name': '火箭实验室 (Rocket Lab)', 'sector': '商业轻中型运载火箭发射与卫星制造'},
    'NBIS': {'name': 'Nebius Group (原YNDX)', 'sector': 'AI基础设施/高性能GPU云算力'},
    'CEG': {'name': '星座能源 (Constellation Energy)', 'sector': '清洁核电与数据中心供电'},
    'AEP': {'name': '美国电力 (American Electric Power)', 'sector': '输配电网基础设施公用事业'},
    'XEL': {'name': '埃克西尔能源 (Xcel Energy)', 'sector': '风光清洁可再生公用事业电网'},
    'EXC': {'name': '爱克斯龙 (Exelon)', 'sector': '美国最大输配电公用事业网络'},
    'FANG': {'name': '响尾蛇能源 (Diamondback)', 'sector': '二叠纪盆地页岩油气低成本勘探'},
    'BKR': {'name': '贝克休斯 (Baker Hughes)', 'sector': '油气上游钻采与碳捕集装备'},
    'LIN': {'name': '林德气体 (Linde)', 'sector': '工业气体与特种材料'},
    'AMGN': {'name': '安进 (Amgen)', 'sector': '生物医药/抗体创新药'},
    'GILD': {'name': '吉利德科学 (Gilead)', 'sector': '抗病毒/生物创新药'},
    'REGN': {'name': '再生元制药 (Regeneron)', 'sector': '单克隆抗体创新药研发'},
    'VRTX': {'name': '福泰制药 (Vertex)', 'sector': '囊性纤维化靶向药物'},
    'ISRG': {'name': '直觉外科 (Intuitive Surgical)', 'sector': '达芬奇手术机器人'},
    'ALNY': {'name': '艾拉伦 (Alnylam)', 'sector': 'RNAi核酸药物基因靶向治疗'},
    'DXCM': {'name': '德康医疗 (DexCom)', 'sector': '连续血糖无创监测可穿戴硬件'},
    'IDXX': {'name': '爱德士生物 (IDEXX Labs)', 'sector': '伴侣动物体外诊断系统与试剂'},
    'GEHC': {'name': 'GE医疗科技 (GE HealthCare)', 'sector': '核磁共振CT超声与医疗影像'},
    'WMT': {'name': '沃尔玛 (Walmart)', 'sector': '零售连锁/消费超市'},
    'COST': {'name': '开市客 (Costco)', 'sector': '仓储会员零售'},
    'ROST': {'name': '罗斯百货 (Ross Stores)', 'sector': '品牌折扣服装零售'},
    'ORLY': {'name': '奥莱利汽车 (O\'Reilly Auto)', 'sector': '专业汽车售后配件连锁'},
    'CPRT': {'name': '科帕特 (Copart)', 'sector': '全球全损车辆残值在线竞价拍卖'},
    'PEP': {'name': '百事公司 (PepsiCo)', 'sector': '非酒精饮料与休闲零食'},
    'KDP': {'name': 'Keurig Dr Pepper', 'sector': '胶囊咖啡系统与特色苏打饮料'},
    'MDLZ': {'name': '亿滋国际 (Mondelez)', 'sector': '奥利奥/趣多多零食制造'},
    'MNST': {'name': '怪物饮料 (Monster Beverage)', 'sector': '功能性能量饮料'},
    'SBUX': {'name': '星巴克 (Starbucks)', 'sector': '精品咖啡连锁餐饮'},
    'CCEP': {'name': '可口可乐欧洲太平洋 (CCEP)', 'sector': '软饮料装瓶商'},
    'TMUS': {'name': 'T-Mobile US', 'sector': '5G无线通信运营商'},
    'CMCSA': {'name': '康卡斯特 (Comcast)', 'sector': '宽带通信服务与影视娱乐'},
    'WBD': {'name': '华纳兄弟探索 (Warner Bros)', 'sector': '影视娱乐媒体与Max流媒体'},
    'TTWO': {'name': 'Take-Two Interactive', 'sector': '3A级互动娱乐与GTA系列游戏'},
    'MAR': {'name': '万豪国际 (Marriott)', 'sector': '全球豪华酒店度假村集团'},
    'TRI': {'name': '汤森路透 (Thomson Reuters)', 'sector': '法律法规与财税信息服务'},
    'FER': {'name': 'Ferrovial', 'sector': '全球机场与特许收费公路基建'},
    'QQQ': {'name': 'Invesco QQQ Trust', 'sector': '纳斯达克100科技指数ETF'},
    'SPY': {'name': 'SPDR S&P 500 ETF', 'sector': '标普500基准宽基ETF'},
    'DIA': {'name': 'SPDR Dow Jones ETF', 'sector': '道琼斯30工业平均指数ETF'},
    'SOXX': {'name': 'iShares Semiconductor ETF', 'sector': '费城半导体行业ETF'},
    'SMH': {'name': 'VanEck Semiconductor ETF', 'sector': '半导体行业基准ETF'},
    'SOXL': {'name': 'Direxion Daily Semi Bull 3X', 'sector': '3倍做多半导体杠杆ETF'},
    'SOXS': {'name': 'Direxion Daily Semi Bear 3X', 'sector': '3倍做空半导体杠杆ETF'},
    'IGV': {'name': 'iShares Expanded Tech-Software', 'sector': '扩展型科技软件行业ETF'},
    'SPCX': {'name': 'The Space Acquisition ETF', 'sector': '商业航天/太空经济主题ETF'},
    'XLU': {'name': 'Utilities Select Sector SPDR', 'sector': '公用事业电力行业ETF'},
    'COHR': {'name': '相干高科 (Coherent Corp)', 'sector': '工业激光器/AI光通信光模块'},
    'CRDO': {'name': '默升科技 (Credo Technology)', 'sector': '高速串行连接芯片/有源光缆AEC'},
    'VRT': {'name': '维谛技术 (Vertiv Holdings)', 'sector': '数据中心关键供配电与液冷系统'},
    'CIEN': {'name': '锡耶纳 (Ciena Corp)', 'sector': '波分复用光传输网络设备'},
    'FN': {'name': '飞尼尔 (Fabrinet)', 'sector': '高端光电子光器件精密先进封装代工'},
    'NXT': {'name': 'Nextracker', 'sector': '智能太阳能跟踪支架控制系统'},
    'NVTS': {'name': '纳微半导体 (Navitas)', 'sector': '第三代氮化镓(GaN)高效功率芯片'},
    'AIPO': {'name': 'AIPO Holdings', 'sector': '纳斯达克高弹性成长标的'},
    'AAOZ': {'name': 'AAOZ Holdings', 'sector': '纳斯达克高弹性科技标的'},
    'AAOI': {'name': '应用光电 (Applied Opto)', 'sector': '高速数据中心光模块/CATV设备'},
    'CBRS': {'name': 'CBRS Group', 'sector': '高弹性通信与计算标的'},
    'RMBS': {'name': '兰巴斯 (Rambus)', 'sector': '高速内存接口芯片/硅智财IP'},
    'SMTC': {'name': '升特半导体 (Semtech)', 'sector': 'LoRa物联网与高性能模拟芯片'},
    'LITX': {'name': '锂能科技 (LITX)', 'sector': '新能源锂电材料高弹性标的'},
    'IONL': {'name': 'IONL Holdings', 'sector': '量子计算概念高弹性标的'},
    'LRNZ': {'name': 'LRNZ Holdings', 'sector': '杠杆成长高弹性标的'},
    'SDGR': {'name': '薛定谔 (Schrodinger)', 'sector': 'AI物理分子动力学药物设计软件'},
    'SEDG': {'name': '太阳边缘 (SolarEdge)', 'sector': '光伏直流优化器与储能逆变器'},
    'BBC': {'name': 'BBC BioTech', 'sector': '纳斯达克生物医药高弹性标的'},
    'HLTH': {'name': 'HealthTech Group', 'sector': '医疗健康数字化服务'},
    'AXTI': {'name': '通用半导体 (AXT Inc)', 'sector': '磷化铟InP/砷化镓GaAs半导体衬底'},
    'SNXX': {'name': 'SNXX Holdings', 'sector': '纳斯达克高弹性成长标的'},
    'TWST': {'name': '拓实生物 (Twist Bioscience)', 'sector': '硅基高通量DNA合成生物学平台'},
    'TXG': {'name': '10x Genomics', 'sector': '单细胞测序与空间转录组学仪器试剂'},
    'LIFE': {'name': 'Life Sciences Group', 'sector': '生命科学与生物技术标的'},
    'NOK': {'name': '诺基亚 (Nokia)', 'sector': '5G电信网络设备与技术专利授权'},
    'KOD': {'name': '柯达克科学 (Kodiak Sciences)', 'sector': '视网膜血管疾病抗体生物医药'}
}


def test_interval(t, cond_key):
    if cond_key == 'ALL':
        return True
    elif cond_key == 'T1_IMM':
        return t['gap'] <= 6
    elif cond_key == 'T2_SHORT':
        return 7 <= t['gap'] <= 24
    elif cond_key == 'T3_INTRADAY':
        return 25 <= t['gap'] <= 78
    elif cond_key == 'T4_NEXT_DAY':
        return 79 <= t['gap'] <= 156
    elif cond_key == 'T5_SWING':
        return 157 <= t['gap'] <= 390
    elif cond_key == 'R1_SHALLOW':
        return t['dd'] >= -0.02
    elif cond_key == 'R2_MODERATE':
        return -0.05 <= t['dd'] < -0.02
    elif cond_key == 'R3_DEEP':
        return t['dd'] < -0.05
    elif cond_key == 'V1_SQUEEZE':
        return t['amp'] <= 0.025
    elif cond_key == 'V2_VOLATILE':
        return t['amp'] > 0.05
    elif cond_key == 'S1_HIGHER':
        return t['higher']
    elif cond_key == 'S2_LOWER':
        return not t['higher']
    return True


def run_full_permutation_analysis():
    t0 = time.time()
    print("=" * 70)
    print("1. Loading regular-session 5m market data for 138 universe tickers...")
    print("=" * 70)

    qqq_tickers = set(json.loads(QQQ_FILE.read_text())['tickers']) if QQQ_FILE.exists() else set()
    all_dirs = sorted([p for p in M5_DIR.iterdir() if p.is_dir() and not p.name.isdigit()])
    symbols_to_scan = [p.name for p in all_dirs]
    print(f"Scanning {len(symbols_to_scan)} symbols...")

    loaded_series = {}
    for s in symbols_to_scan:
        p = M5_DIR / s / "2026.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            if 'time_key' not in df.columns:
                continue
            time_str = df['time_key'].astype(str)
            dates = time_str.str[:10]
            if 'session_type' in df.columns:
                reg = df[(dates >= '2026-07-01') & (df['session_type'] == 'regular')].sort_values('time_key').reset_index(drop=True)
            else:
                times = time_str.str[11:16]
                reg = df[(dates >= '2026-07-01') & (times >= '09:35') & (times <= '16:00')].sort_values('time_key').reset_index(drop=True)

            if len(reg) < 70:
                continue

            loaded_series[s] = {
                'close': reg['close'].to_numpy(dtype=np.float64),
                'high': reg['high'].to_numpy(dtype=np.float64),
                'low': reg['low'].to_numpy(dtype=np.float64),
                'time_keys': reg['time_key'].tolist(),
                'len': len(reg)
            }
        except Exception as err:
            print(f"Error loading {s}: {err}")

    print(f"Loaded {len(loaded_series)} series successfully in {time.time() - t0:.2f}s")

    # 2. Detect momentum episodes (N=24 bars = 2h, M=0.05, D=0.05)
    print("\n2. Detecting 8 micro-patterns across all tickers...")
    N = 24
    M = 0.05
    D = 0.05

    stock_shapes = {}
    total_events_count = 0

    for s, s_data in loaded_series.items():
        closes = s_data['close']
        highs = s_data['high']
        lows = s_data['low']
        time_keys = s_data['time_keys']
        tot = s_data['len']

        if tot < N + 10:
            continue

        rets = np.full(tot, np.nan)
        rets[N:] = (closes[N:] - closes[:-N]) / closes[:-N]

        in_s, s_st = False, 0
        in_d, d_st = False, 0
        s_events, d_events = [], []

        for i in range(N, tot):
            r = rets[i]
            if r >= M:
                if not in_s:
                    in_s, s_st = True, i
            else:
                if in_s:
                    s_events.append({'type': 'SURGE', 'st': s_st - N, 'end': i - 1})
                    in_s = False
            if r <= -D:
                if not in_d:
                    in_d, d_st = True, i
            else:
                if in_d:
                    d_events.append({'type': 'DROP', 'st': d_st - N, 'end': i - 1})
                    in_d = False

        if in_s:
            s_events.append({'type': 'SURGE', 'st': s_st - N, 'end': tot - 1})
        if in_d:
            d_events.append({'type': 'DROP', 'st': d_st - N, 'end': tot - 1})

        evts = s_events + d_events
        for e in evts:
            c_sub = closes[e['st']: e['end'] + 1]
            cnt = len(c_sub)
            p0 = closes[e['st']]
            sub_r = (c_sub - p0) / p0
            min_idx = int(np.argmin(sub_r))
            max_idx = int(np.argmax(sub_r))
            min_ret = sub_r[min_idx]
            max_ret = sub_r[max_idx]
            min_pos = min_idx / max(1, cnt - 1)
            max_pos = max_idx / max(1, cnt - 1)
            end_ret = (closes[e['end']] - p0) / p0

            if e['type'] == 'SURGE':
                if min_ret >= -0.005 and max_pos >= 0.75:
                    pat = '▲ 单边拉升'
                elif min_pos <= 0.45 and min_ret <= -0.015:
                    pat = 'V型深弹'
                elif max_pos <= 0.55 and end_ret < max_ret - 0.015:
                    pat = '冲高回落'
                else:
                    pat = '阶梯中继'
            else:
                if max_ret <= 0.005 and min_pos >= 0.75:
                    pat = '▼ 单边下杀'
                elif max_pos <= 0.45 and max_ret >= 0.015:
                    pat = '倒V冲顶'
                elif min_pos <= 0.55 and end_ret > min_ret + 0.015:
                    pat = '探底回抽'
                else:
                    pat = '破位阴跌'
            e['pattern'] = pat

        # Merge contiguous overlapping windows into standardized discrete shapes per pattern
        shapes = []
        for p_name in PATTERNS:
            p_evts = [e for e in evts if e['pattern'] == p_name]
            active = np.zeros(tot, dtype=int)
            for e in p_evts:
                active[e['st']: e['end'] + 1] = 1

            in_shape, s_idx = False, 0
            for i in range(tot):
                if active[i] == 1:
                    if not in_shape:
                        in_shape, s_idx = True, i
                else:
                    if in_shape:
                        shapes.append({
                            'symbol': s,
                            'pattern': p_name,
                            'st': s_idx,
                            'end': i - 1,
                            'st_time': time_keys[s_idx],
                            'end_time': time_keys[i - 1],
                            'bars': i - s_idx,
                            'p_start': float(closes[s_idx]),
                            'p_end': float(closes[i - 1])
                        })
                        in_shape = False
            if in_shape:
                shapes.append({
                    'symbol': s,
                    'pattern': p_name,
                    'st': s_idx,
                    'end': tot - 1,
                    'st_time': time_keys[s_idx],
                    'end_time': time_keys[tot - 1],
                    'bars': tot - s_idx,
                    'p_start': float(closes[s_idx]),
                    'p_end': float(closes[tot - 1])
                })

        shapes.sort(key=lambda x: (x['st'], x['end']))
        stock_shapes[s] = shapes
        total_events_count += len(shapes)

    print(f"Total merged discrete shapes: {total_events_count}")

    # 3. Extract transitions (Consecutive & All 5-Day) and compute forward metrics from B
    print("\n3. Building sequential transition pairs and calculating forward performance...")

    consec_list = []
    all5d_list = []

    for s, shapes in stock_shapes.items():
        s_data = loaded_series[s]
        closes = s_data['close']
        highs = s_data['high']
        lows = s_data['low']
        time_keys = s_data['time_keys']
        tot = s_data['len']

        is_qqq = (s in qqq_tickers)
        is_fav = (s in FAVORITES_TICKERS)
        is_held = (s in HOLDINGS_0086)
        is_etf = (s in ETF_TICKERS)
        is_semi = (s in SEMI_HARDWARE)

        # For direct consecutive: first B where B.st >= A.end
        for i in range(len(shapes)):
            A = shapes[i]
            first_B = None
            for j in range(i + 1, len(shapes)):
                if shapes[j]['st'] >= A['end']:
                    first_B = shapes[j]
                    break

            if first_B and first_B['st'] - A['end'] <= 390:
                t_obj = build_transition_record(s, A, first_B, closes, highs, lows, time_keys, tot,
                                                is_qqq, is_fav, is_held, is_etf, is_semi)
                consec_list.append(t_obj)

            # For all 5-day pairs: any B where B.st >= A.end and gap <= 390
            for j in range(i + 1, len(shapes)):
                B = shapes[j]
                if B['st'] >= A['end']:
                    gap = B['st'] - A['end']
                    if gap <= 390:
                        t_obj = build_transition_record(s, A, B, closes, highs, lows, time_keys, tot,
                                                        is_qqq, is_fav, is_held, is_etf, is_semi)
                        all5d_list.append(t_obj)
                    else:
                        break

    print(f"Total Direct Consecutive Transitions: {len(consec_list)}")
    print(f"Total All 5-Day Transitions: {len(all5d_list)}")

    # 4. Aggregating 8x8 Permutation Matrices for 13 interval conditions
    print("\n4. Aggregating 8x8 Permutation Matrices for 13 interval conditions...")
    dataset = {
        'meta': {
            'universe_count': len(loaded_series),
            'total_shapes': total_events_count,
            'time_range': '2026-07-01 ~ 2026-09-29',
            'consecutive_count': len(consec_list),
            'all5d_count': len(all5d_list),
            'patterns': PATTERNS,
            'pattern_meta': PATTERN_META,
            'intervals': INTERVAL_CONFIGS,
            'ticker_meta': TICKER_META
        },
        'scopes': {
            'consecutive': process_scope_transitions(consec_list),
            'all_5d': process_scope_transitions(all5d_list)
        }
    }

    # Save to JSON
    print(f"\nWriting output to {JSON_OUT}...")
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)
    print(f"Saved ab_permutation_data.json ({JSON_OUT.stat().st_size / 1024 / 1024:.2f} MB)")

    # 5. Generate interactive HTML dashboard
    print(f"\n5. Generating interactive HTML dashboard to {HTML_OUT}...")
    generate_html(dataset)
    print(f"Saved {HTML_OUT} successfully ({HTML_OUT.stat().st_size / 1024 / 1024:.2f} MB)")

    return dataset


def build_transition_record(s, A, B, closes, highs, lows, time_keys, tot,
                            is_qqq, is_fav, is_held, is_etf, is_semi):
    st_int = A['end']
    end_int = B['st']
    gap = end_int - st_int
    p_A = closes[st_int]
    p_B = closes[end_int]

    sub_low = np.min(lows[st_int: end_int + 1]) if end_int >= st_int else p_A
    sub_high = np.max(highs[st_int: end_int + 1]) if end_int >= st_int else p_A
    dd = float((sub_low - p_A) / p_A)
    amp = float((sub_high - sub_low) / p_A)
    higher = bool(p_B >= p_A)

    # Forward metrics starting from B['end']
    b_end = B['end']
    p_entry = closes[b_end]

    max_fwd_1 = min(tot - 1, b_end + 78)
    fwd_high_1 = np.max(highs[b_end + 1: max_fwd_1 + 1]) if b_end + 1 <= tot - 1 else p_entry
    hit_1d_3 = bool(fwd_high_1 >= p_entry * 1.03)
    hit_1d_5 = bool(fwd_high_1 >= p_entry * 1.05)

    max_fwd_3 = min(tot - 1, b_end + 234)
    fwd_high_3 = np.max(highs[b_end + 1: max_fwd_3 + 1]) if b_end + 1 <= tot - 1 else p_entry
    hit_3d_5 = bool(fwd_high_3 >= p_entry * 1.05)
    hit_3d_10 = bool(fwd_high_3 >= p_entry * 1.10)

    max_fwd_5 = min(tot - 1, b_end + 390)
    fwd_high_5 = np.max(highs[b_end + 1: max_fwd_5 + 1]) if b_end + 1 <= tot - 1 else p_entry
    hit_5d_10 = bool(fwd_high_5 >= p_entry * 1.10)

    max_gain_3d = float((fwd_high_3 - p_entry) / p_entry * 100)

    return {
        'symbol': s,
        'A': A['pattern'],
        'B': B['pattern'],
        'A_start': A['st_time'],
        'A_end': A['end_time'],
        'B_start': B['st_time'],
        'B_end': B['end_time'],
        'gap': int(gap),
        'gap_hours': round(gap * 5 / 60, 1),
        'dd': round(dd, 4),
        'amp': round(amp, 4),
        'higher': higher,
        'p_A': round(float(p_A), 2),
        'p_B_start': round(float(p_B), 2),
        'p_entry': round(float(p_entry), 2),
        'fwd_high_3': round(float(fwd_high_3), 2),
        'max_gain_3d': round(max_gain_3d, 2),
        'hit_1d_3': hit_1d_3,
        'hit_1d_5': hit_1d_5,
        'hit_3d_5': hit_3d_5,
        'hit_3d_10': hit_3d_10,
        'hit_5d_10': hit_5d_10,
        'is_qqq': is_qqq,
        'is_fav': is_fav,
        'is_held': is_held,
        'is_etf': is_etf,
        'is_semi': is_semi
    }


def process_scope_transitions(transitions_list):
    scope_data = {}

    for cond_key in INTERVAL_CONFIGS.keys():
        filtered_transitions = [t for t in transitions_list if test_interval(t, cond_key)]
        tot_filtered = len(filtered_transitions)

        # 64 combinations matrix
        matrix = {}
        combos_list = []

        for p_A in PATTERNS:
            matrix[p_A] = {}
            for p_B in PATTERNS:
                c_trans = [t for t in filtered_transitions if t['A'] == p_A and t['B'] == p_B]
                c_cnt = len(c_trans)

                if c_cnt > 0:
                    hit_1d_3 = sum(1 for t in c_trans if t['hit_1d_3'])
                    hit_1d_5 = sum(1 for t in c_trans if t['hit_1d_5'])
                    hit_3d_5 = sum(1 for t in c_trans if t['hit_3d_5'])
                    hit_3d_10 = sum(1 for t in c_trans if t['hit_3d_10'])
                    hit_5d_10 = sum(1 for t in c_trans if t['hit_5d_10'])

                    r_1d_3 = round(hit_1d_3 / c_cnt * 100, 1)
                    r_1d_5 = round(hit_1d_5 / c_cnt * 100, 1)
                    r_3d_5 = round(hit_3d_5 / c_cnt * 100, 1)
                    r_3d_10 = round(hit_3d_10 / c_cnt * 100, 1)
                    r_5d_10 = round(hit_5d_10 / c_cnt * 100, 1)
                    avg_gain_3d = round(float(np.mean([t['max_gain_3d'] for t in c_trans])), 2)

                    # Stock breakdown
                    stock_groups = {}
                    for t in c_trans:
                        sym = t['symbol']
                        if sym not in stock_groups:
                            stock_groups[sym] = {'count': 0, 'hit_3d_5': 0, 'hit_1d_3': 0, 'hit_1d_5': 0}
                        stock_groups[sym]['count'] += 1
                        if t['hit_3d_5']: stock_groups[sym]['hit_3d_5'] += 1
                        if t['hit_1d_3']: stock_groups[sym]['hit_1d_3'] += 1
                        if t['hit_1d_5']: stock_groups[sym]['hit_1d_5'] += 1

                    stocks_summary = []
                    for sym, s_stat in stock_groups.items():
                        stocks_summary.append({
                            'symbol': sym,
                            'count': s_stat['count'],
                            'hit_3d_5': s_stat['hit_3d_5'],
                            'rate_3d_5': round(s_stat['hit_3d_5'] / s_stat['count'] * 100, 1)
                        })
                    stocks_summary.sort(key=lambda x: (x['rate_3d_5'], x['count']), reverse=True)

                    # Sample historical events (up to 8)
                    sample_events = []
                    for t in c_trans[:8]:
                        sample_events.append({
                            'symbol': t['symbol'],
                            'A_start': t['A_start'],
                            'A_end': t['A_end'],
                            'B_start': t['B_start'],
                            'B_end': t['B_end'],
                            'gap_hours': t['gap_hours'],
                            'dd': round(t['dd'] * 100, 1),
                            'p_entry': t['p_entry'],
                            'max_gain_3d': t['max_gain_3d'],
                            'hit_3d_5': t['hit_3d_5'],
                            'hit_1d_3': t['hit_1d_3']
                        })

                    combo_stat = {
                        'A': p_A,
                        'B': p_B,
                        'count': c_cnt,
                        'hit_1d_3': hit_1d_3,
                        'rate_1d_3': r_1d_3,
                        'hit_1d_5': hit_1d_5,
                        'rate_1d_5': r_1d_5,
                        'hit_3d_5': hit_3d_5,
                        'rate_3d_5': r_3d_5,
                        'hit_3d_10': hit_3d_10,
                        'rate_3d_10': r_3d_10,
                        'hit_5d_10': hit_5d_10,
                        'rate_5d_10': r_5d_10,
                        'avg_gain_3d': avg_gain_3d,
                        'stocks_count': len(stock_groups),
                        'stocks': stocks_summary,
                        'sample_events': sample_events
                    }
                else:
                    combo_stat = {
                        'A': p_A,
                        'B': p_B,
                        'count': 0,
                        'hit_1d_3': 0,
                        'rate_1d_3': 0.0,
                        'hit_1d_5': 0,
                        'rate_1d_5': 0.0,
                        'hit_3d_5': 0,
                        'rate_3d_5': 0.0,
                        'hit_3d_10': 0,
                        'rate_3d_10': 0.0,
                        'hit_5d_10': 0,
                        'rate_5d_10': 0.0,
                        'avg_gain_3d': 0.0,
                        'stocks_count': 0,
                        'stocks': [],
                        'sample_events': []
                    }

                matrix[p_A][p_B] = combo_stat
                if c_cnt > 0:
                    combos_list.append(combo_stat)

        # Sort combos by 3d+5% win rate and count
        combos_list.sort(key=lambda x: (x['rate_3d_5'], x['count']), reverse=True)

        scope_data[cond_key] = {
            'total_transitions': tot_filtered,
            'active_combos_count': len([c for c in combos_list if c['count'] > 0]),
            'matrix': matrix,
            'leaderboard': combos_list[:12],
            'all_combos': combos_list
        }

    return scope_data


def generate_html(dataset):
    """Generate self-contained interactive HTML dashboard."""
    # Build lightweight summary for instant offline embedding
    summary_scopes = {}
    for scope_key, scope_val in dataset['scopes'].items():
        summary_scopes[scope_key] = {}
        for cond_key, cond_val in scope_val.items():
            mat = {}
            for pA, row in cond_val['matrix'].items():
                mat[pA] = {}
                for pB, c in row.items():
                    c_copy = dict(c)
                    c_copy.pop('sample_events', None)
                    mat[pA][pB] = c_copy

            combos = []
            for c in cond_val['all_combos']:
                c_copy = dict(c)
                c_copy.pop('sample_events', None)
                combos.append(c_copy)

            summary_scopes[scope_key][cond_key] = {
                'total_transitions': cond_val['total_transitions'],
                'active_combos_count': cond_val['active_combos_count'],
                'matrix': mat,
                'leaderboard': combos[:12],
                'all_combos': combos
            }

    embedded_payload = {
        'meta': dataset['meta'],
        'scopes': summary_scopes
    }
    json_embedded = json.dumps(embedded_payload, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>微观形态复合双核特征全排列分析看板 · A 【间隔 N】 B</title>
  <style>
    :root {{
      --bg-dark: #070c18;
      --bg-card: #0f172a;
      --bg-toolbar: #131d33;
      --border-color: #1e293b;
      --border-active: #38bdf8;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;

      --color-cyan: #06b6d4;
      --color-amber: #f59e0b;
      --color-lime: #84cc16;
      --color-emerald: #10b981;
      --color-rose: #e11d48;
      --color-purple: #8b5cf6;
      --color-orange: #f97316;
      --color-red: #dc2626;

      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-dark);
      color: var(--text-main);
      font-family: var(--font-sans);
      padding: 20px 28px;
      line-height: 1.5;
      font-size: 13px;
    }}

    /* Header */
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-color);
      flex-wrap: wrap;
      gap: 12px;
    }}
    .title-area h1 {{
      font-size: 22px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .badge-scope-tag {{
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.4);
      padding: 3px 9px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--font-mono);
      font-weight: 600;
    }}
    .badge-alpha-tag {{
      background: rgba(16, 185, 129, 0.16);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.4);
      padding: 3px 9px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--font-mono);
      font-weight: 600;
    }}
    .subtitle {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 6px;
    }}

    /* Control Bars */
    .control-bars {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 18px;
    }}

    .toolbar-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--bg-toolbar);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 16px;
      flex-wrap: wrap;
    }}
    .toolbar-label {{
      font-size: 12px;
      font-weight: 700;
      color: #f1f5f9;
      font-family: var(--font-mono);
      margin-right: 6px;
      display: flex;
      align-items: center;
      gap: 5px;
    }}

    .btn-pill {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #94a3b8;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
      white-space: nowrap;
    }}
    .btn-pill:hover {{
      color: #fff;
      border-color: #38bdf8;
    }}
    .btn-pill.active {{
      background: rgba(56, 189, 248, 0.22);
      color: #38bdf8;
      border-color: #38bdf8;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
    }}

    .btn-pill.active-emerald {{
      background: rgba(16, 185, 129, 0.22);
      color: #10b981;
      border-color: #10b981;
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.25);
    }}

    /* Interval Tab Groups */
    .interval-group {{
      display: flex;
      align-items: center;
      gap: 4px;
      background: rgba(15, 23, 42, 0.6);
      padding: 3px 6px;
      border-radius: 6px;
      border: 1px solid #24324d;
    }}
    .interval-cat-tag {{
      font-size: 10px;
      color: #94a3b8;
      font-weight: 700;
      padding: 0 4px;
      text-transform: uppercase;
    }}

    /* Stats Overview Cockpit */
    .cockpit-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .cockpit-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px 16px;
    }}
    .cockpit-title {{
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }}
    .cockpit-val {{
      font-size: 20px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: #fff;
    }}
    .cockpit-desc {{
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 3px;
    }}

    /* Main Sections Layout */
    .section-title {{
      font-size: 16px;
      font-weight: 700;
      color: #f8fafc;
      margin: 24px 0 12px 0;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    /* 8x8 Heatmap Matrix Container */
    .heatmap-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px 20px;
      overflow-x: auto;
      margin-bottom: 24px;
    }}
    .matrix-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 6px;
      font-family: var(--font-sans);
    }}
    .matrix-corner {{
      background: #111a2e;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 8px;
      font-size: 11px;
      font-weight: 700;
      color: #94a3b8;
      text-align: center;
      width: 130px;
      min-width: 130px;
    }}
    .matrix-col-th {{
      background: #111a2e;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 8px 4px;
      text-align: center;
      font-size: 11px;
      font-weight: 700;
      min-width: 105px;
    }}
    .matrix-row-th {{
      background: #111a2e;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 10px 10px;
      font-size: 12px;
      font-weight: 700;
      text-align: left;
      white-space: nowrap;
    }}

    .matrix-cell {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 10px 6px;
      text-align: center;
      cursor: pointer;
      transition: all 0.15s ease;
      position: relative;
    }}
    .matrix-cell:hover {{
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
      border-color: #38bdf8 !important;
      z-index: 10;
    }}
    .cell-val {{
      font-size: 15px;
      font-weight: 700;
      font-family: var(--font-mono);
      line-height: 1.1;
    }}
    .cell-sub {{
      font-size: 10px;
      color: #94a3b8;
      margin-top: 4px;
      font-family: var(--font-mono);
    }}
    .cell-diag {{
      position: absolute;
      top: 3px;
      right: 4px;
      font-size: 9px;
      color: #64748b;
    }}

    /* Standout Alpha Cards */
    .alpha-cards-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .alpha-card {{
      background: #111a2e;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 14px 18px;
      position: relative;
      cursor: pointer;
      transition: all 0.18s ease;
    }}
    .alpha-card:hover {{
      border-color: #10b981;
      transform: translateY(-2px);
      box-shadow: 0 4px 16px rgba(16, 185, 129, 0.15);
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .combo-title {{
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .card-rate-badge {{
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.4);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 13px;
      font-weight: 700;
      font-family: var(--font-mono);
    }}
    .ladder-bar {{
      display: flex;
      gap: 8px;
      margin: 10px 0;
      background: #090e1c;
      padding: 8px 12px;
      border-radius: 6px;
    }}
    .ladder-item {{
      flex: 1;
      text-align: center;
    }}
    .ladder-label {{
      font-size: 10px;
      color: var(--text-dim);
    }}
    .ladder-val {{
      font-size: 13px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: #f1f5f9;
      margin-top: 2px;
    }}
    .card-stocks {{
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }}
    .stock-chip {{
      background: rgba(56, 189, 248, 0.12);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 3px;
      padding: 1px 5px;
      font-size: 10px;
      font-family: var(--font-mono);
    }}

    /* Rankings Table */
    .table-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px 18px;
      overflow-x: auto;
      margin-bottom: 24px;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .data-table th {{
      background: #111a2e;
      color: #94a3b8;
      font-weight: 700;
      padding: 10px 12px;
      text-align: left;
      border-bottom: 2px solid #1e293b;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}
    .data-table th:hover {{
      color: #fff;
    }}
    .data-table td {{
      padding: 9px 12px;
      border-bottom: 1px solid #162033;
      white-space: nowrap;
    }}
    .data-table tr:hover td {{
      background: rgba(56, 189, 248, 0.05);
    }}

    /* Drawer / Modal */
    .drawer-overlay {{
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(4px);
      z-index: 1000;
      justify-content: flex-end;
    }}
    .drawer-overlay.active {{
      display: flex;
    }}
    .drawer-panel {{
      width: 680px;
      max-width: 90vw;
      background: #0d1527;
      height: 100%;
      border-left: 1px solid #1e293b;
      box-shadow: -8px 0 32px rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      animation: slideIn 0.2s ease-out;
    }}
    @keyframes slideIn {{
      from {{ transform: translateX(100%); }}
      to {{ transform: translateX(0); }}
    }}
    .drawer-header {{
      padding: 18px 22px;
      border-bottom: 1px solid #1e293b;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #111a2e;
    }}
    .drawer-title {{
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .drawer-close {{
      background: transparent;
      border: 1px solid #334155;
      color: #94a3b8;
      border-radius: 4px;
      padding: 4px 10px;
      cursor: pointer;
      font-size: 13px;
    }}
    .drawer-close:hover {{
      color: #fff;
      border-color: #e11d48;
    }}
    .drawer-content {{
      padding: 18px 22px;
      overflow-y: auto;
      flex: 1;
    }}

    /* Tooltip */
    #global-tooltip {{
      position: fixed;
      display: none;
      background: #0f172a;
      border: 1px solid #38bdf8;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 12px;
      color: #fff;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.7);
      pointer-events: none;
      z-index: 2000;
      max-width: 320px;
      line-height: 1.4;
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <div class="header">
    <div class="title-area">
      <h1>
        🧩 微观形态复合双核特征全排列分析看板
        <span class="badge-scope-tag">A 【间隔 N】 B</span>
        <span class="badge-alpha-tag">8×8 = 64 组形态对</span>
      </h1>
      <div class="subtitle">
        全域标的池 (138 只美股/ETF · QQQ + 自选 + 0086持仓) · 13 维结构/时空跨度约束 · 连续 3 个月 5 分钟 K 线量化挖掘
      </div>
    </div>
    <div style="text-align: right;">
      <span style="font-family: var(--font-mono); font-size: 12px; color: var(--text-dim);">
        回测周期: 2026-07-01 ~ 2026-09-29 · 5m Regular
      </span>
    </div>
  </div>

  <!-- Control Cockpit -->
  <div class="control-bars">

    <!-- Scope Row -->
    <div class="toolbar-row">
      <div class="toolbar-label">⚡ 承接范围 (Scope):</div>
      <button class="btn-pill active" id="btn-scope-consec" onclick="setScope('consecutive')">
        直接紧邻承接 (Direct Next · 1,713 笔)
      </button>
      <button class="btn-pill" id="btn-scope-all5d" onclick="setScope('all_5d')">
        5日衍生全排列 (All 5-Day · 9,006 笔)
      </button>
      <span style="margin-left: auto; color: var(--text-dim); font-size: 11px;">
        💡 紧邻承接：A结束后首个出现的新形态；5日全排列：5个交易日内任意后续形态
      </span>
    </div>

    <!-- Interval N Tabs Row -->
    <div class="toolbar-row" id="interval-tabs-bar">
      <div class="toolbar-label">⏳ 间隔条件 (N):</div>

      <!-- ALL -->
      <button class="btn-pill active" id="btn-int-ALL" onclick="setIntervalCond('ALL')">
        ALL (全量基准)
      </button>

      <!-- Time Gaps -->
      <div class="interval-group">
        <span class="interval-cat-tag">时间</span>
        <button class="btn-pill" id="btn-int-T1_IMM" onclick="setIntervalCond('T1_IMM')">T1 紧邻 ≤30m</button>
        <button class="btn-pill" id="btn-int-T2_SHORT" onclick="setIntervalCond('T2_SHORT')">T2 短憩 30m~2h</button>
        <button class="btn-pill" id="btn-int-T3_INTRADAY" onclick="setIntervalCond('T3_INTRADAY')">T3 日内 2h~1d</button>
        <button class="btn-pill" id="btn-int-T4_NEXT_DAY" onclick="setIntervalCond('T4_NEXT_DAY')">T4 次日 1d~2d</button>
        <button class="btn-pill" id="btn-int-T5_SWING" onclick="setIntervalCond('T5_SWING')">T5 波段 2d~5d</button>
      </div>

      <!-- Pullback DD -->
      <div class="interval-group">
        <span class="interval-cat-tag">回撤</span>
        <button class="btn-pill" id="btn-int-R1_SHALLOW" onclick="setIntervalCond('R1_SHALLOW')">R1 极浅 ≤2%</button>
        <button class="btn-pill" id="btn-int-R2_MODERATE" onclick="setIntervalCond('R2_MODERATE')">R2 中撤 2%~5%</button>
        <button class="btn-pill" id="btn-int-R3_DEEP" onclick="setIntervalCond('R3_DEEP')">R3 深坑 &gt;5%</button>
      </div>

      <!-- Volatility -->
      <div class="interval-group">
        <span class="interval-cat-tag">振幅</span>
        <button class="btn-pill" id="btn-int-V1_SQUEEZE" onclick="setIntervalCond('V1_SQUEEZE')">V1 收敛 ≤2.5%</button>
        <button class="btn-pill" id="btn-int-V2_VOLATILE" onclick="setIntervalCond('V2_VOLATILE')">V2 宽幅 &gt;5%</button>
      </div>

      <!-- Structural Level -->
      <div class="interval-group">
        <span class="interval-cat-tag">重心</span>
        <button class="btn-pill" id="btn-int-S1_HIGHER" onclick="setIntervalCond('S1_HIGHER')">S1 抬高 Pb≥Pa</button>
        <button class="btn-pill" id="btn-int-S2_LOWER" onclick="setIntervalCond('S2_LOWER')">S2 下移 Pb&lt;Pa</button>
      </div>
    </div>

    <!-- Metric & Filter Row -->
    <div class="toolbar-row">
      <div class="toolbar-label">🎯 观察指标 (Metric):</div>
      <button class="btn-pill active-emerald" id="btn-met-rate_3d_5" onclick="setMetric('rate_3d_5')">
        ★ 后续 3天 +5% 达成率
      </button>
      <button class="btn-pill" id="btn-met-rate_1d_3" onclick="setMetric('rate_1d_3')">后续 1天 +3%</button>
      <button class="btn-pill" id="btn-met-rate_1d_5" onclick="setMetric('rate_1d_5')">后续 1天 +5%</button>
      <button class="btn-pill" id="btn-met-rate_3d_10" onclick="setMetric('rate_3d_10')">后续 3天 +10%</button>
      <button class="btn-pill" id="btn-met-rate_5d_10" onclick="setMetric('rate_5d_10')">后续 5天 +10%</button>
      <button class="btn-pill" id="btn-met-avg_gain_3d" onclick="setMetric('avg_gain_3d')">3日最大涨幅均值</button>
      <button class="btn-pill" id="btn-met-count" onclick="setMetric('count')">样本触发频次 N</button>

      <div style="margin-left: auto; display: flex; align-items: center; gap: 8px;">
        <span class="toolbar-label">样本门槛:</span>
        <select id="sel-min-count" onchange="setMinCount(this.value)" style="background:#0f172a; border:1px solid #334155; color:#fff; border-radius:4px; padding:4px 8px; font-size:12px;">
          <option value="1">全部 (≥1次)</option>
          <option value="5">初筛 (≥5次)</option>
          <option value="10" selected>推荐高信度 (≥10次)</option>
          <option value="20">统计稳健 (≥20次)</option>
        </select>
      </div>
    </div>
  </div>

  <!-- Cockpit KPI Banner -->
  <div class="cockpit-grid">
    <div class="cockpit-card">
      <div class="cockpit-title">当前间隔有效样本数</div>
      <div class="cockpit-val" id="kpi-total-trans">--</div>
      <div class="cockpit-desc" id="kpi-interval-desc">--</div>
    </div>
    <div class="cockpit-card">
      <div class="cockpit-title">活跃形态组合覆盖</div>
      <div class="cockpit-val" id="kpi-active-combos">-- / 64</div>
      <div class="cockpit-desc">全排列矩阵有效填充率</div>
    </div>
    <div class="cockpit-card">
      <div class="cockpit-title">Top 1 复合形态胜率王</div>
      <div class="cockpit-val" id="kpi-top-combo" style="color:#10b981; font-size:16px;">--</div>
      <div class="cockpit-desc" id="kpi-top-rate">--</div>
    </div>
    <div class="cockpit-card">
      <div class="cockpit-title">全矩阵胜率中位数</div>
      <div class="cockpit-val" id="kpi-median-rate" style="color:#38bdf8;">--</div>
      <div class="cockpit-desc">3天 +5% 整体基准达成水平</div>
    </div>
  </div>

  <!-- Section 1: 8x8 Permutation Heatmap Matrix -->
  <div class="section-title">
    <span>📊 8 × 8 复合微观形态热力矩阵 (前置形态 A × 后续形态 B)</span>
    <span style="font-size: 11px; font-weight: normal; color: var(--text-dim);">
      纵轴：前置触发形态 A · 横轴：后续发生形态 B · 格内数据随上方指标切换
    </span>
  </div>
  <div class="heatmap-card">
    <table class="matrix-table" id="matrix-table">
      <!-- Generated by JavaScript -->
    </table>
  </div>

  <!-- Section 2: Top Alpha Standout Combinations -->
  <div class="section-title">
    <span>🏆 高信度复合形态杀手锏榜单 (Top Alpha Leaders)</span>
    <span style="font-size: 11px; font-weight: normal; color: var(--text-dim);" id="leaderboard-subtitle">
      满足样本门槛的高胜率形态序列
    </span>
  </div>
  <div class="alpha-cards-grid" id="alpha-cards-container">
    <!-- Generated by JavaScript -->
  </div>

  <!-- Section 3: Full 64 Combinations Table -->
  <div class="section-title">
    <span>📋 全部 64 组全排列组合明细统计 (可排序 & 筛选)</span>
    <input type="text" id="filter-search" placeholder="🔍 搜索形态名称或股票代码 (如 NVDA, CRDO, 深弹)..." oninput="renderTable()" style="margin-left: auto; background:#0f172a; border:1px solid #334155; color:#fff; padding:5px 12px; border-radius:6px; font-size:12px; width:280px;">
  </div>
  <div class="table-card">
    <table class="data-table" id="rankings-table">
      <thead>
        <tr>
          <th onclick="sortTable('rank')">#</th>
          <th onclick="sortTable('combo')">复合形态组合 (A → B)</th>
          <th onclick="sortTable('role')">转化机制</th>
          <th onclick="sortTable('count')">样本量 N ⬍</th>
          <th onclick="sortTable('rate_1d_3')">1天 +3% ⬍</th>
          <th onclick="sortTable('rate_1d_5')">1天 +5% ⬍</th>
          <th onclick="sortTable('rate_3d_5')">★ 3天 +5% ⬍</th>
          <th onclick="sortTable('rate_3d_10')">3天 +10% ⬍</th>
          <th onclick="sortTable('rate_5d_10')">5天 +10% ⬍</th>
          <th onclick="sortTable('avg_gain_3d')">3日均涨幅 ⬍</th>
          <th onclick="sortTable('stocks_count')">覆盖标的数 ⬍</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody id="table-tbody">
        <!-- Generated by JavaScript -->
      </tbody>
    </table>
  </div>

  <!-- Drawer Modal -->
  <div class="drawer-overlay" id="drawer-overlay" onclick="closeDrawer(event)">
    <div class="drawer-panel" onclick="event.stopPropagation()">
      <div class="drawer-header">
        <div class="drawer-title" id="drawer-title">复合形态明细穿透</div>
        <button class="drawer-close" onclick="closeDrawer()">✕ 关闭</button>
      </div>
      <div class="drawer-content" id="drawer-content">
        <!-- Populated dynamically -->
      </div>
    </div>
  </div>

  <!-- Tooltip -->
  <div id="global-tooltip"></div>

  <!-- Application Script -->
  <script>
    // Embedded Data Payload (Standalone Offline Ready)
    const EMBEDDED_DATA = {json_embedded};
    let appData = EMBEDDED_DATA;

    // Optional dynamic enhancement: load extended historical events from JSON if available via HTTP
    fetch('./ab_permutation_data.json')
      .then(res => res.json())
      .then(fullData => {{
        appData = fullData;
        console.log('Successfully loaded full extended event logs from ab_permutation_data.json.');
      }})
      .catch(err => {{
        console.log('Running with embedded dataset.');
      }});

    // State
    let currentScope = 'consecutive';
    let currentInterval = 'ALL';
    let currentMetric = 'rate_3d_5';
    let currentMinCount = 10;
    let sortColumn = 'rate_3d_5';
    let sortAsc = false;

    const PATTERNS_ORDER = appData.meta.patterns;
    const PATTERN_META = appData.meta.pattern_meta;
    const INTERVALS = appData.meta.intervals;

    // Metric display configs
    const METRIC_CONFIGS = {{
      'rate_3d_5': {{ name: '★ 3天+5% 达成率', unit: '%', isRate: true }},
      'rate_1d_3': {{ name: '1天+3% 达成率', unit: '%', isRate: true }},
      'rate_1d_5': {{ name: '1天+5% 达成率', unit: '%', isRate: true }},
      'rate_3d_10': {{ name: '3天+10% 达成率', unit: '%', isRate: true }},
      'rate_5d_10': {{ name: '5天+10% 达成率', unit: '%', isRate: true }},
      'avg_gain_3d': {{ name: '3日最大涨幅均值', unit: '%', isRate: false }},
      'count': {{ name: '触发样本量 N', unit: '次', isRate: false }}
    }};

    function setScope(scope) {{
      currentScope = scope;
      document.getElementById('btn-scope-consec').className = (scope === 'consecutive') ? 'btn-pill active' : 'btn-pill';
      document.getElementById('btn-scope-all5d').className = (scope === 'all_5d') ? 'btn-pill active' : 'btn-pill';
      refreshAll();
    }}

    function setIntervalCond(condKey) {{
      currentInterval = condKey;
      Object.keys(INTERVALS).forEach(k => {{
        const btn = document.getElementById('btn-int-' + k);
        if (btn) {{
          btn.className = (k === condKey) ? 'btn-pill active' : 'btn-pill';
        }}
      }});
      refreshAll();
    }}

    function setMetric(metricKey) {{
      currentMetric = metricKey;
      Object.keys(METRIC_CONFIGS).forEach(k => {{
        const btn = document.getElementById('btn-met-' + k);
        if (btn) {{
          btn.className = (k === metricKey) ? 'btn-pill active-emerald' : 'btn-pill';
        }}
      }});
      renderHeatmap();
    }}

    function setMinCount(val) {{
      currentMinCount = parseInt(val, 10);
      renderLeaderboard();
      renderTable();
    }}

    function getHeatmapColor(val, metric, count) {{
      if (count === 0) return {{ bg: 'rgba(30, 41, 59, 0.25)', border: '#1e293b', text: '#64748b' }};
      if (metric === 'count') {{
        if (val >= 60) return {{ bg: 'rgba(56, 189, 248, 0.45)', border: '#38bdf8', text: '#fff' }};
        if (val >= 30) return {{ bg: 'rgba(56, 189, 248, 0.3)', border: '#0284c7', text: '#f0f9ff' }};
        if (val >= 10) return {{ bg: 'rgba(56, 189, 248, 0.18)', border: '#0369a1', text: '#e0f2fe' }};
        return {{ bg: 'rgba(56, 189, 248, 0.08)', border: '#075985', text: '#94a3b8' }};
      }}
      if (metric === 'avg_gain_3d') {{
        if (val >= 25) return {{ bg: 'rgba(16, 185, 129, 0.45)', border: '#10b981', text: '#fff' }};
        if (val >= 15) return {{ bg: 'rgba(16, 185, 129, 0.3)', border: '#059669', text: '#ecfdf5' }};
        if (val >= 8) return {{ bg: 'rgba(16, 185, 129, 0.18)', border: '#047857', text: '#d1fae5' }};
        return {{ bg: 'rgba(245, 158, 11, 0.15)', border: '#d97706', text: '#fde68a' }};
      }}
      // Rate metric (0% - 100%)
      if (val >= 80) return {{ bg: 'rgba(16, 185, 129, 0.45)', border: '#10b981', text: '#fff' }};
      if (val >= 70) return {{ bg: 'rgba(6, 182, 212, 0.35)', border: '#06b6d4', text: '#ecfeff' }};
      if (val >= 55) return {{ bg: 'rgba(59, 130, 246, 0.28)', border: '#3b82f6', text: '#eff6ff' }};
      if (val >= 40) return {{ bg: 'rgba(245, 158, 11, 0.22)', border: '#f59e0b', text: '#fffbeb' }};
      return {{ bg: 'rgba(225, 29, 72, 0.2)', border: '#e11d48', text: '#ffe4e6' }};
    }}

    function refreshAll() {{
      renderKPIs();
      renderHeatmap();
      renderLeaderboard();
      renderTable();
    }}

    function renderKPIs() {{
      const scopeData = appData.scopes[currentScope][currentInterval];
      const intMeta = INTERVALS[currentInterval];

      document.getElementById('kpi-total-trans').innerText = scopeData.total_transitions.toLocaleString() + ' 笔';
      document.getElementById('kpi-interval-desc').innerText = intMeta.name + ' · ' + intMeta.desc;
      document.getElementById('kpi-active-combos').innerText = scopeData.active_combos_count + ' / 64';

      const validCombos = scopeData.all_combos.filter(c => c.count >= currentMinCount);
      if (validCombos.length > 0) {{
        const top = validCombos[0];
        document.getElementById('kpi-top-combo').innerHTML = `${{top.A}} → ${{top.B}}`;
        document.getElementById('kpi-top-rate').innerText = `3天+5%: ${{top.rate_3d_5}}% (N=${{top.count}})`;

        const rates = validCombos.map(c => c.rate_3d_5).sort((a,b) => a - b);
        const med = rates[Math.floor(rates.length / 2)];
        document.getElementById('kpi-median-rate').innerText = med.toFixed(1) + '%';
      }} else {{
        document.getElementById('kpi-top-combo').innerText = '无满足门槛组合';
        document.getElementById('kpi-top-rate').innerText = '请调低样本门槛';
        document.getElementById('kpi-median-rate').innerText = '--';
      }}
    }}

    function renderHeatmap() {{
      const scopeData = appData.scopes[currentScope][currentInterval];
      const matrix = scopeData.matrix;
      const table = document.getElementById('matrix-table');

      let html = '<thead><tr>';
      html += '<th class="matrix-corner">A (前) \\ B (后)</th>';

      // Header row for B
      PATTERNS_ORDER.forEach(pB => {{
        const meta = PATTERN_META[pB];
        html += `<th class="matrix-col-th" style="color:${{meta.color}};">
          ${{pB}}<br><span style="font-size:9px; color:${{meta.color}}99;">${{meta.role}}</span>
        </th>`;
      }});
      html += '</tr></thead><tbody>';

      // Rows for A
      PATTERNS_ORDER.forEach(pA => {{
        const metaA = PATTERN_META[pA];
        html += `<tr><th class="matrix-row-th" style="color:${{metaA.color}};">
          ${{pA}}<br><span style="font-size:10px; color:${{metaA.color}}99;">${{metaA.role}}</span>
        </th>`;

        PATTERNS_ORDER.forEach(pB => {{
          const cell = matrix[pA][pB];
          const val = cell[currentMetric];
          const colors = getHeatmapColor(val, currentMetric, cell.count);
          const isDiag = (pA === pB);

          let displayVal = cell.count > 0 ? (METRIC_CONFIGS[currentMetric].isRate ? val.toFixed(1) + '%' : (currentMetric === 'count' ? val : '+' + val.toFixed(1) + '%')) : '-';

          html += `<td class="matrix-cell" style="background:${{colors.bg}}; border-color:${{colors.border}};"
            onmouseenter="showTooltip(event, '${{pA}}', '${{pB}}')"
            onmousemove="moveTooltip(event)"
            onmouseleave="hideTooltip()"
            onclick="openDrawer('${{pA}}', '${{pB}}')">
            ${{isDiag ? '<span class="cell-diag">同</span>' : ''}}
            <div class="cell-val" style="color:${{colors.text}};">${{displayVal}}</div>
            <div class="cell-sub">${{cell.count > 0 ? 'N=' + cell.count : '无样本'}}</div>
          </td>`;
        }});

        html += '</tr>';
      }});

      html += '</tbody>';
      table.innerHTML = html;
    }}

    function renderLeaderboard() {{
      const scopeData = appData.scopes[currentScope][currentInterval];
      const container = document.getElementById('alpha-cards-container');
      const validCombos = scopeData.all_combos.filter(c => c.count >= currentMinCount);

      if (validCombos.length === 0) {{
        container.innerHTML = '<div style="color:var(--text-muted); grid-column:1/-1; padding:20px; background:#111a2e; border-radius:8px; text-align:center;">当前间隔条件下，暂无样本数 ≥ ' + currentMinCount + ' 的组合，请调低样本门槛</div>';
        return;
      }}

      let html = '';
      validCombos.slice(0, 6).forEach((c, idx) => {{
        const metaA = PATTERN_META[c.A];
        const metaB = PATTERN_META[c.B];

        html += `<div class="alpha-card" onclick="openDrawer('${{c.A}}', '${{c.B}}')">
          <div class="card-top">
            <div class="combo-title">
              <span style="color:${{metaA.color}};">${{c.A}}</span>
              <span style="color:var(--text-dim); font-size:11px;">➔</span>
              <span style="color:${{metaB.color}};">${{c.B}}</span>
            </div>
            <div class="card-rate-badge">★ 3天+5%: ${{c.rate_3d_5}}%</div>
          </div>
          <div class="ladder-bar">
            <div class="ladder-item">
              <div class="ladder-label">1天+3%</div>
              <div class="ladder-val">${{c.rate_1d_3}}%</div>
            </div>
            <div class="ladder-item">
              <div class="ladder-label">1天+5%</div>
              <div class="ladder-val">${{c.rate_1d_5}}%</div>
            </div>
            <div class="ladder-item">
              <div class="ladder-label">3天+10%</div>
              <div class="ladder-val">${{c.rate_3d_10}}%</div>
            </div>
            <div class="ladder-item">
              <div class="ladder-label">3日均涨</div>
              <div class="ladder-val">+${{c.avg_gain_3d}}%</div>
            </div>
          </div>
          <div style="font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between; margin-top:6px;">
            <span>触发样本: ${{c.count}} 笔 (${{c.hit_3d_5}} 胜)</span>
            <span>涉及标的: ${{c.stocks_count}} 只</span>
          </div>
          <div class="card-stocks">
            ${{c.stocks.slice(0, 5).map(s => `<span class="stock-chip">${{s.symbol}} (${{s.rate_3d_5}}%)</span>`).join('')}}
          </div>
        </div>`;
      }});

      container.innerHTML = html;
    }}

    function renderTable() {{
      const scopeData = appData.scopes[currentScope][currentInterval];
      const tbody = document.getElementById('table-tbody');
      const search = document.getElementById('filter-search').value.trim().toUpperCase();

      let list = scopeData.all_combos.slice();

      if (search) {{
        list = list.filter(c => {{
          return c.A.includes(search) || c.B.includes(search) ||
            c.stocks.some(s => s.symbol.includes(search));
        }});
      }}

      // Sort
      list.sort((a, b) => {{
        let vA = a[sortColumn];
        let vB = b[sortColumn];
        if (sortColumn === 'combo') {{
          vA = a.A + a.B;
          vB = b.A + b.B;
        }}
        if (vA < vB) return sortAsc ? -1 : 1;
        if (vA > vB) return sortAsc ? 1 : -1;
        return 0;
      }});

      let html = '';
      list.forEach((c, idx) => {{
        const metaA = PATTERN_META[c.A];
        const metaB = PATTERN_META[c.B];

        html += `<tr>
          <td style="color:var(--text-dim); font-family:var(--font-mono);">${{idx + 1}}</td>
          <td>
            <span style="font-weight:700; color:${{metaA.color}};">${{c.A}}</span>
            <span style="color:var(--text-dim); margin:0 4px;">➔</span>
            <span style="font-weight:700; color:${{metaB.color}};">${{c.B}}</span>
          </td>
          <td style="color:var(--text-muted); font-size:11px;">
            ${{metaA.role}} ➔ ${{metaB.role}}
          </td>
          <td style="font-family:var(--font-mono); font-weight:700;">${{c.count}}</td>
          <td style="font-family:var(--font-mono); color:${{c.rate_1d_3 >= 70 ? '#10b981' : '#f1f5f9'}};">${{c.rate_1d_3}}%</td>
          <td style="font-family:var(--font-mono); color:${{c.rate_1d_5 >= 60 ? '#10b981' : '#f1f5f9'}};">${{c.rate_1d_5}}%</td>
          <td style="font-family:var(--font-mono); font-weight:700; color:${{c.rate_3d_5 >= 75 ? '#10b981' : (c.rate_3d_5 >= 60 ? '#38bdf8' : '#f59e0b')}};">
            ${{c.rate_3d_5}}% (${{c.hit_3d_5}}/${{c.count}})
          </td>
          <td style="font-family:var(--font-mono); color:${{c.rate_3d_10 >= 50 ? '#10b981' : '#f1f5f9'}};">${{c.rate_3d_10}}%</td>
          <td style="font-family:var(--font-mono);">${{c.rate_5d_10}}%</td>
          <td style="font-family:var(--font-mono); color:#10b981;">+${{c.avg_gain_3d}}%</td>
          <td style="font-family:var(--font-mono);">${{c.stocks_count}}</td>
          <td>
            <button class="btn-pill" style="padding:2px 8px; font-size:11px;" onclick="openDrawer('${{c.A}}', '${{c.B}}')">
              穿透明细
            </button>
          </td>
        </tr>`;
      }});

      tbody.innerHTML = html;
    }}

    function sortTable(col) {{
      if (sortColumn === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortColumn = col;
        sortAsc = false;
      }}
      renderTable();
    }}

    function showTooltip(e, pA, pB) {{
      const cell = appData.scopes[currentScope][currentInterval].matrix[pA][pB];
      const tip = document.getElementById('global-tooltip');
      const metaA = PATTERN_META[pA];
      const metaB = PATTERN_META[pB];

      tip.innerHTML = `
        <div style="font-weight:700; font-size:13px; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
          <span style="color:${{metaA.color}};">${{pA}}</span>
          <span style="color:var(--text-dim);">➔</span>
          <span style="color:${{metaB.color}};">${{pB}}</span>
        </div>
        <div style="color:var(--text-muted); font-size:11px; margin-bottom:6px;">
          当前约束: ${{INTERVALS[currentInterval].name}}
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px 8px; font-size:11px;">
          <div>样本频次: <b style="color:#fff;">${{cell.count}} 笔</b></div>
          <div>3日均涨: <b style="color:#10b981;">+${{cell.avg_gain_3d}}%</b></div>
          <div>后续 1天+3%: <b style="color:#fff;">${{cell.rate_1d_3}}%</b></div>
          <div>后续 1天+5%: <b style="color:#fff;">${{cell.rate_1d_5}}%</b></div>
          <div>★ 3天+5%: <b style="color:#10b981;">${{cell.rate_3d_5}}%</b></div>
          <div>后续 3天+10%: <b style="color:#fff;">${{cell.rate_3d_10}}%</b></div>
          <div>后续 5天+10%: <b style="color:#fff;">${{cell.rate_5d_10}}%</b></div>
          <div>涉及标的: <b style="color:#38bdf8;">${{cell.stocks_count}} 只</b></div>
        </div>
        <div style="margin-top:8px; padding-top:6px; border-top:1px solid #334155; font-size:10px; color:#38bdf8;">
          👉 点击单元格穿透查看个股分布与历史实盘事件
        </div>
      `;
      tip.style.display = 'block';
      moveTooltip(e);
    }}

    function moveTooltip(e) {{
      const tip = document.getElementById('global-tooltip');
      const pad = 12;
      let left = e.clientX + pad;
      let top = e.clientY + pad;

      if (left + 330 > window.innerWidth) {{
        left = e.clientX - 330;
      }}
      if (top + 200 > window.innerHeight) {{
        top = e.clientY - 200;
      }}

      tip.style.left = left + 'px';
      tip.style.top = top + 'px';
    }}

    function hideTooltip() {{
      document.getElementById('global-tooltip').style.display = 'none';
    }}

    function openDrawer(pA, pB) {{
      hideTooltip();
      const scopeData = appData.scopes[currentScope][currentInterval];
      const cell = scopeData.matrix[pA][pB];
      const metaA = PATTERN_META[pA];
      const metaB = PATTERN_META[pB];
      const intMeta = INTERVALS[currentInterval];

      const drawer = document.getElementById('drawer-overlay');
      const title = document.getElementById('drawer-title');
      const content = document.getElementById('drawer-content');

      title.innerHTML = `
        <span style="color:${{metaA.color}};">${{pA}}</span>
        <span style="color:var(--text-dim); margin:0 4px;">➔</span>
        <span style="color:${{metaB.color}};">${{pB}}</span>
        <span style="font-size:11px; background:#1e293b; color:#94a3b8; padding:2px 8px; border-radius:4px; font-weight:normal;">
          ${{intMeta.badge}}
        </span>
      `;

      let html = `
        <!-- KPI Cockpit in Drawer -->
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; margin-bottom:18px;">
          <div style="background:#111a2e; padding:10px; border-radius:6px; border:1px solid #1e293b; text-align:center;">
            <div style="font-size:10px; color:var(--text-dim);">样本量 / 命中数</div>
            <div style="font-size:16px; font-weight:700; color:#fff; font-family:var(--font-mono); margin-top:2px;">
              ${{cell.hit_3d_5}} / ${{cell.count}}
            </div>
          </div>
          <div style="background:#111a2e; padding:10px; border-radius:6px; border:1px solid #1e293b; text-align:center;">
            <div style="font-size:10px; color:var(--text-dim);">★ 3天+5% 达成率</div>
            <div style="font-size:16px; font-weight:700; color:#10b981; font-family:var(--font-mono); margin-top:2px;">
              ${{cell.rate_3d_5}}%
            </div>
          </div>
          <div style="background:#111a2e; padding:10px; border-radius:6px; border:1px solid #1e293b; text-align:center;">
            <div style="font-size:10px; color:var(--text-dim);">3日最大涨幅均值</div>
            <div style="font-size:16px; font-weight:700; color:#38bdf8; font-family:var(--font-mono); margin-top:2px;">
              +${{cell.avg_gain_3d}}%
            </div>
          </div>
        </div>

        <div style="background:#111a2e; border:1px solid #1e293b; border-radius:6px; padding:12px; margin-bottom:18px;">
          <div style="font-size:12px; font-weight:700; color:#f1f5f9; margin-bottom:6px;">📈 全梯度达成表现</div>
          <div style="display:flex; justify-content:space-between; font-size:11px; font-family:var(--font-mono);">
            <div>1天+3%: <b>${{cell.rate_1d_3}}%</b> (${{cell.hit_1d_3}}/${{cell.count}})</div>
            <div>1天+5%: <b>${{cell.rate_1d_5}}%</b> (${{cell.hit_1d_5}}/${{cell.count}})</div>
            <div>3天+10%: <b>${{cell.rate_3d_10}}%</b> (${{cell.hit_3d_10}}/${{cell.count}})</div>
            <div>5天+10%: <b>${{cell.rate_5d_10}}%</b> (${{cell.hit_5d_10}}/${{cell.count}})</div>
          </div>
        </div>

        <!-- Stock Breakdown -->
        <div style="font-size:13px; font-weight:700; color:#fff; margin-bottom:10px;">
          🏛️ 触发个股分布与胜率统计 (${{cell.stocks.length}} 只标的)
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:8px; margin-bottom:20px;">
          ${{cell.stocks.map(s => {{
            const tMeta = appData.meta.ticker_meta[s.symbol] || {{ name: s.symbol, sector: '美股标的' }};
            return `<div style="background:#111a2e; border:1px solid #1e293b; border-radius:6px; padding:8px 10px; display:flex; justify-content:space-between; align-items:center;">
              <div>
                <b style="color:#38bdf8; font-family:var(--font-mono);">${{s.symbol}}</b>
                <span style="font-size:10px; color:var(--text-dim); margin-left:4px;">${{tMeta.name.split(' ')[0]}}</span>
              </div>
              <div style="text-align:right;">
                <span style="font-weight:700; font-family:var(--font-mono); color:${{s.rate_3d_5 >= 75 ? '#10b981' : '#f1f5f9'}};">
                  ${{s.rate_3d_5}}%
                </span>
                <span style="font-size:10px; color:var(--text-dim); margin-left:4px;">(${{s.hit_3d_5}}/${{s.count}})</span>
              </div>
            </div>`;
          }}).join('')}}
        </div>
      `;

      // Sample events if present
      if (cell.sample_events && cell.sample_events.length > 0) {{
        html += `
          <div style="font-size:13px; font-weight:700; color:#fff; margin-bottom:10px;">
            🔍 历史实盘触发事件还原 (展示前 ${{cell.sample_events.length}} 笔)
          </div>
          <table style="width:100%; border-collapse:collapse; font-size:11px; font-family:var(--font-mono);">
            <thead>
              <tr style="background:#111a2e; color:#94a3b8; text-align:left;">
                <th style="padding:6px 8px;">标的</th>
                <th style="padding:6px 8px;">形态A时点</th>
                <th style="padding:6px 8px;">间隔(小时/回撤)</th>
                <th style="padding:6px 8px;">形态B时点</th>
                <th style="padding:6px 8px;">入场价</th>
                <th style="padding:6px 8px;">3日均涨</th>
                <th style="padding:6px 8px;">3d+5%</th>
              </tr>
            </thead>
            <tbody>
              ${{cell.sample_events.map(ev => `
                <tr style="border-bottom:1px solid #1e293b;">
                  <td style="padding:6px 8px; color:#38bdf8; font-weight:700;">${{ev.symbol}}</td>
                  <td style="padding:6px 8px; color:var(--text-muted);">${{ev.A_start.slice(5, 16)}}</td>
                  <td style="padding:6px 8px; color:var(--text-dim);">${{ev.gap_hours}}h (${{ev.dd}}%)</td>
                  <td style="padding:6px 8px; color:var(--text-muted);">${{ev.B_start.slice(5, 16)}}</td>
                  <td style="padding:6px 8px;">$${{ev.p_entry}}</td>
                  <td style="padding:6px 8px; color:#10b981; font-weight:700;">+${{ev.max_gain_3d}}%</td>
                  <td style="padding:6px 8px;">
                    ${{ev.hit_3d_5 ? '<span style="color:#10b981; font-weight:700;">✓ 达成</span>' : '<span style="color:#e11d48;">✗ 未达</span>'}}
                  </td>
                </tr>
              `).join('')}}
            </tbody>
          </table>
        `;
      }}

      content.innerHTML = html;
      drawer.className = 'drawer-overlay active';
    }}

    function closeDrawer() {{
      document.getElementById('drawer-overlay').className = 'drawer-overlay';
    }}

    // Close drawer on Esc key
    window.addEventListener('keydown', e => {{
      if (e.key === 'Escape') closeDrawer();
    }});

    // Initialize
    refreshAll();
  </script>
</body>
</html>
"""

    with open(HTML_OUT, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    run_full_permutation_analysis()
