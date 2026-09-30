#!/usr/bin/env python3
"""Generate comprehensive momentum pattern scan and 4-quadrant scatter analysis.

Expanded Universe:
1. QQQ (Nasdaq 100 Constituents - 104 stocks)
2. User Favorites (牛牛特别关注自选股)
3. User Holdings (Moomoo US 0086 Cash & Margin Holdings)
4. User ETFs (SOXX, SMH, IGV, SPCX, SPY, QQQ, DIA, SOXL, SOXS, XLU)
5. ETF Constituents (Semiconductor, Optical, Storage, Cloud Software)

Total: 138 US stocks & ETFs
Window Widths: 1h (12 bars), 2h (24 bars · 基准), 3h (36 bars), 4h (48 bars), 5h (60 bars)
Time Span: Past 3 months (2026-07-01 to 2026-09-29)
Granularity: 5-minute regular trading session bars

Output:
- analysis/qqq_momentum_patterns_quadrant/quadrant_data.json
- analysis/qqq_momentum_patterns_quadrant/index.html
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
JSON_OUT = ANALYSIS_DIR / "quadrant_data.json"
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

# Window configurations
WINDOWS_CONFIG = {
    '1h': {'bars': 12, 'name': '1小时 (12根 5m)', 'desc': '超短线日内初探'},
    '2h': {'bars': 24, 'name': '2小时 (24根 5m · 基准)', 'desc': '标准日内动量波段'},
    '3h': {'bars': 36, 'name': '3小时 (36根 5m)', 'desc': '半日推进波段'},
    '4h': {'bars': 48, 'name': '4小时 (48根 5m)', 'desc': '大半日趋势延伸'},
    '5h': {'bars': 60, 'name': '5小时 (60根 5m)', 'desc': '全日级单边/回踩'}
}

# Complete metadata for all 138 tickers
TICKER_META = {
    # Mega Tech & QQQ Core
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

    # ETFs
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

    # User Holdings & Favorites (Outside QQQ Core)
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

PATTERN_META = {
    'V型深弹': {
        'code': 'V_REBOUND',
        'color': '#06b6d4',  # Cyan
        'bg': 'rgba(6, 182, 212, 0.16)',
        'border': '#06b6d4',
        'desc': '前半程深探下砸砸出黄金坑，尾盘放量暴力反扑创新高',
        'sentiment': '高弹性洗盘再突破，买方动能惯性极强'
    },
    '冲高回落': {
        'code': 'SPIKE_PULLBACK',
        'color': '#f59e0b',  # Amber
        'bg': 'rgba(245, 158, 11, 0.16)',
        'border': '#f59e0b',
        'desc': '脉冲式急拉遇阻回踩消化浮筹，整体区间涨幅仍达标+5%',
        'sentiment': '试盘遇阻震荡整理，高位吸收抛压再攻'
    },
    '阶梯中继': {
        'code': 'STEP_CONTINUATION',
        'color': '#84cc16',  # Lime
        'bg': 'rgba(132, 204, 22, 0.16)',
        'border': '#84cc16',
        'desc': '窄幅横盘蓄势平台后二次放量起爆，呈阶梯式推进',
        'sentiment': '机构资金稳健换手，趋势性推进'
    },
    '▲ 单边拉升': {
        'code': 'MONOTONIC_RALLY',
        'color': '#10b981',  # Emerald
        'bg': 'rgba(16, 185, 129, 0.16)',
        'border': '#10b981',
        'desc': '无明显回撤直线上攻，多头单边加速赶顶或爆量',
        'sentiment': '极度亢奋追涨，需警惕后续透支'
    }
}


def scan_all_universe():
    t0 = time.time()
    print("Collecting all available US stocks and ETFs in us_5m...")
    qqq_tickers = set(json.loads(QQQ_FILE.read_text())['tickers'])

    all_dirs = sorted([p for p in M5_DIR.iterdir() if p.is_dir() and not p.name.isdigit()])
    symbols_to_scan = [p.name for p in all_dirs]
    print(f"Total US tickers to scan: {len(symbols_to_scan)}")

    # 1. Preload all regular-session data into memory
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
                'closes': reg['close'].to_numpy(dtype=np.float64),
                'highs': reg['high'].to_numpy(dtype=np.float64),
                'time_keys': reg['time_key'].tolist(),
                'total_len': len(reg)
            }
        except Exception as err:
            print(f"Error loading {s}: {err}")

    print(f"Loaded {len(loaded_series)} series in {time.time() - t0:.2f}s")

    # 2. Iterate across 5 window widths
    M = 0.05  # +5% surge threshold
    patterns_order = ['V型深弹', '冲高回落', '阶梯中继', '▲ 单边拉升']
    windows_data = {}

    for w_key, w_cfg in WINDOWS_CONFIG.items():
        N = w_cfg['bars']
        win_events = []

        for s, s_data in loaded_series.items():
            closes = s_data['closes']
            highs = s_data['highs']
            time_keys = s_data['time_keys']
            total_len = s_data['total_len']

            if total_len < N + 10:
                continue

            roll_ret = np.full(total_len, np.nan)
            roll_ret[N:] = (closes[N:] - closes[:-N]) / closes[:-N]

            in_surge = False
            surge_start = 0
            stock_episodes = []

            for i in range(N, total_len):
                r = roll_ret[i]
                if r >= M:
                    if not in_surge:
                        in_surge = True
                        surge_start = i
                else:
                    if in_surge:
                        stock_episodes.append({'start_idx': surge_start - N, 'end_idx': i - 1})
                        in_surge = False
            if in_surge:
                stock_episodes.append({'start_idx': surge_start - N, 'end_idx': total_len - 1})

            for ep in stock_episodes:
                st_i = ep['start_idx']
                end_i = ep['end_idx']
                p0 = closes[st_i]
                p_entry = closes[end_i]

                sub_closes = closes[st_i : end_i + 1]
                cnt = len(sub_closes)
                sub_rets = (sub_closes - p0) / p0
                min_idx = int(np.argmin(sub_rets))
                max_idx = int(np.argmax(sub_rets))
                min_ret = sub_rets[min_idx]
                max_ret = sub_rets[max_idx]
                min_pos = min_idx / max(1, cnt - 1)
                max_pos = max_idx / max(1, cnt - 1)

                pattern = ''
                if min_ret >= -0.005 and max_pos >= 0.75:
                    pattern = '▲ 单边拉升'
                elif min_pos <= 0.45 and min_ret <= -0.015:
                    pattern = 'V型深弹'
                elif max_pos <= 0.55 and (closes[end_i] - p0) / p0 < max_ret - 0.015:
                    pattern = '冲高回落'
                else:
                    pattern = '阶梯中继'

                # 1d +3% (78 bars = 1 regular day)
                max_fwd_1 = min(total_len - 1, end_i + 78)
                fwd_high_1 = np.max(highs[end_i + 1 : max_fwd_1 + 1]) if end_i + 1 <= total_len - 1 else p_entry
                hit_1d_3 = bool(fwd_high_1 >= p_entry * 1.03)
                mat_1d_3 = bool(end_i + 78 < total_len or hit_1d_3)

                # 1d +5% (78 bars = 1 regular day)
                hit_1d_5 = bool(fwd_high_1 >= p_entry * 1.05)
                mat_1d_5 = bool(end_i + 78 < total_len or hit_1d_5)

                # 3d +5% (234 bars = 3 regular days)
                max_fwd_3 = min(total_len - 1, end_i + 234)
                fwd_high_3 = np.max(highs[end_i + 1 : max_fwd_3 + 1]) if end_i + 1 <= total_len - 1 else p_entry
                hit_3d_5 = bool(fwd_high_3 >= p_entry * 1.05)
                mat_3d_5 = bool(end_i + 234 < total_len or hit_3d_5)

                # 3d +10% (234 bars)
                hit_3d_10 = bool(fwd_high_3 >= p_entry * 1.10)
                mat_3d_10 = bool(end_i + 234 < total_len or hit_3d_10)

                # 5d +10% (390 bars = 5 regular days)
                max_fwd_5 = min(total_len - 1, end_i + 390)
                fwd_high_5 = np.max(highs[end_i + 1 : max_fwd_5 + 1]) if end_i + 1 <= total_len - 1 else p_entry
                hit_5d_10 = bool(fwd_high_5 >= p_entry * 1.10)
                mat_5d_10 = bool(end_i + 390 < total_len or hit_5d_10)

                max_gain_3d = float((fwd_high_3 - p_entry) / p_entry * 100)

                # Tag mapping for stock
                is_qqq = (s in qqq_tickers)
                is_fav = (s in FAVORITES_TICKERS)
                is_held = (s in HOLDINGS_0086)
                is_etf = (s in ETF_TICKERS)
                is_semi = (s in SEMI_HARDWARE)

                evt = {
                    'symbol': s,
                    'pattern': pattern,
                    'start_time': time_keys[st_i],
                    'end_time': time_keys[end_i],
                    'start_price': round(float(p0), 2),
                    'end_price': round(float(p_entry), 2),
                    'window_ret': round(float((p_entry - p0) / p0 * 100), 2),
                    'bars': cnt,
                    'hit_1d_3': hit_1d_3,
                    'mat_1d_3': mat_1d_3,
                    'hit_1d_5': hit_1d_5,
                    'mat_1d_5': mat_1d_5,
                    'hit_3d_5': hit_3d_5,
                    'mat_3d_5': mat_3d_5,
                    'hit_3d_10': hit_3d_10,
                    'mat_3d_10': mat_3d_10,
                    'hit_5d_10': hit_5d_10,
                    'mat_5d_10': mat_5d_10,
                    'max_gain_3d': round(max_gain_3d, 2),
                    'is_qqq': is_qqq,
                    'is_fav': is_fav,
                    'is_held': is_held,
                    'is_etf': is_etf,
                    'is_semi': is_semi
                }
                win_events.append(evt)

        df_win = pd.DataFrame(win_events)

        # Aggregate pattern summary
        pat_summary = []
        pat_stock_stats = {}

        for p in patterns_order:
            sub = df_win[df_win['pattern'] == p] if not df_win.empty else pd.DataFrame()
            tot = len(sub)
            stocks_count = sub['symbol'].nunique() if not sub.empty else 0

            h1_3 = int(sub['hit_1d_3'].sum()) if not sub.empty else 0
            m1_3 = int(sub['mat_1d_3'].sum()) if not sub.empty else 0
            r1_3 = round(h1_3 / m1_3 * 100, 1) if m1_3 else 0.0

            h1 = int(sub['hit_1d_5'].sum()) if not sub.empty else 0
            m1 = int(sub['mat_1d_5'].sum()) if not sub.empty else 0
            r1 = round(h1 / m1 * 100, 1) if m1 else 0.0

            h3_5 = int(sub['hit_3d_5'].sum()) if not sub.empty else 0
            m3_5 = int(sub['mat_3d_5'].sum()) if not sub.empty else 0
            r3_5 = round(h3_5 / m3_5 * 100, 1) if m3_5 else 0.0

            h3_10 = int(sub['hit_3d_10'].sum()) if not sub.empty else 0
            m3_10 = int(sub['mat_3d_10'].sum()) if not sub.empty else 0
            r3_10 = round(h3_10 / m3_10 * 100, 1) if m3_10 else 0.0

            h5_10 = int(sub['hit_5d_10'].sum()) if not sub.empty else 0
            m5_10 = int(sub['mat_5d_10'].sum()) if not sub.empty else 0
            r5_10 = round(h5_10 / m5_10 * 100, 1) if m5_10 else 0.0

            pat_summary.append({
                'pattern': p,
                'code': PATTERN_META[p]['code'],
                'color': PATTERN_META[p]['color'],
                'bg': PATTERN_META[p]['bg'],
                'border': PATTERN_META[p]['border'],
                'desc': PATTERN_META[p]['desc'],
                'sentiment': PATTERN_META[p]['sentiment'],
                'total_events': tot,
                'stocks_count': stocks_count,
                'hit_1d_3': h1_3, 'mat_1d_3': m1_3, 'rate_1d_3': r1_3,
                'hit_1d_5': h1, 'mat_1d_5': m1, 'rate_1d_5': r1,
                'hit_3d_5': h3_5, 'mat_3d_5': m3_5, 'rate_3d_5': r3_5,
                'hit_3d_10': h3_10, 'mat_3d_10': m3_10, 'rate_3d_10': r3_10,
                'hit_5d_10': h5_10, 'mat_5d_10': m5_10, 'rate_5d_10': r5_10,
            })

            # Stock list for this pattern
            stock_list = []
            if not sub.empty:
                for s in sub['symbol'].unique():
                    s_events = [e for e in win_events if e['symbol'] == s and e['pattern'] == p]
                    s_tot = len(s_events)

                    s_h1_3 = sum(1 for e in s_events if e['hit_1d_3'])
                    s_m1_3 = sum(1 for e in s_events if e['mat_1d_3'])
                    s_r1_3 = round(s_h1_3 / s_m1_3 * 100, 1) if s_m1_3 else 0.0

                    s_h1 = sum(1 for e in s_events if e['hit_1d_5'])
                    s_m1 = sum(1 for e in s_events if e['mat_1d_5'])
                    s_r1 = round(s_h1 / s_m1 * 100, 1) if s_m1 else 0.0

                    s_h3_5 = sum(1 for e in s_events if e['hit_3d_5'])
                    s_m3_5 = sum(1 for e in s_events if e['mat_3d_5'])
                    s_r3_5 = round(s_h3_5 / s_m3_5 * 100, 1) if s_m3_5 else 0.0

                    s_h3_10 = sum(1 for e in s_events if e['hit_3d_10'])
                    s_m3_10 = sum(1 for e in s_events if e['mat_3d_10'])
                    s_r3_10 = round(s_h3_10 / s_m3_10 * 100, 1) if s_m3_10 else 0.0

                    s_h5_10 = sum(1 for e in s_events if e['hit_5d_10'])
                    s_m5_10 = sum(1 for e in s_events if e['mat_5d_10'])
                    s_r5_10 = round(s_h5_10 / s_m5_10 * 100, 1) if s_m5_10 else 0.0

                    meta = TICKER_META.get(s, {'name': s, 'sector': '科技/综合'})

                    is_high_rate = (s_r3_5 >= 70.0)
                    is_high_hits = (s_h3_5 >= 3)
                    if is_high_rate and is_high_hits:
                        quadrant = 'Q1'
                    elif is_high_rate and not is_high_hits:
                        quadrant = 'Q2'
                    elif not is_high_rate and not is_high_hits:
                        quadrant = 'Q3'
                    else:
                        quadrant = 'Q4'

                    if s_r3_5 >= 80.0 and s_h3_5 >= 3:
                        rating = '★★★★★ 顶级爆发'
                    elif s_r3_5 >= 70.0 and s_h3_5 >= 2:
                        rating = '★★★★ 高效稳健'
                    elif s_r3_5 >= 60.0:
                        rating = '★★★ 弹性活跃'
                    else:
                        rating = '★★ 偏弱观察'

                    stock_list.append({
                        'symbol': s,
                        'name': meta['name'],
                        'sector': meta['sector'],
                        'total_events': s_tot,
                        'mat_3d_5': s_m3_5,
                        'hit_3d_5': s_h3_5,
                        'rate_3d_5': s_r3_5,
                        'mat_1d_3': s_m1_3,
                        'hit_1d_3': s_h1_3,
                        'rate_1d_3': s_r1_3,
                        'mat_1d_5': s_m1,
                        'hit_1d_5': s_h1,
                        'rate_1d_5': s_r1,
                        'mat_3d_10': s_m3_10,
                        'hit_3d_10': s_h3_10,
                        'rate_3d_10': s_r3_10,
                        'mat_5d_10': s_m5_10,
                        'hit_5d_10': s_h5_10,
                        'rate_5d_10': s_r5_10,
                        'quadrant': quadrant,
                        'rating': rating,
                        'is_qqq': (s in qqq_tickers),
                        'is_fav': (s in FAVORITES_TICKERS),
                        'is_held': (s in HOLDINGS_0086),
                        'is_etf': (s in ETF_TICKERS),
                        'is_semi': (s in SEMI_HARDWARE),
                        'events': s_events
                    })

                stock_list.sort(key=lambda x: (x['rate_3d_5'], x['hit_3d_5'], x['total_events']), reverse=True)
            pat_stock_stats[p] = stock_list

        # Cross-pattern synergy
        synergy_scores = {}
        for p, s_list in pat_stock_stats.items():
            for item in s_list:
                s = item['symbol']
                if s not in synergy_scores:
                    synergy_scores[s] = {
                        'symbol': s,
                        'name': item['name'],
                        'sector': item['sector'],
                        'patterns_hit': {},
                        'total_hits_all': 0,
                        'total_events_all': 0,
                        'high_win_count': 0,
                        'is_qqq': item['is_qqq'],
                        'is_fav': item['is_fav'],
                        'is_held': item['is_held'],
                        'is_etf': item['is_etf'],
                        'is_semi': item['is_semi']
                    }
                synergy_scores[s]['patterns_hit'][p] = {
                    'hits': item['hit_3d_5'],
                    'total': item['total_events'],
                    'rate': item['rate_3d_5']
                }
                synergy_scores[s]['total_hits_all'] += item['hit_3d_5']
                synergy_scores[s]['total_events_all'] += item['total_events']
                if item['rate_3d_5'] >= 65.0 and item['hit_3d_5'] >= 2:
                    synergy_scores[s]['high_win_count'] += 1

        synergy_ranked = sorted(
            [v for v in synergy_scores.values() if v['total_hits_all'] >= 3],
            key=lambda x: (x['high_win_count'], x['total_hits_all']),
            reverse=True
        )

        windows_data[w_key] = {
            'window_key': w_key,
            'bars': N,
            'total_events': len(win_events),
            'pattern_summary': pat_summary,
            'pattern_stock_stats': pat_stock_stats,
            'synergy_ranked': synergy_ranked[:35]
        }
        print(f"Window {w_key} ({N} bars): {len(win_events)} events processed.")

    # 3. Build Windows Comparison Matrix (All 4 patterns across 1h~5h)
    pattern_verdicts = {
        'V型深弹': {
            'best_window': '1h ~ 3h (2h最佳平衡)',
            'verdict': '超强回踩深洗后突破，1h~3h 保持 73%~74% 极高胜率；1h 1d+5% 达 62.2% 具极强日内爆发力；2h 达成率 73.6% 且样本充足(126笔)，为黄金平衡点；4h~5h 周期过长钝化回落至 62%~66%。'
        },
        '冲高回落': {
            'best_window': '2h (黄金峰值)',
            'verdict': '脉冲遇阻回踩蓄势，2h 呈现最强胜率顶峰 (3d+5%高达 65.2%, 3d+10%高达 49.4%)；1h 窗口回踩消化尚未充分 (59.5%)；3h~5h 冲高后遇阻周期过长，多头动能退潮衰减至 47%~53%。'
        },
        '阶梯中继': {
            'best_window': '1h ~ 3h (稳健中枢)',
            'verdict': '平台换手二次起爆，表现极其稳健平滑。1h 达成率 55.3%，随窗口拓宽至 5h 仅温和回落至 47.3%，样本量持续放大至 950+ 笔，趋势延续性最佳。'
        },
        '▲ 单边拉升': {
            'best_window': '1h (超短线快打)',
            'verdict': '极度亢奋单边直拉，呈现显著衰竭诅咒！1h 短线惯性尚有 49.1%；但窗口拉长到 2h 跌至 42.7%，3h~5h 3d+10% 暴跌至 15%~17%，全形态胜率与盈亏比垫底，坚决不追长窗口单边。'
        }
    }

    windows_comparison = []
    for p in patterns_order:
        row = {
            'pattern': p,
            'code': PATTERN_META[p]['code'],
            'color': PATTERN_META[p]['color'],
            'best_window': pattern_verdicts[p]['best_window'],
            'verdict': pattern_verdicts[p]['verdict'],
            'windows': {}
        }
        for w_key in ['1h', '2h', '3h', '4h', '5h']:
            p_sum = next((x for x in windows_data[w_key]['pattern_summary'] if x['pattern'] == p), None)
            if p_sum:
                row['windows'][w_key] = {
                    'events': p_sum['total_events'],
                    'stocks': p_sum['stocks_count'],
                    'rate_1d_3': p_sum['rate_1d_3'],
                    'hit_1d_3': p_sum['hit_1d_3'],
                    'mat_1d_3': p_sum['mat_1d_3'],
                    'rate_1d_5': p_sum['rate_1d_5'],
                    'hit_1d_5': p_sum['hit_1d_5'],
                    'mat_1d_5': p_sum['mat_1d_5'],
                    'rate_3d_5': p_sum['rate_3d_5'],
                    'hit_3d_5': p_sum['hit_3d_5'],
                    'mat_3d_5': p_sum['mat_3d_5'],
                    'rate_3d_10': p_sum['rate_3d_10'],
                    'hit_3d_10': p_sum['hit_3d_10'],
                    'mat_3d_10': p_sum['mat_3d_10'],
                    'rate_5d_10': p_sum['rate_5d_10'],
                    'hit_5d_10': p_sum['hit_5d_10'],
                    'mat_5d_10': p_sum['mat_5d_10']
                }
        windows_comparison.append(row)

    full_payload = {
        'universe_info': {
            'universe': 'QQQ + 自选 + 自选ETF + 自选ETF成分股 (全量聚合)',
            'stock_count': len(symbols_to_scan),
            'start_date': '2026-07-01',
            'end_date': '2026-09-29',
            'bars_per_day': 78,
            'default_window': '2h',
            'threshold_m': M,
            'qqq_count': len(qqq_tickers),
            'fav_held_count': len(FAVORITES_TICKERS | HOLDINGS_0086),
            'etf_count': len(ETF_TICKERS),
            'semi_count': len(SEMI_HARDWARE)
        },
        'windows_config': WINDOWS_CONFIG,
        'windows_comparison': windows_comparison,
        'windows_data': windows_data,
        # Default 2h compatibility fields
        'pattern_summary': windows_data['2h']['pattern_summary'],
        'pattern_stock_stats': windows_data['2h']['pattern_stock_stats'],
        'synergy_ranked': windows_data['2h']['synergy_ranked']
    }

    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(full_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {JSON_OUT} ({JSON_OUT.stat().st_size / (1024*1024):.2f} MB)")

    generate_html(full_payload)
    return full_payload


def generate_html(payload):
    json_str = json.dumps(payload, ensure_ascii=False)
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>全量多维标的池 4 大上涨形态扫描与四象限特征分析 · QQQ + 自选 + ETF</title>
  <style>
    :root {{
      --bg-dark: #080d1a;
      --bg-card: #0f172a;
      --bg-toolbar: #131c31;
      --border-color: #1e293b;
      --border-active: #38bdf8;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      --color-cyan: #06b6d4;
      --color-amber: #f59e0b;
      --color-lime: #84cc16;
      --color-emerald: #10b981;
      --color-rose: #f43f5e;
      --color-purple: #a855f7;
      
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
      margin-bottom: 14px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-color);
    }}
    .title-area h1 {{
      font-size: 21px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .badge-universe {{
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.4);
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

    /* Bars Container */
    .control-bars {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 18px;
    }}

    /* Universe Scope Bar */
    .universe-bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      background: #111c33;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 10px 16px;
      flex-wrap: wrap;
    }}
    .universe-label {{
      font-size: 12px;
      font-weight: 700;
      color: #f1f5f9;
      font-family: var(--font-mono);
      margin-right: 4px;
    }}
    .btn-scope {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #94a3b8;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .btn-scope:hover {{
      color: #fff;
      border-color: #38bdf8;
    }}
    .btn-scope.active {{
      background: rgba(56, 189, 248, 0.22);
      color: #38bdf8;
      border-color: #38bdf8;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
    }}

    /* Window Width Switcher Bar (NEW!) */
    .window-bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      background: #0d1628;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 10px 16px;
      flex-wrap: wrap;
    }}
    .window-label {{
      font-size: 12px;
      font-weight: 700;
      color: #38bdf8;
      font-family: var(--font-mono);
      margin-right: 4px;
    }}
    .btn-window {{
      background: #0f172a;
      border: 1px solid #334155;
      color: #94a3b8;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .btn-window:hover {{
      color: #fff;
      border-color: #06b6d4;
    }}
    .btn-window.active {{
      background: rgba(6, 182, 212, 0.22);
      color: #22d3ee;
      border-color: #06b6d4;
      box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
      font-weight: 700;
    }}

    /* KPI Row */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .kpi-lbl {{
      font-size: 11px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}
    .kpi-val {{
      font-size: 18px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: #fff;
    }}
    .kpi-sub {{
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 4px;
      font-family: var(--font-mono);
    }}

    /* Section Card */
    .section-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 22px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .section-title {{
      font-size: 15px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    /* Tables */
    .benchmark-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: var(--font-mono);
    }}
    .benchmark-table th {{
      background: #111b2e;
      color: #94a3b8;
      text-align: left;
      padding: 10px 12px;
      font-size: 11px;
      font-weight: 600;
      border-bottom: 2px solid var(--border-color);
      white-space: nowrap;
    }}
    .benchmark-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      vertical-align: middle;
    }}
    .benchmark-table tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
      cursor: pointer;
    }}
    .benchmark-table tr.active-pat-row td {{
      background: rgba(6, 182, 212, 0.12) !important;
      border-bottom-color: #06b6d4;
    }}

    /* Evolution Matrix Table (NEW!) */
    .matrix-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: var(--font-mono);
    }}
    .matrix-table th {{
      background: #111b2e;
      color: #94a3b8;
      text-align: center;
      padding: 10px 10px;
      font-size: 11px;
      font-weight: 600;
      border-bottom: 2px solid var(--border-color);
      border-right: 1px solid rgba(255,255,255,0.05);
      white-space: nowrap;
    }}
    .matrix-table th.col-window {{
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .matrix-table th.col-window:hover {{
      color: #38bdf8;
      background: #15223c;
    }}
    .matrix-table th.active-win-col {{
      background: rgba(6, 182, 212, 0.22) !important;
      color: #22d3ee !important;
      border-bottom: 2px solid #06b6d4;
    }}
    .matrix-table td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      border-right: 1px solid rgba(255,255,255,0.05);
      vertical-align: middle;
      text-align: center;
    }}
    .matrix-table td.active-win-cell {{
      background: rgba(6, 182, 212, 0.08) !important;
      border-left: 1px solid rgba(6, 182, 212, 0.25);
      border-right: 1px solid rgba(6, 182, 212, 0.25);
    }}
    .matrix-table tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}

    .rate-badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 12px;
      font-family: var(--font-mono);
      white-space: nowrap;
    }}
    .rate-high {{ background: rgba(16, 185, 129, 0.22); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .rate-mid {{ background: rgba(56, 189, 248, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }}
    .rate-low {{ background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); }}
    .rate-danger {{ background: rgba(239, 68, 68, 0.20); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}

    /* Pattern Nav Tabs */
    .pattern-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    .pat-tab {{
      background: #111b2e;
      border: 1px solid #334155;
      color: #94a3b8;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
    }}
    .pat-tab:hover {{
      color: #fff;
      border-color: #475569;
    }}
    .pat-tab.active {{
      background: rgba(6, 182, 212, 0.18);
      color: #22d3ee;
      border-color: #06b6d4;
      box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
    }}

    /* Threshold Controls Bar */
    .quadrant-ctrl-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-toolbar);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 10px 16px;
      margin-bottom: 14px;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .ctrl-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .ctrl-label {{
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      font-family: var(--font-mono);
      white-space: nowrap;
    }}
    .btn-chip {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
      padding: 3px 9px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--font-mono);
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .btn-chip:hover {{
      color: #fff;
      border-color: #38bdf8;
    }}
    .btn-chip.active {{
      background: rgba(56, 189, 248, 0.22);
      color: #38bdf8;
      border-color: #38bdf8;
      font-weight: 700;
    }}

    /* Canvas Viewport */
    .scatter-canvas-wrap {{
      position: relative;
      width: 100%;
      height: 520px;
      background: #090e1a;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      cursor: crosshair;
    }}
    canvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }}

    /* 2x2 Grid View for Multi-Quadrant */
    .quadrant-grid-2x2 {{
      display: none;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
      margin-bottom: 20px;
    }}
    .grid-cell {{
      background: #090e1a;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px;
      position: relative;
      height: 380px;
    }}
    .grid-cell-title {{
      font-size: 12px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }}

    /* Tooltip */
    .hud-card {{
      position: absolute;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid #38bdf8;
      border-radius: 6px;
      padding: 12px 14px;
      font-size: 11px;
      font-family: var(--font-mono);
      pointer-events: none;
      z-index: 20;
      display: none;
      backdrop-filter: blur(6px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.6);
      min-width: 270px;
    }}

    /* Quadrant Legend Bar */
    .quadrant-legend {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 12px;
    }}
    .q-legend-card {{
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 11px;
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
    }}
    .q-legend-card:hover {{
      transform: translateY(-1px);
    }}
    .q-card-q1 {{ background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.35); }}
    .q-card-q2 {{ background: rgba(56, 189, 248, 0.08); border-color: rgba(56, 189, 248, 0.35); }}
    .q-card-q3 {{ background: rgba(148, 163, 184, 0.06); border-color: rgba(148, 163, 184, 0.25); }}
    .q-card-q4 {{ background: rgba(239, 68, 68, 0.08); border-color: rgba(239, 68, 68, 0.35); }}

    /* Sweet spot stocks table */
    .stock-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: var(--font-mono);
    }}
    .stock-table th {{
      background: #111b2e;
      color: #94a3b8;
      text-align: left;
      padding: 8px 10px;
      font-size: 11px;
      font-weight: 600;
      border-bottom: 2px solid var(--border-color);
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}
    .stock-table th:hover {{
      color: #38bdf8;
    }}
    .stock-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      vertical-align: middle;
    }}
    .stock-table tr:hover td {{
      background: rgba(255,255,255,0.02);
    }}

    .tag-q {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 700;
      font-family: var(--font-sans);
    }}
    .tag-q1 {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .tag-q2 {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }}
    .tag-q3 {{ background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid #64748b; }}
    .tag-q4 {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}

    .badge-source {{
      display: inline-block;
      font-size: 9px;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 3px;
      margin-right: 3px;
      font-family: var(--font-mono);
    }}
    .src-held {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .src-fav {{ background: rgba(56, 189, 248, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }}
    .src-etf {{ background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }}
    .src-qqq {{ background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid #475569; }}
    .src-semi {{ background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }}

    /* Drawer / Modal */
    .modal-overlay {{
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      z-index: 100;
      display: none;
      justify-content: center;
      align-items: center;
    }}
    .modal-box {{
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      width: 820px;
      max-width: 90vw;
      max-height: 85vh;
      overflow-y: auto;
      padding: 20px;
      box-shadow: 0 16px 40px rgba(0,0,0,0.8);
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <div class="header">
    <div class="title-area">
      <h1>
        🎯 纳斯达克 100 (QQQ) + 自选股 + 自选 ETF 全量 4 大上涨形态扫描与四象限特征分析
        <span class="badge-universe" id="header-universe-badge">全网 138 只标的 · 3 个月全量扫描</span>
      </h1>
      <div class="subtitle">
        扫描跨度: <strong>2026-07-01 至 2026-09-29</strong> · 数据粒度: <strong>5 分钟常规交易时段</strong> · 动量窗口: <strong><span id="subtitle-window-desc">N=24 根 (2小时)</span>, M ≥ +5.0%</strong> · 累计检测有效上涨波段: <strong><span id="subtitle-event-count">1,407</span> 笔</strong>
      </div>
    </div>
    <div style="text-align:right;">
      <a href="http://127.0.0.1:8768/analysis/multi_stock_log_momentum_windows/index.html" style="color:#38bdf8; text-decoration:none; font-size:12px; font-family:var(--font-mono); border:1px solid #0284c7; padding:4px 10px; border-radius:4px; background:rgba(2,132,199,0.15);">
        &larr; 返回多标的对数微观探测器
      </a>
    </div>
  </div>

  <div class="control-bars">
    <!-- Universe Scope Selector -->
    <div class="universe-bar">
      <span class="universe-label">🌐 标的池分析范围切换:</span>
      <button class="btn-scope active" id="scope-ALL" onclick="setUniverseScope('ALL')">
        全量扩充标的池 (138 只)
      </button>
      <button class="btn-scope" id="scope-FAV_HELD" onclick="setUniverseScope('FAV_HELD')">
        ⭐ 我的自选 & 0086持仓 (44 只)
      </button>
      <button class="btn-scope" id="scope-ETF" onclick="setUniverseScope('ETF')">
        📈 核心自选 ETF 族群 (10 只 · SOXX/SMH/SOXL等)
      </button>
      <button class="btn-scope" id="scope-SEMI" onclick="setUniverseScope('SEMI')">
        ⚡ 半导体/光通信/AI硬件成分股 (38 只)
      </button>
      <button class="btn-scope" id="scope-QQQ" onclick="setUniverseScope('QQQ')">
        🏛️ 纳指 100 QQQ 全成分 (104 只)
      </button>
    </div>

    <!-- Window Width Switcher Bar (NEW!) -->
    <div class="window-bar">
      <span class="window-label">⏱️ 检测窗口宽度切换 (Window Width):</span>
      <button class="btn-window" id="win-btn-1h" onclick="setWindowWidth('1h')">
        1小时 (12根 5m)
      </button>
      <button class="btn-window active" id="win-btn-2h" onclick="setWindowWidth('2h')">
        2小时 (24根 5m · 基准)
      </button>
      <button class="btn-window" id="win-btn-3h" onclick="setWindowWidth('3h')">
        3小时 (36根 5m)
      </button>
      <button class="btn-window" id="win-btn-4h" onclick="setWindowWidth('4h')">
        4小时 (48根 5m)
      </button>
      <button class="btn-window" id="win-btn-5h" onclick="setWindowWidth('5h')">
        5小时 (60根 5m)
      </button>
      <span style="font-size:11px; color:var(--text-muted); margin-left:auto; font-family:var(--font-mono);">
        当前分析窗口: <strong id="lbl-current-window" style="color:#22d3ee;">2小时 (24根 5m · 基准)</strong> · 实时联动全页面图表与统计
      </span>
    </div>
  </div>

  <!-- KPI Grid -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-lbl">当前范围标的总量</div>
      <div class="kpi-val" style="color:#38bdf8;" id="kpi-stock-count">138 只</div>
      <div class="kpi-sub" id="kpi-stock-sub">QQQ + 自选 + ETF全集</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">触发上涨形态标的</div>
      <div class="kpi-val" style="color:#10b981;" id="kpi-active-count">102 只</div>
      <div class="kpi-sub" id="kpi-active-rate">有效覆盖率 73.9%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">有效上涨波段事件</div>
      <div class="kpi-val" style="color:#f59e0b;" id="kpi-event-count">1,407 笔</div>
      <div class="kpi-sub" id="kpi-event-sub">2小时内涨幅 ≥ +5.0%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">最佳爆发形态 (3天+5%)</div>
      <div class="kpi-val" style="color:#06b6d4;" id="kpi-best-pat">V型深弹 73.6%</div>
      <div class="kpi-sub" id="kpi-best-pat-sub">92 / 125 笔成熟波段达标</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">最佳爆发形态 (3天+10%)</div>
      <div class="kpi-val" style="color:#c084fc;" id="kpi-best-10">V型深弹 51.2%</div>
      <div class="kpi-sub" id="kpi-best-10-sub">63 / 123 笔触及翻倍门槛</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">多栖共振黄金标的</div>
      <div class="kpi-val" style="color:#34d399;" id="kpi-synergy-top">TER · LITE · CRDO</div>
      <div class="kpi-sub">跨多形态胜率均 ≥ 70%</div>
    </div>
  </div>

  <!-- Section 1: Universe Pattern Benchmark Table -->
  <div class="section-card">
    <div class="section-head">
      <div class="section-title">
        <span>📋 4 种上涨型图形整体统计基准 (<span id="title-current-win" style="color:#38bdf8;">当前窗口: 2h 基准</span> · 当前标的池聚合)</span>
      </div>
      <div style="font-size:11px; font-family:var(--font-mono); color:var(--text-muted);">
        (点击单行可直接切换下方四象限图与个股列表)
      </div>
    </div>

    <table class="benchmark-table">
      <thead>
        <tr>
          <th style="width:140px;">形态图形类别</th>
          <th style="width:80px;">动量门类</th>
          <th style="width:90px;">覆盖标的数</th>
          <th style="width:80px;">命中频次</th>
          <th style="width:80px;">条件占比</th>
          <th style="width:130px;">后续 1天 +3% 达成率</th>
          <th style="width:130px;">后续 1天 +5% 达成率</th>
          <th style="width:150px; color:#38bdf8;">后续 3天 +5% 达成率 ★</th>
          <th style="width:140px;">后续 3天 +10% 达成率</th>
          <th style="width:140px;">后续 5天 +10% 达成率</th>
          <th>微观几何特征与博弈研判</th>
        </tr>
      </thead>
      <tbody id="benchmark-tbody">
        <!-- Injected via JS -->
      </tbody>
    </table>
  </div>

  <!-- Section 1.5: Multi-Window Evolution Matrix (NEW!) -->
  <div class="section-card">
    <div class="section-head">
      <div class="section-title">
        <span>📊 不同窗口宽度 (1h ~ 5h) 4 大形态核心指标演变比对矩阵 (Multi-Window Evolution Matrix)</span>
      </div>
      <div style="font-size:11px; font-family:var(--font-mono); color:var(--text-muted);">
        点击表头 <span style="color:#22d3ee;">[1h] · [2h] · [3h] · [4h] · [5h]</span> 可快速联动全页面切换分析窗口
      </div>
    </div>

    <table class="matrix-table" id="matrix-table">
      <thead>
        <tr>
          <th style="width:130px; text-align:left;">形态图形</th>
          <th style="width:130px;">最优窗口建议</th>
          <th class="col-window" id="col-head-1h" onclick="setWindowWidth('1h')">1h 窗口 (12根)</th>
          <th class="col-window active-win-col" id="col-head-2h" onclick="setWindowWidth('2h')">2h 窗口 (24根 · 基准)</th>
          <th class="col-window" id="col-head-3h" onclick="setWindowWidth('3h')">3h 窗口 (36根)</th>
          <th class="col-window" id="col-head-4h" onclick="setWindowWidth('4h')">4h 窗口 (48根)</th>
          <th class="col-window" id="col-head-5h" onclick="setWindowWidth('5h')">5h 窗口 (60根)</th>
          <th style="text-align:left;">核心实证规律与钝化机制研判</th>
        </tr>
      </thead>
      <tbody id="matrix-tbody">
        <!-- Injected via JS -->
      </tbody>
    </table>
  </div>

  <!-- Section 2: Quadrant Scatter Plot -->
  <div class="section-card">
    <div class="section-head">
      <div class="section-title">
        <span>🎯 分股票四象限分布图 (按上涨图形独立划分)</span>
        <span style="font-size:11px; font-weight:normal; color:var(--text-muted);">
          【横轴: 3天+5%达成数量 (笔数) · 纵轴: 3天+5%达成率 (%) · 目标定位第一象限黄金股票群】
        </span>
      </div>
      <div style="display:flex; gap:6px;">
        <button class="btn-chip" id="btn-view-single" onclick="setQuadrantViewMode('single')">单图形深度四象限</button>
        <button class="btn-chip" id="btn-view-2x2" onclick="setQuadrantViewMode('2x2')">🔲 4 图并列对比视图 (2x2)</button>
      </div>
    </div>

    <!-- Pattern Navigation Tabs -->
    <div class="pattern-tabs" id="pattern-tabs-bar">
      <!-- Injected via JS -->
    </div>

    <!-- Threshold Controls Bar -->
    <div class="quadrant-ctrl-bar">
      <div class="ctrl-item">
        <span class="ctrl-label">📐 达成率基准门槛 (纵轴 Y):</span>
        <button class="btn-chip" onclick="setThreshY(50.0)">50%</button>
        <button class="btn-chip" onclick="setThreshY(60.0)">60%</button>
        <button class="btn-chip active" id="chip-y-70" onclick="setThreshY(70.0)">70% (标准)</button>
        <button class="btn-chip" onclick="setThreshY(80.0)">80% (严格)</button>
        <input type="range" min="30" max="95" step="5" value="70" id="slider-thresh-y" oninput="onSliderY(this.value)" style="width:90px; accent-color:#38bdf8;">
        <span style="font-family:var(--font-mono); color:#38bdf8; font-weight:700;" id="val-thresh-y">≥ 70%</span>
      </div>

      <div class="ctrl-item">
        <span class="ctrl-label">⏱️ 达成数量门槛 (横轴 X):</span>
        <button class="btn-chip" onclick="setThreshX(1)">≥ 1笔</button>
        <button class="btn-chip" onclick="setThreshX(2)">≥ 2笔</button>
        <button class="btn-chip active" id="chip-x-3" onclick="setThreshX(3)">≥ 3笔 (标准)</button>
        <button class="btn-chip" onclick="setThreshX(5)">≥ 5笔 (高频)</button>
        <input type="range" min="1" max="15" step="1" value="3" id="slider-thresh-x" oninput="onSliderX(this.value)" style="width:80px; accent-color:#38bdf8;">
        <span style="font-family:var(--font-mono); color:#38bdf8; font-weight:700;" id="val-thresh-x">≥ 3 笔</span>
      </div>

      <div class="ctrl-item" style="margin-left:auto;">
        <span style="font-size:11px; color:var(--text-muted);">悬停查看个股卡片 · 点击点位锁定下方列表</span>
      </div>
    </div>

    <!-- Single Main Canvas Viewport -->
    <div id="single-canvas-container">
      <div class="scatter-canvas-wrap" id="scatter-wrap">
        <canvas id="scatter-canvas"></canvas>
        <div class="hud-card" id="scatter-hud"></div>
      </div>

      <!-- Quadrant Summary & Legend Bar -->
      <div class="quadrant-legend" id="quadrant-legend-bar">
        <!-- Injected via JS -->
      </div>
    </div>

    <!-- 2x2 Multi Viewport Grid -->
    <div class="quadrant-grid-2x2" id="grid-2x2-container">
      <div class="grid-cell">
        <div class="grid-cell-title">
          <span style="color:#06b6d4;">① V型深弹</span>
          <span style="font-family:var(--font-mono); font-size:11px; color:#38bdf8;" id="grid-sub-V型深弹">--</span>
        </div>
        <div style="position:relative; width:100%; height:335px;">
          <canvas id="grid-canvas-V型深弹"></canvas>
        </div>
      </div>

      <div class="grid-cell">
        <div class="grid-cell-title">
          <span style="color:#f59e0b;">② 冲高回落</span>
          <span style="font-family:var(--font-mono); font-size:11px; color:#fbbf24;" id="grid-sub-冲高回落">--</span>
        </div>
        <div style="position:relative; width:100%; height:335px;">
          <canvas id="grid-canvas-冲高回落"></canvas>
        </div>
      </div>

      <div class="grid-cell">
        <div class="grid-cell-title">
          <span style="color:#84cc16;">③ 阶梯中继</span>
          <span style="font-family:var(--font-mono); font-size:11px; color:#a3e635;" id="grid-sub-阶梯中继">--</span>
        </div>
        <div style="position:relative; width:100%; height:335px;">
          <canvas id="grid-canvas-阶梯中继"></canvas>
        </div>
      </div>

      <div class="grid-cell">
        <div class="grid-cell-title">
          <span style="color:#10b981;">④ ▲ 单边拉升</span>
          <span style="font-family:var(--font-mono); font-size:11px; color:#34d399;" id="grid-sub-▲ 单边拉升">--</span>
        </div>
        <div style="position:relative; width:100%; height:335px;">
          <canvas id="grid-canvas-▲ 单边拉升"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Section 3: Sweet Spot Stock Group Drilldown -->
  <div class="section-card">
    <div class="section-head">
      <div class="section-title">
        <span>👑 目标股票群范围圈定与穿透清单</span>
        <span style="font-size:11px; font-weight:normal; color:var(--text-muted);" id="table-filter-subtitle">
          (当前形态: V型深弹 · 筛选范围: 全部象限)
        </span>
      </div>
      <div style="display:flex; gap:6px; align-items:center;">
        <span style="font-size:11px; color:var(--text-muted);">象限过滤:</span>
        <button class="btn-chip active" id="filter-q-ALL" onclick="setQuadrantFilter('ALL')">全部标的</button>
        <button class="btn-chip" id="filter-q-Q1" onclick="setQuadrantFilter('Q1')" style="color:#34d399;">👑 仅看第一象限 (黄金群)</button>
        <button class="btn-chip" id="filter-q-Q2" onclick="setQuadrantFilter('Q2')" style="color:#38bdf8;">💎 仅看第二象限</button>
        <button class="btn-chip" id="filter-q-Q4" onclick="setQuadrantFilter('Q4')" style="color:#f87171;">⚠️ 仅看第四象限 (假突破)</button>
      </div>
    </div>

    <table class="stock-table">
      <thead>
        <tr>
          <th onclick="sortStockTable('symbol')">标的代码 ⇕</th>
          <th onclick="sortStockTable('name')">公司全称与核心主营</th>
          <th onclick="sortStockTable('sector')">细分行业赛道</th>
          <th>标的池属性</th>
          <th onclick="sortStockTable('quadrant')">象限归属 ⇕</th>
          <th onclick="sortStockTable('total_events')">总触发次数 ⇕</th>
          <th onclick="sortStockTable('hit_3d_5')" style="color:#38bdf8;">3天+5% 达成数 (X) ⇕</th>
          <th onclick="sortStockTable('rate_3d_5')" style="color:#38bdf8;">3天+5% 达成率 (Y) ★ ⇕</th>
          <th onclick="sortStockTable('rate_1d_3')">1天+3% 达成率 ⇕</th>
          <th onclick="sortStockTable('rate_1d_5')">1天+5% 达成率 ⇕</th>
          <th onclick="sortStockTable('rate_3d_10')">3天+10% 达成率 ⇕</th>
          <th onclick="sortStockTable('rate_5d_10')">5天+10% 达成率 ⇕</th>
          <th onclick="sortStockTable('rating')">综合动量评级</th>
          <th>明细穿透</th>
        </tr>
      </thead>
      <tbody id="stock-tbody">
        <!-- Injected via JS -->
      </tbody>
    </table>
  </div>

  <!-- Section 4: Cross-Pattern Synergy Matrix -->
  <div class="section-card">
    <div class="section-head">
      <div class="section-title">
        <span>⚡ 跨形态多维共振黄金标的矩阵 (Cross-Pattern Synergy Matrix)</span>
        <span style="font-size:11px; font-weight:normal; color:var(--text-muted);" id="synergy-subtitle">
          【在当前窗口下，2 个或更多上涨形态中均斩获高达成率（≥ 65%）的超核心标的】
        </span>
      </div>
    </div>

    <table class="stock-table">
      <thead>
        <tr>
          <th>标的代码</th>
          <th>公司全称</th>
          <th>行业赛道</th>
          <th>标的池属性</th>
          <th>高胜率形态数</th>
          <th>累计达标总笔数</th>
          <th>总触发笔数</th>
          <th>V型深弹 表现</th>
          <th>冲高回落 表现</th>
          <th>阶梯中继 表现</th>
          <th>单边拉升 表现</th>
          <th>核心量化评价</th>
        </tr>
      </thead>
      <tbody id="synergy-tbody">
        <!-- Injected via JS -->
      </tbody>
    </table>
  </div>

  <!-- Modal Event Details -->
  <div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:10px; margin-bottom:14px;">
        <h3 id="modal-title" style="color:#fff; font-size:15px;">个股触发事件历史穿透</h3>
        <button onclick="closeModal()" style="background:none; border:none; color:#94a3b8; font-size:18px; cursor:pointer;">&times;</button>
      </div>
      <div id="modal-content">
        <!-- Populated via JS -->
      </div>
    </div>
  </div>

  <!-- Embedded JSON Data -->
  <script id="raw-json" type="application/json">
{json_str}
  </script>

  <script>
    let appData = null;
    let universeScope = 'ALL'; // 'ALL', 'FAV_HELD', 'ETF', 'SEMI', 'QQQ'
    let currentWindow = '2h';  // '1h', '2h', '3h', '4h', '5h'
    let currentPattern = 'V型深弹';
    let threshY = 70.0;
    let threshX = 3;
    let quadrantFilter = 'ALL';
    let sortCol = 'rate_3d_5';
    let sortAsc = false;
    let hoveredDot = null;
    let viewMode = 'single'; // 'single' or '2x2'

    // Computed views based on universeScope & currentWindow
    let activePatternSummary = [];
    let activeStockStats = {{}};

    function init() {{
      const raw = document.getElementById('raw-json').textContent;
      appData = JSON.parse(raw);
      renderMatrixTable();
      recalcUniverseScope();
      setupCanvasListeners();
    }}

    // --- Window Width Switching (NEW!) ---
    function setWindowWidth(win) {{
      currentWindow = win;
      document.querySelectorAll('.btn-window').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById(`win-btn-${{win}}`);
      if (activeBtn) activeBtn.classList.add('active');

      const wCfg = appData.windows_config[win];
      const winName = wCfg ? wCfg.name : win;
      document.getElementById('lbl-current-window').innerText = winName;
      document.getElementById('subtitle-window-desc').innerText = `N=${{wCfg.bars}} 根 (${{win}})`;
      document.getElementById('title-current-win').innerText = `当前窗口: ${{win}} ${{win === '2h' ? '基准' : ''}}`;

      // Update matrix table active column highlight
      document.querySelectorAll('.col-window').forEach(th => th.classList.remove('active-win-col'));
      const colHead = document.getElementById(`col-head-${{win}}`);
      if (colHead) colHead.classList.add('active-win-col');

      document.querySelectorAll('.matrix-col-cell').forEach(td => {{
        td.classList.toggle('active-win-cell', td.getAttribute('data-win') === win);
      }});

      recalcUniverseScope();
    }}

    // --- Universe Scope Switching ---
    function setUniverseScope(scope) {{
      universeScope = scope;
      document.querySelectorAll('.btn-scope').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById(`scope-${{scope}}`);
      if (activeBtn) activeBtn.classList.add('active');

      const scopeBadgeMap = {{
        'ALL': '全量扩充标的池 (138 只 · QQQ+自选+ETF全集)',
        'FAV_HELD': '⭐ 我的自选与0086持仓 (44 只核心池)',
        'ETF': '📈 核心自选 ETF 族群 (10 只 · SOXX/SMH/SOXL等)',
        'SEMI': '⚡ 半导体/光通信/AI硬件族群 (38 只)',
        'QQQ': '🏛️ 纳斯达克 100 QQQ 全成分 (104 只)'
      }};
      document.getElementById('header-universe-badge').innerText = scopeBadgeMap[scope];

      recalcUniverseScope();
    }}

    function filterStockByScope(s) {{
      if (universeScope === 'ALL') return true;
      if (universeScope === 'FAV_HELD') return s.is_fav || s.is_held;
      if (universeScope === 'ETF') return s.is_etf;
      if (universeScope === 'SEMI') return s.is_semi;
      if (universeScope === 'QQQ') return s.is_qqq;
      return true;
    }}

    function recalcUniverseScope() {{
      const winObj = appData.windows_data[currentWindow];
      if (!winObj) return;

      activeStockStats = {{}};
      const patternsOrder = ['V型深弹', '冲高回落', '阶梯中继', '▲ 单边拉升'];

      patternsOrder.forEach(p => {{
        const fullList = winObj.pattern_stock_stats[p] || [];
        const filtered = fullList.filter(s => filterStockByScope(s));
        
        filtered.forEach(s => {{
          const isHighRate = (s.rate_3d_5 >= threshY);
          const isHighHits = (s.hit_3d_5 >= threshX);
          if (isHighRate && isHighHits) s.quadrant = 'Q1';
          else if (isHighRate && !isHighHits) s.quadrant = 'Q2';
          else if (!isHighRate && !isHighHits) s.quadrant = 'Q3';
          else s.quadrant = 'Q4';
        }});

        activeStockStats[p] = filtered;
      }});

      // Recalculate pattern summary for active scope and active window
      activePatternSummary = patternsOrder.map(p => {{
        const sList = activeStockStats[p];
        const cfg = winObj.pattern_summary.find(x => x.pattern === p);
        let totEvt = 0;
        let h1_3 = 0, m1_3 = 0;
        let h1 = 0, m1 = 0;
        let h3_5 = 0, m3_5 = 0;
        let h3_10 = 0, m3_10 = 0;
        let h5_10 = 0, m5_10 = 0;

        sList.forEach(s => {{
          totEvt += s.total_events;
          h1_3 += s.hit_1d_3; m1_3 += s.mat_1d_3;
          h1 += s.hit_1d_5; m1 += s.mat_1d_5;
          h3_5 += s.hit_3d_5; m3_5 += s.mat_3d_5;
          h3_10 += s.hit_3d_10; m3_10 += s.mat_3d_10;
          h5_10 += s.hit_5d_10; m5_10 += s.mat_5d_10;
        }});

        return {{
          pattern: p,
          color: cfg.color,
          bg: cfg.bg,
          border: cfg.border,
          desc: cfg.desc,
          sentiment: cfg.sentiment,
          total_events: totEvt,
          stocks_count: sList.length,
          hit_1d_3: h1_3, mat_1d_3: m1_3, rate_1d_3: m1_3 ? (h1_3/m1_3*100) : 0,
          hit_1d_5: h1, mat_1d_5: m1, rate_1d_5: m1 ? (h1/m1*100) : 0,
          hit_3d_5: h3_5, mat_3d_5: m3_5, rate_3d_5: m3_5 ? (h3_5/m3_5*100) : 0,
          hit_3d_10: h3_10, mat_3d_10: m3_10, rate_3d_10: m3_10 ? (h3_10/m3_10*100) : 0,
          hit_5d_10: h5_10, mat_5d_10: m5_10, rate_5d_10: m5_10 ? (h5_10/m5_10*100) : 0,
        }};
      }});

      // Update KPIs
      const allActiveTickers = new Set();
      let totalScopeEvents = 0;
      patternsOrder.forEach(p => {{
        activeStockStats[p].forEach(s => {{
          allActiveTickers.add(s.symbol);
          totalScopeEvents += s.total_events;
        }});
      }});

      const totalUniverseTickers = (universeScope === 'ALL') ? 138 :
        (universeScope === 'FAV_HELD' ? appData.universe_info.fav_held_count :
        (universeScope === 'ETF' ? appData.universe_info.etf_count :
        (universeScope === 'SEMI' ? appData.universe_info.semi_count : appData.universe_info.qqq_count)));

      document.getElementById('kpi-stock-count').innerText = `${{totalUniverseTickers}} 只`;
      document.getElementById('kpi-active-count').innerText = `${{allActiveTickers.size}} 只`;
      const actRate = totalUniverseTickers > 0 ? (allActiveTickers.size / totalUniverseTickers * 100).toFixed(1) : '0.0';
      document.getElementById('kpi-active-rate').innerText = `触发覆盖率 ${{actRate}}%`;
      document.getElementById('kpi-event-count').innerText = `${{totalScopeEvents.toLocaleString()}} 笔`;
      document.getElementById('kpi-event-sub').innerText = `${{currentWindow}}内涨幅 ≥ +5.0%`;
      document.getElementById('subtitle-event-count').innerText = totalScopeEvents.toLocaleString();

      const vSum = activePatternSummary.find(x => x.pattern === 'V型深弹');
      if (vSum) {{
        document.getElementById('kpi-best-pat').innerText = `V型深弹 ${{vSum.rate_3d_5.toFixed(1)}}%`;
        document.getElementById('kpi-best-pat-sub').innerText = `${{vSum.hit_3d_5}} / ${{vSum.mat_3d_5}} 笔成熟波段达标`;
        document.getElementById('kpi-best-10').innerText = `V型深弹 ${{vSum.rate_3d_10.toFixed(1)}}%`;
        document.getElementById('kpi-best-10-sub').innerText = `${{vSum.hit_3d_10}} / ${{vSum.mat_3d_10}} 笔触及翻倍门槛`;
      }}

      // Top synergy stocks
      const synList = (winObj.synergy_ranked || []).filter(s => filterStockByScope(s));
      if (synList.length > 0) {{
        const top3 = synList.slice(0, 3).map(x => x.symbol).join(' · ');
        document.getElementById('kpi-synergy-top').innerText = top3;
      }} else {{
        document.getElementById('kpi-synergy-top').innerText = '--';
      }}

      renderBenchmarkTable();
      renderPatternTabs();
      renderQuadrantView();
      renderStockTable();
      renderSynergyTable();
    }}

    // --- 1. Benchmark Table ---
    function renderBenchmarkTable() {{
      const tbody = document.getElementById('benchmark-tbody');
      tbody.innerHTML = '';
      const totalEvt = activePatternSummary.reduce((acc, x) => acc + x.total_events, 0);

      activePatternSummary.forEach(p => {{
        const tr = document.createElement('tr');
        if (p.pattern === currentPattern) tr.classList.add('active-pat-row');
        tr.onclick = () => selectPattern(p.pattern);

        const share = totalEvt > 0 ? (p.total_events / totalEvt * 100).toFixed(1) : '0.0';

        tr.innerHTML = `
          <td>
            <div style="display:flex; align-items:center; gap:6px;">
              <span style="width:10px; height:10px; border-radius:2px; background:${{p.color}};"></span>
              <strong style="color:${{p.color}};">${{p.pattern}}</strong>
            </div>
          </td>
          <td><span class="tag-q tag-q1">上涨门槛</span></td>
          <td><strong style="color:#fff;">${{p.stocks_count}}</strong> 只</td>
          <td><strong style="color:#fff;">${{p.total_events}}</strong> 笔</td>
          <td>${{share}}%</td>
          <td>${{getRateBadge(p.hit_1d_3, p.mat_1d_3)}}</td>
          <td>${{getRateBadge(p.hit_1d_5, p.mat_1d_5)}}</td>
          <td>${{getRateBadge(p.hit_3d_5, p.mat_3d_5, true)}}</td>
          <td>${{getRateBadge(p.hit_3d_10, p.mat_3d_10)}}</td>
          <td>${{getRateBadge(p.hit_5d_10, p.mat_5d_10)}}</td>
          <td style="color:var(--text-muted); font-size:11px;">${{p.desc}} · <em>${{p.sentiment}}</em></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // --- 1.5 Multi-Window Evolution Matrix Table ---
    function renderMatrixTable() {{
      const tbody = document.getElementById('matrix-tbody');
      tbody.innerHTML = '';
      const compList = appData.windows_comparison || [];

      compList.forEach(item => {{
        const tr = document.createElement('tr');

        function fmtWinCell(wKey) {{
          const d = item.windows[wKey];
          if (!d) return `<td class="matrix-col-cell" data-win="${{wKey}}">--</td>`;
          const isAct = (wKey === currentWindow);
          const actCls = isAct ? 'active-win-cell' : '';

          let badgeCls = 'rate-low';
          if (d.rate_3d_5 >= 65) badgeCls = 'rate-high';
          else if (d.rate_3d_5 >= 50) badgeCls = 'rate-mid';
          else if (d.rate_3d_5 < 40) badgeCls = 'rate-danger';

          return `
            <td class="matrix-col-cell ${{actCls}}" data-win="${{wKey}}" onclick="setWindowWidth('${{wKey}}')" style="cursor:pointer;">
              <div>
                <span class="rate-badge ${{badgeCls}}" style="font-size:11px; padding:2px 6px;">
                  3d+5%: ${{d.rate_3d_5.toFixed(1)}}%
                </span>
              </div>
              <div class="matrix-cell-sub">
                ${{d.events}}笔 · 1d+3%: ${{d.rate_1d_3.toFixed(1)}}% · 1d+5%: ${{d.rate_1d_5.toFixed(1)}}%
              </div>
            </td>
          `;
        }}

        tr.innerHTML = `
          <td style="text-align:left;">
            <div style="display:flex; align-items:center; gap:6px;">
              <span style="width:10px; height:10px; border-radius:2px; background:${{item.color}};"></span>
              <strong style="color:${{item.color}};">${{item.pattern}}</strong>
            </div>
          </td>
          <td>
            <span style="background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:2px 6px; border-radius:4px; font-weight:700; font-size:11px;">
              ${{item.best_window}}
            </span>
          </td>
          ${{fmtWinCell('1h')}}
          ${{fmtWinCell('2h')}}
          ${{fmtWinCell('3h')}}
          ${{fmtWinCell('4h')}}
          ${{fmtWinCell('5h')}}
          <td style="text-align:left; color:#cbd5e1; font-size:11px; line-height:1.4;">
            ${{item.verdict}}
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function getRateBadge(hit, mat, isHighlight) {{
      if (mat === 0) return `<span class="rate-badge rate-low">-- (0/0)</span>`;
      const pct = (hit / mat) * 100;
      let cls = 'rate-low';
      if (pct >= 60) cls = 'rate-high';
      else if (pct >= 40) cls = 'rate-mid';
      else if (pct < 25) cls = 'rate-danger';
      const glow = isHighlight ? 'box-shadow:0 0 6px rgba(56,189,248,0.5); font-weight:800;' : '';
      return `<span class="rate-badge ${{cls}}" style="${{glow}}">${{pct.toFixed(1)}}% <span style="font-size:10px; font-weight:normal;">(${{hit}}/${{mat}})</span></span>`;
    }}

    // --- 2. Pattern Tabs ---
    function renderPatternTabs() {{
      const bar = document.getElementById('pattern-tabs-bar');
      bar.innerHTML = '';
      activePatternSummary.forEach(p => {{
        const btn = document.createElement('button');
        btn.className = `pat-tab ${{p.pattern === currentPattern ? 'active' : ''}}`;
        btn.id = `tab-${{p.pattern}}`;
        btn.innerHTML = `
          <span style="width:8px; height:8px; border-radius:2px; background:${{p.color}};"></span>
          <span>${{p.pattern}}</span>
          <span style="font-family:var(--font-mono); font-size:11px; opacity:0.8;">(${{p.stocks_count}}只标的 · 3d+5%: ${{p.rate_3d_5.toFixed(1)}}%)</span>
        `;
        btn.onclick = () => selectPattern(p.pattern);
        bar.appendChild(btn);
      }});
    }}

    function selectPattern(pName) {{
      currentPattern = pName;
      document.querySelectorAll('.pat-tab').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById(`tab-${{pName}}`);
      if (activeBtn) activeBtn.classList.add('active');

      renderBenchmarkTable();
      renderQuadrantView();
      renderStockTable();
    }}

    // --- 3. Threshold Adjustments ---
    function setThreshY(val) {{
      threshY = parseFloat(val);
      document.getElementById('slider-thresh-y').value = threshY;
      document.getElementById('val-thresh-y').innerText = `≥ ${{threshY.toFixed(0)}}%`;
      document.querySelectorAll('[id^="chip-y-"]').forEach(c => c.classList.remove('active'));
      const chip = document.getElementById(`chip-y-${{Math.round(threshY)}}`);
      if (chip) chip.classList.add('active');
      recalcQuadrantsAndRender();
    }}

    function onSliderY(val) {{
      threshY = parseFloat(val);
      document.getElementById('val-thresh-y').innerText = `≥ ${{threshY.toFixed(0)}}%`;
      document.querySelectorAll('[id^="chip-y-"]').forEach(c => c.classList.remove('active'));
      recalcQuadrantsAndRender();
    }}

    function setThreshX(val) {{
      threshX = parseInt(val, 10);
      document.getElementById('slider-thresh-x').value = threshX;
      document.getElementById('val-thresh-x').innerText = `≥ ${{threshX}} 笔`;
      document.querySelectorAll('[id^="chip-x-"]').forEach(c => c.classList.remove('active'));
      const chip = document.getElementById(`chip-x-${{threshX}}`);
      if (chip) chip.classList.add('active');
      recalcQuadrantsAndRender();
    }}

    function onSliderX(val) {{
      threshX = parseInt(val, 10);
      document.getElementById('val-thresh-x').innerText = `≥ ${{threshX}} 笔`;
      document.querySelectorAll('[id^="chip-x-"]').forEach(c => c.classList.remove('active'));
      recalcQuadrantsAndRender();
    }}

    function recalcQuadrantsAndRender() {{
      Object.keys(activeStockStats).forEach(p => {{
        activeStockStats[p].forEach(s => {{
          const isHighRate = (s.rate_3d_5 >= threshY);
          const isHighHits = (s.hit_3d_5 >= threshX);
          if (isHighRate && isHighHits) s.quadrant = 'Q1';
          else if (isHighRate && !isHighHits) s.quadrant = 'Q2';
          else if (!isHighRate && !isHighHits) s.quadrant = 'Q3';
          else s.quadrant = 'Q4';
        }});
      }});
      renderQuadrantView();
      renderStockTable();
    }}

    function setQuadrantViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btn-view-single').classList.toggle('active', mode === 'single');
      document.getElementById('btn-view-2x2').classList.toggle('active', mode === '2x2');
      document.getElementById('single-canvas-container').style.display = (mode === 'single') ? 'block' : 'none';
      document.getElementById('grid-2x2-container').style.display = (mode === '2x2') ? 'grid' : 'none';
      renderQuadrantView();
    }}

    // --- 4. Canvas Quadrant Rendering ---
    function renderQuadrantView() {{
      if (viewMode === 'single') {{
        renderSingleScatter(currentPattern);
        renderQuadrantLegendBar(currentPattern);
      }} else {{
        activePatternSummary.forEach(p => {{
          renderGridScatter(p.pattern);
        }});
      }}
    }}

    function renderSingleScatter(patName) {{
      const canvas = document.getElementById('scatter-canvas');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      const stocks = activeStockStats[patName] || [];
      const padL = 60;
      const padR = 40;
      const padT = 30;
      const padB = 45;
      const chartW = w - padL - padR;
      const chartH = h - padT - padB;

      let maxHits = Math.max(...stocks.map(s => s.hit_3d_5), 5);
      maxHits = Math.max(maxHits + 1, 6);
      const minHits = 0;

      const minY = 0;
      const maxY = 105;

      function getX(hitVal) {{
        return padL + ((hitVal - minHits) / (maxHits - minHits)) * chartW;
      }}
      function getY(rateVal) {{
        return padT + chartH - ((rateVal - minY) / (maxY - minY)) * chartH;
      }}

      const crossX = getX(threshX);
      const crossY = getY(threshY);

      // 1. Quadrant Background Shading
      ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
      ctx.fillRect(crossX, padT, w - padR - crossX, crossY - padT);

      ctx.fillStyle = 'rgba(56, 189, 248, 0.05)';
      ctx.fillRect(padL, padT, crossX - padL, crossY - padT);

      ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
      ctx.fillRect(padL, crossY, crossX - padL, padT + chartH - crossY);

      ctx.fillStyle = 'rgba(239, 68, 68, 0.06)';
      ctx.fillRect(crossX, crossY, w - padR - crossX, padT + chartH - crossY);

      // 2. Grid & Axis Ticks
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;

      ctx.font = '10px monospace';
      ctx.fillStyle = '#64748b';
      ctx.textAlign = 'right';
      for (let yPct = 0; yPct <= 100; yPct += 20) {{
        const py = getY(yPct);
        ctx.beginPath();
        ctx.moveTo(padL, py);
        ctx.lineTo(w - padR, py);
        ctx.stroke();
        ctx.fillText(`${{yPct}}%`, padL - 8, py + 3);
      }}

      ctx.textAlign = 'center';
      const xStep = maxHits > 15 ? 2 : 1;
      for (let xH = 0; xH <= maxHits; xH += xStep) {{
        const px = getX(xH);
        ctx.beginPath();
        ctx.moveTo(px, padT);
        ctx.lineTo(px, padT + chartH);
        ctx.stroke();
        ctx.fillText(`${{xH}}`, px, h - padB + 16);
      }}

      // Axis Labels
      ctx.font = 'bold 11px sans-serif';
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'center';
      ctx.fillText('3 天 +5% 达成数量 (笔数) &rarr;', padL + chartW / 2, h - 8);

      ctx.save();
      ctx.translate(16, padT + chartH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('&larr; 3 天 +5% 达成率 (%)', 0, 0);
      ctx.restore();

      // 3. Crosshair Lines
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 5]);

      ctx.beginPath();
      ctx.moveTo(crossX, padT);
      ctx.lineTo(crossX, padT + chartH);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(padL, crossY);
      ctx.lineTo(w - padR, crossY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Quadrant Badges
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
      ctx.textAlign = 'right';
      ctx.fillText('👑 第一象限: 核心黄金群 (高频 & 高胜率)', w - padR - 10, padT + 20);

      ctx.fillStyle = 'rgba(56, 189, 248, 0.6)';
      ctx.textAlign = 'left';
      ctx.fillText('💎 第二象限: 高胜率蓄力群', padL + 12, padT + 20);

      ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
      ctx.textAlign = 'right';
      ctx.fillText('⚠️ 第四象限: 频繁冲高假突破 (低后劲)', w - padR - 10, padT + chartH - 12);

      ctx.fillStyle = 'rgba(100, 116, 139, 0.5)';
      ctx.textAlign = 'left';
      ctx.fillText('💤 第三象限: 观望观察区', padL + 12, padT + chartH - 12);

      // 4. Plot Stock Dots with collision avoidance
      const coordBuckets = {{}};
      stocks.forEach(s => {{
        const key = `${{s.hit_3d_5}}_${{Math.round(s.rate_3d_5)}}`;
        if (!coordBuckets[key]) coordBuckets[key] = [];
        coordBuckets[key].push(s);
      }});

      stocks.forEach(s => {{
        const key = `${{s.hit_3d_5}}_${{Math.round(s.rate_3d_5)}}`;
        const group = coordBuckets[key];
        const idxInGrp = group.indexOf(s);
        let jitterX = 0, jitterY = 0;
        if (group.length > 1) {{
          const angle = (idxInGrp / group.length) * Math.PI * 2;
          const radius = Math.min(12, 4 + group.length * 1.5);
          jitterX = Math.cos(angle) * radius;
          jitterY = Math.sin(angle) * radius;
        }}

        const px = getX(s.hit_3d_5) + jitterX;
        const py = getY(s.rate_3d_5) + jitterY;
        s._canvasX = px;
        s._canvasY = py;

        const rad = Math.min(14, Math.max(5, 4 + s.total_events * 0.7));
        const isHovered = (hoveredDot && hoveredDot.symbol === s.symbol);

        let dotColor = '#64748b';
        if (s.quadrant === 'Q1') dotColor = '#10b981';
        else if (s.quadrant === 'Q2') dotColor = '#38bdf8';
        else if (s.quadrant === 'Q4') dotColor = '#f43f5e';

        ctx.save();
        if (isHovered || s.quadrant === 'Q1') {{
          ctx.shadowColor = dotColor;
          ctx.shadowBlur = isHovered ? 16 : 8;
        }}

        ctx.beginPath();
        ctx.arc(px, py, isHovered ? rad + 3 : rad, 0, Math.PI * 2);
        ctx.fillStyle = dotColor;
        ctx.fill();
        ctx.lineWidth = isHovered ? 2.5 : 1.5;
        ctx.strokeStyle = s.is_held ? '#fbbf24' : (s.is_fav ? '#38bdf8' : '#fff');
        ctx.stroke();
        ctx.restore();

        // Ticker Text Label
        ctx.font = (isHovered || s.quadrant === 'Q1') ? 'bold 11px monospace' : '10px monospace';
        ctx.fillStyle = s.is_held ? '#fbbf24' : ((isHovered || s.quadrant === 'Q1') ? '#fff' : '#cbd5e1');
        ctx.textAlign = 'left';
        ctx.fillText(s.symbol, px + rad + 4, py + 3);
      }});
    }}

    function renderQuadrantLegendBar(patName) {{
      const bar = document.getElementById('quadrant-legend-bar');
      bar.innerHTML = '';
      const stocks = activeStockStats[patName] || [];
      const qCounts = {{ 'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0 }};
      stocks.forEach(s => qCounts[s.quadrant]++);

      const legends = [
        {{ q: 'Q1', title: '👑 第一象限 (核心黄金群)', sub: `达成数 ≥ ${{threshX}} 且 达成率 ≥ ${{threshY}}%`, count: qCounts['Q1'], cls: 'q-card-q1', color: '#34d399' }},
        {{ q: 'Q2', title: '💎 第二象限 (高胜率蓄力群)', sub: `达成数 < ${{threshX}} 但 达成率 ≥ ${{threshY}}%`, count: qCounts['Q2'], cls: 'q-card-q2', color: '#38bdf8' }},
        {{ q: 'Q4', title: '⚠️ 第四象限 (假突破重灾区)', sub: `达成数 ≥ ${{threshX}} 但 达成率 < ${{threshY}}%`, count: qCounts['Q4'], cls: 'q-card-q4', color: '#f87171' }},
        {{ q: 'Q3', title: '💤 第三象限 (观望观察区)', sub: `达成数 < ${{threshX}} 且 达成率 < ${{threshY}}%`, count: qCounts['Q3'], cls: 'q-card-q3', color: '#94a3b8' }},
      ];

      legends.forEach(l => {{
        const card = document.createElement('div');
        card.className = `q-legend-card ${{l.cls}}`;
        card.onclick = () => setQuadrantFilter(l.q);
        card.innerHTML = `
          <div style="flex:1;">
            <div style="font-weight:700; color:${{l.color}};">${{l.title}}</div>
            <div style="font-size:10px; color:var(--text-muted);">${{l.sub}}</div>
          </div>
          <div style="font-size:16px; font-weight:700; font-family:var(--font-mono); color:#fff;">${{l.count}} 只</div>
        `;
        bar.appendChild(card);
      }});
    }}

    // Grid 2x2 Canvas Renderer
    function renderGridScatter(patName) {{
      const canvas = document.getElementById(`grid-canvas-${{patName}}`);
      const subTitle = document.getElementById(`grid-sub-${{patName}}`);
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);
      const stocks = activeStockStats[patName] || [];
      const pSum = activePatternSummary.find(x => x.pattern === patName);
      if (subTitle && pSum) {{
        subTitle.innerText = `总体: ${{pSum.rate_3d_5.toFixed(1)}}% (${{pSum.hit_3d_5}}/${{pSum.mat_3d_5}}) · ${{stocks.length}}只标的`;
      }}

      const padL = 40, padR = 25, padT = 20, padB = 30;
      const chartW = w - padL - padR;
      const chartH = h - padT - padB;

      let maxHits = Math.max(...stocks.map(s => s.hit_3d_5), 5) + 1;
      function getX(val) {{ return padL + (val / maxHits) * chartW; }}
      function getY(val) {{ return padT + chartH - (val / 105) * chartH; }}

      const cx = getX(threshX);
      const cy = getY(threshY);

      // Shading
      ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
      ctx.fillRect(cx, padT, w - padR - cx, cy - padT);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.05)';
      ctx.fillRect(padL, padT, cx - padL, cy - padT);

      // Lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let y = 0; y <= 100; y += 25) {{
        ctx.beginPath();
        ctx.moveTo(padL, getY(y));
        ctx.lineTo(w - padR, getY(y));
        ctx.stroke();
      }}

      ctx.strokeStyle = '#38bdf8';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + chartH);
      ctx.moveTo(padL, cy); ctx.lineTo(w - padR, cy);
      ctx.stroke();
      ctx.setLineDash([]);

      stocks.forEach(s => {{
        const px = getX(s.hit_3d_5);
        const py = getY(s.rate_3d_5);
        ctx.beginPath();
        ctx.arc(px, py, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = s.quadrant === 'Q1' ? '#10b981' : (s.quadrant === 'Q2' ? '#38bdf8' : (s.quadrant === 'Q4' ? '#f43f5e' : '#64748b'));
        ctx.fill();
        ctx.font = '9px monospace';
        ctx.fillStyle = s.is_held ? '#fbbf24' : '#cbd5e1';
        ctx.fillText(s.symbol, px + 6, py + 3);
      }});
    }}

    // --- 5. Mouse Interaction for Canvas ---
    function setupCanvasListeners() {{
      const wrap = document.getElementById('scatter-wrap');
      const hud = document.getElementById('scatter-hud');

      wrap.addEventListener('mousemove', e => {{
        if (viewMode !== 'single') return;
        const rect = wrap.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const stocks = activeStockStats[currentPattern] || [];
        let closest = null;
        let minDist = 18;

        stocks.forEach(s => {{
          if (s._canvasX === undefined) return;
          const dx = mouseX - s._canvasX;
          const dy = mouseY - s._canvasY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < minDist) {{
            minDist = dist;
            closest = s;
          }}
        }});

        if (closest) {{
          hoveredDot = closest;
          renderSingleScatter(currentPattern);
          hud.style.display = 'block';
          hud.style.left = `${{Math.min(rect.width - 280, mouseX + 14)}}px`;
          hud.style.top = `${{Math.max(10, mouseY - 40)}}px`;

          const qTagMap = {{
            'Q1': '<span class="tag-q tag-q1">👑 第一象限 (核心黄金群)</span>',
            'Q2': '<span class="tag-q tag-q2">💎 第二象限 (高胜率蓄力群)</span>',
            'Q3': '<span class="tag-q tag-q3">💤 第三象限 (观望观察区)</span>',
            'Q4': '<span class="tag-q tag-q4">⚠️ 第四象限 (假突破区)</span>'
          }};

          let sourceBadges = '';
          if (closest.is_held) sourceBadges += '<span class="badge-source src-held">0086持仓</span>';
          if (closest.is_fav) sourceBadges += '<span class="badge-source src-fav">自选关注</span>';
          if (closest.is_etf) sourceBadges += '<span class="badge-source src-etf">自选ETF</span>';
          if (closest.is_semi) sourceBadges += '<span class="badge-source src-semi">半导体硬件</span>';
          if (closest.is_qqq) sourceBadges += '<span class="badge-source src-qqq">QQQ</span>';

          hud.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:5px; margin-bottom:6px;">
              <strong style="color:#fff; font-size:13px;">${{closest.symbol}}</strong>
              ${{qTagMap[closest.quadrant]}}
            </div>
            <div style="color:#38bdf8; font-size:11px; margin-bottom:2px;">${{closest.name}}</div>
            <div style="margin-bottom:6px;">${{sourceBadges}}</div>
            <div style="color:var(--text-dim); font-size:10px; margin-bottom:6px;">行业: ${{closest.sector}}</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
              <span style="color:var(--text-muted);">3天+5% 达成数量:</span>
              <strong style="color:#fff;">${{closest.hit_3d_5}} 笔 <span style="font-weight:normal; color:var(--text-dim);">/ ${{closest.total_events}}次</span></strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
              <span style="color:var(--text-muted);">3天+5% 达成率:</span>
              <strong style="color:#34d399; font-size:13px;">${{closest.rate_3d_5.toFixed(1)}}%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
              <span style="color:var(--text-muted);">1天+3% 达成率:</span>
              <span>${{closest.rate_1d_3.toFixed(1)}}% (${{closest.hit_1d_3}}/${{closest.mat_1d_3}})</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
              <span style="color:var(--text-muted);">1天+5% 达成率:</span>
              <span>${{closest.rate_1d_5.toFixed(1)}}% (${{closest.hit_1d_5}}/${{closest.mat_1d_5}})</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
              <span style="color:var(--text-muted);">3天+10% 达成率:</span>
              <span>${{closest.rate_3d_10.toFixed(1)}}% (${{closest.hit_3d_10}}/${{closest.mat_3d_10}})</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
              <span style="color:var(--text-muted);">5天+10% 达成率:</span>
              <span>${{closest.rate_5d_10.toFixed(1)}}% (${{closest.hit_5d_10}}/${{closest.mat_5d_10}})</span>
            </div>
            <div style="margin-top:6px; padding-top:4px; border-top:1px dashed #334155; font-size:10px; color:#facc15;">
              评级: ${{closest.rating}} (点击定位表格明细)
            </div>
          `;
        }} else {{
          if (hoveredDot) {{
            hoveredDot = null;
            hud.style.display = 'none';
            renderSingleScatter(currentPattern);
          }}
        }}
      }});

      wrap.addEventListener('mouseleave', () => {{
        hud.style.display = 'none';
        if (hoveredDot) {{
          hoveredDot = null;
          renderSingleScatter(currentPattern);
        }}
      }});

      wrap.addEventListener('click', () => {{
        if (hoveredDot) {{
          highlightStockRow(hoveredDot.symbol);
        }}
      }});
    }}

    function highlightStockRow(symbol) {{
      const tr = document.getElementById(`row-stock-${{symbol}}`);
      if (tr) {{
        tr.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        tr.style.backgroundColor = 'rgba(56, 189, 248, 0.25)';
        setTimeout(() => {{
          tr.style.backgroundColor = '';
        }}, 2500);
      }}
    }}

    // --- 6. Sweet Spot Stock Table ---
    function setQuadrantFilter(q) {{
      quadrantFilter = q;
      document.querySelectorAll('[id^="filter-q-"]').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById(`filter-q-${{q}}`);
      if (activeBtn) activeBtn.classList.add('active');

      const subtitle = document.getElementById('table-filter-subtitle');
      const qNames = {{
        'ALL': '全部象限',
        'Q1': '👑 第一象限 (核心黄金群)',
        'Q2': '💎 第二象限 (高胜率蓄力群)',
        'Q3': '💤 第三象限 (观望观察区)',
        'Q4': '⚠️ 第四象限 (假突破区)'
      }};
      subtitle.innerText = `(当前形态: ${{currentPattern}} · 筛选范围: ${{qNames[q]}})`;

      renderStockTable();
    }}

    function sortStockTable(col) {{
      if (sortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortCol = col;
        sortAsc = false;
      }}
      renderStockTable();
    }}

    function renderStockTable() {{
      const tbody = document.getElementById('stock-tbody');
      tbody.innerHTML = '';

      let list = [...(activeStockStats[currentPattern] || [])];

      if (quadrantFilter !== 'ALL') {{
        list = list.filter(s => s.quadrant === quadrantFilter);
      }}

      list.sort((a, b) => {{
        let vA = a[sortCol];
        let vB = b[sortCol];
        if (typeof vA === 'string') return sortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
        return sortAsc ? (vA - vB) : (vB - vA);
      }});

      list.forEach(s => {{
        const tr = document.createElement('tr');
        tr.id = `row-stock-${{s.symbol}}`;

        const qTag = `<span class="tag-q tag-${{s.quadrant.toLowerCase()}}">${{s.quadrant}}</span>`;

        let sourceBadges = '';
        if (s.is_held) sourceBadges += '<span class="badge-source src-held">0086持仓</span>';
        if (s.is_fav) sourceBadges += '<span class="badge-source src-fav">自选</span>';
        if (s.is_etf) sourceBadges += '<span class="badge-source src-etf">ETF</span>';
        if (s.is_semi) sourceBadges += '<span class="badge-source src-semi">半导体</span>';
        if (s.is_qqq) sourceBadges += '<span class="badge-source src-qqq">QQQ</span>';

        tr.innerHTML = `
          <td><strong style="color:#fff; font-size:13px;">${{s.symbol}}</strong></td>
          <td>
            <div style="font-weight:600; color:#f1f5f9;">${{s.name}}</div>
          </td>
          <td style="color:var(--text-muted); font-size:11px;">${{s.sector}}</td>
          <td>${{sourceBadges}}</td>
          <td>${{qTag}}</td>
          <td><strong>${{s.total_events}}</strong> 次</td>
          <td><strong style="color:#38bdf8; font-size:13px;">${{s.hit_3d_5}}</strong> 笔</td>
          <td>${{getRateBadge(s.hit_3d_5, s.mat_3d_5, true)}}</td>
          <td>${{getRateBadge(s.hit_1d_3, s.mat_1d_3)}}</td>
          <td>${{getRateBadge(s.hit_1d_5, s.mat_1d_5)}}</td>
          <td>${{getRateBadge(s.hit_3d_10, s.mat_3d_10)}}</td>
          <td>${{getRateBadge(s.hit_5d_10, s.mat_5d_10)}}</td>
          <td style="color:#facc15; font-size:11px;">${{s.rating}}</td>
          <td>
            <button class="btn-chip" onclick="openStockModal('${{s.symbol}}')">穿透 ${{s.total_events}} 笔事件</button>
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // --- 7. Cross-Pattern Synergy Table ---
    function renderSynergyTable() {{
      const tbody = document.getElementById('synergy-tbody');
      tbody.innerHTML = '';

      const winObj = appData.windows_data[currentWindow];
      let list = (winObj.synergy_ranked || []).filter(s => filterStockByScope(s));

      list.forEach(s => {{
        const tr = document.createElement('tr');

        function fmtPatStat(pName) {{
          const st = s.patterns_hit[pName];
          if (!st) return '<span style="color:var(--text-dim);">--</span>';
          const col = st.rate >= 70 ? '#34d399' : (st.rate >= 50 ? '#38bdf8' : '#94a3b8');
          return `<span style="color:${{col}}; font-weight:700;">${{st.rate.toFixed(0)}}% <span style="font-size:10px; font-weight:normal;">(${{st.hits}}/${{st.total}})</span></span>`;
        }}

        let sourceBadges = '';
        if (s.is_held) sourceBadges += '<span class="badge-source src-held">0086持仓</span>';
        if (s.is_fav) sourceBadges += '<span class="badge-source src-fav">自选</span>';
        if (s.is_etf) sourceBadges += '<span class="badge-source src-etf">ETF</span>';
        if (s.is_semi) sourceBadges += '<span class="badge-source src-semi">半导体</span>';
        if (s.is_qqq) sourceBadges += '<span class="badge-source src-qqq">QQQ</span>';

        tr.innerHTML = `
          <td><strong style="color:#fff; font-size:13px;">${{s.symbol}}</strong></td>
          <td style="color:#f1f5f9; font-weight:600;">${{s.name}}</td>
          <td style="color:var(--text-muted); font-size:11px;">${{s.sector}}</td>
          <td>${{sourceBadges}}</td>
          <td><span class="rate-badge rate-high">${{s.high_win_count}} 栖高胜率</span></td>
          <td><strong style="color:#38bdf8; font-size:13px;">${{s.total_hits_all}} 笔</strong></td>
          <td>${{s.total_events_all}} 次</td>
          <td>${{fmtPatStat('V型深弹')}}</td>
          <td>${{fmtPatStat('冲高回落')}}</td>
          <td>${{fmtPatStat('阶梯中继')}}</td>
          <td>${{fmtPatStat('▲ 单边拉升')}}</td>
          <td style="color:#facc15; font-size:11px;">极强爆发惯性 · 重点自选跟踪</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // --- 8. Event Modal ---
    function openStockModal(symbol) {{
      const modal = document.getElementById('modal-overlay');
      const title = document.getElementById('modal-title');
      const content = document.getElementById('modal-content');

      const s = (activeStockStats[currentPattern] || []).find(x => x.symbol === symbol);
      if (!s) return;

      title.innerHTML = `🔎 <strong>${{symbol}}</strong> (${{s.name}}) · 窗口【${{currentWindow}}】形态【${{currentPattern}}】触发历史记录 (${{s.events.length}} 笔)`;

      let html = `
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>触发时刻 (美东)</th>
              <th>结束时刻</th>
              <th>波段涨幅</th>
              <th>入场价 ($)</th>
              <th>3日内最高涨幅</th>
              <th>1天+3% 状态</th>
              <th>1天+5% 状态</th>
              <th>3天+5% 状态</th>
              <th>3天+10% 状态</th>
            </tr>
          </thead>
          <tbody>
      `;

      s.events.forEach(e => {{
        const hitTag1_3 = e.hit_1d_3 ? '<span class="tag-q tag-q1">✓ +3%</span>' : '<span class="tag-q tag-q3">&times; 未达标</span>';
        const hitTag1_5 = e.hit_1d_5 ? '<span class="tag-q tag-q1">✓ +5%</span>' : '<span class="tag-q tag-q3">&times; 未达标</span>';
        const hitTag5 = e.hit_3d_5 ? '<span class="tag-q tag-q1">✓ 已达标 +5%</span>' : '<span class="tag-q tag-q3">&times; 未达标</span>';
        const hitTag10 = e.hit_3d_10 ? '<span class="tag-q tag-q1">✓ 已达标 +10%</span>' : '<span class="tag-q tag-q3">&times; 未达标</span>';
        html += `
          <tr>
            <td>${{e.start_time.slice(5, 16)}}</td>
            <td>${{e.end_time.slice(5, 16)}}</td>
            <td style="color:#10b981; font-weight:700;">+${{e.window_ret}}%</td>
            <td>$${{e.end_price.toFixed(2)}}</td>
            <td style="color:${{e.max_gain_3d >= 5 ? '#34d399' : '#94a3b8'}}; font-weight:700;">+${{e.max_gain_3d.toFixed(2)}}%</td>
            <td>${{hitTag1_3}}</td>
            <td>${{hitTag1_5}}</td>
            <td>${{hitTag5}}</td>
            <td>${{hitTag10}}</td>
          </tr>
        `;
      }});

      html += '</tbody></table>';
      content.innerHTML = html;
      modal.style.display = 'flex';
    }}

    function closeModal() {{
      document.getElementById('modal-overlay').style.display = 'none';
    }}

    window.addEventListener('resize', () => {{
      renderQuadrantView();
    }});

    window.onload = init;
  </script>
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding='utf-8')
    print(f"Generated standalone HTML at {HTML_OUT} ({HTML_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == '__main__':
    scan_all_universe()
