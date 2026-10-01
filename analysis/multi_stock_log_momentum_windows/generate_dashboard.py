#!/usr/bin/env python3
"""Generate self-contained HTML wireframe for multi-stock 5m log-scale momentum window analysis.

Target tickers: ARM, AMD, LITE, MU
Time span: Past 3 months (2026-07-01 to 2026-09-29, 4,892 5m bars)
Coordinates: Vertical Logarithmic Scale
Enhanced Interactive Features:
1. Chart 1 (Main chart): Wheel zoom disabled; Horizontal time pan slider and drag pan added.
2. Chart 2 (Navigator overview): Draggable brush range selector with left/right handles to zoom Chart 1.
3. Pattern checkboxes to individually toggle visibility of each of the 8 micro-patterns.
4. Independent continuous shape construction per pattern (solves the occlusion bug where V-shape was masked by others).
5. Comprehensive Pattern Statistics Cockpit:
   - Frequency count and percentage distribution
   - Forward reach rates: 1-day +5%, 3-day +5% (New!), 3-day +10%, 5-day +10%
   - Interactive pattern filtering and isolation
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DIR_PATH = Path(__file__).resolve().parent
DATA_PATH = DIR_PATH / "data.json"
HTML_PATH = DIR_PATH / "index.html"


def prepare_data_if_missing():
    if DATA_PATH.exists():
        return
    symbols = ['ARM', 'AMD', 'LITE', 'MU']
    dfs = {}
    for s in symbols:
        p = ROOT / "market_data" / "us_5m" / s / "2026.parquet"
        df = pd.read_parquet(p)
        df = df[df['session_date'] >= '2026-07-01']
        reg = df[df['session_type'] == 'regular'].sort_values('time_key').reset_index(drop=True)
        dfs[s] = reg

    times = dfs['ARM']['time_key'].tolist()
    data = {
        'symbols': symbols,
        'time_keys': times,
        'stocks': {}
    }
    for s in symbols:
        reg = dfs[s]
        p0 = float(reg['close'].iloc[0])
        data['stocks'][s] = {
            'p0': round(p0, 2),
            'open': [round(float(x), 2) for x in reg['open']],
            'high': [round(float(x), 2) for x in reg['high']],
            'low': [round(float(x), 2) for x in reg['low']],
            'close': [round(float(x), 2) for x in reg['close']],
            'volume': [int(x) for x in reg['volume']],
        }
    with open(DATA_PATH, 'w') as f:
        json.dump(data, f)
    print(f"Generated {DATA_PATH} ({DATA_PATH.stat().st_size / 1024:.1f} KB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>多标的 5m 对数坐标走势与动态动量窗口探测器 · ARM / AMD / LITE / MU</title>
  <style>
    :root {
      --bg-dark: #0a0f1d;
      --bg-card: #111827;
      --bg-toolbar: #0f172a;
      --border-color: #1e293b;
      --border-focus: #38bdf8;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      /* Ticker Colors */
      --color-arm: #38bdf8;  /* Cyan */
      --color-amd: #fbbf24;  /* Amber */
      --color-lite: #10b981; /* Emerald */
      --color-mu: #c084fc;   /* Purple */

      /* Signal Colors */
      --surge-color: #10b981;
      --surge-bg: rgba(16, 185, 129, 0.16);
      --drop-color: #ef4444;
      --drop-bg: rgba(239, 68, 68, 0.16);
      
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-dark);
      color: var(--text-main);
      font-family: var(--font-sans);
      padding: 18px 24px;
      line-height: 1.45;
      font-size: 13px;
    }

    /* Top Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-color);
    }
    .title-area h1 {
      font-size: 20px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-log {
      background: rgba(56, 189, 248, 0.15);
      color: var(--color-arm);
      border: 1px solid rgba(56, 189, 248, 0.35);
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--font-mono);
      font-weight: 600;
    }
    .subtitle {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 5px;
    }

    /* KPI Summary Cards */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px 14px;
      position: relative;
      overflow: hidden;
    }
    .kpi-card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
    }
    .kpi-card.c-arm::before { background: var(--color-arm); }
    .kpi-card.c-amd::before { background: var(--color-amd); }
    .kpi-card.c-lite::before { background: var(--color-lite); }
    .kpi-card.c-mu::before { background: var(--color-mu); }

    .kpi-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 14px;
    }
    .kpi-p0 {
      font-size: 11px;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }
    .kpi-ret {
      font-size: 20px;
      font-weight: 800;
      font-family: var(--font-mono);
      margin: 4px 0 2px;
    }
    .kpi-stats {
      font-size: 11px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      border-top: 1px dashed rgba(255,255,255,0.08);
      padding-top: 4px;
      margin-top: 4px;
      font-family: var(--font-mono);
    }

    /* Control Panel */
    .cockpit-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .control-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 16px;
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .control-lbl {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }
    .control-sub {
      font-size: 11px;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }
    .val-badge {
      background: #1e293b;
      color: #38bdf8;
      border: 1px solid #334155;
      padding: 2px 7px;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 12px;
      min-width: 44px;
      text-align: center;
    }
    input[type="range"] {
      cursor: pointer;
      accent-color: #38bdf8;
      width: 110px;
    }
    .btn-chip {
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text-muted);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-family: var(--font-mono);
      transition: all 0.15s ease;
    }
    .btn-chip:hover {
      background: #27354f;
      color: #fff;
    }
    .btn-chip.active {
      background: #0284c7;
      color: #fff;
      border-color: #38bdf8;
      font-weight: 700;
    }

    /* Pattern Checkbox Chips */
    .pattern-check-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: #1e293b;
      border: 1px solid #334155;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--font-mono);
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .pattern-check-chip:hover {
      background: #27354f;
      border-color: #475569;
    }
    .pattern-check-chip input[type="checkbox"] {
      width: 12px;
      height: 12px;
      cursor: pointer;
      margin: 0;
      accent-color: #38bdf8;
    }

    /* Mode Toggles */
    .mode-btn-group {
      display: flex;
      background: #090e1a;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 2px;
      gap: 2px;
    }
    .mode-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 11px;
      padding: 4px 10px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.15s ease;
    }
    .mode-btn.active {
      background: #1e293b;
      color: #38bdf8;
      font-weight: 700;
      border: 1px solid #334155;
    }

    /* Pattern Statistics Cockpit */
    .stats-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .stats-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding-bottom: 8px;
    }
    .stats-title {
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .dist-bar-wrap {
      display: flex;
      height: 22px;
      border-radius: 4px;
      overflow: hidden;
      background: #090e1a;
      border: 1px solid #1e293b;
      margin-bottom: 14px;
    }
    .dist-bar-seg {
      height: 100%;
      position: relative;
      transition: all 0.15s ease;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-weight: 700;
      color: #fff;
      font-family: var(--font-mono);
      text-shadow: 0 1px 2px rgba(0,0,0,0.8);
      overflow: hidden;
      white-space: nowrap;
    }
    .dist-bar-seg:hover {
      filter: brightness(1.25);
      z-index: 2;
    }

    .matrix-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: var(--font-mono);
    }
    .matrix-table th {
      background: #0f172a;
      color: var(--text-muted);
      padding: 8px 10px;
      text-align: left;
      font-weight: 700;
      border-bottom: 1px solid var(--border-color);
    }
    .matrix-table td {
      padding: 8px 10px;
      border-bottom: 1px solid #162032;
    }
    .matrix-table tr:hover td {
      background: #172338;
      cursor: pointer;
    }
    .matrix-table tr.filtered-active td {
      background: rgba(56, 189, 248, 0.15);
      border-bottom-color: #38bdf8;
    }

    .rate-badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 3px;
      font-weight: 700;
      font-size: 11px;
      white-space: nowrap;
    }
    .rate-high {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .rate-mid {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .rate-low {
      background: rgba(148, 163, 184, 0.15);
      color: #94a3b8;
      border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Chart Card */
    .chart-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      position: relative;
    }
    .chart-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .chart-title {
      font-size: 13px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .legend-box {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      font-family: var(--font-mono);
      cursor: pointer;
      user-select: none;
    }
    .legend-item.disabled {
      opacity: 0.3;
      text-decoration: line-through;
    }
    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 2px;
    }

    /* Canvas Viewports */
    .canvas-container {
      position: relative;
      width: 100%;
      height: 480px;
      background: #090e1a;
      border: 1px solid #1e293b;
      border-radius: 6px;
      overflow: hidden;
      cursor: crosshair;
    }
    canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }

    /* Time Panning Bar (Under Chart 1) */
    .time-pan-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
      padding: 8px 14px;
      background: #090e1a;
      border: 1px solid var(--border-color);
      border-radius: 6px;
    }
    .btn-pan {
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 4px;
      cursor: pointer;
      font-family: var(--font-mono);
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .btn-pan:hover {
      background: #0284c7;
      color: #fff;
      border-color: #38bdf8;
    }
    .time-slider {
      flex: 1;
      accent-color: #38bdf8;
      cursor: ew-resize;
      height: 6px;
    }
    .viewport-info-badge {
      background: #1e293b;
      color: #38bdf8;
      border: 1px solid #334155;
      padding: 3px 8px;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
    }

    /* Sub-Chart & Navigator Brush Viewport */
    .sub-chart-container {
      position: relative;
      width: 100%;
      height: 110px;
      background: #090e1a;
      border: 1px solid #1e293b;
      border-radius: 6px;
      margin-top: 8px;
      overflow: hidden;
      cursor: pointer;
    }

    /* Tooltip */
    .hud-tooltip {
      position: absolute;
      top: 12px;
      right: 12px;
      background: rgba(15, 23, 42, 0.94);
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 11px;
      font-family: var(--font-mono);
      pointer-events: none;
      z-index: 10;
      display: none;
      backdrop-filter: blur(4px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      min-width: 250px;
    }

    /* Detected Episodes Table */
    .table-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
    }
    .table-head-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .table-title {
      font-weight: 700;
      font-size: 14px;
      color: #fff;
    }
    .filter-tabs {
      display: flex;
      gap: 6px;
    }

    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: var(--font-mono);
    }
    .data-table th {
      background: #0f172a;
      color: var(--text-muted);
      text-align: left;
      padding: 8px 12px;
      font-weight: 700;
      border-bottom: 1px solid var(--border-color);
    }
    .data-table td {
      padding: 8px 12px;
      border-bottom: 1px solid #162032;
    }
    .data-table tr:hover td {
      background: #172338;
    }
    .badge-tag {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 11px;
      font-weight: 700;
    }
    .badge-surge {
      background: rgba(16, 185, 129, 0.18);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-drop {
      background: rgba(239, 68, 68, 0.18);
      color: #ef4444;
      border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .btn-locate {
      background: #1e293b;
      border: 1px solid #334155;
      color: #38bdf8;
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 4px;
      cursor: pointer;
    }
    .btn-locate:hover {
      background: #0284c7;
      color: #fff;
    }

    /* Hit Shape Badge & Micro-pattern */
    .badge-shape {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      font-family: var(--font-mono);
      cursor: pointer;
      border: none;
      transition: all 0.15s ease;
    }
    .shape-pattern-tag {
      display: inline-block;
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 3px;
      margin-left: 4px;
      vertical-align: middle;
      font-family: var(--font-sans);
    }
  </style>
</head>
<body>

  <!-- Header -->
  <div class="header">
    <div class="title-area">
      <h1>
        📈 5m 对数坐标多标的走势与动态动量窗口探测器
        <span class="badge-log">纵轴对数坐标系 (Log Scale)</span>
      </h1>
      <div class="subtitle">
        标的池: <strong>ARM · AMD · LITE · MU</strong> · 跨度: 2026-07-01 至 2026-09-29 (3 个月 · 4,892 根 5 分钟常规交易时段 K 线) · 动态滑动窗口 (N, M%, D%) 触发区域实时计算
      </div>
    </div>
    <div style="text-align:right;">
      <div class="mode-btn-group">
        <button class="mode-btn active" id="btn-mode-norm" onclick="setScaleMode('norm')">归一化对数走势 (基准100)</button>
        <button class="mode-btn" id="btn-mode-abs" onclick="setScaleMode('abs')">绝对价格对数坐标 ($)</button>
        <button class="mode-btn" id="btn-mode-candle" onclick="setScaleMode('candle')">单标的 5m K线焦点</button>
      </div>
    </div>
  </div>

  <!-- KPI Cards -->
  <div class="kpi-row" id="kpi-container">
    <!-- Populated by JS -->
  </div>

  <!-- Interactive Control Cockpit -->
  <div class="cockpit-panel">
    <div class="control-row">
      <!-- N Control -->
      <div class="control-group">
        <span class="control-lbl">⏱️ 窗口长度 N:</span>
        <input type="range" id="input-n-range" min="2" max="234" value="24" oninput="updateN(this.value)">
        <span class="val-badge" id="val-n">24 根</span>
        <span class="control-sub" id="lbl-n-time">(2小时)</span>
        <div style="display:flex; gap:4px; margin-left:6px;">
          <button class="btn-chip" onclick="updateN(6)">30m(6)</button>
          <button class="btn-chip" onclick="updateN(12)">1h(12)</button>
          <button class="btn-chip active" id="chip-n-24" onclick="updateN(24)">2h(24)</button>
          <button class="btn-chip" onclick="updateN(78)">1天(78)</button>
          <button class="btn-chip" onclick="updateN(234)">3天(234)</button>
        </div>
      </div>

      <!-- M% Surge Control -->
      <div class="control-group">
        <span class="control-lbl" style="color:var(--surge-color);">🚀 上涨门槛 M%:</span>
        <input type="range" id="input-m-range" min="1.0" max="25.0" step="0.5" value="5.0" oninput="updateM(this.value)">
        <span class="val-badge" style="color:var(--surge-color);" id="val-m">+5.0%</span>
        <div style="display:flex; gap:4px; margin-left:4px;">
          <button class="btn-chip" onclick="updateM(3.0)">+3%</button>
          <button class="btn-chip active" id="chip-m-5" onclick="updateM(5.0)">+5%</button>
          <button class="btn-chip" onclick="updateM(8.0)">+8%</button>
          <button class="btn-chip" onclick="updateM(10.0)">+10%</button>
        </div>
      </div>

      <!-- D% Drop Control -->
      <div class="control-group">
        <span class="control-lbl" style="color:var(--drop-color);">📉 下跌门槛 D%:</span>
        <input type="range" id="input-d-range" min="1.0" max="25.0" step="0.5" value="5.0" oninput="updateD(this.value)">
        <span class="val-badge" style="color:var(--drop-color);" id="val-d">-5.0%</span>
        <div style="display:flex; gap:4px; margin-left:4px;">
          <button class="btn-chip" onclick="updateD(3.0)">-3%</button>
          <button class="btn-chip active" id="chip-d-5" onclick="updateD(5.0)">-5%</button>
          <button class="btn-chip" onclick="updateD(8.0)">-8%</button>
          <button class="btn-chip" onclick="updateD(10.0)">-10%</button>
        </div>
      </div>
    </div>

    <!-- Second Control Row: Ticker focus & Pattern Checkboxes (Requirement 1) -->
    <div class="control-row" style="border-top:1px solid rgba(255,255,255,0.05); padding-top:8px;">
      <div class="control-group">
        <span class="control-lbl">🎯 标记检测标的:</span>
        <button class="btn-chip active" id="chip-tag-ALL" onclick="setTagTarget('ALL')">全部 4 只交集/并集</button>
        <button class="btn-chip" id="chip-tag-ARM" onclick="setTagTarget('ARM')">仅 ARM</button>
        <button class="btn-chip" id="chip-tag-AMD" onclick="setTagTarget('AMD')">仅 AMD</button>
        <button class="btn-chip" id="chip-tag-LITE" onclick="setTagTarget('LITE')">仅 LITE</button>
        <button class="btn-chip" id="chip-tag-MU" onclick="setTagTarget('MU')">仅 MU</button>
      </div>

      <!-- Pattern Checkboxes to toggle rendering of each shape -->
      <div class="control-group" style="margin-left:6px; display:flex; align-items:center; gap:5px; flex-wrap:wrap;">
        <span class="control-lbl">🎨 渲染图形形态:</span>
        <label class="pattern-check-chip">
          <input type="checkbox" id="chk-pat-▲ 单边拉升" checked onchange="togglePatternVisibility('▲ 单边拉升', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#10b981;"></span>
          <span style="color:#10b981; font-weight:700;">▲ 单边拉升</span>
        </label>
        <label class="pattern-check-chip" style="border-color:#06b6d4;">
          <input type="checkbox" id="chk-pat-V型深弹" checked onchange="togglePatternVisibility('V型深弹', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#06b6d4;"></span>
          <span style="color:#06b6d4; font-weight:700;">V型深弹</span>
        </label>
        <label class="pattern-check-chip">
          <input type="checkbox" id="chk-pat-阶梯中继" checked onchange="togglePatternVisibility('阶梯中继', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#84cc16;"></span>
          <span style="color:#84cc16; font-weight:700;">阶梯中继</span>
        </label>
        <label class="pattern-check-chip">
          <input type="checkbox" id="chk-pat-冲高回落" checked onchange="togglePatternVisibility('冲高回落', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#f59e0b;"></span>
          <span style="color:#f59e0b; font-weight:700;">冲高回落</span>
        </label>
        <label class="pattern-check-chip">
          <input type="checkbox" id="chk-pat-▼ 单边下杀" checked onchange="togglePatternVisibility('▼ 单边下杀', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#f43f5e;"></span>
          <span style="color:#f43f5e; font-weight:700;">▼ 单边下杀</span>
        </label>
        <label class="pattern-check-chip">
          <input type="checkbox" id="chk-pat-倒V冲顶" checked onchange="togglePatternVisibility('倒V冲顶', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#a855f7;"></span>
          <span style="color:#a855f7; font-weight:700;">倒V冲顶</span>
        </label>
        <label class="pattern-check-chip">
          <input type="checkbox" id="chk-pat-探底回抽" checked onchange="togglePatternVisibility('探底回抽', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#3b82f6;"></span>
          <span style="color:#3b82f6; font-weight:700;">探底回抽</span>
        </label>
        <label class="pattern-check-chip">
          <input type="checkbox" id="chk-pat-破位阴跌" checked onchange="togglePatternVisibility('破位阴跌', this.checked)">
          <span style="width:8px; height:8px; border-radius:2px; background:#e11d48;"></span>
          <span style="color:#e11d48; font-weight:700;">破位阴跌</span>
        </label>

        <!-- Quick selection helper chips -->
        <button class="btn-chip" onclick="setAllPatternVisibility(true)" style="padding:2px 6px;">全选</button>
        <button class="btn-chip" onclick="setAllPatternVisibility(false)" style="padding:2px 6px;">全关</button>
        <button class="btn-chip" onclick="isolatePattern('V型深弹')" style="background:rgba(6,182,212,0.22); color:#22d3ee; border:1px solid #06b6d4; font-weight:700; padding:2px 8px;">★ 仅看 V型深弹</button>
      </div>

      <div class="control-group" id="candle-focus-group" style="display:none;">
        <span class="control-lbl">🕯️ 5m K线焦点标的:</span>
        <select id="candle-select" onchange="setCandleTarget(this.value)" style="background:#1e293b; color:#fff; border:1px solid #334155; padding:3px 8px; border-radius:4px; font-family:var(--font-mono);">
          <option value="ARM">ARM</option>
          <option value="AMD">AMD</option>
          <option value="LITE">LITE</option>
          <option value="MU">MU</option>
        </select>
      </div>

      <div class="control-group" style="margin-left:auto;">
        <span class="control-lbl">🔍 时间预设:</span>
        <button class="btn-chip active" id="chip-zoom-all" onclick="setTimeZoom('ALL')">全部 3 个月</button>
        <button class="btn-chip" id="chip-zoom-1m" onclick="setTimeZoom('1M')">近 1 个月</button>
        <button class="btn-chip" id="chip-zoom-2w" onclick="setTimeZoom('2W')">近 2 周</button>
        <button class="btn-chip" id="chip-zoom-3d" onclick="setTimeZoom('3D')">近 3 天</button>
        <button class="btn-chip" id="chip-zoom-1d" onclick="setTimeZoom('1D')">最新 1 天</button>
      </div>
    </div>
  </div>

  <!-- Pattern Statistics Cockpit (Requirement 2 & 3) -->
  <div class="stats-card">
    <div class="stats-header">
      <div class="stats-title">
        <span>📊 8 种微观图形特征统计与后续收益达成率矩阵</span>
        <span style="font-size:11px; font-weight:normal; color:var(--text-muted);">
          (当前条件命中区间分布 · 8色微观形态独立区隔 · 后续 1d/3d/5d 收益穿透追踪 · 点击单行可筛选下方列表)
        </span>
      </div>
      <div style="font-size:11px; font-family:var(--font-mono); color:var(--text-muted);" id="stats-summary-info">
        总命中事件: -- 笔
      </div>
    </div>

    <!-- Segmented Distribution Bar -->
    <div class="dist-bar-wrap" id="pattern-dist-bar">
      <!-- Populated via JS -->
    </div>

    <!-- Matrix Table (Includes New 3-day +5% Reach Metric) -->
    <table class="matrix-table" id="pattern-matrix-table">
      <thead>
        <tr>
          <th style="width:125px;">形态图形类别</th>
          <th style="width:70px;">动量门类</th>
          <th style="width:75px;">命中频次</th>
          <th style="width:80px;">条件占比</th>
          <th style="width:130px;">后续 1天 +5% 达成率</th>
          <th style="width:130px; color:#38bdf8;">后续 3天 +5% 达成率 ★</th>
          <th style="width:130px;">后续 3天 +10% 达成率</th>
          <th style="width:130px;">后续 5天 +10% 达成率</th>
          <th>微观几何特征与博弈研判</th>
        </tr>
      </thead>
      <tbody id="pattern-matrix-tbody">
        <!-- Populated via JS -->
      </tbody>
    </table>
  </div>

  <!-- Main Chart Card -->
  <div class="chart-card">
    <div class="chart-toolbar">
      <div class="chart-title">
        <span>① 主图：5m 对数坐标曲线走势</span>
        <span id="chart-mode-tag" style="font-size:11px; color:var(--text-muted); font-weight:normal;">[归一化基准 100 对数刻度 · 垂直等距代表等比例涨跌]</span>
      </div>
      <div class="legend-box" id="legend-box">
        <!-- Injected via JS -->
      </div>
    </div>

    <!-- Main Canvas Viewport (Chart 1) -->
    <div class="canvas-container" id="canvas-wrapper">
      <canvas id="main-canvas"></canvas>
      <div class="hud-tooltip" id="hud-tooltip"></div>
    </div>

    <!-- Horizontal Time Panning Toolbar (Requirement 1) -->
    <div class="time-pan-bar">
      <div style="display:flex; align-items:center; gap:6px;">
        <button class="btn-pan" onclick="panViewport(-150)" title="快速前移150根K线">◀◀ 快速前移</button>
        <button class="btn-pan" onclick="panViewport(-30)" title="微调前移30根K线">◀ 微调</button>
      </div>
      <div style="flex:1; display:flex; align-items:center; gap:10px;">
        <span style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono); white-space:nowrap;">时间轴平移:</span>
        <input type="range" class="time-slider" id="time-pan-slider" min="0" max="4891" value="0" oninput="onPanSliderInput(this.value)">
      </div>
      <div style="display:flex; align-items:center; gap:6px;">
        <span class="viewport-info-badge" id="viewport-info-lbl">视口: -- ~ --</span>
        <button class="btn-pan" onclick="panViewport(30)" title="微调后移30根K线">微调 ▶</button>
        <button class="btn-pan" onclick="panViewport(150)" title="快速后移150根K线">快速后移 ▶▶</button>
        <button class="btn-pan" onclick="resetViewport()" title="重置展示全部3个月">全景重置</button>
      </div>
    </div>

    <!-- Sub-Chart: Overview & Range Brush Selector (Requirement 1 & 2) -->
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:14px;">
      <div style="font-size:12px; font-weight:700; color:var(--text-main);">
        ② 副图：3个月全景概览 & 首尾区间选择器 (在下方全景图拖动左手柄 ◀| 和右手柄 |▶ 放大主图)
      </div>
      <div style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">
        全景时间轴 07-01 ~ 09-29 · 8色高亮块与主图同色对应 · 拖动中间平移，拖动两端手柄缩放
      </div>
    </div>
    <div class="sub-chart-container" id="sub-canvas-wrapper">
      <canvas id="sub-canvas"></canvas>
    </div>
  </div>

  <!-- Detected Episodes Table -->
  <div class="table-card">
    <div class="table-head-bar">
      <div class="table-title">
        📋 动态动量窗口触发事件清单 (共 <span id="event-count" style="color:#38bdf8;">0</span> 个有效事件)
        <span id="pattern-filter-status" style="font-size:11px; font-weight:normal; margin-left:10px; color:#38bdf8;"></span>
      </div>
      <div class="filter-tabs">
        <button class="btn-chip active" id="tab-evt-all" onclick="setEventFilter('ALL')">全部触发</button>
        <button class="btn-chip" id="tab-evt-surge" onclick="setEventFilter('SURGE')">仅上涨 &ge; +M%</button>
        <button class="btn-chip" id="tab-evt-drop" onclick="setEventFilter('DROP')">仅下跌 &le; -D%</button>
        <button class="btn-chip" id="btn-clear-pattern" onclick="setPatternFilter('ALL')" style="display:none; color:#facc15; border-color:#ca8a04;">清除形态筛选 ✕</button>
      </div>
    </div>
    <div style="max-height:360px; overflow-y:auto;">
      <table class="data-table" id="event-table">
        <thead>
          <tr>
            <th style="width:68px;">类型</th>
            <th style="width:60px;">标的</th>
            <th style="width:180px;">命中图形编号 / 微观形态</th>
            <th style="width:125px;">窗口起始时间</th>
            <th style="width:125px;">达标结束时间</th>
            <th style="width:85px;">窗口幅度</th>
            <th style="width:135px;">起始价 &rarr; 达标价</th>
            <th style="width:85px;">历时</th>
            <th style="width:75px;">操作</th>
          </tr>
        </thead>
        <tbody id="event-tbody">
          <!-- Injected via JS -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- Raw Embedded JSON Data -->
  <script id="raw-data" type="application/json">
__RAW_DATA_JSON__
  </script>

  <script>
    // --- Application State ---
    let appData = null;
    let N = 24;      // default 24 bars (2 hours)
    let M = 0.05;    // default +5%
    let D = 0.05;    // default -5%
    let tagTarget = 'ALL';
    let candleTarget = 'ARM';
    let scaleMode = 'norm'; // 'norm', 'abs', 'candle'
    let timeRange = [0, 4891];
    let eventFilter = 'ALL';
    let patternFilter = 'ALL'; // Filter by specific micro-pattern
    let activeStockVisibility = { 'ARM': true, 'AMD': true, 'LITE': true, 'MU': true };
    let mouseHoverIdx = null;
    let allShapes = [];
    let focusedShapeId = null;

    // Pattern visibility state controlled by checkboxes (Requirement 1)
    let patternVisibility = {
      '▲ 单边拉升': true,
      'V型深弹': true,
      '阶梯中继': true,
      '冲高回落': true,
      '▼ 单边下杀': true,
      '倒V冲顶': true,
      '探底回抽': true,
      '破位阴跌': true
    };

    // Pattern configurations with unique colors (Requirement 2)
    const PATTERNS_CONFIG = {
      '▲ 单边拉升': {
        type: 'SURGE',
        color: '#10b981', // Emerald
        bg: 'rgba(16, 185, 129, 0.18)',
        border: '#10b981',
        badgeBg: 'rgba(6, 78, 59, 0.92)',
        prefix: '拉升',
        desc: '无明显回撤直线上攻，多头单边加速'
      },
      'V型深弹': {
        type: 'SURGE',
        color: '#06b6d4', // Cyan
        bg: 'rgba(6, 182, 212, 0.22)',
        border: '#06b6d4',
        badgeBg: 'rgba(22, 78, 99, 0.92)',
        prefix: 'V弹',
        desc: '前半程深探下砸，尾盘暴力反扑创高'
      },
      '阶梯中继': {
        type: 'SURGE',
        color: '#84cc16', // Lime
        bg: 'rgba(132, 204, 22, 0.18)',
        border: '#84cc16',
        badgeBg: 'rgba(54, 83, 20, 0.92)',
        prefix: '阶梯',
        desc: '平台横盘蓄势后二次放量突破'
      },
      '冲高回落': {
        type: 'SURGE',
        color: '#f59e0b', // Amber
        bg: 'rgba(245, 158, 11, 0.20)',
        border: '#f59e0b',
        badgeBg: 'rgba(120, 53, 15, 0.92)',
        prefix: '冲高',
        desc: '脉冲急冲遇阻回踩，整体仍达标+M%'
      },
      '▼ 单边下杀': {
        type: 'DROP',
        color: '#f43f5e', // Rose
        bg: 'rgba(244, 63, 94, 0.18)',
        border: '#f43f5e',
        badgeBg: 'rgba(136, 19, 55, 0.92)',
        prefix: '下杀',
        desc: '空头完全控盘，单边阴跌破位无抵抗'
      },
      '倒V冲顶': {
        type: 'DROP',
        color: '#a855f7', // Purple
        bg: 'rgba(168, 85, 247, 0.20)',
        border: '#a855f7',
        badgeBg: 'rgba(88, 28, 135, 0.92)',
        prefix: '倒V',
        desc: '虚张声势诱多冲高后突遭瀑布砸盘'
      },
      '探底回抽': {
        type: 'DROP',
        color: '#3b82f6', // Blue
        bg: 'rgba(59, 130, 246, 0.20)',
        border: '#3b82f6',
        badgeBg: 'rgba(30, 58, 138, 0.92)',
        prefix: '探底',
        desc: '中途急砸击穿-D%后抄底盘托底反抽'
      },
      '破位阴跌': {
        type: 'DROP',
        color: '#e11d48', // Ruby Red
        bg: 'rgba(225, 29, 72, 0.18)',
        border: '#e11d48',
        badgeBg: 'rgba(76, 5, 25, 0.92)',
        prefix: '破位',
        desc: '高位震荡后关键位失守，多杀多直下'
      }
    };

    const COLORS = {
      'ARM': '#38bdf8',
      'AMD': '#fbbf24',
      'LITE': '#10b981',
      'MU': '#c084fc',
    };

    // Load Data
    function init() {
      const rawText = document.getElementById('raw-data').textContent.trim();
      if (rawText && !rawText.startsWith('__')) {
        appData = JSON.parse(rawText);
        setupDashboard();
      } else {
        fetch('data.json').then(r => r.json()).then(d => {
          appData = d;
          setupDashboard();
        });
      }
    }

    function setupDashboard() {
      timeRange = [0, appData.time_keys.length - 1];
      setupKPIs();
      setupLegend();
      setupChart1Listeners();
      setupChart2NavigatorListeners();
      recalcAndRender();
    }

    function setupKPIs() {
      const container = document.getElementById('kpi-container');
      container.innerHTML = '';
      appData.symbols.forEach(s => {
        const info = appData.stocks[s];
        const p0 = info.p0;
        const pLast = info.close[info.close.length - 1];
        const ret = ((pLast - p0) / p0) * 100;
        const pMin = Math.min(...info.low);
        const pMax = Math.max(...info.high);

        const card = document.createElement('div');
        card.className = `kpi-card c-${s.toLowerCase()}`;
        card.innerHTML = `
          <div class="kpi-head">
            <span style="color:${COLORS[s]}">${s}</span>
            <span class="kpi-p0">7/1基准: $${p0.toFixed(2)}</span>
          </div>
          <div class="kpi-ret" style="color:${ret >= 0 ? '#10b981' : '#ef4444'}">
            ${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%
          </div>
          <div class="kpi-stats">
            <span>最新: $${pLast.toFixed(2)}</span>
            <span>低: $${pMin.toFixed(2)}</span>
            <span>高: $${pMax.toFixed(2)}</span>
          </div>
          <div class="kpi-stats" id="kpi-events-${s}" style="border-top:none; padding-top:2px;">
            <span style="color:#10b981;">🚀 激增: --</span>
            <span style="color:#ef4444;">📉 急跌: --</span>
          </div>
        `;
        container.appendChild(card);
      });
    }

    function setupLegend() {
      const box = document.getElementById('legend-box');
      box.innerHTML = '';
      appData.symbols.forEach(s => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.id = `legend-item-${s}`;
        item.innerHTML = `
          <div class="legend-dot" style="background:${COLORS[s]}"></div>
          <span style="color:${COLORS[s]}; font-weight:700;">${s}</span>
        `;
        item.onclick = () => toggleStockVisibility(s);
        box.appendChild(item);
      });
    }

    function toggleStockVisibility(s) {
      activeStockVisibility[s] = !activeStockVisibility[s];
      document.getElementById(`legend-item-${s}`).classList.toggle('disabled', !activeStockVisibility[s]);
      render();
    }

    // --- Checkbox Handlers for Pattern Visibility (Requirement 1) ---
    function togglePatternVisibility(pName, isChecked) {
      patternVisibility[pName] = isChecked;
      const chk = document.getElementById(`chk-pat-${pName}`);
      if (chk) chk.checked = isChecked;
      renderTable();
      render();
    }

    function setAllPatternVisibility(val) {
      Object.keys(patternVisibility).forEach(p => {
        patternVisibility[p] = val;
        const chk = document.getElementById(`chk-pat-${p}`);
        if (chk) chk.checked = val;
      });
      renderTable();
      render();
    }

    function isolatePattern(targetPattern) {
      Object.keys(patternVisibility).forEach(p => {
        const val = (p === targetPattern);
        patternVisibility[p] = val;
        const chk = document.getElementById(`chk-pat-${p}`);
        if (chk) chk.checked = val;
      });
      renderTable();
      render();
    }

    // --- Parameter Updates ---
    function updateN(val) {
      N = parseInt(val, 10);
      document.getElementById('input-n-range').value = N;
      document.getElementById('val-n').innerText = `${N} 根`;
      const mins = N * 5;
      const hours = (mins / 60).toFixed(1);
      document.getElementById('lbl-n-time').innerText = `(${hours}小时)`;
      
      document.querySelectorAll('[id^="chip-n-"]').forEach(c => c.classList.remove('active'));
      const activeChip = document.getElementById(`chip-n-${N}`);
      if (activeChip) activeChip.classList.add('active');

      recalcAndRender();
    }

    function updateM(val) {
      M = parseFloat(val) / 100;
      document.getElementById('input-m-range').value = (M * 100).toFixed(1);
      document.getElementById('val-m').innerText = `+${(M * 100).toFixed(1)}%`;
      
      document.querySelectorAll('[id^="chip-m-"]').forEach(c => c.classList.remove('active'));
      const activeChip = document.getElementById(`chip-m-${Math.round(M * 100)}`);
      if (activeChip) activeChip.classList.add('active');

      recalcAndRender();
    }

    function updateD(val) {
      D = parseFloat(val) / 100;
      document.getElementById('input-d-range').value = (D * 100).toFixed(1);
      document.getElementById('val-d').innerText = `-${(D * 100).toFixed(1)}%`;
      
      document.querySelectorAll('[id^="chip-d-"]').forEach(c => c.classList.remove('active'));
      const activeChip = document.getElementById(`chip-d-${Math.round(D * 100)}`);
      if (activeChip) activeChip.classList.add('active');

      recalcAndRender();
    }

    function setTagTarget(t) {
      tagTarget = t;
      document.querySelectorAll('[id^="chip-tag-"]').forEach(c => c.classList.remove('active'));
      document.getElementById(`chip-tag-${t}`).classList.add('active');
      recalcAndRender();
    }

    function setScaleMode(mode) {
      scaleMode = mode;
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      if (mode === 'norm') document.getElementById('btn-mode-norm').classList.add('active');
      if (mode === 'abs') document.getElementById('btn-mode-abs').classList.add('active');
      if (mode === 'candle') document.getElementById('btn-mode-candle').classList.add('active');

      document.getElementById('candle-focus-group').style.display = (mode === 'candle') ? 'flex' : 'none';
      
      const tagMap = {
        'norm': '[归一化基准 100 对数刻度 · 垂直等距代表等比例涨跌]',
        'abs': '[绝对美元价格对数坐标 · $80 至 $1100 · 对数间距]',
        'candle': `[${candleTarget} 5m K线蜡烛图 + 其余标的对数曲线叠加]`
      };
      document.getElementById('chart-mode-tag').innerText = tagMap[mode];
      render();
    }

    function setCandleTarget(s) {
      candleTarget = s;
      if (scaleMode === 'candle') setScaleMode('candle');
    }

    function setTimeZoom(preset) {
      document.querySelectorAll('[id^="chip-zoom-"]').forEach(c => c.classList.remove('active'));
      const totalBars = appData.time_keys.length;
      if (preset === 'ALL') {
        timeRange = [0, totalBars - 1];
        document.getElementById('chip-zoom-all').classList.add('active');
      } else if (preset === '1M') {
        timeRange = [Math.max(0, totalBars - 78 * 22), totalBars - 1];
        document.getElementById('chip-zoom-1m').classList.add('active');
      } else if (preset === '2W') {
        timeRange = [Math.max(0, totalBars - 78 * 10), totalBars - 1];
        document.getElementById('chip-zoom-2w').classList.add('active');
      } else if (preset === '3D') {
        timeRange = [Math.max(0, totalBars - 78 * 3), totalBars - 1];
        document.getElementById('chip-zoom-3d').classList.add('active');
      } else if (preset === '1D') {
        timeRange = [Math.max(0, totalBars - 78), totalBars - 1];
        document.getElementById('chip-zoom-1d').classList.add('active');
      }
      render();
    }

    // --- Rolling Calculation & Event Detection ---
    let detectedEvents = [];
    let rollingReturns = {};
    let surgeMask = [];
    let dropMask = [];

    function recalcAndRender() {
      const len = appData.time_keys.length;
      surgeMask = new Array(len).fill(false);
      dropMask = new Array(len).fill(false);
      detectedEvents = [];
      rollingReturns = {};

      const symbolsToScan = (tagTarget === 'ALL') ? appData.symbols : [tagTarget];

      // Calculate rolling returns for all symbols
      appData.symbols.forEach(s => {
        const closes = appData.stocks[s].close;
        const rets = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          if (i < N) {
            rets[i] = 0;
          } else {
            const pPrev = closes[i - N];
            rets[i] = pPrev > 0 ? (closes[i] - pPrev) / pPrev : 0;
          }
        }
        rollingReturns[s] = rets;
      });

      // Populate surge & drop masks and episodes
      const sCounts = {};
      appData.symbols.forEach(s => sCounts[s] = { surge: 0, drop: 0 });

      symbolsToScan.forEach(s => {
        const rets = rollingReturns[s];
        const closes = appData.stocks[s].close;

        let inSurge = false;
        let surgeStart = 0;
        let inDrop = false;
        let dropStart = 0;

        for (let i = N; i < len; i++) {
          const r = rets[i];
          if (r >= M) {
            surgeMask[i] = true;
            sCounts[s].surge++;
            if (!inSurge) {
              inSurge = true;
              surgeStart = i;
            }
          } else {
            if (inSurge) {
              detectedEvents.push({
                type: 'SURGE',
                symbol: s,
                startIdx: surgeStart - N,
                endIdx: i - 1,
                startTime: appData.time_keys[surgeStart - N],
                endTime: appData.time_keys[i - 1],
                ret: (closes[i - 1] - closes[surgeStart - N]) / closes[surgeStart - N],
                startPrice: closes[surgeStart - N],
                endPrice: closes[i - 1],
                bars: (i - 1) - (surgeStart - N) + 1,
              });
              inSurge = false;
            }
          }

          if (r <= -D) {
            dropMask[i] = true;
            sCounts[s].drop++;
            if (!inDrop) {
              inDrop = true;
              dropStart = i;
            }
          } else {
            if (inDrop) {
              detectedEvents.push({
                type: 'DROP',
                symbol: s,
                startIdx: dropStart - N,
                endIdx: i - 1,
                startTime: appData.time_keys[dropStart - N],
                endTime: appData.time_keys[i - 1],
                ret: (closes[i - 1] - closes[dropStart - N]) / closes[dropStart - N],
                startPrice: closes[dropStart - N],
                endPrice: closes[i - 1],
                bars: (i - 1) - (dropStart - N) + 1,
              });
              inDrop = false;
            }
          }
        }
      });

      // 1. Classify micro-pattern for each event
      detectedEvents.forEach(e => {
        const closes = appData.stocks[e.symbol].close;
        const p0 = e.startPrice;
        let minRet = 0, maxRet = 0, minIdx = 0, maxIdx = 0;
        const count = Math.max(1, e.endIdx - e.startIdx + 1);
        for (let i = e.startIdx; i <= e.endIdx; i++) {
          const r = (closes[i] - p0) / p0;
          if (r < minRet) { minRet = r; minIdx = i - e.startIdx; }
          if (r > maxRet) { maxRet = r; maxIdx = i - e.startIdx; }
        }
        const minPos = minIdx / count;
        const maxPos = maxIdx / count;

        let pattern = '';
        if (e.type === 'SURGE') {
          if (minRet >= -0.005 && maxPos >= 0.75) pattern = '▲ 单边拉升';
          else if (minPos <= 0.45 && minRet <= -0.015) pattern = 'V型深弹';
          else if (maxPos <= 0.55 && (closes[e.endIdx] - p0) / p0 < maxRet - 0.015) pattern = '冲高回落';
          else pattern = '阶梯中继';
        } else {
          if (maxRet <= 0.005 && minPos >= 0.75) pattern = '▼ 单边下杀';
          else if (maxPos <= 0.45 && maxRet >= 0.015) pattern = '倒V冲顶';
          else if (minPos <= 0.55 && (closes[e.endIdx] - p0) / p0 > minRet + 0.015) pattern = '探底回抽';
          else pattern = '破位阴跌';
        }
        e.patternName = pattern;
      });

      // 2. Build Independent Continuous Shapes for EACH of the 8 Patterns (Solves Occlusion Bug!)
      allShapes = [];
      const patternKeys = Object.keys(PATTERNS_CONFIG);

      patternKeys.forEach(p => {
        const pEvents = detectedEvents.filter(e => e.patternName === p);
        const active = new Uint8Array(len);
        pEvents.forEach(e => {
          for (let i = e.startIdx; i <= e.endIdx; i++) active[i] = 1;
        });

        const shapes = [];
        let inS = false, sIdx = 0;
        for (let i = 0; i < len; i++) {
          if (active[i] === 1) {
            if (!inS) { inS = true; sIdx = i; }
          } else {
            if (inS) { shapes.push({ startIdx: sIdx, endIdx: i - 1 }); inS = false; }
          }
        }
        if (inS) shapes.push({ startIdx: sIdx, endIdx: len - 1 });

        shapes.forEach((s, idx) => {
          s.patternName = p;
          s.type = PATTERNS_CONFIG[p].type;
          s.typeId = `${PATTERNS_CONFIG[p].prefix}-${String(idx + 1).padStart(2, '0')}`;
          s.startTime = appData.time_keys[s.startIdx];
          s.endTime = appData.time_keys[s.endIdx];
          s.events = pEvents.filter(e => !(e.endIdx < s.startIdx || e.startIdx > s.endIdx));
          allShapes.push(s);
        });
      });

      // Sort all shapes chronologically
      allShapes.sort((a, b) => a.startIdx - b.startIdx);
      allShapes.forEach((s, idx) => {
        s.globalId = idx + 1;
        s.fullCode = `图形 #${String(idx + 1).padStart(2, '0')} [${s.typeId}]`;
      });

      // 3. Map events to their own pattern's shape
      detectedEvents.forEach(e => {
        const matched = allShapes.find(s => s.patternName === e.patternName && !(e.endIdx < s.startIdx || e.startIdx > s.endIdx));
        if (matched) {
          e.hitShapeId = matched.globalId;
          e.hitShapeCode = matched.fullCode;
          e.hitShapeTypeId = matched.typeId;
        } else {
          e.hitShapeId = 0;
          e.hitShapeCode = '未匹配';
          e.hitShapeTypeId = '--';
        }
      });

      // Update KPI card counts
      appData.symbols.forEach(s => {
        const el = document.getElementById(`kpi-events-${s}`);
        if (el) {
          el.innerHTML = `
            <span style="color:#10b981;">🚀 激增(${N}根): ${sCounts[s].surge}</span>
            <span style="color:#ef4444;">📉 急跌(${N}根): ${sCounts[s].drop}</span>
          `;
        }
      });

      // Sort events newest first
      detectedEvents.sort((a, b) => b.endIdx - a.endIdx);

      // Render Pattern Statistics Matrix & Cockpit (includes 3d+5%)
      renderPatternStatistics();

      renderTable();
      render();
    }

    // --- Pattern Statistics Cockpit & Forward Reach Rates (Requirement 2 & 3) ---
    function renderPatternStatistics() {
      const len = appData.time_keys.length;
      const stats = {};
      const patternKeys = Object.keys(PATTERNS_CONFIG);
      
      patternKeys.forEach(p => {
        stats[p] = {
          name: p,
          config: PATTERNS_CONFIG[p],
          count: 0,
          hit_1d: 0, mat_1d: 0,
          hit_3d_5: 0, mat_3d_5: 0,
          hit_3d_10: 0, mat_3d_10: 0,
          hit_5d_10: 0, mat_5d_10: 0
        };
      });

      const totalEvt = detectedEvents.length;
      document.getElementById('stats-summary-info').innerText = `当前命中有效事件: ${totalEvt} 笔 (N=${N}, M=+${(M*100).toFixed(1)}%, D=-${(D*100).toFixed(1)}%)`;

      // Track forward reach for each event
      detectedEvents.forEach(e => {
        const p = e.patternName;
        if (!stats[p]) return;
        const st = stats[p];
        st.count++;

        const end_i = e.endIdx;
        const s = e.symbol;
        const highs = appData.stocks[s].high;
        const p_entry = appData.stocks[s].close[end_i];

        function checkReach(hBars, targetPct) {
          const maxFwd = Math.min(len - 1, end_i + hBars);
          const isMature = (end_i + hBars < len);
          if (end_i + 1 > len - 1) return { hit: false, mature: isMature };
          let maxH = p_entry;
          for (let j = end_i + 1; j <= maxFwd; j++) {
            if (highs[j] > maxH) maxH = highs[j];
          }
          const hit = (maxH >= p_entry * (1.0 + targetPct));
          return { hit, mature: (isMature || hit) };
        }

        // 1d = 78 bars (+5%)
        const r1 = checkReach(78, 0.05);
        if (r1.mature) { st.mat_1d++; if (r1.hit) st.hit_1d++; }

        // 3d = 234 bars (+5%) (Requirement from User!)
        const r3_5 = checkReach(234, 0.05);
        if (r3_5.mature) { st.mat_3d_5++; if (r3_5.hit) st.hit_3d_5++; }

        // 3d = 234 bars (+10%)
        const r3_10 = checkReach(234, 0.10);
        if (r3_10.mature) { st.mat_3d_10++; if (r3_10.hit) st.hit_3d_10++; }

        // 5d = 390 bars (+10%)
        const r5 = checkReach(390, 0.10);
        if (r5.mature) { st.mat_5d++; if (r5.hit) st.hit_5d++; }
      });

      // 1. Render Segmented Distribution Bar
      const distBar = document.getElementById('pattern-dist-bar');
      distBar.innerHTML = '';
      patternKeys.forEach(p => {
        const st = stats[p];
        if (st.count === 0 && totalEvt > 0) return;
        const share = totalEvt > 0 ? (st.count / totalEvt) * 100 : 0;
        const seg = document.createElement('div');
        seg.className = 'dist-bar-seg';
        seg.style.width = `${Math.max(2, share)}%`;
        seg.style.backgroundColor = st.config.color;
        seg.title = `${p}: ${st.count}次 (${share.toFixed(1)}%) - 点击按此形态筛选`;
        seg.onclick = () => setPatternFilter(p);
        if (share >= 6) {
          seg.innerText = `${p} ${share.toFixed(0)}%`;
        }
        distBar.appendChild(seg);
      });

      // 2. Render Matrix Table
      const tbody = document.getElementById('pattern-matrix-tbody');
      tbody.innerHTML = '';

      function getRateBadge(hit, mat, isHighlight) {
        if (mat === 0) return `<span class="rate-badge rate-low">-- (0/0)</span>`;
        const pct = (hit / mat) * 100;
        let cls = 'rate-low';
        if (pct >= 40) cls = 'rate-high';
        else if (pct >= 20) cls = 'rate-mid';
        const hlBorder = isHighlight ? 'box-shadow:0 0 4px rgba(56,189,248,0.5);' : '';
        return `<span class="rate-badge ${cls}" style="${hlBorder}">${pct.toFixed(1)}% <span style="font-size:10px; font-weight:normal;">(${hit}/${mat})</span></span>`;
      }

      patternKeys.forEach(p => {
        const st = stats[p];
        const cfg = st.config;
        const share = totalEvt > 0 ? (st.count / totalEvt) * 100 : 0;
        const tr = document.createElement('tr');
        if (patternFilter === p) tr.classList.add('filtered-active');
        tr.onclick = () => setPatternFilter(patternFilter === p ? 'ALL' : p);

        const typeBadge = cfg.type === 'SURGE'
          ? `<span class="badge-tag badge-surge">上涨门槛</span>`
          : `<span class="badge-tag badge-drop">下跌门槛</span>`;

        tr.innerHTML = `
          <td>
            <div style="display:flex; align-items:center; gap:6px;">
              <span style="width:10px; height:10px; border-radius:2px; background:${cfg.color}; display:inline-block;"></span>
              <strong style="color:${cfg.color};">${p}</strong>
            </div>
          </td>
          <td>${typeBadge}</td>
          <td><strong style="color:#fff;">${st.count}</strong> 次</td>
          <td>
            <div style="display:flex; align-items:center; gap:6px;">
              <div style="flex:1; height:6px; background:#1e293b; border-radius:3px; overflow:hidden;">
                <div style="width:${share}%; height:100%; background:${cfg.color};"></div>
              </div>
              <span>${share.toFixed(1)}%</span>
            </div>
          </td>
          <td>${getRateBadge(st.hit_1d, st.mat_1d, false)}</td>
          <td>${getRateBadge(st.hit_3d_5, st.mat_3d_5, true)}</td>
          <td>${getRateBadge(st.hit_3d_10, st.mat_3d_10, false)}</td>
          <td>${getRateBadge(st.hit_5d, st.mat_5d, false)}</td>
          <td style="color:var(--text-muted); font-size:11px;">${cfg.desc}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function setPatternFilter(p) {
      patternFilter = p;
      const statusEl = document.getElementById('pattern-filter-status');
      const clearBtn = document.getElementById('btn-clear-pattern');
      if (p === 'ALL') {
        statusEl.innerText = '';
        clearBtn.style.display = 'none';
      } else {
        statusEl.innerText = `[当前形态锁定: ${p}]`;
        clearBtn.style.display = 'inline-block';
      }
      renderPatternStatistics();
      renderTable();
    }

    function setEventFilter(f) {
      eventFilter = f;
      document.getElementById('tab-evt-all').classList.toggle('active', f === 'ALL');
      document.getElementById('tab-evt-surge').classList.toggle('active', f === 'SURGE');
      document.getElementById('tab-evt-drop').classList.toggle('active', f === 'DROP');
      renderTable();
    }

    function renderTable() {
      const tbody = document.getElementById('event-tbody');
      tbody.innerHTML = '';
      
      const filtered = detectedEvents.filter(e => {
        if (eventFilter === 'SURGE' && e.type !== 'SURGE') return false;
        if (eventFilter === 'DROP' && e.type !== 'DROP') return false;
        if (patternFilter !== 'ALL' && e.patternName !== patternFilter) return false;
        // Check pattern checkbox visibility
        if (!patternVisibility[e.patternName]) return false;
        return true;
      });

      document.getElementById('event-count').innerText = filtered.length;

      const toShow = filtered.slice(0, 100); // top 100
      toShow.forEach(e => {
        const tr = document.createElement('tr');
        const badge = e.type === 'SURGE'
          ? `<span class="badge-tag badge-surge">🚀 激增 ≥ +${(M*100).toFixed(1)}%</span>`
          : `<span class="badge-tag badge-drop">📉 急跌 ≤ -${(D*100).toFixed(1)}%</span>`;
        
        const retCls = e.ret >= 0 ? '#10b981' : '#ef4444';
        const mins = e.bars * 5;
        const pCfg = PATTERNS_CONFIG[e.patternName] || { color: '#94a3b8', bg: 'rgba(148,163,184,0.15)', border: '#94a3b8' };

        tr.innerHTML = `
          <td>${badge}</td>
          <td style="color:${COLORS[e.symbol]}; font-weight:700;">${e.symbol}</td>
          <td>
            <button class="badge-shape" style="background:${pCfg.bg}; color:${pCfg.color}; border:1px solid ${pCfg.border};" onclick="locateShape(${e.hitShapeId})" title="点击主图定位该图形 #${e.hitShapeId} (${e.hitShapeTypeId})">
              ${e.hitShapeCode}
            </button>
            <span class="shape-pattern-tag" style="background:${pCfg.bg}; color:${pCfg.color}; border:1px solid ${pCfg.border};">${e.patternName}</span>
          </td>
          <td>${e.startTime.slice(5, 16)}</td>
          <td>${e.endTime.slice(5, 16)}</td>
          <td style="color:${retCls}; font-weight:700;">${e.ret >= 0 ? '+' : ''}${(e.ret * 100).toFixed(2)}%</td>
          <td>$${e.startPrice.toFixed(2)} &rarr; $${e.endPrice.toFixed(2)}</td>
          <td>${e.bars}根 (${mins}m)</td>
          <td>
            <button class="btn-locate" onclick="locateEvent(${e.startIdx}, ${e.endIdx})">定位视口</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    function locateShape(shapeId) {
      const s = allShapes.find(x => x.globalId === shapeId);
      if (!s) return;
      focusedShapeId = shapeId;
      const span = s.endIdx - s.startIdx;
      const pad = Math.max(30, Math.floor(span * 1.5));
      timeRange = [
        Math.max(0, s.startIdx - pad),
        Math.min(appData.time_keys.length - 1, s.endIdx + pad)
      ];
      render();
    }

    function locateEvent(startIdx, endIdx) {
      focusedShapeId = null;
      const pad = Math.max(30, Math.floor((endIdx - startIdx) * 1.5));
      const s = Math.max(0, startIdx - pad);
      const e = Math.min(appData.time_keys.length - 1, endIdx + pad);
      timeRange = [s, e];
      render();
    }

    // --- Viewport Horizontal Pan Controls (Requirement 1) ---
    function panViewport(deltaBars) {
      const len = appData.time_keys.length;
      const span = timeRange[1] - timeRange[0];
      let newS = Math.max(0, Math.min(len - 1 - span, timeRange[0] + deltaBars));
      let newE = newS + span;
      timeRange = [newS, newE];
      render();
    }

    function onPanSliderInput(val) {
      const len = appData.time_keys.length;
      const span = timeRange[1] - timeRange[0];
      const s = Math.max(0, Math.min(len - 1 - span, parseInt(val, 10)));
      timeRange = [s, s + span];
      render();
    }

    function resetViewport() {
      timeRange = [0, appData.time_keys.length - 1];
      focusedShapeId = null;
      render();
    }

    function updatePanSliderUI() {
      const slider = document.getElementById('time-pan-slider');
      const infoLbl = document.getElementById('viewport-info-lbl');
      if (!slider || !infoLbl || !appData) return;
      const len = appData.time_keys.length;
      slider.max = len - 1 - (timeRange[1] - timeRange[0]);
      slider.value = timeRange[0];

      const sTime = appData.time_keys[timeRange[0]].slice(5, 16);
      const eTime = appData.time_keys[timeRange[1]].slice(5, 16);
      const bars = timeRange[1] - timeRange[0] + 1;
      infoLbl.innerText = `视口: ${sTime} ~ ${eTime} (${bars}根 5m Bar)`;
    }

    // --- Chart 1 Canvas Listeners: Drag-to-Pan (Wheel Zoom Disabled) ---
    function setupChart1Listeners() {
      const container = document.getElementById('canvas-wrapper');
      let isDragging = false;
      let dragStartX = 0;
      let startRange = [0, 0];

      container.addEventListener('mousemove', e => {
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const width = rect.width;
        const [startIdx, endIdx] = timeRange;
        const count = endIdx - startIdx + 1;
        const padL = 65;
        const padR = 25;
        const chartW = width - padL - padR;

        if (isDragging) {
          const deltaX = e.clientX - dragStartX;
          const span = startRange[1] - startRange[0];
          const deltaBars = Math.round(-(deltaX / chartW) * span);
          const len = appData.time_keys.length;
          let newS = Math.max(0, Math.min(len - 1 - span, startRange[0] + deltaBars));
          timeRange = [newS, newS + span];
          render();
          return;
        }

        if (mouseX >= padL && mouseX <= width - padR) {
          const frac = (mouseX - padL) / chartW;
          mouseHoverIdx = Math.min(endIdx, Math.max(startIdx, Math.round(startIdx + frac * (count - 1))));
        } else {
          mouseHoverIdx = null;
        }
        render();
      });

      container.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        isDragging = true;
        dragStartX = e.clientX;
        startRange = [...timeRange];
        container.style.cursor = 'grabbing';
      });

      window.addEventListener('mouseup', () => {
        if (isDragging) {
          isDragging = false;
          container.style.cursor = 'crosshair';
        }
      });

      container.addEventListener('mouseleave', () => {
        if (!isDragging) {
          mouseHoverIdx = null;
          document.getElementById('hud-tooltip').style.display = 'none';
          render();
        }
      });
    }

    // --- Chart 2 (Sub-Chart) Navigator Brush Listeners (Requirement 1) ---
    let navDragMode = null;
    let navDragStartX = 0;
    let navDragStartRange = [0, 0];

    function setupChart2NavigatorListeners() {
      const container = document.getElementById('sub-canvas-wrapper');

      function getNavIdxFromEvent(e) {
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const padL = 65;
        const padR = 25;
        const chartW = rect.width - padL - padR;
        const frac = Math.max(0, Math.min(1, (x - padL) / chartW));
        return Math.round(frac * (appData.time_keys.length - 1));
      }

      function getNavXFromIdx(idx) {
        const rect = container.getBoundingClientRect();
        const padL = 65;
        const padR = 25;
        const chartW = rect.width - padL - padR;
        return padL + (idx / (appData.time_keys.length - 1)) * chartW;
      }

      container.addEventListener('mousedown', e => {
        if (e.button !== 0 || !appData) return;
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const xStart = getNavXFromIdx(timeRange[0]);
        const xEnd = getNavXFromIdx(timeRange[1]);
        const handleThreshold = 9;

        navDragStartX = e.clientX;
        navDragStartRange = [...timeRange];

        if (Math.abs(mouseX - xStart) <= handleThreshold) {
          navDragMode = 'left';
        } else if (Math.abs(mouseX - xEnd) <= handleThreshold) {
          navDragMode = 'right';
        } else if (mouseX > xStart && mouseX < xEnd) {
          navDragMode = 'pan';
        } else {
          // Clicked outside: jump window center
          const clickIdx = getNavIdxFromEvent(e);
          const span = timeRange[1] - timeRange[0];
          let newS = Math.max(0, Math.min(appData.time_keys.length - 1 - span, Math.round(clickIdx - span / 2)));
          timeRange = [newS, newS + span];
          render();
          navDragMode = 'pan';
          navDragStartRange = [...timeRange];
        }
      });

      window.addEventListener('mousemove', e => {
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const xStart = getNavXFromIdx(timeRange[0]);
        const xEnd = getNavXFromIdx(timeRange[1]);
        const handleThreshold = 9;

        if (!navDragMode) {
          if (Math.abs(mouseX - xStart) <= handleThreshold || Math.abs(mouseX - xEnd) <= handleThreshold) {
            container.style.cursor = 'ew-resize';
          } else if (mouseX > xStart && mouseX < xEnd) {
            container.style.cursor = 'grab';
          } else {
            container.style.cursor = 'pointer';
          }
          return;
        }

        const len = appData.time_keys.length;
        const padL = 65;
        const padR = 25;
        const chartW = rect.width - padL - padR;
        const deltaX = e.clientX - navDragStartX;
        const deltaBars = Math.round((deltaX / chartW) * (len - 1));

        if (navDragMode === 'left') {
          container.style.cursor = 'ew-resize';
          const newStart = Math.max(0, Math.min(navDragStartRange[1] - 20, navDragStartRange[0] + deltaBars));
          timeRange = [newStart, navDragStartRange[1]];
          render();
        } else if (navDragMode === 'right') {
          container.style.cursor = 'ew-resize';
          const newEnd = Math.min(len - 1, Math.max(navDragStartRange[0] + 20, navDragStartRange[1] + deltaBars));
          timeRange = [navDragStartRange[0], newEnd];
          render();
        } else if (navDragMode === 'pan') {
          container.style.cursor = 'grabbing';
          const span = navDragStartRange[1] - navDragStartRange[0];
          let newS = Math.max(0, Math.min(len - 1 - span, navDragStartRange[0] + deltaBars));
          timeRange = [newS, newS + span];
          render();
        }
      });

      window.addEventListener('mouseup', () => {
        if (navDragMode) {
          navDragMode = null;
          container.style.cursor = 'pointer';
        }
      });
    }

    // --- Canvas Rendering Engine ---
    function render() {
      updatePanSliderUI();
      renderMainChart();
      renderSubChart();
    }

    // --- Chart 1: Main Log-Scale Chart ---
    function renderMainChart() {
      const canvas = document.getElementById('main-canvas');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      const padL = 65;
      const padR = 25;
      const padT = 20;
      const padB = 30;
      const chartW = w - padL - padR;
      const chartH = h - padT - padB;

      const [startIdx, endIdx] = timeRange;
      const count = endIdx - startIdx + 1;

      // 1. Calculate Min and Max in visible window based on scaleMode
      let minVal = Infinity;
      let maxVal = -Infinity;

      appData.symbols.forEach(s => {
        if (!activeStockVisibility[s]) return;
        const info = appData.stocks[s];
        const p0 = info.p0;
        for (let i = startIdx; i <= endIdx; i++) {
          let val;
          if (scaleMode === 'norm') {
            val = Math.log(info.close[i] / p0);
          } else {
            val = Math.log(info.close[i]);
          }
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
      });

      if (!isFinite(minVal) || !isFinite(maxVal)) {
        minVal = Math.log(100);
        maxVal = Math.log(110);
      }

      // Add 6% padding to log bounds
      const span = maxVal - minVal;
      minVal -= span * 0.06;
      maxVal += span * 0.06;

      function getX(idx) {
        return padL + ((idx - startIdx) / Math.max(1, count - 1)) * chartW;
      }
      function getY(logVal) {
        return padT + chartH - ((logVal - minVal) / (maxVal - minVal)) * chartH;
      }

      // 2. Draw Background Grid and Y-Axis Ticks (Log Scale)
      ctx.strokeStyle = '#162032';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#64748b';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';

      if (scaleMode === 'norm') {
        const pctTicks = [-0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60];
        pctTicks.forEach(pct => {
          const lVal = Math.log(1 + pct);
          if (lVal >= minVal && lVal <= maxVal) {
            const y = getY(lVal);
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(w - padR, y);
            ctx.stroke();
            ctx.fillText(`${pct >= 0 ? '+' : ''}${(pct * 100).toFixed(0)}%`, padL - 8, y + 3);
          }
        });
      } else {
        const priceTicks = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200];
        priceTicks.forEach(p => {
          const lVal = Math.log(p);
          if (lVal >= minVal && lVal <= maxVal) {
            const y = getY(lVal);
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(w - padR, y);
            ctx.stroke();
            ctx.fillText(`$${p}`, padL - 8, y + 3);
          }
        });
      }

      // 3. Draw X-Axis Date Ticks
      ctx.textAlign = 'center';
      const step = Math.max(1, Math.floor(count / 8));
      for (let i = startIdx; i <= endIdx; i += step) {
        const x = getX(i);
        const tStr = appData.time_keys[i].slice(5, 16);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + chartH);
        ctx.stroke();
        ctx.fillText(tStr, x, h - 10);
      }

      // 4. Render Highlight Shapes and Shape ID Badges (Filtered by patternVisibility checkboxes!)
      allShapes.forEach(shape => {
        // Skip if this pattern's checkbox is unchecked
        if (!patternVisibility[shape.patternName]) return;
        if (shape.endIdx < startIdx || shape.startIdx > endIdx) return;

        const x0 = Math.max(padL, getX(shape.startIdx));
        const x1 = Math.min(w - padR, getX(shape.endIdx));
        const shapeW = Math.max(3, x1 - x0);

        const pCfg = PATTERNS_CONFIG[shape.patternName];
        const isFocused = (focusedShapeId === shape.globalId);

        const fillCol = isFocused ? 'rgba(250, 204, 21, 0.35)' : pCfg.bg;
        const borderCol = isFocused ? '#facc15' : pCfg.border;

        // Fill band
        ctx.fillStyle = fillCol;
        ctx.fillRect(x0, padT, shapeW, chartH);

        // Top line
        ctx.strokeStyle = borderCol;
        ctx.lineWidth = isFocused ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.moveTo(x0, padT);
        ctx.lineTo(x0 + shapeW, padT);
        ctx.stroke();

        // If focused, draw side vertical lines
        if (isFocused) {
          ctx.beginPath();
          ctx.moveTo(x0, padT);
          ctx.lineTo(x0, padT + chartH);
          ctx.moveTo(x0 + shapeW, padT);
          ctx.lineTo(x0 + shapeW, padT + chartH);
          ctx.stroke();
        }

        // Draw shape ID label if width allows or zoomed in
        if (shapeW >= 12 || count <= 650) {
          const badgeText = shape.typeId; // e.g. V弹-01, 拉升-01
          ctx.font = 'bold 9px monospace';
          const tw = ctx.measureText(badgeText).width;
          const bw = tw + 6;
          const bh = 14;
          const bx = Math.min(w - padR - bw, Math.max(padL, x0 + 1));
          const by = padT + 2;

          ctx.fillStyle = isFocused ? '#ca8a04' : pCfg.badgeBg;
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, 3);
          else ctx.rect(bx, by, bw, bh);
          ctx.fill();
          ctx.strokeStyle = borderCol;
          ctx.stroke();

          ctx.fillStyle = isFocused ? '#fff' : pCfg.color;
          ctx.textAlign = 'center';
          ctx.fillText(badgeText, bx + bw / 2, by + 10);
        }
      });

      // 5. Draw Stock Price Lines / Candlesticks
      if (scaleMode === 'candle') {
        appData.symbols.forEach(s => {
          if (s === candleTarget || !activeStockVisibility[s]) return;
          const info = appData.stocks[s];
          ctx.strokeStyle = COLORS[s];
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.35;
          ctx.beginPath();
          for (let i = startIdx; i <= endIdx; i++) {
            const y = getY(Math.log(info.close[i]));
            const x = getX(i);
            if (i === startIdx) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        });

        // Draw Candle for selected
        const cInfo = appData.stocks[candleTarget];
        const barW = Math.max(1, Math.floor((chartW / count) * 0.7));
        for (let i = startIdx; i <= endIdx; i++) {
          const o = getY(Math.log(cInfo.open[i]));
          const c = getY(Math.log(cInfo.close[i]));
          const hi = getY(Math.log(cInfo.high[i]));
          const lo = getY(Math.log(cInfo.low[i]));
          const x = getX(i);

          const isUp = cInfo.close[i] >= cInfo.open[i];
          ctx.strokeStyle = isUp ? '#10b981' : '#ef4444';
          ctx.fillStyle = isUp ? '#10b981' : '#ef4444';

          ctx.beginPath();
          ctx.moveTo(x, hi);
          ctx.lineTo(x, lo);
          ctx.stroke();

          const top = Math.min(o, c);
          const bot = Math.max(o, c);
          ctx.fillRect(x - barW / 2, top, Math.max(1, barW), Math.max(1, bot - top));
        }
      } else {
        appData.symbols.forEach(s => {
          if (!activeStockVisibility[s]) return;
          const info = appData.stocks[s];
          const p0 = info.p0;
          ctx.strokeStyle = COLORS[s];
          ctx.lineWidth = 1.8;
          ctx.beginPath();

          for (let i = startIdx; i <= endIdx; i++) {
            const val = (scaleMode === 'norm') ? Math.log(info.close[i] / p0) : Math.log(info.close[i]);
            const y = getY(val);
            const x = getX(i);
            if (i === startIdx) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        });
      }

      // 6. Draw Crosshair & Tooltip
      if (mouseHoverIdx !== null && mouseHoverIdx >= startIdx && mouseHoverIdx <= endIdx) {
        const hx = getX(mouseHoverIdx);
        ctx.strokeStyle = '#38bdf8';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(hx, padT);
        ctx.lineTo(hx, padT + chartH);
        ctx.stroke();
        ctx.setLineDash([]);

        const tooltip = document.getElementById('hud-tooltip');
        tooltip.style.display = 'block';
        const tKey = appData.time_keys[mouseHoverIdx];
        
        let tipHtml = `<div style="font-weight:700; color:#fff; border-bottom:1px solid #334155; padding-bottom:4px; margin-bottom:6px;">🕒 ${tKey}</div>`;
        appData.symbols.forEach(s => {
          const info = appData.stocks[s];
          const price = info.close[mouseHoverIdx];
          const retCum = ((price - info.p0) / info.p0) * 100;
          const roll = rollingReturns[s] ? (rollingReturns[s][mouseHoverIdx] * 100).toFixed(2) : '0.00';
          const rollNum = parseFloat(roll);

          let rollTag = '';
          if (rollNum >= M * 100) rollTag = ' <span style="color:#10b981; font-weight:700;">[🚀+M%]</span>';
          else if (rollNum <= -D * 100) rollTag = ' <span style="color:#ef4444; font-weight:700;">[📉-D%]</span>';

          tipHtml += `
            <div style="display:flex; justify-content:space-between; gap:16px;">
              <span style="color:${COLORS[s]}; font-weight:700;">${s}:</span>
              <span>$${price.toFixed(2)} (${retCum >= 0 ? '+' : ''}${retCum.toFixed(1)}%) | <strong>${rollNum >= 0 ? '+' : ''}${roll}%</strong>${rollTag}</span>
            </div>
          `;
        });

        // Show all active matching shapes at hover index
        const hoveredShapes = allShapes.filter(s => patternVisibility[s.patternName] && mouseHoverIdx >= s.startIdx && mouseHoverIdx <= s.endIdx);
        if (hoveredShapes.length > 0) {
          tipHtml += `<div style="margin-top:6px; padding-top:4px; border-top:1px dashed #334155; font-size:11px;">`;
          hoveredShapes.forEach(hs => {
            const pCfg = PATTERNS_CONFIG[hs.patternName] || { color: '#38bdf8' };
            tipHtml += `
              <div style="color:${pCfg.color}; display:flex; align-items:center; gap:4px; margin-top:2px;">
                🎯 命中图形: <strong>${hs.fullCode}</strong> · ${hs.patternName}
              </div>
            `;
          });
          tipHtml += `</div>`;
        }
        tooltip.innerHTML = tipHtml;
      }
    }

    // --- Chart 2: Navigator Brush Overview (Requirement 1 & 2) ---
    function renderSubChart() {
      const canvas = document.getElementById('sub-canvas');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      const padL = 65;
      const padR = 25;
      const padT = 10;
      const padB = 22;
      const chartW = w - padL - padR;
      const chartH = h - padT - padB;
      const totalBars = appData.time_keys.length;

      function getNavX(idx) {
        return padL + (idx / Math.max(1, totalBars - 1)) * chartW;
      }

      let rMin = -0.15;
      let rMax = 0.20;
      appData.symbols.forEach(s => {
        if (!rollingReturns[s]) return;
        const rets = rollingReturns[s];
        for (let i = 0; i < totalBars; i += 4) {
          if (rets[i] < rMin) rMin = rets[i];
          if (rets[i] > rMax) rMax = rets[i];
        }
      });
      rMin = Math.min(-D * 1.2, rMin * 1.05);
      rMax = Math.max(M * 1.2, rMax * 1.05);

      function getNavY(retVal) {
        return padT + chartH - ((retVal - rMin) / (rMax - rMin)) * chartH;
      }

      // 1. Draw Background Shapes (All Checked Colors) across 3 months
      allShapes.forEach(shape => {
        if (!patternVisibility[shape.patternName]) return;
        const x0 = getNavX(shape.startIdx);
        const x1 = getNavX(shape.endIdx);
        const shapeW = Math.max(2, x1 - x0);
        const pCfg = PATTERNS_CONFIG[shape.patternName];
        ctx.fillStyle = pCfg.bg;
        ctx.fillRect(x0, padT, shapeW, chartH);
        ctx.strokeStyle = pCfg.border;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, padT);
        ctx.lineTo(x0 + shapeW, padT);
        ctx.stroke();
      });

      // 2. Draw Zero and Threshold Lines
      const y0 = getNavY(0);
      ctx.strokeStyle = '#27354f';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y0);
      ctx.lineTo(w - padR, y0);
      ctx.stroke();

      const yM = getNavY(M);
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, yM);
      ctx.lineTo(w - padR, yM);
      ctx.stroke();

      const yD = getNavY(-D);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
      ctx.beginPath();
      ctx.moveTo(padL, yD);
      ctx.lineTo(w - padR, yD);
      ctx.stroke();
      ctx.setLineDash([]);

      // 3. Draw Mini Curves across 3 months
      appData.symbols.forEach(s => {
        if (!activeStockVisibility[s] || !rollingReturns[s]) return;
        const rets = rollingReturns[s];
        ctx.strokeStyle = COLORS[s];
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        const step = 2;
        for (let i = 0; i < totalBars; i += step) {
          const x = getNavX(i);
          const y = getNavY(rets[i]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      });

      // 4. X-Axis Date Ticks for Navigator
      ctx.fillStyle = '#64748b';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      const navTicks = [0, Math.floor(totalBars * 0.25), Math.floor(totalBars * 0.5), Math.floor(totalBars * 0.75), totalBars - 1];
      navTicks.forEach(idx => {
        const x = getNavX(idx);
        const tStr = appData.time_keys[idx].slice(5, 10);
        ctx.fillText(tStr, x, h - 5);
      });

      // 5. Interactive Selection Window & Handles (Requirement 1: 首尾区间选择器)
      const xStart = getNavX(timeRange[0]);
      const xEnd = getNavX(timeRange[1]);
      const selW = Math.max(4, xEnd - xStart);

      // Dimmed masks outside selection
      ctx.fillStyle = 'rgba(9, 14, 26, 0.65)';
      ctx.fillRect(padL, padT, Math.max(0, xStart - padL), chartH);
      ctx.fillRect(xEnd, padT, Math.max(0, (w - padR) - xEnd), chartH);

      // Active Selection Box
      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.fillRect(xStart, padT, selW, chartH);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(xStart, padT, selW, chartH);

      // Left Handle (首端手柄 ◀|)
      ctx.fillStyle = '#0284c7';
      ctx.fillRect(xStart - 4, padT + chartH / 2 - 12, 8, 24);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(xStart - 4, padT + chartH / 2 - 12, 8, 24);
      ctx.fillStyle = '#fff';
      ctx.fillRect(xStart - 1, padT + chartH / 2 - 6, 2, 12);

      // Right Handle (尾端手柄 |▶)
      ctx.fillStyle = '#0284c7';
      ctx.fillRect(xEnd - 4, padT + chartH / 2 - 12, 8, 24);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(xEnd - 4, padT + chartH / 2 - 12, 8, 24);
      ctx.fillStyle = '#fff';
      ctx.fillRect(xEnd - 1, padT + chartH / 2 - 6, 2, 12);
    }

    // Initialize on window load
    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""


def main():
    prepare_data_if_missing()
    with open(DATA_PATH, 'r') as f:
        raw_json_str = f.read()

    html_content = HTML_TEMPLATE.replace("__RAW_DATA_JSON__", raw_json_str)
    with open(HTML_PATH, 'w') as f:
        f.write(html_content)
    print(f"Generated standalone wireframe HTML at {HTML_PATH} ({HTML_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
