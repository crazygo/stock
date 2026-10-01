#!/usr/bin/env python3
"""
Generate complete interactive HTML visualization for:
1. KOD vs NVTS combined single-chart overlay & NVTS 04:01 green bar mystery
2. User's exact strategy cases:
   - 3 True Continuation Successes (Pre-market > 5% AND Regular continues to gain > 5%): KOD, NCPL_WIN, IVVD
   - 3 Premarket Traps (Pre-market > 5% BUT Regular crashes): MRM, NCPL_TRAP, AUID_TRAP
3. Macro statistical account of all 119 Pre-market > 5% cases across 2,451 stock-days
4. Mathematical calibration matrix
"""
import os
import json

SCRATCH_DIR = "/Users/admin/.gemini/antigravity-cli/brain/f2466b1a-bce0-48dc-9661-08ba17afc795/scratch"

with open(os.path.join(SCRATCH_DIR, "combined_kod_nvts.json")) as f:
    combined_kod_nvts = json.load(f)

with open(os.path.join(SCRATCH_DIR, "strategy_signal_cases.json")) as f:
    strategy_cases = json.load(f)

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>盘前信号买入策略（PM > 5% 且 Regular > 5%）微观量化与算法校准看板</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #3b82f6;
      --success: #10b981;
      --danger: #ef4444;
      --warning: #f59e0b;
      --cyan: #06b6d4;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 24px;
    }}
    .header {{
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
    }}
    .title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{ font-size: 22px; font-weight: 700; color: #fff; }}
    .subtitle {{ color: var(--text-muted); font-size: 13.5px; margin-top: 4px; }}
    
    .stats-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
    }}
    .stat-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
    .stat-val {{ font-size: 22px; font-weight: 700; margin-top: 4px; font-family: var(--font-mono); }}
    .stat-desc {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

    .tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 8px;
      overflow-x: auto;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 10px 18px;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }}
    .tab-btn:hover {{ color: #fff; background: rgba(255,255,255,0.05); }}
    .tab-btn.active {{
      background: #1e293b;
      color: #38bdf8;
      border-color: #0284c7;
    }}

    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}

    .case-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      margin-bottom: 24px;
    }}
    .case-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .case-badge {{
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
    .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }}

    .quant-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      background: rgba(0,0,0,0.3);
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 16px;
    }}
    .q-lbl {{ font-size: 11px; color: var(--text-muted); }}
    .q-val {{ font-size: 15px; font-weight: 700; font-family: var(--font-mono); margin-top: 2px; }}

    .chart-container {{
      width: 100%;
      height: 380px;
      position: relative;
      background: #0d131f;
      border-radius: 8px;
      border: 1px solid #1f2937;
      overflow: hidden;
    }}
    .chart-container-tall {{
      height: 440px;
    }}
    svg.stock-chart {{
      width: 100%;
      height: 100%;
      display: block;
    }}

    .legend-bar {{
      display: flex;
      gap: 16px;
      font-size: 12px;
      margin-top: 10px;
      color: var(--text-muted);
      flex-wrap: wrap;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 2px; }}

    .matrix-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 13px;
    }}
    .matrix-table th, .matrix-table td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--card-border);
    }}
    .matrix-table th {{
      background: #172033;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
    }}
    .matrix-table tr:hover td {{ background: rgba(255,255,255,0.02); }}

    .formula-box {{
      background: #172033;
      border-left: 4px solid var(--accent);
      padding: 14px;
      border-radius: 4px;
      margin-bottom: 16px;
      font-family: var(--font-mono);
      font-size: 13px;
    }}

    .callout-box {{
      background: rgba(245, 158, 11, 0.08);
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: 8px;
      padding: 18px;
      margin-top: 20px;
    }}
    .callout-title {{
      font-size: 16px;
      font-weight: 700;
      color: #fbbf24;
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .toggle-group {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .toggle-btn {{
      background: #1f2937;
      border: 1px solid #374151;
      color: #d1d5db;
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
    }}
    .toggle-btn.active {{
      background: #0284c7;
      color: #fff;
      border-color: #38bdf8;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title-row">
      <div>
        <h1>盘前信号买入策略（Premarket &gt; 5% 且 Regular 延续 &gt; 5%）微观量化与算法校准看板</h1>
        <div class="subtitle">严格面向实盘交易入场逻辑：盘前发出强度信号买入，检验常规盘能否真正接力爆拉</div>
      </div>
      <div>
        <span style="font-size: 12px; color: var(--text-muted); background: #172033; padding: 6px 12px; border-radius: 4px; border: 1px solid var(--card-border);">
          当前系统时间: 2026-09-29 18:05 (美东 06:05)
        </span>
      </div>
    </div>

    <!-- Macro Strategy Stats Bar -->
    <div class="stats-bar">
      <div class="stat-card">
        <div class="stat-label">近30天全量样本基底</div>
        <div class="stat-val" style="color: #fff;">2,451 个</div>
        <div class="stat-desc">涵盖 129 只异动活跃股票日度样本</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">盘前涨幅 &ge; +5% 触发信号</div>
        <div class="stat-val" style="color: #38bdf8;">119 次</div>
        <div class="stat-desc">占全部交易日样本的 4.85%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">常规盘自开盘继续涨 &ge; +5%</div>
        <div class="stat-val" style="color: #10b981;">42.86%</div>
        <div class="stat-desc">51 / 119 次成功打出日内 +5% 溢价</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">常规盘收盘稳赢 &ge; +5%</div>
        <div class="stat-val" style="color: #f59e0b;">12.61%</div>
        <div class="stat-desc">仅 15 次收盘收益守住 +5% 以上</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">假突破开盘即闷杀比例</div>
        <div class="stat-val" style="color: #ef4444;">28.57%</div>
        <div class="stat-desc">34 次开盘后完全不涨且暴跌 &gt; 15%</div>
      </div>
    </div>
  </div>

  <!-- Navigation Tabs -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-compare')">① KOD 与 NVTS 同图对比 & 绿柱解密</button>
    <button class="tab-btn" onclick="switchTab('tab-strategy-success')">② 3 个真实策略延续大肉案例 (KOD, NCPL, IVVD)</button>
    <button class="tab-btn" onclick="switchTab('tab-strategy-traps')">③ 3 个盘前高开闷杀陷阱案例 (MRM, NCPL, AUID)</button>
    <button class="tab-btn" onclick="switchTab('tab-matrix')">④ 算法校准指标数学量化矩阵</button>
  </div>

  <!-- TAB 1: KOD vs NVTS -->
  <div id="tab-compare" class="tab-pane active">
    <div class="case-card">
      <div class="case-header">
        <div>
          <span style="font-size: 20px; font-weight: 700; color: #fff;">KOD vs NVTS 同图归一化对比看板</span>
          <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">
            同时间轴（2026-09-28 04:00 至 2026-09-29 05:35）· 308 根 5m K 线同步对齐
          </div>
        </div>
        <div class="toggle-group">
          <button id="btn-mode-pct" class="toggle-btn active" onclick="toggleCombinedMode('pct')">归一化百分比走势 (%)</button>
          <button id="btn-mode-dual" class="toggle-btn" onclick="toggleCombinedMode('dual')">双 Y 轴价格走势 ($)</button>
        </div>
      </div>

      <div class="quant-grid">
        <div class="quant-item">
          <div class="q-lbl">KOD 累计极值涨幅</div>
          <div class="q-val" style="color: #06b6d4;">+150.8% ($89.89)</div>
        </div>
        <div class="quant-item">
          <div class="q-lbl">NVTS 累计极值涨幅</div>
          <div class="q-val" style="color: #f59e0b;">+14.5% ($13.50)</div>
        </div>
        <div class="quant-item">
          <div class="q-lbl">KOD 盘前接力比 (VR)</div>
          <div class="q-val" style="color: #10b981;">179.2x (313万股)</div>
        </div>
        <div class="quant-item">
          <div class="q-lbl">NVTS 盘前接力比 (VR)</div>
          <div class="q-val" style="color: #ef4444;">1.11x (平移无增量)</div>
        </div>
        <div class="quant-item">
          <div class="q-lbl">KOD 盘前动能斜率</div>
          <div class="q-val" style="color: #10b981;">+0.42% / 5m</div>
        </div>
        <div class="quant-item">
          <div class="q-lbl">NVTS 盘前动能斜率</div>
          <div class="q-val" style="color: #ef4444;">-0.04% / 5m</div>
        </div>
      </div>

      <div class="chart-container chart-container-tall" id="chart-combined"></div>

      <div class="legend-bar">
        <div class="legend-item"><div class="dot" style="background: #06b6d4;"></div><strong style="color:#06b6d4;">KOD</strong> (青色线)</div>
        <div class="legend-item"><div class="dot" style="background: #f59e0b;"></div><strong style="color:#f59e0b;">NVTS</strong> (橙色线)</div>
        <div class="legend-item"><div class="dot" style="background: rgba(139, 92, 246, 0.4);"></div>夜盘 (20:00-04:00)</div>
        <div class="legend-item"><div class="dot" style="background: rgba(245, 158, 11, 0.4);"></div>盘前 (04:00-09:30)</div>
        <div class="legend-item"><div class="dot" style="background: rgba(16, 185, 129, 0.4);"></div>常规盘 (09:30-16:00)</div>
        <div class="legend-item"><span style="color:#ef4444; font-weight:bold;">★ 标红点:</span> 04:01 NVTS 巨型长绿柱位置</div>
      </div>
    </div>

    <!-- EXPLANATION OF NVTS 04:01 GREEN LINE -->
    <div class="callout-box">
      <div class="callout-title">
        <span>🔍</span> NVTS 在 Pre-Market 初期（04:01）那条“极高绿线”究竟代表了什么？
      </div>
      <div style="font-size: 13.5px; color: #e5e7eb; line-height: 1.8;">
        <p><strong>1. 盘面现象：</strong> 在 04:01:00（或 04:05 5m K 线），NVTS 出现垂直暴拉的<strong>实体大阳线</strong>：Open <strong>$11.7300</strong>，High <strong>$13.2800</strong>，Close <strong>$13.2400</strong>，单根蜡烛垂直跨度高达 <strong>+$1.51 (+12.87%)</strong>！</p>
        
        <p style="margin-top: 8px;"><strong>2. 微观撮合机制解密：</strong></p>
        <ul style="padding-left: 20px; margin-top: 4px;">
          <li><strong>$11.73 是前一日常规盘的收盘基准价</strong>：美东 04:00 盘前 ECN 正式开市，系统登记的首笔撮合为“开市基准对盘单（Opening Cross）”或跨市场 Form T 延迟报单，其锚定价格为前一日 16:00 的收盘价 <strong>$11.7300</strong>。</li>
          <li><strong>夜盘真实市价早已在 $13.20 附近</strong>：在此之前的夜盘中，NVTS 早已在 $13.15~$13.25 交易了整整 8 个小时；04:01 盘前挂单撮合的实际成交价直接在 <strong>$13.20 ~ $13.28</strong>。</li>
          <li><strong>K 线合成算法导致的“视觉巨阳错觉”</strong>：由于该分钟记录了 <code>Open = $11.73</code>，<code>Close = $13.20</code>，图表系统按规则画出了一根 +12.87% 的实体火箭柱。</li>
        </ul>

        <p style="margin-top: 8px;"><strong>3. 算法校准必做：</strong> 必须在数据清洗中剔除 04:00~04:02 昨收价基准孤点，否则量化系统会误认为“盘前 1 分钟放量突发暴拉 +13%”而发出虚假的追涨买入指令！</p>
      </div>
    </div>
  </div>

  <!-- TAB 2: TRUE SUCCESS CASES -->
  <div id="tab-strategy-success" class="tab-pane">
    <div class="case-card" style="border-left: 4px solid #10b981;">
      <h3 style="color:#10b981; margin-bottom: 6px;">符合期望的延续案例特征：盘前确认放量突破（PM &gt; +5%）+ 常规盘开盘继续狂飙（Regular &gt; +5%）</h3>
      <p style="font-size: 13px; color: var(--text-muted);">
        此类标的在 09:30 常规盘开盘直接买入，不仅能够稳稳斩获 +10% ~ +50% 的日内极值冲刺，且收盘均守住巨大浮盈（零回撤或极小回撤）。
      </p>
    </div>
    <div id="container-strategy-success"></div>
  </div>

  <!-- TAB 3: SEVERE TRAP CASES -->
  <div id="tab-strategy-traps" class="tab-pane">
    <div class="case-card" style="border-left: 4px solid #ef4444;">
      <h3 style="color:#ef4444; margin-bottom: 6px;">致命诱多陷阱案例特征：盘前看似暴涨（PM &gt; +5% ~ +20%），但常规盘开盘瞬间见顶、单边崩跌 -25% 以上</h3>
      <p style="font-size: 13px; color: var(--text-muted);">
        此类标的盘前涨幅虚高（通常伴随缩量或盘前末端阴跌），在 09:30 开盘一买入立刻成为接盘侠，开盘后连 1 分钱都没有再涨过，全天下杀 -20% ~ -28%！
      </p>
    </div>
    <div id="container-strategy-traps"></div>
  </div>

  <!-- TAB 4: CALIBRATION MATRIX -->
  <div id="tab-matrix" class="tab-pane">
    <div class="case-card">
      <h2 style="margin-bottom: 14px; color: #fff;">算法校准：区分“真突破翻倍”与“缩量闷杀陷阱”的五大数学指标</h2>
      
      <div class="formula-box">
        1. 盘前放量倍率 (Premarket Volume Ratio): Vol_PM &gt; 200,000 股<br>
        2. 常规/盘前换手接力比: Ratio = Vol_Regular / Vol_Premarket &gt; 3.0x<br>
        3. 盘前末端动能斜率 (Late PM Slope): (Close_0925 - Open_0900) / Open_0900 &gt; 0.0% (严禁末端漏水)<br>
        4. 盘前高点保留率 (Retention): Close_0925 / High_PM &ge; 90.0%<br>
        5. 开盘防破位硬止损: 买入后若跌破 Open_0930 的 -3.0%，无条件市价止损！
      </div>

      <table class="matrix-table" id="matrix-strategy-table">
        <thead>
          <tr>
            <th>标的代码</th>
            <th>标的名称</th>
            <th>类型标签</th>
            <th>前日收盘价</th>
            <th>盘前收盘价</th>
            <th>盘前涨幅</th>
            <th>盘前成交量</th>
            <th>常规盘开盘价</th>
            <th>常规盘最高价</th>
            <th>常规盘自开盘涨幅</th>
            <th>开盘最大下杀浮亏</th>
            <th>常规盘收盘收益</th>
          </tr>
        </thead>
        <tbody id="matrix-strategy-tbody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const combinedData = {json.dumps(combined_kod_nvts)};
    const strategyCases = {json.dumps(strategy_cases)};

    let currentCombinedMode = 'pct';

    function switchTab(tabId) {{
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
    }}

    function toggleCombinedMode(mode) {{
      currentCombinedMode = mode;
      document.getElementById('btn-mode-pct').classList.toggle('active', mode === 'pct');
      document.getElementById('btn-mode-dual').classList.toggle('active', mode === 'dual');
      renderCombinedChart();
    }}

    // Render Combined KOD & NVTS Chart
    function renderCombinedChart() {{
      const container = document.getElementById('chart-combined');
      if (!container || !combinedData || combinedData.length === 0) return;

      const width = 1000;
      const height = 420;
      const padding = {{ top: 30, right: 70, bottom: 40, left: 60 }};
      const chartW = width - padding.left - padding.right;
      const chartH = height - padding.top - padding.bottom;

      const n = combinedData.length;
      const step = chartW / n;

      const kPcts = combinedData.map(d => d.k_pct);
      const nPcts = combinedData.map(d => d.n_pct);
      const kPrices = combinedData.map(d => d.k_close);
      const nPrices = combinedData.map(d => d.n_close);

      let minK, maxK, minN, maxN;
      if (currentCombinedMode === 'pct') {{
        const allPcts = [...kPcts, ...nPcts];
        minK = Math.min(...allPcts) - 0.05;
        maxK = Math.max(...allPcts) + 0.10;
        minN = minK;
        maxN = maxK;
      }} else {{
        minK = Math.min(...kPrices) * 0.95;
        maxK = Math.max(...kPrices) * 1.05;
        minN = Math.min(...nPrices) * 0.95;
        maxN = Math.max(...nPrices) * 1.05;
      }}

      function yK(v) {{ return padding.top + (1 - (v - minK) / (maxK - minK)) * chartH; }}
      function yN(v) {{ return padding.top + (1 - (v - minN) / (maxN - minN)) * chartH; }}

      // Session bands
      let sessionBands = '';
      let curSession = null;
      let startX = 0;
      combinedData.forEach((b, i) => {{
        const sType = b.session;
        if (sType !== curSession) {{
          if (curSession !== null) {{
            const bandW = i * step - startX;
            let col = 'rgba(255,255,255,0.015)';
            if (curSession === 'Overnight') col = 'rgba(139, 92, 246, 0.08)';
            if (curSession === 'Pre-Market') col = 'rgba(245, 158, 11, 0.08)';
            if (curSession === 'Regular') col = 'rgba(16, 185, 129, 0.08)';
            if (curSession === 'Post-Market') col = 'rgba(59, 130, 246, 0.08)';
            sessionBands += `<rect x="${{padding.left + startX}}" y="${{padding.top}}" width="${{bandW}}" height="${{chartH}}" fill="${{col}}" />`;
            sessionBands += `<text x="${{padding.left + startX + 6}}" y="${{padding.top + 14}}" fill="#64748b" font-size="10" font-family="monospace">${{curSession}}</text>`;
          }}
          curSession = sType;
          startX = i * step;
        }}
      }});
      if (curSession !== null) {{
        const bandW = chartW - startX;
        let col = 'rgba(255,255,255,0.015)';
        if (curSession === 'Overnight') col = 'rgba(139, 92, 246, 0.08)';
        if (curSession === 'Pre-Market') col = 'rgba(245, 158, 11, 0.08)';
        if (curSession === 'Regular') col = 'rgba(16, 185, 129, 0.08)';
        if (curSession === 'Post-Market') col = 'rgba(59, 130, 246, 0.08)';
        sessionBands += `<rect x="${{padding.left + startX}}" y="${{padding.top}}" width="${{bandW}}" height="${{chartH}}" fill="${{col}}" />`;
        sessionBands += `<text x="${{padding.left + startX + 6}}" y="${{padding.top + 14}}" fill="#64748b" font-size="10" font-family="monospace">${{curSession}}</text>`;
      }}

      // Path lines
      let pathK = '';
      let pathN = '';
      combinedData.forEach((d, i) => {{
        const x = padding.left + i * step + step / 2;
        const valK = currentCombinedMode === 'pct' ? d.k_pct : d.k_close;
        const valN = currentCombinedMode === 'pct' ? d.n_pct : d.n_close;
        const yCoordK = yK(valK);
        const yCoordN = yN(valN);
        if (i === 0) {{ pathK += `M ${{x}} ${{yCoordK}}`; pathN += `M ${{x}} ${{yCoordN}}`; }}
        else {{ pathK += ` L ${{x}} ${{yCoordK}}`; pathN += ` L ${{x}} ${{yCoordN}}`; }}
      }});

      // 04:01 green line callout marker
      const targetIdx = 289;
      const markerX = padding.left + targetIdx * step;
      const markerY = yN(currentCombinedMode === 'pct' ? combinedData[targetIdx].n_pct : combinedData[targetIdx].n_close);
      const calloutElement = `
        <line x1="${{markerX}}" y1="${{padding.top}}" x2="${{markerX}}" y2="${{padding.top + chartH}}" stroke="#ef4444" stroke-dasharray="3,3" stroke-width="1.5"/>
        <circle cx="${{markerX}}" cy="${{markerY}}" r="6" fill="#ef4444" stroke="#fff" stroke-width="2"/>
        <rect x="${{markerX - 165}}" y="${{markerY - 38}}" width="160" height="28" rx="4" fill="#1e1e2d" stroke="#ef4444" stroke-width="1"/>
        <text x="${{markerX - 85}}" y="${{markerY - 20}}" fill="#fbbf24" font-size="11" font-weight="bold" text-anchor="middle">★ 04:01 巨型绿柱位置</text>
      `;

      // Y-axis labels
      let yLabelsLeft = '';
      let yLabelsRight = '';
      const gridCount = 5;
      for (let g = 0; g <= gridCount; g++) {{
        const frac = g / gridCount;
        const yPos = padding.top + frac * chartH;
        sessionBands += `<line x1="${{padding.left}}" y1="${{yPos}}" x2="${{padding.left + chartW}}" y2="${{yPos}}" stroke="#1f2937" stroke-dasharray="2,2"/>`;
        if (currentCombinedMode === 'pct') {{
          const pctVal = maxK - frac * (maxK - minK);
          yLabelsLeft += `<text x="${{padding.left - 8}}" y="${{yPos + 4}}" fill="#94a3b8" font-size="10" font-family="monospace" text-anchor="end">${{(pctVal*100).toFixed(0)}}%</text>`;
        }} else {{
          const kVal = maxK - frac * (maxK - minK);
          const nVal = maxN - frac * (maxN - minN);
          yLabelsLeft += `<text x="${{padding.left - 8}}" y="${{yPos + 4}}" fill="#06b6d4" font-size="10" font-family="monospace" text-anchor="end">$${{kVal.toFixed(1)}}</text>`;
          yLabelsRight += `<text x="${{padding.left + chartW + 8}}" y="${{yPos + 4}}" fill="#f59e0b" font-size="10" font-family="monospace" text-anchor="start">$${{nVal.toFixed(2)}}</text>`;
        }}
      }}

      // X-axis time labels
      let xLabels = '';
      const xInterval = Math.floor(n / 6);
      combinedData.forEach((d, i) => {{
        if (i % xInterval === 0 || i === n - 1) {{
          const x = padding.left + i * step + step / 2;
          const dtText = d.time_key.slice(5, 16);
          xLabels += `<text x="${{x}}" y="${{height - 10}}" fill="#64748b" font-size="10" text-anchor="middle" font-family="monospace">${{dtText}}</text>`;
        }}
      }});

      container.innerHTML = `
        <svg class="stock-chart" viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
          ${{sessionBands}}
          <path d="${{pathK}}" fill="none" stroke="#06b6d4" stroke-width="2.5" />
          <path d="${{pathN}}" fill="none" stroke="#f59e0b" stroke-width="2.5" />
          ${{calloutElement}}
          ${{yLabelsLeft}}
          ${{yLabelsRight}}
          ${{xLabels}}
        </svg>
      `;
    }}

    // Render Strategy Card
    function renderStrategyCard(c, containerElemId) {{
      const container = document.getElementById(containerElemId);
      if (!container) return;

      const isSuccess = c.type === 'true_success';
      const badgeClass = isSuccess ? 'badge-success' : 'badge-fail';
      const badgeText = isSuccess ? '策略成功：盘中狂飙' : '致命陷阱：开盘崩跌';
      const regGainCol = c.gain_from_open >= 0.05 ? '#10b981' : '#ef4444';
      const regCloseCol = c.close_gain_from_open >= 0 ? '#10b981' : '#ef4444';

      const cardHtml = `
        <div class="case-card">
          <div class="case-header">
            <div>
              <span style="font-size: 19px; font-weight: 700; color: #fff;">${{c.id}}</span>
              <span style="font-size: 14px; color: var(--text-muted); margin-left: 8px;">${{c.name}} (${{c.symbol}}) · 日期: ${{c.date}}</span>
              <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">${{c.desc}}</div>
            </div>
            <span class="case-badge ${{badgeClass}}">${{badgeText}}</span>
          </div>

          <div class="quant-grid">
            <div class="quant-item">
              <div class="q-lbl">前日收盘价</div>
              <div class="q-val">$${{c.prev_close.toFixed(2)}}</div>
            </div>
            <div class="quant-item">
              <div class="q-lbl">盘前收盘涨幅 (信号发出)</div>
              <div class="q-val" style="color: #38bdf8;">+${{(c.pm_gain*100).toFixed(1)}}% ($${{c.pm_close.toFixed(2)}})</div>
            </div>
            <div class="quant-item">
              <div class="q-lbl">常规盘开盘价 (买入入场价)</div>
              <div class="q-val" style="color: #fff;">$${{c.reg_open.toFixed(2)}}</div>
            </div>
            <div class="quant-item">
              <div class="q-lbl">常规盘自开盘继续冲高</div>
              <div class="q-val" style="color: ${{regGainCol}};">+${{(c.gain_from_open*100).toFixed(1)}}% ($${{c.reg_high.toFixed(2)}})</div>
            </div>
            <div class="quant-item">
              <div class="q-lbl">常规盘开盘最大下杀浮亏</div>
              <div class="q-val" style="color: #ef4444;">${{(c.dd_from_open*100).toFixed(1)}}% ($${{c.reg_low.toFixed(2)}})</div>
            </div>
            <div class="quant-item">
              <div class="q-lbl">16:00 常规盘收盘收益</div>
              <div class="q-val" style="color: ${{regCloseCol}};">${{c.close_gain_from_open >= 0 ? '+' : ''}}${{(c.close_gain_from_open*100).toFixed(1)}}% ($${{c.reg_close.toFixed(2)}})</div>
            </div>
            <div class="quant-item">
              <div class="q-lbl">常规/盘前成交量比 (接力比)</div>
              <div class="q-val" style="color: ${{c.vol_ratio_reg_pm > 3.0 ? '#10b981' : '#f59e0b'}};">${{c.vol_ratio_reg_pm.toFixed(1)}}x</div>
            </div>
          </div>

          <div class="chart-container" id="chart-${{c.id}}"></div>

          <div class="legend-bar">
            <div class="legend-item"><div class="dot" style="background: rgba(245, 158, 11, 0.4);"></div>盘前 (04:00-09:30)</div>
            <div class="legend-item"><div class="dot" style="background: rgba(16, 185, 129, 0.4);"></div>常规盘 (09:30-16:00)</div>
            <div class="legend-item"><span style="color:#38bdf8; font-weight:bold;">-- 蓝虚线:</span> 09:30 开盘入场价 ($${{c.reg_open.toFixed(2)}})</div>
            <div class="legend-item"><span style="color:#10b981; font-weight:bold;">-- 绿虚线:</span> 入场 +5% 目标位 ($${{(c.reg_open*1.05).toFixed(2)}})</div>
          </div>
        </div>
      `;
      container.innerHTML += cardHtml;
    }}

    // Render Strategy SVG Candlestick
    function renderStrategyCandles(c) {{
      const container = document.getElementById(`chart-${{c.id}}`);
      if (!container || !c.bars || c.bars.length === 0) return;

      const bars = c.bars;
      const width = 1000;
      const height = 360;
      const padding = {{ top: 20, right: 65, bottom: 40, left: 10 }};
      const chartW = width - padding.left - padding.right;
      const priceH = height * 0.7 - padding.top;
      const volH = height * 0.25;

      const prices = bars.map(b => [b.open, b.high, b.low, b.close]).flat();
      prices.push(c.reg_open * 1.05);
      const minP = Math.min(...prices) * 0.98;
      const maxP = Math.max(...prices) * 1.02;
      const maxV = Math.max(...bars.map(b => b.volume)) || 1;

      const n = bars.length;
      const barW = Math.max(2.0, (chartW / n) * 0.75);
      const step = chartW / n;

      function pY(p) {{ return padding.top + (1 - (p - minP) / (maxP - minP)) * priceH; }}
      function vH(v) {{ return (v / maxV) * volH; }}

      // Session background
      let sessionBands = '';
      let curSession = null;
      let startX = 0;
      bars.forEach((b, i) => {{
        const sType = b.session;
        if (sType !== curSession) {{
          if (curSession !== null) {{
            const bandW = i * step - startX;
            let col = curSession === 'Pre-Market' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(16, 185, 129, 0.08)';
            sessionBands += `<rect x="${{startX}}" y="${{padding.top}}" width="${{bandW}}" height="${{height - padding.top - padding.bottom}}" fill="${{col}}" />`;
            sessionBands += `<text x="${{startX + 8}}" y="${{padding.top + 14}}" fill="#64748b" font-size="10" font-family="monospace">${{curSession}}</text>`;
          }}
          curSession = sType;
          startX = i * step;
        }}
      }});
      if (curSession !== null) {{
        const bandW = chartW - startX;
        let col = curSession === 'Pre-Market' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(16, 185, 129, 0.08)';
        sessionBands += `<rect x="${{startX}}" y="${{padding.top}}" width="${{bandW}}" height="${{height - padding.top - padding.bottom}}" fill="${{col}}" />`;
        sessionBands += `<text x="${{startX + 8}}" y="${{padding.top + 14}}" fill="#64748b" font-size="10" font-family="monospace">${{curSession}}</text>`;
      }}

      // 09:30 entry line
      const openY = pY(c.reg_open);
      const openLine = `<line x1="0" y1="${{openY}}" x2="${{chartW}}" y2="${{openY}}" stroke="#38bdf8" stroke-dasharray="3,3" stroke-width="1.2" opacity="0.8"/>
      <text x="${{chartW + 4}}" y="${{openY + 4}}" fill="#38bdf8" font-size="10" font-family="monospace">Open ($${{c.reg_open.toFixed(2)}})</text>`;

      // +5% Target line from open
      const targetP = c.reg_open * 1.05;
      const targetY = pY(targetP);
      let targetLine = '';
      if (targetY >= padding.top && targetY <= padding.top + priceH) {{
        targetLine = `<line x1="0" y1="${{targetY}}" x2="${{chartW}}" y2="${{targetY}}" stroke="#10b981" stroke-dasharray="4,4" stroke-width="1.5" opacity="0.9"/>
        <text x="${{chartW + 4}}" y="${{targetY + 4}}" fill="#10b981" font-size="10" font-family="monospace">+5% ($${{targetP.toFixed(2)}})</text>`;
      }}

      // Candles
      let candleElements = '';
      let volumeElements = '';
      let timeLabels = '';
      const labelInterval = Math.floor(n / 6);

      bars.forEach((b, i) => {{
        const x = i * step + step / 2;
        const isUp = b.close >= b.open;
        const col = isUp ? '#10b981' : '#ef4444';

        const yH = pY(b.high);
        const yL = pY(b.low);
        candleElements += `<line x1="${{x}}" y1="${{yH}}" x2="${{x}}" y2="${{yL}}" stroke="${{col}}" stroke-width="1.2"/>`;

        const yO = pY(b.open);
        const yC = pY(b.close);
        const topY = Math.min(yO, yC);
        const bodyH = Math.max(1.5, Math.abs(yC - yO));
        candleElements += `<rect x="${{x - barW/2}}" y="${{topY}}" width="${{barW}}" height="${{bodyH}}" fill="${{col}}" />`;

        const vHeight = vH(b.volume);
        volumeElements += `<rect x="${{x - barW/2}}" y="${{height - padding.bottom - vHeight}}" width="${{barW}}" height="${{vHeight}}" fill="${{col}}" opacity="0.4" />`;

        if (i % labelInterval === 0 || i === n - 1) {{
          const tText = b.time_key.slice(11, 16);
          timeLabels += `<text x="${{x}}" y="${{height - 10}}" fill="#64748b" font-size="10" text-anchor="middle" font-family="monospace">${{tText}}</text>`;
        }}
      }});

      container.innerHTML = `
        <svg class="stock-chart" viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
          ${{sessionBands}}
          ${{openLine}}
          ${{targetLine}}
          ${{candleElements}}
          ${{volumeElements}}
          ${{timeLabels}}
        </svg>
      `;
    }}

    function renderStrategyTable() {{
      const tbody = document.getElementById('matrix-strategy-tbody');
      if (!tbody) return;

      tbody.innerHTML = strategyCases.map(c => {{
        const isSuccess = c.type === 'true_success';
        const badge = isSuccess ? '<span class="case-badge badge-success">真实延续</span>' : '<span class="case-badge badge-fail">诱多闷杀</span>';
        const regGainCol = c.gain_from_open >= 0.05 ? '#10b981' : '#ef4444';
        const regCloseCol = c.close_gain_from_open >= 0 ? '#10b981' : '#ef4444';

        return `
          <tr>
            <td style="font-weight:700; color:#fff;">${{c.id}}</td>
            <td>${{c.name}} (${{c.date}})</td>
            <td>${{badge}}</td>
            <td style="font-family:monospace;">$${{c.prev_close.toFixed(2)}}</td>
            <td style="font-family:monospace;">$${{c.pm_close.toFixed(2)}}</td>
            <td style="font-family:monospace; font-weight:700; color:#38bdf8;">+${{(c.pm_gain*100).toFixed(1)}}%</td>
            <td style="font-family:monospace;">${{c.pm_vol.toLocaleString()}}</td>
            <td style="font-family:monospace;">$${{c.reg_open.toFixed(2)}}</td>
            <td style="font-family:monospace;">$${{c.reg_high.toFixed(2)}}</td>
            <td style="font-family:monospace; font-weight:700; color:${{regGainCol}};">+${{(c.gain_from_open*100).toFixed(1)}}%</td>
            <td style="font-family:monospace; color:#ef4444;">${{(c.dd_from_open*100).toFixed(1)}}%</td>
            <td style="font-family:monospace; font-weight:700; color:${{regCloseCol}};">${{c.close_gain_from_open>=0?'+':''}}${{(c.close_gain_from_open*100).toFixed(1)}}%</td>
          </tr>
        `;
      }}).join('');
    }}

    window.onload = function() {{
      renderCombinedChart();
      
      // Render success cards
      strategyCases.filter(c => c.type === 'true_success').forEach(c => {{
        renderStrategyCard(c, 'container-strategy-success');
        renderStrategyCandles(c);
      }});

      // Render trap cards
      strategyCases.filter(c => c.type === 'severe_trap').forEach(c => {{
        renderStrategyCard(c, 'container-strategy-traps');
        renderStrategyCandles(c);
      }});

      renderStrategyTable();
    }};
  </script>
</body>
</html>
"""

target_file = "/Users/admin/Code/stock/analysis/overnight_surge_case_study/index.html"
with open(target_file, "w") as f:
    f.write(html_content)

print(f"Generated strategy-focused {target_file} successfully! Size: {len(html_content)} bytes")
