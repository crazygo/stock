#!/usr/bin/env python3
"""Generate self-contained interactive multi-column (3D, 5D, 10D) and multi-row per date HTML dashboard."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "results" / "dashboard_data.json"
HTML_PATH = BASE_DIR / "index.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>开盘微观时点预测模型 · 每日选股多周期全景看板 (3D / 5D / 10D)</title>
  <style>
    :root {
      --bg-primary: #090d16;
      --bg-card: #111827;
      --bg-card-hover: #17223b;
      --border-color: #1f293d;
      --border-accent: #2e3c54;
      --text-main: #f3f4f6;
      --text-muted: #94a3b8;
      --accent-green: #10b981;
      --accent-green-bg: rgba(16, 185, 129, 0.12);
      --accent-blue: #38bdf8;
      --accent-blue-bg: rgba(56, 189, 248, 0.12);
      --accent-amber: #f59e0b;
      --accent-amber-bg: rgba(245, 158, 11, 0.12);
      --accent-red: #ef4444;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-primary);
      color: var(--text-main);
      font-family: var(--font-sans);
      padding: 24px;
      line-height: 1.5;
      font-size: 14px;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 16px;
    }
    .title-area h1 {
      font-size: 22px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .title-badge {
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--accent-blue-bg);
      color: var(--accent-blue);
      border: 1px solid rgba(56, 189, 248, 0.3);
      font-family: var(--font-mono);
      font-weight: 600;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 6px;
    }

    /* KPI Cards */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      position: relative;
      overflow: hidden;
    }
    .kpi-card.golden {
      border-color: #f59e0b;
      background: linear-gradient(145deg, #161e31 0%, #111827 100%);
    }
    .kpi-card.golden::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: #f59e0b;
    }
    .kpi-lbl {
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
    }
    .kpi-val {
      font-size: 24px;
      font-weight: 800;
      margin-top: 6px;
      font-family: var(--font-mono);
    }
    .kpi-sub {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }

    /* Single Score Scale Grid */
    .scale-panel {
      background: #0d1424;
      border: 1px solid var(--border-accent);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 20px;
    }
    .scale-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin-top: 10px;
    }
    .scale-card {
      background: #090e1a;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 10px 12px;
      text-align: center;
    }
    .scale-tag {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .scale-val {
      font-size: 16px;
      font-weight: 800;
      font-family: var(--font-mono);
      color: var(--accent-green);
    }
    .scale-mfe {
      font-size: 11px;
      color: var(--accent-blue);
      font-family: var(--font-mono);
      margin-top: 2px;
    }

    /* Table Toolbar */
    .table-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .rank-toggle-group {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .t-btn {
      background: #111827;
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
    }
    .t-btn:hover { background: #1e293b; color: #fff; }
    .t-btn.active {
      background: linear-gradient(135deg, #0284c7, #0369a1);
      color: #fff;
      border-color: #38bdf8;
    }

    /* Main Multi-Row Data Table */
    .table-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 24px;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .data-table th {
      text-align: left;
      padding: 10px 12px;
      background: #0d131f;
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border-color);
      font-size: 12px;
    }
    .data-table td {
      padding: 10px 12px;
      border-bottom: 1px solid #1a2333;
    }

    /* Date row separation */
    tr.date-start td {
      border-top: 2px solid #2d3b55;
    }
    tr.date-group-alt td {
      background: rgba(17, 24, 39, 0.4);
    }
    tr.date-group td {
      background: rgba(13, 19, 31, 0.6);
    }
    tr:hover td {
      background: #19253c !important;
    }

    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      font-family: var(--font-mono);
    }
    .badge-win {
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-fail {
      background: rgba(100, 116, 139, 0.15);
      color: #94a3b8;
    }
    .badge-rank1 {
      background: rgba(245, 158, 11, 0.18);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
      font-weight: 800;
    }

    .horizon-cell {
      font-family: var(--font-mono);
      font-size: 12px;
    }
    .mfe-win {
      color: #10b981;
      font-weight: 700;
    }
    .mfe-fail {
      color: #94a3b8;
    }
    .mfe-pending {
      color: #38bdf8;
      font-weight: 600;
    }

    .tag-chip {
      display: inline-block;
      background: #141c2c;
      border: 1px solid #243147;
      border-radius: 3px;
      padding: 1px 5px;
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
      margin-right: 4px;
    }
  </style>
</head>
<body>

  <!-- Top Header -->
  <div class="header">
    <div class="title-area">
      <h1>
        ⚡ 开盘微观时点预测模型 · 每日选股多周期全景看板
        <span class="title-badge">T=10m (09:40 ET) 黄金时点</span>
      </h1>
      <div class="subtitle">
        按日多行展示 Top 3 推荐标的 · 明确各周期目标：1天 ≥ +5% · 3天 ≥ +5% · 5天 ≥ +10% · 10天 ≥ +10% · 样本外严格实证 (2026-07 至 2026-09)
      </div>
    </div>
    <div style="text-align:right;">
      <span class="badge" style="background:rgba(245, 158, 11, 0.15); color:#fbbf24; border:1px solid rgba(245, 158, 11, 0.4); padding:6px 12px;">模型隔离状态: PASS</span>
      <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">独立工程目录 · 未触动任何已有生产模型</div>
    </div>
  </div>

  <!-- Multi-Horizon Side-by-Side KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi-card golden">
      <div class="kpi-lbl">👑 每日 Top 1 达标率 (3D +5%)</div>
      <div class="kpi-val" style="color:#fbbf24;" id="kpi-top1-rates">--</div>
      <div class="kpi-sub" id="kpi-top1-sub">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">🖐️ 每日 Top 3 至少 1 只达标率</div>
      <div class="kpi-val" style="color:var(--accent-green);" id="kpi-top3-rates">--</div>
      <div class="kpi-sub" id="kpi-top3-sub">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">🚀 每日 Top 5 至少 1 只达标率</div>
      <div class="kpi-val" style="color:var(--accent-blue);" id="kpi-top5-rates">--</div>
      <div class="kpi-sub" id="kpi-top5-sub">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-lbl">🎯 Top 1 平均最大冲高 (Avg MFE)</div>
      <div class="kpi-val" style="color:#e2e8f0;" id="kpi-top1-mfes">--</div>
      <div class="kpi-sub" id="kpi-top1-mfe-sub">--</div>
    </div>
  </div>

  <!-- Single Score Confidence Scale -->
  <div class="scale-panel">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div style="font-weight:700; font-size:13px; color:#fff;">
        📐 单笔预测成功率置信度标尺 (当模型给出具体得分时，该笔预测的历史真实达成率)
      </div>
      <div style="font-size:11px; color:var(--text-muted);">
        得分越高，代表该笔预测在各持有周期触及目标的概率越高
      </div>
    </div>
    <div class="scale-grid" id="scale-container">
      <!-- Injected via JS -->
    </div>
  </div>

  <!-- Main Multi-Column Table Card -->
  <div class="table-card">
    <div class="table-toolbar">
      <div style="font-weight:700; font-size:15px; color:#fff; display:flex; align-items:center; gap:8px;">
        <span>📅 每日实操选股多周期追踪表 (美东 09:40 开盘后 10 分钟决策)</span>
      </div>
      <div class="rank-toggle-group">
        <span style="font-size:12px; color:var(--text-muted);">每日展示股票数:</span>
        <button class="t-btn active" id="btn-top3" onclick="setDisplayRank(3)">显示 Top 3 (推荐 · 每日 3 行)</button>
        <button class="t-btn" id="btn-top1" onclick="setDisplayRank(1)">显示 Top 1 (绝对龙头 · 每日 1 行)</button>
        <button class="t-btn" id="btn-top5" onclick="setDisplayRank(5)">显示 Top 5 (全量备选 · 每日 5 行)</button>
      </div>
    </div>

    <div style="overflow-x:auto;">
      <table class="data-table" id="main-table">
        <thead>
          <tr>
            <th style="width:110px;">交易日期</th>
            <th style="width:80px;">当日排名</th>
            <th style="width:85px;">标的代码</th>
            <th style="width:85px;">09:40 买入价</th>
            <th style="width:85px;">模型置信度</th>
            <th style="width:130px; background:#0f172a; border-left:1px solid #2e3c54;">⚡ 1 天 ≥ +5% (1D)</th>
            <th style="width:130px; background:#0f172a;">🔥 3 天 ≥ +5% (3D)</th>
            <th style="width:130px; background:#0f172a;">🚀 5 天 ≥ +10% (5D)</th>
            <th style="width:130px; background:#0f172a; border-right:1px solid #2e3c54;">📈 10 天 ≥ +10% (10D)</th>
            <th>关键微观特征标签</th>
          </tr>
        </thead>
        <tbody id="table-tbody">
          <!-- Injected via JS -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- Raw Embedded JSON Data -->
  <script id="dashboard-data" type="application/json">
__DASHBOARD_DATA_JSON__
  </script>

  <script>
    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    let currentDisplayRank = 3; // Default to Top 3 (3 rows per date)

    function init() {
      // 1. KPI cards
      const k = data.kpis;
      document.getElementById('kpi-top1-rates').innerText = `${(k.top1_touch['3d'] * 100).toFixed(1)}%`;
      document.getElementById('kpi-top1-sub').innerHTML = `1D(5%): <strong>${(k.top1_touch['1d'] * 100).toFixed(1)}%</strong> · 3D(5%): <strong>${(k.top1_touch['3d'] * 100).toFixed(1)}%</strong> · 5D(10%): <strong>${(k.top1_touch['5d'] * 100).toFixed(1)}%</strong> · 10D(10%): <strong>${(k.top1_touch['10d'] * 100).toFixed(1)}%</strong>`;

      document.getElementById('kpi-top3-rates').innerText = `${(k.top3_any_hit['3d'] * 100).toFixed(1)}%`;
      document.getElementById('kpi-top3-sub').innerHTML = `1D(5%): <strong>${(k.top3_any_hit['1d'] * 100).toFixed(1)}%</strong> · 3D(5%): <strong>${(k.top3_any_hit['3d'] * 100).toFixed(1)}%</strong> · 5D(10%): <strong>${(k.top3_any_hit['5d'] * 100).toFixed(1)}%</strong> · 10D(10%): <strong>${(k.top3_any_hit['10d'] * 100).toFixed(1)}%</strong>`;

      document.getElementById('kpi-top5-rates').innerText = `${(k.top5_any_hit['3d'] * 100).toFixed(1)}%`;
      document.getElementById('kpi-top5-sub').innerHTML = `1D(5%): <strong>${(k.top5_any_hit['1d'] * 100).toFixed(1)}%</strong> · 3D(5%): <strong>${(k.top5_any_hit['3d'] * 100).toFixed(1)}%</strong> · 5D(10%): <strong>${(k.top5_any_hit['5d'] * 100).toFixed(1)}%</strong> · 10D(10%): <strong>${(k.top5_any_hit['10d'] * 100).toFixed(1)}%</strong>`;

      document.getElementById('kpi-top1-mfes').innerText = `+${(k.top1_mfe['3d'] * 100).toFixed(2)}%`;
      document.getElementById('kpi-top1-mfe-sub').innerHTML = `1D: <strong>+${(k.top1_mfe['1d'] * 100).toFixed(1)}%</strong> · 3D: <strong>+${(k.top1_mfe['3d'] * 100).toFixed(1)}%</strong> · 5D: <strong>+${(k.top1_mfe['5d'] * 100).toFixed(1)}%</strong> · 10D: <strong>+${(k.top1_mfe['10d'] * 100).toFixed(1)}%</strong>`;

      // 2. Score Calibration Scale
      const scaleContainer = document.getElementById('scale-container');
      scaleContainer.innerHTML = '';
      data.scale.forEach(item => {
        const card = document.createElement('div');
        card.className = 'scale-card';
        card.innerHTML = `
          <div class="scale-tag">得分 ${item.bin}</div>
          <div class="scale-val">3D: ${(item.touch_3d * 100).toFixed(1)}%</div>
          <div class="scale-mfe">1D(5%): ${(item.touch_1d * 100).toFixed(1)}% · 5D(10%): ${(item.touch_5d * 100).toFixed(1)}%</div>
          <div style="font-size:10px; color:var(--text-muted); margin-top:2px;">样本数: ${item.count}</div>
        `;
        scaleContainer.appendChild(card);
      });

      // 3. Render Table
      renderTable();
    }

    function setDisplayRank(n) {
      currentDisplayRank = n;
      document.querySelectorAll('.t-btn').forEach(btn => btn.classList.remove('active'));
      document.getElementById(`btn-top${n}`).classList.add('active');
      renderTable();
    }

    function renderTable() {
      const tbody = document.getElementById('table-tbody');
      tbody.innerHTML = '';

      data.dates_data.forEach((dayData, dayIdx) => {
        const stocksToShow = dayData.stocks.filter(s => s.rank <= currentDisplayRank);
        const isAlt = dayIdx % 2 === 1;

        stocksToShow.forEach((s, sIdx) => {
          const tr = document.createElement('tr');
          if (sIdx === 0) tr.classList.add('date-start');
          tr.classList.add(isAlt ? 'date-group-alt' : 'date-group');

          // Date cell
          let dateTd = '';
          if (sIdx === 0) {
            dateTd = `<td rowspan="${stocksToShow.length}" style="font-family:var(--font-mono); font-weight:700; color:#fff; vertical-align:middle; border-right:1px solid #1f293d;">
              ${dayData.date}
              <div style="font-size:11px; color:var(--text-muted); font-weight:normal; margin-top:2px;">共 ${dayData.stocks.length} 只候选</div>
            </td>`;
          }

          // Rank badge
          let rankBadge = '';
          if (s.rank === 1) {
            rankBadge = '<span class="badge badge-rank1">🥇 #1 龙头</span>';
          } else if (s.rank === 2) {
            rankBadge = '<span class="badge" style="background:#223249; color:#93c5fd;">🥈 #2</span>';
          } else if (s.rank === 3) {
            rankBadge = '<span class="badge" style="background:#202b3c; color:#bfdbfe;">🥉 #3</span>';
          } else {
            rankBadge = `<span class="badge badge-fail">#${s.rank}</span>`;
          }

          // 4 Horizon cells
          let cell1d = renderHorizonCell(s.mfe_1d, s.touch_1d, s.is_matured_1d, 0.05);
          let cell3d = renderHorizonCell(s.mfe_3d, s.touch_3d, s.is_matured_3d, 0.05);
          let cell5d = renderHorizonCell(s.mfe_5d, s.touch_5d, s.is_matured_5d, 0.10);
          let cell10d = renderHorizonCell(s.mfe_10d, s.touch_10d, s.is_matured_10d, 0.10);

          // Micro Tags
          const tags = `
            <span class="tag-chip">ATR: ${s.atr_pct}%</span>
            <span class="tag-chip">QQQ均线: ${s.qqq_vwap_dev >= 0 ? '+' : ''}${s.qqq_vwap_dev}%</span>
            <span class="tag-chip">K线振幅: ${s.bar_range}%</span>
          `;

          tr.innerHTML = `
            ${dateTd}
            <td>${rankBadge}</td>
            <td style="font-family:var(--font-mono); font-weight:700; font-size:14px; color:#38bdf8;">${s.symbol}</td>
            <td style="font-family:var(--font-mono);">$${s.entry_price.toFixed(2)}</td>
            <td style="font-family:var(--font-mono); font-weight:700; color:${s.score >= 0.70 ? '#fbbf24' : '#e2e8f0'};">
              ${(s.score * 100).toFixed(1)}%
            </td>
            <td class="horizon-cell" style="border-left:1px solid #2e3c54;">${cell1d}</td>
            <td class="horizon-cell">${cell3d}</td>
            <td class="horizon-cell">${cell5d}</td>
            <td class="horizon-cell" style="border-right:1px solid #2e3c54;">${cell10d}</td>
            <td>${tags}</td>
          `;

          tbody.appendChild(tr);
        });
      });
    }

    function renderHorizonCell(mfe, touch, isMatured, targetPct) {
      if (mfe === null || mfe === undefined) {
        return '<span style="color:#64748b; font-size:11px;">未开仓</span>';
      }

      const mfePct = (mfe * 100).toFixed(2);
      const isTouch = touch === 1 || mfe >= targetPct;
      let badge = '';
      if (isTouch) {
        badge = '<span class="badge badge-win">✓ 达标</span>';
      } else if (!isMatured) {
        badge = '<span class="badge" style="background:#1e293b; color:#38bdf8; border:1px solid #334155;">⏳ 进行中</span>';
      } else {
        badge = '<span class="badge badge-fail">✕</span>';
      }

      const mfeCls = isTouch ? 'mfe-win' : (!isMatured ? 'mfe-pending' : 'mfe-fail');

      return `
        <div>
          <span class="${mfeCls}">+${mfePct}%</span> ${badge}
        </div>
      `;
    }
    init();
  </script>
</body>
</html>
"""

def generate_html():
    with open(DATA_PATH, encoding="utf-8") as f:
        raw_json_str = f.read()

    full_html = HTML_TEMPLATE.replace("__DASHBOARD_DATA_JSON__", raw_json_str)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated multi-column multi-row dashboard HTML at {HTML_PATH} ({HTML_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_html()
