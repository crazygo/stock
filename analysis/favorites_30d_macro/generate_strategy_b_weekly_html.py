#!/usr/bin/env python3
"""
Generate interactive HTML report for Strategy B (2026-01 to 2026-09 Weekly Backtest)
Outputs:
  analysis/favorites_30d_macro/strategy_b_weekly_report.html
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_JSON = ROOT_DIR / "analysis" / "favorites_30d_macro" / "strategy_b_weekly_2026_results.json"
OUT_HTML = ROOT_DIR / "analysis" / "favorites_30d_macro" / "strategy_b_weekly_report.html"

def generate_html():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_str = json.dumps(data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>策略 B (Score >= 75) 2026 全年逐周回测报告</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent: #38bdf8;
      --accent-subtle: rgba(56, 189, 248, 0.12);
      --green: #10b981;
      --green-bg: rgba(16, 185, 129, 0.15);
      --red: #f43f5e;
      --red-bg: rgba(244, 63, 94, 0.15);
      --yellow: #f59e0b;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg);
      color: var(--text-main);
      line-height: 1.5;
      font-size: 14px;
      padding-bottom: 80px;
    }}

    .container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}

    /* Header & Critical Notice */
    .header-card {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}

    .header-card h1 {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .badge-tag {{
      display: inline-block;
      padding: 3px 10px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 4px;
      background: var(--accent-subtle);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      font-family: var(--font-mono);
    }}

    /* Execution Bias Correction Alert */
    .alert-box {{
      background: rgba(30, 41, 59, 0.8);
      border-left: 4px solid var(--accent);
      border-radius: 0 6px 6px 0;
      padding: 16px 20px;
      margin-top: 16px;
      border-top: 1px solid var(--card-border);
      border-right: 1px solid var(--card-border);
      border-bottom: 1px solid var(--card-border);
    }}

    .alert-title {{
      font-weight: 700;
      font-size: 14px;
      color: var(--accent);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    /* KPI Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}

    @media (max-width: 1200px) {{
      .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    @media (max-width: 768px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 16px;
    }}

    .kpi-label {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.03em;
      margin-bottom: 6px;
    }}

    .kpi-val {{
      font-family: var(--font-mono);
      font-size: 26px;
      font-weight: 700;
      line-height: 1.1;
      color: #fff;
    }}

    .kpi-sub {{
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 6px;
    }}

    /* Chart Section */
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}

    @media (max-width: 992px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
    }}

    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 18px 20px;
    }}

    .chart-title {{
      font-size: 14px;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .canvas-container {{
      position: relative;
      width: 100%;
      height: 220px;
    }}

    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}

    /* Table Section */
    .table-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 20px;
    }}

    .table-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 16px;
    }}

    .table-title {{
      font-size: 16px;
      font-weight: 700;
    }}

    .search-box {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 4px;
      padding: 6px 12px;
      color: #fff;
      font-size: 13px;
      width: 220px;
    }}

    .table-wrap {{
      overflow-x: auto;
    }}

    table.weekly-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}

    table.weekly-table th {{
      background: #0f172a;
      border-bottom: 2px solid var(--card-border);
      padding: 10px 12px;
      font-weight: 600;
      color: var(--text-muted);
      white-space: nowrap;
    }}

    table.weekly-table td {{
      border-bottom: 1px solid var(--card-border);
      padding: 10px 12px;
      color: var(--text-main);
      white-space: nowrap;
    }}

    table.weekly-table tr:hover td {{
      background: rgba(255, 255, 255, 0.03);
    }}

    table.weekly-table tr.expanded-row td {{
      background: rgba(15, 23, 42, 0.6);
      padding: 0;
    }}

    .num {{
      font-family: var(--font-mono);
      text-align: right;
    }}
    th.num {{ text-align: right; }}

    .badge-signal {{
      display: inline-block;
      min-width: 28px;
      text-align: center;
      padding: 2px 8px;
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 700;
      border-radius: 4px;
      background: #334155;
      color: #f8fafc;
    }}

    .badge-signal.high {{
      background: var(--accent-subtle);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}

    .pos-val {{ color: var(--green); }}
    .neg-val {{ color: var(--red); }}

    .expand-btn {{
      background: #334155;
      border: none;
      color: #fff;
      border-radius: 3px;
      width: 22px;
      height: 22px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 12px;
      transition: background 0.15s;
    }}
    .expand-btn:hover {{ background: #475569; }}

    /* Trade Details Container inside Row */
    .trades-detail-wrap {{
      padding: 16px 20px;
      border-top: 1px dashed var(--card-border);
      border-bottom: 1px dashed var(--card-border);
      background: #0b1120;
    }}

    .trade-subtable {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .trade-subtable th {{
      background: transparent;
      border-bottom: 1px solid #334155;
      color: var(--text-dim);
      padding: 6px 10px;
      font-size: 11px;
    }}
    .trade-subtable td {{
      border-bottom: 1px solid #1e293b;
      padding: 6px 10px;
      color: var(--text-muted);
    }}
    .trade-subtable tr:last-child td {{ border-bottom: none; }}
  </style>
</head>
<body>

  <div class="container">

    <!-- Header Card -->
    <header class="header-card">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
          <h1>
            <span>📈 策略 B (Score &ge; 75) 2026 全年逐周回测大盘</span>
            <span class="badge-tag">5M ROLLING · CAUSAL ENTRY</span>
          </h1>
          <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">
            回测时间：<strong>2026-01-02 至 2026-09-25 (全网 39 周)</strong> · 样本池：<strong>49 只自选核心股</strong> · 触发粒度：<strong>5 分钟无未来函数因果滚动</strong>
          </p>
        </div>
        <div style="text-align: right; font-size: 12px; color: var(--text-dim); font-family: var(--font-mono);">
          生成时间: {data['generated_at'][:19].replace('T', ' ')}
        </div>
      </div>

      <!-- Critical Notice on Look-Ahead Bias Correction -->
      <div class="alert-box">
        <div class="alert-title">
          <span>⚡ 核心机制修正说明：彻底剔除“事后全天最显著”未来数据偏误</span>
        </div>
        <p style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
          <strong>用户质疑证实有效</strong>：原先探索脚本中“日内挑选形态最显著的那 1 次”属于事后统计偏误（Look-Ahead Bias）——在盘中实时交易时，无法预知下午是否会出现更显著的形态，等待就会错失买点。<br>
          <strong>本次全量重构规则（Strictly Causal Real-Time Entry）</strong>：<br>
          ① <strong>盘中首破即买</strong>：当 5 分钟滑动窗口评估该股票 Score &ge; 75 的<strong>第一个瞬间</strong>，立即按该 5m 柱收盘价作为入场买入价；<br>
          ② <strong>持仓冷静期保护</strong>：买入后该标的锁定最多 3 个交易日（或触及 +5% 止盈退出），持仓期间该股票后续 5m 的重复越线不重复加仓，彻底还原真实实盘！
        </p>
      </div>
    </header>

    <!-- Top KPI Grid -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">全网跨度周数</div>
        <div class="kpi-val">{data['total_weeks']} <span style="font-size: 14px; font-weight: normal; color: var(--text-dim);">周</span></div>
        <div class="kpi-sub">184 个完整交易日</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">首破即买总开仓</div>
        <div class="kpi-val">{data['total_trades']} <span style="font-size: 14px; font-weight: normal; color: var(--text-dim);">次</span></div>
        <div class="kpi-sub">平均每周 10.9 笔 (日均 2.3 笔)</div>
      </div>
      <div class="kpi-card" style="border-color: rgba(56, 189, 248, 0.4);">
        <div class="kpi-label">3天触及+5%胜率</div>
        <div class="kpi-val" style="color: var(--accent);">{data['overall_hit_5pct_rate']}%</div>
        <div class="kpi-sub">3天触及+3%率达 {data['overall_hit_3pct_rate']}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">平均最大冲高</div>
        <div class="kpi-val pos-val">+{data['overall_avg_max_gain']}%</div>
        <div class="kpi-sub">平均最大回撤: {data['overall_avg_max_dd']}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">单笔平均期末收益</div>
        <div class="kpi-val pos-val">+{data['overall_avg_end_ret']}%</div>
        <div class="kpi-sub">每笔固定持仓 3 天收盘均益</div>
      </div>
      <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.4);">
        <div class="kpi-label">全年复利累计净值</div>
        <div class="kpi-val pos-val">+{data['final_cum_return_hold']}%</div>
        <div class="kpi-sub">按每周均益复合增长</div>
      </div>
    </section>

    <!-- Charts Grid -->
    <section class="charts-grid">
      <!-- Chart 1: Signals Count & Win Rate -->
      <div class="chart-card">
        <div class="chart-title">
          <span>📊 逐周发出信号次数与 +5% 达标胜率走势 (W01 ~ W39)</span>
          <span style="font-size: 11px; color: var(--text-dim);">柱状: 信号数 | 折线: +5%胜率</span>
        </div>
        <div class="canvas-container">
          <canvas id="chartWeeklySignals"></canvas>
        </div>
      </div>

      <!-- Chart 2: Cumulative Equity Curve -->
      <div class="chart-card">
        <div class="chart-title">
          <span>💰 2026 全年逐周资产净值复利增长曲线 (Equity Curve)</span>
          <span style="font-size: 11px; color: var(--green);">期末净值: +{data['final_cum_return_hold']}%</span>
        </div>
        <div class="canvas-container">
          <canvas id="chartEquityCurve"></canvas>
        </div>
      </div>
    </section>

    <!-- Full Weekly Breakdown Table -->
    <section class="table-card">
      <div class="table-toolbar">
        <div class="table-title">
          <span>📅 2026 全年 39 周逐周收益与交易清单 (点击左侧 '+' 可展开当周具体交易)</span>
        </div>
        <div>
          <input type="text" id="weekSearch" class="search-box" placeholder="搜索周号 / 股票代码..." oninput="filterTable()">
        </div>
      </div>

      <div class="table-wrap">
        <table class="weekly-table" id="mainTable">
          <thead>
            <tr>
              <th style="width: 36px;"></th>
              <th>周度 ID</th>
              <th>日期区间</th>
              <th class="num">交易日数</th>
              <th class="num">信号次数</th>
              <th>覆盖标的数</th>
              <th class="num">3天+5%胜率</th>
              <th class="num">3天+3%胜率</th>
              <th class="num">周均冲高</th>
              <th class="num">周均回撤</th>
              <th class="num">单笔均益</th>
              <th class="num">当周累计</th>
              <th class="num">资产净值</th>
            </tr>
          </thead>
          <tbody id="tableBody">
            <!-- Rendered by JS -->
          </tbody>
        </table>
      </div>
    </section>

  </div>

  <script>
    const BACKTEST_RESULTS = {json_str};

    // Render Table
    function renderWeeklyTable(filterQuery = "") {{
      const tbody = document.getElementById("tableBody");
      tbody.innerHTML = "";
      const q = filterQuery.toLowerCase().trim();

      BACKTEST_RESULTS.weekly_data.forEach((w, idx) => {{
        const matchWeek = w.week_id.toLowerCase().includes(q) || w.date_range.toLowerCase().includes(q);
        const matchSym = w.symbols.some(s => s.toLowerCase().includes(q));

        if (q && !matchWeek && !matchSym) return;

        const tr = document.createElement("tr");
        tr.id = `row-${{w.week_id}}`;

        const hasTrades = w.signals_count > 0;
        const expandBtn = hasTrades ? `<button class="expand-btn" onclick="toggleWeek('${{w.week_id}}')">+</button>` : '';

        const hit5Color = w.hit_5pct_rate >= 60 ? 'var(--green)' : (w.hit_5pct_rate >= 40 ? 'var(--text-main)' : 'var(--red)');
        const retHoldColor = w.weekly_ret_hold > 0 ? 'var(--green)' : (w.weekly_ret_hold < 0 ? 'var(--red)' : 'var(--text-dim)');
        const cumColor = w.cum_equity_hold >= 0 ? 'var(--green)' : 'var(--red)';

        tr.innerHTML = `
          <td>${{expandBtn}}</td>
          <td><strong>${{w.week_id}}</strong></td>
          <td style="color: var(--text-muted); font-size: 12px;">${{w.date_range}}</td>
          <td class="num">${{w.trading_days}}</td>
          <td class="num"><span class="badge-signal ${{w.signals_count >= 15 ? 'high' : ''}}">${{w.signals_count}}</span></td>
          <td><span style="font-size: 12px; color: var(--text-dim);">${{w.symbols_count}} 只</span></td>
          <td class="num" style="color: ${{hit5Color}}; font-weight: 700;">${{hasTrades ? w.hit_5pct_rate.toFixed(1) + '%' : '-'}}</td>
          <td class="num">${{hasTrades ? w.hit_3pct_rate.toFixed(1) + '%' : '-'}}</td>
          <td class="num pos-val">${{hasTrades ? '+' + w.avg_max_gain.toFixed(1) + '%' : '-'}}</td>
          <td class="num neg-val">${{hasTrades ? w.avg_max_dd.toFixed(1) + '%' : '-'}}</td>
          <td class="num" style="color: ${{retHoldColor}}; font-weight: 600;">${{hasTrades ? (w.avg_end_ret > 0 ? '+' : '') + w.avg_end_ret.toFixed(2) + '%' : '-'}}</td>
          <td class="num" style="color: ${{retHoldColor}}; font-weight: 700;">${{hasTrades ? (w.weekly_ret_hold > 0 ? '+' : '') + w.weekly_ret_hold.toFixed(2) + '%' : '0.00%'}}</td>
          <td class="num" style="color: ${{cumColor}}; font-weight: 700;">${{w.cum_equity_hold > 0 ? '+' : ''}}${{w.cum_equity_hold.toFixed(1)}}%</td>
        `;
        tbody.appendChild(tr);

        // Child row for expanded trades
        if (hasTrades) {{
          const subTr = document.createElement("tr");
          subTr.id = `detail-${{w.week_id}}`;
          subTr.className = "expanded-row";
          subTr.style.display = "none";

          let subTableRows = "";
          w.trades.forEach(t => {{
            const tWin = t.hit_5pct ? '<span style="color: var(--green); font-weight: bold;">✔ 达标(+5%)</span>' : '<span style="color: var(--text-dim);">未达标</span>';
            const retCol = t.end_ret > 0 ? 'var(--green)' : 'var(--red)';
            subTableRows += `
              <tr>
                <td><strong>${{t.ticker}}</strong></td>
                <td>${{t.name}}</td>
                <td style="font-family: var(--font-mono);">${{t.entry_time}}</td>
                <td><span style="font-family: var(--font-mono); font-size: 11px; background: #1e293b; padding: 2px 4px; border-radius: 3px;">${{t.atom}}</span></td>
                <td class="num">${{t.score.toFixed(1)}}</td>
                <td>${{tWin}}</td>
                <td class="num pos-val">+${{t.max_gain.toFixed(1)}}%</td>
                <td class="num neg-val">${{t.max_dd.toFixed(1)}}%</td>
                <td class="num" style="color: ${{retCol}}; font-weight: 600;">${{t.end_ret > 0 ? '+' : ''}}${{t.end_ret.toFixed(2)}}%</td>
              </tr>
            `;
          }});

          subTr.innerHTML = `
            <td colspan="13">
              <div class="trades-detail-wrap">
                <div style="font-size: 12px; font-weight: 700; margin-bottom: 8px; color: var(--accent);">
                  ${{w.week_id}} 当周触发的 ${{w.signals_count}} 笔首破开仓明细 (5m 实时因果入场):
                </div>
                <table class="trade-subtable">
                  <thead>
                    <tr>
                      <th>代码</th>
                      <th>名称</th>
                      <th>入场时点 (5m收盘)</th>
                      <th>形态</th>
                      <th class="num">综合打分</th>
                      <th>+5%达成状态</th>
                      <th class="num">3天最高冲高</th>
                      <th class="num">3天最大回撤</th>
                      <th class="num">3天收盘收益</th>
                    </tr>
                  </thead>
                  <tbody>${{subTableRows}}</tbody>
                </table>
              </div>
            </td>
          `;
          tbody.appendChild(subTr);
        }}
      }});
    }}

    function toggleWeek(wid) {{
      const detailRow = document.getElementById(`detail-${{wid}}`);
      const parentRow = document.getElementById(`row-${{wid}}`);
      const btn = parentRow.querySelector(".expand-btn");
      if (!detailRow) return;

      if (detailRow.style.display === "none") {{
        detailRow.style.display = "table-row";
        btn.textContent = "-";
        btn.style.background = "var(--accent)";
      }} else {{
        detailRow.style.display = "none";
        btn.textContent = "+";
        btn.style.background = "#334155";
      }}
    }}

    function filterTable() {{
      const q = document.getElementById("weekSearch").value;
      renderWeeklyTable(q);
    }}

    // Render Canvas Charts
    function drawCharts() {{
      const weeks = BACKTEST_RESULTS.weekly_data;
      const labels = weeks.map(w => w.week_id.replace("2026-", ""));
      const signals = weeks.map(w => w.signals_count);
      const winRates = weeks.map(w => w.signals_count > 0 ? w.hit_5pct_rate : 0);
      const equity = weeks.map(w => w.cum_equity_hold);

      // --- Chart 1: Signals & Win Rate ---
      const cvs1 = document.getElementById("chartWeeklySignals");
      const ctx1 = cvs1.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const w1 = cvs1.parentElement.clientWidth;
      const h1 = cvs1.parentElement.clientHeight;
      cvs1.width = w1 * dpr;
      cvs1.height = h1 * dpr;
      ctx1.scale(dpr, dpr);

      const padL = 36, padR = 36, padT = 20, padB = 30;
      const chartW = w1 - padL - padR;
      const chartH = h1 - padT - padB;
      const n = weeks.length;
      const stepX = chartW / (n - 1);

      // Max signals
      const maxSig = Math.max(...signals, 25);

      // Draw grid
      ctx1.strokeStyle = "#334155";
      ctx1.lineWidth = 0.5;
      for (let i = 0; i <= 4; i++) {{
        const y = padT + chartH * (i / 4);
        ctx1.beginPath();
        ctx1.moveTo(padL, y);
        ctx1.lineTo(w1 - padR, y);
        ctx1.stroke();
      }}

      // Draw Signal Bars
      const barW = Math.max(3, stepX * 0.6);
      weeks.forEach((w, i) => {{
        const x = padL + i * stepX - barW / 2;
        const bH = (w.signals_count / maxSig) * chartH;
        const y = padT + chartH - bH;
        ctx1.fillStyle = "rgba(56, 189, 248, 0.4)";
        ctx1.fillRect(x, y, barW, bH);
      }});

      // Draw Win Rate Line (right scale 0 ~ 100%)
      ctx1.beginPath();
      ctx1.strokeStyle = "#10b981";
      ctx1.lineWidth = 2;
      weeks.forEach((w, i) => {{
        const x = padL + i * stepX;
        const y = padT + chartH * (1.0 - (w.signals_count > 0 ? w.hit_5pct_rate : 0) / 100.0);
        if (i === 0) ctx1.moveTo(x, y);
        else ctx1.lineTo(x, y);
      }});
      ctx1.stroke();

      // Axis text
      ctx1.fillStyle = "#64748b";
      ctx1.font = "10px sans-serif";
      ctx1.fillText("W01", padL, h1 - 10);
      ctx1.fillText("W20", padL + stepX * 19, h1 - 10);
      ctx1.fillText("W39", w1 - padR - 15, h1 - 10);

      // --- Chart 2: Cumulative Equity Curve ---
      const cvs2 = document.getElementById("chartEquityCurve");
      const ctx2 = cvs2.getContext("2d");
      cvs2.width = w1 * dpr;
      cvs2.height = h1 * dpr;
      ctx2.scale(dpr, dpr);

      const minEq = Math.min(...equity, -10);
      const maxEq = Math.max(...equity, 70);
      const eqSpan = maxEq - minEq;

      // Draw zero line
      const zeroY = padT + chartH * ((maxEq - 0) / eqSpan);
      ctx2.strokeStyle = "#475569";
      ctx2.lineWidth = 1;
      ctx2.setLineDash([4, 4]);
      ctx2.beginPath();
      ctx2.moveTo(padL, zeroY);
      ctx2.lineTo(w1 - padR, zeroY);
      ctx2.stroke();
      ctx2.setLineDash([]);

      // Draw Equity Area & Line
      ctx2.beginPath();
      weeks.forEach((w, i) => {{
        const x = padL + i * stepX;
        const y = padT + chartH * ((maxEq - w.cum_equity_hold) / eqSpan);
        if (i === 0) ctx2.moveTo(x, y);
        else ctx2.lineTo(x, y);
      }});

      // Line style
      ctx2.strokeStyle = "#38bdf8";
      ctx2.lineWidth = 2.5;
      ctx2.stroke();

      // Area under curve
      ctx2.lineTo(padL + (n - 1) * stepX, padT + chartH);
      ctx2.lineTo(padL, padT + chartH);
      ctx2.closePath();
      const grad = ctx2.createLinearGradient(0, padT, 0, padT + chartH);
      grad.addColorStop(0, "rgba(56, 189, 248, 0.35)");
      grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");
      ctx2.fillStyle = grad;
      ctx2.fill();

      // X Axis text
      ctx2.fillStyle = "#64748b";
      ctx2.font = "10px sans-serif";
      ctx2.fillText("W01", padL, h1 - 10);
      ctx2.fillText("W20", padL + stepX * 19, h1 - 10);
      ctx2.fillText("W39", w1 - padR - 15, h1 - 10);
      ctx2.fillText("+0%", padL - 25, zeroY + 3);
      ctx2.fillText(`+${{maxEq.toFixed(0)}}%`, padL - 32, padT + 10);
    }}

    window.onload = function() {{
      renderWeeklyTable();
      drawCharts();
      window.addEventListener("resize", drawCharts);
    }};
  </script>
</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated HTML report at {OUT_HTML} ({len(html_content)} bytes)")

if __name__ == "__main__":
    generate_html()
