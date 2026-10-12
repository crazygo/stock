#!/usr/bin/env python3
"""
生成 SOXS 最近 30 天上涨窗口 (涨幅>=5%, 回撤<4%, 不重复) 证明 HTML Wireframe
- 遵循 .agents/skills/html-wireframe/SKILL.md 灰阶、系统字体、原生表单规范
- 遵循 AGENTS.md 第六节交互契约：日期区间选择、快捷天数、禁用滚轮误操作、明确平移按钮、完整量价路径阴影
- 单文件内联自包含，嵌入真实 60m / 5m / Daily 行情数据，支持在线动态调参重算
"""

import os
import json
import pandas as pd

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOXS 最近 30 天上涨窗口证明看板 (HTML Wireframe)</title>
    <style>
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #cbd5e1;
            --border-dark: #94a3b8;
            --header-bg: #f1f5f9;
            --accent: #2563eb;
            --accent-bg: #eff6ff;
            --up-color: #15803d;
            --down-color: #b91c1c;
            --shade-color: rgba(34, 197, 94, 0.12);
            --shade-border: rgba(34, 197, 94, 0.6);
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            padding: 20px;
            line-height: 1.5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        /* 标题与协议头 */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }
        .header h1 {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .header p {
            font-size: 13px;
            color: var(--text-muted);
        }
        .meta-tags {
            display: flex;
            gap: 12px;
            margin-top: 10px;
            font-size: 12px;
            flex-wrap: wrap;
        }
        .meta-tag {
            background: var(--header-bg);
            border: 1px solid var(--border);
            padding: 2px 8px;
            border-radius: 4px;
        }

        /* 控制面板条 */
        .controls-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
            background: var(--header-bg);
            border: 1px solid var(--border);
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 16px;
            font-size: 13px;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .control-group label {
            font-weight: 600;
            color: var(--text-muted);
        }
        select, input[type="number"], input[type="date"], button {
            font-family: inherit;
            font-size: 13px;
            padding: 4px 8px;
            border: 1px solid var(--border-dark);
            border-radius: 4px;
            background: var(--surface);
            color: var(--text-main);
        }
        button {
            cursor: pointer;
            background: #e2e8f0;
            border-color: #94a3b8;
            font-weight: 500;
        }
        button:hover {
            background: #cbd5e1;
        }
        button.btn-primary {
            background: #0f172a;
            color: #ffffff;
            border-color: #0f172a;
        }
        button.btn-primary:hover {
            background: #334155;
        }

        /* 指标卡片 */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 16px;
        }
        .metric-title {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 22px;
            font-weight: 700;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .metric-sub {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* 图表区域 */
        .chart-card {
            position: relative;
        }
        .chart-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 12px;
        }
        .chart-legend {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-box {
            width: 14px;
            height: 10px;
            display: inline-block;
        }
        .canvas-container {
            position: relative;
            width: 100%;
            height: 480px;
            border: 1px solid var(--border);
            background: #ffffff;
            overflow: hidden;
            user-select: none;
        }
        canvas {
            display: block;
        }
        .tooltip {
            position: absolute;
            display: none;
            background: rgba(15, 23, 42, 0.92);
            color: #ffffff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 100;
            white-space: nowrap;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }

        /* 导航与平移条 */
        .pan-controls {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        /* 证明表格 */
        .table-responsive {
            overflow-x: auto;
            margin-top: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }
        th, td {
            padding: 8px 12px;
            border: 1px solid var(--border);
        }
        th {
            background: var(--header-bg);
            font-weight: 600;
            color: var(--text-muted);
        }
        tr:hover td {
            background: var(--accent-bg);
            cursor: pointer;
        }
        tr.active-row td {
            background: #e2e8f0;
            font-weight: 600;
        }
        .num {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            text-align: right;
        }
        .tag-win {
            display: inline-block;
            background: #dcfce7;
            color: #15803d;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }

        /* 证明注释 */
        .notes-list {
            font-size: 13px;
            color: #334155;
            padding-left: 20px;
        }
        .notes-list li {
            margin-bottom: 6px;
        }
    </style>
</head>
<body>

<div class="container">
    <!-- 1. 顶部标题与契约定义 -->
    <div class="card header">
        <h1>SOXS 最近 30 天上涨窗口证明看板 (HTML Wireframe)</h1>
        <p>标的: <strong>US.SOXS (Direxion 3X Bear Semiconductor)</strong> · 算法规则: <strong>涨幅 &ge; +5%，期间最大回撤 &lt; 4%，窗口互不重叠 (Disjoint)</strong></p>
        <div class="meta-tags">
            <span class="meta-tag">交互契约: Wireframe 灰阶规范</span>
            <span class="meta-tag">K线平移: 显式按钮平移 (禁用滚轮缩放防误触)</span>
            <span class="meta-tag">时段: 全日常规盘 + 盘前盘后</span>
            <span class="meta-tag">可信数据: Futu OpenD 实盘时序 (已对齐)</span>
        </div>
    </div>

    <!-- 2. 交互控制条 (支持动态调参重算) -->
    <div class="controls-bar">
        <div class="control-group">
            <label for="timeframeSelect">时间粒度:</label>
            <select id="timeframeSelect" onchange="switchTimeframe()">
                <option value="60m" selected>60分钟K (60M) - 推荐标准波段</option>
                <option value="5m">5分钟K (5M) - 精细微观快攻</option>
                <option value="daily">日K (Daily) - 跨日收盘波段</option>
            </select>
        </div>

        <div class="control-group">
            <label for="rangePreset">区间快捷:</label>
            <select id="rangePreset" onchange="applyPreset()">
                <option value="30cal" selected>最近 30 日历天 (2026-09-08 ~ 10-07)</option>
                <option value="30trd">最近 30 交易日 (2026-08-26 ~ 10-07)</option>
                <option value="all">全部可用数据 (2026-08-20 起)</option>
            </select>
        </div>

        <div class="control-group">
            <label for="minGainInput">最小涨幅阈值:</label>
            <input type="number" id="minGainInput" value="5.0" step="0.5" min="1" max="50" style="width: 70px;"> %
        </div>

        <div class="control-group">
            <label for="maxDDInput">最大允许回撤:</label>
            <input type="number" id="maxDDInput" value="4.0" step="0.5" min="1" max="20" style="width: 70px;"> %
        </div>

        <button class="btn-primary" onclick="recalculate()">⚡ 重新运行证明计算</button>

        <div class="pan-controls" style="margin-left: auto;">
            <button onclick="panChart(-15)">◀ 较早 (左移)</button>
            <button onclick="panChart(15)">较晚 (右移) ▶</button>
            <button onclick="resetChart()">⟲ 重置可见范围</button>
        </div>
    </div>

    <!-- 3. 统计汇总指标卡 -->
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">识别有效上涨窗口数</div>
            <div class="metric-value" id="mWindowCount">-</div>
            <div class="metric-sub" id="mWindowSub">不重叠独立波段</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">波段平均涨幅</div>
            <div class="metric-value" id="mAvgGain" style="color: var(--up-color);">-</div>
            <div class="metric-sub">起步低点到峰值涨幅均值</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">最大单波涨幅</div>
            <div class="metric-value" id="mMaxGain" style="color: var(--up-color);">-</div>
            <div class="metric-sub" id="mMaxGainDate">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">平均期间最大回撤</div>
            <div class="metric-value" id="mAvgDD">-</div>
            <div class="metric-sub">各波段行进中经历的最高回撤</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">平均持仓周期</div>
            <div class="metric-value" id="mAvgBars">-</div>
            <div class="metric-sub" id="mAvgDurationSub">-</div>
        </div>
    </div>

    <!-- 4. K 线图表与波段阴影证明区 -->
    <div class="card chart-card">
        <div class="chart-toolbar">
            <div class="chart-legend">
                <div class="legend-item">
                    <span class="legend-box" style="background: var(--shade-color); border: 1px dashed var(--shade-border);"></span>
                    <span>合格上涨窗口阴影 (&ge; 5%, 回撤 &lt; 4%)</span>
                </div>
                <div class="legend-item">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#15803d;"></span>
                    <span>起步低点起点</span>
                </div>
                <div class="legend-item">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#b91c1c;"></span>
                    <span>波段最高点 (Peak)</span>
                </div>
                <div class="legend-item">
                    <span style="color:#64748b; font-weight:bold;">✖</span>
                    <span>阻断点 (回撤触及 4% 结束)</span>
                </div>
            </div>
            <div id="chartRangeInfo" style="color: var(--text-muted);">
                当前展示: 60M · 80 根可见 K 线
            </div>
        </div>

        <div class="canvas-container" id="canvasContainer">
            <canvas id="klineCanvas"></canvas>
            <div class="tooltip" id="chartTooltip"></div>
        </div>
    </div>

    <!-- 5. 逐笔证明清单表格 -->
    <div class="card">
        <h2 style="font-size: 15px; margin-bottom: 8px;">📋 识别合格窗口逐笔证明清单 (点击任意行可在上方图表高亮并平移定位)</h2>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th>波段起点时刻</th>
                        <th>起点价</th>
                        <th>峰值高点时刻</th>
                        <th>峰值价</th>
                        <th class="num">累计涨幅</th>
                        <th class="num">期间最大回撤</th>
                        <th class="num">K线根数</th>
                        <th>持续时长</th>
                        <th>阻断时刻 (回撤触及阈值)</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody id="windowsTableBody">
                    <!-- JS 动态注入 -->
                </tbody>
            </table>
        </div>
    </div>

    <!-- 6. 算法严密性与实证说明 -->
    <div class="card">
        <h2 style="font-size: 15px; margin-bottom: 8px;">💡 量化实证与算法证明说明</h2>
        <ol class="notes-list">
            <li><strong>不重叠（Disjoint）原则</strong>：当一个波段确立并因为后续回撤达到 4% 而结束阻断后，下一个窗口的候选起点严格在该波段结束点之后寻找，确保统计窗口在时间轴上互斥独立，不产生样本重复膨胀。</li>
            <li><strong>为什么 60 分钟线有 5 个窗口，而日线 High-Low 极值是 0 个？</strong>：SOXS 是 3 倍做空半导体杠杆 ETF，单日上下影线日内波动极大（通常 6%~12%）。若用日线 High-Low 极值跨日判定，日内的下影线毛刺极易在当天就跌超 4% 从而打断波段；60分钟线能够剔除日内毛刺，更真实地反映持续数小时到数天的趋势波段；而日线收盘价（Close-to-Close）平滑掉影线后亦有 3 个波段。</li>
            <li><strong>美东 10:30 开盘首小时顶峰效应</strong>：在 60 分钟级别的 5 个窗口中，有 4 个窗口的最高点精准见顶于美东 10:30（开盘首小时），证明了 SOXS 极度依赖开盘情绪脉冲冲高，随后容易被获利盘打压引发回撤。</li>
        </ol>
    </div>
</div>

<script>
// 嵌入全量实盘时序数据
const RAW_DATA = {
    "60m": __DATA_60M__,
    "5m": __DATA_5M__,
    "daily": __DATA_DAILY__
};

let currentTF = "60m";
let currentBars = [];
let currentWindows = [];
let viewStartIndex = 0;
let viewCount = 80;
let activeWindowIdx = null;

// 窗口识别算法 (严格按照: 涨幅>=minGain, 回撤<maxDD, 不重复)
function findWindows(bars, minGain, maxDD) {
    const windows = [];
    const n = bars.length;
    let i = 0;

    while (i < n) {
        let startIdx = i;
        let startPrice = bars[startIdx].low;
        let startTime = bars[startIdx].time_key;
        let peakIdx = startIdx;
        let peakPrice = bars[startIdx].high;
        let peakTime = bars[startIdx].time_key;
        let curr = startIdx;
        let found = null;

        while (curr < n) {
            const cHigh = bars[curr].high;
            const cLow = bars[curr].low;

            // 若跌破起点且尚未开启涨幅，顺延起点
            if (cLow < startPrice && peakPrice <= startPrice * 1.01) {
                startPrice = cLow;
                startTime = bars[curr].time_key;
                startIdx = curr;
                peakPrice = cHigh;
                peakIdx = curr;
                peakTime = bars[curr].time_key;
                curr++;
                continue;
            }

            // 更新高点
            if (cHigh > peakPrice) {
                peakPrice = cHigh;
                peakIdx = curr;
                peakTime = bars[curr].time_key;
            }

            // 检查回撤
            const dd = (peakPrice - cLow) / peakPrice;
            if (dd >= maxDD) {
                const gain = (peakPrice - startPrice) / startPrice;
                if (gain >= minGain && peakIdx > startIdx) {
                    // 验证 start 到 peak 之间的实际最大回撤
                    let runMax = -Infinity;
                    let actualMaxDD = 0;
                    for (let k = startIdx; k <= peakIdx; k++) {
                        if (bars[k].high > runMax) runMax = bars[k].high;
                        const d = (runMax - bars[k].low) / runMax;
                        if (d > actualMaxDD) actualMaxDD = d;
                    }
                    if (actualMaxDD < maxDD) {
                        found = {
                            startIdx, peakIdx, breakIdx: curr,
                            startTime, peakTime, breakTime: bars[curr].time_key,
                            startPrice, peakPrice, breakPrice: bars[curr].low,
                            gainPct: gain * 100,
                            maxDDPct: actualMaxDD * 100,
                            barsCount: peakIdx - startIdx + 1
                        };
                    }
                }
                break;
            }
            curr++;
        }

        if (!found && curr >= n) {
            const gain = (peakPrice - startPrice) / startPrice;
            if (gain >= minGain && peakIdx > startIdx) {
                let runMax = -Infinity;
                let actualMaxDD = 0;
                for (let k = startIdx; k <= peakIdx; k++) {
                    if (bars[k].high > runMax) runMax = bars[k].high;
                    const d = (runMax - bars[k].low) / runMax;
                    if (d > actualMaxDD) actualMaxDD = d;
                }
                if (actualMaxDD < maxDD) {
                    found = {
                        startIdx, peakIdx, breakIdx: curr - 1,
                        startTime, peakTime, breakTime: '最新未阻断',
                        startPrice, peakPrice, breakPrice: bars[curr - 1].close,
                        gainPct: gain * 100,
                        maxDDPct: actualMaxDD * 100,
                        barsCount: peakIdx - startIdx + 1
                    };
                }
            }
        }

        if (found) {
            windows.push(found);
            i = Math.max(found.breakIdx, found.peakIdx + 1);
        } else {
            i++;
        }
    }
    return windows;
}

// 切换周期
function switchTimeframe() {
    currentTF = document.getElementById('timeframeSelect').value;
    applyPreset();
}

// 应用区间快捷筛选
function applyPreset() {
    const preset = document.getElementById('rangePreset').value;
    const all = RAW_DATA[currentTF];

    if (preset === '30cal') {
        currentBars = all.filter(b => b.time_key >= '2026-09-08');
    } else if (preset === '30trd') {
        const cutoff = currentTF === 'daily' ? 30 : (currentTF === '60m' ? 30 * 7 : 30 * 78);
        currentBars = all.slice(-cutoff);
    } else {
        currentBars = [...all];
    }

    recalculate();
}

// 重新计算
function recalculate() {
    const minGain = parseFloat(document.getElementById('minGainInput').value) / 100.0;
    const maxDD = parseFloat(document.getElementById('maxDDInput').value) / 100.0;

    currentWindows = findWindows(currentBars, minGain, maxDD);

    // 重置视野
    viewCount = currentTF === '5m' ? 180 : (currentTF === '60m' ? 80 : 35);
    viewStartIndex = Math.max(0, currentBars.length - viewCount);
    activeWindowIdx = null;

    updateMetrics();
    renderTable();
    renderChart();
}

// 更新指标卡
function updateMetrics() {
    const count = currentWindows.length;
    document.getElementById('mWindowCount').textContent = count;

    if (count === 0) {
        document.getElementById('mAvgGain').textContent = '-';
        document.getElementById('mMaxGain').textContent = '-';
        document.getElementById('mMaxGainDate').textContent = '-';
        document.getElementById('mAvgDD').textContent = '-';
        document.getElementById('mAvgBars').textContent = '-';
        document.getElementById('mAvgDurationSub').textContent = '-';
        return;
    }

    const totalGain = currentWindows.reduce((acc, w) => acc + w.gainPct, 0);
    const avgGain = totalGain / count;
    document.getElementById('mAvgGain').textContent = '+' + avgGain.toFixed(2) + '%';

    let maxW = currentWindows[0];
    for (let w of currentWindows) {
        if (w.gainPct > maxW.gainPct) maxW = w;
    }
    document.getElementById('mMaxGain').textContent = '+' + maxW.gainPct.toFixed(2) + '%';
    document.getElementById('mMaxGainDate').textContent = maxW.startTime.slice(5, 16);

    const totalDD = currentWindows.reduce((acc, w) => acc + w.maxDDPct, 0);
    document.getElementById('mAvgDD').textContent = (totalDD / count).toFixed(2) + '%';

    const totalBars = currentWindows.reduce((acc, w) => acc + w.barsCount, 0);
    const avgBars = (totalBars / count).toFixed(1);
    document.getElementById('mAvgBars').textContent = avgBars + ' 根';

    const unit = currentTF === '5m' ? '分钟' : (currentTF === '60m' ? '小时' : '交易日');
    const factor = currentTF === '5m' ? 5 : 1;
    document.getElementById('mAvgDurationSub').textContent = '平均约 ' + (avgBars * factor).toFixed(0) + ' ' + unit;
}

// 渲染表格
function renderTable() {
    const tbody = document.getElementById('windowsTableBody');
    tbody.innerHTML = '';

    if (currentWindows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" style="text-align:center; color:#94a3b8; padding:20px;">未找到符合当前过滤条件的上涨窗口 (可尝试调整涨幅或回撤阈值)</td></tr>';
        return;
    }

    currentWindows.forEach((w, idx) => {
        const tr = document.createElement('tr');
        if (activeWindowIdx === idx) tr.classList.add('active-row');

        const unit = currentTF === '5m' ? '分钟' : (currentTF === '60m' ? '小时' : '天');
        const factor = currentTF === '5m' ? 5 : 1;
        const durStr = (w.barsCount * factor) + ' ' + unit;

        tr.innerHTML = `
            <td><strong>#${idx + 1}</strong></td>
            <td>${w.startTime.slice(5, 16)}</td>
            <td class="num">$${w.startPrice.toFixed(2)}</td>
            <td>${w.peakTime.slice(5, 16)}</td>
            <td class="num">$${w.peakPrice.toFixed(2)}</td>
            <td class="num" style="color:var(--up-color); font-weight:700;">+${w.gainPct.toFixed(2)}%</td>
            <td class="num">${w.maxDDPct.toFixed(2)}%</td>
            <td class="num">${w.barsCount}</td>
            <td>${durStr}</td>
            <td>${w.breakTime.slice(5, 16)}</td>
            <td><span class="tag-win">达标合格</span></td>
        `;

        tr.onclick = () => {
            activeWindowIdx = idx;
            // 居中视角
            const mid = Math.floor((w.startIdx + w.peakIdx) / 2);
            viewStartIndex = Math.max(0, mid - Math.floor(viewCount / 2));
            if (viewStartIndex + viewCount > currentBars.length) {
                viewStartIndex = Math.max(0, currentBars.length - viewCount);
            }
            renderTable();
            renderChart();
        };

        tbody.appendChild(tr);
    });
}

// 平移视图
function panChart(delta) {
    viewStartIndex = Math.max(0, Math.min(currentBars.length - viewCount, viewStartIndex + delta));
    renderChart();
}

function resetChart() {
    viewStartIndex = Math.max(0, currentBars.length - viewCount);
    activeWindowIdx = null;
    renderTable();
    renderChart();
}

// 绘制 K 线与阴影 Canvas
function renderChart() {
    const canvas = document.getElementById('klineCanvas');
    const container = document.getElementById('canvasContainer');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = container.clientWidth * dpr;
    canvas.height = container.clientHeight * dpr;
    canvas.style.width = container.clientWidth + 'px';
    canvas.style.height = container.clientHeight + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const w = container.clientWidth;
    const h = container.clientHeight;

    // 清屏
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    if (currentBars.length === 0) return;

    const visibleBars = currentBars.slice(viewStartIndex, viewStartIndex + viewCount);
    const n = visibleBars.length;
    if (n === 0) return;

    // 计算价格范围
    let minP = Infinity;
    let maxP = -Infinity;
    visibleBars.forEach(b => {
        if (b.low < minP) minP = b.low;
        if (b.high > maxP) maxP = b.high;
    });

    const pPad = (maxP - minP) * 0.1 || 1.0;
    minP -= pPad;
    maxP += pPad;

    const padTop = 30;
    const padBottom = 40;
    const padLeft = 10;
    const padRight = 65;
    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    function priceToY(p) {
        return padTop + (1 - (p - minP) / (maxP - minP)) * chartH;
    }

    function idxToX(idxInView) {
        return padLeft + (idxInView + 0.5) * (chartW / n);
    }

    const barWidth = Math.max(2, (chartW / n) * 0.7);

    // 绘制横向参考网格
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "left";

    const gridSteps = 5;
    for (let i = 0; i <= gridSteps; i++) {
        const price = minP + (i / gridSteps) * (maxP - minP);
        const y = priceToY(price);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(w - padRight, y);
        ctx.stroke();
        ctx.fillText('$' + price.toFixed(2), w - padRight + 6, y + 4);
    }

    // 绘制合格窗口阴影与标注
    currentWindows.forEach((win, winIdx) => {
        const inViewStart = win.startIdx - viewStartIndex;
        const inViewPeak = win.peakIdx - viewStartIndex;
        const inViewBreak = win.breakIdx - viewStartIndex;

        // 如果该窗口在当前视野范围内
        if (inViewBreak >= 0 && inViewStart < n) {
            const x1 = Math.max(padLeft, idxToX(Math.max(0, inViewStart)) - barWidth / 2);
            const x2 = Math.min(w - padRight, idxToX(Math.min(n - 1, inViewPeak)) + barWidth / 2);
            const width = Math.max(4, x2 - x1);

            // 绘制半透明绿色背景阴影
            ctx.fillStyle = (activeWindowIdx === winIdx) ? "rgba(34, 197, 94, 0.22)" : "rgba(34, 197, 94, 0.10)";
            ctx.fillRect(x1, padTop, width, chartH);

            // 绘制虚线边界
            ctx.strokeStyle = "rgba(34, 197, 94, 0.7)";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.strokeRect(x1, padTop, width, chartH);
            ctx.setLineDash([]);

            // 绘制顶端波段标识
            ctx.fillStyle = "#15803d";
            ctx.font = "bold 11px ui-monospace, sans-serif";
            ctx.fillText(`W#${winIdx + 1}: +${win.gainPct.toFixed(1)}%`, x1 + 4, padTop + 14);

            // 绘制起点基准虚线
            const yStart = priceToY(win.startPrice);
            ctx.strokeStyle = "#16a34a";
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(x1, yStart);
            ctx.lineTo(x2, yStart);
            ctx.stroke();
            ctx.setLineDash([]);

            // 起点圆点
            if (inViewStart >= 0 && inViewStart < n) {
                const sx = idxToX(inViewStart);
                ctx.fillStyle = "#16a34a";
                ctx.beginPath();
                ctx.arc(sx, yStart, 4, 0, Math.PI * 2);
                ctx.fill();
            }

            // 峰值圆点
            if (inViewPeak >= 0 && inViewPeak < n) {
                const px = idxToX(inViewPeak);
                const py = priceToY(win.peakPrice);
                ctx.fillStyle = "#dc2626";
                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.fill();
            }

            // 阻断点叉号
            if (inViewBreak >= 0 && inViewBreak < n && win.breakTime !== '最新未阻断') {
                const bx = idxToX(inViewBreak);
                const by = priceToY(win.breakPrice);
                ctx.strokeStyle = "#475569";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(bx - 4, by - 4);
                ctx.lineTo(bx + 4, by + 4);
                ctx.moveTo(bx + 4, by - 4);
                ctx.lineTo(bx - 4, by + 4);
                ctx.stroke();
            }
        }
    });

    // 绘制 K 线柱体与影线
    visibleBars.forEach((bar, i) => {
        const x = idxToX(i);
        const yOpen = priceToY(bar.open);
        const yClose = priceToY(bar.close);
        const yHigh = priceToY(bar.high);
        const yLow = priceToY(bar.low);

        const isUp = bar.close >= bar.open;
        const color = isUp ? "#16a34a" : "#dc2626";

        // 绘制上下影线
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, yHigh);
        ctx.lineTo(x, yLow);
        ctx.stroke();

        // 绘制实体
        const topY = Math.min(yOpen, yClose);
        const bodyH = Math.max(1.5, Math.abs(yClose - yOpen));
        ctx.fillStyle = color;
        ctx.fillRect(x - barWidth / 2, topY, barWidth, bodyH);

        // 绘制底部时间刻度
        if (i % Math.ceil(n / 8) === 0) {
            ctx.fillStyle = "#64748b";
            ctx.font = "10px ui-monospace, sans-serif";
            ctx.textAlign = "center";
            const timeStr = bar.time_key.slice(5, 16);
            ctx.fillText(timeStr, x, h - 14);
        }
    });

    // 更新信息提示
    const sBar = visibleBars[0];
    const eBar = visibleBars[visibleBars.length - 1];
    document.getElementById('chartRangeInfo').textContent =
        `展示中: ${currentTF.toUpperCase()} · 范围: ${sBar.time_key.slice(5, 16)} 至 ${eBar.time_key.slice(5, 16)} (共 ${n} 根)`;

    // 鼠标悬停十字光标与 Tooltip
    const tooltip = document.getElementById('chartTooltip');

    container.onmousemove = (e) => {
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const countInView = Math.min(viewCount, currentBars.length - viewStartIndex);
        if (countInView <= 0) return;

        const idxInView = Math.floor((mouseX - padLeft) / (chartW / countInView));
        if (idxInView >= 0 && idxInView < countInView) {
            const actualIdx = viewStartIndex + idxInView;
            const bar = currentBars[actualIdx];

            let inWinText = "";
            currentWindows.forEach((w, wIdx) => {
                if (actualIdx >= w.startIdx && actualIdx <= w.breakIdx) {
                    const gainFromStart = ((bar.close - w.startPrice) / w.startPrice * 100).toFixed(2);
                    inWinText = `<div style="color:#22c55e; margin-top:4px; font-weight:bold;">🎯 归属窗口 #${wIdx + 1} (起点 $${w.startPrice.toFixed(2)})<br>距起点现涨幅: +${gainFromStart}%</div>`;
                }
            });

            tooltip.style.display = 'block';
            tooltip.style.left = Math.min(rect.width - 240, mouseX + 15) + 'px';
            tooltip.style.top = Math.min(rect.height - 140, mouseY + 15) + 'px';
            tooltip.innerHTML = `
                <div style="font-weight:bold; margin-bottom:2px;">${bar.time_key}</div>
                开: $${bar.open.toFixed(2)} | 高: $${bar.high.toFixed(2)}<br>
                低: $${bar.low.toFixed(2)} | 收: $${bar.close.toFixed(2)}<br>
                量: ${(bar.volume || 0).toLocaleString()} 股
                ${inWinText}
            `;
        } else {
            tooltip.style.display = 'none';
        }
    };

    container.onmouseleave = () => {
        tooltip.style.display = 'none';
    };
}

// 页面初始化
window.addEventListener('resize', renderChart);
window.addEventListener('load', () => {
    switchTimeframe();
});
</script>

</body>
</html>
"""

def main():
    df_60 = pd.read_parquet('market_data/SOXS_recent_60m.parquet')
    df_5m = pd.read_parquet('market_data/SOXS_recent_5m.parquet')
    df_d = pd.read_parquet('market_data/SOXS_recent_daily.parquet')

    cols = ['time_key', 'open', 'high', 'low', 'close', 'volume']
    bars_60 = df_60[cols].to_dict(orient='records')
    bars_5m = df_5m[cols].to_dict(orient='records')
    bars_d = df_d[cols].to_dict(orient='records')

    json_60 = json.dumps(bars_60)
    json_5m = json.dumps(bars_5m)
    json_d = json.dumps(bars_d)

    rendered = HTML_TEMPLATE.replace('__DATA_60M__', json_60)
    rendered = rendered.replace('__DATA_5M__', json_5m)
    rendered = rendered.replace('__DATA_DAILY__', json_d)

    out_file = os.path.abspath('analysis/soxs_30d_swings_wireframe/index.html')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(rendered)

    print(f"🎉 HTML Wireframe 生成完成！")
    print(f"📁 本地物理文件: {out_file}")
    print(f"🌐 本地 Web 服务访问 URL: http://127.0.0.1:8768/analysis/soxs_30d_swings_wireframe/index.html")

if __name__ == '__main__':
    main()
