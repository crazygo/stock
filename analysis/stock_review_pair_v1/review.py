"""Measured before/after review of input identity, results and storage."""
import html
from pathlib import Path

from common import fingerprint, read, write
from storage import price_bars

TRAIT_FIELDS = ['v', 'status', 'uncertainty', 'history', 'mapped', 'bars',
                'evaluation_cutoff', 'zero_volume_warning']


def compare_run(baseline, run, details, recompute):
    old = read(baseline / 'inputs/data_manifest.json')
    new = read(run / 'inputs/data_manifest.json')
    old_inputs = {d['security_id']: d for d in old['datasets']}
    old_traits = {r['code']: r for r in read(baseline / 'results/trend/results.json')['records']}
    new_traits = {r['code']: r for r in read(run / 'results/trend/results.json')['records']}
    old_behavior = {r['code']: r for r in read(baseline / 'results/behavior.json')['records']}
    new_behavior = {r['code']: r for r in read(run / 'results/behavior.json')['records']}
    rows = []
    for entry in new['datasets']:
        code = entry['security_id']
        old_entry = old_inputs.get(code)
        left, right = old_traits.get(code, {}), new_traits.get(code, {})
        issues = []
        if old_entry is None or fingerprint(price_bars(baseline, old_entry)) != fingerprint(price_bars(run, entry)):
            issues.append('OHLCV')
        if old_entry is None or any(old_entry.get(k) != entry.get(k) for k in ['dataset_id', 'sha256', 'coverage']):
            issues.append('输入版本/覆盖')
        if any(left.get(k) != right.get(k) for k in TRAIT_FIELDS):
            issues.append('五标签/历史/不确定性')
        if old_behavior.get(code) != new_behavior.get(code):
            issues.append('历史形态/峰谷证据')
        deltas = [abs(a-b) for a, b in zip(left.get('v', []), right.get('v', [])) if a is not None and b is not None]
        behavior = new_behavior.get(code) or {}
        shares, folds = behavior.get('shares') or {}, behavior.get('folds') or {}
        rows.append({'code': code, 'name': right.get('name', code), 'special': code in new_behavior,
                     'issues': issues, 'old_v': left.get('v'), 'new_v': right.get('v'),
                     'states': right.get('status'), 'max_delta': max(deltas, default=0),
                     'last_day': entry['coverage']['actual_end'], 'coverage': entry['coverage']['status'],
                     'up': shares.get('up'), 'windows': shares.get('n'),
                     'frequency': folds.get('frequency'), 'amplitude': folds.get('amplitudeMedian')})
    old_count = len(old_traits)
    changed = sum(bool(r['issues']) for r in rows)
    logical_before = sum(p.stat().st_size for p in (baseline / 'inputs').rglob('*') if p.is_file())
    physical_after = sum(p.stat().st_size for p in (run / 'inputs').rglob('*') if p.is_file() and not p.is_symlink())
    report = {'schema_version': 'shared_data_recompute_review_v1', 'run_id': new['run_id'],
              'baseline_run_id': old['run_id'], 'data_run_id': new['data_run_id'],
              'mode': 'frozen_input_fresh_recompute' if recompute else 'frozen_input_replay',
              'market_cutoffs': new['market_cutoffs'], 'securities': len(rows),
              'behavior_securities': len(new_behavior), 'unchanged': len(rows)-changed, 'changed': changed,
              'same_members': old_count == len(new_traits) and set(old_traits) == set(new_traits),
              'same_data_run_id': old['data_run_id'] == new['data_run_id'],
              'same_behavior_members': set(old_behavior) == set(new_behavior),
              'max_absolute_trait_delta': max((r['max_delta'] for r in rows), default=0),
              'calculations': {k: details[k] for k in ['trend_fresh_calculations', 'behavior_fresh_calculations',
                                                      'calculation_cache_reused', 'evaluation_reused']},
              'storage': {**details['shared_storage'], 'run_parquet_files': details['run_parquet_files'],
                          'baseline_logical_input_bytes': logical_before,
                          'new_run_physical_input_bytes': physical_after},
              'rows': sorted(rows, key=lambda r: (not r['special'], r['code']))}
    passed = changed == 0 and all(report[k] for k in ['same_members', 'same_data_run_id', 'same_behavior_members'])
    report['status'] = 'passed' if passed else 'failed'
    write(run / 'validation/recompute_review.json', report)
    render_review(run, report)
    if not passed:
        raise ValueError('storage refactor changed frozen input or calculation results; see recompute_review.json')
    return report


def render_review(run, report):
    data = __import__('json').dumps(report, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    h = html.escape
    mi = lambda n: f'{n / 1024**2:.2f} MiB'
    storage = report['storage']
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数据共用改造 · 纯计算核对</title><style>
:root{--ink:#222;--line:#aaa;--wash:#eee}*{box-sizing:border-box}body{margin:0;color:var(--ink);font:14px/1.6 system-ui;background:#fff}
main{max-width:1440px;margin:auto;padding:20px}h1{font-size:26px;margin:0 0 8px}h2{font-size:18px;margin:20px 0 8px}p{margin:6px 0}
a{color:inherit}code{overflow-wrap:anywhere}button,input{font:inherit;max-width:100%;border:1px solid #888;background:white;padding:5px}
button{cursor:pointer}:focus-visible{outline:3px solid #555}.tools{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:14px 0}
.summary{border:1px solid var(--line);padding:12px;background:var(--wash)}.wrap{max-width:100%;overflow:auto;border:1px solid var(--line)}
table{border-collapse:collapse;min-width:1050px;width:100%;font-size:12px}th,td{border-bottom:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}th{background:var(--wash);position:sticky;top:0}
.values{font-variant-numeric:tabular-nums;white-space:nowrap}.difference{font-weight:bold}.note{font-size:12px;color:#555}.meta{overflow-wrap:anywhere}
th:last-child,td:last-child{min-width:100px;white-space:nowrap}td:first-child{min-width:190px;max-width:230px}td:nth-child(4){min-width:160px}td:nth-child(5){min-width:85px}td:nth-child(6){min-width:120px}
@media(max-width:600px){main{padding:12px}h1{font-size:22px}.tools{gap:8px}.tools input[type=search]{width:100%}}
</style><main><h1>数据共用改造 · 纯计算核对</h1>'''
    page += f'<div class="summary"><b>{report["unchanged"]}/{report["securities"]} 个证券的输入与计算结果一致；差异 {report["changed"]} 个。</b>'
    page += f'<p>最大五标签绝对差值：{report["max_absolute_trait_delta"]}。特别关注 {report["behavior_securities"]} 个，连同完整窗口、峰谷证据逐项核对。</p></div>'
    page += f'<p class="meta">基线 <code>{h(report["baseline_run_id"])}</code><br>本次 <code>{h(report["run_id"])}</code><br>数据 <code>{h(report["data_run_id"])}</code></p>'
    page += f'<p>行情截止：{h(str(report["market_cutoffs"]))}。使用冻结输入，未调用采集链路；历史点仍为回算。</p>'
    calc = report['calculations']
    page += f'<p>五标签新算 {calc["trend_fresh_calculations"]} 个，缓存取值 {calc["calculation_cache_reused"]} 个；历史形态新算 {calc["behavior_fresh_calculations"]} 个。既有评估记录复用：{h(str(calc["evaluation_reused"]))}。</p>'
    page += '<p><a href="../pages/stock_behavior_description_v3/index.html">打开本次历史形态比较墙（日周图）</a> · <a href="../pages/ai_trend_quadrant_v2/index.html">打开本次趋势象限与多股票图</a></p>'
    page += '<h2>实际存储变化</h2>'
    page += f'<p>基线输入逻辑大小 {mi(storage["baseline_logical_input_bytes"])} → 新 run 独占输入 {mi(storage["new_run_physical_input_bytes"])}；run 内 Parquet {storage["run_parquet_files"]} 个。</p>'
    page += f'<p>共用库本次新增对象 {storage["objects_created"]} 个、{mi(storage["bytes_created"])}，复用对象 {storage["objects_reused"]} 个。首次迁移会建立共用版本，后续同输入重跑不再复制；旧 run 保留。</p>'
    page += '<h2>逐股核对</h2><p class="note">五标签顺序 G / V / M / J / S，显示原始值；— 为未知。G、V 为年化量，M 为相关系数，J、S 为比例。此页核对改造有没有改变计算；形态语义请对照上面的 K 线页面。</p>'
    page += '<div class="tools"><input id="search" type="search" aria-label="搜索证券" placeholder="代码 / 名称"><label><input id="special" type="checkbox">仅特别关注</label><label><input id="changed" type="checkbox">仅差异</label><button id="reset">清除筛选</button><span id="count" aria-live="polite"></span></div>'
    page += '<div class="wrap"><table><thead><tr><th>证券 / 截止 / 覆盖</th><th>旧五标签</th><th>新五标签</th><th>最大差值 / 状态</th><th>偏上行</th><th>折返频率 / 幅度</th><th>核对</th></tr></thead><tbody id="rows"></tbody></table></div></main>'
    page += '<script id="data" type="application/json">' + data + '</script>'
    page += '''<script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const val=x=>x==null?'—':Number(x).toFixed(6),pct=x=>x==null?'—':(x*100).toFixed(1)+'%';
function render(){const q=$('search').value.trim().toLowerCase();const rows=D.rows.filter(r=>(!q||(r.code+' '+r.name).toLowerCase().includes(q))&&(!$('special').checked||r.special)&&(!$('changed').checked||r.issues.length));
$('count').textContent=rows.length+' / '+D.rows.length+' 个';$('rows').innerHTML=rows.map(r=>`<tr><td><b>${esc(r.code)}</b>${r.special?' · 特别关注':''}<br>${esc(r.name)}<br>${esc(r.last_day)} · ${esc(r.coverage)}</td><td class="values">${r.old_v.map(val).join(' / ')}</td><td class="values">${r.new_v.map(val).join(' / ')}</td><td>${r.max_delta}<br>${esc(r.states.join(' / '))}</td><td>${pct(r.up)}<br>${r.windows??'—'} 个窗口</td><td>${val(r.frequency)} 次/20日<br>${pct(r.amplitude)}</td><td class="difference">${r.issues.length?esc(r.issues.join('；')):'一致'}${r.special?'<br><a href="../pages/stock_behavior_description_v3/detail.html?code='+encodeURIComponent(r.code)+'">日线证据</a>':''}</td></tr>`).join('')||'<tr><td colspan="7">没有匹配项。</td></tr>';}
for(const id of ['search','special','changed'])$(id).addEventListener('input',render);
$('reset').onclick=()=>{$('search').value='';$('special').checked=$('changed').checked=false;render()};render();
</script></html>'''
    (run / 'validation/recompute_review.html').write_text(page)
