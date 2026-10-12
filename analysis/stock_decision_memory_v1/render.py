#!/usr/bin/env python3
"""Publish self-contained stock dossier HTML from frozen local research only."""

from __future__ import annotations

import copy
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from initialize import REPO, encoded, sha, verify_snapshot


HERE = Path(__file__).resolve().parent
STORE = REPO / ".cache/stock_decision_memory_v1"
ANALOG = REPO / "analysis/premarket_self_analogs_v1"


def esc(value):
    return html.escape(str(value if value is not None else ""), quote=True)


def number(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def percent(value):
    return number(value * 100) + "%" if isinstance(value, (int, float)) else "—"


def json_script(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).replace("<", "\\u003c")


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Reference template changed near: {before[:80]}")
    return text.replace(before, after, 1)


EXTRA_CSS = """
main,section,.box,.grid,.grid>*{min-width:0}p{max-width:76ch}a{color:var(--ink)}
.grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.reader-head{display:grid;gap:10px}
.stock-nav{border-bottom:1px solid var(--line);padding-bottom:14px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.stock-nav label{display:flex;gap:8px;align-items:center;min-width:0}.stock-nav select{max-width:420px;min-width:0}
.memory-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.memory-metrics strong{font-size:24px;display:block}.memory-metrics small{display:block}.fact-range{font-size:13px;color:var(--muted)}
.plain-table{white-space:normal}.plain-table th,.plain-table td{text-align:left}.plain-table td{vertical-align:top}
.status-label{border:1px solid var(--line);padding:2px 7px;display:inline-block;font-size:12px}
.reading-list{display:grid;gap:8px;margin:0;padding-left:20px}.stage-note{background:var(--fill);padding:12px}
details>p{margin-top:8px}.controls input,.controls select{max-width:100%}.stock-directory th,.stock-directory td{white-space:nowrap}
@media(max-width:700px){.grid,.memory-metrics{grid-template-columns:1fr}.stock-nav label{display:grid;width:100%;gap:4px}.stock-nav select{max-width:100%;width:100%}.controls label{min-width:0}.controls label select{display:block;width:100%}.memory-metrics strong{font-size:22px}}
@media print{.stock-nav,.controls,.directory-tools{display:none}main{max-width:none;padding:0}.table-scroll{overflow:visible;max-height:none}details{break-inside:avoid}section{break-before:auto}.chart{height:260px}}
"""


def navigation(code, codes, names):
    index = codes.index(code)
    options = "".join(f'<option value="{esc(c)}"{" selected" if c == code else ""}>{esc(c)} · {esc(names[c])}</option>' for c in codes)
    return f'''<div class="stock-nav"><a href="../index.html">全部档案</a>
<label for="dossierStock">股票档案 <select id="dossierStock" aria-label="股票档案">{options}</select></label>
<a href="{esc(codes[(index-1) % len(codes)])}.html">上一只</a><a href="{esc(codes[(index+1) % len(codes)])}.html">下一只</a></div>'''


def trait_section(profile):
    if profile is None:
        return '''<section id="traits"><h2>日线历史特征</h2><p>本轮尚未导入这只证券的日线特征。偏上行、折返频率和幅度留空。</p></section>'''
    o = profile["observation"]
    up, turns = o["upward_windows"], o["reversals"]
    start, end = o["observed_range"]
    settings = o["settings"]
    return f'''<section id="traits"><h2>日线历史特征</h2>
<p class="fact-range">实际观察 {esc(start)} 至 {esc(end)} · {o['coverage']['observed_daily_bars']} 个日级收盘 · 历史描述</p>
<div class="memory-metrics">
<div class="box">偏上行窗口占比<strong>{percent(up['fraction'])}</strong><small>{up['qualifying']} / {up['complete_windows']} 个完整窗口；窗口重叠</small></div>
<div class="box">折返频率<strong>{number(turns['frequency'])}</strong><small>次 / 20 次交易日变化；{turns['confirmations']} 次反向确认</small></div>
<div class="box">完整峰谷幅度中位数<strong>{percent(turns['amplitude_median'])}</strong><small>{turns['complete_segments']} 段；上涨和下跌段共同统计</small></div></div>
<p>这段历史里，{up['complete_windows']} 个完整窗口中有 {up['qualifying']} 个满足偏上行条件。折返与幅度用于描述已经发生的路径；它们没有给出继续持有的收益或卖出门槛。</p>
<details><summary>计算口径与覆盖</summary><p>偏上行：{settings['window']} 次交易日变化，净上涨且方向效率至少 {number(settings['efficiency'])}。折返：反向收盘变化达到 {percent(settings['threshold'])} 才确认。幅度：完整波段高收盘 / 低收盘 − 1。</p>
<p>请求范围 {esc(o['requested_range'][0])} 至 {esc(o['requested_range'][1])}；实际范围如上。缺失日期 {esc('、'.join(map(str, o['coverage']['missing'])) or '本范围未登记缺日')}；日历状态 {('已知' if o['coverage']['calendar_known'] else '未知')}。参数固定，下方小时日期与门槛控件不会改变这三个日线数值。</p></details></section>'''


def decision_section(code):
    extra = ""
    if code == "US.SOXS":
        extra = '''<div class="box"><h3>SOXS 已保存的退出研究问题</h3><p>首次净盈利达到 5% 后，部分兑现能否减少回吐？目前该条件结果尚未计算。</p>
<p>已有波段描述：2026-09-08 至 10-07，共 22 个交易日，涨幅至少 5%、算法测得回撤小于 4%；小时线识别 5 段，5 分钟线识别 12 段。它们从事后低点到高点统计，没有真实入场失败案例的分母。</p>
<p>其中四个小时高点落在 10:30 标签的开盘首小时柱里，较细 K 线标签分别为 10:00、09:45、10:05、09:50，不能写成精准在 10:30 见顶。</p>
<p>单日反向三倍机制需与本次目的一起核对：获利仓与组合对冲仓的退出问题分别研究。<a href="https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs">发行人产品说明</a></p></div>'''
    return f'''<section id="decisions"><h2>操作前参考</h2><p>先确定本次交易目的和原计划，再查对应证据。这里尚未接入你的实际成本、持仓计划或个人决策记录。</p>
<div class="table-scroll"><table class="plain-table"><thead><tr><th>准备做的决定</th><th>先核对什么</th><th>档案当前能回答什么</th></tr></thead><tbody>
<tr><td>买入或加仓</td><td>入场理由、持有期限、损失预算、行业与事件是否改变</td><td>可查已有历史路径与情境；没有获准的买入规则</td></tr>
<tr><td>盈利后继续等或兑现</td><td>是否到达原计划检查点；继续持有的增益与回吐</td><td>净盈利达到某水平后的退出比较尚未计算</td></tr>
<tr><td>亏损后退出或继续持有</td><td>原理由是否失效，是否把短线亏损改成长期持有</td><td>没有你的入场成本和计划，不能代填本次止损结论</td></tr>
</tbody></table></div>{extra}</section>'''


def segment_section(profile):
    if profile is None:
        return ""
    folds = profile["observation"]["reversals"]
    rows = []
    for x in folds["completed"]:
        rows.append(f'''<tr><td>{esc(x['start'])}</td><td>{esc(x['end'])}</td><td>{'上行' if x['direction'] > 0 else '下行'}</td><td>{number(x['from'])} → {number(x['to'])}</td><td>{percent(x['amplitude'])}</td><td>{esc(x['confirmed'])}</td><td>{x['days']}</td></tr>''')
    return f'''<section id="segments"><h2>完整峰谷证据</h2><p>极值日期与反向确认日期分别列出。幅度从实际极值计算，波段结束通常要等后续价格才能确认。</p>
<div class="table-scroll"><table><thead><tr><th>起点</th><th>极值终点</th><th>方向</th><th>复权收盘价端点</th><th>幅度</th><th>反向确认日</th><th>交易日变化</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="7">本范围没有可计入的完整波段</td></tr>'}</tbody></table></div>
<p class="muted">首段截断和未结束末段不计入幅度。这里列出的端点不是当时已经知道的入场或卖出时点。</p></section>'''


def history_section(profiles, analog):
    rows = []
    for p, manifest in profiles:
        o = p["observation"]
        rows.append(f'''<tr><td>日线路径描述</td><td>{esc(' 至 '.join(str(x) for x in o['observed_range']))}</td><td>历史描述</td><td>{o['upward_windows']['complete_windows']} 个完整滑动窗口；{o['reversals']['complete_segments']} 段完整峰谷</td></tr>''')
    if analog and analog.get("status") == "ready":
        rows.append('''<tr><td>小时曲线相似研究</td><td>2026-10-08 09:00 ET 截止</td><td>探索研究</td><td>匹配以前的同股曲线；未来概率和退出策略未验证</td></tr>''')
    elif analog:
        rows.append(f'''<tr><td>小时研究覆盖检查</td><td>2026-10-08 批次</td><td>不适用</td><td>{esc(analog.get('reason', analog.get('status')))}</td></tr>''')
    rows.append('''<tr><td>个人计划与决策</td><td>尚未登记</td><td>待积累</td><td>交易前写计划，准备操作时保存依据，结果成熟后复盘</td></tr>''')
    return f'''<section id="history"><h2>档案材料与更新</h2><div class="table-scroll"><table class="plain-table"><thead><tr><th>材料</th><th>时点</th><th>状态</th><th>范围</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>新增观察与反证按日期继续积累，旧快照保留。重复渲染不会增加历史样本，也不会取得新行情。</p></section>'''


def leaf_header(code, name, profiles, analog, codes, names):
    observations = "已有日线特征" if profiles else "日线特征未导入"
    observations += (" · 已有小时情境研究" if analog.get("status") == "ready" else " · 小时研究不适用") if analog else " · 本批小时研究未覆盖"
    segment_link = '<a href="#segments">峰谷证据</a>' if profiles else ''
    return f'''<header class="reader-head">{navigation(code, codes, names)}<h1>{esc(code)} · {esc(name)}</h1>
<p>每股决策档案 <span class="status-label">{observations}</span></p><p class="muted">操作前查历史特征、相似情境和原计划；每项材料保留自己的日期与口径。</p>
<nav><a href="#traits">历史特征</a><a href="#decisions">操作前参考</a><a href="#overview">小时情境</a>{segment_link}<a href="#history">档案材料</a></nav></header>'''


def leaf_document(code, name, profiles, analog, codes, names, reference):
    latest = profiles[-1][0] if profiles else None
    prefix = leaf_header(code, name, profiles, analog, codes, names)
    prefix += trait_section(latest) + decision_section(code)
    suffix = segment_section(latest) + history_section(profiles, analog)
    nav_script = "document.getElementById('dossierStock').addEventListener('change',e=>{location.href=encodeURIComponent(e.target.value)+'.html'});"
    if analog and analog.get("status") == "ready":
        doc = replace_once(reference, '<title>持仓 · 小时曲线历史相似窗口</title>', f'<title>{esc(code)} · 每股决策档案</title>')
        doc = replace_once(doc, '</style>', EXTRA_CSS + '</style>')
        begin, finish = doc.index('<header>'), doc.index('</header>') + len('</header>')
        doc = doc[:begin] + prefix + doc[finish:]
        doc = replace_once(doc, '<h2>逐股概览</h2>', '<h2>小时相似情境</h2><p id="cutoff" class="fact-range"></p>')
        doc = replace_once(doc, '<section id="detail"><div class="controls"><h2>逐股详情</h2>', '<section id="detail"><div class="controls" hidden style="display:none"><h2>逐股详情</h2>')
        doc = replace_once(doc, "let current='US.NRG'", f"let current={json.dumps(code)}")
        doc = replace_once(doc, '20只持仓（19美股、1港股） · ${state.start}', '${current} · ${state.start}')
        doc = doc.replace('盘前实时报价约09:13 ET · 0086账户', '盘前报价为该批09:13快照')
        doc = doc.replace('持仓${s.qty}股 ·', '该批持仓${s.qty}股 ·')
        doc = doc.replace('按当前09:00价格映射', '按该批09:00参考价映射')
        doc = doc.replace('<th>盘前最新价</th>', '<th>该批盘前价</th>')
        doc = replace_once(doc, "current=b.dataset.code;$('stock').value=current;chosenCase='';renderDetail()})}", "current=b.dataset.code;$('stock').value=current;chosenCase='';renderDetail();$('detail').scrollIntoView({block:'start'})})}")
        doc = replace_once(doc, 'const D=JSON.parse', nav_script + 'const D=JSON.parse')
        doc = replace_once(doc, '<footer>', suffix + '<footer>')
        doc = re.sub(r'<footer>.*?</footer>', '<footer>日线特征与小时情境各自保留原批次。静态页面不会取得新报价。<a href="../../premarket_self_analogs_v1/index.html#detail">原小时研究报告</a> · <a href="../index.html">全部档案</a></footer>', doc, count=1, flags=re.S)
        return replace_once(doc, '__DATA__', json_script(analog["_payload"]))
    reason = analog.get("reason", analog.get("status")) if analog else '此证券不在这份小时相似研究的持仓快照中。'
    body = prefix + f'''<section id="overview"><h2>小时相似情境</h2><p>{esc(reason)}</p><p class="muted">此处没有匹配数量或后续涨跌统计，不以零值替代未知。</p></section>''' + suffix
    style = re.search(r'<style>(.*?)</style>', reference, re.S).group(1) + EXTRA_CSS
    return f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(code)} · 每股决策档案</title><style>{style}</style></head><body><main>{body}<footer>静态档案；个人决策记录尚未接入。<a href="../index.html">返回全部档案</a></footer></main><script>{nav_script}</script></body></html>'''


def directory(codes, names, profiles, analogs, style):
    rows = []
    for code in codes:
        p = profiles.get(code, [])
        o = p[-1][0]["observation"] if p else None
        a = analogs.get(code)
        kind = ('特别关注 / 该批持仓' if p and a else '特别关注' if p else '该批持仓')
        counts = f"{a.get('matched_n', '—')} 个匹配" if a and a.get('status') == 'ready' else '不适用' if a else '未覆盖'
        scope = ('both favorite holding' if p and a else 'favorite' if p else 'holding')
        rows.append(f'''<tr data-search="{esc((code+' '+names[code]).lower())}" data-scope="{scope}"><td><a href="pages/{esc(code)}.html">{esc(code)}</a></td><td class="wrap">{esc(names[code])}</td><td>{kind}</td><td>{percent(o['upward_windows']['fraction']) if o else '未导入'}</td><td>{number(o['reversals']['frequency']) if o else '—'}</td><td>{percent(o['reversals']['amplitude_median']) if o else '—'}</td><td>{counts}</td></tr>''')
    return f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>每股决策档案</title><style>{style}{EXTRA_CSS}</style></head><body><main>
<header><h1>每股决策档案</h1><p>每只证券一个独立 HTML。先看历史特征与相似情境，再核对这次交易的目的和原计划。</p>
<p class="muted">55 只特别关注底稿与参考报告的 20 只持仓合并为 {len(codes)} 份档案。日线特征实际截至美股 10-07／港股 10-08；小时相似研究是 2026-10-08 09:00 ET 固定批次，持仓与报价均为该批快照。</p>
<nav><a href="pages/US.SOXS.html">先读 SOXS</a><a href="pages/US.NRG.html">查看 NRG 曲线与案例</a></nav></header>
<section><h2>股票目录</h2><div class="controls directory-tools"><label for="search">查找股票 <input id="search" type="search" placeholder="代码或名称"></label><label for="scope">范围 <select id="scope"><option value="all">全部档案</option><option value="favorite">特别关注底稿</option><option value="holding">参考报告持仓</option><option value="both">两者都有</option></select></label><span id="count" aria-live="polite">{len(codes)} 份档案</span></div>
<div class="table-scroll"><table class="stock-directory"><thead><tr><th>股票</th><th>名称</th><th>来源分组</th><th>偏上行</th><th>折返次 / 20日变化</th><th>完整波段幅度中位</th><th>小时相似研究</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p id="noResults" hidden>没有匹配的股票，清空搜索或切换范围。</p>
<p class="muted">日线三个指标为固定历史描述，不是未来胜率；小时匹配使用另一份独立日期与协议，未导入和未覆盖分别保留。</p></section>
<section><h2>每份档案的阅读顺序</h2><ol class="reading-list"><li>查看日线特征及样本数。</li><li>核对准备操作的问题与尚缺的证据。</li><li>查看小时相似曲线、同股基准和逐个历史案例。</li><li>检查峰谷确认依据与档案材料日期。</li></ol><p>净盈利达到 5% 后的退出比较和个人交易记录尚未接通；历史描述不会自行升级为买卖规则。</p></section>
<footer>页面内嵌阅读所需数据，可离线阅读。静态报告不会自行取新行情或更新持仓。</footer></main><script>
const search=document.getElementById('search'),scope=document.getElementById('scope');function filter(){{let n=0;for(const r of document.querySelectorAll('[data-search]')){{r.hidden=!(r.dataset.search.includes(search.value.trim().toLowerCase())&&(scope.value==='all'||r.dataset.scope.split(' ').includes(scope.value)));if(!r.hidden)n++}}document.getElementById('count').textContent=n+' 份档案';document.getElementById('noResults').hidden=n!==0}}search.addEventListener('input',filter);scope.addEventListener('change',filter);
</script></body></html>'''


def publish_link(name, target):
    path = HERE / name
    if path.exists() and not path.is_symlink():
        raise ValueError(f"Refuse to replace an unmanaged output: {path}")
    temporary = HERE / (".publish-" + name)
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(target, HERE), target_is_directory=target.is_dir())
    os.replace(temporary, path)


def main():
    index_raw = (STORE / "index.json").read_bytes()
    catalog = json.loads(index_raw)
    profiles = {}
    for code, refs in catalog["securities"].items():
        versions = []
        for ref in refs:
            path = STORE / ref["profile"]
            manifest = verify_snapshot(path.parent.parent)
            versions.append((json.loads(path.read_text()), manifest))
        versions.sort(key=lambda item: item[1]["source_information_cutoff"] or "")
        profiles[code] = versions
    source_manifest = json.loads((ANALOG / "delivery_manifest.json").read_text())
    raw = (ANALOG / "results.json").read_bytes()
    analog_result = json.loads(raw)
    reference_raw = (ANALOG / "report.template.html").read_bytes()
    reference = reference_raw.decode()
    displayed_raw = (ANALOG / "index.html").read_bytes()
    displayed = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', displayed_raw.decode(), re.S).group(1))
    projected = copy.deepcopy(analog_result)
    for stock in projected["stocks"]:
        for sample in stock.get("samples", []):
            sample.pop("feature_keys", None)
            sample.pop("future_24h_comparable", None)
            if sample.get("forward_3_sessions"):
                sample["forward_3_sessions"].pop("keys", None)
                sample["forward_3_sessions"].pop("missing_keys", None)
            if sample.get("future_rth"):
                for key in ("curve", "labels", "ohlc"):
                    sample["future_rth"].pop(key, None)
    if projected != displayed:
        raise ValueError("Analog source data and displayed report disagree; retry after source publication")
    reconstructed = reference.replace('__DATA__', json.dumps(projected, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    if reconstructed.encode() != displayed_raw:
        raise ValueError("Analog source template and displayed report disagree")
    analogs = {}
    for stock in analog_result["stocks"]:
        stock = copy.deepcopy(stock)
        for sample in stock.get("samples", []):
            sample.pop("feature_keys", None)
            sample.pop("future_24h_comparable", None)
            if sample.get("forward_3_sessions"):
                sample["forward_3_sessions"].pop("keys", None)
                sample["forward_3_sessions"].pop("missing_keys", None)
            if sample.get("future_rth"):
                for key in ("curve", "labels", "ohlc"):
                    sample["future_rth"].pop(key, None)
        payload = {k: v for k, v in analog_result.items() if k != "stocks"}
        payload["stocks"] = [copy.deepcopy(stock)]
        stock["_payload"] = payload
        analogs[stock["code"]] = stock
    codes = sorted(set(profiles) | set(analogs))
    names = {code: (profiles[code][-1][0]["observation"]["name"] if code in profiles else analogs[code]["name"]) for code in codes}
    output = {f"pages/{code}.html": leaf_document(code, names[code], profiles.get(code, []), analogs.get(code), codes, names, reference) for code in codes}
    style = re.search(r'<style>(.*?)</style>', reference, re.S).group(1)
    output["index.html"] = directory(codes, names, profiles, analogs, style)
    checksums = {path: sha(text.encode()) for path, text in output.items()}
    view_hash = sha(encoded(checksums))
    destination = STORE / "views" / ("view-" + view_hash[:24])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        for relative, digest in checksums.items():
            if sha((destination / relative).read_bytes()) != digest:
                raise ValueError(f"Published HTML modified: {relative}")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))
        try:
            for relative, text in output.items():
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            manifest = {"version": "stock_decision_memory_html_v1", "view_sha256": view_hash,
                        "securities": len(codes), "trait_dossiers": len(profiles),
                        "analog_dossiers": len(analogs), "analog_usable": sum(a.get("status") == "ready" for a in analogs.values()),
                        "sources": {"dossier_index": sha(index_raw), "analog_results": sha(raw),
                                    "reference_template": sha(reference_raw), "displayed_reference": sha(displayed_raw)},
                        "source_consistency": {"data_matches_display": True, "template_matches_display": True,
                                               "legacy_manifest_results_hash_matches": sha(raw) == source_manifest['files']['results.json'],
                                               "legacy_manifest_display_hash_matches": sha(displayed_raw) == source_manifest['files']['index.html']},
                        "files": checksums, "new_market_data_requests": 0,
                        "validated_trade_rules_added": 0}
            (staging / "manifest.json").write_bytes(encoded(manifest))
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    publish_link("pages", destination / "pages")
    publish_link("index.html", destination / "index.html")
    print(json.dumps({"securities": len(codes), "trait_dossiers": len(profiles),
                      "analog_dossiers": len(analogs), "view": str(destination),
                      "index": str(HERE / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
