"""One self-contained page per qualified combo and per conceptual research family."""
from __future__ import annotations
import base64,gzip,html,json,pickle
import pandas as pd
import numpy as np
import build as b
import evaluate as ev
OUT=b.OUT
FAMILIES={
 'nonlinear':('单股非线性量价',['solo_forest']),
 'smooth':('紧凑与平滑量价',['logistic_solo_01','logistic_solo_1','compact_solo_short','compact_solo_long']),
 'related':('相关股与市场环境',['macro_related_long','related_short','related_long','compact_related_short','compact_related_long']),
 'regime':('匹配行业环境',['regime_related']),
 'analogs':('历史相似日',['analog_solo_15','analog_solo_30','analog_related_60','macro_analog_solo']),
 'macro':('前日风险环境与单股结构',['macro_solo_short','macro_solo_long','macro_logistic_solo'])}
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
 if isinstance(x,np.generic):return x.item()
 return x

def anchor(c):return (pd.Timestamp(c['start'])-pd.Timestamp('2025-04-01')).days%30

def main():
 report=json.loads((OUT/'results.json').read_text());reports=[report];preds=[pd.read_parquet(OUT/'enrolled_predictions.parquet')]
 if (OUT/'alignment_results.json').exists():reports.append(json.loads((OUT/'alignment_results.json').read_text()));preds.append(pd.read_parquet(OUT/'alignment_enrolled_predictions.parquet'))
 pred=pd.concat(preds,ignore_index=True);allcells=sum((r['cells'] for r in reports),[]);enrolled=sum((r['enrolled_combinations'] for r in reports),[]);passed=[c for c in enrolled if c['pass_research']];passed.sort(key=lambda c:(c['symbol'],c['start'],c['selected']))
 daily=json.loads((OUT/'daily.json').read_text());calendar={r['session_date']:dict(close=570+r['duration_minutes'],duration=r['duration_minutes']) for r in json.loads((b.legacy.HISTORY/'calendar.json').read_text())['sessions'] if b.START<=r['session_date']<=b.END}
 template=(OUT/'page.html').read_text();cache={};pages=[]
 targets=[(f"{c['symbol']}_{c['start']}_{c['selected']}.html",c,'历史合格组合') for c in passed]
 for key,(title,names) in FAMILIES.items():
  pool=[c for c in passed if c['selected'] in names] or [c for c in enrolled if c['selected'] in names]
  if pool:targets.append((f'idea_{key}.html',pool[0],title))
 for filename,c,kind in targets:
  s=c['symbol'];name=c['selected'];phase=anchor(c)
  if s not in cache:
   f=pd.read_parquet(OUT/'raw'/f'{s}.parquet');bars=[]
   for r in f.itertuples():
    d=(r.start+pd.Timedelta(days=1) if r.minute>=1200 else r.start).strftime('%Y-%m-%d')
    if d not in calendar:continue
    k='night' if r.minute>=1200 or r.minute<240 else 'pre' if r.minute<570 else 'regular' if r.minute<calendar[d]['close'] else 'post'
    bars.append([r.start.strftime('%Y-%m-%dT%H:%M'),r.end.strftime('%Y-%m-%dT%H:%M'),*[round(float(v),5) for v in [r.open,r.high,r.low,r.close]],int(r.volume),d,k])
   cache[s]=bars
  registered={x['start']:x for x in enrolled if x['symbol']==s and x['selected']==name and anchor(x)==phase};cells=[]
  for x in sorted((x for x in allcells if x['symbol']==s and anchor(x)==phase),key=lambda x:x['start']):
   if x['start'] in registered:cells.append(registered[x['start']])
   else:cells.append(dict(symbol=s,start=x['start'],end=x['end'],selected=None,pass_research=False,reason='该模型未通过周期前校准',variants=[]))
  idx=next(i for i,x in enumerate(cells) if x['start']==c['start']);selectedpred=pred[pred.symbol.eq(s)&pred.model.eq(name)].copy();selectedpred=selectedpred[selectedpred.cycle_start.map(lambda d:(pd.Timestamp(d)-pd.Timestamp('2025-04-01')).days%30)==phase]
  assert not selectedpred.duplicated('day').any()
  # Feature importance is explanatory only. Coefficients are not causal effects.
  with (OUT/'models'/c['artifact']).open('rb') as f:a=pickle.load(f)
  model=a['model'];importance=model.feature_importances_ if hasattr(model,'feature_importances_') else model[-1].feature_importances_[:len(a['features'])] if hasattr(model[-1],'feature_importances_') else np.abs(model[-1].coef_[0][:len(a['features'])]) if hasattr(model[-1],'coef_') else np.zeros(len(a['features']))
  c['importance']=sorted(zip(a['features'],importance.tolist()),key=lambda p:-p[1])[:20]
  payload=dict(symbol=s,title=f"{s} · {report['models'][name]} · {kind}",intro=f'09:25 盘前输入 → 当天常规盘从 09:30 Open 触及 +3%。该模型须在周期前通过校准；这是逐模型登记结果，另见总览的唯一择模策略整体结果。日期锚点偏移 {phase} 天，跨锚点窗口可能重叠。',models=report['models'],cells=cells,defaultCycle=idx,bars=cache[s],calendar=calendar,daily=[x for x in daily[s] if b.START<=x['day']<=b.END],predictions=selectedpred.to_dict('records'))
  encoded=base64.b64encode(gzip.compress(json.dumps(clean(payload),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),mtime=0)).decode();(OUT/filename).write_text(template.replace('__PAYLOAD__',encoded));pages.append(dict(path=filename,symbol=s,start=c['start'],model=name,kind=kind,pass_research=c['pass_research']))
 def ledger(cells,variant=False):
  rows=[]
  for c in sorted(cells,key=lambda c:(not c['pass_research'],c['symbol'],c['start'])):
   m=c.get('metrics',{});link=next((p['path'] for p in pages if p['symbol']==c['symbol'] and p['model']==c['selected'] and p['start']==c['start']),None)
   sym=f'<a href="{link}">{c["symbol"]}</a>' if link else c['symbol'];base=[sym,c['start']+'—'+c['end'],report['models'].get(c['selected'],'无前置合格模型')]
   if m and m.get('precision') is not None:base += [f'{m["tp"]}/{m["n"]}',f'{m["precision"]*100:.1f}%',f'{m["base"]*100:.1f}%',f'{m["lift"]*100:.1f}pp','通过' if c['pass_research'] else c['reason']]
   else:base += ['0','—','—','—',c['reason']]
   rows.append(f'<tr data-pass="{int(c["pass_research"])}">'+''.join('<td>'+str(v)+'</td>' for v in base)+'</tr>')
  return ''.join(rows)
 links=''.join(f'<li><a href="{p["path"]}">{p["symbol"]} · {report["models"][p["model"]]} · {p["start"]} · {p["kind"]}</a>（{"通过" if p["pass_research"] else "未通过"}）</li>' for p in pages)
 distinct=sorted(set(c['symbol'] for c in passed));policy=report['policy'];intro=f'逐模型登记：{len(passed)} 个历史合格组合，{len(distinct)} 只特别关注股票（'+', '.join(distinct)+f'）。原锚点的唯一择模策略：{policy["tp"]}/{policy["n"]}，precision {policy["precision"]*100:.1f}%；仍未达到整体 70%。多个组合共享股票 / 日期，不能算独立信号。'
 style='body{font:14px system-ui;margin:20px;color:#222;background:#fafafa}main{max-width:1450px;margin:auto}section{border:1px solid #999;padding:14px;margin:14px 0;background:white}a{color:#222}p{line-height:1.7}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #aaa;padding:7px;white-space:nowrap}th{background:#eee}.scroll{overflow:auto}input,select{font:inherit;padding:5px;border:1px solid #777}h1{font-size:24px}'
 head='<tr><th>股票</th><th>周期</th><th>周期前登记模型</th><th>命中/信号</th><th>precision</th><th>同股基准</th><th>增量</th><th>结果</th></tr>'
 doc=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单股周期盘前研究</title><style>{style}</style><main><h1>盘前量价 → 当天触及 +3%：单股 × 30 天周期</h1><p>{intro}</p><p>两年行情 2024-10-03—2026-10-02。这是已暴露历史上的前向开发回测；合格周期是后验发现，下一周期仍需重新校准。没有以调低原门禁制造通过。</p><section><h2>可调试页面</h2><ul>{links}</ul><a href="REPORT.md">研究结果 / 失败解释 / 复算</a> · <a href="PROTOCOL.md">原冻结协议</a> · <a href="MACRO_ENROLLMENT_PROTOCOL.md">宏观与逐模型登记协议</a> · <a href="ALIGNMENT_PROTOCOL.md">周期对齐协议</a> · <a href="audit.json">数据审计</a></section><section><h2>全部前置登记模型账本</h2><label>股票 <input id="stock" placeholder="如 AAOI"></label> <label><input type="checkbox" id="pass">只看合格组合</label><p id="count"></p><div class="scroll"><table><thead>{head}</thead><tbody id="enrolledRows">{ledger(enrolled,True)}</tbody></table></div></section><details><summary>所有股票周期：唯一择模策略（包含全部未登记 / 失败周期）</summary><div class="scroll"><table><thead>{head}</thead><tbody>{ledger(allcells)}</tbody></table></div></details><p>32 只特别关注 STOCK、QQQ、静态相关股与行业 ETF已使用；全自选 / ETF 成分范围快照继承 v2，尚未验证全部 1858 个名称，不称全池覆盖。当前成分回溯不是历史 PIT。原始行情与模型不进 Git，不自动上传，不接交易。</p></main><script>function filter(){{let n=0;document.querySelectorAll("#enrolledRows tr").forEach(r=>{{r.hidden=(!r.cells[0].innerText.includes(document.getElementById("stock").value.toUpperCase()))||(document.getElementById("pass").checked&&r.dataset.pass!="1");if(!r.hidden)n++}});document.getElementById("count").textContent=n+" 个模型周期"}}document.getElementById("stock").oninput=filter;document.getElementById("pass").onchange=filter;filter()</script></html>'
 (OUT/'index.html').write_text(doc);(OUT/'pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2));(OUT/'delivery_combinations.json').write_text(json.dumps(dict(passed=clean(passed),distinct_stocks=distinct,stock_goal_pass=len(distinct)>=5,combination_goal_pass=len(passed)>=5,total_registered=len(enrolled),total_stock_cycles=len(allcells)),ensure_ascii=False,indent=2));print(json.dumps({'pages':len(pages),'qualified':len(passed),'stocks':distinct},ensure_ascii=False))
if __name__=='__main__':main()
