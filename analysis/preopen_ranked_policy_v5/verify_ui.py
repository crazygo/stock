"""Exercise each live wireframe at desktop/mobile without touching user browser."""
from pathlib import Path
import subprocess,json,datetime,time,os
os.environ.setdefault("AGENT_BROWSER_DEFAULT_TIMEOUT","45000")
OUT=Path(__file__).resolve().parent
CMD=['/Users/admin/.npm/_npx/6de2aa2fded2970c/node_modules/agent-browser/bin/agent-browser-darwin-arm64','--session','preopen-v5','--json']
def call(*args):
 if args[0]=='click':
  call('eval','document.querySelector('+json.dumps(args[1])+').scrollIntoView({block:"center"})')
 p=subprocess.run(CMD+list(args),text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stdout[:500]+p.stderr[:500])
 r=json.loads(p.stdout)
 if not r['success']:raise RuntimeError(str(r)[:1000])
 return r.get('data',{}).get('result',r.get('data'))
def js(s):return call('eval',s)
def wait(s):return call('wait','--fn',s)
def main():
 logs=[];(OUT/'screenshots').mkdir(exist_ok=True)
 for route in ['','hazard_linear','relative_flow','recovery_forest']:
  name=route or 'index';call('set','viewport','1440','1000');call('open','http://127.0.0.1:8770/'+(route+'.html' if route else ''));wait('APP.ready&&(!STATE.job.running)&&CHART&&APP.loadedRange.join()==[$("start").value,$("end").value].join()');call('snapshot','-i')
  state=js('({ready:APP.ready,route:$("route").value,overflow:document.documentElement.scrollWidth>innerWidth,mini:APP.miniDays,spacing:APP.spacing,action:STATE.latest.action,current:STATE.latest.current_probability,rankRows:$("rankRows").children.length,tradeCount:TRADES.length,error:$("error").textContent})');assert state['ready'] and not state['overflow'] and state['spacing']>=5 and not state['error'];assert state['mini']==501 and state['action']=='no_action' and not state['current'];assert state['route']==(route or STATE_ROUTE)
  wait('APP.signalRange.join()=="2026-05-01,2026-09-30"');weekly=js('({n:SIGNALS.length,reported:STATE.weekly_results.routes.find(r=>r.route===$("route").value).metrics.n,weeks:$("weeklyRows").children.length,models:APP.optionModels,counts:APP.optionCounts})');assert weekly['n']==weekly['reported'] and weekly['weeks']==23 and weekly['models']==3 and weekly['counts']==[3,3,3]
  call('select','#signalFilter','fail');call('click','#signalRows button');wait('APP.inspectReady&&FOCUS.is_signal&&FOCUS.hit===0');assert js('$("detail").innerText.includes("有效但未达成")');call('select','#signalFilter','all')
  if not route:
   call('click','#allOptions');wait('!$("allOptions").disabled&&$("optionsStatus").innerText.includes("分别计算")');assert js('APP.optionCounts.join()=="3,3,3"')
  call('select','#signalPeriod','2026-08_09');call('click','#both');wait('APP.signalRange.join()=="2026-08-01,2026-09-30"');signal=js('({n:SIGNALS.length,reported:STATE.signal_results.routes.find(r=>r.route===$("route").value).metrics.n,precision:STATE.signal_results.routes.find(r=>r.route===$("route").value).metrics.precision,main:$("signalMetrics").innerText})');assert signal['n']==signal['reported']
  if signal['n']:
   call('select','#signalFilter','fail');call('click','#signalRows button');wait('APP.inspectReady&&FOCUS.is_signal&&FOCUS.hit===0');assert js('$("detail").innerText.includes("有效但未达成")&&$("detail").innerText.includes("仍计入准确率分母")');call('select','#signalFilter','hit');call('click','#signalRows button');wait('APP.inspectReady&&FOCUS.is_signal&&FOCUS.hit===1');js('$("chart").scrollIntoView({block:"center"})');call('screenshot',str(OUT/'screenshots'/f'{name}_signal_desktop.png'));call('set','viewport','390','844');js('$("chart").scrollIntoView({block:"center"})');assert js('document.documentElement.scrollWidth<=innerWidth');call('screenshot',str(OUT/'screenshots'/f'{name}_signal_mobile.png'));call('set','viewport','1440','1000');call('select','#signalFilter','all');call('click','#both');wait('FOCUS===null')
  else:assert js('$("signalRows").innerText.includes("不可评分")')
  signal_csv=js('(async()=>{let original=URL.createObjectURL;URL.createObjectURL=b=>{window.__signalBlob=b;return original.call(URL,b)};$("signalExport").click();let t=await __signalBlob.text();URL.createObjectURL=original;return t.split("\\n")[0]})()');assert 'threshold' in signal_csv and 'hit' in signal_csv
  call('check','#fav');favorites=js('$("rankRows").children.length');assert favorites<=state['rankRows'];call('uncheck','#fav');call('check','#qualified');assert js('$("rankRows").innerText.includes("没有")');call('uncheck','#qualified')
  call('click','#aug');wait('$("end").value==="2026-08-31"&&CHART.bars.length>0');call('click','#sep');wait('$("start").value==="2026-09-01"');call('click','#both');wait('$("end").value==="2026-09-30"&&$("start").value==="2026-08-01"')
  for n in [30,60,90,180]:
   call('click',f'[data-days="{n}"]');wait('CHART&&CHART.bars.length>0&&APP.loadedRange.join()==[$("start").value,$("end").value].join()&&APP.ledgerRange.join()==[$("start").value,$("end").value].join()');d=js('[$("start").value,$("end").value]');assert (datetime.date.fromisoformat(d[1])-datetime.date.fromisoformat(d[0])).days+1==n
  call('click','#twoYears');wait('$("start").value==="2024-10-03"&&$("end").value==="2026-10-02"');call('click','#mini');before=js('$("start").value');call('press','ArrowLeft');after=js('$("start").value');assert (datetime.date.fromisoformat(before)-datetime.date.fromisoformat(after)).days==30;call('click','#both')
  nozoom=js('(()=>{let a=[POS,$("grain").value,$("start").value,$("end").value];$("chart").dispatchEvent(new WheelEvent("wheel",{deltaY:300,bubbles:true}));$("chart").dispatchEvent(new MouseEvent("mousedown",{clientX:100}));$("chart").dispatchEvent(new MouseEvent("mousemove",{clientX:900}));return {before:a,after:[POS,$("grain").value,$("start").value,$("end").value],touch:getComputedStyle($("chart")).touchAction}})()');assert nozoom['before']==nozoom['after'] and nozoom['touch']=='pan-y'
  for grain in ['30','15','5','60']:call('select','#grain',grain)
  wait('APP.loadedRange.join()==[$("start").value,$("end").value].join()');call('click','#first');assert js('POS')==0;call('click','#right');right=js('POS');assert right>0;call('click','#left');assert js('POS')<right;call('click','#last');assert js('POS==+$("pan").max')
  wait('APP.loadedRange.join()==[$("start").value,$("end").value].join()&&APP.ledgerRange.join()==[$("start").value,$("end").value].join()');call('select','#tradeFilter','fail');failure=None
  if state['tradeCount']:
   call('click','#tradeRows button');wait('APP.inspectReady&&FOCUS&&$("grain").value==="5"&&$("decisionRows").children.length>0');assert js('AGG.slice(POS,POS+visibleCount()).some(r=>r.calendarDay===FOCUS.day&&r.minute===FOCUS.entry_minute)');failure=js('({stock:FOCUS.symbol,hit:FOCUS.hit,rows:$("decisionRows").children.length,detail:$("detail").innerText,featureKeys:Object.keys(JSON.parse($("features").innerText)).length})');assert failure['hit']!=1 and failure['rows']>0 and failure['featureKeys']>40 and '未触及' in failure['detail']
  if state['tradeCount']:
   call('select','#tradeFilter','hit');call('click','#tradeRows button');wait('APP.inspectReady&&FOCUS.hit===1');assert js('$("detail").innerText.includes("首次触及")')
   call('screenshot',str(OUT/'screenshots'/f'{name}_path_desktop.png'));call('set','viewport','390','844');assert js('AGG.slice(POS,POS+visibleCount()).some(r=>r.calendarDay===FOCUS.day&&r.minute===FOCUS.entry_minute)');js('$("chart").scrollIntoView({block:"center"})');call('screenshot',str(OUT/'screenshots'/f'{name}_path_mobile.png'));call('set','viewport','1440','1000')
  call('click','#both');wait('APP.ledgerRange.join()=="2026-08-01,2026-09-30"');assert js('FOCUS===null&&$("features").innerText==="未选择信号 / 买入"');call('select','#tradeFilter','all');call('click','#noActions');wait('$("noActionDetail").open');assert js('$("noActionRows").innerText.includes("不操作")')
  exported=js('(async()=>{let original=URL.createObjectURL;URL.createObjectURL=b=>{window.__blob=b;return original.call(URL,b)};$("export").click();let t=await __blob.text();URL.createObjectURL=original;return t.split("\\n")[0]})()');assert 'target' in exported and 'net' in exported
  call('check','#auto');assert js('!!AUTO');call('uncheck','#auto');js('scrollTo(0,0)');call('screenshot',str(OUT/'screenshots'/f'{name}_desktop.png'));call('set','viewport','390','844');mobile=js('({overflow:document.documentElement.scrollWidth>innerWidth,chart:$("chart").clientWidth,spacing:APP.spacing})');assert not mobile['overflow'] and mobile['chart']>280;js('scrollTo(0,0)');call('screenshot',str(OUT/'screenshots'/f'{name}_mobile.png'));logs.append(dict(page=name,initial=state,favorites=favorites,nozoom=nozoom,failure=failure,mobile=mobile,csv=exported));print(json.dumps({'page':name,'passed':True}),flush=True)
  call('set','viewport','1440','1000');call('select','#policyVersion','capacity');wait('runId().startsWith("v5capacity:")&&APP.ledgerRange.join()=="2026-08-01,2026-09-30"&&TRADES.length===backtestResults().routes.find(r=>r.route===$("route").value).metrics.n')
  assert js('$("modelInfo").innerText.includes("限量小仓位")&&$("currentModel").innerText.includes("回撤恢复")&&$("coverage").textContent.includes("可评分覆盖")')
  logs[-1]['capacity']=js('({run:runId(),trades:TRADES.length,precision:backtestResults().routes.find(r=>r.route===$("route").value).metrics.precision})');logs[-1]['weekly']=weekly;logs[-1]['signals']=signal;logs[-1]['signal_csv']=signal_csv
  if js('TRADES.length'):
   call('select','#tradeFilter','fail');call('click','#tradeRows button');wait('APP.inspectReady&&FOCUS&&FOCUS.hit!==1');assert js('$("detail").innerText.includes("未触及")&&$("decisionRows").children.length>0')
  call('select','#policyVersion','phase');wait('runId().startsWith("v5phase:")&&FOCUS===null')
 # One real closed-market refresh checks gateway -> features -> ledger -> UI.
 call('click','#refresh');wait('STATE.job.running||$("job").innerText.includes("开始")');wait('!STATE.job.running&&$("job").innerText.includes("完成")');assert js('STATE.latest.rows.length>100&&STATE.latest.action==="no_action"');call('close');(OUT/'ui_verification.json').write_text(json.dumps(dict(pages=logs,actual_refresh=True,status='passed',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),ensure_ascii=False,indent=2))
STATE_ROUTE='recovery_forest'
if __name__=='__main__':main()
