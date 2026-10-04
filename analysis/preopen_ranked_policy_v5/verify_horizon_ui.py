"""Desktop/mobile target switch, raw paths, no zoom and cloud-option UI."""
import json,datetime
from common import OUT
from verify_ui import call,js,wait

def main():
 logs=[]
 for route in ['', 'hazard_linear','relative_flow','recovery_forest']:
  call('set','viewport','1440','1000');call('open','http://127.0.0.1:8770/'+(route+'.html' if route else ''));wait('APP.ready&&HORIZON&&HORIZON_EVENTS.length===horizonHead().routes.find(r=>r.route===$("horizonRoute").value).metrics.n');call('snapshot','-s','#horizonTarget');expected=route or 'recovery_forest';assert js('$("horizonRoute").value')==expected;record=dict(page=route or 'index',targets=[])
  for target in ['5d5pct','10d10pct']:
   call('select','#horizonTarget',target);wait('HORIZON_EVENTS.length===horizonHead().routes.find(r=>r.route===$("horizonRoute").value).metrics.n');assert js('APP.horizonWeeks')==23;assert js('APP.horizonModels.join()')=='LogisticRegression,LightGBM,ExtraTrees';assert js('$("horizonSummary").children.length')==3
   call('select','#horizonFilter','fail');call('click','#horizonRows button');wait('APP.inspectReady&&FOCUS.horizon_end&&FOCUS.hit===0');assert js('$("detail").innerText.includes("有效但失败")&&$("features").innerText.includes("h_span_20")&&$("decisionRows").children.length>0');assert js('$("start").value===FOCUS.day&&$("end").value===FOCUS.horizon_end')
   shade=js('({first:FOCUS.day,last:FOCUS.horizon_end,dates:[...new Set(AGG.filter(r=>r.kind==="regular"&&focusWindow(r)).map(r=>r.calendarDay))],target:FOCUS.target_gain})');assert len(shade['dates'])==(5 if target=='5d5pct' else 10)
   call('select','#horizonFilter','hit');call('click','#horizonRows button');wait('APP.inspectReady&&FOCUS.hit===1');assert js('$("detail").innerText.includes("首次触及")')
   call('select','#horizonFilter','pending');call('click','#horizonRows button');wait('APP.inspectReady&&FOCUS.hit===null');assert js('$("detail").innerText.includes("待成熟或不可评分")');assert js('$("end").value<="2026-09-30"')
   call('select','#horizonFilter','all');call('select','#horizonWeek','1');assert js('selectedHorizonEvents().every(e=>e.day>=horizonHead().weeks[1].first_session&&e.day<=horizonHead().weeks[1].last_session)');call('select','#horizonWeek','all')
   csv=js('(async()=>{let old=URL.createObjectURL;URL.createObjectURL=b=>{window.__horizonBlob=b;return old.call(URL,b)};$("horizonExport").click();let text=await __horizonBlob.text();URL.createObjectURL=old;return text.split("\\n")[0]})()');assert csv.startswith('target,route,algorithm,')
   nozoom=js('(()=>{let before=[POS,$("grain").value,$("start").value,$("end").value];$("chart").dispatchEvent(new WheelEvent("wheel",{deltaY:300}));return {before,after:[POS,$("grain").value,$("start").value,$("end").value]}})()');assert nozoom['before']==nozoom['after'];call('click','#first');call('click','#right');assert js('POS')>0
   record['targets'].append(dict(target=target,signals=js('HORIZON_EVENTS.length'),shade=shade,csv=csv,nozoom=True))
  call('set','viewport','390','844');assert not js('document.documentElement.scrollWidth>innerWidth');js('document.querySelector("section[aria-label=跨日目标回测]").scrollIntoView({block:"start"})');call('screenshot',str(OUT/'screenshots'/f'horizon_{route or "index"}_mobile.png'));record['mobile_overflow']=False
  if not route:
   call('set','viewport','1440','1000')
   for target in ['5d5pct','10d10pct','intraday3pct']:
    call('select','#optionsTarget',target);wait('!$("allOptions").disabled&&$("optionsStatus").innerText.includes("分别计算")&&APP.optionCounts.join()=="3,3,3"');assert js('$("allModelOptions").innerText.includes("LightGBM")');assert js('$("allModelOptions").innerText.includes("1.05")') if target=='5d5pct' else True
   record['all_target_options']=True
  logs.append(record);print(json.dumps(dict(page=record['page'],passed=True)),flush=True)
 call('close');(OUT/'horizon_v1/ui_verification.json').write_text(json.dumps(dict(status='passed',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pages=logs),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
