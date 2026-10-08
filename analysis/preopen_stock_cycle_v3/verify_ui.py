"""Real desktop/mobile browser verification of every delivered page."""
from pathlib import Path
import subprocess,json,datetime
OUT=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8768/analysis/preopen_stock_cycle_v3/'
CMD=['/Users/admin/.npm/_npx/6de2aa2fded2970c/node_modules/agent-browser/bin/agent-browser-darwin-arm64','--session','preopen-v3','--json']
def call(*args):
 p=subprocess.run(CMD+list(args),text=True,capture_output=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 r=json.loads(p.stdout)
 if not r['success']:raise RuntimeError(str(r))
 return r.get('data',{}).get('result',r.get('data'))
def js(s):return call('eval',s)
def main():
 pages=json.loads((OUT/'pages.json').read_text());logs=[];(OUT/'screenshots').mkdir(exist_ok=True)
 for page in pages:
  path=page['path'];call('set','viewport','1440','1000');call('open',BASE+path);call('wait','--fn','!!window.APP');call('snapshot','-i')
  initial=js('({title:$("title").innerText,ready:!$("app").hidden,start:$("start").value,end:$("end").value,overflow:document.documentElement.scrollWidth>innerWidth,spacing:DATA.viewport.spacing,signals:PRED.filter(p=>p.signal).length,tp:PRED.filter(p=>p.signal&&p.y).length,cell:APP.cell.metrics,grain:$("grain").value,night:RAW.some(r=>r[8]=="night"),official:RAW.every(r=>DATA.calendar[r[7]])})')
  assert initial['ready'] and not initial['overflow'] and initial['spacing']>=5 and initial['grain']=='60' and initial['official'] and initial['night']
  assert initial['signals']==initial['cell']['n'] and initial['tp']==initial['cell']['tp']
  assert js('MINIS.length==Object.keys(DATA.calendar).length&&MINIS[0].day=="2024-10-03"&&MINIS.at(-1).day=="2026-10-02"')
  presets=[]
  for days in [30,60,90,180]:
   call('click',f'[data-days="{days}"]');r=js('({start:$("start").value,end:$("end").value,text:$("period").innerText})');assert (datetime.date.fromisoformat(r['end'])-datetime.date.fromisoformat(r['start'])).days+1==days;presets.append(days)
  call('click','#mini');before=js('$("start").value');call('press','ArrowLeft');after=js('$("start").value');assert (datetime.date.fromisoformat(before)-datetime.date.fromisoformat(after)).days==30
  call('click','#twoYears');assert js('[$("start").value,$("end").value]')==['2024-10-03','2026-10-02']
  nozoom=js('(()=>{let before=[$("start").value,$("end").value,POS,$("grain").value];$("chart").dispatchEvent(new WheelEvent("wheel",{deltaY:800,bubbles:true}));$("chart").dispatchEvent(new MouseEvent("mousedown",{clientX:200,bubbles:true}));$("chart").dispatchEvent(new MouseEvent("mousemove",{clientX:800,bubbles:true}));$("chart").dispatchEvent(new MouseEvent("mouseup",{clientX:800,bubbles:true}));return {before,after:[$("start").value,$("end").value,POS,$("grain").value],touch:getComputedStyle($("chart")).touchAction}})()');assert nozoom['before']==nozoom['after'] and nozoom['touch']=='pan-y'
  call('click','#reset');call('select','#filter','failures');failure=js('(()=>{let b=document.querySelector("[data-date]");if(!b)return null;let d=b.dataset.date;b.click();return {day:d,grain:$("grain").value,detail:$("detail").innerText,signal:DATA.predictions.find(p=>p.day==d).signal,y:DATA.predictions.find(p=>p.day==d).y,viewport:DATA.viewport,rows:$("featureRows").children.length}})()')
  if failure:assert failure['grain']=='5' and failure['signal'] and not failure['y'] and '未触及' in failure['detail'] and failure['rows']>=15
  call('select','#grain','30');assert js('$("grain").value')=='30';call('select','#grain','15');assert js('$("grain").value')=='15';call('select','#grain','5');call('click','#last');assert js('POS==+$("pan").max');call('click','#focus');call('click','#right');right=js('POS');call('click','#left');assert js('POS')<=right
  call('select','#spacing','9');assert js('DATA.viewport.spacing')==9;call('select','#spacing','5')
  call('select','#filter','signals');exported=js('(async()=>{let original=URL.createObjectURL;URL.createObjectURL=b=>{window.__blob=b;return original.call(URL,b)};$("export").click();let text=await __blob.text();URL.createObjectURL=original;return {header:text.split("\\n")[0],lines:text.split("\\n").length}})()');assert 'feature_end' in exported['header'] and 'threshold' in exported['header']
  call('screenshot',str(OUT/'screenshots'/path.replace('.html','_desktop.png')));call('set','viewport','390','844');mobile=js('({overflow:document.documentElement.scrollWidth>innerWidth,chart:$("chart").clientWidth,spacing:DATA.viewport.spacing,mini:$("mini").clientWidth})');assert not mobile['overflow'] and mobile['chart']>=280 and mobile['spacing']>=5
  call('screenshot',str(OUT/'screenshots'/path.replace('.html','_mobile.png')))
  logs.append(dict(page=path,initial=initial,presets=presets,nozoom=nozoom,failure=failure,mobile=mobile,exported_csv=exported,status='passed'));print(json.dumps({'page':path,'passed':True}),flush=True)
 call('open',BASE+'index.html');call('snapshot','-i');call('fill','#stock','AAOI');count=js('document.querySelectorAll("#enrolledRows tr:not([hidden])").length');assert count>0;call('check','#pass');countpass=js('document.querySelectorAll("#enrolledRows tr:not([hidden])").length');assert countpass<=count
 (OUT/'ui_verification.json').write_text(json.dumps(dict(status='passed',tool='agent-browser',pages=logs,index_filter=True),ensure_ascii=False,indent=2));print('Every delivered page passed desktop/mobile interactions.')
if __name__=='__main__':main()
