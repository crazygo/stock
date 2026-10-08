"""Real browser flow checks against the rendered frozen research artifacts."""
from pathlib import Path
import subprocess,json,datetime
OUT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8768/analysis/preopen_intraday_v2/'
CMD=['npx','--yes','agent-browser','--session','preopen-v2','--json']
def call(*args):
    p=subprocess.run(CMD+list(args),text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(' '.join(args[:2])+': '+p.stderr+p.stdout)
    r=json.loads(p.stdout)
    if not r['success']:raise RuntimeError(str(r))
    return r.get('data',{}).get('result',r.get('data'))
def js(script):return call('eval',script)
def main():
    monthly=json.loads((OUT/'adaptive_results.json').read_text());fixed=json.loads((OUT/'results.json').read_text());logs=[]
    (OUT/'screenshots').mkdir(exist_ok=True)
    for route in ['continuation','repair','relative']:
        call('set','viewport','1440','1000');call('open',BASE+route+'.html');call('wait','--fn','!!document.getElementById("app")&&!document.getElementById("app").hidden')
        base=js('({title:document.title,ready:!document.getElementById("app").hidden,grain:$("grain").value,target:$("target").value,version:$("version").value,start:$("start").value,end:$("end").value,overflow:document.documentElement.scrollWidth>innerWidth,forecast:$("dayForecast").innerText})')
        assert base['ready'] and base['grain']=='hourly' and base['target']=='0.03' and base['version']=='monthly' and not base['overflow']
        presets=[]
        for days in [30,60,90,180]:
            call('click',f'[data-days="{days}"]');r=js('({start:$("start").value,end:$("end").value})');span=(datetime.date.fromisoformat(r['end'])-datetime.date.fromisoformat(r['start'])).days+1;assert span==days;presets.append(span)
        call('click','#mini');before=js('$("start").value');call('press','ArrowLeft');after=js('$("start").value');assert (datetime.date.fromisoformat(before)-datetime.date.fromisoformat(after)).days==7
        call('click','#twoYears');r=js('({start:$("start").value,end:$("end").value})');assert r=={'start':'2024-10-03','end':'2026-10-02'}
        nozoom=js('(()=>{const before=[$("start").value,$("end").value];$("chart").dispatchEvent(new WheelEvent("wheel",{deltaY:500,bubbles:true}));$("chart").dispatchEvent(new MouseEvent("mousedown",{clientX:300,bubbles:true}));$("chart").dispatchEvent(new MouseEvent("mousemove",{clientX:800,bubbles:true}));$("chart").dispatchEvent(new MouseEvent("mouseup",{clientX:800,bubbles:true}));return {before,after:[$("start").value,$("end").value],touch:getComputedStyle($("chart")).touchAction}})()');assert nozoom['before']==nozoom['after'] and nozoom['touch']=='pan-y'
        call('select','#pool','all');call('click','#frozenPeriod');counts=[]
        for version,report in [('monthly',monthly),('fixed',fixed)]:
            call('select','#version',version);call('click','#reset');m=js('({precision:$("metrics").children[0].querySelector("strong").innerText,n:Number($("metrics").children[2].querySelector("strong").innerText),frozen:$("frozen").innerText})');expected=report['routes'][route]['summary'];assert m['n']==expected['n'];assert m['precision']==f"{expected['precision']*100:.1f}%";counts.append({'version':version,**m})
        call('fill','#threshold','0.8');mode=js('$("threshold").dispatchEvent(new Event("change"));$("mode").innerText');assert '探索' in mode
        call('click','#reset');call('select','#target','0.01');assert '探索' in js('$("mode").innerText');call('click','#reset');call('select','#entry','delay');assert '探索' in js('$("mode").innerText');call('click','#reset')
        call('select','#pool','favorites');call('select','#symbol','CRDO');call('select','#version','monthly')
        detail=js('(()=>{let b=document.querySelector("[data-day]");let d=b.dataset.day;b.click();return {day:d,start:$("start").value,end:$("end").value,grain:$("grain").value,features:$("dayFeatures").children.length}})()');assert detail['day']==detail['start']==detail['end'] and detail['features']>5 and detail['grain']=='hourly'
        exported=js('(async()=>{const original=URL.createObjectURL;URL.createObjectURL=b=>{window.__exportBlob=b;return original.call(URL,b)};$("export").click();const s=await window.__exportBlob.text();URL.createObjectURL=original;return {header:s.split("\\n")[0],lines:s.split("\\n").length,hasProbability:s.includes("score_"+ROUTE)}})()');assert exported['lines']==2 and exported['hasProbability'] and 'model_version' in exported['header']
        call('screenshot',str(OUT/'screenshots'/f'{route}_desktop.png'))
        call('select','#pool','etf');call('select','#symbol','ZYME');missing=js('({coverage:$("coverage").innerText,events:$("events").innerText})');assert '没有可评分行' in missing['events']
        call('select','#pool','favorites');call('select','#symbol','CRDO');call('click','[data-days="30"]');call('set','viewport','390','844');mobile=js('({overflow:document.documentElement.scrollWidth>innerWidth,width:$("chart").getBoundingClientRect().width,mini:$("mini").getBoundingClientRect().width})');assert not mobile['overflow'] and mobile['width']>=250 and mobile['mini']>=250;call('screenshot',str(OUT/'screenshots'/f'{route}_mobile.png'))
        logs.append({'route':route,'default':base,'preset_days':presets,'mini_keyboard_shift_days':7,'mouse_zoom':nozoom,'frozen_counts':counts,'exploration_label':mode,'day_detail':detail,'csv':exported,'missing_data':missing,'mobile':mobile,'status':'passed'})
        print(json.dumps({'route':route,'status':'passed'}),flush=True)
    result={'status':'passed','tool':'agent-browser','viewports':[[1440,1000],[390,844]],'routes':logs};(OUT/'ui_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('All three browser flows passed.')
if __name__=='__main__':main()
