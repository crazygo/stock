/* Offline source/data/render checks. This does not automate a browser. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const TL=require('./timeline.js'),out=__dirname,data=JSON.parse(fs.readFileSync(path.join(out,'results.json'))),checks=[];
function check(name,ok){checks.push({name,passed:!!ok});if(!ok)throw Error(name);}
function sha(p){return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');}
function main(){
  const html=fs.readFileSync(path.join(out,'index.html'),'utf8');
  const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];check('one self-contained script',scripts.length===1);new vm.Script(scripts[0][1]);check('inline JavaScript compiles',true);
  const js=scripts[0][1],begin=js.indexOf('const D=')+8,embedded=JSON.parse(js.slice(begin,js.indexOf(', $=id=>',begin)));
  check('embedded snapshot and member count match',embedded.run_id===data.run_id&&embedded.as_of===data.as_of&&embedded.records.length===data.records.length);
  for(const r of data.records){const actual=embedded.records.find(x=>x.code===r.code);check('embedded source values retained '+r.code,actual&&actual.history.length===r.history.length&&actual.history.every((h,i)=>h.day===r.history[i].day&&h.v.every((v,k)=>v==null?r.history[i].v[k]==null:Math.abs(v-r.history[i].v[k])<=.0000006)));}
  check('one shared timeline container',(html.match(/id="time-spectrum"/g)||[]).length===1);
  check('progress alignment and period-line controls removed',!html.includes('id="time-axis"')&&!html.includes('id="period-legend"')&&!html.includes('周期进度'));
  check('all three calendar lookback buttons', [14,30,60].every(n=>html.includes('data-lookback="'+n+'"')));
  check('unknown transforms do not become zero',TL.KEYS.every(k=>TL.score(k,null)===null));
  check('fixed scale reference values',TL.score('G',0)===50&&TL.score('V',.35)===50&&TL.score('M',-1)===0&&TL.score('M',1)===100&&TL.score('J',.5)===50&&TL.score('S',1)===100);
  const fixture={history:[{day:'2026-09-23',v:[0,.35,0,.5,.8],status:Array(5).fill('scored')},{day:'2026-09-25',v:Array(5).fill(null),status:Array(5).fill('unavailable')},{day:'2026-10-02',v:[.1,.4,.1,.6,.9],status:Array(5).fill('scored')}]};
  const sparse=TL.layout(fixture,data.as_of,14,700);
  check('irregular evaluation dates are retained exactly',sparse.series.every(s=>s.points.map(p=>p.day).join(',')==='2026-09-23,2026-10-02'));
  check('missing evaluation splits each line',sparse.series.every(s=>s.segments.length===2&&s.segments.every(a=>a.length===1)));
  check('calendar spacing is preserved',Math.abs(sparse.series[0].points[1].x-sparse.series[0].points[0].x-(sparse.frame.right-sparse.frame.left)*9/13)<1e-9);
  let total=0;
  for(const r of data.records)for(const n of [14,30,60]){
    const l=TL.layout(r,data.as_of,n,700);const expectedStart=new Date(Date.parse(data.as_of+'T00:00:00Z')-(n-1)*86400000).toISOString().slice(0,10);
    check(r.code+' '+n+' date range',l.from===expectedStart&&l.end===data.as_of&&l.series.length===5);
    for(let k=0;k<5;k++){
      const s=l.series[k],source=r.history.filter(h=>h.day>=expectedStart&&h.day<=data.as_of&&h.v[k]!=null&&h.status[k]==='scored');
      check(r.code+' '+n+' '+s.key+' source points',s.points.length===source.length&&s.points.every((p,i)=>p.day===source[i].day&&p.value===source[i].v[k]));
      check(r.code+' '+n+' '+s.key+' calendar geometry',s.points.every(p=>Math.abs(p.x-(l.frame.left+(Date.parse(p.day)-Date.parse(expectedStart))/(Date.parse(data.as_of)-Date.parse(expectedStart))*(l.frame.right-l.frame.left)))<1e-9));
      check(r.code+' '+n+' '+s.key+' bounded fixed display scale',s.points.every(p=>Number.isFinite(p.score)&&p.score>=0&&p.score<=100));
      total+=s.points.length;
    }
    const svg=TL.svg(l,data.as_of),markers=(svg.match(/class="timeline-point"/g)||[]).length;
    check(r.code+' '+n+' every valid evaluation has a marker',markers===l.series.reduce((sum,s)=>sum+s.points.length,0));
    check(r.code+' '+n+' one line per dimension',(svg.match(/class="timeline-path"/g)||[]).length===5);
  }
  const nvda=embedded.records.find(r=>r.code==='US.NVDA'),l=TL.layout(nvda,data.as_of,60,700),hidden=TL.layout(nvda,data.as_of,60,700,['G']);
  check('hiding dimensions does not change date or value coordinates',l.series.every((s,k)=>JSON.stringify(s.points)===JSON.stringify(hidden.series[k].points)));
  check('hiding dimensions changes visibility only',hidden.series.filter(s=>s.visible).map(s=>s.key).join(',')==='G');
  for(const n of [14,30]){const shorter=TL.layout(nvda,data.as_of,n,700);check(n+'-day view crops existing values',shorter.series.every((s,k)=>s.points.every(p=>l.series[k].points.some(q=>q.day===p.day&&q.value===p.value&&q.score===p.score))));}
  for(const width of [700,390,276,256]){const v=TL.layout(nvda,data.as_of,60,width);check('all plot geometry fits width '+width,v.series.every(s=>s.points.every(p=>p.x>=42-1e-8&&p.x<=width-13+1e-8&&p.y>=v.frame.top-1e-8&&p.y<=v.frame.bottom+1e-8)));}
  check('original model and source files unchanged',data.input_sha256.model===sha(path.join(out,'model.py'))&&data.input_sha256.build===sha(path.join(out,'build.py'))&&data.input_sha256.protocol===sha(path.join(out,'PROTOCOL.md'))&&data.input_sha256.v1_results===sha(path.join(out,'../ai_trend_quadrant_v1/results.json')));
  const cache=path.resolve(out,'../../.cache/ai_trend_quadrant_v2');fs.mkdirSync(cache,{recursive:true});
  // Serialize the exact chart generator used by index.html for static SVG inspection.
  for(const [name,width] of [['timeline-desktop',700],['timeline-mobile',276]]){
    const svg=TL.svg(TL.layout(nvda,data.as_of,60,width),data.as_of).replace('<svg ', '<svg style="font-family:system-ui,sans-serif;font-size:12px;background:#fff" ').replaceAll('var(--fg)','#262626').replaceAll('var(--secondary)','#626262').replaceAll('var(--random)','#777').replaceAll('var(--line)','#c7c7c7').replaceAll('var(--bg)','#fff');
    fs.writeFileSync(path.join(cache,name+'.svg'),svg);
  }
  const report={run_id:data.run_id,view_revision:'five_traits_comparison_three_columns_v1',component:'five_traits_timeline_v1',verified_at:new Date().toISOString(),verification_kind:'offline_source_data_and_svg_geometry',live_browser_verified:false,point_values_checked:total,checks,errors:[],files:{index:sha(path.join(out,'index.html')),timeline:sha(path.join(out,'timeline.js'))}};
  fs.writeFileSync(path.join(out,'timeline_verification.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:checks.length,point_values_checked:total,live_browser_verified:false}));
}
main();
