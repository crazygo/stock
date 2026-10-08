/* Offline price projection checks. No browser access or live market requests. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm');
const PC=require('./price_chart.js'),OUT=__dirname,P=JSON.parse(fs.readFileSync(path.join(OUT,'prices.json'))),D=JSON.parse(fs.readFileSync(path.join(OUT,'results.json'))),checks=[];
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function check(name,ok){checks.push({name,passed:!!ok});if(!ok)throw Error(name);}
const fixture={code:'fixture',currency:'USD',bars:[['2026-06-29',10,14,9,11],['2026-06-30',11,15,10,12],['2026-07-01',12,13,8,9],['2026-07-02',9,11,7,10],['2026-07-06',10,13,9,12],['2026-07-07',12,14,10,11]]};
const cal={start:'2026-06-29',end:'2026-07-12',days:['2026-06-29','2026-06-30','2026-07-01','2026-07-02','2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10'],early_close:[]};
const ws=PC.weeks(fixture,cal,'2026-07-07');
check('weekly first open, max high, min low, last close',ws[0].o===10&&ws[0].h===15&&ws[0].l===7&&ws[0].c===10);
check('official holiday four-session week is complete',ws[0].complete&&ws[0].bars.length===4&&ws[0].expected.length===4);
check('current incomplete week is identified',ws[1].partial&&!ws[1].complete&&ws[1].c===11);
const historical=PC.weeks(fixture,cal,'2026-07-01')[0];check('selected historical end excludes later prices',historical.partial&&historical.c===9&&historical.l===8&&historical.bars.length===3);
const gap={...fixture,bars:fixture.bars.filter(b=>b[0]!=='2026-06-30')},gw=PC.weeks(gap,cal,'2026-07-07')[0];check('missing official session is not a holiday or completed week',gw.missing.join(',')==='2026-06-30'&&!gw.complete);
const gl=PC.layout(gap,cal,'2026-06-29','2026-07-07',700);check('missing session keeps a slot without synthetic OHLC',gl.visible.find(d=>d.day==='2026-06-30').bar===null);
check('holiday and weekend occupy no main-chart slot',!gl.days.includes('2026-07-03')&&!gl.days.includes('2026-07-04')&&!gl.days.includes('2026-07-05'));
check('calendar gaps are not interpolated into mini OHLC',(PC.mini(gap,'2024-07-07','2026-07-07','2026-06-29','2026-07-07',700).match(/class="mini-ohlc"/g)||[]).length===gap.bars.length);
check('unverified calendar cannot claim a complete week',PC.weeks(fixture,null,'2026-07-07').every(w=>!w.complete&&!w.known));
check('empty market remains without fabricated candles',PC.layout({bars:[],currency:'JPY'},null,'2026-08-08',P.as_of,276).visible.length===0);
check('snapshot identity and cutoff match',P.source_run_id===D.run_id&&P.as_of===D.as_of&&P.source_results_sha256===sha(path.join(OUT,'results.json')));
let total=0,weeksChecked=0;
for(const r of P.records){
 const source=D.provenance.find(p=>p.code===r.code);if(source)check('frozen OHLC input '+r.code,r.sha256===source.sha256&&sha(path.resolve(OUT,'../..',r.path))===source.sha256);
 check('valid cutoff, listing and OHLC '+r.code,r.bars.every((b,i)=>b[0]<=P.as_of&&b[0]>=P.mini_start&&(!r.listing_date||b[0]>=r.listing_date)&&(!i||b[0]>r.bars[i-1][0])&&b.slice(1).every(v=>Number.isFinite(v)&&v>0)&&b[2]>=Math.max(b[1],b[3],b[4])&&b[3]<=Math.min(b[1],b[2],b[4])));
 total+=r.bars.length;const calendar=P.calendars[r.market];
 if(calendar)check('captured dates inside verified calendar are official sessions '+r.code,r.bars.filter(b=>b[0]>=calendar.start&&b[0]<=calendar.end).every(b=>calendar.days.includes(b[0])));
 for(const w of PC.weeks(r,calendar,P.as_of)){
  const bars=r.bars.filter(b=>PC.weekStart(b[0])===w.day&&b[0]<=P.as_of);
  check('independent weekly reduction '+r.code+' '+w.day,w.o===bars[0][1]&&w.c===bars.at(-1)[4]&&w.h===Math.max(...bars.map(b=>b[2]))&&w.l===Math.min(...bars.map(b=>b[3])));
  if(w.complete)check('completed weekly coverage '+r.code+' '+w.day,w.expected.every(day=>bars.some(b=>b[0]===day))&&w.expected.every(day=>day<=P.as_of));
  weeksChecked++;
 }
 for(const n of [30,60,90,180]){
  const from=PC.start(P.as_of,n),l=PC.layout(r,calendar,from,P.as_of,276);
  check('minimum daily spacing '+r.code+' '+n,l.spacing>=5.5-1e-9&&l.weeklySpacing>=5.5-1e-9&&l.visible.length<=l.capacity&&l.weekly.length<=l.capacity);
  check('exact daily source and price geometry '+r.code+' '+n,l.visible.every(d=>!d.bar||r.bars.some(b=>JSON.stringify(b)===JSON.stringify(d.bar)))&&l.visible.every(d=>d.x>=l.frame.left&&d.x<=l.frame.right));
  check('shared weekly and daily price domain '+r.code+' '+n,l.weekly.filter(w=>w.o!=null).every(w=>l.y(w.h)>=l.frame.top-1e-7&&l.y(w.l)<=l.frame.bottom+1e-7)&&l.visible.every(d=>!d.bar||l.y(d.bar[2])>=l.frame.top-1e-7&&l.y(d.bar[3])<=l.frame.bottom+1e-7));
  const s=PC.svg(l),noWeeks=PC.svg(l,false);check('daily candle count retained on toggle '+r.code+' '+n,(s.match(/class="daily-candle"/g)||[]).length===l.visible.filter(d=>d.bar).length&&(noWeeks.match(/class="daily-candle"/g)||[]).length===l.visible.filter(d=>d.bar).length&&!noWeeks.includes('class="weekly-candle"'));
  check('weekly candles and date ticks without repeated headers '+r.code+' '+n,(s.match(/class="weekly-candle"/g)||[]).length===l.weekly.filter(w=>w.o!=null).length&&s.includes(`y="${l.frame.top-13}"`)&&!s.includes('下轴 · 日线')&&!s.includes('上轴 · 周线')&&!s.includes('上下轴日期不同'));
  // Every selected date can be reached through the explicit pan range; none is discarded for width.
  const reached=new Set(),weeksReached=new Set();for(let offset=0;offset<=l.maxOffset;offset+=Math.max(1,l.count)){const v=PC.layout(r,calendar,from,P.as_of,276,offset);v.visible.forEach(d=>reached.add(d.day));v.weekly.forEach(w=>weeksReached.add(w.day));}
  const final=PC.layout(r,calendar,from,P.as_of,276,l.maxOffset);final.visible.forEach(d=>reached.add(d.day));final.weekly.forEach(w=>weeksReached.add(w.day));check('all dates on both independent axes reachable '+r.code+' '+n,l.days.every(d=>reached.has(d))&&l.allWeeks.every(w=>weeksReached.has(w.day)));
  check('independent ranges expand exactly sevenfold '+r.code+' '+n,l.dailyDays===n&&l.weeklyDays===7*n&&l.weeklyFrom===PC.start(P.as_of,7*n)&&l.from===from);
  check('weekly data obeys expanded start and shared end '+r.code+' '+n,l.allWeeks.every(w=>w.bars.every(b=>b[0]>=l.weeklyFrom&&b[0]<=P.as_of)));
  check('independent candle coordinates '+r.code+' '+n,l.visible.every(d=>Math.abs(d.x-(l.frame.left+(d.u-l.u0)/(l.u1-l.u0)*(l.frame.right-l.frame.left)))<1e-9)&&l.weekly.every(w=>Math.abs(w.x-(l.frame.left+(w.u-l.u0)/(l.u1-l.u0)*(l.frame.right-l.frame.left)))<1e-9));
 }
}
check('all securities and unavailable retained',P.records.length===166&&P.records.filter(r=>r.bars.length===0).length===3);
const html=fs.readFileSync(path.join(OUT,'index.html'),'utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];new vm.Script(script);check('single-file JavaScript compiles',true);
const begin=script.indexOf('const P=')+8,embedded=JSON.parse(script.slice(begin,script.indexOf(',PC=PriceOverlay',begin)));check('all embedded prices exactly match projection',JSON.stringify(P)===JSON.stringify(embedded));
check('native range and shortcut controls', ['price-from','price-to','price-week-from','price-hover','price-pan-slider','price-inspect','price-mini','show-week'].every(id=>html.includes('id="'+id+'"'))&&[30,60,90,180].every(n=>html.includes('data-price-days="'+n+'"')));
check('no wheel or drag zoom handlers',!html.includes('onwheel')&&!html.includes("addEventListener('wheel'")&&!html.includes('onpointerdown')&&!html.includes('ondrag'));
const nvda=P.records.find(r=>r.code==='US.NVDA'),current=PC.weeks(nvda,P.calendars.US,P.as_of).at(-1);check('Oct 6 current week is a two-session incomplete week',current.day==='2026-10-05'&&current.partial&&current.bars.length===2);
const first=PC.layout(nvda,P.calendars.US,PC.start(P.as_of,180),P.as_of,276,0),last=PC.layout(nvda,P.calendars.US,PC.start(P.as_of,180),P.as_of,276,null);check('pan moves daily date window without losing source',first.visible[0].day<last.visible[0].day&&first.days.length===last.days.length);
const double=PC.layout(nvda,P.calendars.US,PC.start(P.as_of,30),P.as_of,700);check('30 calendar days expands to 210 calendar days',double.from==='2026-09-07'&&double.weeklyFrom==='2026-03-11'&&double.dailyDays===30&&double.weeklyDays===210);
const mid=PC.probe(double,.5);check('one horizontal position returns two different dates',mid.daily.day!==mid.weekly.periodFrom&&mid.daily.day>'2026-09-01'&&mid.weekly.periodFrom<'2026-08-01');
check('daily and weekly right boundaries use the same cutoff',double.end===P.as_of&&double.allWeeks.at(-1).periodEnd===P.as_of&&PC.probe(double,1).daily.day===P.as_of);
const clipped=PC.weeks(fixture,cal,'2026-07-07','2026-06-30')[0];check('expanded start clips boundary week without earlier prices',clipped.clipped&&!clipped.complete&&clipped.o===11&&clipped.bars.every(b=>b[0]>='2026-06-30'));
const mini=PC.mini(nvda,P.mini_start,P.as_of,double.from,double.end,700,double.weeklyFrom);check('mini marks both requested ranges',mini.includes('class="mini-week-range"'));
const cache=path.resolve(OUT,'../../.cache/ai_trend_quadrant_v2');fs.mkdirSync(cache,{recursive:true});
function colors(s){return s.replace('<svg ','<svg style="font-family:system-ui,sans-serif;font-size:12px" ').replaceAll('var(--bg)','#fff').replaceAll('var(--fg)','#262626').replaceAll('var(--muted)','#efefef').replaceAll('var(--secondary)','#626262').replaceAll('var(--line)','#c7c7c7');}
for(const [name,width]of [['price-desktop',700],['price-mobile',276]]){
 const l=PC.layout(nvda,P.calendars.US,PC.start(P.as_of,30),P.as_of,width);fs.writeFileSync(path.join(cache,name+'.svg'),colors(PC.svg(l,true,1-1e-10)));fs.writeFileSync(path.join(cache,name+'-mini.svg'),colors(PC.mini(nvda,P.mini_start,P.as_of,l.from,l.end,width,l.weeklyFrom)));
}
const report={view_revision:'five_traits_comparison_three_columns_v1',component:'five_traits_dual_range_v1',source_run_id:P.source_run_id,verified_at:new Date().toISOString(),verification_kind:'offline_source_and_svg',live_browser_verified:false,daily_bars_checked:total,weeks_checked:weeksChecked,checks,errors:[],files:{prices:sha(path.join(OUT,'prices.json')),chart:sha(path.join(OUT,'price_chart.js')),index:sha(path.join(OUT,'index.html'))}};
fs.writeFileSync(path.join(OUT,'price_verification.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:checks.length,daily_bars_checked:total,weeks_checked:weeksChecked,live_browser_verified:false}));
