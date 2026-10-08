const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert'),crypto=require('crypto');
const out=__dirname,B=require('./core.js'),PC=require('./price_overlay.js'),root=path.join(out,'../..');
const html=fs.readFileSync(path.join(out,'index.html'),'utf8'),data=JSON.parse(html.match(/<script type="application\/json" id="dataset">([\s\S]*?)<\/script>/)[1]);
for(const s of html.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(s[1]);
assert.equal(fs.readFileSync(path.join(out,'price_overlay.js'),'utf8'),fs.readFileSync(path.join(out,'../ai_trend_quadrant_v2/price_chart.js'),'utf8'),'Exact reference renderer snapshot');
assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(out,'price_overlay.js'))).digest('hex'),data.price_renderer_sha256);
const P=data.prices,old=JSON.parse(fs.readFileSync(path.join(out,'descriptions.json'))),calendars={US:{start:'2024-01-01',end:'2026-12-31',days:P.us_sessions,early_close:P.us_early_close},HK:{start:P.hk_calendar.start,end:P.hk_calendar.end,days:B.calendar(P,'HK.CALENDAR',P.hk_calendar.start,P.hk_calendar.end),early_close:[]}};
let aggregateChecks=0,charts=0,fullDefault=0;
for(const r of data.records){
 const previous=old.records.find(x=>x.code===r.code),now=B.describe(P,r.code,'2026-08-09','2026-10-07');assert.deepEqual(now.shares,previous.shares,'Description percentages preserved '+r.code);
 const record={...r,market:r.code.split('.')[0],currency:r.code.startsWith('US.')?'USD':'HKD',listing_date:r.listing_date==='1970-01-01'?null:r.listing_date,bars:P.records[r.code]};
 for(const end of ['2026-09-30','2026-10-07'])for(const n of [30,60,90,180]){
  let l=PC.layout(record,calendars[record.market],PC.start(end,n),end,344,null,{compact:true});assert.equal(l.weeklyDays,n*7);assert.equal(l.weeklyFrom,PC.start(end,n*7));
  assert(l.source.every(b=>b[0]<=end));assert(l.spacing>=5.5-1e-10&&l.weeklySpacing>=5.5-1e-10);
  for(const w of l.allWeeks){let raw=record.bars.filter(b=>b[0]>=w.periodFrom&&b[0]<=w.periodEnd&&PC.weekStart(b[0])===w.day);
   if(!raw.length){assert.equal(w.o,null);continue}
   assert.deepEqual([w.o,w.h,w.l,w.c],[raw[0][1],Math.max(...raw.map(b=>b[2])),Math.min(...raw.map(b=>b[3])),raw.at(-1)[4]]);assert(raw.every(b=>b[0]<=end));aggregateChecks++;
  }
  let left=PC.layout(record,calendars[record.market],l.from,end,344,0,{compact:true}),right=l;
  assert.equal(left.u0,0);assert.equal(right.u1,1);assert.equal(right.offset,right.maxOffset);
  let withWeek=PC.svg(l,true,null,{id:r.code.replace('.','-')}),withoutWeek=PC.svg(l,false);
  assert(withWeek.includes('weekly-candle'));assert(!withoutWeek.includes('class="weekly-candle"'));assert(withWeek.includes('daily-candle'));
  const knownWeek=l.allWeeks.find(w=>w.known&&w.o!=null&&w.partial);if(end==='2026-10-07')assert(knownWeek?.flags.includes('未完周'));
  if(n===60&&end==='2026-10-07')fullDefault+=l.source.length;
  charts++;
 }
}
assert.equal(data.records.length,55);assert(!('intraday' in data),'Wall excludes hourly payload');
for(const [name,v] of Object.entries(data.manifest))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,v.path))).digest('hex'),v.sha256,name);
const report={status:'passed',view:data.view_version,exact_reference_renderer:true,retained_description_records_unchanged:55,chart_range_checks:charts,weekly_ohlc_checks:aggregateChecks,default_daily_bars:fullDefault,checks:['embedded script syntax','same reference renderer bytes and SHA','55 two-indicator records match current model','daily N / weekly7N independent axes','weekly aggregation from original daily OHLC','no end-after data','>=5.5px in both layers','all positions reachable by explicit pan','week visibility does not change axis/domain','unfinished October7 week flagged','offline daily-only payload','immutable input hashes']};
fs.writeFileSync(path.join(out,'wall_verification.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
