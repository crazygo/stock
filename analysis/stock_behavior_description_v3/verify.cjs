const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert'),vm=require('vm');
const B=require('./core.js'),previous=require('./baseline/core_v2.js'),out=__dirname;
const P=JSON.parse(fs.readFileSync(path.join(out,'../operation_cadence_v1/prices.json'))),members=JSON.parse(fs.readFileSync(path.join(out,'../operation_cadence_v1/membership.json'))).members;
const dates=P.us_sessions.filter(d=>d>='2026-06-01').slice(0,65),fixture=values=>values.map((v,i)=>[dates[i],v,v,v,v,100]);
const near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const simple=fixture([100,110,120,115,110,100,106,110,104,100,106]),f=B.folds(simple);
assert.equal(f.turnCount,4);near(f.frequency,8);assert.equal(f.excluded.length,1);assert.equal(f.amplitudeN,3);near(f.amplitudeMedian,.1);
assert.deepEqual(f.completed.map(w=>[w.startIndex,w.endIndex,w.confirmIndex]),[[2,5,6],[5,7,8],[7,9,10]]);
assert.equal(f.unfinished.anchorIndex,9);assert.equal(f.unfinished.confirmIndex,10);
const alternating=n=>Array.from({length:21},(_,i)=>i%2?n:100),fast=B.folds(fixture(alternating(110))),larger=B.folds(fixture(alternating(130)));
near(fast.frequency,19);near(larger.frequency,fast.frequency);assert(larger.amplitudeMedian>fast.amplitudeMedian);near(fast.amplitudeMedian,.1);near(larger.amplitudeMedian,.3);
const slow=B.folds(fixture(alternating(110).flatMap((v,i)=>i===20?[v]:[v,v])));near(slow.amplitudeMedian,fast.amplitudeMedian);assert(slow.frequency<fast.frequency);
const scaled=B.folds(simple.map(b=>[b[0],...b.slice(1,5).map(v=>v*13),b[5]]));near(scaled.frequency,f.frequency);near(scaled.amplitudeMedian,f.amplitudeMedian);
for(const values of [Array(21).fill(100),Array.from({length:21},(_,i)=>100*1.02**i),Array.from({length:21},(_,i)=>100*.98**i),[100,100,130,130,130]]){const x=B.folds(fixture(values));assert.equal(x.frequency,0);assert.equal(x.amplitudeMedian,null);}
const Q={...P,records:{'US.TEST':simple}},a=B.describe(Q,'US.TEST',dates[2],dates[8]);
const modified=JSON.parse(JSON.stringify(Q));modified.records['US.TEST'][0][4]=1e9;modified.records['US.TEST'][10][4]=.01;assert.deepEqual(a,B.describe(modified,'US.TEST',dates[2],dates[8]));
const gap=B.describe({...P,records:{'US.TEST':simple.filter((_,i)=>i!==5)}},'US.TEST',dates[0],dates[10]);assert.equal(gap.folds,null);
assert.equal(B.describe({...P,records:{'US.TEST':fixture([100])}},'US.TEST',dates[0],dates[0]).folds,null);
assert.equal(B.describe(P,'HK.03006','2025-01-01',P.as_of).folds,null);
const cases=[],summaries=[];
for(const member of members)for(const days of [30,60,180,365]){
 const start=B.shifts(P.as_of,1-days),end=P.as_of,x=B.describe(P,member.code,start,end),old=previous.describe(P,member.code,start,end);
 assert.deepEqual(x.shares,old.shares?{up:old.shares.up,n:old.shares.n,upN:old.shares.upN}:null);
 assert.deepEqual(x.metrics,old.metrics);assert.deepEqual(x.traits,old.traits);
 if(x.folds){const g=x.folds;assert.equal(g.turnCount,Math.max(0,x.metrics.moves.length-1));near(g.frequency,g.turnCount*20/g.returns);assert(g.frequency<20);assert.equal(g.amplitudeN,g.completed.length);assert(g.completed.every(w=>w.start>=start&&w.confirmed<=end&&w.start<=w.end&&w.end<w.confirmed));assert(g.completed.every(w=>w.startIndex>0));assert.deepEqual(g, B.describe(P,member.code,start,end,{window:10}).folds,'Up window cannot affect frequency or amplitude');}
 assert(!('repeated' in (x.shares||{})));
 cases.push({code:member.code,start,end,settings:x.settings,metrics:x.metrics,folds:x.folds,shares:x.shares,traits:x.traits,windows:x.windows,missing:x.missing});
 if(days===60)summaries.push({code:member.code,name:member.name,kind:member.kind,start,end,actual_start:x.observedFirst,actual_end:x.observedLast,status:x.status,shares:x.shares,folds:x.folds,previous_repeated:old.shares?.repeated});
}
for(const file of ['index.html','detail.html']){const h=fs.readFileSync(path.join(out,file),'utf8');for(const s of h.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(s[1]);assert(!h.includes('NaN%'));}
const manifest=JSON.parse(fs.readFileSync(path.join(out,'input_manifest.json')));for(const v of Object.values(manifest))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.resolve(out,'../..',v.path))).digest('hex'),v.sha256);
const saturated=summaries.filter(r=>r.previous_repeated===1),frequencies=saturated.map(r=>r.folds.frequency),amp=saturated.map(r=>r.folds.amplitudeMedian).filter(v=>v!=null);
assert.equal(saturated.length,25);assert(Math.max(...frequencies)>Math.min(...frequencies));assert(new Set(amp).size>20);
fs.writeFileSync(path.join(out,'descriptions.json'),JSON.stringify({version:B.VERSION,settings:B.DEFAULTS,range:[B.shifts(P.as_of,-59),P.as_of],records:summaries},null,2));
fs.writeFileSync(path.join(out,'verification_cases.json'),JSON.stringify(cases));
const report={status:'passed',version:B.VERSION,real_range_cases:cases.length,up_definition_unchanged:cases.length,formerly_saturated:saturated.length,formerly_saturated_frequency_range:[Math.min(...frequencies),Math.max(...frequencies)],formerly_saturated_amplitude_range:[Math.min(...amp),Math.max(...amp)],formerly_saturated_distinct_amplitudes:new Set(amp).size,checks:['same timing different amplitudes separates amplitude only','same amplitude different timing separates frequency only','initial direction excluded from turns','completed extrema vs threshold confirmation','left boundary and unfinished waves excluded from amplitude','monotone/flat/jump paths have zero frequency and unknown amplitude','price scale invariance','no outside-range influence','missing/unknown calendar abstains','220 unchanged up and original raw metrics','up window does not affect fold metrics','immutable input hashes','both embedded page scripts parse'],semantic_alignment:'pending human review'};
fs.writeFileSync(path.join(out,'calculation_verification.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
