const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert');
const B=require('./core.js'),out=__dirname,old=path.join(out,'../operation_cadence_v1');
const P=JSON.parse(fs.readFileSync(path.join(old,'prices.json'))),M=JSON.parse(fs.readFileSync(path.join(old,'membership.json')));
const previous=require('./archive/three_indicators_v1/core.js');
const close=(a,b,tol=1e-11)=>assert(Math.abs(a-b)<tol,`${a} != ${b}`);
const dates=P.us_sessions.filter(d=>d>='2026-06-01').slice(0,65),fixture=values=>values.map((v,i)=>[dates[i],v,v,v,v,100]);
const oscillation=fixture([100,106,100,106,100,106]),moves=B.moves(oscillation);
assert.equal(moves.length,5);assert.deepEqual(moves.map(x=>x.direction),[1,-1,1,-1,1]);
assert.deepEqual(moves.map(x=>x.anchorIndex),[0,1,2,3,4]);assert.deepEqual(moves.map(x=>x.confirmIndex),[1,2,3,4,5]);
const rising=fixture(Array.from({length:61},(_,i)=>100*1.01**i)),m=B.metrics(rising);
assert.equal(m.moveCount,1);close(m.signedEfficiency,1);assert.equal(m.drawdown,0);
const flat=B.metrics(fixture(Array(25).fill(100)));assert.equal(flat.moveCount,0);assert.equal(flat.signedEfficiency,0);assert.equal(flat.top3Share,null);
const falling=B.metrics(fixture(Array.from({length:61},(_,i)=>100*.99**i)));close(falling.signedEfficiency,-1);assert.equal(falling.moveCount,1);
const scaled=B.metrics(oscillation.map(b=>[b[0],...b.slice(1,5).map(v=>v*13),b[5]])),base=B.metrics(oscillation);
for(const k of ['net','signedEfficiency','vol','drawdown','top3Share'])close(base[k],scaled[k]);assert.equal(base.moveCount,scaled.moveCount);
const Q={...P,records:{'US.TEST':rising}};
let full=B.describe(Q,'US.TEST',dates[0],dates[60]);assert.equal(full.shares.n,41);assert.equal(full.shares.up,1);assert.equal(full.shares.repeated,0);
let sliced=B.describe(Q,'US.TEST',dates[20],dates[50]),mutated=JSON.parse(JSON.stringify(Q));
mutated.records['US.TEST'][0][4]=1e9;mutated.records['US.TEST'][60][4]=.01;
assert.deepEqual(sliced,B.describe(mutated,'US.TEST',dates[20],dates[50]));
const gapQ={...P,records:{'US.TEST':rising.filter((_,i)=>i!==30)}};let gap=B.describe(gapQ,'US.TEST',dates[0],dates[60]);
assert.equal(gap.metrics,null);assert.deepEqual(gap.missing,[dates[30]]);assert.equal(gap.windows.length,20);
assert(gap.windows.every(x=>x.last<dates[30]||x.first>dates[30]));
let one=B.describe({...P,records:{'US.TEST':fixture([100])}},'US.TEST',dates[0],dates[0]);assert.equal(one.metrics,null);assert.equal(one.shares,null);
assert.equal(B.calendar(P,'HK.TEST','2025-01-01','2025-12-31'),null);
let summaries=[],cases=[];
for(const member of M.members){
 for(const days of [30,60,180,365]){
  const end=P.as_of,start=B.shifts(end,1-days),x=B.describe(P,member.code,start,end);
  const baseline=previous.describe(P,member.code,start,end);
  const project=s=>s?Object.fromEntries(['up','repeated','n','upN','repeatedN'].map(k=>[k,s[k]])):null;
  assert.deepEqual(x.shares,project(baseline.shares),'Two retained indicators unchanged '+member.code+' '+days);
  assert.deepEqual(x.metrics,baseline.metrics);assert.deepEqual(x.traits,baseline.traits);
  assert(!('quietVol' in x.settings));assert(x.windows.every(w=>!('quietUp' in w)));
  assert(x.windows.every(w=>w.first>=start&&w.last<=end));
  if(x.metrics){assert(x.metrics.moves.every(v=>v.anchor>=x.observedFirst&&v.confirmed<=x.observedLast));
   x.metrics.moves.forEach((v,i)=>{assert(v.direction>0?v.change>=.05-1e-10:v.change<=-.05+1e-10);if(i)assert(v.direction!==x.metrics.moves[i-1].direction)});
  }
  if(x.shares)for(const key of ['up','repeated'])assert(x.shares[key]>=0&&x.shares[key]<=1);
  cases.push({code:member.code,start,end,metrics:x.metrics,traits:x.traits,shares:x.shares,windows:x.windows,missing:x.missing});
  if(days===60)summaries.push({code:member.code,name:member.name,kind:member.kind,start,end,actual_start:x.observedFirst,actual_end:x.observedLast,status:x.status,metrics:x.metrics,shares:x.shares});
 }
}
const manifest=JSON.parse(fs.readFileSync(path.join(out,'input_manifest.json')));
for(const [name,v] of Object.entries(manifest))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(old,name))).digest('hex'),v.sha256);
fs.writeFileSync(path.join(out,'descriptions.json'),JSON.stringify({version:B.VERSION,settings:B.DEFAULTS,range:['2026-08-09','2026-10-07'],records:summaries},null,2));
fs.writeFileSync(path.join(out,'verification_cases.json'),JSON.stringify(cases));
const report={status:'passed',version:B.VERSION,real_range_cases:cases.length,unchanged_two_indicator_cases:cases.length,checks:['exact alternating threshold moves','one monotone direction segment','flat and downward paths','scale invariance','no prices outside selected interval','missing trading day forbids global path metrics and crossing windows','one-bar and missing cases retained','unknown HK calendar abstains','55 securities including ETFs','all move amplitudes and dates auditable','historical counts and proportions','two retained metrics match archived model across all 220 cases; removed label absent','input file SHA preservation']};
fs.writeFileSync(path.join(out,'calculation_verification.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
