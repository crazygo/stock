/* Daily-close morphology, descriptive and causal. No future-price outcomes. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.DailyPatterns=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const VERSION='daily_close_patterns_v2',N=10,DEFAULTS=Object.freeze({vDip:.05});
const TYPES=[{id:'rise',short:'▲',name:'单边拉升'},{id:'v',short:'V',name:'V型深弹'},{id:'pullback',short:'P',name:'冲高回落'},{id:'step',short:'S',name:'阶梯中继'}];
const QUADRANTS=[{id:'up_low',name:'上行 · 低波动'},{id:'up_high',name:'上行 · 高波动'},{id:'down_low',name:'下行 · 低波动'},{id:'down_high',name:'下行 · 高波动'}];
function efficiency(a){let distance=0;for(let i=1;i<a.length;i++)distance+=Math.abs(a[i]-a[i-1]);return distance?(a.at(-1)-a[0])/distance:0;}
function parameters(options={}){const vDip=options.vDip??DEFAULTS.vDip;if(!Number.isFinite(vDip)||vDip<=0||vDip>1)throw new RangeError('V dip threshold must be greater than 0 and at most 1');return {vDip};}
function classify(closes,options={}){
 const {vDip}=parameters(options);
 if(closes.length!==N+1||!closes.every(v=>Number.isFinite(v)&&v>0))return {matches:[],reason:'invalid_window'};
 const r=closes.map(v=>v/closes[0]-1),end=r.at(-1),min=Math.min(...r),max=Math.max(...r),minPos=r.indexOf(min)/N,maxPos=r.indexOf(max)/N,eff=efficiency(closes);
 let positive=0;for(let i=1;i<closes.length;i++)if(closes[i]>closes[i-1])positive++;
 const metrics={end,min,max,minPos,maxPos,eff,positive},matches=[];
 if(end<.05-1e-12)return {matches,metrics};
 if(min>=-.005-1e-12&&maxPos>=.75&&eff>=.65&&positive/N>=.6)matches.push('rise');
 if(minPos<=.45&&min<=-vDip+1e-12)matches.push('v');
 if(maxPos<=.55&&max-end>=.015-1e-12)matches.push('pullback');
 let platform=null;
 for(let a=2;a<=N-4&&!platform;a++)for(let b=a+2;b<=N-2;b++){
  const first=closes[a]/closes[0]-1,second=closes.at(-1)/closes[b]-1,band=Math.max(...closes.slice(a,b+1))/Math.min(...closes.slice(a,b+1))-1;
  const drift=Math.abs(closes[b]/closes[a]-1);
  if(first>=.025-1e-12&&second>=.025-1e-12&&band<=.015+1e-12&&drift<=.0075+1e-12&&efficiency(closes.slice(0,a+1))>=.55&&efficiency(closes.slice(b))>=.55){platform={a,b,first,second,band,drift};break;}
 }
 if(platform)matches.push('step');return {matches,metrics:{...metrics,platform}};
}
function completeWindow(bars,calendar){
 if(!calendar)return 'unverified_calendar';
 if(bars[0][0]<calendar.start||bars.at(-1)[0]>calendar.end)return 'unverified_calendar';
 const expected=calendar.days.filter(d=>d>=bars[0][0]&&d<=bars.at(-1)[0]);
 return expected.length===bars.length&&expected.every((d,i)=>d===bars[i][0])?'scored':'missing_sessions';
}
function scan(record,calendar,from,end,options={}){
 const config=parameters(options);
 const bars=record.bars.filter(b=>b[0]<=end),evaluations=[],events=[];
 for(let i=0;i<bars.length;i++){
  if(bars[i][0]<from)continue;
  if(i<N){evaluations.push({day:bars[i][0],status:'insufficient_history'});continue;}
  const window=bars.slice(i-N,i+1);if(window[0][0]<from){evaluations.push({day:bars[i][0],status:'window_crosses_start'});continue;}
  const status=completeWindow(window,calendar);evaluations.push({day:bars[i][0],status});if(status!=='scored')continue;
  const result=classify(window.map(b=>b[4]),config);if(result.matches.length)events.push({start:window[0][0],day:bars[i][0],matches:result.matches,metrics:result.metrics,closes:window.map(b=>b[4]),days:window.map(b=>b[0])});
 }
 const possible=calendar?calendar.days.filter(d=>d>=from&&d<=end&&(!record.listing_date||d>=record.listing_date)):[];
 const have=new Set(bars.map(b=>b[0])),missing=possible.filter(d=>!have.has(d));
 const counts=Object.fromEntries(TYPES.map(t=>[t.id,events.filter(e=>e.matches.includes(t.id)).length]));
 return {version:VERSION,parameters:config,from,end,events,evaluations,counts,scored:evaluations.filter(e=>e.status==='scored').length,
  missing,unverified:!calendar||from<calendar.start||end>calendar.end,status:!bars.length?'unavailable':evaluations.some(e=>e.status==='scored')?'evaluated':'insufficient_or_unverified'};
}
function quadrant(record,calendar,end){
 const bars=record.bars.filter(b=>b[0]<=end);if(bars.length<121)return {id:null,G:null,V:null,status:'insufficient_history'};
 const expected=calendar?.days.filter(d=>d<=end).at(-1);if(!expected||end>calendar.end)return {id:null,G:null,V:null,status:'unverified_calendar'};
 if(bars.at(-1)[0]!==expected)return {id:null,G:null,V:null,status:'stale'};
 const window=bars.slice(-121),status=completeWindow(window,calendar);if(status!=='scored')return {id:null,G:null,V:null,status};
 const returns=[];for(let i=1;i<window.length;i++)returns.push(Math.log(window[i][4]/window[i-1][4]));
 const G=returns.reduce((a,b)=>a+b,0)/120*252,last=returns.slice(-60),mean=last.reduce((a,b)=>a+b,0)/60,V=Math.sqrt(last.reduce((a,b)=>a+(b-mean)**2,0)/59)*Math.sqrt(252);
 return {id:(G>=0?'up_':'down_')+(V<=.35?'low':'high'),G,V,status:'scored',day:bars.at(-1)[0]};
}
function matchesGroup(record,scanResult,quad,selection){
 const source=!selection.sources.length||selection.sources.some(id=>record.groups.includes(id));
 const chains=!selection.chains?.length||selection.chains.some(id=>(record.chains||[]).includes(id));
 const shape=!selection.patterns.length||selection.patterns.some(id=>scanResult.counts[id]>0);
 const q=!selection.quadrants.length||selection.quadrants.includes(quad.id);
 return source&&chains&&shape&&q;
}
return {VERSION,N,DEFAULTS,TYPES,QUADRANTS,efficiency,classify,completeWindow,scan,quadrant,matchesGroup};
});
