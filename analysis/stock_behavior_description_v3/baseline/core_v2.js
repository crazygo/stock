/* Historical path descriptions only. No future events, fitted probabilities or orders. */
(function(root){
'use strict';
const VERSION='historical_path_description_v2';
const DEFAULTS={threshold:.05,window:20,efficiency:.20};
const sum=a=>a.reduce((s,x)=>s+x,0), mean=a=>a.length?sum(a)/a.length:null;
const std=a=>a.length>1?Math.sqrt(sum(a.map(x=>(x-mean(a))**2))/(a.length-1)):null;
const median=a=>{if(!a.length)return null;const b=[...a].sort((x,y)=>x-y);return (b[Math.floor((b.length-1)/2)]+b[Math.floor(b.length/2)])/2};
function shifts(day,n){let d=new Date(day+'T00:00:00Z');d.setUTCDate(d.getUTCDate()+n);return d.toISOString().slice(0,10)}
function calendar(prices,code,start,end){
 if(code.startsWith('US.'))return prices.us_sessions.filter(d=>d>=start&&d<=end);
 const c=prices.hk_calendar;
 if(!c||start<c.start||end>c.end)return null;
 let days=[];const holiday=new Set(c.holidays);
 for(let d=start;d<=end;d=shifts(d,1)){let w=new Date(d+'T00:00:00Z').getUTCDay();if(w!==0&&w!==6&&!holiday.has(d))days.push(d)}return days;
}
/* Each move is confirmed only on crossing q from the running extreme.
   The extreme/confirmation span is retrospective evidence, never an entry/exit. */
function moves(rows,q=.05){
 if(rows.length<2)return [];
 const p=rows.map(b=>b[4]);let state=0,lo=0,hi=0,ext=0,out=[];
 const push=(a,b,dir)=>out.push({anchor:rows[a][0],confirmed:rows[b][0],anchorIndex:a,confirmIndex:b,
  from:p[a],to:p[b],direction:dir,change:p[b]/p[a]-1,days:b-a});
 for(let i=1;i<p.length;i++){
  if(state===0){if(p[i]<=p[lo])lo=i;if(p[i]>=p[hi])hi=i;
   if(p[i]/p[lo]-1>=q-1e-12){push(lo,i,1);state=1;ext=i}
   else if(p[i]/p[hi]-1<=-q+1e-12){push(hi,i,-1);state=-1;ext=i}
  }else if(state===1){if(p[i]>=p[ext])ext=i;else if(p[i]/p[ext]-1<=-q+1e-12){push(ext,i,-1);state=-1;ext=i}}
  else{if(p[i]<=p[ext])ext=i;else if(p[i]/p[ext]-1>=q-1e-12){push(ext,i,1);state=1;ext=i}}
 }return out;
}
function metrics(rows,q=.05){
 if(rows.length<2)return null;
 let p=rows.map(b=>b[4]),r=p.slice(1).map((v,i)=>Math.log(v/p[i])),total=sum(r),travel=sum(r.map(Math.abs));
 let peak=p[0],dd=0,ddFrom=0,ddTo=0,peakIndex=0;
 p.forEach((v,i)=>{if(v>peak){peak=v;peakIndex=i}let d=1-v/peak;if(d>dd){dd=d;ddFrom=peakIndex;ddTo=i}});
 let positive=r.map((x,i)=>({value:x,day:rows[i+1][0],from:rows[i][0]})).filter(a=>a.value>0).sort((a,b)=>b.value-a.value);
 let posTotal=sum(positive.map(a=>a.value)),legs=moves(rows,q);
 const intervals=legs.slice(1).map((l,i)=>l.confirmIndex-legs[i].confirmIndex);
 return {bars:rows.length,returns:r.length,first:rows[0][0],last:rows.at(-1)[0],net:p.at(-1)/p[0]-1,
  signedEfficiency:travel>1e-14?total/travel:0,vol:std(r),drawdown:dd,drawdownFrom:rows[ddFrom][0],drawdownTo:rows[ddTo][0],
  moveCount:legs.length,roundTrips:Math.floor(legs.length/2),medianConfirmationGap:median(intervals),moves:legs,
  top3Share:posTotal?sum(positive.slice(0,3).map(a=>a.value))/posTotal:null,topDays:positive.slice(0,3),
  positiveDays:positive.length,logReturns:r};
}
function four(r){
 let G=null,V=null,M=null,J=null;
 if(r.length>=120)G=mean(r.slice(-120))*252;
 if(r.length>=60){let a=r.slice(-60);V=std(a)*Math.sqrt(252);let l=a.slice(0,-1),h=a.slice(1),ml=mean(l),mh=mean(h);
  let den=Math.sqrt(sum(l.map(x=>(x-ml)**2))*sum(h.map(x=>(x-mh)**2)));M=den>1e-20?sum(l.map((x,i)=>(x-ml)*(h[i]-mh)))/den:null;
  let en=a.map(x=>(x-mean(a))**2),tot=sum(en);J=tot>1e-20?sum(en.sort((x,y)=>y-x).slice(0,6))/tot:null;
 }return [G,V,M,J];
}
function traits(r){
 let v=four(r),S=null;
 if(r.length>=140){let seq=[];
  for(let i=20;i>=0;i--){let f=four(r.slice(0,r.length-i));seq.push([Math.tanh(f[0]/.5),Math.tanh((f[1]-.35)/.25),f[2],Math.tanh((f[3]-.55)/.2)])}
  if(seq.every(a=>a.every(x=>x!=null&&Number.isFinite(x)))){let d=0;for(let i=1;i<seq.length;i++)for(let j=0;j<4;j++)d+=Math.abs(seq[i][j]-seq[i-1][j]);S=Math.exp(-10*d/(20*4)/2)}
 }return [...v,S];
}
function describe(prices,code,start,end,settings={}){
 const cfg={...DEFAULTS,...settings};let rows=(prices.records[code]||[]).filter(b=>b[0]>=start&&b[0]<=end);
 if(!rows.length)return {code,start,end,settings:cfg,rows:[],metrics:null,windows:[],shares:null,traits:[null,null,null,null,null],missing:[],calendarKnown:false,status:'no_data'};
 let expected=calendar(prices,code,rows[0][0],rows.at(-1)[0]),present=new Set(rows.map(b=>b[0])),missing=expected?expected.filter(d=>!present.has(d)):[];
 let ordinal=expected?new Map(expected.map((d,i)=>[d,i])):null,known=!!expected;
 let m=known&&!missing.length?metrics(rows,cfg.threshold):null,windows=[];
 if(ordinal)for(let i=cfg.window;i<rows.length;i++){
  let a=rows.slice(i-cfg.window,i+1);if(ordinal.get(a.at(-1)[0])-ordinal.get(a[0][0])!==cfg.window)continue;
  let x=metrics(a,cfg.threshold),up=x.net>0&&x.signedEfficiency>=cfg.efficiency;
  windows.push({first:x.first,last:x.last,net:x.net,efficiency:x.signedEfficiency,vol:x.vol,moves:x.moveCount,
   up,repeated:x.moveCount>=3});
 }
 let count=windows.length,shares=count?{up:windows.filter(x=>x.up).length/count,repeated:windows.filter(x=>x.repeated).length/count,
  n:count,upN:windows.filter(x=>x.up).length,repeatedN:windows.filter(x=>x.repeated).length}:null;
 return {code,start,end,settings:cfg,rows,metrics:m,windows,shares,traits:m?traits(m.logReturns):[null,null,null,null,null],missing,calendarKnown:known,
  observedFirst:rows[0][0],observedLast:rows.at(-1)[0],endpointNet:rows.length>1?rows.at(-1)[4]/rows[0][4]-1:null,
  status:!known?'unknown_calendar':missing.length?'internal_gaps':'described'};
}
function year(prices,code,end){
 let anchor=new Date(end+'T00:00:00Z');anchor.setUTCFullYear(anchor.getUTCFullYear()-1);let start=anchor.toISOString().slice(0,10);
 let x=describe(prices,code,start,end),w=x.windows;
 return {start,end,first:x.observedFirst,last:x.observedLast,metrics:x.metrics,best20:w.length?w.reduce((a,b)=>a.net>b.net?a:b):null,
  adequate:!!x.metrics&&x.observedFirst<=shifts(start,7)&&x.observedLast>=shifts(end,-7),status:x.status};
}
const API={VERSION,DEFAULTS,shifts,calendar,moves,metrics,traits,describe,year};
if(typeof module!=='undefined'&&module.exports)module.exports=API;else root.Behavior=API;
})(typeof window!=='undefined'?window:globalThis);
