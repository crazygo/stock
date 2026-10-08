/* Display-only half-range log-price fits. Historical descriptors remain in core.js. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.SlopeOverlay=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const signed=v=>(v>=0?'+':'')+v.toFixed(3);
function fit(days,lookup,start,stop,known){
 const dates=days.slice(start,stop),rows=dates.map(day=>lookup.get(day));
 const missing=dates.filter((day,i)=>!rows[i]||!Number.isFinite(rows[i][4])||rows[i][4]<=0);
 const base={start,stop,first:dates[0]||null,last:dates.at(-1)||null,n:dates.length,missing};
 if(!known)return {...base,beta:null,reason:'日历未核验'};
 if(dates.length<2)return {...base,beta:null,reason:'不足2个收盘'};
 if(missing.length)return {...base,beta:null,reason:'缺'+missing.length+'日收盘'};
 const values=rows.map(b=>Math.log(b[4])),mid=(values.length-1)/2,mean=values.reduce((a,b)=>a+b,0)/values.length;
 let numerator=0,denominator=0;
 for(let i=0;i<values.length;i++){numerator+=(i-mid)*(values[i]-mean);denominator+=(i-mid)**2;}
 const beta=numerator/denominator,alpha=mean-beta*mid;
 return {...base,beta,alpha,logPercent:beta*100,reason:null};
}
function describe(layout){
 // Split the whole daily axis, never the visible/panned portion or the weekly background.
 const n=layout.days.length,cut=Math.floor(n/2),lookup=new Map(layout.source.map(b=>[b[0],b]));
 const known=!!layout.calendar&&layout.from>=layout.calendar.start&&layout.end<=layout.calendar.end;
 const halves=[fit(layout.days,lookup,0,cut,known),fit(layout.days,lookup,cut,n,known)];
 const delta=halves.every(h=>h.beta!=null)?halves[1].logPercent-halves[0].logPercent:null;
 const direction=delta==null?'覆盖不足':Math.abs(delta)<.0005?'趋势速度近似不变':delta>0?(halves[1].beta>0?'近期提速':'下行减缓'):(halves[1].beta<0?'下行加快':'近期放缓');
 return {halves,cut,n,delta,direction,splitBefore:layout.days[cut]||null};
}
function svg(original,layout,description,options={}){
 const l=layout,{frame:f}=l,d=description,clip=esc(options.id||'slope-clip'),out=[];
 const x=i=>f.left+((i+.5)/Math.max(1,d.n)-l.u0)/(l.u1-l.u0)*(f.right-f.left);
 const split=f.left+(d.cut/Math.max(1,d.n)-l.u0)/(l.u1-l.u0)*(f.right-f.left);
 out.push(`<g class="slope-overlay" pointer-events="none"><defs><clipPath id="${clip}"><rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}"/></clipPath></defs>`);
 if(d.n>1&&split>=f.left&&split<=f.right)out.push(`<line class="slope-midpoint" data-before="${esc(d.splitBefore)}" x1="${split}" x2="${split}" y1="${f.top}" y2="${f.bottom}" stroke="var(--secondary)" stroke-dasharray="2 5" opacity=".65"/>`);
 d.halves.forEach((h,k)=>{
  if(h.beta==null)return;
  // Log-linear fits become exponential paths on the existing, unchanged linear price axis.
  const points=[];
  for(let i=0;i<h.n;i++)points.push(`${i?'L':'M'}${x(h.start+i).toFixed(3)},${l.y(Math.exp(h.alpha+h.beta*i)).toFixed(3)}`);
  out.push(`<path class="slope-fit slope-fit-${k?'back':'front'}" data-half="${k}" data-from="${h.first}" data-to="${h.last}" data-slope="${h.logPercent}" d="${points.join(' ')}" fill="none" stroke="var(--fg)" stroke-width="2.2" ${k?'':'stroke-dasharray="6 4"'} clip-path="url(#${clip})"/>`);
 });
 out.push('</g>');
 const y=l.height+11;
 d.halves.forEach((h,k)=>{
  const left=8+k*l.width/2,label=`${k?'后':'前'}50% ${h.beta==null?'—':signed(h.logPercent)+'%'}`;
  const note=`日线${k?'后':'前'}半 ${h.first||'—'}—${h.last||'—'} · ${h.n}个交易日收盘 · ${h.reason||'对数拟合斜率 '+signed(h.logPercent)+'%/交易日'}`;
  out.push(`<g class="slope-label" data-half="${k}" data-from="${h.first||''}" data-to="${h.last||''}" data-slope="${h.logPercent??''}" aria-label="${esc(note)}"><title>${esc(note)}</title><line x1="${left}" x2="${left+15}" y1="${y-4}" y2="${y-4}" stroke="var(--fg)" stroke-width="2.2" ${k?'':'stroke-dasharray="5 3"'}/><text x="${left+20}" y="${y}" style="font-size:11px;font-variant-numeric:tabular-nums">${esc(label)}</text></g>`);
 });
 const summary=d.delta==null?'斜率覆盖不足 · '+d.halves.filter(h=>h.reason).map(h=>h.reason).join(' / '):`后−前 ${signed(d.delta)} pp/日 · ${d.direction}`;
 out.push(`<text class="slope-delta" data-delta="${d.delta??''}" x="${l.width/2}" y="${l.height+27}" text-anchor="middle" style="font-size:11px;font-variant-numeric:tabular-nums">${esc(summary)}</text>`);
 const height=l.height+34;
 // Reserve the original card height; both candles and annotations share the same projection.
 return original.replace(`viewBox="0 0 ${l.width} ${l.height}"`,`viewBox="0 0 ${l.width} ${height}" preserveAspectRatio="none" style="height:${l.height}px"`).replace(`height="${l.height}" role="img"`,`height="${height}" role="img"`).replace('</svg>',out.join('')+'</svg>');
}
return {describe,svg};
});
