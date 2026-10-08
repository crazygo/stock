/* Pure daily/weekly OHLC projection and SVG, shared by the page and offline checks. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.PriceOverlay=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const DAY=86400000,ts=d=>Date.parse(d+'T00:00:00Z'),iso=t=>new Date(t).toISOString().slice(0,10);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=v=>Number(v).toFixed(v<1?4:v<10?3:2);
function weekStart(day){const t=ts(day),dow=new Date(t).getUTCDay();return iso(t-((dow+6)%7)*DAY);}
function start(end,n){return iso(ts(end)-(n-1)*DAY);}
function weeks(record,calendar,end){
 const groups=new Map();
 for(const b of record.bars){if(b[0]>end)continue;const key=weekStart(b[0]);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(b);}
 return [...groups].map(([day,bars])=>{
  const stop=iso(ts(day)+6*DAY),known=!!calendar&&day>=calendar.start&&stop<=calendar.end;
  const expected=known?calendar.days.filter(d=>d>=day&&d<=stop&&(!record.listing_date||d>=record.listing_date)):[];
  const seen=new Set(bars.map(b=>b[0])),missing=expected.filter(d=>d<=end&&!seen.has(d));
  const partial=known?expected.some(d=>d>end):stop>end;
  const flags=[];if(partial)flags.push('未完周');if(missing.length)flags.push('缺 '+missing.length+' 个交易日');if(!known)flags.push('日历未核验');
  if(record.listing_date&&record.listing_date>day&&record.listing_date<=stop)flags.push('上市首周');
  return {day,stop,first:bars[0][0],last:bars.at(-1)[0],o:bars[0][1],h:Math.max(...bars.map(b=>b[2])),l:Math.min(...bars.map(b=>b[3])),c:bars.at(-1)[4],
   bars,expected,missing,partial,known,flags,complete:known&&!partial&&!missing.length};
 });
}
function layout(record,calendar,from,end,width,offset=null){
 width=Math.max(230,Number(width)||700);
 const frame={left:55,right:width-9,top:59,bottom:282},span=frame.right-frame.left;
 const source=record.bars.filter(b=>b[0]>=from&&b[0]<=end),lookup=new Map(source.map(b=>[b[0],b]));
 // Only an official calendar supplies expected sessions. Outside it, show observed dates, not guessed weekdays.
 const dates=new Set(source.map(b=>b[0]));
 if(calendar)for(const d of calendar.days)if(d>=from&&d<=end&&(!record.listing_date||d>=record.listing_date))dates.add(d);
 const days=[...dates].sort(),capacity=Math.max(1,Math.floor(span/5.5)),count=Math.min(days.length,capacity),maxOffset=Math.max(0,days.length-count);
 const at=offset==null?maxOffset:Math.max(0,Math.min(maxOffset,Math.round(offset))),shown=days.slice(at,at+count),spacing=span/Math.max(1,count);
 const x=i=>frame.left+(i+.5)*spacing,visible=shown.map((day,i)=>({day,x:x(i),bar:lookup.get(day)||null,early:calendar?.early_close.includes(day)||false}));
 const allWeeks=weeks(record,calendar,end);
 const weekly=allWeeks.filter(w=>shown.some(d=>weekStart(d)===w.day)).map(w=>{
  const ids=days.map((d,i)=>weekStart(d)===w.day?i:-1).filter(i=>i>=0),a=ids[0]-at,b=ids.at(-1)-at;
  const left=frame.left+a*spacing,right=frame.left+(b+1)*spacing;
  return {...w,left,right,x:(left+right)/2,labelX:(Math.max(frame.left,left)+Math.min(frame.right,right))/2,clipped:w.first<shown[0]||w.last>shown.at(-1)};
 });
 // Include the full displayed weekly candle in the common price domain. Layer toggles cannot rescale it.
 const highs=[...visible.filter(d=>d.bar).map(d=>d.bar[2]),...weekly.map(w=>w.h)],lows=[...visible.filter(d=>d.bar).map(d=>d.bar[3]),...weekly.map(w=>w.l)];
 const hi=highs.length?Math.max(...highs):1,lo=lows.length?Math.min(...lows):0,pad=Math.max((hi-lo)*.08,hi*.005);
 const domain=[Math.max(0,lo-pad),hi+pad],y=v=>frame.bottom-(v-domain[0])/(domain[1]-domain[0])*(frame.bottom-frame.top);
 return {record,calendar,from,end,width,height:325,frame,days,visible,weekly,allWeeks,source,capacity,count,offset:at,maxOffset,spacing,domain,y};
}
function title(b){return `${b[0]} · 开 ${number(b[1])} / 高 ${number(b[2])} / 低 ${number(b[3])} / 收 ${number(b[4])}`;}
function svg(l,showWeek=true,selected=null){
 const {frame:f,y,width,height}=l,out=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="日K与周K共用价格轴；上轴周一，下轴交易日期"><rect width="100%" height="100%" fill="var(--bg)"/><defs><clipPath id="price-plot-clip"><rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}"/></clipPath></defs>`];
 out.push(`<text x="${f.left}" y="16" fill="var(--secondary)">上轴 · 周一起点</text><text x="${width-9}" y="16" text-anchor="end">${esc(l.record.currency)} · 前复权价格</text>`);
 for(let i=0;i<5;i++){const v=l.domain[0]+(l.domain[1]-l.domain[0])*i/4,yy=y(v);out.push(`<line x1="${f.left}" x2="${f.right}" y1="${yy}" y2="${yy}" stroke="var(--line)"/><text x="${f.left-6}" y="${yy+4}" text-anchor="end">${number(v)}</text>`);}
 out.push(`<rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}" fill="none" stroke="var(--line)"/>`);
 const labelBox=(x,text)=>{const size=text.length*6.5,xx=Math.max(f.left+size/2,Math.min(f.right-size/2,x));return {x:xx,left:xx-size/2,right:xx+size/2};};
 let lastLabel=-Infinity;const finalWeek=l.weekly.at(-1),finalBox=finalWeek?labelBox(finalWeek.labelX,finalWeek.day.slice(5)+(finalWeek.partial?'*':'')):null;
 for(const w of l.weekly){
  const label=w.day.slice(5)+(w.partial?'*':'');
  if(w.left>=f.left)out.push(`<line class="week-boundary" x1="${w.left}" x2="${w.left}" y1="${f.top-25}" y2="${f.bottom}" stroke="var(--secondary)" stroke-width="1.2" opacity=".45"/>`);
  const box=labelBox(w.labelX,label),isLast=w===finalWeek;
  if(isLast||box.left>lastLabel+5&&box.right+5<finalBox.left){out.push(`<text x="${box.x}" y="${f.top-12}" text-anchor="middle">${esc(label)}</text>`);lastLabel=box.right;}
  if(!showWeek)continue;
  const ww=(w.right-w.left)*.78,top=Math.min(y(w.o),y(w.c)),hh=Math.max(1.2,Math.abs(y(w.o)-y(w.c))),flags=[...w.flags,...(w.clipped?['显示边界裁切']:[])],note=flags.length?flags.join(' / '):'完整周';
  const tt=`周 ${w.day} → ${w.last} · ${note} · 开 ${number(w.o)} / 高 ${number(w.h)} / 低 ${number(w.l)} / 收 ${number(w.c)}`;
  out.push(`<g class="weekly-candle" data-week="${w.day}" data-complete="${w.complete}" opacity=".65" clip-path="url(#price-plot-clip)"><title>${esc(tt)}</title><line x1="${w.x}" x2="${w.x}" y1="${y(w.h)}" y2="${y(w.l)}" stroke="var(--secondary)" stroke-width="2"/><rect x="${w.x-ww/2}" y="${top}" width="${ww}" height="${hh}" fill="var(--secondary)" fill-opacity="${w.c>=w.o?'.035':'.20'}" stroke="var(--secondary)" stroke-width="1.4" ${w.flags.length?'stroke-dasharray="4 3"':''}/></g>`);
 }
 for(const d of l.visible){
  if(!d.bar){out.push(`<path class="missing-price-day" data-day="${d.day}" d="M${d.x-2},${f.bottom-8}l4,4m-4,0l4,-4" stroke="var(--secondary)"><title>${d.day} · 官方交易日，无有效OHLC，保持缺失</title></path>`);continue;}
  const b=d.bar,bw=Math.min(7,l.spacing*.58),up=b[4]>=b[1],yy=Math.min(y(b[1]),y(b[4])),bh=Math.max(1,Math.abs(y(b[1])-y(b[4])));
  out.push(`<g class="daily-candle" data-day="${d.day}"><title>${esc(title(b)+(d.early?' · 半日市':''))}</title><line x1="${d.x}" x2="${d.x}" y1="${y(b[2])}" y2="${y(b[3])}" stroke="var(--fg)" stroke-width="1"/><rect x="${d.x-bw/2}" y="${yy}" width="${bw}" height="${bh}" fill="${up?'var(--bg)':'var(--fg)'}" stroke="var(--fg)" stroke-width="1"/></g>`);
  if(selected===d.day)out.push(`<line x1="${d.x}" x2="${d.x}" y1="${f.top}" y2="${f.bottom}" stroke="var(--fg)" stroke-dasharray="2 4" opacity=".5"/>`);
  out.push(`<rect class="price-day-hit" data-day="${d.day}" x="${d.x-l.spacing/2}" y="${f.top}" width="${l.spacing}" height="${f.bottom-f.top}" fill="transparent"><title>${esc(title(b))}</title></rect>`);
 }
 const n=l.visible.length;if(n){let last=-Infinity;const final=labelBox(l.visible.at(-1).x,l.visible.at(-1).day.slice(5));for(let i=0;i<n;i++){const d=l.visible[i],box=labelBox(d.x,d.day.slice(5)),isLast=i===n-1;if(!isLast&&(box.left-last<26||box.right+12>final.left))continue;out.push(`<text x="${box.x}" y="${f.bottom+19}" text-anchor="middle">${d.day.slice(5)}</text>`);last=box.right;}}
 out.push(`<text x="${f.left}" y="${height-2}" fill="var(--secondary)">下轴 · 交易日期（休市折叠）</text>`);
 if(!l.source.length)out.push(`<text x="${(f.left+f.right)/2}" y="${(f.top+f.bottom)/2}" text-anchor="middle">所选范围没有有效日线</text>`);
 return out.join('')+'</svg>';
}
function mini(record,from,end,selectedFrom,selectedEnd,width){
 width=Math.max(230,Number(width)||700);const f={left:8,right:width-8,top:13,bottom:61},x=d=>f.left+(ts(d)-ts(from))/(ts(end)-ts(from))*(f.right-f.left);
 const bars=record.bars.filter(b=>b[0]>=from&&b[0]<=end),hi=bars.length?Math.max(...bars.map(b=>b[2])):1,lo=bars.length?Math.min(...bars.map(b=>b[3])):0,range=Math.max(.001,hi-lo),y=v=>f.bottom-(v-lo)/range*(f.bottom-f.top);
 const out=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} 88" width="${width}" height="88" role="img" aria-label="固定两年真实日级OHLC预览"><rect width="100%" height="100%" fill="var(--bg)"/><rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}" fill="var(--muted)"/><rect x="${x(selectedFrom)}" y="${f.top}" width="${Math.max(1,x(selectedEnd)-x(selectedFrom))}" height="${f.bottom-f.top}" fill="var(--bg)" stroke="var(--fg)"/>`];
 for(const b of bars){const xx=x(b[0]);out.push(`<path class="mini-ohlc" data-day="${b[0]}" d="M${xx},${y(b[2])}V${y(b[3])}M${xx-.8},${y(b[1])}H${xx}M${xx},${y(b[4])}H${xx+.8}" fill="none" stroke="var(--secondary)" stroke-width=".8"/>`);}
 out.push(`<text x="${f.left}" y="81">${from}</text><text x="${f.right}" y="81" text-anchor="end">${end}</text></svg>`);
 return out.join('');
}
function miniTarget(clientX,box,from,end){const ratio=Math.max(0,Math.min(1,(clientX-box.left-8)/(box.width-16)));return iso(ts(from)+Math.round((ts(end)-ts(from))/DAY*ratio)*DAY);}
return {DAY,ts,iso,start,weekStart,weeks,layout,svg,mini,miniTarget,number};
});
