/* Pure daily/weekly OHLC projection and SVG, shared by the page and offline checks. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.PriceOverlay=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const DAY=86400000,ts=d=>Date.parse(d+'T00:00:00Z'),iso=t=>new Date(t).toISOString().slice(0,10);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=v=>Number(v).toFixed(v<1?4:v<10?3:2);
function weekStart(day){const t=ts(day),dow=new Date(t).getUTCDay();return iso(t-((dow+6)%7)*DAY);}
function start(end,n){return iso(ts(end)-(n-1)*DAY);}
function weeks(record,calendar,end,from=null){
 const groups=new Map();
 if(from)for(let t=ts(weekStart(from));t<=ts(end);t+=7*DAY)groups.set(iso(t),[]);
 for(const b of record.bars){if(b[0]>end||(from&&b[0]<from))continue;const key=weekStart(b[0]);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(b);}
 return [...groups].map(([day,bars])=>{
  const stop=iso(ts(day)+6*DAY),periodFrom=from&&from>day?from:day,periodEnd=end<stop?end:stop;
  const known=!!calendar&&periodFrom>=calendar.start&&stop<=calendar.end;
  const expected=known?calendar.days.filter(d=>d>=periodFrom&&d<=stop&&(!record.listing_date||d>=record.listing_date)):[];
  const seen=new Set(bars.map(b=>b[0])),missing=expected.filter(d=>d<=end&&!seen.has(d));
  const partial=known?expected.some(d=>d>end):stop>end,clipped=!!from&&from>day;
  const flags=[];if(partial)flags.push('未完周');if(missing.length)flags.push('缺 '+missing.length+' 个交易日');if(!known)flags.push('日历未核验');
  if(clipped)flags.push('起点截断周');if(!bars.length)flags.push('无OHLC');
  if(record.listing_date&&stop<record.listing_date)flags.push('上市前');
  if(record.listing_date&&record.listing_date>day&&record.listing_date<=stop)flags.push('上市首周');
  return {day,stop,periodFrom,periodEnd,first:bars[0]?.[0]||null,last:bars.at(-1)?.[0]||null,o:bars[0]?.[1]??null,h:bars.length?Math.max(...bars.map(b=>b[2])):null,l:bars.length?Math.min(...bars.map(b=>b[3])):null,c:bars.at(-1)?.[4]??null,
   bars,expected,missing,partial,known,clipped,flags,complete:known&&!partial&&!clipped&&!missing.length&&bars.length>0};
 });
}
function layout(record,calendar,from,end,width,offset=null,options={}){
 width=Math.max(230,Number(width)||700);
 const compact=!!options.compact,top=compact?24:46,frame={left:55,right:width-9,top,bottom:top+224},span=frame.right-frame.left;
 const dailyDays=Math.round((ts(end)-ts(from))/DAY)+1,weeklyDays=dailyDays*7,weeklyFrom=start(end,weeklyDays);
 const source=record.bars.filter(b=>b[0]>=from&&b[0]<=end),lookup=new Map(source.map(b=>[b[0],b]));
 // Only an official calendar supplies expected sessions. Outside it, show observed dates, not guessed weekdays.
 const dates=new Set(source.map(b=>b[0]));
 if(calendar)for(const d of calendar.days)if(d>=from&&d<=end)dates.add(d);
 const days=[...dates].sort(),allWeeks=weeks(record,calendar,end,weeklyFrom),total=Math.max(days.length,allWeeks.length,1);
 const capacity=Math.max(1,Math.floor(span/5.5)),count=Math.min(total,capacity),maxOffset=total-count;
 const at=offset==null?maxOffset:Math.max(0,Math.min(maxOffset,Math.round(offset))),u0=at/total,u1=(at+count)/total;
 const x=u=>frame.left+(u-u0)/(u1-u0)*span;
 const visible=days.map((day,i)=>({day,i,u:(i+.5)/days.length,x:x((i+.5)/days.length),bar:lookup.get(day)||null,notListed:!!record.listing_date&&day<record.listing_date,early:calendar?.early_close.includes(day)||false})).filter(d=>d.u>=u0&&d.u<=u1);
 const weekly=allWeeks.map((w,i)=>({...w,i,u:(i+.5)/allWeeks.length,x:x((i+.5)/allWeeks.length),left:x(i/allWeeks.length),right:x((i+1)/allWeeks.length)})).filter(w=>w.u>=u0&&w.u<=u1);
 const spacing=span/Math.max(1,days.length)/(u1-u0),weeklySpacing=span/Math.max(1,allWeeks.length)/(u1-u0);
 // Both independent histories contribute to the same price domain, even when a layer is hidden.
 const highs=[...visible.filter(d=>d.bar).map(d=>d.bar[2]),...weekly.filter(w=>w.h!=null).map(w=>w.h)],lows=[...visible.filter(d=>d.bar).map(d=>d.bar[3]),...weekly.filter(w=>w.l!=null).map(w=>w.l)];
 const hi=highs.length?Math.max(...highs):1,lo=lows.length?Math.min(...lows):0,pad=Math.max((hi-lo)*.08,hi*.005);
 const domain=[Math.max(0,lo-pad),hi+pad],y=v=>frame.bottom-(v-domain[0])/(domain[1]-domain[0])*(frame.bottom-frame.top);
 return {record,calendar,from,end,dailyDays,weeklyDays,weeklyFrom,width,height:frame.bottom+30,compact,frame,days,visible,weekly,allWeeks,source,capacity,count,total,offset:at,maxOffset,u0,u1,spacing,weeklySpacing,domain,y};
}
function probe(l,u){
 u=Math.max(0,Math.min(1-1e-10,u));const dailyIndex=Math.min(l.days.length-1,Math.floor(u*l.days.length)),weeklyIndex=Math.min(l.allWeeks.length-1,Math.floor(u*l.allWeeks.length));
 const day=l.days[dailyIndex],daily=day?{day,bar:l.source.find(b=>b[0]===day)||null,i:dailyIndex,notListed:!!l.record.listing_date&&day<l.record.listing_date}:null;
 return {u,daily,weekly:l.allWeeks[weeklyIndex]||null,x:l.frame.left+(u-l.u0)/(l.u1-l.u0)*(l.frame.right-l.frame.left)};
}
function title(b){return `${b[0]} · 开 ${number(b[1])} / 高 ${number(b[2])} / 低 ${number(b[3])} / 收 ${number(b[4])}`;}
function svg(l,showWeek=true,selectedU=null,options={}){
 const clip=esc(options.id||'price-plot-clip');
 const {frame:f,y,width,height}=l,out=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="日线与周线独立横轴范围，周范围是日范围七倍，共用价格轴"><rect width="100%" height="100%" fill="var(--bg)"/><defs><clipPath id="${clip}"><rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}"/></clipPath></defs>`];
 if(!l.compact)out.push(`<text x="${width-9}" y="16" text-anchor="end">${esc(l.record.currency)} · 前复权</text>`);
 for(let i=0;i<5;i++){const v=l.domain[0]+(l.domain[1]-l.domain[0])*i/4,yy=y(v);out.push(`<line x1="${f.left}" x2="${f.right}" y1="${yy}" y2="${yy}" stroke="var(--line)"/><text x="${f.left-6}" y="${yy+4}" text-anchor="end">${number(v)}</text>`);}
 out.push(`<rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}" fill="none" stroke="var(--line)"/>`);
 if(options.selectedEvent&&(!options.visiblePatterns||options.selectedEvent.matches.some(k=>options.visiblePatterns.includes(k)))){const event=options.selectedEvent,a=l.days.indexOf(event.start),b=l.days.indexOf(event.day);if(a>=0&&b>=0){const x0=Math.max(f.left,f.left+(a/l.days.length-l.u0)/(l.u1-l.u0)*(f.right-f.left)),x1=Math.min(f.right,f.left+((b+1)/l.days.length-l.u0)/(l.u1-l.u0)*(f.right-f.left));if(x1>x0)out.push(`<rect class="pattern-span" data-from="${event.start}" data-to="${event.day}" x="${x0}" y="${f.top}" width="${x1-x0}" height="${f.bottom-f.top}" fill="var(--fg)" fill-opacity=".055" stroke="var(--secondary)" stroke-dasharray="3 4"><title>所选日线形态 ${event.start} → ${event.day}</title></rect>`);}}
 // Independent tick positions; no week boundary is projected through the daily calendar.
 function axis(items,from,end,labelY,tickY){
  const first=l.offset===0?from:items[0]?.periodFrom||items[0]?.day,last=l.offset===l.maxOffset?end:items.at(-1)?.periodEnd||items.at(-1)?.day;
  if(!first||!last)return;
  out.push(`<text x="${f.left}" y="${labelY}">${first.slice(5)}</text><text x="${f.right}" y="${labelY}" text-anchor="end">${last.slice(5)}</text>`);
  let prev=f.left+38;for(const item of items){if(item.x<prev+50||item.x>f.right-52)continue;const d=item.periodFrom||item.day;out.push(`<line x1="${item.x}" x2="${item.x}" y1="${tickY}" y2="${tickY+5}" stroke="var(--secondary)"/><text x="${item.x}" y="${labelY}" text-anchor="middle">${d.slice(5)}${item.partial?'*':''}</text>`);prev=item.x;}
 }
 axis(l.weekly,l.weeklyFrom,l.end,f.top-13,f.top-6);
 if(showWeek)for(const w of l.weekly){
  if(w.o==null){if(w.known&&w.missing.length)out.push(`<path class="missing-week" data-week="${w.day}" d="M${w.x-2},${f.bottom-12}l4,4m-4,0l4,-4" stroke="var(--secondary)" opacity=".5"/>`);continue;}
  const ww=l.weeklySpacing*.74,top=Math.min(y(w.o),y(w.c)),hh=Math.max(1.2,Math.abs(y(w.o)-y(w.c))),note=w.flags.join(' / ')||'完整周';
  const tt=`周 ${w.periodFrom} → ${w.periodEnd} · ${note} · 开 ${number(w.o)} / 高 ${number(w.h)} / 低 ${number(w.l)} / 收 ${number(w.c)}`;
  out.push(`<g class="weekly-candle" data-week="${w.day}" data-complete="${w.complete}" opacity=".55" clip-path="url(#${clip})"><title>${esc(tt)}</title><line x1="${w.x}" x2="${w.x}" y1="${y(w.h)}" y2="${y(w.l)}" stroke="var(--secondary)" stroke-width="1.5"/><rect x="${w.x-ww/2}" y="${top}" width="${ww}" height="${hh}" fill="var(--secondary)" fill-opacity="${w.c>=w.o?'.06':'.26'}" stroke="var(--secondary)" stroke-width="1.4" ${w.flags.length?'stroke-dasharray="4 3"':''}/></g>`);
 }
 for(const d of l.visible){
  if(!d.bar){if(!d.notListed)out.push(`<path class="missing-price-day" data-day="${d.day}" d="M${d.x-2},${f.bottom-6}l4,4m-4,0l4,-4" stroke="var(--secondary)"><title>${d.day} · 官方交易日，无有效OHLC，保持缺失</title></path>`);continue;}
  const b=d.bar,bw=Math.min(6,l.spacing*.32),up=b[4]>=b[1],yy=Math.min(y(b[1]),y(b[4])),bh=Math.max(1,Math.abs(y(b[1])-y(b[4])));
  out.push(`<g class="daily-candle" data-day="${d.day}"><title>${esc(title(b)+(d.early?' · 半日市':''))}</title><line x1="${d.x}" x2="${d.x}" y1="${y(b[2])}" y2="${y(b[3])}" stroke="var(--fg)" stroke-width="1"/><rect x="${d.x-bw/2}" y="${yy}" width="${bw}" height="${bh}" fill="${up?'var(--bg)':'var(--fg)'}" stroke="var(--fg)" stroke-width="1"/></g>`);
 }
 if(selectedU!=null&&selectedU>=l.u0&&selectedU<=l.u1){const p=probe(l,selectedU);out.push(`<line class="price-probe-guide" x1="${p.x}" x2="${p.x}" y1="${f.top}" y2="${f.bottom}" stroke="var(--fg)" stroke-dasharray="2 4" opacity=".5"/>`);}
 axis(l.visible,l.from,l.end,f.bottom+19,f.bottom);
 if(!l.source.length&&!l.weekly.some(w=>w.o!=null))out.push(`<text x="${(f.left+f.right)/2}" y="${(f.top+f.bottom)/2}" text-anchor="middle">所选双范围均没有有效价格</text>`);
 out.push(`<rect class="price-probe-hit" x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}" fill="transparent"/>`);
 const glyph={rise:'▲',v:'V',pullback:'P',step:'S'};
 for(const event of options.events||[]){const matches=event.matches.filter(k=>!options.visiblePatterns||options.visiblePatterns.includes(k)),d=l.visible.find(d=>d.day===event.day&&d.bar);if(!d||!matches.length)continue;const yy=Math.max(f.top+11,y(d.bar[2])-9),label=matches.map(k=>glyph[k]).join('');out.push(`<g class="pattern-marker" data-day="${event.day}" data-start="${event.start}" tabindex="0" role="button" aria-label="日线形态 ${esc(label)} ${event.start} 至 ${event.day}"><title>${esc(label)} · 日线 ${event.start} → ${event.day} · 区间涨幅 ${(event.metrics.end*100).toFixed(1)}%</title><circle cx="${d.x}" cy="${yy}" r="5" fill="var(--bg)" stroke="var(--fg)"/><text x="${d.x}" y="${yy+3}" text-anchor="middle" style="font-size:8px">${esc(label)}</text></g>`);}
 return out.join('')+'</svg>';
}
function mini(record,from,end,selectedFrom,selectedEnd,width,weeklyFrom=null){
 width=Math.max(230,Number(width)||700);const f={left:8,right:width-8,top:13,bottom:61},x=d=>f.left+Math.max(0,Math.min(1,(ts(d)-ts(from))/(ts(end)-ts(from))))*(f.right-f.left);
 const bars=record.bars.filter(b=>b[0]>=from&&b[0]<=end),hi=bars.length?Math.max(...bars.map(b=>b[2])):1,lo=bars.length?Math.min(...bars.map(b=>b[3])):0,range=Math.max(.001,hi-lo),y=v=>f.bottom-(v-lo)/range*(f.bottom-f.top);
 const out=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} 88" width="${width}" height="88" role="img" aria-label="固定两年真实日级OHLC预览"><rect width="100%" height="100%" fill="var(--bg)"/><rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}" fill="var(--muted)"/><rect x="${x(selectedFrom)}" y="${f.top}" width="${Math.max(1,x(selectedEnd)-x(selectedFrom))}" height="${f.bottom-f.top}" fill="var(--bg)" stroke="var(--fg)"/>`];
 if(weeklyFrom)out.push(`<rect class="mini-week-range" x="${x(weeklyFrom)}" y="${f.top}" width="${Math.max(1,x(selectedEnd)-x(weeklyFrom))}" height="${f.bottom-f.top}" fill="none" stroke="var(--secondary)" stroke-dasharray="4 3"/>`);
 for(const b of bars){const xx=x(b[0]);out.push(`<path class="mini-ohlc" data-day="${b[0]}" d="M${xx},${y(b[2])}V${y(b[3])}M${xx-.8},${y(b[1])}H${xx}M${xx},${y(b[4])}H${xx+.8}" fill="none" stroke="var(--secondary)" stroke-width=".8"/>`);}
 out.push(`<text x="${f.left}" y="81">${from}</text><text x="${f.right}" y="81" text-anchor="end">${end}</text></svg>`);
 return out.join('');
}
function miniTarget(clientX,box,from,end){const ratio=Math.max(0,Math.min(1,(clientX-box.left-8)/(box.width-16)));return iso(ts(from)+Math.round((ts(end)-ts(from))/DAY*ratio)*DAY);}
return {DAY,ts,iso,start,weekStart,weeks,layout,probe,svg,mini,miniTarget,number};
});
