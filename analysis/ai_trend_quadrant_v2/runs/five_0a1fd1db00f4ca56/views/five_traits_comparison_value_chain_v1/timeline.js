/* Pure rendering of captured evaluations; no browser, network, fitting or clock. */
(function(root){
  const KEYS=['G','V','M','J','S'];
  const STYLES={G:{dash:'',shape:'circle',color:'var(--fg)'},V:{dash:'8 4',shape:'diamond',color:'var(--secondary)'},M:{dash:'2 4',shape:'triangle',color:'var(--fg)'},J:{dash:'7 3 2 3',shape:'square',color:'var(--secondary)'},S:{dash:'12 5',shape:'cross',color:'var(--random)'}};
  const ts=d=>Date.parse(d+'T00:00:00Z');
  const start=(end,n)=>new Date(ts(end)-(n-1)*86400000).toISOString().slice(0,10);
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function score(key,value){
    if(!Number.isFinite(value))return null;
    if(key==='G')return 50*(1+Math.tanh(value/.50));
    if(key==='V')return 50*(1+Math.tanh((value-.35)/.25));
    if(key==='M')return 50*(1+value);
    return 100*value;
  }
  function raw(key,value){return value==null?'—':key==='M'||key==='S'?value.toFixed(2):(100*value).toFixed(1)+'%';}
  function layout(record,end,n,width,visible=KEYS){
    const from=start(end,n),height=350,frame={left:42,right:width-13,top:43,bottom:height-42};
    const x=day=>frame.left+(ts(day)-ts(from))/(ts(end)-ts(from))*(frame.right-frame.left),y=s=>frame.bottom-s/100*(frame.bottom-frame.top);
    const rows=(record.history||[]).filter(h=>h.day>=from&&h.day<=end);
    const series=KEYS.map((key,k)=>{
      const segments=[];let current=[];const points=[];
      for(const h of rows){const v=h.v[k],s=score(key,v);if(s==null||h.status?.[k]!=='scored'){if(current.length)segments.push(current);current=[];continue;}
        const p={day:h.day,value:v,score:s,x:x(h.day),y:y(s)};points.push(p);current.push(p);
      }
      if(current.length)segments.push(current);
      return {key,visible:visible.includes(key),points,segments,style:STYLES[key]};
    });
    const ticks=[from,...(width>=500?[new Date(ts(from)+(ts(end)-ts(from))/2).toISOString().slice(0,10)]:[]),end].map((day,i,a)=>({day,x:x(day),anchor:i===0?'start':i===a.length-1?'end':'middle'}));
    return {from,end,n,width,height,frame,series,ticks,rows};
  }
  function marker(shape,x,y,r,attrs=''){
    if(shape==='diamond')return `<path ${attrs} d="M${x},${y-r}L${x+r},${y}L${x},${y+r}L${x-r},${y}Z"/>`;
    if(shape==='triangle')return `<path ${attrs} d="M${x},${y-r}L${x+r},${y+r}L${x-r},${y+r}Z"/>`;
    if(shape==='square')return `<rect ${attrs} x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}"/>`;
    if(shape==='cross')return `<path ${attrs} d="M${x-r},${y-r}L${x+r},${y+r}M${x-r},${y+r}L${x+r},${y-r}"/>`;
    return `<circle ${attrs} cx="${x}" cy="${y}" r="${r}"/>`;
  }
  function svg(l,selectedDay){
    const f=l.frame,parts=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${l.width} ${l.height}" height="${l.height}" role="img" aria-label="真实日期轴，五标签线与有效计算点" data-from="${l.from}" data-end="${l.end}" data-lookback="${l.n}" data-y-domain="0,100">`];
    parts.push(`<rect width="${l.width}" height="${l.height}" fill="var(--bg)"/>`);
    parts.push('<text x="0" y="15">标签显示分数 0–100</text>');
    parts.push(`<text x="${l.width-2}" y="15" text-anchor="end">${l.from.slice(0,4)}年</text>`);
    for(let i=0;i<KEYS.length;i++){const key=KEYS[i],style=STYLES[key],x=f.left+i*(f.right-f.left)/5;parts.push(`<path d="M${x},30L${x+17},30" fill="none" stroke="${style.color}" stroke-width="1.5" stroke-dasharray="${style.dash}"/>${marker(style.shape,x+8,30,2,`fill="var(--bg)" stroke="${style.color}" stroke-width="1"`)}<text x="${x+21}" y="34">${key}</text>`);}
    parts.push(`<rect x="${f.left}" y="${f.top}" width="${f.right-f.left}" height="${f.bottom-f.top}" fill="none" stroke="var(--line)" data-chart-frame=""/>`);
    for(const s of [0,25,50,75,100]){const y=f.bottom-s/100*(f.bottom-f.top);parts.push(`<line x1="${f.left}" x2="${f.right}" y1="${y}" y2="${y}" stroke="var(--line)" opacity=".55"/><text x="${f.left-7}" y="${y+4}" text-anchor="end">${s}</text>`);}
    if(selectedDay>=l.from&&selectedDay<=l.end){const x=f.left+(ts(selectedDay)-ts(l.from))/(ts(l.end)-ts(l.from))*(f.right-f.left);parts.push(`<line x1="${x}" x2="${x}" y1="${f.top}" y2="${f.bottom}" stroke="var(--line)" stroke-dasharray="3 3"/>`);}
    for(const s of l.series){
      const d=s.segments.map(a=>a.map((p,i)=>(i?'L':'M')+p.x+','+p.y).join('')).join('');
      parts.push(`<g data-trait="${s.key}" ${s.visible?'':'style="display:none"'}><path class="timeline-path" data-trait-path="${s.key}" d="${d}" fill="none" stroke="${s.style.color}" stroke-width="1.5" stroke-dasharray="${s.style.dash}"/>`);
      for(const p of s.points){const radius=p.day===selectedDay?3.3:l.width<500?2:2.5;const a=`class="timeline-point" data-day="${p.day}" data-trait="${s.key}" data-value="${p.value}" data-score="${p.score}" fill="${s.key==='G'?s.style.color:'var(--bg)'}" stroke="${s.style.color}" stroke-width="1.1"`;
        parts.push(`<g data-calculated-day="${p.day}">${marker(s.style.shape,p.x,p.y,radius,a)}<title>${p.day} · ${s.key} ${esc(raw(s.key,p.value))} · 显示分数 ${p.score.toFixed(1)}</title></g>`);
      }
      parts.push('</g>');
    }
    if(!l.series.some(s=>s.visible&&s.points.length))parts.push(`<text x="${(f.left+f.right)/2}" y="${(f.top+f.bottom)/2}" text-anchor="middle">${l.series.some(s=>s.visible)?'该范围没有可显示的计算结果':'勾选标签显示曲线'}</text>`);
    for(const t of l.ticks)parts.push(`<text x="${t.x}" y="${l.height-20}" text-anchor="${t.anchor}">${t.day.slice(5)}</text>`);
    parts.push(`<text x="${(f.left+f.right)/2}" y="${l.height-3}" text-anchor="middle">日期</text></svg>`);
    return parts.join('');
  }
  const api={KEYS,STYLES,start,score,raw,layout,marker,svg};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.FiveTraitTimeline=api;
})(typeof globalThis!=='undefined'?globalThis:this);
