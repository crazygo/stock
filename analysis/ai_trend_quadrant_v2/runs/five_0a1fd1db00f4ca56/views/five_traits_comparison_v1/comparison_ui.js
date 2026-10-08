// Inlined inside the page's scope; rendering reads captured data only.
const cmp={active:'map',from:PC.start(P.as_of,60),end:P.as_of,sources:new Set(['holdings']),patterns:new Set(),quadrants:new Set(),search:'',page:0,week:true};
const compareDerived=new Map(),compareScanCache=new Map(),compareViews=new Map(),PAGE_SIZE=12;
const sourceGroups=[{id:'holdings',name:'持仓'},{id:'watchlist',name:'自选'},{id:'special',name:'特别关注'},{id:'ai_plus',name:'AI 产业池 A+'},{id:'ai_a',name:'AI 产业池 A'}];
const statusNames={unavailable:'无行情',insufficient_history:'历史不足',unverified_calendar:'日历未核验',missing_sessions:'交易日有缺失',stale:'行情未到终点',insufficient_or_unverified:'可评分窗口不足',evaluated:'已按日线识别',scored:'已计算'};
const statusText=s=>statusNames[s]||s;
function selectView(view){
 if(state.open)close();cmp.active=view;$('map-panel').hidden=view!=='map';$('compare-panel').hidden=view!=='compare';
 for(const [id,v]of [['tab-map','map'],['tab-compare','compare']]){$(id).setAttribute('aria-selected',String(v===view));$(id).tabIndex=v===view?0:-1;}
 $('meta').textContent=view==='map'?`四象限快照 · 行情截止 ${D.as_of} · ${D.summary.securities} 个证券 · 五标签为探索描述代理`:`多股票分组 · 行情截止 ${C.as_of} · ${C.summary.members} 个证券 · 当前成员快照`;
 $('tab-'+(view==='map'?'map':'compare')).focus();requestAnimationFrame(()=>{if(cmp.active==='map')renderMap();else renderComparison();});
}
function setupComparison(){
 for(const [family,label,items]of [['sources','股票组 · 可多选',sourceGroups],['patterns','日线形态 · 范围内至少一次',DP.TYPES],['quadrants','所选终点的四个象限',DP.QUADRANTS]]){
  const field=document.createElement('fieldset');field.className='compare-family';field.innerHTML=`<legend>${label}</legend><div class="compare-options">${items.map(t=>`<label><input type="checkbox" data-compare-family="${family}" value="${t.id}" ${cmp[family].has(t.id)?'checked':''}><span>${esc(t.name)}</span><small data-count-for="${t.id}"></small></label>`).join('')}</div>`;
  field.querySelectorAll('input').forEach(input=>input.onchange=()=>{if(input.checked)cmp[family].add(input.value);else cmp[family].delete(input.value);cmp.page=0;renderComparison();});$('compare-options').append(field);
 }
 $('compare-from').min=$('compare-to').min=P.mini_start;$('compare-from').max=$('compare-to').max=P.as_of;
 $('compare-range').onsubmit=e=>{e.preventDefault();setCompareRange($('compare-from').value,$('compare-to').value);};
 document.querySelectorAll('[data-compare-days]').forEach(b=>b.onclick=()=>setCompareRange(PC.start(P.as_of,Number(b.dataset.compareDays)),P.as_of));
 $('tab-map').onclick=()=>selectView('map');$('tab-compare').onclick=()=>selectView('compare');
 $('view-tabs').onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();selectView(e.key==='Home'?'map':e.key==='End'?'compare':cmp.active==='map'?'compare':'map');};
 let searchTimer;$('compare-search').oninput=e=>{cmp.search=e.target.value.trim().toLowerCase();cmp.page=0;clearTimeout(searchTimer);searchTimer=setTimeout(renderComparison,100);};
 $('compare-clear').onclick=()=>{for(const key of ['sources','patterns','quadrants'])cmp[key].clear();$('compare-options').querySelectorAll('input').forEach(i=>i.checked=false);cmp.search='';$('compare-search').value='';cmp.page=0;renderComparison();};
 $('compare-week').onchange=e=>{cmp.week=e.target.checked;renderComparisonCharts();};
 $('compare-prev').onclick=()=>{cmp.page=Math.max(0,cmp.page-1);renderComparison();};$('compare-next').onclick=()=>{cmp.page++;renderComparison();};
 $('compare-method').innerHTML='<p>算法 daily_close_patterns_v1：每个形态使用连续10个交易日、11个前复权收盘。共同条件：首尾涨幅≥5%。单边拉升：最低偏离≥−0.5%、峰值在后25%、方向效率≥0.65、上涨日≥60%。V型深弹：谷值在前45%、下探≥1.5%。冲高回落：峰值在前55%、从峰值回落≥1.5个百分点。阶梯中继：两段上涨分别≥2.5%，中间至少三个收盘构成≤1.5%的平台、平台首尾偏移≤0.75%，两段效率分别≥0.55；有实际平台才打标。</p><p>形态须完整包含在所选日期内，只有已确认终点使用当时及此前收盘；重叠窗口保留，计数是匹配窗口数，不是独立交易次数。一只股票可命中多个形态。未核验日历或缺日不拼接成形态。</p><p>四象限沿用当前地图的G=0、V=35%分界，在所选终点按G120／V60点估计分组；不是原v1 Pugh归属概率。A+／A沿用AI产业池快照，其中研究初筛与正式核查分别标记。成员和评级是当前快照，历史图不代表当时持仓或评级。</p>';
 let previousWidth=0,resizeTimer;new ResizeObserver(()=>{const width=$('compare-grid').clientWidth;if(width===previousWidth)return;previousWidth=width;clearTimeout(resizeTimer);if(cmp.active==='compare')resizeTimer=setTimeout(renderComparisonCharts,80);}).observe($('compare-grid'));
}
function setCompareRange(from,end){
 if(!from||!end||from>end||from<P.mini_start||end>P.as_of){$('compare-status').textContent='请选择导航范围内且起点不晚于终点的日期。';return;}
 Object.assign(cmp,{from,end,page:0});compareViews.clear();renderComparison();
}
function jumpCompareRange(day,code){
 const n=Math.round((PC.ts(cmp.end)-PC.ts(cmp.from))/PC.DAY)+1,span=(n-1)*PC.DAY;
 const first=Math.max(PC.ts(P.mini_start),Math.min(PC.ts(P.as_of)-span,PC.ts(day)-Math.floor((n-1)/2)*PC.DAY));
 setCompareRange(PC.iso(first),PC.iso(first+span));
 requestAnimationFrame(()=>{const mini=$('compare-grid').querySelector(`[data-code="${code}"] .compare-mini`);(mini||$('compare-from')).focus();});
}
function deriveComparison(){
 compareDerived.clear();for(const r of C.records){const price=comparePrices.get(r.code),cal=P.calendars[price.market],key=r.code+'|'+cmp.from+'|'+cmp.end;let d=compareScanCache.get(key);
  if(!d){d={scan:DP.scan(price,cal,cmp.from,cmp.end),quad:DP.quadrant(price,cal,cmp.end)};compareScanCache.set(key,d);}compareDerived.set(r.code,{record:r,price,...d});
 }
 if(compareScanCache.size>4000){const current=[...compareDerived].map(([code,d])=>[code+'|'+cmp.from+'|'+cmp.end,d]);compareScanCache.clear();current.forEach(([key,d])=>compareScanCache.set(key,d));}
}
function renderComparison(){
 if(cmp.active!=='compare')return;deriveComparison();
 const selection={sources:[...cmp.sources],patterns:[...cmp.patterns],quadrants:[...cmp.quadrants]},rows=[...compareDerived.values()],counts={};
 for(const t of sourceGroups)counts[t.id]=C.records.filter(r=>r.groups.includes(t.id)).length;
 for(const t of DP.TYPES)counts[t.id]=rows.filter(d=>d.scan.counts[t.id]>0).length;
 for(const t of DP.QUADRANTS)counts[t.id]=rows.filter(d=>d.quad.id===t.id).length;
 $('compare-options').querySelectorAll('[data-count-for]').forEach(e=>e.textContent='('+counts[e.dataset.countFor]+')');
 const search=r=>!cmp.search||(r.code+' '+r.name).toLowerCase().includes(cmp.search),source=r=>!selection.sources.length||selection.sources.some(g=>r.groups.includes(g));
 const filteredRows=rows.filter(d=>search(d.record)&&DP.matchesGroup(d.record,d.scan,d.quad,selection));
 const pending=rows.filter(d=>{if(!search(d.record)||!source(d.record))return false;const pUnknown=selection.patterns.length&&!selection.patterns.some(p=>d.scan.counts[p]>0)&&(d.scan.status!=='evaluated'||d.scan.missing.length>0||d.scan.unverified),qUnknown=selection.quadrants.length&&!d.quad.id;
  return (pUnknown||qUnknown)&&(!selection.patterns.length||pUnknown||selection.patterns.some(p=>d.scan.counts[p]>0))&&(!selection.quadrants.length||qUnknown||selection.quadrants.includes(d.quad.id));});
 filteredRows.sort((a,b)=>Number(b.record.groups.includes('holdings'))-Number(a.record.groups.includes('holdings'))||a.record.code.localeCompare(b.record.code));
 const pages=Math.max(1,Math.ceil(filteredRows.length/PAGE_SIZE));cmp.page=Math.min(cmp.page,pages-1);
 $('compare-from').value=cmp.from;$('compare-to').value=cmp.end;const n=Math.round((PC.ts(cmp.end)-PC.ts(cmp.from))/PC.DAY)+1;
 $('compare-panel').querySelector('h2').textContent=`多股票 · 日线 ${n} 天 × 周线 ${n*7} 天`;
 $('compare-axis').textContent=`日范围 ${cmp.from} → ${cmp.end}；周范围 ${PC.start(cmp.end,n*7)} → ${cmp.end}。两轴日期独立，价格轴按每只证券的本币显示。`;
 $('compare-data').textContent=`行情截止 ${C.as_of} · 分组读取 ${new Date(C.group_observed_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})} 北京时间 · AI评级 ${C.quality_as_of} · ${C.summary.members} 个证券 / ${C.summary.with_prices} 个有日线。括号为全池证券数，形态与象限按所选日期更新。`;
 document.querySelectorAll('[data-compare-days]').forEach(b=>b.setAttribute('aria-pressed',String(cmp.end===P.as_of&&cmp.from===PC.start(P.as_of,Number(b.dataset.compareDays)))));
 $('compare-status').textContent=`匹配 ${filteredRows.length} 只 · 待确认 ${pending.length} 只 · 每页最多 ${PAGE_SIZE} 张价格图。▲ / V / P / S 是日线形态终点标记；选择命中窗口查看整段形态。`;
 $('compare-page-info').textContent=`第 ${cmp.page+1} / ${pages} 页 · ${filteredRows.length?cmp.page*PAGE_SIZE+1:0}—${Math.min((cmp.page+1)*PAGE_SIZE,filteredRows.length)} / ${filteredRows.length}`;
 $('compare-prev').disabled=cmp.page===0;$('compare-next').disabled=cmp.page===pages-1;$('compare-empty').hidden=filteredRows.length>0;
 $('compare-grid').replaceChildren();for(const d of filteredRows.slice(cmp.page*PAGE_SIZE,(cmp.page+1)*PAGE_SIZE))appendComparisonCard(d);
 $('compare-pending-title').textContent=`筛选依据待确认（${pending.length} 只，保留缺失）`;$('compare-pending-list').replaceChildren();for(const d of pending){const button=document.createElement('button');button.type='button';const reason=!d.quad.id&&selection.quadrants.length?d.quad.status:d.scan.missing.length?'missing_sessions':d.scan.unverified?'unverified_calendar':d.scan.status;button.textContent=d.record.code+' · '+statusText(reason);button.onclick=()=>open(d.record.code,true);$('compare-pending-list').append(button);}
 renderComparisonCharts();
}
function appendComparisonCard(d){
 const r=d.record,p=d.price,card=document.createElement('article');card.className='compare-card';card.dataset.code=r.code;
 const tags=r.groups.map(g=>sourceGroups.find(t=>t.id===g)?.name||g),quality=r.quality,quad=DP.QUADRANTS.find(t=>t.id===d.quad.id);
 card.innerHTML=`<header class="compare-card-head"><div><h3>${esc(r.code)} · ${esc(r.name)}</h3><div class="small muted">${esc(p.currency)} · ${quad?esc(quad.name):'象限待确认：'+esc(statusText(d.quad.status))}</div></div><button type="button" class="compare-detail">详情</button></header><div class="compare-badges">${tags.map(t=>`<span>${esc(t)}</span>`).join('')}${quality?`<span>${quality.status==='verified'?'正式核查':'研究初筛'} ${esc(quality.grade)} · ${esc(quality.as_of)}</span>`:''}</div><div class="compare-chart"></div><div class="price-pan"><button type="button" data-card-pan="-1">← 较早</button><input class="compare-pan-slider" type="range" min="0" step="1" aria-label="${esc(r.code)} 双轴平移"><button type="button" data-card-pan="1">较近 →</button></div><div class="small muted compare-visible"></div><div class="small compare-probe"></div><div class="compare-mini price-mini" role="slider" tabindex="0" aria-label="${esc(r.code)} 两年真实日级OHLC导航，调整全部股票观察范围"></div><div class="small muted">两年导航：实框为日范围，虚框为周范围；点击或方向键调整全部股票。</div><label class="small">日线命中窗口<select class="compare-case"><option value="">选择形态（${d.scan.events.length} 个匹配窗口）</option>${d.scan.events.map((e,i)=>`<option value="${i}">${e.start} → ${e.day} · ${e.matches.map(id=>DP.TYPES.find(t=>t.id===id).name).join(' / ')}</option>`).join('')}</select></label><div class="small muted compare-case-summary"></div><details><summary>命中依据与覆盖</summary><div class="compare-case-body small"></div></details><div class="small muted">实际行情 ${p.first_day||'无'} → ${p.last_day||'无'}；有效窗口 ${d.scan.scored}；${d.scan.missing.length?'缺 '+d.scan.missing.length+' 个交易日；':''}${d.scan.unverified?'日历有未核验范围；':''}${statusText(d.scan.status)}。</div>`;
 card.querySelector('.compare-detail').onclick=()=>open(r.code,true);card.querySelector('.compare-case').disabled=!d.scan.events.length;
 card.querySelector('.compare-case').onchange=e=>chooseComparisonEvent(card,e.target.value===''?null:Number(e.target.value));
 card.querySelectorAll('[data-card-pan]').forEach(b=>b.onclick=()=>{const v=compareViews.get(r.code),l=v.layout;v.offset=Math.max(0,Math.min(l.maxOffset,l.offset+Number(b.dataset.cardPan)*Math.max(1,Math.floor(l.count*.8))));v.probe=null;renderComparisonCard(card);});
 card.querySelector('.compare-pan-slider').oninput=e=>{const v=compareViews.get(r.code);v.offset=Number(e.target.value);v.probe=null;renderComparisonCard(card);};
 const mini=card.querySelector('.compare-mini');mini.onclick=e=>jumpCompareRange(PC.miniTarget(e.clientX,mini.getBoundingClientRect(),P.mini_start,P.as_of),r.code);
 mini.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const center=PC.iso(Math.round((PC.ts(cmp.from)+PC.ts(cmp.end))/2));jumpCompareRange(e.key==='Home'?P.mini_start:e.key==='End'?P.as_of:PC.iso(PC.ts(center)+(e.key==='ArrowLeft'?-1:1)*(e.shiftKey?30:7)*PC.DAY),r.code);};
 $('compare-grid').append(card);
}
function chooseComparisonEvent(card,index){
 const d=compareDerived.get(card.dataset.code),v=compareViews.get(card.dataset.code);v.event=index;
 if(index!=null){const e=d.scan.events[index],a=v.layout.days.indexOf(e.start),b=v.layout.days.indexOf(e.day);v.probe=(b+.5)/v.layout.days.length;const lo=a/v.layout.days.length,hi=(b+1)/v.layout.days.length;
  if(lo<v.layout.u0||hi>v.layout.u1)v.offset=Math.max(0,Math.min(v.layout.maxOffset,Math.floor((lo+hi)/2*v.layout.total-v.layout.count/2)));
 }
 renderComparisonCard(card);
}
function renderComparisonCard(card){
 const d=compareDerived.get(card.dataset.code);if(!d)return;let v=compareViews.get(card.dataset.code);if(!v){v={offset:null,probe:null,event:null};compareViews.set(card.dataset.code,v);}
 const chart=card.querySelector('.compare-chart'),l=PC.layout(d.price,P.calendars[d.price.market],cmp.from,cmp.end,chart.clientWidth,v.offset);v.layout=l;v.offset=l.offset;if(v.probe==null||v.probe<l.u0||v.probe>l.u1)v.probe=l.u1-1e-10;
 const e=v.event==null?null:d.scan.events[v.event],id='compare-clip-'+card.dataset.code.replace(/[^a-zA-Z0-9]/g,'-');
 chart.innerHTML=PC.svg(l,cmp.week,v.probe,{id,events:d.scan.events,selectedEvent:e});const svg=chart.querySelector('svg');
 const pointer=e=>{const box=svg.getBoundingClientRect(),x=(e.clientX-box.left)*l.width/box.width;return Math.max(l.u0,Math.min(l.u1-1e-10,l.u0+(x-l.frame.left)/(l.frame.right-l.frame.left)*(l.u1-l.u0)));};
 svg.onclick=event=>{const mark=event.target.closest('.pattern-marker');if(mark){chooseComparisonEvent(card,d.scan.events.findIndex(e=>e.day===mark.dataset.day&&e.start===mark.dataset.start));return;}if(event.target.closest('.price-probe-hit')){v.probe=pointer(event);renderComparisonCard(card);}};
 svg.onkeydown=event=>{const mark=event.target.closest('.pattern-marker');if(mark&&['Enter',' '].includes(event.key)){event.preventDefault();chooseComparisonEvent(card,d.scan.events.findIndex(e=>e.day===mark.dataset.day&&e.start===mark.dataset.start));card.querySelector('.compare-case').focus();}};
 svg.onpointermove=event=>{if(event.target.closest('.price-probe-hit'))card.querySelector('.compare-probe').textContent=priceProbeText(l,pointer(event));};svg.onpointerleave=()=>card.querySelector('.compare-probe').textContent=priceProbeText(l,v.probe);
 card.querySelector('.compare-probe').textContent=priceProbeText(l,v.probe);const slider=card.querySelector('.compare-pan-slider');slider.max=String(l.maxOffset);slider.value=String(l.offset);slider.disabled=!l.maxOffset;
 const mini=card.querySelector('.compare-mini');mini.innerHTML=PC.mini(d.price,P.mini_start,P.as_of,cmp.from,cmp.end,chart.clientWidth,l.weeklyFrom);
 mini.setAttribute('aria-valuemin','0');mini.setAttribute('aria-valuemax',String(Math.round((PC.ts(P.as_of)-PC.ts(P.mini_start))/PC.DAY)));mini.setAttribute('aria-valuenow',String(Math.round(((PC.ts(cmp.from)+PC.ts(cmp.end))/2-PC.ts(P.mini_start))/PC.DAY)));mini.setAttribute('aria-valuetext','日线 '+cmp.from+' 至 '+cmp.end+'；周线 '+l.weeklyFrom+' 至 '+cmp.end);
 card.querySelector('[data-card-pan="-1"]').disabled=l.offset===0;card.querySelector('[data-card-pan="1"]').disabled=l.offset===l.maxOffset;
 card.querySelector('.compare-visible').textContent=`日线可见 ${l.visible[0]?.day||'无'} → ${l.visible.at(-1)?.day||'无'}；周线可见 ${l.weekly[0]?.periodFrom||'无'} → ${l.weekly.at(-1)?.periodEnd||'无'}。`;
 card.querySelector('.compare-case').value=v.event==null?'':String(v.event);card.querySelector('.compare-case-summary').textContent=e?`${e.start} → ${e.day} · 日线 ${(e.metrics.end*100).toFixed(1)}% · 阴影是所选日线形态范围。`:'命中窗口：'+DP.TYPES.map(t=>t.short+' '+d.scan.counts[t.id]).join(' · ');
 card.querySelector('.compare-case-body').innerHTML=e?`<p>${e.matches.map(id=>DP.TYPES.find(t=>t.id===id).name).join(' / ')}：首尾 ${(e.metrics.end*100).toFixed(2)}%，谷值 ${(e.metrics.min*100).toFixed(2)}%，谷位置 ${(e.metrics.minPos*100).toFixed(0)}%；峰值 ${(e.metrics.max*100).toFixed(2)}%，峰位置 ${(e.metrics.maxPos*100).toFixed(0)}%；效率 ${e.metrics.eff.toFixed(3)}，上涨 ${e.metrics.positive}/10 日。${e.metrics.platform?'平台波幅 '+(e.metrics.platform.band*100).toFixed(2)+'%。':''}</p><p>确认日 ${e.day}；输入只到该日收盘。连续窗口可能重叠，不是独立操作信号。</p>`:`<p>扫描 ${d.scan.scored} 个可评分窗口；匹配 ${d.scan.events.length} 个窗口。每种形态仅在区间内完整匹配至少一次时加入对应组。</p><p>${d.scan.missing.length?'缺失日：'+esc(d.scan.missing.join('、')):'没有已核验交易日缺口。'}${d.scan.unverified?' 部分日期超出已核验日历，保留未知。':''}</p>`;
}
function renderComparisonCharts(){if(cmp.active!=='compare')return;$('compare-grid').querySelectorAll('.compare-card').forEach(renderComparisonCard);}
setupComparison();
