/* Offline checks of captured memberships, causal daily tags, filters and SVG geometry. */
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict'),vm=require('vm');
const OUT=__dirname,ROOT=path.resolve(OUT,'../..'),PC=require('./price_chart.js'),DP=require('./daily_patterns.js');
const read=n=>JSON.parse(fs.readFileSync(path.join(OUT,n),'utf8')),sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const P=read('prices.json'),C=read('comparison.json'),D=read('results.json'),checks=[];
function check(name,pass){assert.ok(pass,name);checks.push(name);}
const prices=new Map([...P.records,...C.extra_prices].map(r=>[r.code,r]));
check('snapshot cutoff is frozen and shared',C.as_of===P.as_of&&C.as_of==='2026-10-06');
check('original descriptors remain unchanged',sha(path.join(OUT,'results.json'))==='591d47626e88daab9353cf7e99c11177a428689c5981a4ceb8e13dcd9c715b68');
check('members are unique and every one has a price or unavailable projection',new Set(C.records.map(r=>r.code)).size===C.records.length&&C.records.every(r=>prices.has(r.code)));
for(const item of C.inputs)check('input hash '+item.path,sha(path.resolve(ROOT,item.path))===item.sha256);
for(const item of C.provenance)check('extra daily source hash '+item.code,sha(path.resolve(ROOT,item.path))===item.sha256);
const capture=read('group_capture.json'),screen=JSON.parse(fs.readFileSync(path.join(ROOT,'analysis/ai_value_chain_map_v1/quality/screen_current.json')));
for(const [group,items]of [['holdings',capture.holdings],['watchlist',capture.watchlists['全部']],['special',capture.watchlists['特别关注']]]){
 const wanted=items.filter(i=>!C.excluded.some(e=>e.code===i.code&&e.group===group)).map(i=>i.code).sort();
 check('captured exact membership '+group,JSON.stringify(C.records.filter(r=>r.groups.includes(group)).map(r=>r.code).sort())===JSON.stringify([...new Set(wanted)]));
}
for(const r of C.records.filter(r=>r.quality))check('AI grade and evidence status '+r.code,JSON.stringify(r.quality)===JSON.stringify(Object.fromEntries(['grade','verified_grade','status','as_of','formal_quality_as_of','expires_at'].map(k=>[k,screen.records[r.code][k]??null]))));
check('non-securities remain excluded',C.excluded.length===15&&C.records.every(r=>['STOCK','ETF'].includes(r.kind)));
check('all held securities have prices',C.records.filter(r=>r.groups.includes('holdings')).every(r=>prices.get(r.code).bars.length));
check('newly listed held ETF keeps insufficient quadrant history',C.records.find(r=>r.code==='US.DISK').kind==='ETF'&&prices.get('US.DISK').listing_date==='2026-06-30'&&DP.quadrant(prices.get('US.DISK'),P.calendars.US,C.as_of).status==='insufficient_history');
check('unknown ETF listing date is not a 1970 placeholder',C.records.find(r=>r.code==='US.IHE').kind==='ETF'&&prices.get('US.IHE').listing_date!=='1970-01-01');
const expectedShapes={rise:[100,101,102,103,104,105,106,107,108,109,110],v:[100,98,96,97,98,100,101,102,103,104,106],pullback:[100,104,108,112,111,110,109,108,107,106,105],step:[100,99,103,105,105.2,105.1,105.3,105.2,106,108,110]};
for(const [kind,a]of Object.entries(expectedShapes))check('known daily-close fixture '+kind,DP.classify(a).matches.includes(kind));
check('a continuous rise is not a platform by default',!DP.classify(expectedShapes.rise).matches.includes('step'));
check('unclassified path is not residual step',DP.classify([100,99,103,102,106,104,107,106,108,107,106]).matches.length===0);
check('10 closes are insufficient for 10 returns',DP.classify(expectedShapes.rise.slice(1)).reason==='invalid_window');
check('low endpoint return fails all upward labels',DP.classify([100,99,98,97,96,97,98,99,100,101,102]).matches.length===0);
check('scale invariance',JSON.stringify(DP.classify(expectedShapes.step))===JSON.stringify(DP.classify(expectedShapes.step.map(v=>v*8))));
const days=P.calendars.US.days.filter(d=>d>='2026-08-01'&&d<='2026-09-01'),fixture={market:'US',currency:'USD',bars:expectedShapes.rise.map((c,i)=>[days[i],c,c,c,c]),listing_date:null};
const full=DP.scan(fixture,P.calendars.US,days[0],days[10]);
check('one fully contained confirmed match',full.events.length===1&&full.events[0].start===days[0]&&full.events[0].day===days[10]);
check('no event whose start precedes selected range',DP.scan(fixture,P.calendars.US,days[1],days[10]).events.length===0);
check('missing official session breaks recognition',DP.scan({...fixture,bars:fixture.bars.filter((b,i)=>i!==4)},P.calendars.US,days[0],days[10]).events.length===0);
check('unknown calendar is not guessed from observed weekdays',DP.scan(fixture,null,days[0],days[10]).events.length===0);
check('later bars cannot change earlier tags',JSON.stringify(full)===JSON.stringify(DP.scan({...fixture,bars:[...fixture.bars,[days[11],1,1,1,1]]},P.calendars.US,days[0],days[10])));
const markerEvent={...full.events[0],matches:['rise','v']},markerLayout=PC.layout(fixture,P.calendars.US,days[0],days[10],700);
const allMarkers=PC.svg(markerLayout,true,null,{events:[markerEvent],visiblePatterns:DP.TYPES.map(t=>t.id)}),onlyV=PC.svg(markerLayout,true,null,{events:[markerEvent],visiblePatterns:['v']}),noMarkers=PC.svg(markerLayout,true,null,{events:[markerEvent],visiblePatterns:[],selectedEvent:markerEvent});
check('all marker types render together by default',allMarkers.includes('日线形态 ▲V'));
check('global visibility hides only the unchecked glyph',onlyV.includes('日线形态 V')&&!onlyV.includes('日线形态 ▲'));
check('unchecking all marker types also clears highlight',!noMarkers.includes('class="pattern-marker"')&&!noMarkers.includes('class="pattern-span"'));
check('marker visibility never removes prices',(allMarkers.match(/class="daily-candle"/g)||[]).length===(noMarkers.match(/class="daily-candle"/g)||[]).length);
const longFixture={...fixture,bars:[...fixture.bars,...days.slice(11,16).map(day=>[day,100,100,100,100])]};
check('at least one historical match survives an unmatched latest date',DP.scan(longFixture,P.calendars.US,days[0],days[15]).counts.rise>0&&DP.scan(longFixture,P.calendars.US,days[0],days[15]).events.at(-1).day<days[15]);
const record={groups:['holdings']},shape={counts:{rise:0,v:2,step:0,pullback:0}},q={id:'up_high'};
check('OR within source groups',DP.matchesGroup(record,shape,q,{sources:['ai_plus','holdings'],patterns:[],quadrants:[]}));
check('OR within daily patterns',DP.matchesGroup(record,shape,q,{sources:[],patterns:['v','rise'],quadrants:[]}));
check('AND across filter families',!DP.matchesGroup(record,shape,q,{sources:['holdings'],patterns:['v'],quadrants:['up_low']}));
check('unknown quadrant cannot pass a quadrant filter',!DP.matchesGroup(record,shape,{id:null},{sources:[],patterns:[],quadrants:['up_low']}));
const derived=new Map(),ranges=[],caseRows=[];let totalEvents=0,totalBars=0;
for(const r of C.records){
 const p=prices.get(r.code),cal=P.calendars[p.market],quad=DP.quadrant(p,cal,C.as_of);
 totalBars+=p.bars.length;
 check('valid increasing daily OHLC '+r.code,p.bars.every((b,i)=>b[0]<=C.as_of&&(!i||p.bars[i-1][0]<b[0])&&b.slice(1).every(v=>Number.isFinite(v)&&v>0)&&b[2]>=Math.max(b[1],b[3],b[4])&&b[3]<=Math.min(b[1],b[2],b[4])));
 if(cal)check('verified daily dates are official sessions '+r.code,p.bars.filter(b=>b[0]>=cal.start&&b[0]<=cal.end).every(b=>cal.days.includes(b[0])));
 if(quad.id){
  const returns=p.bars.slice(-121).slice(1).map((b,i)=>Math.log(b[4]/p.bars.slice(-121)[i][4])),g=Math.log(p.bars.at(-1)[4]/p.bars.at(-121)[4])*252/120,rr=returns.slice(-60),m=rr.reduce((a,b)=>a+b,0)/60,v=Math.sqrt(rr.reduce((a,b)=>a+(b-m)**2,0)/59)*Math.sqrt(252);
  check('independent G/V formula '+r.code,Math.abs(g-quad.G)<1e-10&&Math.abs(v-quad.V)<1e-10);
  const original=D.records.find(o=>o.code===r.code);if(original?.mapped)check('same original map coordinates '+r.code,Math.abs(original.v[0]-quad.G)<1e-10&&Math.abs(original.v[1]-quad.V)<1e-10);
 }
 derived.set(r.code,{record:r,price:p,quad});
}
for(const n of [30,60,90,180]){
 const from=PC.start(C.as_of,n),counts=Object.fromEntries(DP.TYPES.map(t=>[t.id,0])),quadrants=Object.fromEntries(DP.QUADRANTS.map(t=>[t.id,0]));let unscored=0;
 for(const d of derived.values()){
  const scan=DP.scan(d.price,P.calendars[d.price.market],from,C.as_of);if(n===60)d.scan=scan;
  if(d.quad.id)quadrants[d.quad.id]++;if(!scan.scored)unscored++;
  for(const t of DP.TYPES)if(scan.counts[t.id])counts[t.id]++;
  for(const e of scan.events){
   check('event only uses complete selected prefix '+d.record.code+' '+n+' '+e.day,e.start>=from&&e.day<=C.as_of&&e.days.length===11&&e.days.at(-1)===e.day&&e.closes.length===11&&e.metrics.end>=.05-1e-12);
   check('event consists of consecutive official sessions '+d.record.code+' '+n+' '+e.day,JSON.stringify(e.days)===JSON.stringify(P.calendars[d.price.market].days.filter(day=>day>=e.start&&day<=e.day)));
   if(e.matches.includes('step')){const a=e.metrics.platform.a,b=e.metrics.platform.b,platform=e.closes.slice(a,b+1);check('actual platform supports step '+d.record.code+' '+n+' '+e.day,b-a>=2&&Math.max(...platform)/Math.min(...platform)-1<=.015+1e-12&&Math.abs(platform.at(-1)/platform[0]-1)<=.0075+1e-12);}
   if(n===60)caseRows.push({code:d.record.code,...e});totalEvents++;
  }
  if(n===60){for(const width of [700,300]){const l=PC.layout(d.price,P.calendars[d.price.market],from,C.as_of,width,null,{compact:true});check('60 day / 420 day and candle width '+d.record.code+' '+width,l.dailyDays===60&&l.weeklyDays===420&&l.weeklyFrom===PC.start(C.as_of,420)&&l.spacing>=5.5-1e-9&&l.weeklySpacing>=5.5-1e-9);}}
 }
 ranges.push({calendar_days:n,from,end:C.as_of,pattern_stock_counts:counts,quadrant_stock_counts:quadrants,unscored_stocks:unscored});
}
const clipIds=C.records.map(r=>'compare-clip-'+r.code.replace(/[^a-zA-Z0-9]/g,'-'));
check('per-card SVG clip ids are unique',new Set(clipIds).size===clipIds.length);
const html=fs.readFileSync(path.join(OUT,'index.html'),'utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];new vm.Script(script);
const begin=script.indexOf(',C=')+3,embedded=JSON.parse(script.slice(begin,script.indexOf(',DP=DailyPatterns',begin)));
check('self-contained embedded comparison matches published projection',JSON.stringify(embedded)===JSON.stringify(C));
check('two accessible native tabs and correct initial panel',html.includes('id="tab-map" role="tab"')&&html.includes('id="tab-compare" role="tab"')&&html.includes('id="compare-panel" role="tabpanel" aria-labelledby="tab-compare" hidden'));
check('comparison dates, shortcuts and compact pan sliders exist',html.includes('id="compare-from" type="date"')&&html.includes('id="compare-to" type="date"')&&[30,60,90,180].every(n=>html.includes('data-compare-days="'+n+'"'))&&html.includes('class="compare-pan-slider" type="range"')&&!html.includes('data-card-pan'));
check('stock cards have no mini navigation or its caption',!html.includes('compare-mini')&&!html.includes('调整全部股票')&&!html.includes('PC.mini(d.price'));
check('card-level case selectors and evidence panels removed',!html.includes('compare-case')&&!html.includes('日线命中窗口')&&!html.includes('命中依据与覆盖'));
check('all pattern types initially enabled through one sidebar control',html.includes('id="compare-display-options"')&&html.includes('displayPatterns:new Set(DP.TYPES.map(t=>t.id))')&&html.includes('data-pattern-display="${t.id}" checked'));
check('no wheel/drag zoom handlers',!html.includes('onwheel')&&!html.includes("addEventListener('wheel'")&&!html.includes('onpointerdown')&&!html.includes('ondrag'));
const snapshot=read('comparison_acquisition.json');check('supplement captures stay ignored and no upload',snapshot.r2_uploaded===false&&snapshot.datasets.every(d=>d.path.startsWith('market_data/trend_quadrant_v2/captures/')));
for(const code of ['US.DISK','US.IHE'])check('two missing held ETFs followed local/R2/OpenD fallback '+code,snapshot.attempts.filter(a=>a.code===code).map(a=>a.stage).join(',')==='local,r2,opend');
const unknown=C.records.filter(r=>!prices.get(r.code).bars.length).map(r=>r.code);check('all unavailable Japanese members are preserved',JSON.stringify(unknown.sort())===JSON.stringify(['JP.2644','JP.4063','JP.4186']));
const cache=path.join(ROOT,'.cache/ai_trend_quadrant_v2');fs.mkdirSync(cache,{recursive:true});
function colors(svg){return svg.replaceAll('var(--bg)','#fff').replaceAll('var(--fg)','#262626').replaceAll('var(--secondary)','#626262').replaceAll('var(--muted)','#efefef').replaceAll('var(--line)','#c7c7c7');}
function makePreview(name,width,columns,codes){
 const pad=18,gap=16,cardWidth=(width-pad*2-gap*(columns-1))/columns,chartWidth=cardWidth-24,cardHeight=390,top=270,height=top+Math.ceil(codes.length/columns)*(cardHeight+gap)+pad;
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><style>text{font-family:system-ui,sans-serif;font-size:12px;fill:#262626}</style><rect width="100%" height="100%" fill="white"/><text x="18" y="30" style="font-size:20px">股票观察 · 分布与多股对比</text><rect x="18" y="52" width="106" height="32" fill="white" stroke="#c7c7c7"/><text x="27" y="74">四象限分布</text><rect x="136" y="52" width="112" height="32" fill="#262626"/><text x="145" y="74" style="fill:white">多股票对比</text><text x="18" y="110" style="font-size:17px">日线 60 天 × 周线 420 天</text><text x="18" y="133">2026-08-08 → 2026-10-06</text><text x="18" y="156">持仓 23　自选 167　特别关注 53</text><text x="18" y="179">AI 产业池 A+ 6　A 303</text><text x="18" y="202">日线形态：▲ / V / P / S　· 范围内至少一次</text><text x="18" y="225">${columns===2?'桌面多卡片':'窄屏单列'}静态 SVG 排版预览 · 非浏览器截图</text>`;
 for(let i=0;i<codes.length;i++){
  const d=derived.get(codes[i]),x=pad+(i%columns)*(cardWidth+gap),y=top+Math.floor(i/columns)*(cardHeight+gap),l=PC.layout(d.price,P.calendars[d.price.market],PC.start(P.as_of,60),P.as_of,chartWidth,null,{compact:true}),event=d.scan.events.at(-1)||null;
  check('marker IDs and dates in actual card '+d.record.code,PC.svg(l,true,null,{id:clipIds[C.records.indexOf(d.record)],events:d.scan.events,selectedEvent:event}).includes('id="'+clipIds[C.records.indexOf(d.record)]+'"'));
  svg+=`<g transform="translate(${x},${y})"><rect width="${cardWidth}" height="${cardHeight}" fill="white" stroke="#c7c7c7"/><text x="12" y="25" style="font-size:15px">${d.record.code}</text><rect x="${cardWidth-62}" y="10" width="50" height="28" fill="white" stroke="#c7c7c7"/><text x="${cardWidth-52}" y="29">详情</text>${l.maxOffset?`<line x1="${cardWidth-158}" x2="${cardWidth-74}" y1="24" y2="24" stroke="#c7c7c7"/><circle cx="${cardWidth-74}" cy="24" r="4" fill="#626262"/>`:``}<text x="12" y="52">${DP.QUADRANTS.find(t=>t.id===d.quad.id)?.name||'象限待确认'} · 持仓</text><g transform="translate(12,67)">${colors(PC.svg(l,true,l.u1-1e-10,{id:'preview-'+i,events:d.scan.events}))}</g></g>`;
 }
 fs.writeFileSync(path.join(cache,name+'.svg'),svg+'</svg>');
}
makePreview('comparison-desktop',1300,2,['US.AMAT','US.AVGO','US.CIEN','US.GOOG']);
makePreview('comparison-mobile',390,1,['US.AVGO','US.CIEN']);
fs.writeFileSync(path.join(OUT,'comparison_cases.json'),JSON.stringify({version:DP.VERSION,from:PC.start(P.as_of,60),end:P.as_of,events:caseRows},null,2)+'\n');
const report={view_revision:'five_traits_comparison_dense_v1',verified_at:new Date().toISOString(),verification_kind:'offline_memberships_daily_patterns_filters_and_SVG',live_browser_verified:false,
 summary:C.summary,total_daily_bars:totalBars,total_matched_windows_checked:totalEvents,unknown_price_members:unknown,ranges,checks,errors:[],files:{comparison:sha(path.join(OUT,'comparison.json')),daily_patterns:sha(path.join(OUT,'daily_patterns.js')),comparison_ui:sha(path.join(OUT,'comparison_ui.js')),index:sha(path.join(OUT,'index.html'))}};
fs.writeFileSync(path.join(OUT,'comparison_verification.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:checks.length,members:C.summary.members,with_prices:C.summary.with_prices,default60:ranges[1],live_browser_verified:false}));
