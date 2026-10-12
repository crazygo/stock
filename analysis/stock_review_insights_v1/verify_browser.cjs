// Browser acceptance for a generated report; Playwright is supplied by the local runtime.
const fs=require('fs'),path=require('path'),{pathToFileURL,fileURLToPath}=require('url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || path.join(process.env.HOME,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const report=path.resolve(process.argv[2]),out=path.resolve(process.argv[3] || path.join(path.dirname(report),'validation/browser'));
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true}),results=[];
 for(const width of [1280,1024,980,768,360,320]){
  const page=await browser.newPage({viewport:{width,height:1000},offline:true}),errors=[],network=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
  await page.goto(pathToFileURL(report).href);
  await page.waitForFunction(()=>document.querySelectorAll('.iev-svg').length>0);
  const check=await page.evaluate(()=>{
   const root=document.querySelector('#industry-evidence-viz'),data=JSON.parse(root.querySelector('#industry-evidence-data').textContent),rows=data.funds.concat(data.groups);
   const mismatches=[],geometry=[];
   const visibleRows=[...root.querySelectorAll('.iev-data-row')];
   for(const row of rows){
    const tr=visibleRows.find(r=>r.dataset.row===row.id);
    if(!tr || tr.querySelectorAll('.iev-name').length!==1 || tr.querySelectorAll('td').length!==4 || !tr.querySelector('td:first-of-type[data-candle-cell]'))mismatches.push([row.id,'shared row/candle column']);
    for(const day of ['5','20','60']){
     const cell=tr.querySelector('td[data-day="'+day+'"]'),v=row.values[day],mark=cell.querySelector('[data-value-mark]');
     if((v.value===null && mark) || (v.value!==null && (!mark || Number(mark.dataset.value)!==v.value)))mismatches.push([row.id,day,'price']);
     if(cell.querySelector('[data-direct-label]').textContent!==v.display || !cell.querySelector('svg').getAttribute('aria-label').includes(v.display))mismatches.push([row.id,day,'label']);
     if(v.scored!==undefined && (v.scored>0?!cell.querySelector('[data-breadth-label]').textContent.includes(v.up+'/'+v.scored+'上涨'):!cell.querySelector('[data-breadth-label]').textContent.includes('不可评分')))mismatches.push([row.id,day,'breadth']);
     if(v.unknown && !cell.querySelector('[data-unknown-label]').textContent.includes('未知'+v.unknown))mismatches.push([row.id,day,'unknown']);
     if(day!=='5')for(const key of ['direction','r2','er']){
      const actual=cell.querySelector('[data-metric="'+key+'"]'),expected=row.chart.metrics[day][key];
      if(!actual || actual.dataset.metricValue!==(expected===null?'':String(expected)))mismatches.push([row.id,day,key]);
     }
     const svg=cell.querySelector('svg'),zero=Number(svg.querySelector('[data-zero-line]').getAttribute('x1')),width=svg.viewBox.baseVal.width;
     if(Math.abs(zero-width/2)>.01)geometry.push([row.id,day,'zero not centered']);
     if(mark){
      const x=Number(mark.getAttribute('x')),w=Number(mark.getAttribute('width')),expected=Math.abs(v.value)/Number(mark.dataset.limit)*(width/2-76);
      if(Math.abs(w-expected)>.01 || (v.value<0?Math.abs(x+w-zero)>.01:Math.abs(x-zero)>.01))geometry.push([row.id,day,'direction/length']);
      if(v.value>0&&!mark.getAttribute('fill').startsWith('hsl(3 '))geometry.push([row.id,day,'positive not red']);
      if(v.value<0&&!mark.getAttribute('fill').startsWith('hsl(145 '))geometry.push([row.id,day,'negative not green']);
     }
    }
   }
   const outside=[...root.querySelectorAll('.iev-svg')].flatMap(s=>[...s.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<-.5||b.x+b.width>s.viewBox.baseVal.width+.5}).map(t=>t.textContent));
   const limits=[...new Set([...root.querySelectorAll('[data-value-mark]')].map(m=>m.dataset.limit))];
   if(limits.length>1)geometry.push('inconsistent shared scale');
   const unknownsInsideCells=[...root.querySelectorAll('[data-unknown-label]')].every(t=>t.closest('td[data-day]'));
   return {marks:root.querySelectorAll('[data-value-mark]').length,expected:rows.reduce((n,r)=>n+['5','20','60'].filter(d=>r.values[d].value!==null).length,0),mismatches,
    outside,geometry,unknown_label_overlap:!unknownsInsideCells,expected_rows:rows.length,shared_headers:root.querySelectorAll('.iev-table thead [data-period]').length,unique_row_labels:root.querySelectorAll('.iev-name').length,invalid:/NaN|undefined/.test(root.innerText),overflow:document.documentElement.scrollWidth>innerWidth+1,
    source_tables:root.querySelector('.iev-source').querySelectorAll('table').length,candle_column:root.querySelector('.iev-table thead tr').children[1].hasAttribute('data-k-header'),market_cutoffs:data.market_cutoff,
    unknowns:root.querySelectorAll('[data-unknown-label]').length,detail_rows:document.querySelectorAll('#rows tr').length};
  });
  if(check.marks!==check.expected||check.mismatches.length||check.geometry.length||check.shared_headers!==3||!check.candle_column||check.unique_row_labels!==check.expected_rows||check.outside.length||check.unknown_label_overlap||check.invalid||check.overflow||check.source_tables!==2||errors.length||network.length)throw Error(JSON.stringify({width,check,errors,network}));
  const auditCandles=async()=>page.evaluate(()=>{
   const root=document.querySelector('#industry-evidence-viz'),data=JSON.parse(root.querySelector('#industry-evidence-data').textContent),byId=new Map(data.funds.concat(data.groups,Object.values(data.security_rows)).map(r=>[r.id,r])),issues=[];
   for(const tr of root.querySelectorAll('.iev-data-row')){
    if(tr.hidden)continue;
    const row=byId.get(tr.dataset.securityId),svg=tr.querySelector('.iev-k-svg'),expected=new Map(row.chart.bars.map(b=>[b[0],b.slice(1)]));
    if(svg.dataset.candleSpacing&&Number(svg.dataset.candleSpacing)<5)issues.push([row.id,'candles under 5px']);
    const displayed=[...svg.querySelectorAll('[data-candle-date]')];
    for(const g of displayed){
     if(JSON.stringify(expected.get(g.dataset.candleDate))!==g.dataset.ohlc)issues.push([row.id,'OHLC']);
     if(g.dataset.candleDate>data.market_cutoff[row.market])issues.push([row.id,'future candle']);
     if(row.chart.kind==='basket_bounds'&&g.querySelectorAll('[data-wick-bound]').length!==2)issues.push([row.id,'unmarked range bound']);
     if(row.chart.kind!=='basket_bounds'&&g.querySelector('[data-wick-bound]'))issues.push([row.id,'true OHLC incorrectly dashed']);
    }
    const selected=Number(root.querySelector('[data-k-days][aria-pressed="true"]').dataset.kDays),wanted=row.chart.dates.slice(-selected).filter(d=>expected.has(d)).length;
    if(displayed.length!==wanted)issues.push([row.id,'candle count']);
    if(tr.querySelectorAll('td').length!==4)issues.push([row.id,'child columns']);
    for(const day of ['20','60'])for(const metric of ['direction','r2','er']){
     const field=tr.querySelector('td[data-day="'+day+'"] [data-metric="'+metric+'"]'),value=row.chart.metrics[day][metric];
     if(!field||field.dataset.metricValue!==(value===null?'':String(value)))issues.push([row.id,day,metric]);
    }
   }return issues;
  });
  if((await auditCandles()).length)throw Error('daily candle/statistics audit failed: '+JSON.stringify(await auditCandles()));
  await page.locator('[data-k-days="60"]').click();
  if((await auditCandles()).length)throw Error('60-session candle audit failed: '+JSON.stringify(await auditCandles()));
  await page.locator('[data-k-days="20"]').click();
  const buttons=page.locator('#industry-evidence-viz .iev-parent-row button[data-security-group]');
  const buttonCount=await buttons.count();
  await buttons.nth(Math.min(3,buttonCount-1)).click();const selection=await page.locator('.iev-selection').innerText();
  if(!selection.includes('5日')||!selection.includes('20日')||!selection.includes('60日'))throw Error('three-period selection failed');
  const keyboardButton=buttons.nth(Math.min(4,buttonCount-1));
  await keyboardButton.focus();await page.keyboard.press('Enter');
  if(await keyboardButton.getAttribute('aria-pressed')!=='true')throw Error('keyboard selection failed');
  if(width===1280){
   const parents=await page.locator('.iev-parent-row').evaluateAll(trs=>trs.map(tr=>tr.dataset.row));
   for(const id of parents){
    const button=page.locator('.iev-parent-row').filter({has:page.locator('button[data-security-group="'+id+'"]')}).locator('button');
    if(await button.getAttribute('aria-expanded')!=='true')await button.click();
    const children=await page.evaluate(id=>{const root=document.querySelector('#industry-evidence-viz'),data=JSON.parse(root.querySelector('#industry-evidence-data').textContent),parent=data.funds.concat(data.groups).find(r=>r.id===id);return {expected:parent.children,actual:[...root.querySelectorAll('.iev-child-row')].filter(tr=>tr.dataset.row.startsWith(id+'::')&&!tr.hidden).map(tr=>tr.dataset.securityId)}},id);
    if(JSON.stringify(children.expected)!==JSON.stringify(children.actual))throw Error('constituent membership mismatch: '+id);
    const issues=await auditCandles();if(issues.length)throw Error(JSON.stringify({id,issues}));
    if(id==='US-compute')await page.locator('#industry-evidence-viz').screenshot({path:path.join(out,'expanded-compute.png')});
    await button.click();
   }
  }
  await page.locator('.iev-source summary').click();
  if(!await page.locator('.iev-source table').first().isVisible())throw Error('source tables did not expand');
  await page.locator('.iev-source summary').click();
  const firstStock=page.locator('#rows tr a').first(),firstCode=(await firstStock.innerText()).split(/\s+/)[0];
  await page.locator('#search').fill(firstCode+' ');
  if(!await page.locator('#count').innerText().then(t=>t.startsWith('1 /')))throw Error('stock directory search regressed');
  await page.locator('#search').fill('');
  if(width<=360){
   const wrap=page.locator('.iev-table-wrap');await wrap.focus();await page.keyboard.press('ArrowRight');await page.waitForFunction(()=>document.querySelector('.iev-table-wrap').scrollLeft>0);
   await wrap.evaluate(e=>e.scrollLeft=e.scrollWidth);await page.locator('#industry-evidence-viz').screenshot({path:path.join(out,'industry-'+width+'-scrolled.png')});await wrap.evaluate(e=>e.scrollLeft=0);
  }
  if(width===1280||width===360)await page.locator('#industry-evidence-viz').screenshot({path:path.join(out,'industry-'+width+'.png')});
  if(width===1280){
   const links=await page.locator('a[href]').evaluateAll(as=>as.map(a=>a.href));
   const broken=links.filter(u=>u.startsWith('file:')).filter(u=>!fs.existsSync(fileURLToPath(new URL(u))));
   if(broken.length)throw Error('broken local report links: '+JSON.stringify(broken));
   await firstStock.click();
   await page.locator('a[href="../index.html"]').first().click();
   await page.waitForFunction(()=>document.querySelectorAll('.iev-svg').length>0);
   if(!page.url().startsWith(pathToFileURL(report).href))throw Error('details returned to the old table report');
  }
  results.push({width,...check,errors,network,selection:true,keyboard:true,original_tables:true,stock_filter:true,detail_return:width===1280?'pass':'checked_at_1280'});
  await page.close();
 }
 await browser.close();
 const result={status:'pass',offline:true,results};fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify({status:'pass',widths:results.map(r=>r.width),marks:results[0].marks,offline:true,detail_return:'pass'}));
})().catch(e=>{console.error(e);process.exit(1)});
