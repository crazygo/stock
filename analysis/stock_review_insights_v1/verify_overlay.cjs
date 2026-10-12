// Acceptance of the real offline report: filters, mixed lines, keyboard, gaps and log axis.
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||path.join(process.env.HOME,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const report=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]||path.join(path.dirname(report),'validation/overlay'));
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true}),results=[];
 for(const width of [1280,1024,980,768,360,320]){
  const page=await browser.newPage({viewport:{width,height:1000},offline:true}),errors=[],network=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
  await page.goto(pathToFileURL(report).href);await page.waitForFunction(()=>document.querySelector('[data-overlay-path]'));
  const audit=async()=>page.evaluate(()=>{
   const root=document.getElementById('industry-evidence-viz'),panel=root.querySelector('.iev-overlay'),data=JSON.parse(root.querySelector('#industry-evidence-data').textContent),svg=panel.querySelector('svg');
   const parents=data.funds.concat(data.groups),byId=new Map(Object.values(data.security_rows).map(r=>[r.id,r]));parents.forEach(r=>byId.set(r.id,r));
   const mode=svg.dataset.overlayPlottedMode,days=Number(svg.dataset.overlaySessions),field=mode==='price'?2:4,issues=[];
   const zero=svg.querySelector('[data-overlay-zero]');if(!zero||Math.abs(Number(zero.getAttribute('y1'))-(32+(svg.viewBox.baseVal.height-42))/2)>.01)issues.push('zero not centered');
   for(const line of svg.querySelectorAll('[data-overlay-path]')){
    const row=byId.get(line.dataset.overlaySeries),points=row.chart.overlay.points.slice(-days).map(p=>[p[0],p[field]]);
    if(line.dataset.overlayPoints!==JSON.stringify(points))issues.push([row.id,'values/window']);
    if(points.some(p=>p[0]>data.market_cutoff[row.market]))issues.push([row.id,'future']);
    const segments=points.reduce((n,p,i)=>n+(p[1]!==null&&(i===0||points[i-1][1]===null)?1:0),0);
    if((line.getAttribute('d').match(/M/g)||[]).length!==segments)issues.push([row.id,'gap bridged']);
    if(!Number.isFinite(Number(svg.dataset.overlayLimit))||/NaN|Infinity/.test(line.getAttribute('d')))issues.push([row.id,'invalid coordinate']);
   }
   const outside=[...svg.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<-.5||b.x+b.width>svg.viewBox.baseVal.width+.5||b.y<-.5||b.y+b.height>svg.viewBox.baseVal.height+.5}).map(t=>t.textContent);
   return {lines:svg.querySelectorAll('[data-overlay-path]').length,series:[...svg.querySelectorAll('[data-overlay-path]')].map(p=>p.dataset.overlaySeries),issues,outside,overflow:document.documentElement.scrollWidth>innerWidth+1,default_sessions:days,mode};
  });
  let check=await audit();if(check.issues.length||check.outside.length||check.overflow||check.default_sessions!==60||check.mode!=='price')throw Error(JSON.stringify({width,check}));
  const defaults=await page.evaluate(()=>{const d=JSON.parse(document.querySelector('#industry-evidence-data').textContent);return d.funds.concat(d.groups).filter(r=>r.chart.overlay.status==='complete').map(r=>r.id);});
  if(JSON.stringify(check.series)!==JSON.stringify(defaults))throw Error('default line set mismatch');
  const categories=await page.locator('[data-overlay-category] option').evaluateAll(xs=>xs.map(x=>x.value));
  for(const category of categories){
   await page.locator('[data-overlay-category]').selectOption(category);check=await audit();
   const expected=await page.evaluate(category=>{const d=JSON.parse(document.querySelector('#industry-evidence-data').textContent);return d.sections.filter(s=>category==='all'||s.id===category).flatMap(s=>s.rows).filter(r=>r.chart.overlay.status==='complete').map(r=>r.id).sort();},category);
   if(JSON.stringify([...check.series].sort())!==JSON.stringify(expected)||check.issues.length||check.outside.length||check.overflow)throw Error(JSON.stringify({width,category,check,expected}));
  }
  await page.locator('[data-overlay-category]').selectOption('US-groups');
  await page.locator('.iev-overlay-picker summary').click();await page.locator('[data-overlay-search]').fill('NVDA');
  const stock=page.locator('[data-overlay-select="US.NVDA"]');await stock.check();check=await audit();
  if(!check.series.includes('US.NVDA')||!check.series.some(id=>id.startsWith('US-'))||check.issues.length)throw Error('mixed group and stock failed');
  await page.locator('[data-overlay-days="20"]').click();check=await audit();if(check.default_sessions!==20||check.issues.length)throw Error('window switch failed');
  await page.locator('[data-overlay-mode]').selectOption('daily');check=await audit();if(check.mode!=='daily'||check.issues.length||check.outside.length)throw Error('daily mean/log switch failed');
  await page.locator('[data-overlay-mode]').selectOption('price');await page.locator('[data-overlay-days="60"]').click();
  const graph=page.locator('.iev-overlay-svg');await graph.focus();await page.keyboard.press('Home');
  const first=await page.locator('[data-overlay-readout]').innerText();await page.keyboard.press('ArrowRight');
  const next=await page.locator('[data-overlay-readout]').innerText();await page.keyboard.press('ArrowDown');const other=await page.locator('[data-overlay-readout]').innerText();
  if(first===next||next===other)throw Error('keyboard date/series failed');
  await page.locator('[data-overlay-search]').fill('');
  if(width===1280){
   await page.locator('[data-overlay-category]').selectOption('all');await page.locator('[data-overlay-all]').click();check=await audit();
   const missing=await page.locator('[data-overlay-select]:disabled').count();if(check.issues.length||check.lines<=defaults.length||!missing)throw Error('all constituents/missing failed');
   await page.locator('[data-overlay-clear]').click();if((await audit()).lines!==0||!(await page.locator('[data-overlay-readout]').innerText()).includes('没有选中'))throw Error('empty selection failed');
  }
  await page.locator('[data-overlay-reset]').click();check=await audit();if(JSON.stringify(check.series)!==JSON.stringify(defaults)||check.default_sessions!==60||check.mode!=='price')throw Error('reset failed');
  await page.locator('[data-overlay-category]').selectOption('US-groups');await page.locator('[data-overlay-search]').fill('NVDA');await page.locator('[data-overlay-select="US.NVDA"]').check();
  await page.locator('.iev-overlay-picker summary').click();
  await page.locator('.iev-overlay').screenshot({path:path.join(out,'overlay-'+width+'.png')});
  if(width===1280){await page.locator('[data-overlay-reset]').click();await page.locator('.iev-overlay').screenshot({path:path.join(out,'overlay-all.png')});}
  check=await audit();if(check.issues.length||check.outside.length||check.overflow||errors.length||network.length)throw Error(JSON.stringify({width,check,errors,network}));
  results.push({width,...check,filters:true,mixed_lines:true,window_switch:true,mode_switch:true,keyboard:true,reset:true,errors,network});await page.close();
 }
 await browser.close();fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({status:'pass',offline:true,results},null,2));console.log(JSON.stringify({status:'pass',offline:true,widths:results.map(r=>r.width),mixed_lines:true}));
})().catch(e=>{console.error(e);process.exit(1)});
