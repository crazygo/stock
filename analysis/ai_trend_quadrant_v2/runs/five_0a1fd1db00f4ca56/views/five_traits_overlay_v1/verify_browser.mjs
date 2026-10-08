import {createRequire} from 'node:module';import {pathToFileURL} from 'node:url';import fs from 'node:fs/promises';import {dirname,resolve} from 'node:path';import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url),{chromium}=require(process.env.PLAYWRIGHT_MODULE||'/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const out=dirname(fileURLToPath(import.meta.url)),cache=resolve(out,'../../.cache/ai_trend_quadrant_v2'),checks=[],errors=[];await fs.mkdir(cache,{recursive:true});
const d=JSON.parse(await fs.readFile(out+'/results.json','utf8'));const browser=await chromium.connectOverCDP(process.env.CDP_ENDPOINT||'http://127.0.0.1:19223');const session=await browser.newBrowserCDPSession();let page;
function check(name,actual,expected){const passed=JSON.stringify(actual)===JSON.stringify(expected);checks.push({name,passed,actual,expected});if(!passed)throw Error(name+': '+JSON.stringify(actual));}
try{
 const url=pathToFileURL(out+'/index.html').href+'?verification='+Date.now();await session.send('Target.createTarget',{url,background:true});
 for(let i=0;i<30&&!page;i++){page=browser.contexts().flatMap(c=>c.pages()).find(p=>p.url()===url);if(!page)await new Promise(r=>setTimeout(r,100));}
 if(!page)throw Error('preview target missing');page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 await page.setViewportSize({width:1440,height:1000});await page.waitForSelector('#map [data-code]');await page.locator('#pool').selectOption('all');
 check('all 150 valid map coordinates',await page.locator('#map [data-code]').count(),150);
 check('drawer initially closed',await page.locator('#drawer').isVisible(),false);
 await page.locator('#security').selectOption('US.NVDA');check('select opens floating drawer',await page.locator('#drawer').isVisible(),true);
 check('five indicator charts',await page.locator('.trait-svg').count(),5);
 check('five traits ordered once',await page.locator('.trait-svg').evaluateAll(v=>v.map(s=>s.dataset.trait)),d.keys);
 check('each indicator has 14 30 60 lines',await page.locator('.trait-svg').evaluateAll(v=>v.map(s=>[...s.querySelectorAll('path[data-period]')].map(p=>Number(p.dataset.period)).sort((a,b)=>a-b))),Array(5).fill([14,30,60]));
 check('three calendar starts',await page.locator('.trait-svg[data-trait="G"] path').evaluateAll(v=>v.map(p=>[Number(p.dataset.period),p.dataset.start]).sort((a,b)=>a[0]-b[0])),[[14,'2026-09-23'],[30,'2026-09-07'],[60,'2026-08-08']]);
 check('all lines end at data cutoff',await page.locator('.trait-svg path').evaluateAll(v=>v.every(s=>s.dataset.end==='2026-10-06')),true);
 check('all curves share their indicator scale',await page.locator('.trait-svg').evaluateAll(v=>v.every(s=>[...s.querySelectorAll('path')].every(p=>p.dataset.domain===s.dataset.domain))),true);
 check('default aligns cycle progress',await page.locator('#time-axis').inputValue(),'progress');
 const actual=await page.locator('.trait-svg').evaluateAll(v=>v.flatMap(s=>[...s.querySelectorAll('path')].map(p=>({trait:s.dataset.trait,period:Number(p.dataset.period),start:p.dataset.start,end:p.dataset.end,domain:JSON.parse(p.dataset.domain),samples:JSON.parse(p.dataset.samples),frame:{x:Number(s.querySelector('rect').getAttribute('x')),w:Number(s.querySelector('rect').getAttribute('width'))}}))));
 const reference=d.records.find(r=>r.code==='US.NVDA');
 check('line samples preserve captured historical values',actual.every(p=>{const k=d.keys.indexOf(p.trait),expected=reference.history.filter(h=>h.day>=p.start&&h.day<=p.end&&h.v[k]!=null);return expected.length===p.samples.length&&expected.every((h,i)=>h.day===p.samples[i].day&&Math.abs(h.v[k]-p.samples[i].value)<.0000006)}),true);
 check('progress mapping uses each calendar range',actual.every(p=>p.samples.every(q=>Math.abs(q.x-(p.frame.x+(Date.parse(q.day)-Date.parse(p.start))/(Date.parse(p.end)-Date.parse(p.start))*p.frame.w))<1e-8)),true);
 const scalesBefore=await page.locator('.trait-svg').evaluateAll(v=>v.map(s=>s.dataset.domain));
 await page.locator('#period-legend input[value="60"]').uncheck();
 check('legend hides 60-day curve in all charts',await page.locator('.trait-svg path[data-period="60"]').evaluateAll(v=>v.every(p=>getComputedStyle(p).display==='none')),true);
 check('hiding a period keeps scales',await page.locator('.trait-svg').evaluateAll(v=>v.map(s=>s.dataset.domain)),scalesBefore);
 await page.locator('#period-legend input[value="60"]').check();
 await page.locator('#time-axis').selectOption('age');
 check('date alignment uses common calendar positions',await page.locator('.trait-svg').evaluateAll(v=>v.every(s=>{const paths=[...s.querySelectorAll('path')].map(p=>JSON.parse(p.dataset.samples)),seen=new Map();return paths.every(a=>a.every(q=>{const old=seen.get(q.day);if(old&&(Math.abs(old.x-q.x)>1e-8||Math.abs(old.y-q.y)>1e-8))return false;seen.set(q.day,q);return true}))})),true);
 await page.screenshot({path:cache+'/v2-date-overlay.png',fullPage:false});
 for(const n of [14,30,60])await page.locator(`#period-legend input[value="${n}"]`).uncheck();
 check('all hidden state prompts selection',await page.locator('.trait-svg').evaluateAll(v=>v.every(s=>s.textContent.includes('勾选周期显示曲线'))),true);
 for(const n of [14,30,60])await page.locator(`#period-legend input[value="${n}"]`).check();
 await page.locator('#time-axis').selectOption('progress');
 await page.locator('#detail summary').click();check('G chart scale follows historical changes',(await page.locator('.trait-svg[data-trait="G"]').first().evaluate(s=>{const [a,b]=JSON.parse(s.dataset.domain);return b-a<1})),true);check('current five-trait detail rows',await page.locator('#detail-table tbody tr').count(),5);
 await page.locator('#detail summary').click();await page.screenshot({path:cache+'/v2-desktop.png',fullPage:false});
 await page.keyboard.press('Escape');check('Escape closes drawer',await page.locator('#drawer').isVisible(),false);check('focus returns to security selector',await page.evaluate(()=>document.activeElement.id),'security');
 const nvdaBefore=await page.locator('#map [data-code="US.NVDA"]').getAttribute('cx');
 await page.locator('#map [data-code="US.QQQ"]').click();if(await page.locator('#cluster').isVisible())await page.locator('#cluster button').filter({hasText:'US.QQQ ·'}).click();
 check('clicking actual map point opens its security',(await page.locator('#drawer-name').innerText()).startsWith('US.QQQ ·'),true);
 await page.locator('#close').click();await page.locator('#pool').selectOption('quality');check('quality filter map points',await page.locator('#map [data-code]').count(),11);
 check('filter does not move coordinates',await page.locator('#map [data-code="US.NVDA"]').getAttribute('cx'),nvdaBefore);
 await page.locator('#pool').selectOption('all');await page.locator('#security').selectOption('US.LYTE');check('insufficient-axis security retains drawer',await page.locator('#drawer').isVisible(),true);check('missing values retained',(await page.locator('#drawer-warning').innerText()).includes('地图坐标不足'),true);
 await page.locator('#security').selectOption('JP.4063');check('unavailable security retains five blank charts',await page.locator('.trait-svg').count(),5);check('unavailable charts keep unknown rather than zero',await page.locator('.trait-svg').evaluateAll(v=>v.every(s=>[...s.querySelectorAll('path')].every(p=>p.getAttribute('d')==='')&&s.textContent.includes('行情不可用'))),true);
 await page.locator('#security').selectOption('HK.03006');check('zero-volume caveat displayed',(await page.locator('#drawer-warning').innerText()).includes('日线量为零'),true);
 await page.locator('#security').selectOption('US.NVDA');
 for(const width of [1024,390,320]){
  await page.setViewportSize({width,height:900});await page.waitForTimeout(350);
  const dims=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,drawer:document.querySelector('#drawer').getBoundingClientRect().right,outside:[...document.querySelectorAll('.trait-svg text')].filter(t=>{const s=t.ownerSVGElement.getBoundingClientRect(),b=t.getBoundingClientRect();return b.left<s.left-.5||b.right>s.right+.5||b.top<s.top-.5||b.bottom>s.bottom+.5}).map(t=>t.textContent)}));
  check('no page overflow '+width,dims.scroll<=width,true);check('drawer fits viewport '+width,dims.drawer<=width,true);check('time labels fit '+width,dims.outside,[]);
  const cols=await page.locator('#spectra').evaluate(e=>getComputedStyle(e).gridTemplateColumns.split(' ').length);check('one chart per indicator at '+width,cols,1);check('all three lines retained at '+width,await page.locator('.trait-svg path[data-period]').count(),15);
  await page.screenshot({path:cache+'/v2-'+width+'.png',fullPage:false});
 }
 await page.emulateMedia({colorScheme:'dark'});await page.waitForTimeout(120);check('dark mode canvas switches',await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()),'#1c1c1c');await page.screenshot({path:cache+'/v2-dark.png',fullPage:false});
 check('no console or JavaScript errors',errors,[]);
}catch(e){errors.push(e.message);process.exitCode=1}
finally{await fs.writeFile(out+'/browser_verification.json',JSON.stringify({run_id:d.run_id,view_revision:'five_traits_overlay_v1',verified_at:new Date().toISOString(),checks,errors,screenshots_directory:cache},null,2));console.log(JSON.stringify({passed:checks.filter(c=>c.passed).length,total:checks.length,errors}));if(page)await page.close();await session.detach();process.exit(process.exitCode||0)}
