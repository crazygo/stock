import {createRequire} from 'node:module';import {pathToFileURL} from 'node:url';import fs from 'node:fs/promises';import {dirname,resolve} from 'node:path';import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url),{chromium}=require(process.env.PLAYWRIGHT_MODULE||'/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const out=dirname(fileURLToPath(import.meta.url)),cache=resolve(out,'../../.cache/ai_trend_quadrant_v2'),checks=[],errors=[];await fs.mkdir(cache,{recursive:true});
const d=JSON.parse(await fs.readFile(out+'/results.json','utf8'));const browser=await chromium.connectOverCDP(process.env.CDP_ENDPOINT||'http://127.0.0.1:19223');const session=await browser.newBrowserCDPSession();let page;
function check(name,actual,expected){const passed=JSON.stringify(actual)===JSON.stringify(expected);checks.push({name,passed,actual,expected});if(!passed)throw Error(name+': '+JSON.stringify(actual));}
try{
 const url=pathToFileURL(out+'/index.html').href;await session.send('Target.createTarget',{url,background:true});
 for(let i=0;i<30&&!page;i++){page=browser.contexts().flatMap(c=>c.pages()).find(p=>p.url()===url);if(!page)await new Promise(r=>setTimeout(r,100));}
 if(!page)throw Error('preview target missing');page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 await page.setViewportSize({width:1440,height:1000});await page.waitForSelector('#map [data-code]');await page.locator('#pool').selectOption('all');
 check('all 150 valid map coordinates',await page.locator('#map [data-code]').count(),150);
 check('drawer initially closed',await page.locator('#drawer').isVisible(),false);
 await page.locator('#security').selectOption('US.NVDA');check('select opens floating drawer',await page.locator('#drawer').isVisible(),true);
 check('three periods simultaneously visible',await page.locator('.period-heading h3').allTextContents(),['最近 14 天','最近 30 天','最近 60 天']);
 check('fifteen charts retain all five traits',await page.locator('.trait-svg').count(),15);
 check('calendar period starts',await page.locator('.period').evaluateAll(v=>v.map(e=>e.querySelector('svg').dataset.start)),['2026-09-23','2026-09-07','2026-08-08']);
 check('all charts end at completed data cutoff',await page.locator('.trait-svg').evaluateAll(v=>v.every(s=>s.dataset.end==='2026-10-06')),true);
 for(const key of d.keys){const domains=await page.locator(`.trait-svg[data-trait="${key}"]`).evaluateAll(v=>v.map(s=>s.dataset.domain));check('shared '+key+' scale across periods',new Set(domains).size,1);}
 await page.locator('#detail summary').click();check('G chart scale follows historical changes',(await page.locator('.trait-svg[data-trait="G"]').first().evaluate(s=>{const [a,b]=JSON.parse(s.dataset.domain);return b-a<1})),true);check('current five-trait detail rows',await page.locator('#detail-table tbody tr').count(),5);
 await page.locator('#detail summary').click();await page.screenshot({path:cache+'/v2-desktop.png',fullPage:false});
 await page.keyboard.press('Escape');check('Escape closes drawer',await page.locator('#drawer').isVisible(),false);check('focus returns to security selector',await page.evaluate(()=>document.activeElement.id),'security');
 const nvdaBefore=await page.locator('#map [data-code="US.NVDA"]').getAttribute('cx');
 await page.locator('#map [data-code="US.QQQ"]').click();if(await page.locator('#cluster').isVisible())await page.locator('#cluster button').filter({hasText:'US.QQQ ·'}).click();
 check('clicking actual map point opens its security',(await page.locator('#drawer-name').innerText()).startsWith('US.QQQ ·'),true);
 await page.locator('#close').click();await page.locator('#pool').selectOption('quality');check('quality filter map points',await page.locator('#map [data-code]').count(),11);
 check('filter does not move coordinates',await page.locator('#map [data-code="US.NVDA"]').getAttribute('cx'),nvdaBefore);
 await page.locator('#pool').selectOption('all');await page.locator('#security').selectOption('US.LYTE');check('insufficient-axis security retains drawer',await page.locator('#drawer').isVisible(),true);check('missing values retained',(await page.locator('#drawer-warning').innerText()).includes('地图坐标不足'),true);
 await page.locator('#security').selectOption('JP.4063');check('unavailable security retains all blank lanes',await page.locator('.trait-svg').count(),15);
 await page.locator('#security').selectOption('HK.03006');check('zero-volume caveat displayed',(await page.locator('#drawer-warning').innerText()).includes('日线量为零'),true);
 await page.locator('#security').selectOption('US.NVDA');
 for(const width of [1024,390,320]){
  await page.setViewportSize({width,height:900});await page.waitForTimeout(350);
  const dims=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,drawer:document.querySelector('#drawer').getBoundingClientRect().right,outside:[...document.querySelectorAll('.trait-svg text')].filter(t=>{const s=t.ownerSVGElement.getBoundingClientRect(),b=t.getBoundingClientRect();return b.left<s.left-.5||b.right>s.right+.5||b.top<s.top-.5||b.bottom>s.bottom+.5}).map(t=>t.textContent)}));
  check('no page overflow '+width,dims.scroll<=width,true);check('drawer fits viewport '+width,dims.drawer<=width,true);check('time labels fit '+width,dims.outside,[]);
  const cols=await page.locator('#periods').evaluate(e=>getComputedStyle(e).gridTemplateColumns.split(' ').length);check('responsive period layout '+width,cols,width<=620?1:3);
  await page.screenshot({path:cache+'/v2-'+width+'.png',fullPage:false});
 }
 await page.emulateMedia({colorScheme:'dark'});await page.waitForTimeout(120);check('dark mode canvas switches',await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()),'#1c1c1c');await page.screenshot({path:cache+'/v2-dark.png',fullPage:false});
 check('no console or JavaScript errors',errors,[]);
}catch(e){errors.push(e.message);process.exitCode=1}
finally{await fs.writeFile(out+'/browser_verification.json',JSON.stringify({run_id:d.run_id,verified_at:new Date().toISOString(),checks,errors,screenshots_directory:cache},null,2));console.log(JSON.stringify({passed:checks.filter(c=>c.passed).length,total:checks.length,errors}));if(page)await page.close();await session.detach();process.exit(process.exitCode||0)}
