const {chromium}=require(process.env.MAP_PLAYWRIGHT_PATH || 'playwright');
const fs=require('fs'),path=require('path'),assert=require('assert').strict;
const dir=__dirname, data=JSON.parse(fs.readFileSync(path.join(dir,'data.json'),'utf8'));
const out=path.join(dir,'quality_browser');fs.mkdirSync(out,{recursive:true});
function current(q){return q&&q.decision==='current'&&q.grade&&!q.preview&&(!q.expires_at||new Date(q.expires_at+'T23:59:59Z')>=new Date());}
async function main(){
 const browser=await chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1512,height:1100},colorScheme:'light'}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('file://'+path.join(dir,'index.html'));
 await page.waitForSelector('#quality-grade',{state:'attached'});
 assert.equal(await page.evaluate(()=>window.__map.D.rows.length),data.rows.length);const aaoi=data.rows.find(r=>r.symbol==='AAOI');const correlation_AAOI=await page.evaluate(code=>{const r=window.__map.D.rows.find(x=>x.code===code);const m=window.__map.metrics(r);return{valid:m.valid,n:m.n,raw:m.raw,residual:m.residual,start:window.__map.D.price.default_start,end:window.__map.D.price.default_end};},aaoi.code);assert.equal(correlation_AAOI.valid,true);assert.equal(correlation_AAOI.n,62);assert.ok(Math.abs(correlation_AAOI.raw-0.6646317217830094)<1e-9);assert.ok(Math.abs(correlation_AAOI.residual-0.4556929928845945)<1e-9);
 assert.equal(await page.locator('#quality-grade').inputValue(),'core');
 await page.click('[data-tab="quality"]');
 const core=data.rows.filter(r=>r.kind==='STOCK'&&current(r.quality)&&['A+','A'].includes(r.quality.grade));
 assert.equal(await page.evaluate(()=>window.__map.rows().length),core.length,'default core');
 await page.screenshot({path:path.join(out,'desktop-current-core.png'),fullPage:false});
 for(const grade of ['all','A+','A','A-','B','stale','pending']){
  await page.selectOption('#quality-grade',grade);
  const expected=data.rows.filter(r=>{if(grade==='all')return true;if(r.kind!=='STOCK')return false;if(grade==='pending')return [1,2,3].includes(r.business.tier)&&(!r.quality||r.quality.decision==='pending'||!r.quality.grade);if(grade==='stale')return !!r.quality?.grade&&r.quality.expires_at&&new Date(r.quality.expires_at+'T23:59:59Z')<new Date();return current(r.quality)&&r.quality.grade===grade;});
  assert.equal(await page.evaluate(()=>window.__map.rows().length),expected.length,grade+' filter');
 }
 await page.selectOption('#quality-grade','core');await page.click('[data-tab="quality"]');
 const qstatus=await page.locator('#quality-status').innerText();assert.match(qstatus,data.quality_rating?.status==='not_run'?/评级尚未执行/:/评级|截至/);if(data.quality_rating?.first_run)assert.match(qstatus,/首轮评级/);
 const counts=await page.locator('#quality-stats .stat').count();assert.equal(counts,6);const expectedStats=[...['A+','A','A-','B'].map(g=>data.rows.filter(r=>r.kind==='STOCK'&&current(r.quality)&&r.quality.grade===g).length),data.rows.filter(r=>r.kind==='STOCK'&&[1,2,3].includes(r.business.tier)&&(!r.quality||r.quality.decision==='pending'||!r.quality.grade)).length,data.rows.filter(r=>r.kind==='STOCK'&&r.quality?.grade&&r.quality.expires_at&&new Date(r.quality.expires_at+'T23:59:59Z')<new Date()).length];const visibleStats=(await page.locator('#quality-stats .stat strong').allTextContents()).map(Number);assert.deepEqual(visibleStats,expectedStats,'summary counts match official snapshot');assert.deepEqual(expectedStats.slice(0,5),[6,11,3,5,433]);
 await page.selectOption('#quality-grade','all');await page.selectOption('#pool','held');
 assert.equal(await page.evaluate(()=>window.__map.rows().length),data.summary.held);
 await page.selectOption('#pool','all');await page.selectOption('#grade','positive');
 const ai=data.rows.filter(r=>r.kind==='STOCK'&&[1,2,3].includes(r.business.tier));
 assert.equal(await page.evaluate(()=>window.__map.stockRows().length),ai.length);
 await page.fill('#search','NVDA');await page.click('[data-tab="basket"]');const basketDL=page.waitForEvent('download');await page.click('#basket-export');const file=path.join(out,'all-ai-export.csv');await(await basketDL).saveAs(file);
 assert.equal(fs.readFileSync(file,'utf8').trim().split(/\r?\n/).length,ai.length+1,'all-AI export independent of quality/search filters');
 await page.click('#reset');await page.click('[data-tab="list"]');
 const r=data.rows.find(x=>x.kind==='STOCK'&&x.quality);
 if(r){await page.fill('#search',r.code);await page.locator('#all-table [data-stock="'+r.code+'"]').click();const detail=await page.locator('#stock-detail').innerText();for(const t of ['质量评级','资料取得','最新申报','营收同比','经营现金流','自由现金流','估值：未评估'])assert.ok(detail.includes(t),t);assert.ok(!detail.includes('[object Object]'));await page.click('#close-stock');}
 await page.click('#reset');const csvDL=page.waitForEvent('download');await page.click('#export');await(csvDL).then(d=>d.saveAs(path.join(out,'current-filter.csv')));const csv=fs.readFileSync(path.join(out,'current-filter.csv'),'utf8');for(const x of ['评级as_of','评级expires_at','规则版本','资料取得日','业务评级官方来源'])assert.ok(csv.includes(x),x);
 await page.setViewportSize({width:390,height:844});for(const tab of ['map','basket','quality','gaps','price','list','method']){await page.click('[data-tab="'+tab+'"]');assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'mobile overflow '+tab);}
 assert.deepEqual(errors,[]);const result={passed:true,source:'built index.html + data.json',desktop:'1512x1100',mobile:'390x844',stocks:data.summary.stocks,quality_counts:core.length,correlation_AAOI,checks:['default current A+/A core','all grade and candidate filters','quality overview counts and not-run state','pool + AI business grade combination','all 458 AI-related export independent of quality/search filters','financial/source detail and current-filter CSV','mobile overflow across seven tabs'],errors};fs.writeFileSync(path.join(dir,'browser_verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
}
main().catch(e=>{console.error(e);process.exit(1)});
