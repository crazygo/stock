/* Browser acceptance of both untouched view engines on the shared dataset. */
const fs=require('fs'),path=require('path'),assert=require('assert');
let pw;try{pw=require('playwright')}catch{pw=require(path.join(process.env.HOME,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'))}
const out=path.resolve(process.argv[2]),run=JSON.parse(fs.readFileSync(path.join(out,'inputs/data_manifest.json')));
(async()=>{
 const browser=await pw.chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000}});await context.setOffline(true);
 const errors=[],requests=[];
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
 const pages=path.join(out,'pages');
 await page.goto('file://'+path.join(pages,'stock_behavior_description_v3/index.html'));
 await page.waitForFunction(()=>window.audit);
 const expected=JSON.parse(fs.readFileSync(path.join(out,'results/behavior_input.json'))).records.length;
 assert.equal(await page.locator('.card').count(),expected);
 assert.equal(await page.locator('#pair-batch').getAttribute('data-run-id'),run.run_id);
 assert(await page.locator('.card svg').count()>=expected);
 await page.locator('[data-half-filter="front"][data-direction="up"]').click();
 assert.equal(await page.locator('[data-half-filter="front"][data-direction="up"]').getAttribute('aria-pressed'),'true');
 await page.locator('#clearFilters').click();assert.equal(await page.locator('.card').count(),expected);
 await page.locator('#search').fill('ZZZZZZ');assert.equal(await page.locator('.card').count(),0);await page.locator('#clearFilters').click();
 await page.locator('#showWeek').count().then(async n=>{if(n){await page.locator('#showWeek').uncheck();await page.locator('#showWeek').check()}});
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.screenshot({path:path.join(out,'validation/behavior-mobile.png')});
 await page.setViewportSize({width:1440,height:1000});
 const href=await page.locator('.card a').first().getAttribute('href');
 await page.goto('file://'+path.join(pages,'stock_behavior_description_v3',href));
 await page.waitForFunction(()=>window.audit);assert(await page.locator('#overview svg').count()>0);
 await page.locator('#reviewNote').fill('自动验收：临时浏览器内核对保存契约');
 await page.locator('#saveReview').click();
 assert((await page.locator('#reviewStatus').textContent()).includes('已保存'));
 await page.goto('file://'+path.join(pages,'ai_trend_quadrant_v2/index.html'));
 await page.locator('#tab-compare').click();
 await page.waitForFunction(()=>document.querySelectorAll('#compare-grid .compare-card').length>0 || document.querySelectorAll('#compare-grid article').length>0);
 assert.equal(await page.locator('#pair-batch').getAttribute('data-run-id'),run.run_id);
 await page.locator('#compare-search').fill('ZZZZZZ');
 await page.waitForFunction(()=>document.querySelectorAll('#compare-grid article').length===0);
 await page.locator('#compare-clear').click();
 assert(await page.locator('#compare-grid svg').count()>0);
 const apple=page.locator('#compare-grid article[data-code="US.AAPL"] .compare-badges');
 if(await apple.count())assert(!(await apple.textContent()).includes('象限待确认'),
                              'HK cutoff must not mark a current US series stale');
 await page.locator('#compare-week').uncheck();await page.locator('#compare-week').check();
 const mapped=JSON.parse(fs.readFileSync(path.join(out,'results/trend/results.json'))).records.find(r=>r.mapped);
 if(mapped){await page.locator('#tab-map').click();await page.locator('#security').selectOption(mapped.code);await page.waitForTimeout(100);assert(await page.locator('#time-spectrum svg').count()>0);await page.locator('#close').click();await page.locator('#tab-compare').click()}
 await page.screenshot({path:path.join(out,'validation/trend-desktop.png')});
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.screenshot({path:path.join(out,'validation/trend-mobile.png')});
 const review=path.join(out,'validation/recompute_review.html');
 if(fs.existsSync(review)){
  await page.setViewportSize({width:1440,height:1000});await page.goto('file://'+review);
  const comparison=JSON.parse(fs.readFileSync(path.join(out,'validation/recompute_review.json')));
  assert.equal(await page.locator('#rows tr').count(),comparison.securities);
  await page.locator('#special').check();assert.equal(await page.locator('#rows tr').count(),expected);
  await page.locator('#changed').check();assert((await page.locator('#count').textContent()).startsWith('0 /'));
  await page.locator('#reset').click();await page.locator('#search').fill('US.AAPL');assert.equal(await page.locator('#rows tr').count(),1);
  await page.locator('#reset').click();await page.screenshot({path:path.join(out,'validation/review-desktop.png')});
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:path.join(out,'validation/review-mobile.png')});
 }
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 fs.writeFileSync(path.join(out,'validation/browser.json'),JSON.stringify({status:'passed',run_id:run.run_id,
  behavior_members:expected,errors,requests,checks:['offline both pages','shared batch','charts','search/reset','day/week','detail evidence/review save','per-market cutoff','map/time spectrum','mobile overflow']}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
