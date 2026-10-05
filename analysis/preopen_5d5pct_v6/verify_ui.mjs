import fs from 'node:fs';
const tabs=await (await fetch('http://127.0.0.1:19223/json/list')).json();
const tab=tabs.find(t=>t.id==='6C932EE5A9FBAF050A128F549BD3C0D6')||tabs.find(t=>t.url.startsWith('http://127.0.0.1:8771'));
if(!tab)throw Error('Dedicated background test tab required');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(resolve=>ws.addEventListener('open',resolve));let id=1;
function send(method,params={}){return new Promise((resolve,reject)=>{let n=id++;let timer=setTimeout(()=>reject(Error('CDP timeout '+method)),15000);const h=e=>{let r=JSON.parse(e.data);if(r.id!==n)return;ws.removeEventListener('message',h);clearTimeout(timer);r.error?reject(Error(r.error.message)):resolve(r.result)};ws.addEventListener('message',h);ws.send(JSON.stringify({id:n,method,params}))})}
async function evaluate(expression){let r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));let checks=[];
const root=new URL('.',import.meta.url).pathname;
for(const page of ['index.html','opportunity.html','sector.html','events.html']){
 for(const width of [1280,390]){
  await send('Emulation.setDeviceMetricsOverride',{width,height:950,deviceScaleFactor:1,mobile:width===390});
  await send('Page.navigate',{url:'http://127.0.0.1:8771/'+page});await sleep(1600);
  const state=await evaluate(`({width:innerWidth,body:document.body.scrollWidth,ready:!!data,bars:data?.bars.length,summary:$('summary').textContent,dates:document.querySelectorAll('input[type=date]').length,mini:!!data?.mini_days,range:$('pan').max,visible:$('visible').textContent})`);
  if(state.body>width||!state.ready||!state.mini||state.dates!==2)throw Error(page+' '+JSON.stringify(state));
  await evaluate(`document.querySelector('[data-days="30"]').click()`);await sleep(300);
  let interactions=await evaluate(`(()=>{let old=offset;$('right').click();let next=offset;$('pan').value=0;$('pan').dispatchEvent(new Event('input'));let before=offset;$('chart').dispatchEvent(new WheelEvent('wheel',{deltaY:400,bubbles:true,cancelable:true}));return {moved:next>old,wheelUnchanged:offset===before,grain:data.grain_minutes}})()`);
  if(!interactions.moved||!interactions.wheelUnchanged||interactions.grain!==5)throw Error('interaction failed '+page);
  if(page==='opportunity.html'){
   const count=await evaluate(`$('signal').options.length`);
   if(count>1){await evaluate(`$('signal').selectedIndex=1;$('signal').dispatchEvent(new Event('change'))`);await sleep(600);let inspect=await evaluate(`$('inspect').textContent`);if(!inspect.includes('features')||!inspect.includes('label_end'))throw Error('inspect failed')}
  }
  const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.mkdirSync(root+'screenshots',{recursive:true});fs.writeFileSync(root+`screenshots/${page.replace('.html','')}_${width}.png`,Buffer.from(shot.data,'base64'));
  if(page!=='index.html'){
   await evaluate(`(async()=>{let failed=evs.find(e=>e.y===0);if(!failed)throw Error('No actual failed signal to inspect');$('symbol').value=failed.symbol;await load();$('signal').value=failed.day;await $('signal').onchange();if(chosen?.y!==0)throw Error('Failed signal did not remain selected')})()`);
   let rect=await evaluate(`(()=>{$('chart').scrollIntoView({block:'center'});let r=$('chart').getBoundingClientRect();return {x:r.left+52,y:r.top+120}})()`);
   await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:rect.x,y:rect.y});
   let detail=await evaluate(`$('bar-details').textContent`);if(!detail.includes('Open')||!detail.includes('成交量'))throw Error('real minute price/volume inspection failed '+page);
   const pathShot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(root+`screenshots/${page.replace('.html','')}_path_${width}.png`,Buffer.from(pathShot.data,'base64'));
   interactions.minutePriceVolumeInspection=true;
  }
  checks.push({page,width,...state,interactions});
 }
}
fs.writeFileSync(root+'ui_verification.json',JSON.stringify({at:new Date().toISOString(),status:'passed',checks},null,2)+'\n');ws.close();console.log(JSON.stringify({status:'passed',checks:checks.length}));
