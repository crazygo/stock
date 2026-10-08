// Headless Chrome (CDP) interaction check: map -> click security -> timeline -> date detail/legend
// -> range buttons -> close panel; desktop and narrow (390px) viewports. Writes screenshots.
// Usage: node --experimental-websocket browser_check.js <page.html> <outdir>
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const [page, outdir] = process.argv.slice(2);
const port = 9333 + Math.floor(Math.random() * 500);
const chrome = spawn('google-chrome', ['--headless=new', '--no-sandbox', '--disable-gpu', `--remote-debugging-port=${port}`,
  `--user-data-dir=/tmp/stv1-chrome-${port}`, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
function getJSON(path, method = 'GET') { return new Promise((res, rej) => { const rq = http.request({ host: '127.0.0.1', port, path, method }, r => { let b = ''; r.on('data', c => b += c); r.on('end', () => { try { res(JSON.parse(b)); } catch (e) { rej(e); } }); }); rq.on('error', rej); rq.end(); }); }
(async () => {
  const results = [];
  let ws;
  try {
    let target;
    for (let i = 0; i < 40; i++) { try { target = await getJSON('/json/new?about:blank', 'PUT'); break; } catch (e) { await sleep(250); } }
    ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise(r => ws.onopen = r);
    let id = 0; const pending = {};
    ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && pending[d.id]) { pending[d.id](d); delete pending[d.id]; } };
    const send = (method, params = {}) => new Promise(r => { const i = ++id; pending[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
    const ev = async expr => (await send('Runtime.evaluate', { expression: expr, returnByValue: true })).result.result.value;
    const click = async (x, y) => { for (const t of ['mousePressed', 'mouseReleased']) await send('Input.dispatchMouseEvent', { type: t, x, y, button: 'left', clickCount: 1 }); await sleep(150); };
    const shot = async name => { const r = await send('Page.captureScreenshot', { format: 'png' }); fs.writeFileSync(`${outdir}/${name}.png`, Buffer.from(r.result.data, 'base64')); };
    const check = (name, ok, info) => results.push({ name, ok: !!ok, info });
    await send('Page.enable'); await send('Runtime.enable');
    for (const vp of [{ name: 'desktop', w: 1280, h: 900, mobile: false }, { name: 'narrow', w: 390, h: 844, mobile: true }]) {
      await send('Emulation.setDeviceMetricsOverride', { width: vp.w, height: vp.h, deviceScaleFactor: 1, mobile: vp.mobile });
      await send('Page.navigate', { url: 'file://' + page }); await sleep(1200);
      const errs = await ev('window.__errs || []');
      const nPts = await ev('document.querySelectorAll("#map g.pt").length');
      check(`${vp.name}: map points rendered`, nPts > 0, nPts);
      if (vp.name === 'desktop' && await ev('!!document.querySelector("[data-filter=grade]")')) {
        // grade / value-chain group / search filters (expected counts computed from the embedded data)
        const D = 'JSON.parse(document.getElementById("data").textContent)';
        const vis = () => ev('+document.getElementById("map").getAttribute("data-visible")');
        const clickSel = async sel => { const p = await ev(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); e.scrollIntoView({block:"center"}); const r = e.getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2}; })()`); await sleep(80); await click(p.x, p.y); };
        const mapped = await ev(`${D}.securities.filter(s => s.map).length`);
        check('filter: all shown initially', await vis() === mapped && nPts === mapped, { mapped, nPts });
        const grades = await ev('[...document.querySelectorAll("[data-grade]")].map(x => x.value)');
        const g0 = grades.includes('A') ? 'A' : grades[0];
        await clickSel(`input[data-grade="${g0}"]`);
        const expG = await ev(`${D}.securities.filter(s => s.map && (s.tags||{}).grade !== ${JSON.stringify(g0)}).length`);
        check(`filter: untick grade ${g0}`, await vis() === expG && await ev('document.querySelectorAll("#map g.pt").length') === expG, { expG, got: await vis() });
        await shot('desktop_filter_grade');
        await clickSel(`input[data-grade="${g0}"]`);
        check('filter: grade restored', await vis() === mapped);
        await clickSel('[data-filter=group] button[data-act=none]');
        check('filter: groups none -> 0 points, unknown list empty', await vis() === 0 && await ev('document.querySelectorAll("#unknown button").length') === 0, await vis());
        const groups = await ev('[...document.querySelectorAll("[data-group]")].map(x => x.value)');
        const counts = await ev(`(() => { const d = ${D}; const o = {}; d.securities.forEach(s => { if (s.map) ((s.tags||{}).groups||[]).forEach(g => o[g] = (o[g]||0) + 1); }); return o; })()`);
        const pick = groups.filter(g => counts[g]).sort((a, b) => counts[b] - counts[a])[1] || groups.find(g => counts[g]);
        await clickSel(`input[data-group="${pick}"]`);
        check(`filter: only group ${pick}`, await vis() === counts[pick], { want: counts[pick], got: await vis(), groups: groups.length });
        check('filter: 16 value-chain groups offered', groups.filter(g => g !== '—').length === 16, groups.length);
        await ev('document.getElementById("flabels").checked || document.getElementById("flabels").click()');
        await sleep(100);
        check('filter: code labels toggle', await ev('document.querySelectorAll("#map text.lbl").length') === counts[pick]);
        await shot('desktop_filter_group');
        await clickSel('[data-filter=group] button[data-act=all]');
        check('filter: groups all restored', await vis() === mapped);
        const tick = await ev(`${D}.securities.find(s => s.map).ticker`);
        await ev(`(() => { const q = document.getElementById("fsearch"); q.value = ${JSON.stringify(tick)}; q.dispatchEvent(new Event("input")); })()`);
        const nSearch = await vis();
        check(`filter: search ${tick}`, nSearch >= 1 && nSearch < mapped, nSearch);
        await ev('(() => { const q = document.getElementById("fsearch"); q.value = ""; q.dispatchEvent(new Event("input")); })()');
        check('filter: search cleared', await vis() === mapped);
        await ev('document.getElementById("flabels").checked && document.getElementById("flabels").click()');
      }
      const pos = await ev('(() => { const g = document.querySelector("#map g.pt"); g.scrollIntoView({block:"center"}); const r = g.getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2, id: g.dataset.id}; })()');
      await sleep(200);
      const pos2 = await ev('(() => { const g = document.querySelector("#map g.pt"); const r = g.getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2}; })()');
      await click(pos2.x, pos2.y);
      const open = await ev('!document.getElementById("panel").hidden');
      check(`${vp.name}: click point opens side panel`, open, pos.id);
      const tl = await ev('(() => { const s = document.getElementById("timeline"); return s ? {circles: s.querySelectorAll("circle.pt").length, legend: document.querySelectorAll(".legend span .sw").length} : null; })()');
      check(`${vp.name}: single timeline chart with points + 5-line legend`, tl && tl.circles >= 1 && tl.legend === 5, tl);
      const dpos = await ev('(() => { const c = document.querySelector("#timeline circle.pt"); c.scrollIntoView({block:"center"}); const r = c.getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2}; })()');
      await sleep(150);
      await click(dpos.x, dpos.y);
      const det = await ev('(() => { const h = [...document.querySelectorAll("#detail h3")].map(x => x.textContent); const rows = document.querySelectorAll("#detail table tr").length; return {h, rows}; })()');
      check(`${vp.name}: date detail table shows`, det.h.some(x => x.startsWith('日期详情')) && det.rows > 8, det);
      for (const n of [14, 60, 30]) {
        const b = await ev(`(() => { const b = [...document.querySelectorAll('.ranges button')].find(x => x.textContent.startsWith('${n} ')); b.scrollIntoView({block:'center'}); const r = b.getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2}; })()`);
        await sleep(100); await click(b.x, b.y);
        const pressed = await ev(`[...document.querySelectorAll('.ranges button')].find(x => x.textContent.startsWith('${n} ')).getAttribute('aria-pressed')`);
        check(`${vp.name}: range ${n} button toggles view`, pressed === 'true', pressed);
      }
      const geo = await ev('(() => { const r = document.getElementById("panel").getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height, vw: innerWidth, vh: innerHeight}; })()');
      check(`${vp.name}: panel geometry`, vp.name === 'narrow' ? (geo.w <= geo.vw + 1 && geo.y > 0) : (geo.x > 0 && geo.w <= 621), geo);
      await shot(`${vp.name}_panel`);
      const cpos = await ev('(() => { const r = document.getElementById("close").getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2}; })()');
      await click(cpos.x, cpos.y);
      check(`${vp.name}: close button hides panel`, await ev('document.getElementById("panel").hidden'));
      const ub = await ev('(() => { const b = document.querySelector("#unknown button"); if (!b) return null; b.scrollIntoView({block:"center"}); const r = b.getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2, t: b.textContent}; })()');
      if (ub) { await sleep(100); await click(ub.x, ub.y); check(`${vp.name}: unknown security still opens detail`, await ev('!document.getElementById("panel").hidden'), ub.t);
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
        await sleep(100); check(`${vp.name}: Escape closes panel`, await ev('document.getElementById("panel").hidden')); }
      await shot(`${vp.name}_map`);
    }
  } catch (e) { results.push({ name: 'script error', ok: false, info: String(e && e.stack || e) }); }
  finally { try { ws && ws.close(); } catch (e) {} chrome.kill('SIGKILL'); }
  console.log(JSON.stringify(results, null, 1));
  process.exit(results.every(r => r.ok) ? 0 : 1);
})();
