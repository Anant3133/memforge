import { chromium } from 'playwright';
const OUT = '/workspace/memforge/benchmarks/results/ui_screenshots';
const widths = [360, 414, 768, 1024, 1440];
const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage', '--no-zygote'] });
const problems = [];
const consoleErrors = [];
for (const w of widths) {
  for (const view of ['analytics', 'live']) {
    const ctx = await browser.newContext({ viewport: { width: w, height: 900 } });
    const page = await ctx.newPage();
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(`${w}/${view}: ${m.text()}`); });
    page.on('pageerror', (e) => consoleErrors.push(`${w}/${view}: PAGEERROR ${e.message}`));
    await page.addInitScript((v) => localStorage.setItem('mf-view', v), view);
    await page.goto('http://127.0.0.1:3000/', { waitUntil: 'load' });
    await page.waitForTimeout(1800);
    // scroll through so in-view animations fire
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 500) { window.scrollTo(0, y); await new Promise(r => setTimeout(r, 60)); } window.scrollTo(0, 0); });
    await page.waitForTimeout(1200);
    const res = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth;
      const out = { vw, scrollW: document.documentElement.scrollWidth, offenders: [] };
      for (const el of document.querySelectorAll('body *')) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        const cs = getComputedStyle(el);
        if (cs.position === 'fixed') continue;
        // ignore anything inside a deliberately scrollable container
        let p = el.parentElement, scrollable = false;
        while (p && p !== document.body) { const o = getComputedStyle(p).overflowX; if (o === 'auto' || o === 'scroll') { scrollable = true; break; } p = p.parentElement; }
        if (scrollable) continue;
        if (r.right > vw + 1 || r.left < -1) out.offenders.push(`${el.tagName.toLowerCase()}.${(el.className && el.className.baseVal === undefined ? el.className : '').toString().slice(0, 40)} right=${Math.round(r.right)} left=${Math.round(r.left)}`);
      }
      out.offenders = out.offenders.slice(0, 8);
      return out;
    });
    if (res.scrollW > res.vw + 1) problems.push(`H-SCROLL ${w}/${view}: scrollWidth ${res.scrollW} > ${res.vw}`);
    if (res.offenders.length) problems.push(`OVERFLOW ${w}/${view}: ${res.offenders.join(' | ')}`);
    await page.screenshot({ path: `${OUT}/${view}-${w}.png`, fullPage: true });
    await ctx.close();
  }
}
console.log(JSON.stringify({ problems, consoleErrors: consoleErrors.slice(0, 12) }, null, 1));
await browser.close();
