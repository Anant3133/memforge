import { chromium } from 'playwright';
const b = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage', '--no-zygote'] });
const out = [];
for (const [w, h] of [[1440, 900], [360, 740]]) {
  const ctx = await b.newContext({ viewport: { width: w, height: h } });
  const p = await ctx.newPage();
  const errs = []; p.on('pageerror', (e) => errs.push(e.message));
  await p.addInitScript(() => localStorage.setItem('mf-view', 'live'));
  await p.goto('http://127.0.0.1:3000/', { waitUntil: 'load' });
  await p.waitForSelector('.mf-know'); await p.waitForTimeout(800);
  const n = await p.locator('.mf-know').count();
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const btn = p.locator('.mf-know').nth(i); await btn.scrollIntoViewIfNeeded(); await btn.click();
    await p.waitForSelector('.mf-modal'); await p.waitForTimeout(80);
    const m = await p.evaluate(() => { const el = document.querySelector('.mf-modal'), r = el.getBoundingClientRect(); return { fits: r.left >= -1 && r.top >= -1 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1, hs: el.scrollWidth > el.clientWidth + 1, txt: el.textContent.length }; });
    if (!m.fits || m.hs || m.txt < 80) { bad++; out.push(`BAD ${w} #${i} ${JSON.stringify(m)}`); }
    await p.keyboard.press('Escape'); await p.waitForTimeout(60);
    if (await p.locator('.mf-modal').count()) { bad++; out.push(`BAD ${w} #${i} did not close`); }
  }
  out.push(`${w}: ${n} live-view Know more buttons, problems=${bad}, pageerrors=${errs.length}`);
  await ctx.close();
}
console.log(out.join('\n')); await b.close();
