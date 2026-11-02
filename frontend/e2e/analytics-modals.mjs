import { chromium } from 'playwright';
const OUT = '/workspace/memforge/benchmarks/results/ui_screenshots';
const b = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage', '--no-zygote'] });
const report = [];
for (const [w, h] of [[1440, 900], [360, 740]]) {
  const ctx = await b.newContext({ viewport: { width: w, height: h } });
  const p = await ctx.newPage();
  await p.addInitScript(() => localStorage.setItem('mf-view', 'analytics'));
  await p.goto('http://127.0.0.1:3000/', { waitUntil: 'load' });
  await p.waitForSelector('.mf-kpi');
  await p.waitForTimeout(1500);
  const buttons = await p.locator('.mf-know').count();
  report.push(`${w}: ${buttons} Know more buttons`);
  // open every Know-more button in turn: it must fit the viewport, show content, close with Esc, restore focus
  let bad = 0;
  for (let i = 0; i < buttons; i++) {
    const btn = p.locator('.mf-know').nth(i);
    await btn.scrollIntoViewIfNeeded();
    await btn.click();
    await p.waitForSelector('.mf-modal');
    await p.waitForTimeout(120);
    const m = await p.evaluate(() => {
      const el = document.querySelector('.mf-modal'); const r = el.getBoundingClientRect();
      return { l: r.left, t: r.top, r: r.right, b: r.bottom, vw: innerWidth, vh: innerHeight, scrollable: el.scrollHeight > el.clientHeight,
        headings: [...el.querySelectorAll('h3')].map(x => x.textContent), title: el.querySelector('h2')?.textContent,
        bodyLocked: document.body.style.overflow === 'hidden', hscroll: el.scrollWidth > el.clientWidth + 1 };
    });
    const fits = m.l >= -1 && m.t >= -1 && m.r <= m.vw + 1 && m.b <= m.vh + 1;
    const hasCore = ['What it is', 'Why we track it', 'How it is measured', 'How to read it', 'Where to look'].every(x => m.headings.includes(x));
    if (!fits || !hasCore || !m.bodyLocked || m.hscroll) { bad++; report.push(`  BAD modal #${i} "${m.title}": fits=${fits} core=${hasCore} locked=${m.bodyLocked} hscroll=${m.hscroll}`); }
    if (i === 3) await p.screenshot({ path: `${OUT}/modal-${w}.png` });
    await p.keyboard.press('Escape');
    await p.waitForTimeout(80);
    const closed = (await p.locator('.mf-modal').count()) === 0;
    const focusBack = await p.evaluate((idx) => document.activeElement === document.querySelectorAll('.mf-know')[idx], i);
    if (!closed || !focusBack) { bad++; report.push(`  BAD close #${i}: closed=${closed} focusRestored=${focusBack}`); }
  }
  report.push(`${w}: ${buttons - bad >= 0 ? 'modals checked' : ''} problems=${bad}`);
  // backdrop click closes; Tab stays trapped inside
  await p.locator('.mf-know').first().click();
  await p.waitForSelector('.mf-modal');
  for (let i = 0; i < 12; i++) await p.keyboard.press('Tab');
  const trapped = await p.evaluate(() => !!document.activeElement?.closest('.mf-modal'));
  await p.mouse.click(4, 4);
  await p.waitForTimeout(100);
  report.push(`${w}: tab trapped=${trapped}, backdrop closes=${(await p.locator('.mf-modal').count()) === 0}`);
  // section close-ups
  for (const id of ['paged', 'spec', 'chunk']) {
    await p.locator('#' + id).scrollIntoViewIfNeeded();
    await p.waitForTimeout(1500);
    await p.locator('#' + id).screenshot({ path: `${OUT}/section-${id}-${w}.png` });
  }
  await ctx.close();
}
console.log(report.join('\n'));
await b.close();
