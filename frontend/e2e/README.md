# UI layout and modal audits

Headless-browser checks that caught real bugs (9 of 24 modals scrolled sideways on a 360px phone; chart text was
unreadable when the SVG was scaled down; reference-line labels overprinted).

```bash
# one-time, inside the project (nothing system-wide is touched)
cd frontend && npm i -D playwright && npx playwright install chromium
# start the API and the dev server
python -m server.app &  cd frontend && npx vite --host 127.0.0.1 &
# run
node e2e/layout-audit.mjs       # 5 widths x 2 views: horizontal overflow, elements past the viewport, console errors; writes screenshots
node e2e/analytics-modals.mjs   # every Know-more modal: fits viewport, has all sections, Esc closes, focus returns, Tab trapped, backdrop closes
node e2e/live-modals.mjs        # same for the Live Engine tab
```

Chromium on a bare server also needs a few shared libraries and a font; if it exits at startup, see
`docs/A100_SERVER_GUIDE.md` (UI section). Launch flags used: `--disable-gpu --disable-dev-shm-usage --no-zygote`.
