const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
  for (const n of process.argv.slice(2)) {
    await p.goto('file://' + path.resolve(n + '.html'));
    await p.evaluate(() => document.fonts.ready);
    await p.waitForTimeout(300);
    await (await p.$('#fig')).screenshot({ path: `../${n}.png` });
  }
  await b.close();
})();
