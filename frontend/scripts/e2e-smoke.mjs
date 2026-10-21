/**
 * Browser smoke: all main routes (requires dev/preview with API proxy for live tabs).
 */
import { chromium } from 'playwright';
import { captureFailure, withScreenshotOnFail } from './qa-browser-utils.mjs';

const BASE = process.env.BASE_URL || 'http://127.0.0.1:5173';

async function assertNoBlank(page, selector, name) {
  const count = await page.locator(selector).count();
  if (!count) throw new Error(`${name} blank or missing selector ${selector}`);
}

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));

  const routes = [
    { hash: 'overview', selector: 'text=Code similarity & logic diagnosis', name: 'Overview' },
    { hash: 'similarity', selector: 'button:has-text("Compare")', name: 'Similarity' },
    { hash: 'classifier', selector: 'text=Logic classifier', name: 'Classifier' },
    { hash: 'evaluation', selector: 'text=Evaluation', name: 'Evaluation' },
    { hash: 'history', selector: 'text=Session history', name: 'History' },
  ];

  for (const r of routes) {
    await withScreenshotOnFail(page, `route-${r.hash}`, async () => {
      await page.goto(`${BASE}/#${r.hash}`, { waitUntil: 'networkidle' });
      await assertNoBlank(page, r.selector, r.name);
    });
  }

  await page.goto(`${BASE}/#classifier`, { waitUntil: 'networkidle' });
  await page.reload({ waitUntil: 'networkidle' });
  await assertNoBlank(page, 'text=Logic classifier', 'Classifier refresh');

  await page.goto(`${BASE}/#similarity`, { waitUntil: 'networkidle' });
  const themeBtn = page.getByRole('button', { name: /Light theme|Dark theme/ });
  await themeBtn.click();
  await page.waitForTimeout(200);
  let theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  if (!theme) throw new Error('Theme toggle did not set data-theme');
  await themeBtn.click();
  theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  if (!theme) throw new Error('Theme toggle second click failed');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${BASE}/#similarity`, { waitUntil: 'networkidle' });
  await assertNoBlank(page, 'button:has-text("Compare")', 'Mobile Similarity');
  const overflow = await page.evaluate(() => {
    const doc = document.documentElement;
    return doc.scrollWidth - doc.clientWidth;
  });
  if (overflow > 24) {
    throw new Error(`Horizontal overflow on mobile layout (${overflow}px)`);
  }

  // Backend unavailable — UI stays usable, similarity shows error (not blank)
  await page.route('**/api/health', (route) =>
    route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"down"}' }),
  );
  await page.route('**/api/similarity/analyze', (route) =>
    route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"down"}' }),
  );
  await page.goto(`${BASE}/#similarity`, { waitUntil: 'networkidle' });
  for (const idx of [0, 1]) {
    const editor = page.locator('.code-editor-wrapper .cm-content').nth(idx);
    await editor.click();
    await page.keyboard.press('Control+A');
    await page.keyboard.press('Backspace');
    await page.keyboard.insertText('def f():\n    pass');
  }
  await page.locator('[data-testid="sim-compare-btn"]').click();
  await page.locator('[data-testid="sim-error"]').waitFor({ timeout: 60000 });
  const errText = await page.locator('[data-testid="sim-error"]').innerText();
  if (!errText.trim()) throw new Error('Expected sim-error when backend down');
  await page.unroute('**/api/health').catch(() => {});
  await page.unroute('**/api/similarity/analyze').catch(() => {});

  if (errors.length) throw new Error(`React errors: ${errors.join('; ')}`);
  console.log('E2E smoke OK: routes, refresh, theme, mobile, backend-down');
  await browser.close();
}

main().catch(async (e) => {
  console.error('E2E smoke FAILED:', e.message);
  process.exit(1);
});
