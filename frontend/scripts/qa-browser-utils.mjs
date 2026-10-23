import fs from 'node:fs';
import path from 'node:path';

const SCREENSHOT_DIR = process.env.QA_SCREENSHOT_DIR || '';

export async function captureFailure(page, name) {
  if (!SCREENSHOT_DIR) return null;
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const file = path.join(SCREENSHOT_DIR, `${name.replace(/[^\w.-]+/g, '_')}.png`);
  await page.screenshot({ path: file, fullPage: true });
  return file;
}

export async function withScreenshotOnFail(page, name, fn) {
  try {
    return await fn();
  } catch (e) {
    const shot = await captureFailure(page, name);
    if (shot) e.message += ` (screenshot: ${shot})`;
    throw e;
  }
}
