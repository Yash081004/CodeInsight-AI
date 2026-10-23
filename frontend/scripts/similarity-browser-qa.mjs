/**
 * Similarity browser E2E: UI must match POST /api/similarity/analyze responses (no score mocks).
 * Requires Vite dev with API proxy to backend at BASE_URL.
 */
import { chromium } from 'playwright';

const BASE = process.env.BASE_URL || 'http://127.0.0.1:5173';
const SIM_ENDPOINT = '/api/similarity/analyze';

const IDENTICAL = 'def add(a, b):\n    return a + b';

const RENAME_A = `def calculate_sum(numbers_list):
    total_value = 0
    for number in numbers_list:
        total_value += number
    return total_value`;

const RENAME_B = `def calculate_sum(arr):
    s = 0
    for x in arr:
        s += x
    return s`;

const DIFF_A = `def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr`;

const DIFF_B = `def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1`;

function primaryScore(api) {
  if (api.decision_model === 'tfidf') return api.tfidf_score;
  return api.codebert_score ?? api.tfidf_score;
}

async function textIfPresent(page, sel) {
  const loc = page.locator(sel);
  if ((await loc.count()) === 0) return '';
  return (await loc.first().innerText()).trim();
}

async function setSnippet(page, index, code) {
  const editor = page.locator('.comparison-stage .code-editor-wrapper .cm-content').nth(index);
  await editor.click();
  await page.keyboard.press('Control+A');
  await page.keyboard.press('Backspace');
  await page.keyboard.insertText(code);
}

async function openThresholdPanel(page) {
  await page.locator('details:has(#similarity-threshold)').evaluate((el) => {
    el.open = true;
  });
}

async function setThreshold(page, value) {
  await openThresholdPanel(page);
  const min = 0.1;
  const max = 0.99;
  const step = 0.05;
  const clamped = Math.min(max, Math.max(min, Math.round(value / step) * step));
  await page.locator('[data-testid="sim-threshold-select"]').selectOption(String(clamped), { force: true });
  await page.waitForFunction(
    (expected) => document.querySelector('[data-testid="sim-threshold-value"]')?.textContent === expected,
    clamped.toFixed(2),
    { timeout: 5000 },
  );
  return clamped;
}

async function compareAndGetApi(page) {
  const responseP = page.waitForResponse(
    (r) => r.url().includes(SIM_ENDPOINT) && r.request().method() === 'POST',
  );
  await page.locator('[data-testid="sim-compare-btn"]').click();
  const response = await responseP;
  if (!response.ok()) {
    const body = await response.text();
    throw new Error(`Similarity API ${response.status()}: ${body}`);
  }
  const api = await response.json();
  const posted = JSON.parse(response.request().postData() || '{}');
  api._request_threshold = posted.threshold;
  return api;
}

async function assertApiUiMatch(page, api, ui, errors, label) {
  const score = primaryScore(api);
  if (score == null) {
    errors.push(`${label}: API returned no primary score`);
    return;
  }
  const uiScore = parseFloat(ui.decisionScore);
  if (Number.isNaN(uiScore) || Math.abs(uiScore - score) > 0.0006) {
    errors.push(`${label}: decision score UI ${ui.decisionScore} !== API ${score.toFixed(3)}`);
  }
  const uiSimilarHidden = ui.isSimilar === 'true';
  if (uiSimilarHidden !== api.is_similar) {
    errors.push(`${label}: is_similar UI ${uiSimilarHidden} !== API ${api.is_similar}`);
  }
  if (api.is_similar && !ui.verdict.toLowerCase().includes('similar')) {
    errors.push(`${label}: expected similar verdict`);
  }
  if (api.is_similar === false && !ui.verdict.toLowerCase().includes('not similar')) {
    errors.push(`${label}: expected not similar verdict`);
  }
  if (api.codebert_score != null && ui.codebert !== api.codebert_score.toFixed(4)) {
    errors.push(`${label}: codebert UI ${ui.codebert} !== API ${api.codebert_score.toFixed(4)}`);
  }
  if (api.tfidf_score != null && ui.tfidf !== api.tfidf_score.toFixed(4)) {
    errors.push(`${label}: tfidf UI ${ui.tfidf} !== API ${api.tfidf_score.toFixed(4)}`);
  }
  const hiddenModel = await textIfPresent(page, '[data-testid="sim-decision-model-id"]');
  if (hiddenModel && hiddenModel !== api.decision_model) {
    errors.push(`${label}: decision_model "${hiddenModel}" !== API "${api.decision_model}"`);
  }
}

async function readUi(page) {
  await page.locator('[data-testid="sim-result"]').waitFor({ timeout: 120000 });
  return {
    decisionScore: (await page.locator('[data-testid="sim-decision-score"]').innerText()).trim(),
    verdict: (await page.locator('[data-testid="sim-verdict"]').innerText()).trim(),
    decisionModel: '',
    codebert: (await page.locator('[data-testid="sim-codebert-score"]').innerText()).trim(),
    tfidf: (await page.locator('[data-testid="sim-tfidf-score"]').innerText()).trim(),
    isSimilar: (await page.locator('[data-testid="sim-is-similar"]').innerText()).trim(),
  };
}

async function runPair(page, codeA, codeB, threshold = 0.8) {
  await page.goto(`${BASE}/#similarity`, { waitUntil: 'networkidle' });
  await setSnippet(page, 0, codeA);
  await setSnippet(page, 1, codeB);
  if (threshold !== 0.8) await setThreshold(page, threshold);
  const api = await compareAndGetApi(page);
  const ui = await readUi(page);
  return { api, ui };
}

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });

  const failures = [];

  for (const { codeA, codeB, label } of [
    { codeA: IDENTICAL, codeB: IDENTICAL, label: 'identical' },
    { codeA: RENAME_A, codeB: RENAME_B, label: 'rename-equivalent' },
    { codeA: DIFF_A, codeB: DIFF_B, label: 'different-operations' },
  ]) {
    const { api, ui } = await runPair(page, codeA, codeB, 0.8);
    const caseErrors = [];
    await assertApiUiMatch(page, api, ui, caseErrors, label);
    if (label === 'different-operations' && api.is_similar !== false) {
      caseErrors.push('different-operations: expected is_similar false (hybrid policy)');
    }
    if (label === 'rename-equivalent' && api.is_similar !== true) {
      caseErrors.push('rename-equivalent: expected is_similar true');
    }
    failures.push(...caseErrors);
    console.log(`${label}: cb=${api.codebert_score} tf=${api.tfidf_score} similar=${api.is_similar} model=${api.decision_model}`);
  }

  // 8: empty inputs — no API call
  await page.goto(`${BASE}/#similarity`, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'Reset' }).click();
  let apiCalled = false;
  page.on('request', (req) => {
    if (req.url().includes(SIM_ENDPOINT) && req.method() === 'POST') apiCalled = true;
  });
  await page.locator('[data-testid="sim-compare-btn"]').click();
  await page.locator('[data-testid="sim-error"]').waitFor();
  const emptyErr = await page.locator('[data-testid="sim-error"]').innerText();
  if (!emptyErr.toLowerCase().includes('both')) failures.push('empty: expected client validation message');
  if (apiCalled) failures.push('empty: API should not be called');
  else console.log('empty inputs OK — client validation, no API');

  // 8: loading — button disabled while in flight
  await setSnippet(page, 0, IDENTICAL);
  await setSnippet(page, 1, IDENTICAL);
  const slowP = page.waitForResponse((r) => r.url().includes(SIM_ENDPOINT));
  await page.locator('[data-testid="sim-compare-btn"]').click();
  const busy = await page.locator('[data-testid="sim-compare-btn"]').getAttribute('aria-busy');
  if (busy !== 'true') failures.push('loading: aria-busy not true immediately after click');
  await slowP;
  await page.locator('[data-testid="sim-result"]').waitFor();
  console.log('loading OK — compare completed');

  // 8: API error + retry
  let failOnce = true;
  await page.unroute('**/api/similarity/analyze').catch(() => {});
  await page.route('**/api/similarity/analyze', async (route) => {
    if (failOnce) {
      failOnce = false;
      await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'temporary failure' }) });
    } else {
      await route.continue();
    }
  });
  await page.locator('[data-testid="sim-compare-btn"]').click();
  await page.locator('[data-testid="sim-error"]').waitFor();
  await page.locator('[data-testid="sim-error"]').getByRole('button', { name: /Retry/i }).click();
  await page.locator('[data-testid="sim-result"]').waitFor({ timeout: 120000 });
  console.log('API error + retry OK');
  await page.unroute('**/api/similarity/analyze');
  consoleErrors.length = 0;

  // 9: threshold consistency
  await page.goto(`${BASE}/#similarity`, { waitUntil: 'networkidle' });
  await setSnippet(page, 0, RENAME_A);
  await setSnippet(page, 1, RENAME_B);
  const baseApi = await compareAndGetApi(page);
  const score = primaryScore(baseApi);
  if (score == null) failures.push('threshold: no score from baseline compare');
  else {
    const highTh = await setThreshold(page, Math.min(0.99, score + 0.1));
    const highApi = await compareAndGetApi(page);
    if (Math.abs(highApi._request_threshold - highTh) > 0.001) {
      failures.push(`threshold high: request sent ${highApi._request_threshold} not ${highTh}`);
    }
    if (highTh > score && highApi.is_similar !== false) {
      failures.push(`threshold high: is_similar ${highApi.is_similar} expected false (score ${score} th ${highTh})`);
    }
    const lowTh = await setThreshold(page, Math.max(0.1, score - 0.1));
    const lowApi = await compareAndGetApi(page);
    if (Math.abs(lowApi._request_threshold - lowTh) > 0.001) {
      failures.push(`threshold low: request sent ${lowApi._request_threshold} not ${lowTh}`);
    }
    if (lowTh < score && lowApi.is_similar !== true) {
      failures.push(`threshold low: is_similar ${lowApi.is_similar} expected true (score ${score} th ${lowTh})`);
    }
    const ui = await readUi(page);
    const thErrors = [];
    await assertApiUiMatch(page, lowApi, ui, thErrors, 'threshold-low');
    failures.push(...thErrors);
    console.log(`threshold OK — score=${score.toFixed(4)} highTh=${highTh} lowTh=${lowTh}`);
  }

  if (pageErrors.length) failures.push(`React page errors: ${pageErrors.join('; ')}`);
  if (consoleErrors.length) failures.push(`Console errors: ${consoleErrors.join('; ')}`);

  await browser.close();

  if (failures.length) {
    failures.forEach((f) => console.error('FAIL:', f));
    process.exit(1);
  }
  console.log('Similarity browser QA passed');
}

main().catch((e) => {
  console.error('Similarity QA FAILED:', e.message);
  process.exit(1);
});
