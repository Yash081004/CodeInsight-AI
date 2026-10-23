/**

 * Classifier browser QA: UI text must match live /api/classification/analyze responses.

 * Requires Vite dev (proxy to :8000) at BASE_URL.

 */

import { chromium } from 'playwright';

import { captureFailure } from './qa-browser-utils.mjs';



const BASE = process.env.BASE_URL || 'http://127.0.0.1:5173';

const API_TIMEOUT = 120_000;



const SOURCE_MAP = {

  codebert_finetuned: 'CodeBERT',

  baseline_fallback: 'TF-IDF baseline (fallback)',

  baseline: 'TF-IDF baseline',

  spec_rule_assist: 'Specification-assisted rule',

};



const REVIEW_MAP = {

  model_primary: 'Model-primary',

  review_required: 'Review required — weak discrimination',

  inconclusive: 'Inconclusive spec check',

  spec_assisted: 'Spec-assisted (deterministic)',

  baseline_routed: 'Baseline routed',

};



/** Diagnostic QA buttons (React state) — avoids CodeMirror / controlled-input drift */

const CASES = [

  { id: '1', diagnosticButton: 'Square (correct)', clearProblem: true },

  { id: '2', diagnosticButton: 'Square (+ bug)' },

  { id: '3', diagnosticButton: 'Inclusive sum' },

  { id: '4', diagnosticButton: 'Adult threshold' },

  { id: '5', diagnosticButton: 'Sum init' },

];



async function textIfPresent(page, sel) {

  const loc = page.locator(sel);

  if ((await loc.count()) === 0) return '';

  return (await loc.innerText()).trim();

}



async function openDiagnosticPanel(page) {
  await page.locator('details:has-text("Examples")').evaluate((el) => {
    el.open = true;
  });
}

async function clickDiagnostic(page, name) {
  await openDiagnosticPanel(page);
  await page.getByRole('button', { name }).click();
}

async function setCodeMirror(page, code) {

  const editor = page.locator('.code-editor-wrapper .cm-content').first();

  await editor.click();

  await page.keyboard.press('Control+A');

  await page.keyboard.press('Backspace');

  await page.keyboard.insertText(code);

  const expectedLines = code.split('\n').length;

  await page.waitForFunction(

    (n) => {

      const t = document.querySelector('.code-editor-wrapper .code-editor-lines')?.textContent || '';

      return t.startsWith(String(n));

    },

    expectedLines,

    { timeout: 10_000 },

  );

}



async function runCase(page, { id, diagnosticButton, clearProblem }) {

  await page.goto(`${BASE}/#classifier`, { waitUntil: 'domcontentloaded' });

  await page.getByRole('button', { name: /^Clear$/ }).click();

  await clickDiagnostic(page, diagnosticButton);

  if (clearProblem) {

    await page.locator('#problem-statement').fill('');

  }

  const responseP = page.waitForResponse(

    (r) => r.url().includes('/api/classification/analyze') && r.request().method() === 'POST',

    { timeout: API_TIMEOUT },

  );

  await page.getByRole('button', { name: /^Analyze$/ }).click();

  const response = await responseP;

  const api = await response.json();

  await page.locator('[data-testid="clf-result"]').waitFor({ timeout: API_TIMEOUT });

  await page.locator('[data-testid="clf-result-label"]').waitFor({ timeout: API_TIMEOUT });



  const uiLabel = (await page.locator('[data-testid="clf-result-label"]').innerText()).trim();

  const uiSource = (await page.locator('[data-testid="clf-result-source"]').innerText()).trim();

  const uiReview = await textIfPresent(page, '[data-testid="clf-review-status"]');

  const uiEvidence = await textIfPresent(page, '[data-testid="clf-evidence"]');

  const uiScoreAttr = await textIfPresent(page, '[data-testid="clf-score-attribution"]');

  const collapseVisible = await page.locator('[data-testid="clf-collapse-warning"]').count();



  const expectedSource = SOURCE_MAP[api.model_type] || api.model_type;

  const expectedReview = REVIEW_MAP[api.diagnostics?.review_status] || api.diagnostics?.review_status || '';

  const scoreForLabel = api.diagnostics?.routing?.score_applies_to_displayed_label !== false

    && api.diagnostics?.score_applies_to_displayed_label !== false;



  const errors = [];

  if (uiLabel !== api.friendly_label) errors.push(`label UI "${uiLabel}" !== API "${api.friendly_label}"`);

  if (!uiSource.includes(expectedSource)) errors.push(`source UI "${uiSource}" missing "${expectedSource}"`);

  if (expectedReview && !uiReview.includes(expectedReview)) errors.push(`review UI "${uiReview}" !== "${expectedReview}"`);

  if (api.diagnostics?.evidence_summary && !uiEvidence.includes(api.diagnostics.evidence_summary.slice(0, 40))) {

    errors.push('evidence summary mismatch (prefix)');

  }

  if (scoreForLabel && !uiScoreAttr.includes('model softmax')) {

    errors.push('score attribution should mention model softmax');

  }

  if (!scoreForLabel && !uiScoreAttr.includes('spec rule')) {

    errors.push('score attribution should mention spec rule');

  }

  if (api.diagnostics?.collapse_warning && collapseVisible === 0) {

    errors.push('collapse warning missing in UI');

  }

  if (!api.diagnostics?.collapse_warning && collapseVisible > 0) {

    errors.push('collapse warning shown but API had none');

  }



  return { id, api, errors, uiLabel, expectedSource: api.model_type };

}



async function main() {

  const browser = await chromium.launch();

  const page = await browser.newPage();

  const pageErrors = [];

  page.on('pageerror', (e) => pageErrors.push(e.message));



  const results = [];

  for (const c of CASES) {

    try {

      const r = await runCase(page, c);

      results.push(r);

      if (r.errors.length) {

        console.error(`Case ${c.id} FAIL:`, r.errors.join('; '));

      } else {

        console.log(`Case ${c.id} OK — ${r.uiLabel} (${r.expectedSource})`);

      }

    } catch (e) {

      await captureFailure(page, `case-${c.id}`);

      throw e;

    }

  }



  await page.goto(`${BASE}/#classifier`, { waitUntil: 'domcontentloaded' });

  await page.getByRole('button', { name: /^Clear$/ }).click();

  await setCodeMirror(page, 'def foo():\n    return 1');

  await page.locator('#problem-statement').fill('Implement binary search on a sorted array of integers.');

  const unsResp = page.waitForResponse((r) => r.url().includes('/api/classification/analyze'), { timeout: API_TIMEOUT });

  await page.getByRole('button', { name: /^Analyze$/ }).click();

  const unsApi = await (await unsResp).json();

  await page.locator('[data-testid="clf-result"]').waitFor({ timeout: API_TIMEOUT });

  await page.locator('[data-testid="clf-result-label"]').waitFor({ timeout: API_TIMEOUT });

  if (unsApi.diagnostics?.spec_assessment?.status !== 'no_rule_applicable') {

    console.error('Unsupported spec FAIL: expected no_rule_applicable');

  } else {

    console.log('Unsupported spec OK — inconclusive / no rule');

  }



  await page.getByRole('button', { name: /^Clear$/ }).click();

  await clickDiagnostic(page, 'Square (+ bug)');

  await page.locator('#problem-statement').fill('fix square');

  const vagueResp = page.waitForResponse((r) => r.url().includes('/api/classification/analyze'), { timeout: API_TIMEOUT });

  await page.getByRole('button', { name: /^Analyze$/ }).click();

  const vagueApi = await (await vagueResp).json();

  await page.locator('[data-testid="clf-result-label"]').waitFor({ timeout: API_TIMEOUT });

  const vagueSource = await page.locator('[data-testid="clf-result-source"]').innerText();

  if (vagueApi.model_type !== 'codebert_finetuned' || !vagueSource.includes('CodeBERT')) {

    console.error('Vague spec FAIL: expected CodeBERT primary');

  } else {

    console.log('Vague spec OK — no spec assist');

  }



  await page.getByRole('button', { name: /^Clear$/ }).click();

  await setCodeMirror(page, 'def broken(\n    return 1');

  await page.locator('#problem-statement').fill('');

  const invalidResp = page.waitForResponse((r) => r.url().includes('/api/classification/analyze'), { timeout: API_TIMEOUT });

  await page.getByRole('button', { name: /^Analyze$/ }).click();

  const invalidApi = await (await invalidResp).json();

  await page.locator('[data-testid="clf-result"]').waitFor({ timeout: API_TIMEOUT });

  await page.locator('[data-testid="clf-result-label"]').waitFor({ timeout: API_TIMEOUT });

  const invalidLabel = (await page.locator('[data-testid="clf-result-label"]').innerText()).trim();

  if (invalidApi.code_valid !== false) {

    console.error('Invalid syntax FAIL: API should report code_valid false');

  } else if (invalidLabel.toLowerCase().includes('correct')) {

    console.error('Invalid syntax FAIL: showed correct in UI');

  } else {

    console.log('Invalid syntax OK');

  }



  await page.reload({ waitUntil: 'domcontentloaded' });

  const stillClassifier = await page.locator('text=Logic classifier').count();

  if (!stillClassifier) console.error('Refresh FAIL: classifier route lost');

  else console.log('Refresh OK — #classifier route');



  if (pageErrors.length) console.error('Page errors:', pageErrors.join('; '));



  await browser.close();

  const failed = results.some((r) => r.errors.length);

  if (failed) process.exit(1);

  console.log('Classifier browser QA passed');

}



main().catch(async (e) => {

  console.error('Classifier QA FAILED:', e.message);

  process.exit(1);

});


