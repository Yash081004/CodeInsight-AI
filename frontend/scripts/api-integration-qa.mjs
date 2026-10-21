/**
 * Live HTTP integration against the real backend (default :8000). No mocks except explicit retry test.
 * Writes JSON report when QA_REPORT_PATH is set.
 */
import fs from 'node:fs';
import path from 'node:path';

const BACKEND = (process.env.BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const REPORT_PATH = process.env.QA_REPORT_PATH || '';

/** @typedef {{ name: string, status: string, category: string, actual?: unknown, details?: string }} Check */

/** @type {Check[]} */
const checks = [];

function record(name, status, category, actual = null, details = '') {
  checks.push({ name, status, category, actual, details });
}

async function postJson(urlPath, body) {
  const res = await fetch(`${BACKEND}${urlPath}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    json = { raw: text };
  }
  return { status: res.status, json };
}

async function getJson(urlPath) {
  const res = await fetch(`${BACKEND}${urlPath}`);
  const json = await res.json();
  return { status: res.status, json };
}

const IDENTICAL = 'def add(a, b):\n    return a + b';

const FORMAT_A = `def total(nums):
    s = 0
    for n in nums:
        s += n
    return s`;

const FORMAT_B = `def total(nums):
    return sum(nums)`;

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

const BUBBLE = `def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr`;

const BINARY = `def binary_search(arr, target):
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

async function main() {
  try {
    const health = await getJson('/api/health');
    record(
      'health',
      health.status === 200 && health.json.status === 'ok' ? 'passed' : 'failed',
      'api_integration',
      health.json,
    );
  } catch (e) {
    record('health', 'blocked', 'api_integration', null, String(e.message));
    writeReport();
    process.exit(2);
  }

  try {
    const ready = await getJson('/api/ready');
    record(
      'ready',
      ready.status === 200 && ready.json.ready === true ? 'passed' : 'warning',
      'api_integration',
      ready.json,
      ready.json.ready ? '' : 'Backend not fully ready',
    );
  } catch (e) {
    record('ready', 'blocked', 'api_integration', null, String(e.message));
  }

  const simCases = [
    { name: 'similarity_identical', a: IDENTICAL, b: IDENTICAL, expectSimilar: true },
    { name: 'similarity_rename', a: RENAME_A, b: RENAME_B, expectSimilar: true },
    { name: 'similarity_format_only', a: FORMAT_A, b: FORMAT_B, expectSimilar: true },
    { name: 'similarity_different_ops', a: BUBBLE, b: BINARY, expectSimilar: false },
  ];

  for (const c of simCases) {
    const { status, json } = await postJson('/api/similarity/analyze', {
      code_a: c.a,
      code_b: c.b,
      threshold: 0.8,
    });
    const ok =
      status === 200 &&
      json.is_similar === c.expectSimilar &&
      json.decision_model &&
      json.decision_score != null;
    record(
      c.name,
      ok ? 'passed' : 'failed',
      'api_integration',
      {
        is_similar: json.is_similar,
        decision_model: json.decision_model,
        decision_policy: json.decision_policy,
        codebert_score: json.codebert_score,
        tfidf_score: json.tfidf_score,
        threshold: json.threshold,
      },
      ok ? '' : `HTTP ${status}; expected is_similar=${c.expectSimilar}`,
    );
  }

  const empty = await postJson('/api/similarity/analyze', { code_a: '', code_b: 'def f(): pass' });
  record(
    'similarity_validation_empty',
    empty.status === 422 ? 'passed' : 'failed',
    'api_integration',
    { status: empty.status },
  );

  const badTh = await postJson('/api/similarity/analyze', {
    code_a: 'def f(): pass',
    code_b: 'def g(): pass',
    threshold: 2,
  });
  record(
    'similarity_validation_threshold',
    badTh.status === 422 ? 'passed' : 'failed',
    'api_integration',
    { status: badTh.status },
  );

  const scoreResp = await postJson('/api/similarity/analyze', {
    code_a: RENAME_A,
    code_b: RENAME_B,
    threshold: 0.8,
  });
  const score = scoreResp.json.decision_score ?? scoreResp.json.codebert_score;
  if (scoreResp.status === 200 && score != null) {
    const highTh = Math.min(0.99, Math.round((score + 0.15) * 20) / 20);
    const high = await postJson('/api/similarity/analyze', {
      code_a: RENAME_A,
      code_b: RENAME_B,
      threshold: highTh,
    });
    record(
      'similarity_threshold_high',
      high.status === 200 && high.json.is_similar === false ? 'passed' : 'failed',
      'api_integration',
      { threshold: highTh, score, is_similar: high.json.is_similar },
    );
    const lowTh = Math.max(0.1, Math.round((score - 0.15) * 20) / 20);
    const low = await postJson('/api/similarity/analyze', {
      code_a: RENAME_A,
      code_b: RENAME_B,
      threshold: lowTh,
    });
    record(
      'similarity_threshold_low',
      low.status === 200 && low.json.is_similar === true ? 'passed' : 'failed',
      'api_integration',
      { threshold: lowTh, score, is_similar: low.json.is_similar },
    );
  } else {
    record('similarity_threshold_high', 'blocked', 'api_integration', null, 'No score for threshold test');
    record('similarity_threshold_low', 'blocked', 'api_integration', null, 'No score for threshold test');
  }

  // Repeated requests — no crash
  for (let i = 0; i < 3; i++) {
    await postJson('/api/similarity/analyze', { code_a: IDENTICAL, code_b: IDENTICAL });
  }
  record('similarity_repeated_requests', 'passed', 'api_integration', { count: 3 });

  const clfCases = [
    {
      name: 'classification_correct',
      body: { code: 'def square(n):\n    return n * n' },
      assert: (j) => j.code_valid && j.predicted_label,
    },
    {
      name: 'classification_wrong_operator',
      body: {
        code: 'def square(n):\n    return n + n',
        problem_statement: 'Return the square of the given number.',
      },
      assert: (j) => j.code_valid && j.diagnostics?.review_status,
    },
    {
      name: 'classification_vague_spec',
      body: {
        code: 'def square(n):\n    return n + n',
        problem_statement: 'fix square',
      },
      assert: (j) => j.model_type === 'codebert_finetuned' || j.model_type === 'baseline_fallback',
    },
    {
      name: 'classification_unsupported_spec',
      body: {
        code: 'def foo():\n    return 1',
        problem_statement: 'Implement binary search on a sorted array of integers.',
      },
      assert: (j) => j.diagnostics?.spec_assessment?.status === 'no_rule_applicable',
    },
    {
      name: 'classification_invalid_syntax',
      body: { code: 'def broken(\n    return 1' },
      assert: (j) => j.code_valid === false || j.predicted_label !== 'correct',
    },
  ];

  for (const c of clfCases) {
    const { status, json } = await postJson('/api/classification/analyze', c.body);
    const ok = status === 200 && c.assert(json);
    if (json.model_type === 'spec_rule_assist') {
      const routing = json.diagnostics?.routing || {};
      if (routing.score_applies_to_displayed_label !== false) {
        record(c.name, 'failed', 'api_integration', json, 'spec assist must not claim model softmax score');
        continue;
      }
    }
    record(c.name, ok ? 'passed' : 'failed', 'api_integration', {
      predicted_label: json.predicted_label,
      model_type: json.model_type,
      review_status: json.diagnostics?.review_status,
    });
  }

  const emptyClf = await postJson('/api/classification/analyze', { code: '' });
  record(
    'classification_validation_empty',
    emptyClf.status === 422 ? 'passed' : 'failed',
    'api_integration',
    { status: emptyClf.status },
  );

  writeReport();
  const failed = checks.filter((c) => c.status === 'failed');
  const blocked = checks.filter((c) => c.status === 'blocked');
  if (failed.length) {
    failed.forEach((f) => console.error('FAIL', f.name, f.details || f.actual));
    process.exit(1);
  }
  if (blocked.length) {
    blocked.forEach((b) => console.warn('BLOCKED', b.name, b.details));
    process.exit(2);
  }
  console.log(`API integration QA OK (${checks.length} checks)`);
}

function writeReport() {
  if (!REPORT_PATH) return;
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(checks, null, 2));
}

main().catch((e) => {
  record('runner_crash', 'failed', 'api_integration', null, e.message);
  writeReport();
  console.error(e);
  process.exit(1);
});
