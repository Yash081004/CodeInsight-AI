/**
 * CodeInsight AI - Frontend API Client
 * Wraps all backend endpoints with unified error handling.
 */

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function analyzeSimilarity(codeA, codeB, threshold = 0.8) {
  const res = await fetch(`${API_BASE}/similarity/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code_a: codeA, code_b: codeB, threshold }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Similarity request failed: ${res.status}`);
  }
  return res.json();
}

export async function analyzeClassification(code, options = {}) {
  const body = {
    code,
    problem_statement: options.problemStatement || undefined,
    function_context: options.functionContext || undefined,
    run_tests: options.runTests || false,
  };
  const res = await fetch(`${API_BASE}/classification/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Classification request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchEvaluationExperiments() {
  const res = await fetch(`${API_BASE}/evaluation/experiments`);
  if (!res.ok) throw new Error(`Evaluation experiments failed: ${res.statusText}`);
  return res.json();
}

export async function fetchEvaluationSummary(runId = 'production') {
  const params = new URLSearchParams({ run_id: runId });
  const res = await fetch(`${API_BASE}/evaluation/summary?${params}`);
  if (!res.ok) throw new Error(`Evaluation summary failed: ${res.statusText}`);
  return res.json();
}

export async function fetchConfusionMatrix(runId = 'production', model = 'codebert') {
  const params = new URLSearchParams({ run_id: runId, model });
  const res = await fetch(`${API_BASE}/evaluation/confusion-matrix?${params}`);
  if (!res.ok) throw new Error(`Confusion matrix failed: ${res.statusText}`);
  return res.json();
}

export async function fetchErrorGallery(runId, filters = {}) {
  const params = new URLSearchParams({ run_id: runId, limit: String(filters.limit ?? 40) });
  if (filters.correct === true) params.set('correct', 'true');
  if (filters.correct === false) params.set('correct', 'false');
  if (filters.trueLabel) params.set('true_label', filters.trueLabel);
  if (filters.predictedLabel) params.set('predicted_label', filters.predictedLabel);
  if (filters.model) params.set('model', filters.model);
  if (filters.familyId) params.set('family_id', filters.familyId);
  const res = await fetch(`${API_BASE}/evaluation/error-gallery?${params}`);
  if (!res.ok) throw new Error(`Error gallery failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelComparison(runId) {
  const params = new URLSearchParams({ run_id: runId });
  const res = await fetch(`${API_BASE}/evaluation/model-comparison?${params}`);
  if (!res.ok) throw new Error(`Model comparison failed: ${res.statusText}`);
  return res.json();
}

export async function fetchBenchmarkQuality(version = 'benchmark_v3') {
  const params = new URLSearchParams({ version });
  const res = await fetch(`${API_BASE}/evaluation/benchmark-quality?${params}`);
  if (!res.ok) throw new Error(`Benchmark quality failed: ${res.statusText}`);
  return res.json();
}

export async function fetchAblations() {
  const res = await fetch(`${API_BASE}/evaluation/ablations`);
  if (!res.ok) throw new Error(`Ablations failed: ${res.statusText}`);
  return res.json();
}

export async function fetchPhase7ErrorAnalysis() {
  const res = await fetch(`${API_BASE}/evaluation/phase7-error-analysis`);
  if (!res.ok) throw new Error(`Phase 7 analysis failed: ${res.statusText}`);
  return res.json();
}

export async function fetchErrorExplorer(params = {}) {
  const qs = new URLSearchParams(params);
  const res = await fetch(`${API_BASE}/evaluation/error-explorer?${qs}`);
  if (!res.ok) throw new Error(`Error explorer failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReviewQueue(dataset = 'benchmark_v3', options = {}) {
  const qs = new URLSearchParams({ dataset });
  if (options.limit != null) qs.append('limit', String(options.limit));
  if (options.priorityOnly) qs.append('priority_only', 'true');
  const res = await fetch(`${API_BASE}/evaluation/review-queue?${qs}`);
  if (!res.ok) throw new Error(`Review queue failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReviewLab(options = {}) {
  const qs = new URLSearchParams();
  if (options.limit != null) qs.append('limit', String(options.limit));
  if (options.priorityOnly) qs.append('priority_only', 'true');
  const res = await fetch(`${API_BASE}/evaluation/review-lab?${qs.toString()}`);
  if (!res.ok) throw new Error(`Review lab failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReviewPacket(sampleId) {
  const qs = new URLSearchParams({ sample_id: sampleId });
  const res = await fetch(`${API_BASE}/evaluation/review-packet?${qs}`);
  if (!res.ok) throw new Error(`Review packet failed: ${res.statusText}`);
  return res.json();
}

export async function fetchBenchmarkV3Codebert(experimentId) {
  const params = new URLSearchParams();
  if (experimentId) params.append('experiment_id', experimentId);
  const qs = params.toString();
  const res = await fetch(
    `${API_BASE}/evaluation/benchmark-v3-codebert${qs ? `?${qs}` : ''}`,
  );
  if (!res.ok) throw new Error(`Benchmark v3 CodeBERT failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetSummary() {
  const res = await fetch(`${API_BASE}/datasets/summary`);
  if (!res.ok) throw new Error(`Dataset summary failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetSamples(page = 1, pageSize = 15, errorLabel = '', split = '') {
  const params = new URLSearchParams({ page, page_size: pageSize });
  if (errorLabel) params.append('error_label', errorLabel);
  if (split) params.append('split', split);
  
  const res = await fetch(`${API_BASE}/datasets/samples?${params.toString()}`);
  if (!res.ok) throw new Error(`Dataset samples failed: ${res.statusText}`);
  return res.json();
}

export async function triggerGenerateDataset() {
  const res = await fetch(`${API_BASE}/datasets/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirm: true }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Dataset generation failed');
  }
  return res.json();
}

export async function fetchHistory(limit = 50) {
  const res = await fetch(`${API_BASE}/history?limit=${limit}`);
  if (!res.ok) throw new Error(`History fetch failed: ${res.statusText}`);
  return res.json();
}

export async function clearHistory() {
  const res = await fetch(`${API_BASE}/history`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirm: true }),
  });
  if (!res.ok) throw new Error(`Clear history failed: ${res.statusText}`);
  return res.json();
}
