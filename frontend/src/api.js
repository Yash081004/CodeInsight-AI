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

export async function analyzeClassification(code) {
  const res = await fetch(`${API_BASE}/classification/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Classification request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchEvaluationSummary() {
  const res = await fetch(`${API_BASE}/evaluation/summary`);
  if (!res.ok) throw new Error(`Evaluation summary failed: ${res.statusText}`);
  return res.json();
}

export async function fetchConfusionMatrix() {
  const res = await fetch(`${API_BASE}/evaluation/confusion-matrix`);
  if (!res.ok) throw new Error(`Confusion matrix failed: ${res.statusText}`);
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
