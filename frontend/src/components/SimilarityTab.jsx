import React, { useState, useEffect, useCallback } from 'react';

import { Play, ArrowLeftRight, RotateCcw, RefreshCw } from 'lucide-react';

import { analyzeSimilarity } from '../api';

import CodeEditor from './ui/CodeEditor';

import AlertBanner from './ui/AlertBanner';

import { useAbortableRequest } from '../hooks/useAbortableRequest';



const THRESHOLD_OPTIONS = [
  ...Array.from({ length: 18 }, (_, i) => Math.round((0.1 + i * 0.05) * 100) / 100),
  0.99,
];

const PRESETS = [

  {

    name: 'Renaming',

    codeA: `def calculate_sum(numbers_list):

    total_value = 0

    for number in numbers_list:

        total_value += number

    return total_value`,

    codeB: `def calculate_sum(arr):

    s = 0

    for x in arr:

        s += x

    return s`,

  },

  {

    name: 'Loop vs comprehension',

    codeA: `def filter_evens(nums):

    result = []

    for n in nums:

        if n % 2 == 0:

            result.append(n)

    return result`,

    codeB: `def filter_evens(nums):

    return [n for n in nums if n % 2 == 0]`,

  },

  {

    name: 'Unrelated',

    codeA: `def bubble_sort(arr):

    n = len(arr)

    for i in range(n):

        for j in range(0, n - i - 1):

            if arr[j] > arr[j + 1]:

                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr`,

    codeB: `def binary_search(arr, target):

    low, high = 0, len(arr) - 1

    while low <= high:

        mid = (low + high) // 2

        if arr[mid] == target:

            return mid

        elif arr[mid] < target:

            low = mid + 1

        else:

            high = mid - 1

    return -1`,

  },

];



export default function SimilarityTab({ onHistoryUpdate }) {

  const [codeA, setCodeA] = useState(PRESETS[0].codeA);

  const [codeB, setCodeB] = useState(PRESETS[0].codeB);

  const [threshold, setThreshold] = useState(0.8);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState(null);

  const [result, setResult] = useState(null);

  const { run } = useAbortableRequest();



  const handleRunAnalysis = useCallback(async () => {

    if (!codeA.trim() || !codeB.trim()) {

      setError('Provide Python code in both Snippet A and Snippet B.');

      return;

    }

    setLoading(true);

    setError(null);

    try {

      const { stale, data } = await run(() => analyzeSimilarity(codeA, codeB, threshold));

      if (stale) return;

      setResult(data);

      if (onHistoryUpdate) onHistoryUpdate();

    } catch (err) {

      setError(err.message || 'Similarity analysis failed. Is the backend running on port 8000?');

      setResult(null);

    } finally {

      setLoading(false);

    }

  }, [codeA, codeB, threshold, onHistoryUpdate, run]);



  useEffect(() => {

    const onKey = (e) => {

      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !loading) {

        e.preventDefault();

        handleRunAnalysis();

      }

    };

    window.addEventListener('keydown', onKey);

    return () => window.removeEventListener('keydown', onKey);

  }, [handleRunAnalysis, loading]);



  const decisionScore = result
    ? (result.decision_model === 'tfidf' ? result.tfidf_score : result.codebert_score ?? result.tfidf_score)
    : null;
  const method = result?.decision_model || 'codebert';



  return (

    <div className="workspace">

      <header className="page-header" style={{ border: 'none', paddingBottom: 0 }}>

        <div>

          <h2 className="page-title">Compare snippets</h2>

          <p className="page-subtitle" style={{ marginTop: 6 }}>

            Side-by-side Python — real CodeBERT and TF-IDF scores from the API.

          </p>

        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>

          <button type="button" onClick={() => { setCodeA(codeB); setCodeB(codeA); setResult(null); }} className="btn btn-subtle btn-sm">

            <ArrowLeftRight size={14} /> Swap

          </button>

          <button type="button" onClick={() => { setCodeA(''); setCodeB(''); setResult(null); setError(null); }} className="btn btn-subtle btn-sm">

            <RotateCcw size={14} /> Reset

          </button>

        </div>

      </header>



      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>

        {PRESETS.map((p) => (

          <button key={p.name} type="button" className="btn btn-secondary btn-sm" onClick={() => { setCodeA(p.codeA); setCodeB(p.codeB); setResult(null); setError(null); }}>

            {p.name}

          </button>

        ))}

      </div>



      <div className="comparison-stage">

        <CodeEditor label="Snippet A" value={codeA} onChange={setCodeA} minHeight="340px" placeholder="# Python…" />

        <CodeEditor label="Snippet B" value={codeB} onChange={setCodeB} minHeight="340px" placeholder="# Python…" />

      </div>



      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>

        <button

          type="button"

          data-testid="sim-compare-btn"

          aria-busy={loading}

          onClick={handleRunAnalysis}

          disabled={loading}

          className={`btn btn-primary btn-lg ${loading ? 'btn-loading' : ''}`}

        >

          {loading ? 'Comparing…' : (<><Play size={16} fill="currentColor" /> Compare</>)}

        </button>

      </div>



      <details className="expand-section">

        <summary>Threshold &amp; shortcut</summary>

        <div className="expand-section__body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          <label htmlFor="similarity-threshold" style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>

            Similarity threshold: <strong className="text-strong" style={{ fontFamily: 'var(--font-mono)' }}>{threshold.toFixed(2)}</strong>

          </label>

          <input
            id="similarity-threshold"
            data-testid="sim-threshold-input"
            type="range"
            min="0.1"
            max="0.99"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            onInput={(e) => setThreshold(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--vermilion)' }}
          />
          <label htmlFor="similarity-threshold-select" className="meta-label" style={{ marginTop: 8 }}>
            Exact threshold
          </label>
          <select
            id="similarity-threshold-select"
            data-testid="sim-threshold-select"
            className="btn btn-secondary btn-sm"
            style={{ width: '100%', marginTop: 4 }}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          >
            {THRESHOLD_OPTIONS.map((t) => (
              <option key={t} value={t}>
                {t.toFixed(2)}
              </option>
            ))}
          </select>
          <span data-testid="sim-threshold-value" className="sr-only">{threshold.toFixed(2)}</span>

          <p className="page-kbd-hint">Compare with <kbd>Ctrl</kbd>+<kbd>Enter</kbd></p>

        </div>

      </details>



      {error && (

        <AlertBanner variant="error" title="Request failed" data-testid="sim-error">

          {error}

          <div style={{ marginTop: 10 }}>

            <button type="button" className="btn btn-secondary btn-sm" onClick={handleRunAnalysis} disabled={loading}>

              <RefreshCw size={14} /> Retry

            </button>

          </div>

        </AlertBanner>

      )}



      {result && (

        <div className="result-hero" data-testid="sim-result">

          <div>

            <div className="meta-label">Decision score ({method})</div>

            <div className="result-hero-score" data-testid="sim-decision-score">{decisionScore != null ? decisionScore.toFixed(3) : '—'}</div>

          </div>

          <div className="result-hero-meta">

            <div data-testid="sim-verdict">

              {result.is_similar ? (

                <span style={{ color: 'var(--mineral)' }}>Similar ({result.decision_model || 'policy'})</span>

              ) : (

                <span style={{ color: 'var(--ochre)' }}>Not similar ({result.decision_model || 'policy'})</span>

              )}

            </div>

            {result.review_recommended && (
              <p data-testid="sim-review" style={{ color: 'var(--ochre)', marginTop: 6, fontSize: '0.78rem' }}>
                Review recommended — embedding match is not verified equivalence.
              </p>
            )}

            {result.explanation && <p data-testid="sim-explanation" style={{ marginTop: 8, maxWidth: '28rem' }}>{result.explanation}</p>}

          </div>

          <span data-testid="sim-codebert-score" className="sr-only">{result.codebert_score != null ? result.codebert_score.toFixed(4) : ''}</span>

          <span data-testid="sim-tfidf-score" className="sr-only">{result.tfidf_score != null ? result.tfidf_score.toFixed(4) : ''}</span>

          <span data-testid="sim-is-similar" className="sr-only">{String(result.is_similar)}</span>
          <span data-testid="sim-decision-model-id" className="sr-only">{result.decision_model || ''}</span>

        </div>

      )}



      {result && (

        <details className="expand-section">

          <summary>Both model scores</summary>

          <div className="expand-section__body kpi-strip">

            <div className="kpi-strip-item">

              <div className="kpi-strip-label">CodeBERT cosine</div>

              <div className="kpi-strip-value">{result.codebert_score != null ? result.codebert_score.toFixed(4) : 'N/A'}</div>

            </div>

            <div className="kpi-strip-item">

              <div className="kpi-strip-label">TF-IDF cosine</div>

              <div className="kpi-strip-value">{result.tfidf_score != null ? result.tfidf_score.toFixed(4) : 'N/A'}</div>

            </div>

          </div>

        </details>

      )}

    </div>

  );

}

