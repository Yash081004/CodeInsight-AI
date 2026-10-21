import React, { useState, useEffect, useCallback } from 'react';
import { Play, Info, Lightbulb, RotateCcw } from 'lucide-react';
import { analyzeClassification } from '../api';
import CodeEditor from './ui/CodeEditor';
import AlertBanner from './ui/AlertBanner';
import { ERROR_LABELS } from '../constants/labels';
import { useAbortableRequest } from '../hooks/useAbortableRequest';

const CLASSIFICATION_PRESETS = [
  { name: 'Correct', code: `def sum_positive_numbers(arr):\n    total = 0\n    for num in arr:\n        if num > 0:\n            total += num\n    return total` },
  { name: 'Wrong init', code: `def compute_factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result` },
  { name: 'Off-by-one', code: `def sum_one_to_n(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total` },
  { name: 'Wrong operator', code: `def find_maximum(a, b):\n    if a < b:\n        return a\n    return b` },
  { name: 'Wrong condition', code: `def is_even_number(n):\n    if n % 2 == 1:\n        return True\n    return False` },
];

const LABEL_STYLE = {
  correct: { bg: 'color-mix(in srgb, var(--mineral) 18%, transparent)', text: 'var(--mineral)', border: 'var(--mineral)' },
  wrong_initialization: { bg: 'color-mix(in srgb, var(--ochre) 18%, transparent)', text: 'var(--ochre)', border: 'var(--ochre)' },
  off_by_one_loop: { bg: 'color-mix(in srgb, var(--vermilion) 15%, transparent)', text: 'var(--vermilion)', border: 'var(--vermilion)' },
  wrong_operator: { bg: 'color-mix(in srgb, var(--ochre) 15%, transparent)', text: 'var(--ochre)', border: 'var(--ochre)' },
  wrong_condition: { bg: 'color-mix(in srgb, var(--mineral) 12%, transparent)', text: 'var(--text-primary)', border: 'var(--mineral)' },
  unknown: { bg: 'color-mix(in srgb, var(--text-muted) 12%, transparent)', text: 'var(--text-muted)', border: 'var(--border-subtle)' },
};

const labelStyles = (predictedLabel) =>
  (predictedLabel && LABEL_STYLE[predictedLabel]) || LABEL_STYLE.unknown;

const labelFriendly = (key) => ERROR_LABELS.find((l) => l.id === key)?.name || key;

const INFERENCE_SOURCE_LABELS = {
  codebert_finetuned: 'CodeBERT',
  baseline_fallback: 'TF-IDF baseline (fallback)',
  baseline: 'TF-IDF baseline',
  spec_rule_assist: 'Specification-assisted rule',
  unavailable: 'No model',
  none: 'None',
};

const REVIEW_STATUS_COPY = {
  model_primary: 'Model-primary',
  review_required: 'Review required — weak discrimination',
  inconclusive: 'Inconclusive spec check',
  spec_assisted: 'Spec-assisted (deterministic)',
  baseline_routed: 'Baseline routed',
};

/** QA / demo cases — same semantics as backend diagnostic tests */
export const DIAGNOSTIC_CASES = [
  { id: '1', name: 'Square (correct)', code: 'def square(n):\n    return n * n', problem: null },
  {
    id: '2',
    name: 'Square (+ bug)',
    code: 'def square(n):\n    return n + n',
    problem: 'Return the square of the given number.',
  },
  {
    id: '3',
    name: 'Inclusive sum',
    code: 'def sum_to_n(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total',
    problem: 'Return the sum of all integers from 1 through n, inclusive.',
  },
  {
    id: '4',
    name: 'Adult threshold',
    code: 'def is_adult(age):\n    if age < 18:\n        return True\n    return False',
    problem: 'Return True if age is at least 18, otherwise return False.',
  },
  {
    id: '5',
    name: 'Sum init',
    code: 'def sum_values(values):\n    total = 1\n    for value in values:\n        total += value\n    return total',
    problem: 'Return the sum of all elements in the list. The sum of an empty list should be zero.',
  },
];

export default function ClassifierTab({ onHistoryUpdate }) {
  const [code, setCode] = useState(CLASSIFICATION_PRESETS[1].code);
  const [problemStatement, setProblemStatement] = useState('');
  const [functionContext, setFunctionContext] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const { run } = useAbortableRequest();

  const handleRunClassification = useCallback(async () => {
    if (!code.trim()) {
      setError('Provide Python source code to analyze.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const { stale, data } = await run(() =>
        analyzeClassification(code, {
          problemStatement: problemStatement.trim() || undefined,
          functionContext: functionContext.trim() || undefined,
        }),
      );
      if (stale) return;
      setResult(data);
      if (onHistoryUpdate) onHistoryUpdate();
    } catch (err) {
      setError(err.message || 'Classification request failed.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [code, problemStatement, functionContext, onHistoryUpdate, run]);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!loading) handleRunClassification();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleRunClassification, loading]);

  const scores = result?.probabilities || {};
  const routing = result?.diagnostics?.routing || {};
  const scoreForLabel = routing.score_applies_to_displayed_label !== false;
  const codebertSnap = result?.diagnostics?.codebert_snapshot;
  const baselineSnap = result?.diagnostics?.baseline_snapshot;
  const styles = result ? labelStyles(result.predicted_label) : null;
  const sourceLabel = result ? (INFERENCE_SOURCE_LABELS[result.model_type] || result.model_type) : '';
  const reviewLabel = result?.diagnostics?.review_status
    ? (REVIEW_STATUS_COPY[result.diagnostics.review_status] || result.diagnostics.review_status)
    : '';
  const scoreAttribution = result
    ? (scoreForLabel
      ? 'Displayed label uses model softmax scores (not calibrated accuracy).'
      : 'Displayed label is from a spec rule — softmax scores are CodeBERT context only.')
    : '';

  return (
    <div className="workspace">
      <header className="page-header" style={{ border: 'none', paddingBottom: 0 }}>
        <div>
          <h2 className="page-title">Logic classifier</h2>
          <p className="page-subtitle" style={{ marginTop: 6 }}>
            Model hypothesis only — not a verified diagnosis.
          </p>
        </div>
        <button type="button" className="btn btn-subtle btn-sm" onClick={() => { setCode(''); setResult(null); setError(null); }}>
          <RotateCcw size={14} /> Clear
        </button>
      </header>

      <CodeEditor label="Python source" value={code} onChange={setCode} minHeight="300px" placeholder="# Paste a function…" />

      <label htmlFor="problem-statement" className="meta-label" style={{ display: 'block' }}>
        Intended behavior (optional)
      </label>
      <textarea
        id="problem-statement"
        value={problemStatement}
        onChange={(e) => setProblemStatement(e.target.value)}
        placeholder="Describe expected behavior for spec-assisted checks when the model is uncertain."
        className="problem-statement-input"
        rows={2}
      />

      <details className="expand-section">
        <summary>Examples &amp; advanced inputs</summary>
        <div className="expand-section__body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {CLASSIFICATION_PRESETS.map((p) => (
              <button key={p.name} type="button" onClick={() => { setCode(p.code); setResult(null); setError(null); }} className="btn btn-secondary btn-sm">
                {p.name}
              </button>
            ))}
          </div>
          <span className="meta-label">Diagnostic QA set</span>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {DIAGNOSTIC_CASES.map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => {
                  setCode(c.code);
                  setProblemStatement(c.problem || '');
                  setResult(null);
                  setError(null);
                }}
                className="btn btn-secondary btn-sm"
              >
                {c.name}
              </button>
            ))}
          </div>
          <label htmlFor="function-context" className="meta-label">Function context (optional API field)</label>
          <input
            id="function-context"
            type="text"
            value={functionContext}
            onChange={(e) => setFunctionContext(e.target.value)}
            className="problem-statement-input"
            style={{ minHeight: 'auto' }}
            placeholder="e.g. loop-heavy, arithmetic"
          />
          <p className="page-kbd-hint">Analyze with <kbd>Ctrl</kbd>+<kbd>Enter</kbd></p>
        </div>
      </details>

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button type="button" onClick={handleRunClassification} disabled={loading} className={`btn btn-primary btn-lg ${loading ? 'btn-loading' : ''}`}>
          {loading ? 'Analyzing…' : (<><Play size={16} fill="currentColor" /> Analyze</>)}
        </button>
      </div>

      {error && <AlertBanner variant="error" data-testid="clf-error">{error}</AlertBanner>}

      {result && (
        <div
          data-testid="clf-result"
          className={`instrument-panel workbench-result ${result.diagnostics?.uncertainty_band === 'high' || result.diagnostics?.review_status === 'review_required' ? 'workbench-result--uncertain' : ''} ${result.model_type === 'spec_rule_assist' ? 'workbench-result--spec' : ''}`}
          style={{ padding: '20px 22px' }}
        >
          <div className="result-hero" style={{ marginBottom: 16, background: 'var(--bg-card)' }}>
            <div>
              <div className="meta-label">Predicted label · not verified</div>
              <span
                data-testid="clf-result-label"
                className="badge badge-display"
                style={{ marginTop: 8, background: styles.bg, color: styles.text, borderColor: styles.border }}
              >
                {result.friendly_label || labelFriendly(result.predicted_label)}
              </span>
            </div>
            <div className="result-hero-meta">
              <div data-testid="clf-result-source">
                Source: <strong className="text-strong">{sourceLabel}</strong>
              </div>
              {reviewLabel && (
                <div data-testid="clf-review-status" style={{ color: 'var(--ochre)', marginTop: 6 }}>
                  {reviewLabel}
                </div>
              )}
              {scoreAttribution && (
                <p data-testid="clf-score-attribution" style={{ marginTop: 8, color: 'var(--text-muted)', maxWidth: '32rem' }}>
                  {scoreAttribution}
                </p>
              )}
              {result.diagnostics?.evidence_summary && (
                <p data-testid="clf-evidence" style={{ marginTop: 8, maxWidth: '32rem' }}>
                  {result.diagnostics.evidence_summary}
                </p>
              )}
            </div>
          </div>

          {result.diagnostics?.collapse_warning && (
            <AlertBanner variant="warning" title="Model quality" data-testid="clf-collapse-warning">
              {result.diagnostics.collapse_warning}
              {result.diagnostics.spec_hint && (
                <span> Spec rule (not neural): {result.diagnostics.spec_hint.message}</span>
              )}
            </AlertBanner>
          )}

          {codebertSnap && result.model_type !== 'codebert_finetuned' && (
            <p style={{ fontSize: '0.78rem', color: 'var(--text-faint)', marginTop: 12 }}>
              CodeBERT alone would predict: {labelFriendly(codebertSnap.predicted_label)}
              {baselineSnap ? ` · TF-IDF: ${labelFriendly(baselineSnap.predicted_label)}` : ''}
            </p>
          )}

          <details className="expand-section" style={{ marginTop: 16 }}>
            <summary>Scores &amp; diagnostics</summary>
            <div className="expand-section__body">
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                {result.model_type === 'spec_rule_assist'
                  ? 'Softmax below is CodeBERT context — not confidence in the spec label.'
                  : result.diagnostics?.confidence_note}
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {Object.entries(scores).map(([labelKey, score]) => {
                  const isSelected = scoreForLabel ? labelKey === result.predicted_label : labelKey === codebertSnap?.predicted_label;
                  const percent = (score * 100).toFixed(1);
                  return (
                    <div key={labelKey} style={{ padding: '8px 10px', border: `1px solid ${isSelected ? 'var(--mineral)' : 'var(--border-subtle)'}`, borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                        <span>{labelFriendly(labelKey)}{!scoreForLabel && labelKey === result.predicted_label ? ' · displayed (spec)' : ''}</span>
                        <span style={{ fontFamily: 'var(--font-mono)' }}>{percent}%</span>
                      </div>
                      <div className="score-bar">
                        <div className={`score-bar-fill ${isSelected ? 'score-bar-fill--selected' : ''}`} style={{ width: `${percent}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              {result.diagnostics?.known_model_limitations?.length > 0 && (
                <ul style={{ fontSize: '0.72rem', color: 'var(--ochre)', marginTop: 14, paddingLeft: 18 }}>
                  {result.diagnostics.known_model_limitations.map((line, i) => (
                    <li key={i}>{line}</li>
                  ))}
                </ul>
              )}
              {result.diagnostics?.static_analysis?.length > 0 && (
                <ul style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 10, paddingLeft: 18 }}>
                  {result.diagnostics.static_analysis.map((f, i) => (
                    <li key={i}>{f.message}</li>
                  ))}
                </ul>
              )}
            </div>
          </details>

          {result.feedback && (
            <details className="expand-section" style={{ marginTop: 12 }}>
              <summary>Pedagogical note</summary>
              <div className="expand-section__body" style={{ display: 'flex', gap: 10 }}>
                <Lightbulb size={18} color="var(--ochre)" aria-hidden />
                <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>{result.feedback}</p>
              </div>
            </details>
          )}

          <p style={{ display: 'flex', gap: 8, fontSize: '0.72rem', color: 'var(--text-faint)', marginTop: 14 }}>
            <Info size={14} aria-hidden />
            No code was executed unless you explicitly enable tests in the API.
          </p>
        </div>
      )}
    </div>
  );
}
