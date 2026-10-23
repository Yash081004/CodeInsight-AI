import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Grid, ShieldAlert, Award, Layers, RefreshCw } from 'lucide-react';
import {
  fetchEvaluationSummary,
  fetchConfusionMatrix,
  fetchEvaluationExperiments,
  fetchErrorGallery,
  fetchModelComparison,
  fetchAblations,
  fetchBenchmarkQuality,
  fetchBenchmarkV3Codebert,
  fetchPhase7ErrorAnalysis,
  fetchErrorExplorer,
  fetchReviewQueue,
  fetchReviewPacket,
} from '../api';

const CLASS_LABELS = [
  'correct',
  'wrong_initialization',
  'off_by_one_loop',
  'wrong_operator',
  'wrong_condition',
];

const SHORT_LABELS = [
  'Correct',
  'Wrong Init',
  'Off-by-One',
  'Wrong Op',
  'Wrong Cond',
];

export default function EvaluationTab() {
  const [summary, setSummary] = useState(null);
  const [cmData, setCmData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('codebert');
  const [selectedRun, setSelectedRun] = useState('production');
  const [runsMeta, setRunsMeta] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [gallery, setGallery] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [galleryFilter, setGalleryFilter] = useState({
    correct: 'all',
    trueLabel: '',
    model: '',
  });
  const [expandedId, setExpandedId] = useState(null);
  const [ablations, setAblations] = useState(null);
  const [benchVersion, setBenchVersion] = useState('benchmark_v3');
  const [benchQuality, setBenchQuality] = useState(null);
  const [v3Codebert, setV3Codebert] = useState(null);
  const [phase7, setPhase7] = useState(null);
  const [errorExplorer, setErrorExplorer] = useState(null);
  const [reviewQueue, setReviewQueue] = useState(null);
  const [reviewPacketDetail, setReviewPacketDetail] = useState(null);
  const [reviewPacketLoading, setReviewPacketLoading] = useState(false);
  const [explorerFilters, setExplorerFilters] = useState({
    split: 'val',
    model: 'codebert',
    trueLabel: '',
    predictedLabel: '',
    familyId: '',
  });

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [experiments, sum, cm, abl, bq, v3cb, p7, rq] = await Promise.all([
        fetchEvaluationExperiments(),
        fetchEvaluationSummary(selectedRun),
        fetchConfusionMatrix(selectedRun, selectedModel),
        fetchAblations(),
        fetchBenchmarkQuality(benchVersion),
        fetchBenchmarkV3Codebert(),
        fetchPhase7ErrorAnalysis(),
        fetchReviewQueue('benchmark_v3', { limit: 40 }),
      ]);
      setAblations(abl);
      setBenchQuality(bq);
      setV3Codebert(v3cb);
      setPhase7(p7);
      setReviewQueue(rq);
      const exParams = {
        split: explorerFilters.split,
        model: explorerFilters.model,
        limit: 40,
      };
      if (explorerFilters.trueLabel) exParams.true_label = explorerFilters.trueLabel;
      if (explorerFilters.predictedLabel) exParams.predicted_label = explorerFilters.predictedLabel;
      if (explorerFilters.familyId) exParams.family_id = explorerFilters.familyId;
      const ex = await fetchErrorExplorer(exParams);
      setErrorExplorer(ex);
      setRunsMeta(experiments);
      setSummary(sum);
      setCmData(cm);
      if (selectedRun.startsWith('benchmark_v2/codebert_')) {
        const gf = {
          limit: 40,
          correct: galleryFilter.correct === 'all' ? undefined : galleryFilter.correct === 'yes',
          trueLabel: galleryFilter.trueLabel || undefined,
          model: galleryFilter.model || undefined,
        };
        const [gal, cmp] = await Promise.all([
          fetchErrorGallery(selectedRun, gf),
          fetchModelComparison(selectedRun),
        ]);
        setGallery(gal);
        setComparison(cmp);
      } else {
        setGallery(null);
        setComparison(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to load evaluation metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [selectedRun, selectedModel, galleryFilter.correct, galleryFilter.trueLabel, galleryFilter.model, benchVersion]);

  const runLabel = (r) => {
    if (r.role === 'production') return 'Production metrics (API checkpoint)';
    if (r.role === 'benchmark_v2') {
      return r.run_id.includes('codebert_')
        ? `benchmark_v2 CodeBERT: ${r.run_id.split('/')[1]}`
        : `benchmark_v2 TF-IDF: ${r.run_id.split('/')[1]}`;
    }
    return `Experiment: ${r.run_id}`;
  };

  const activeMetrics = selectedModel === 'codebert'
    ? (summary?.codebert_metrics || summary?.codebert_classification)
    : (summary?.baseline_metrics || summary?.baseline_classification);

  const rawMatrix = cmData?.data?.matrix || cmData?.matrix || summary?.confusion_matrix?.matrix;
  const matrix = Array.isArray(rawMatrix) ? rawMatrix : null;

  const maxVal = matrix ? Math.max(...matrix.flat(), 1) : 1;
  const simMetrics = summary?.similarity_metrics || summary?.similarity;
  const simF1 = simMetrics?.tfidf_f1 ?? simMetrics?.f1;
  const codebertSimF1 = simMetrics?.codebert_f1 ?? null;
  const totalPairsCount = simMetrics?.test_pairs ?? simMetrics?.total_pairs ?? null;

  return (
    <div className="workspace">
      <header className="page-header" style={{ border: 'none', paddingBottom: 0 }}>
        <div>
          <h2 className="page-title">Evaluation</h2>
          <p className="page-subtitle" style={{ marginTop: 6 }}>
            Saved artifact metrics — synthetic labels; limitations preserved in detail views.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <select
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="btn btn-secondary btn-sm"
            style={{ minWidth: '200px' }}
            aria-label="Evaluation run"
          >
            {(runsMeta?.runs || [{ run_id: 'production', role: 'production' }]).map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {runLabel(r)}
              </option>
            ))}
          </select>
          <button type="button" onClick={() => setSelectedModel('codebert')} className={`btn btn-sm ${selectedModel === 'codebert' ? 'btn-primary' : 'btn-secondary'}`}>
            CodeBERT
          </button>
          <button type="button" onClick={() => setSelectedModel('baseline')} className={`btn btn-sm ${selectedModel === 'baseline' ? 'btn-primary' : 'btn-secondary'}`}>
            TF-IDF
          </button>
          <button type="button" onClick={loadData} disabled={loading} className="btn btn-secondary btn-sm" aria-label="Refresh">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </header>

      {error && (
        <div className="alert-banner alert-banner--error" role="alert">{error}</div>
      )}

      <div className="kpi-strip">
        <div className="kpi-strip-item">
          <div className="kpi-strip-label">{selectedModel === 'codebert' ? 'CodeBERT' : 'Baseline'} macro F1</div>
          <div className="kpi-strip-value">
            {activeMetrics?.macro_f1 != null ? `${(activeMetrics.macro_f1 * 100).toFixed(1)}%` : loading ? '…' : 'N/A'}
          </div>
        </div>
        <div className="kpi-strip-item">
          <div className="kpi-strip-label">Test accuracy</div>
          <div className="kpi-strip-value">
            {activeMetrics?.accuracy != null ? `${(activeMetrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
        <div className="kpi-strip-item">
          <div className="kpi-strip-label">Similarity F1 (test)</div>
          <div className="kpi-strip-value">
            {simF1 != null ? `${(simF1 * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
        <div className="kpi-strip-item">
          <div className="kpi-strip-label">Split / note</div>
          <div className="kpi-strip-value" style={{ fontSize: '0.85rem' }}>
            {summary?.eval_split || 'test'}
            {runsMeta?.similarity_preliminary ? ' · sim preliminary' : ''}
          </div>
        </div>
      </div>

      <details className="expand-section">
        <summary>Confusion matrix, per-class metrics &amp; experiments</summary>
        <div className="expand-section__body" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Confusion Matrix Heatmap & Per-Class Breakdown */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '20px' }}>
        
        {/* Confusion Matrix */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Grid size={18} color="var(--primary-light)" />
              Test Set Confusion Matrix
            </h3>
            <span className="badge badge-primary">{selectedModel}</span>
          </div>

          <p style={{ fontSize: '0.775rem', color: 'var(--text-faint)', marginBottom: '16px' }}>
            Rows represent Ground Truth labels; Columns represent Model Predictions.
          </p>

          {matrix ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', fontSize: '0.8rem' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '8px', color: 'var(--text-muted)', textAlign: 'left', fontWeight: 600 }}>Actual \ Predicted</th>
                    {SHORT_LABELS.map((lbl, idx) => (
                      <th key={idx} style={{ padding: '8px', color: 'var(--primary-light)', fontWeight: 600 }}>
                        {lbl}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((row, rIdx) => (
                    <tr key={rIdx}>
                      <td style={{ padding: '10px 8px', color: 'var(--text-muted)', textAlign: 'left', fontWeight: 600 }}>
                        {SHORT_LABELS[rIdx]}
                      </td>
                      {row.map((val, cIdx) => {
                        const isDiagonal = rIdx === cIdx;
                        const intensity = val > 0 ? (val / maxVal) : 0;
                        const bg = val === 0
                          ? 'rgba(255, 255, 255, 0.02)'
                          : isDiagonal
                            ? `rgba(99, 102, 241, ${0.25 + intensity * 0.55})`
                            : `rgba(244, 63, 94, ${0.15 + intensity * 0.45})`;

                        return (
                          <td
                            key={cIdx}
                            style={{
                              padding: '10px 6px',
                              background: bg,
                              border: '1px solid var(--border-subtle)',
                              color: val > 0 ? '#ffffff' : 'var(--text-faint)',
                              fontFamily: 'var(--font-mono)',
                              fontWeight: val > 0 ? 700 : 400,
                            }}
                          >
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No confusion matrix data available for this model.
            </div>
          )}
        </div>

        {/* Per-Class Metrics Table */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Award size={18} color="var(--accent-cyan)" />
              Per-Class Diagnostic Precision & Recall
            </h3>
          </div>

          <p style={{ fontSize: '0.775rem', color: 'var(--text-faint)', marginBottom: '16px' }}>
            Evaluation metrics decomposed across individual logical fault categories.
          </p>

          {activeMetrics?.per_class_metrics && (
            <div style={{ marginBottom: '20px' }}>
              {CLASS_LABELS.map((lbl, idx) => {
                const f1 = activeMetrics.per_class_metrics[lbl]?.f1 ?? 0;
                return (
                  <div key={lbl} style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                      <span>{SHORT_LABELS[idx]}</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{(f1 * 100).toFixed(0)}% F1</span>
                    </div>
                    <div style={{ height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', marginTop: '4px' }}>
                      <div
                        style={{
                          width: `${Math.min(100, f1 * 100)}%`,
                          height: '100%',
                          background: 'var(--primary)',
                          borderRadius: '4px',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '10px 8px', color: 'var(--text-muted)' }}>Class</th>
                  <th style={{ padding: '10px 8px', color: 'var(--text-muted)' }}>Precision</th>
                  <th style={{ padding: '10px 8px', color: 'var(--text-muted)' }}>Recall</th>
                  <th style={{ padding: '10px 8px', color: 'var(--text-muted)' }}>F1-Score</th>
                  <th style={{ padding: '10px 8px', color: 'var(--text-muted)' }}>Support</th>
                </tr>
              </thead>
              <tbody>
                {CLASS_LABELS.map((lbl, idx) => {
                  const perClass = activeMetrics?.per_class_metrics?.[lbl] || {};
                  return (
                    <tr key={lbl} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.03)' }}>
                      <td style={{ padding: '10px 8px', fontWeight: 600, color: '#ffffff' }}>
                        {SHORT_LABELS[idx]}
                      </td>
                      <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {perClass.precision !== undefined && perClass.precision !== null ? `${(perClass.precision * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {perClass.recall !== undefined && perClass.recall !== null ? `${(perClass.recall * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--primary-light)', fontWeight: 700 }}>
                        {perClass.f1 !== undefined && perClass.f1 !== null ? `${(perClass.f1 * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-faint)' }}>
                        {perClass.support !== undefined && perClass.support !== null ? perClass.support : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      <div className="glass-panel" style={{ padding: '16px 24px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '8px' }}>Dataset quality (Phase 5)</h3>
        <select
          value={benchVersion}
          onChange={(e) => setBenchVersion(e.target.value)}
          className="btn btn-secondary btn-sm"
          aria-label="Benchmark version"
        >
          <option value="benchmark_v2">benchmark_v2</option>
          <option value="benchmark_v3">benchmark_v3</option>
        </select>
        {!benchQuality?.available && (
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            No quality report for this version (generate v3 first).
          </p>
        )}
        {benchQuality?.available && (
          <div style={{ fontSize: '0.78rem', marginTop: '10px', color: 'var(--text-muted)' }}>
            <div>Families: {benchQuality.quality_report?.independent_program_families ?? '—'}</div>
            <div>Synthetic / unreviewed: {String(benchQuality.synthetic_unreviewed ?? true)}</div>
            <div>Coverage (train): {JSON.stringify(benchQuality.quality_report?.class_coverage_by_split?.train ?? {})}</div>
            {benchQuality.integrity?.errors?.length > 0 && (
              <div style={{ color: 'var(--accent-amber)' }}>Integrity: {benchQuality.integrity.errors.join('; ')}</div>
            )}
          </div>
        )}
      </div>

      {v3Codebert?.available && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '12px' }}>
            Phase 6 benchmark_v3 CodeBERT (validation selection)
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--accent-amber)', marginBottom: '10px' }}>
            Labels are synthetic and not human-reviewed. Validation selects checkpoints; final test (if shown) is
            a single post-selection eval — do not retune from it.
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
            TF-IDF validation macro F1:{' '}
            <strong>
              {v3Codebert.tfidf_baseline_val?.macro_f1 != null
                ? `${(v3Codebert.tfidf_baseline_val.macro_f1 * 100).toFixed(1)}%`
                : '—'}
            </strong>
            . Selected:{' '}
            <strong>{v3Codebert.selection?.selected_experiment_id ?? '—'}</strong>
            {v3Codebert.selection?.beats_tfidf_val_macro_f1 === false && (
              <span> (does not beat TF-IDF on validation macro F1)</span>
            )}
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.78rem', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ textAlign: 'left', padding: '8px' }}>Experiment</th>
                  <th style={{ padding: '8px' }}>Device</th>
                  <th style={{ padding: '8px' }}>Val macro F1</th>
                  <th style={{ padding: '8px' }}>Collapse?</th>
                  <th style={{ padding: '8px' }}>Final test F1</th>
                </tr>
              </thead>
              <tbody>
                {(v3Codebert.experiments || []).map((e) => (
                  <tr key={e.experiment_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>{e.experiment_id}</td>
                    <td style={{ padding: '8px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      {e.gpu_run ? 'GPU' : e.device_used || '—'}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>
                      {e.validation_macro_f1 != null ? `${(e.validation_macro_f1 * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>
                      {e.validation_collapse_warning ? 'yes' : 'no'}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>
                      {e.final_test_macro_f1 != null
                        ? `${(e.final_test_macro_f1 * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {phase7?.available && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldAlert size={18} color="var(--accent-amber)" />
            Phase 7 error analysis (synthetic v3 — not human-reviewed)
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
            {errorExplorer?.comparison_warning}
          </p>
          <div style={{ fontSize: '0.78rem', marginBottom: '12px' }}>
            Verified: {phase7.report?.verified_facts?.length ?? 0} · Both models wrong (val):{' '}
            {(phase7.report?.both_models_wrong_val_sample_ids || []).join(', ') || '—'}
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
            <select
              className="btn btn-secondary btn-sm"
              value={explorerFilters.split}
              onChange={(e) => setExplorerFilters((f) => ({ ...f, split: e.target.value }))}
            >
              <option value="val">val</option>
              <option value="test">test (historical)</option>
            </select>
            <select
              className="btn btn-secondary btn-sm"
              value={explorerFilters.model}
              onChange={(e) => setExplorerFilters((f) => ({ ...f, model: e.target.value }))}
            >
              <option value="codebert">CodeBERT gpu_3_balanced</option>
              <option value="tfidf">TF-IDF val replay</option>
            </select>
            <select
              className="btn btn-secondary btn-sm"
              value={explorerFilters.trueLabel}
              onChange={(e) => setExplorerFilters((f) => ({ ...f, trueLabel: e.target.value }))}
            >
              <option value="">All true labels</option>
              {CLASS_LABELS.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
            <button type="button" className="btn btn-secondary btn-sm" onClick={loadData}>
              Apply filters
            </button>
          </div>
          {errorExplorer?.available && (
            <div style={{ overflowX: 'auto', maxHeight: '320px' }}>
              <table style={{ width: '100%', fontSize: '0.72rem', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <th style={{ textAlign: 'left', padding: '6px' }}>ID</th>
                    <th style={{ padding: '6px' }}>Family</th>
                    <th style={{ padding: '6px' }}>True</th>
                    <th style={{ padding: '6px' }}>Pred</th>
                    <th style={{ padding: '6px' }}>OK?</th>
                  </tr>
                </thead>
                <tbody>
                  {(errorExplorer.entries || []).map((row) => (
                    <tr key={row.sample_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '6px', fontFamily: 'var(--font-mono)' }}>{row.sample_id}</td>
                      <td style={{ padding: '6px' }}>{row.family_id}</td>
                      <td style={{ padding: '6px' }}>{row.true_label}</td>
                      <td style={{ padding: '6px' }}>{row.predicted_label}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>{row.correct ? '✓' : '✗'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '12px' }}>
          Phase 8 — Human review &amp; reviewed eval slice
        </h3>
        {!reviewQueue?.available && (
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Review queue unavailable. Generate benchmark_v3 review artifacts first.
          </p>
        )}
        {reviewQueue?.available && (
          <>
            <p style={{ fontSize: '0.78rem', color: 'var(--accent-amber)', marginBottom: '10px' }}>
              Synthetic until adjudicated — not human-validated. CSV labels unchanged by review workflow.
            </p>
            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', fontSize: '0.8rem', marginBottom: '12px' }}>
              <span>Queue total: <strong>{reviewQueue.counts?.total_queue ?? '—'}</strong></span>
              <span>Pending: <strong>{reviewQueue.counts?.pending ?? '—'}</strong></span>
              <span>Adjudicated: <strong>{reviewQueue.counts?.adjudicated ?? 0}</strong></span>
              <span>Export-eligible: <strong>{reviewQueue.counts?.export_eligible ?? 0}</strong></span>
            </div>
            <div
              style={{
                padding: '12px',
                borderRadius: '8px',
                background: 'rgba(245, 158, 11, 0.08)',
                border: '1px solid rgba(245, 158, 11, 0.25)',
                fontSize: '0.78rem',
                marginBottom: '14px',
              }}
            >
              <div>
                Slice status:{' '}
                <strong>{reviewQueue.reviewed_eval_slice?.status ?? 'unknown'}</strong>
                {reviewQueue.reviewed_eval_slice?.blocked !== false && (
                  <span> (blocked)</span>
                )}
              </div>
              {reviewQueue.reviewed_eval_slice?.blocked_reason && (
                <div style={{ color: 'var(--text-muted)', marginTop: '4px' }}>
                  Reason: {reviewQueue.reviewed_eval_slice.blocked_reason}
                </div>
              )}
              <div style={{ marginTop: '6px' }}>
                Next action: {reviewQueue.reviewed_eval_slice?.next_action ?? reviewQueue.note}
              </div>
            </div>
            {reviewQueue.integrity_warnings?.length > 0 && (
              <p style={{ fontSize: '0.75rem', color: 'var(--accent-rose)', marginBottom: '8px' }}>
                Integrity: {reviewQueue.integrity_warnings.join('; ')}
              </p>
            )}
            <div style={{ overflowX: 'auto', maxHeight: '280px' }}>
              <table style={{ width: '100%', fontSize: '0.72rem', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <th style={{ textAlign: 'left', padding: '6px' }}>Review ID</th>
                    <th style={{ padding: '6px' }}>Sample</th>
                    <th style={{ padding: '6px' }}>Family</th>
                    <th style={{ padding: '6px' }}>Label</th>
                    <th style={{ padding: '6px' }}>Status</th>
                    <th style={{ padding: '6px' }}>Export?</th>
                  </tr>
                </thead>
                <tbody>
                  {(reviewQueue.items || []).map((row) => (
                    <tr
                      key={row.review_id}
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', cursor: 'pointer' }}
                      onClick={async () => {
                        if (!row.sample_id) return;
                        setReviewPacketLoading(true);
                        try {
                          const pkt = await fetchReviewPacket(row.sample_id);
                          setReviewPacketDetail(pkt);
                        } catch {
                          setReviewPacketDetail({ available: false });
                        } finally {
                          setReviewPacketLoading(false);
                        }
                      }}
                    >
                      <td style={{ padding: '6px', fontFamily: 'var(--font-mono)' }}>{row.review_id}</td>
                      <td style={{ padding: '6px' }}>{row.sample_id}</td>
                      <td style={{ padding: '6px' }}>{row.family_id}</td>
                      <td style={{ padding: '6px' }}>{row.current_label}</td>
                      <td style={{ padding: '6px' }}>{row.status}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>
                        {row.export_eligibility?.export_eligible ? 'yes' : 'no'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {reviewPacketLoading && (
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>Loading packet…</p>
            )}
            {reviewPacketDetail?.available && (
              <div style={{ marginTop: '12px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <div>
                  <strong>{reviewPacketDetail.packet?.sample_id}</strong> —{' '}
                  {reviewPacketDetail.packet?.problem_specification?.problem_statement ?? 'No problem statement in registry'}
                </div>
                <div>Mutation: {reviewPacketDetail.packet?.mutation_rationale ?? '—'}</div>
                {(() => {
                  const row = reviewQueue.items?.find(
                    (i) => i.sample_id === reviewPacketDetail.packet?.sample_id,
                  );
                  return (
                    <>
                      <div>Status: {row?.status ?? reviewPacketDetail.packet?.status}</div>
                      <div>Rationale: {row?.rationale ?? '(none — pending adjudication)'}</div>
                      <div>Expected behavior (dataset field): {row?.expected_behavior ?? '—'}</div>
                      {(row?.audit_history?.length ?? 0) > 0 && (
                        <pre style={{ fontSize: '0.7rem', overflow: 'auto', maxHeight: '100px' }}>
                          {JSON.stringify(row.audit_history, null, 2)}
                        </pre>
                      )}
                    </>
                  );
                })()}
              </div>
            )}
          </>
        )}
      </div>

      {ablations?.available && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '12px' }}>
            Phase 4 ablation comparison (validation selection)
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Selected by validation macro F1:{' '}
            <strong>{ablations.selection_summary?.selected_by_val_macro_f1 ?? '—'}</strong>.
            Test columns are historical / post-selection only — not mixed with validation charts.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.78rem', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ textAlign: 'left', padding: '8px' }}>Experiment</th>
                  <th style={{ padding: '8px' }}>Val macro F1</th>
                  <th style={{ padding: '8px' }}>Val collapse?</th>
                  <th style={{ padding: '8px' }}>Historical test F1</th>
                  <th style={{ padding: '8px' }}>LR / weights</th>
                </tr>
              </thead>
              <tbody>
                {(ablations.experiments || []).map((e) => (
                  <tr key={e.experiment_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>{e.experiment_id}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>
                      {e.validation_macro_f1 != null ? `${(e.validation_macro_f1 * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>
                      {e.validation_collapse_warning ? 'yes' : 'no'}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>
                      {e.historical_test_macro_f1 != null
                        ? `${(e.historical_test_macro_f1 * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      {e.training_config?.learning_rate ?? '—'} /{' '}
                      {e.training_config?.class_weights === false ? 'unweighted' : 'weighted'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {selectedRun.startsWith('benchmark_v2/codebert_') && comparison?.available && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={18} color="var(--accent-emerald)" />
            benchmark_v2 matched test comparison
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Same {comparison.data?.test_sample_count ?? '?'} test IDs, {comparison.data?.test_family_count ?? '?'} families.
            Softmax scores are not calibrated confidence.
          </p>
          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
            <span>TF-IDF macro F1: {(comparison.data?.baseline?.macro_f1 * 100).toFixed(1)}%</span>
            <span>CodeBERT macro F1: {(comparison.data?.codebert?.macro_f1 * 100).toFixed(1)}%</span>
            <span>Best val epoch: {comparison.data?.codebert?.best_epoch ?? 'N/A'}</span>
          </div>
        </div>
      )}

      {selectedRun.startsWith('benchmark_v2/codebert_') && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldAlert size={18} color="var(--accent-amber)" />
            Classification error gallery
          </h3>
          {!gallery?.available && (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              {gallery?.message || 'Gallery artifacts not found for this run (train CodeBERT v2 first).'}
            </p>
          )}
          {gallery?.available && (
            <>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '16px' }}>
                <select
                  className="btn btn-secondary btn-sm"
                  value={galleryFilter.correct}
                  onChange={(e) => setGalleryFilter((f) => ({ ...f, correct: e.target.value }))}
                  aria-label="Correct filter"
                >
                  <option value="all">All outcomes</option>
                  <option value="no">Any model wrong</option>
                  <option value="yes">Both correct</option>
                </select>
                <select
                  className="btn btn-secondary btn-sm"
                  value={galleryFilter.trueLabel}
                  onChange={(e) => setGalleryFilter((f) => ({ ...f, trueLabel: e.target.value }))}
                  aria-label="True label filter"
                >
                  <option value="">All true labels</option>
                  {CLASS_LABELS.map((l, i) => (
                    <option key={l} value={l}>{SHORT_LABELS[i]}</option>
                  ))}
                </select>
                <select
                  className="btn btn-secondary btn-sm"
                  value={galleryFilter.model}
                  onChange={(e) => setGalleryFilter((f) => ({ ...f, model: e.target.value }))}
                  aria-label="Model error filter"
                >
                  <option value="">Both models</option>
                  <option value="baseline">TF-IDF errors only</option>
                  <option value="codebert">CodeBERT errors only</option>
                </select>
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginBottom: '12px' }}>
                {gallery.probability_note}
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {(gallery.entries || []).map((entry) => (
                  <div
                    key={entry.sample_id}
                    style={{
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-md)',
                      padding: '12px',
                      background: 'rgba(255,255,255,0.02)',
                    }}
                  >
                    <button
                      type="button"
                      onClick={() => setExpandedId(expandedId === entry.sample_id ? null : entry.sample_id)}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        background: 'transparent',
                        border: 'none',
                        color: 'inherit',
                        cursor: 'pointer',
                      }}
                    >
                      <strong>{entry.sample_id}</strong>
                      <span style={{ marginLeft: '8px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                        family {entry.family_id} · true {entry.true_label}
                      </span>
                      <div style={{ fontSize: '0.8rem', marginTop: '4px' }}>
                        TF-IDF → {entry.baseline?.predicted_label}{' '}
                        {entry.baseline?.correct ? '✓' : '✗'} · CodeBERT → {entry.codebert?.predicted_label}{' '}
                        {entry.codebert?.correct ? '✓' : '✗'}
                      </div>
                    </button>
                    {expandedId === entry.sample_id && (
                      <pre
                        style={{
                          marginTop: '10px',
                          fontSize: '0.72rem',
                          overflow: 'auto',
                          maxHeight: '220px',
                          padding: '10px',
                          background: 'rgba(0,0,0,0.25)',
                          borderRadius: '6px',
                        }}
                      >
                        {entry.code}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

        </div>
      </details>

    </div>
  );
}
