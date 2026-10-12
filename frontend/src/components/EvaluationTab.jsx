import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Grid, ShieldAlert, Award, Layers, RefreshCw } from 'lucide-react';
import { fetchEvaluationSummary, fetchConfusionMatrix } from '../api';

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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sum, cm] = await Promise.all([
        fetchEvaluationSummary(),
        fetchConfusionMatrix(),
      ]);
      setSummary(sum);
      setCmData(cm);
    } catch (err) {
      setError(err.message || 'Failed to load evaluation metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const activeMetrics = selectedModel === 'codebert'
    ? (summary?.codebert_metrics || summary?.codebert_classification)
    : (summary?.baseline_metrics || summary?.baseline_classification);

  const rawMatrix = cmData?.data?.matrix || cmData?.matrix || summary?.confusion_matrix?.matrix;
  const matrix = Array.isArray(rawMatrix) ? rawMatrix : null;

  const maxVal = matrix ? Math.max(...matrix.flat(), 1) : 1;
  const simMetrics = summary?.similarity_metrics || summary?.similarity;
  const simF1 = simMetrics?.tfidf_f1 ?? simMetrics?.f1;
  const totalPairsCount = simMetrics?.total_pairs ?? null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      
      {/* Title & Model Selector Header */}
      <div className="glass-panel" style={{ padding: '20px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 800 }}>Model Benchmark & Empirical Evaluation</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            Empirical validation results on held-out test splits comparing Deep Learning vs Statistical Baseline
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.05)', borderRadius: 'var(--radius-sm)', padding: '3px' }}>
            <button
              onClick={() => setSelectedModel('codebert')}
              className="btn btn-sm"
              style={{
                background: selectedModel === 'codebert' ? 'var(--primary)' : 'transparent',
                color: selectedModel === 'codebert' ? '#ffffff' : 'var(--text-muted)',
                borderRadius: '6px',
              }}
            >
              CodeBERT (Fine-Tuned)
            </button>
            <button
              onClick={() => setSelectedModel('baseline')}
              className="btn btn-sm"
              style={{
                background: selectedModel === 'baseline' ? 'var(--primary)' : 'transparent',
                color: selectedModel === 'baseline' ? '#ffffff' : 'var(--text-muted)',
                borderRadius: '6px',
              }}
            >
              TF-IDF Baseline
            </button>
          </div>

          <button onClick={loadData} disabled={loading} className="btn btn-secondary btn-sm">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* KPI Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        
        <div className="glass-panel glow-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Test Accuracy</span>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--primary-light)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
            {activeMetrics?.accuracy !== undefined && activeMetrics?.accuracy !== null ? `${(activeMetrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
            Overall sample correctness
          </p>
        </div>

        <div className="glass-panel glow-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Macro F1-Score</span>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-cyan)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
            {activeMetrics?.macro_f1 !== undefined && activeMetrics?.macro_f1 !== null ? `${(activeMetrics.macro_f1 * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
            Unweighted class balance
          </p>
        </div>

        <div className="glass-panel glow-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Macro Precision</span>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-emerald)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
            {activeMetrics?.macro_precision !== undefined && activeMetrics?.macro_precision !== null ? `${(activeMetrics.macro_precision * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
            Positive predictive value
          </p>
        </div>

        <div className="glass-panel glow-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Macro Recall</span>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-amber)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
            {activeMetrics?.macro_recall !== undefined && activeMetrics?.macro_recall !== null ? `${(activeMetrics.macro_recall * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
            True positive sensitivity
          </p>
        </div>

        <div className="glass-panel glow-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Similarity Baseline F1</span>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-purple)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
            {simF1 !== undefined && simF1 !== null ? `${(simF1 * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
            {totalPairsCount ? `Evaluated on ${totalPairsCount} code pairs` : 'Evaluated on similarity split'}
          </p>
        </div>

      </div>

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

      {/* Comparative Analysis Table */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers size={18} color="var(--accent-emerald)" />
          Comparative Analysis: Deep Transformer vs Frequency Baseline
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div style={{ padding: '16px', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)' }}>
            <strong style={{ fontSize: '0.9rem', color: 'var(--primary-light)' }}>Representation Depth</strong>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px', lineHeight: 1.5 }}>
              CodeBERT encodes multi-head self-attention mechanisms over 12 transformer layers, understanding variable scopes and syntax dependencies. TF-IDF merely counts n-gram tokens.
            </p>
          </div>
          
          <div style={{ padding: '16px', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)' }}>
            <strong style={{ fontSize: '0.9rem', color: 'var(--accent-cyan)' }}>Identifier Invariance</strong>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px', lineHeight: 1.5 }}>
              Renaming variables degrades TF-IDF similarity significantly due to out-of-vocabulary or mismatched terms, while CodeBERT retains robust semantic similarity.
            </p>
          </div>

          <div style={{ padding: '16px', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)' }}>
            <strong style={{ fontSize: '0.9rem', color: 'var(--accent-amber)' }}>Sample Efficiency & Fine-tuning</strong>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px', lineHeight: 1.5 }}>
              With pre-trained bimodal weights from CodeSearchNet, CodeBERT provides transfer learning foundations that can be specialized to nuanced logic fault detection with domain-specific datasets.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}
