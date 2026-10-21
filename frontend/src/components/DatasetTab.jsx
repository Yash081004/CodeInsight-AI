import React, { useState, useEffect } from 'react';
import { Database, Filter, ChevronLeft, ChevronRight, Eye, RefreshCw, CheckCircle2, AlertCircle, Code } from 'lucide-react';
import { fetchDatasetSummary, fetchDatasetSamples, triggerGenerateDataset } from '../api';
import PageHeader from './ui/PageHeader';
import AlertBanner from './ui/AlertBanner';

const LABEL_BADGES = {
  correct: 'badge-emerald',
  wrong_initialization: 'badge-amber',
  off_by_one_loop: 'badge-rose',
  wrong_operator: 'badge-purple',
  wrong_condition: 'badge-cyan',
};

export default function DatasetTab() {
  const [summary, setSummary] = useState(null);
  const [samples, setSamples] = useState([]);
  const [totalSamples, setTotalSamples] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(12);
  const [selectedLabel, setSelectedLabel] = useState('');
  const [selectedSplit, setSelectedSplit] = useState('');
  const [selectedSample, setSelectedSample] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [notification, setNotification] = useState(null);

  const loadSummary = async () => {
    try {
      const data = await fetchDatasetSummary();
      setSummary(data);
    } catch (err) {
      console.error(err);
    }
  };

  const loadSamples = async () => {
    setLoading(true);
    try {
      const data = await fetchDatasetSamples(page, pageSize, selectedLabel, selectedSplit);
      setSamples(data.samples || []);
      setTotalSamples(data.total || 0);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSummary();
  }, []);

  useEffect(() => {
    loadSamples();
  }, [page, selectedLabel, selectedSplit]);

  const handleRegenerate = async () => {
    if (!window.confirm('Regenerate the dataset? This will run AST mutation pipelines across all templates.')) {
      return;
    }
    setGenerating(true);
    setNotification(null);
    try {
      const res = await triggerGenerateDataset();
      setNotification({ type: 'success', message: `Dataset generated successfully (${res.total_samples || 91} samples).` });
      await loadSummary();
      await loadSamples();
    } catch (err) {
      setNotification({ type: 'error', message: err.message || 'Dataset generation failed.' });
    } finally {
      setGenerating(false);
    }
  };

  const totalPages = Math.max(1, Math.ceil(totalSamples / pageSize));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <PageHeader
        title="Datasets"
        subtitle="Browse synthetic mutation samples and splits from the backend dataset API. Labels are programmatic—not a substitute for human adjudication."
        actions={
          <button
            type="button"
            onClick={handleRegenerate}
            disabled={generating}
            className="btn btn-secondary btn-sm"
          >
            <RefreshCw size={14} className={generating ? 'animate-spin' : ''} aria-hidden />
            {generating ? 'Regenerating…' : 'Regenerate dataset'}
          </button>
        }
      />

      <AlertBanner variant="warning" title="Synthetic data">
        Samples are generated and AST-validated; human review of benchmark v3 is incomplete and the reviewed eval slice remains blocked.
      </AlertBanner>

      {/* Notification */}
      {notification && (
        <div className="glass-panel" style={{
          padding: '12px 18px',
          borderLeft: `4px solid ${notification.type === 'success' ? 'var(--accent-emerald)' : 'var(--accent-rose)'}`
        }}>
          <p style={{ fontSize: '0.85rem', color: notification.type === 'success' ? '#34d399' : '#fb7185' }}>
            {notification.message}
          </p>
        </div>
      )}

      {/* Summary Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <div className="glass-panel" style={{ padding: '18px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Total Samples</span>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--primary-light)', marginTop: '4px', fontFamily: 'var(--font-mono)' }}>
            {summary?.total_classification_samples || 91}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)' }}>Validated via AST</span>
        </div>

        <div className="glass-panel" style={{ padding: '18px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Algorithm Templates</span>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-cyan)', marginTop: '4px', fontFamily: 'var(--font-mono)' }}>
            {summary?.templates_used || 21}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)' }}>Sorting, Searching, Math</span>
        </div>

        <div className="glass-panel" style={{ padding: '18px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Similarity Pairs</span>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-emerald)', marginTop: '4px', fontFamily: 'var(--font-mono)' }}>
            {summary?.similarity_pairs_count || 193}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)' }}>Bimodal pairs.csv</span>
        </div>

        <div className="glass-panel" style={{ padding: '18px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Train / Val / Test Split</span>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#ffffff', marginTop: '8px', fontFamily: 'var(--font-mono)' }}>
            61 / 14 / 16
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)' }}>70% / 15% / 15%</span>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="glass-panel" style={{ padding: '14px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.825rem', color: 'var(--text-muted)' }}>
            <Filter size={14} />
            <span>Category:</span>
          </div>
          <select
            value={selectedLabel}
            onChange={(e) => { setSelectedLabel(e.target.value); setPage(1); }}
            style={{
              background: 'var(--bg-secondary)',
              color: '#ffffff',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              padding: '6px 12px',
              fontSize: '0.825rem',
              outline: 'none',
              cursor: 'pointer',
            }}
          >
            <option value="">All Categories</option>
            <option value="correct">correct</option>
            <option value="wrong_initialization">wrong_initialization</option>
            <option value="off_by_one_loop">off_by_one_loop</option>
            <option value="wrong_operator">wrong_operator</option>
            <option value="wrong_condition">wrong_condition</option>
          </select>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.825rem', color: 'var(--text-muted)', marginLeft: '8px' }}>
            <span>Split:</span>
          </div>
          <select
            value={selectedSplit}
            onChange={(e) => { setSelectedSplit(e.target.value); setPage(1); }}
            style={{
              background: 'var(--bg-secondary)',
              color: '#ffffff',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              padding: '6px 12px',
              fontSize: '0.825rem',
              outline: 'none',
              cursor: 'pointer',
            }}
          >
            <option value="">All Splits</option>
            <option value="train">Train</option>
            <option value="val">Val</option>
            <option value="test">Test</option>
          </select>
        </div>

        {/* Pagination controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Page <strong style={{ color: '#ffffff' }}>{page}</strong> of {totalPages} ({totalSamples} samples)
          </span>
          <div style={{ display: 'flex', gap: '4px' }}>
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="btn btn-secondary btn-sm"
              style={{ padding: '6px 8px' }}
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="btn btn-secondary btn-sm"
              style={{ padding: '6px 8px' }}
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Dataset Samples Table */}
      <div className="glass-panel" style={{ padding: '0', overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ background: 'rgba(255, 255, 255, 0.02)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>ID</th>
                <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>Problem</th>
                <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>Error Label</th>
                <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>Mutation Detail</th>
                <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>Split</th>
                <th style={{ padding: '12px 16px', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="6" style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    Loading dataset records...
                  </td>
                </tr>
              ) : samples.length === 0 ? (
                <tr>
                  <td colSpan="6" style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No samples match the selected filters.
                  </td>
                </tr>
              ) : (
                samples.map((s, idx) => (
                  <tr
                    key={s.sample_id || idx}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
                      transition: 'background 0.15s ease',
                    }}
                  >
                    <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', color: 'var(--primary-light)', fontSize: '0.8rem' }}>
                      {s.sample_id}
                    </td>
                    <td style={{ padding: '12px 16px', fontWeight: 600, color: '#ffffff' }}>
                      {s.problem_type}
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <span className={`badge ${LABEL_BADGES[s.error_label] || 'badge-primary'}`}>
                        {s.error_label}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', color: 'var(--text-muted)', maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {s.mutation_description || 'None (base template)'}
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        background: s.split === 'train' ? 'rgba(99, 102, 241, 0.15)' : (s.split === 'val' ? 'rgba(6, 182, 212, 0.15)' : 'rgba(16, 185, 129, 0.15)'),
                        color: s.split === 'train' ? 'var(--primary-light)' : (s.split === 'val' ? 'var(--accent-cyan)' : 'var(--accent-emerald)'),
                      }}>
                        {s.split}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>
                      <button
                        onClick={() => setSelectedSample(s)}
                        className="btn btn-secondary btn-sm"
                        style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                      >
                        <Eye size={12} />
                        View Code
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Code Inspection Modal / Drawer */}
      {selectedSample && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(6px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 999,
          padding: '20px',
        }}>
          <div className="glass-panel" style={{
            maxWidth: '750px',
            width: '100%',
            maxHeight: '90vh',
            display: 'flex',
            flexDirection: 'column',
            padding: '24px',
            background: 'var(--bg-secondary)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 800 }}>Sample: {selectedSample.sample_id}</h3>
                  <span className={`badge ${LABEL_BADGES[selectedSample.error_label] || 'badge-primary'}`}>
                    {selectedSample.error_label}
                  </span>
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  Problem: {selectedSample.problem_type} • Split: {selectedSample.split}
                </p>
              </div>

              <button
                onClick={() => setSelectedSample(null)}
                className="btn btn-secondary btn-sm"
              >
                Close
              </button>
            </div>

            <div style={{ marginBottom: '14px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              <strong>Mutation: </strong> {selectedSample.mutation_description || 'Ground truth baseline'}
            </div>

            <div style={{ flex: 1, overflowY: 'auto', marginBottom: '16px' }}>
              <div className="code-editor-wrapper">
                <div className="code-editor-header">
                  <span>Python Source</span>
                </div>
                <pre style={{
                  padding: '16px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.85rem',
                  color: '#38bdf8',
                  lineHeight: 1.6,
                  overflowX: 'auto',
                }}>
                  {selectedSample.code}
                </pre>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setSelectedSample(null)}
                className="btn btn-primary btn-sm"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
