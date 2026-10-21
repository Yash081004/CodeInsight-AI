import React, { useState, useEffect } from 'react';
import { History, Trash2, Download, RefreshCw, GitCompare, Bug } from 'lucide-react';
import { fetchHistory, clearHistory } from '../api';
import PageHeader from './ui/PageHeader';
import AlertBanner from './ui/AlertBanner';

export default function HistoryTab({ refreshTrigger }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [loadError, setLoadError] = useState(null);

  const loadHistory = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const data = await fetchHistory(50);
      setHistory(data.entries || []);
    } catch (err) {
      setLoadError(err.message || 'Could not load history.');
      setHistory([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [refreshTrigger]);

  const handleClearHistory = async () => {
    if (!window.confirm('Clear all analysis records on this server?')) return;
    setClearing(true);
    try {
      await clearHistory();
      setHistory([]);
    } catch (err) {
      setLoadError(err.message || 'Clear failed.');
    } finally {
      setClearing(false);
    }
  };

  const handleExportJSON = () => {
    const dataStr = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(history, null, 2))}`;
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `codeinsight_history_${Date.now()}.json`;
    a.click();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <PageHeader
        title="Session history"
        subtitle="Recent similarity and classification requests logged in the local server SQLite store—not a cross-device account history."
        actions={
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <button type="button" onClick={handleExportJSON} disabled={history.length === 0} className="btn btn-secondary btn-sm">
              <Download size={14} aria-hidden /> Export JSON
            </button>
            <button
              type="button"
              onClick={handleClearHistory}
              disabled={history.length === 0 || clearing}
              className="btn btn-secondary btn-sm"
              style={{ color: 'var(--accent-rose)' }}
            >
              <Trash2 size={14} aria-hidden /> {clearing ? 'Clearing…' : 'Clear'}
            </button>
            <button type="button" onClick={loadHistory} disabled={loading} className="btn btn-secondary btn-sm" aria-label="Refresh">
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
        }
      />

      {loadError && <AlertBanner variant="error">{loadError}</AlertBanner>}

      <div className="glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="history-table">
            <thead>
              <tr>
                <th scope="col">Time</th>
                <th scope="col">Type</th>
                <th scope="col">Model</th>
                <th scope="col">Summary</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={4} className="history-empty">Loading…</td>
                </tr>
              ) : history.length === 0 ? (
                <tr>
                  <td colSpan={4} className="history-empty">
                    <History size={32} color="var(--text-faint)" aria-hidden />
                    <p>No analyses yet.</p>
                    <span>Run similarity or classification to populate this log.</span>
                  </td>
                </tr>
              ) : (
                history.map((item, idx) => (
                  <tr key={item.id || idx}>
                    <td className="history-mono">
                      {item.timestamp ? new Date(item.timestamp).toLocaleString() : '—'}
                    </td>
                    <td>
                      <span className={`badge ${item.analysis_type === 'similarity' ? 'badge-cyan' : 'badge-primary'}`}>
                        {item.analysis_type === 'similarity' ? <GitCompare size={12} aria-hidden /> : <Bug size={12} aria-hidden />}
                        {item.analysis_type}
                      </span>
                    </td>
                    <td className="text-strong" style={{ textTransform: 'capitalize', fontWeight: 600 }}>
                      {item.model_used || '—'}
                    </td>
                    <td className="history-mono history-summary">{item.result_summary}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
