import React, { useState, useEffect } from 'react';
import { History, Trash2, Download, RefreshCw, GitCompare, Bug } from 'lucide-react';
import { fetchHistory, clearHistory } from '../api';

export default function HistoryTab({ refreshTrigger }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await fetchHistory(50);
      setHistory(data.entries || []);
    } catch (err) {
      console.error('Failed to load history', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [refreshTrigger]);

  const handleClearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all analysis history records?')) {
      return;
    }
    setClearing(true);
    try {
      await clearHistory();
      setHistory([]);
    } catch (err) {
      console.error('Failed to clear history', err);
    } finally {
      setClearing(false);
    }
  };

  const handleExportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `codeinsight_history_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Top Banner & Actions */}
      <div className="glass-panel" style={{ padding: '20px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 800 }}>Audit Trail & Prediction History</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            Persistent SQLite database log of code similarity and error diagnosis queries
          </p>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={handleExportJSON}
            disabled={history.length === 0}
            className="btn btn-secondary btn-sm"
          >
            <Download size={14} />
            Export JSON
          </button>
          
          <button
            onClick={handleClearHistory}
            disabled={history.length === 0 || clearing}
            className="btn btn-secondary btn-sm"
            style={{ color: '#fb7185' }}
          >
            <Trash2 size={14} />
            {clearing ? 'Clearing...' : 'Clear History'}
          </button>

          <button onClick={loadHistory} disabled={loading} className="btn btn-secondary btn-sm">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* History Table */}
      <div className="glass-panel" style={{ padding: '0', overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ background: 'rgba(255, 255, 255, 0.02)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '12px 18px', color: 'var(--text-muted)', fontWeight: 600 }}>Timestamp</th>
                <th style={{ padding: '12px 18px', color: 'var(--text-muted)', fontWeight: 600 }}>Type</th>
                <th style={{ padding: '12px 18px', color: 'var(--text-muted)', fontWeight: 600 }}>Model Used</th>
                <th style={{ padding: '12px 18px', color: 'var(--text-muted)', fontWeight: 600 }}>Result Summary</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="4" style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    Loading history...
                  </td>
                </tr>
              ) : history.length === 0 ? (
                <tr>
                  <td colSpan="4" style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                      <History size={32} color="var(--text-faint)" />
                      <p>No analysis queries logged yet.</p>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-faint)' }}>
                        Run tests in the Semantic Similarity or Error Classifier tabs to populate the audit log.
                      </span>
                    </div>
                  </td>
                </tr>
              ) : (
                history.map((item, idx) => (
                  <tr
                    key={item.id || idx}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
                      transition: 'background 0.15s ease',
                    }}
                  >
                    <td style={{ padding: '12px 18px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.775rem' }}>
                      {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}
                    </td>
                    <td style={{ padding: '12px 18px' }}>
                      <span className={`badge ${item.analysis_type === 'similarity' ? 'badge-cyan' : 'badge-primary'}`}>
                        {item.analysis_type === 'similarity' ? <GitCompare size={12} /> : <Bug size={12} />}
                        {item.analysis_type}
                      </span>
                    </td>
                    <td style={{ padding: '12px 18px', fontWeight: 600, color: '#ffffff', textTransform: 'capitalize' }}>
                      {item.model_used || 'CodeBERT'}
                    </td>
                    <td style={{ padding: '12px 18px', color: 'var(--text-main)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                      {item.result_summary}
                    </td>
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
