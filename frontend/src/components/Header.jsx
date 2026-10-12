import React from 'react';
import { Cpu, Database, Server, RefreshCw, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';

export default function Header({ health, onRefreshHealth, isRefreshing }) {
  const isHealthy = health?.status === 'ok';

  return (
    <header className="glass-panel" style={{ padding: '16px 24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        
        {/* Brand & Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '46px',
            height: '46px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)'
          }}>
            <Sparkles size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em', background: 'linear-gradient(90deg, #ffffff, #c7d2fe)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                CodeInsight AI
              </h1>
              <span className="badge badge-primary">Deep Learning</span>
              <span className="badge badge-cyan">v1.0.0</span>
            </div>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Semantic Code Similarity & Logic Error Classification System
            </p>
          </div>
        </div>

        {/* System Health Indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          
          {/* Device pill */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8rem',
          }}>
            <Cpu size={14} color="var(--primary-light)" />
            <span style={{ color: 'var(--text-muted)' }}>Device:</span>
            <strong style={{ color: '#ffffff', textTransform: 'uppercase' }}>
              {health?.device || 'CPU'}
            </strong>
          </div>

          {/* CodeBERT Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8rem',
          }}>
            <Server size={14} color={health?.codebert_available ? 'var(--accent-emerald)' : 'var(--accent-amber)'} />
            <span style={{ color: 'var(--text-muted)' }}>CodeBERT:</span>
            <span style={{ color: health?.codebert_available ? '#34d399' : '#fbbf24', fontWeight: 600 }}>
              {health?.codebert_available ? 'Ready' : 'Loaded on demand'}
            </span>
          </div>

          {/* Classifier Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8rem',
          }}>
            <span className={`pulse-dot ${health?.classifier_available ? 'pulse-green' : (health?.baseline_available ? 'pulse-amber' : 'pulse-rose')}`} />
            <span style={{ color: 'var(--text-muted)' }}>Classifier:</span>
            <span style={{ color: '#ffffff', fontWeight: 600 }}>
              {health?.classifier_available ? 'CodeBERT (Finetuned)' : (health?.baseline_available ? 'TF-IDF Baseline' : 'Not Loaded')}
            </span>
          </div>

          {/* Dataset Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8rem',
          }}>
            <Database size={14} color={health?.dataset_available ? 'var(--accent-cyan)' : 'var(--accent-amber)'} />
            <span style={{ color: 'var(--text-muted)' }}>Dataset:</span>
            <span style={{ color: health?.dataset_available ? '#38bdf8' : '#fbbf24', fontWeight: 600 }}>
              {health?.dataset_available ? 'Available' : 'Missing'}
            </span>
          </div>

          {/* Refresh Button */}
          <button
            onClick={onRefreshHealth}
            disabled={isRefreshing}
            className="btn btn-secondary btn-sm"
            title="Refresh system status"
            style={{ padding: '8px 10px' }}
          >
            <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
          </button>

        </div>

      </div>
    </header>
  );
}
