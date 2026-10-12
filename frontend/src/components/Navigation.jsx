import React from 'react';
import { BookOpen, GitCompare, Bug, BarChart3, Database, History } from 'lucide-react';

export default function Navigation({ activeTab, onSelectTab, historyCount = 0 }) {
  const tabs = [
    { id: 'overview', label: 'Architecture & Overview', icon: BookOpen },
    { id: 'similarity', label: 'Semantic Similarity', icon: GitCompare, badge: 'CodeBERT' },
    { id: 'classifier', label: 'Logic Error Classifier', icon: Bug, badge: '5 Classes' },
    { id: 'evaluation', label: 'Evaluation & Metrics', icon: BarChart3 },
    { id: 'dataset', label: 'Dataset Explorer', icon: Database },
    { id: 'history', label: 'History & Audit', icon: History, count: historyCount },
  ];

  return (
    <nav style={{
      display: 'flex',
      gap: '8px',
      overflowX: 'auto',
      paddingBottom: '6px',
      marginBottom: '24px',
      borderBottom: '1px solid var(--border-subtle)',
    }}>
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onSelectTab(tab.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 18px',
              borderRadius: 'var(--radius-md) var(--radius-md) 0 0',
              background: isActive ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
              border: 'none',
              borderBottom: isActive ? '2px solid var(--primary)' : '2px solid transparent',
              color: isActive ? '#ffffff' : 'var(--text-muted)',
              fontSize: '0.875rem',
              fontWeight: isActive ? 600 : 500,
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              transition: 'all 0.2s ease',
            }}
          >
            <Icon size={16} color={isActive ? 'var(--primary-light)' : 'var(--text-faint)'} />
            <span>{tab.label}</span>
            {tab.badge && (
              <span style={{
                fontSize: '0.65rem',
                padding: '2px 6px',
                borderRadius: '6px',
                background: isActive ? 'rgba(99, 102, 241, 0.3)' : 'rgba(255, 255, 255, 0.05)',
                color: isActive ? 'var(--primary-light)' : 'var(--text-faint)',
                fontWeight: 700,
              }}>
                {tab.badge}
              </span>
            )}
            {typeof tab.count === 'number' && tab.count > 0 && (
              <span style={{
                fontSize: '0.7rem',
                padding: '1px 6px',
                borderRadius: '999px',
                background: 'rgba(255, 255, 255, 0.1)',
                color: 'var(--text-muted)',
              }}>
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
