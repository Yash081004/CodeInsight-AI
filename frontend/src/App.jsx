import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Navigation from './components/Navigation';
import OverviewTab from './components/OverviewTab';
import SimilarityTab from './components/SimilarityTab';
import ClassifierTab from './components/ClassifierTab';
import EvaluationTab from './components/EvaluationTab';
import DatasetTab from './components/DatasetTab';
import HistoryTab from './components/HistoryTab';
import { fetchHealth, fetchHistory } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [health, setHealth] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  const refreshHealth = async () => {
    setIsRefreshing(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch (err) {
      console.warn('Backend not responding yet:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const refreshHistoryCount = async () => {
    try {
      const data = await fetchHistory(100);
      setHistoryCount(data.total || data.entries?.length || 0);
    } catch (err) {
      // ignore
    }
  };

  useEffect(() => {
    refreshHealth();
    refreshHistoryCount();
  }, []);

  const handleHistoryUpdate = () => {
    refreshHistoryCount();
    setHistoryRefreshKey(k => k + 1);
  };

  return (
    <div className="app-container">
      {/* System Status Header */}
      <Header
        health={health}
        onRefreshHealth={refreshHealth}
        isRefreshing={isRefreshing}
      />

      {/* Navigation Bar */}
      <Navigation
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        historyCount={historyCount}
      />

      {/* Main Tab Content */}
      <main>
        {activeTab === 'overview' && (
          <OverviewTab onNavigateTab={setActiveTab} />
        )}

        {activeTab === 'similarity' && (
          <SimilarityTab onHistoryUpdate={handleHistoryUpdate} />
        )}

        {activeTab === 'classifier' && (
          <ClassifierTab onHistoryUpdate={handleHistoryUpdate} />
        )}

        {activeTab === 'evaluation' && (
          <EvaluationTab />
        )}

        {activeTab === 'dataset' && (
          <DatasetTab />
        )}

        {activeTab === 'history' && (
          <HistoryTab refreshTrigger={historyRefreshKey} />
        )}
      </main>

      {/* University & Academic Project Footer */}
      <footer style={{
        marginTop: '60px',
        paddingTop: '24px',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        fontSize: '0.8rem',
        color: 'var(--text-faint)',
      }}>
        <div>
          <strong style={{ color: 'var(--text-muted)' }}>CodeInsight AI</strong> • Deep Learning Subject Demonstration & Academic Project
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <span>Transformer NLP for Code</span>
          <span>•</span>
          <span>CodeBERT (RoBERTa Architecture)</span>
          <span>•</span>
          <span>AST Mutation Engine</span>
        </div>
      </footer>
    </div>
  );
}
