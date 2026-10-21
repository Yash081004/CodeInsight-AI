import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import Navigation from './components/Navigation';
import ErrorBoundary from './components/ErrorBoundary';
import OverviewTab from './components/OverviewTab';
import SimilarityTab from './components/SimilarityTab';
import ClassifierTab from './components/ClassifierTab';
import EvaluationTab from './components/EvaluationTab';
import DatasetTab from './components/DatasetTab';
import HistoryTab from './components/HistoryTab';
import { fetchHealth, fetchHistory } from './api';
import { tabFromHash, hashForTab } from './routes';

export default function App() {
  const [activeTab, setActiveTab] = useState(() => tabFromHash());
  const [health, setHealth] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [theme, setTheme] = useState(() => localStorage.getItem('codeinsight-theme') || 'dark');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [tabResetKey, setTabResetKey] = useState(0);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('codeinsight-theme', theme);
  }, [theme]);

  const selectTab = useCallback((tabId) => {
    setActiveTab(tabId);
    setMobileNavOpen(false);
    const hash = hashForTab(tabId);
    if (window.location.hash !== hash) {
      window.history.pushState(null, '', hash);
    }
  }, []);

  useEffect(() => {
    const onHash = () => setActiveTab(tabFromHash());
    window.addEventListener('hashchange', onHash);
    window.addEventListener('popstate', onHash);
    if (!window.location.hash) {
      window.history.replaceState(null, '', hashForTab(activeTab));
    }
    return () => {
      window.removeEventListener('hashchange', onHash);
      window.removeEventListener('popstate', onHash);
    };
  }, []);

  const refreshHealth = async () => {
    setIsRefreshing(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch (err) {
      console.warn('Backend not responding:', err);
      setHealth((h) => h || { status: 'unavailable' });
    } finally {
      setIsRefreshing(false);
    }
  };

  const refreshHistoryCount = async () => {
    try {
      const data = await fetchHistory(100);
      setHistoryCount(data.total || data.entries?.length || 0);
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    refreshHealth();
    refreshHistoryCount();
  }, []);

  const handleHistoryUpdate = () => {
    refreshHistoryCount();
    setHistoryRefreshKey((k) => k + 1);
  };

  const boundaryRetry = () => setTabResetKey((k) => k + 1);

  const renderTab = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab onNavigateTab={selectTab} health={health} />;
      case 'similarity':
        return <SimilarityTab onHistoryUpdate={handleHistoryUpdate} />;
      case 'classifier':
        return <ClassifierTab onHistoryUpdate={handleHistoryUpdate} />;
      case 'evaluation':
        return <EvaluationTab />;
      case 'dataset':
        return <DatasetTab />;
      case 'history':
        return <HistoryTab refreshTrigger={historyRefreshKey} />;
      default:
        return <OverviewTab onNavigateTab={selectTab} health={health} />;
    }
  };

  return (
    <div className="app-shell">
      <Navigation
        activeTab={activeTab}
        onSelectTab={selectTab}
        historyCount={historyCount}
        mobileOpen={mobileNavOpen}
        onCloseMobile={() => setMobileNavOpen(false)}
      />

      <div className="app-main">
        <Header
          activeTab={activeTab}
          health={health}
          onRefreshHealth={refreshHealth}
          isRefreshing={isRefreshing}
          theme={theme}
          onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
          onOpenMobileNav={() => setMobileNavOpen(true)}
        />

        <div className="app-content">
          <main className="workspace" id="main-content">
            <ErrorBoundary key={`${activeTab}-${tabResetKey}`} onRetry={boundaryRetry}>
              {renderTab()}
            </ErrorBoundary>
          </main>

          <footer className="app-footer">
            <span>Artifact-backed metrics · local history</span>
          </footer>
        </div>
      </div>
    </div>
  );
}
