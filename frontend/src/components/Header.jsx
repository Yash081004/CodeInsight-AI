import React from 'react';
import { RefreshCw, Sun, Moon, Menu } from 'lucide-react';
import { TAB_PAGE_TITLES } from '../routes';

export default function Header({
  activeTab,
  health,
  onRefreshHealth,
  isRefreshing,
  theme,
  onToggleTheme,
  onOpenMobileNav,
}) {
  return (
    <header className="topbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button
          type="button"
          className="btn btn-secondary btn-sm mobile-menu-btn"
          onClick={onOpenMobileNav}
          aria-label="Open navigation"
        >
          <Menu size={16} />
        </button>
        <div>
          <h2 className="topbar-title">{TAB_PAGE_TITLES[activeTab] || 'CodeInsight AI'}</h2>
          <p className="topbar-meta">
            {health?.status === 'ok' ? 'Backend connected' : health ? 'Backend degraded' : 'Checking backend…'}
          </p>
        </div>
      </div>

      <div className="topbar-actions">
        <div className="status-pill" title="Backend status">
          <span
            className={`pulse-dot ${health?.status === 'ok' ? 'pulse-green' : health ? 'pulse-amber' : 'pulse-rose'}`}
            aria-hidden
          />
          <span className="status-pill-value">{health?.device || 'offline'}</span>
        </div>
        <button
          type="button"
          onClick={onToggleTheme}
          className="btn btn-secondary btn-sm"
          aria-label={theme === 'dark' ? 'Light theme' : 'Dark theme'}
        >
          {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
        </button>
        <button
          type="button"
          onClick={onRefreshHealth}
          disabled={isRefreshing}
          className="btn btn-secondary btn-sm"
          aria-label="Refresh status"
        >
          <RefreshCw size={14} className={isRefreshing ? 'animate-spin' : ''} />
        </button>
      </div>
    </header>
  );
}
