import React from 'react';

import { BookOpen, GitCompare, Bug, BarChart3, Database, History } from 'lucide-react';

import { PRIMARY_TABS, SECONDARY_TABS } from '../routes';



const ICONS = {

  overview: BookOpen,

  similarity: GitCompare,

  classifier: Bug,

  evaluation: BarChart3,

  dataset: Database,

  history: History,

};



function NavGroup({ tabs, activeTab, onSelectTab, historyCount }) {

  return tabs.map((tab) => {

    const Icon = ICONS[tab.id];

    const isActive = activeTab === tab.id;

    const count = tab.id === 'history' ? historyCount : null;

    return (

      <button

        key={tab.id}

        type="button"

        role="tab"

        aria-selected={isActive}

        className={`nav-item ${isActive ? 'nav-item--active' : ''}`}

        onClick={() => onSelectTab(tab.id)}

      >

        <Icon size={18} aria-hidden strokeWidth={isActive ? 2.25 : 2} />

        <span>{tab.label}</span>

        {typeof count === 'number' && count > 0 && (

          <span className="nav-item-badge">{count}</span>

        )}

      </button>

    );

  });

}



export default function Navigation({ activeTab, onSelectTab, historyCount = 0, mobileOpen, onCloseMobile }) {

  return (

    <>

      {mobileOpen && (

        <button

          type="button"

          className="sidebar-overlay"

          aria-label="Close menu"

          onClick={onCloseMobile}

        />

      )}

      <aside className={`sidebar ${mobileOpen ? 'sidebar--open' : ''}`} aria-label="Primary">

        <div className="sidebar-brand">

          <h1>CodeInsight</h1>

          <span className="meta-label">Midnight Atelier</span>

        </div>

        <nav className="sidebar-nav" role="tablist">

          <NavGroup tabs={PRIMARY_TABS} activeTab={activeTab} onSelectTab={onSelectTab} historyCount={historyCount} />

        </nav>

        <div className="sidebar-nav-secondary">

          <span className="sidebar-section-label">More</span>

          <nav className="sidebar-nav" role="tablist">

            <NavGroup tabs={SECONDARY_TABS} activeTab={activeTab} onSelectTab={onSelectTab} historyCount={historyCount} />

          </nav>

        </div>

      </aside>

    </>

  );

}

