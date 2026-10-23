import React from 'react';

export default function PageHeader({ title, subtitle, actions, kbdHint }) {
  return (
    <header className="page-header">
      <div className="page-header-text">
        <h2 className="page-title">{title}</h2>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
        {kbdHint && (
          <p className="page-kbd-hint" aria-live="polite">
            {kbdHint}
          </p>
        )}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </header>
  );
}
