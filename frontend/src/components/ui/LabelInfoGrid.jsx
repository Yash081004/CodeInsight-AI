import React from 'react';
import { ERROR_LABELS } from '../../constants/labels';

const BADGE_MAP = {
  correct: 'badge-emerald',
  wrong_initialization: 'badge-amber',
  off_by_one_loop: 'badge-rose',
  wrong_operator: 'badge-cyan',
  wrong_condition: 'badge-primary',
};

export default function LabelInfoGrid({ compact = false }) {
  return (
    <div className={`label-info-grid ${compact ? 'label-info-grid--compact' : ''}`}>
      {ERROR_LABELS.map((label) => (
        <article key={label.id} className="label-info-card glass-panel">
          <span className={`badge ${BADGE_MAP[label.id] || 'badge-primary'}`}>{label.short}</span>
          <h4 className="label-info-name">{label.name}</h4>
          <p className="label-info-desc" title={label.description}>
            {label.description}
          </p>
        </article>
      ))}
    </div>
  );
}
