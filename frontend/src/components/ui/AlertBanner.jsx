import React from 'react';
import { AlertCircle, Info, AlertTriangle } from 'lucide-react';

const ICONS = {
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

export default function AlertBanner({ variant = 'info', title, children, ...rest }) {
  const Icon = ICONS[variant] || Info;
  return (
    <div className={`alert-banner alert-banner--${variant}`} role={variant === 'error' ? 'alert' : 'status'} {...rest}>
      <Icon size={18} aria-hidden className="alert-banner-icon" />
      <div>
        {title && <strong className="alert-banner-title">{title}</strong>}
        <div className="alert-banner-body">{children}</div>
      </div>
    </div>
  );
}
