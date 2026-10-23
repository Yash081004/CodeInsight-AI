import React, { useState, useCallback } from 'react';
import { Copy, Check } from 'lucide-react';

export default function CopyButton({ text, label = 'Copy', size = 'default' }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  }, [text]);

  const cls = size === 'sm' ? 'btn btn-secondary btn-sm copy-btn' : 'btn btn-secondary copy-btn';

  return (
    <button
      type="button"
      className={cls}
      onClick={handleCopy}
      disabled={!text}
      aria-label={label}
      title={copied ? 'Copied' : label}
    >
      {copied ? <Check size={14} aria-hidden /> : <Copy size={14} aria-hidden />}
      {size !== 'sm' && <span>{copied ? 'Copied' : label}</span>}
    </button>
  );
}
