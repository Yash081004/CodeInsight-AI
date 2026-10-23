import React, { useMemo } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { oneDark } from '@codemirror/theme-one-dark';
import CopyButton from './CopyButton';

/** Code panels use warm charcoal surface regardless of app theme (spec). */
const editorTheme = oneDark;

export default function CodeEditor({
  value,
  onChange,
  label,
  placeholder = '',
  minHeight = '220px',
  id,
  footer,
}) {
  const lineCount = (value || '').split('\n').length;

  const extensions = useMemo(() => [python()], []);

  return (
    <div className="code-editor-wrapper" id={id}>
      <div className="code-editor-header">
        <span>{label}</span>
        <div className="code-editor-header-actions">
          {footer}
          <CopyButton text={value} label={`Copy ${label}`} size="sm" />
          <span className="code-editor-lines">{lineCount} lines</span>
        </div>
      </div>
      <CodeMirror
        value={value}
        height={minHeight}
        theme={editorTheme}
        extensions={extensions}
        onChange={(v) => onChange(v)}
        placeholder={placeholder}
        basicSetup={{
          lineNumbers: true,
          foldGutter: false,
          highlightActiveLine: true,
          bracketMatching: true,
        }}
        style={{ fontSize: '0.85rem' }}
      />
    </div>
  );
}
