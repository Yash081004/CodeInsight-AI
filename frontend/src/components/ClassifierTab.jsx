import React, { useState } from 'react';
import { Bug, Play, CheckCircle2, AlertTriangle, AlertCircle, Info, Lightbulb, Check, Sparkles } from 'lucide-react';
import { analyzeClassification } from '../api';

const CLASSIFICATION_PRESETS = [
  {
    name: '1. Correct Code',
    label: 'correct',
    desc: 'Functional algorithm passing all specifications',
    code: `def sum_positive_numbers(arr):
    total = 0
    for num in arr:
        if num > 0:
            total += num
    return total`,
  },
  {
    name: '2. Wrong Initialization Bug',
    label: 'wrong_initialization',
    desc: 'Accumulator starting at wrong identity value (0 instead of 1 for multiplication)',
    code: `def compute_factorial(n):
    result = 0  # BUG: Multiplying with 0 will always yield 0! Should be 1
    for i in range(1, n + 1):
        result *= i
    return result`,
  },
  {
    name: '3. Off-by-One Loop Bug',
    label: 'off_by_one_loop',
    desc: 'Upper boundary omitted from range iteration',
    code: `def sum_one_to_n(n):
    total = 0
    for i in range(1, n):  # BUG: range stops before n; misses the last element n
        total += i
    return total`,
  },
  {
    name: '4. Wrong Operator Bug',
    label: 'wrong_operator',
    desc: 'Inverted comparison operator in conditional',
    code: `def find_maximum(a, b):
    # BUG: using '<' returns the smaller element instead of maximum
    if a < b:
        return a
    return b`,
  },
  {
    name: '5. Wrong Condition Bug',
    label: 'wrong_condition',
    desc: 'Inverted modulo parity check',
    code: `def is_even_number(n):
    # BUG: n % 2 == 1 evaluates to True for odd numbers
    if n % 2 == 1:
        return True
    return False`,
  },
];

const LABEL_COLORS = {
  correct: { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', border: 'rgba(16, 185, 129, 0.3)' },
  wrong_initialization: { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24', border: 'rgba(245, 158, 11, 0.3)' },
  off_by_one_loop: { bg: 'rgba(244, 63, 94, 0.15)', text: '#fb7185', border: 'rgba(244, 63, 94, 0.3)' },
  wrong_operator: { bg: 'rgba(168, 85, 247, 0.15)', text: '#c084fc', border: 'rgba(168, 85, 247, 0.3)' },
  wrong_condition: { bg: 'rgba(6, 182, 212, 0.15)', text: '#38bdf8', border: 'rgba(6, 182, 212, 0.3)' },
};

export default function ClassifierTab({ onHistoryUpdate }) {
  const [code, setCode] = useState(CLASSIFICATION_PRESETS[1].code);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleApplyPreset = (preset) => {
    setCode(preset.code);
    setResult(null);
    setError(null);
  };

  const handleRunClassification = async () => {
    if (!code.trim()) {
      setError('Please provide Python source code to analyze.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeClassification(code);
      setResult(data);
      if (onHistoryUpdate) onHistoryUpdate();
    } catch (err) {
      setError(err.message || 'Classification request failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Preset Buttons */}
      <div className="glass-panel" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>
            Error Taxonomy Presets:
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)' }}>
            Select a standard code sample to test the model's diagnostic capability
          </span>
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {CLASSIFICATION_PRESETS.map((p, idx) => (
            <button
              key={idx}
              onClick={() => handleApplyPreset(p)}
              className="btn btn-secondary btn-sm"
              style={{ fontSize: '0.775rem' }}
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Code Editor */}
      <div className="code-editor-wrapper">
        <div className="code-editor-header">
          <span>Python Source Snippet</span>
          <span style={{ color: 'var(--text-faint)' }}>{code.split('\n').length} lines</span>
        </div>
        <textarea
          className="code-textarea"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          placeholder="# Paste Python function to inspect..."
          spellCheck={false}
          style={{ minHeight: '260px' }}
        />
      </div>

      {/* Action Bar */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button
          onClick={handleRunClassification}
          disabled={loading}
          className="btn btn-primary"
          style={{ minWidth: '220px', padding: '12px 28px' }}
        >
          {loading ? (
            <span>Running Deep Inference...</span>
          ) : (
            <>
              <Play size={16} fill="currentColor" />
              Classify Logic Error
            </>
          )}
        </button>
      </div>

      {/* Error notification */}
      {error && (
        <div className="glass-panel" style={{ padding: '16px 20px', borderLeft: '4px solid var(--accent-rose)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#fb7185' }}>
            <AlertCircle size={18} />
            <span style={{ fontSize: '0.875rem' }}>{error}</span>
          </div>
        </div>
      )}

      {/* Prediction Output */}
      {result && (
        <div className="glass-panel" style={{ padding: '28px', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
          
          {/* Header row */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 800 }}>Model Diagnosis</h3>
                
                {result.predicted_label && (
                  <span
                    className="badge"
                    style={{
                      fontSize: '0.85rem',
                      padding: '6px 14px',
                      background: LABEL_COLORS[result.predicted_label]?.bg || 'rgba(99, 102, 241, 0.15)',
                      color: LABEL_COLORS[result.predicted_label]?.text || '#ffffff',
                      border: `1px solid ${LABEL_COLORS[result.predicted_label]?.border || 'rgba(99, 102, 241, 0.3)'}`,
                    }}
                  >
                    {result.friendly_label || result.predicted_label}
                  </span>
                )}
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Diagnosed using: <strong style={{ color: '#ffffff' }}>{result.model_type}</strong>
              </p>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <span className={`badge ${result.is_valid_python ? 'badge-emerald' : 'badge-rose'}`}>
                AST Syntax: {result.is_valid_python ? 'Valid Python' : 'Syntax Error'}
              </span>
            </div>
          </div>

          {/* Probabilities Distribution */}
          <div style={{ marginBottom: '28px' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '14px', color: 'var(--text-muted)' }}>
              Class Probability Distribution
            </h4>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {Object.entries(result.confidence_scores || {}).map(([labelKey, score]) => {
                const isSelected = labelKey === result.predicted_label;
                const percent = (score * 100).toFixed(1);
                const colors = LABEL_COLORS[labelKey] || { text: '#ffffff' };

                return (
                  <div key={labelKey} style={{
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-sm)',
                    background: isSelected ? 'rgba(255, 255, 255, 0.04)' : 'rgba(255, 255, 255, 0.01)',
                    border: isSelected ? '1px solid rgba(99, 102, 241, 0.3)' : '1px solid transparent',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', fontSize: '0.85rem' }}>
                      <span style={{ fontWeight: isSelected ? 700 : 500, color: isSelected ? colors.text : 'var(--text-muted)' }}>
                        {labelKey} {isSelected && ' (Predicted)'}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: isSelected ? colors.text : 'var(--text-faint)' }}>
                        {percent}%
                      </span>
                    </div>

                    <div style={{ width: '100%', height: '6px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${percent}%`,
                        height: '100%',
                        background: isSelected ? 'linear-gradient(90deg, #6366f1, #a855f7)' : 'rgba(255, 255, 255, 0.2)',
                        transition: 'width 0.5s ease',
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Educational Feedback Card */}
          <div style={{
            padding: '20px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(99, 102, 241, 0.08)',
            border: '1px solid rgba(99, 102, 241, 0.25)',
            marginBottom: '20px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <Lightbulb size={18} color="var(--primary-light)" />
              <strong style={{ fontSize: '0.95rem', color: '#ffffff' }}>Pedagogical Feedback Guide</strong>
            </div>
            <p style={{ fontSize: '0.875rem', color: '#e0e7ff', lineHeight: 1.6 }}>
              {result.feedback}
            </p>
          </div>

          {/* Academic Note */}
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', fontSize: '0.775rem', color: 'var(--text-faint)', borderTop: '1px solid var(--border-subtle)', paddingTop: '16px' }}>
            <Info size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
            <span>
              <strong>Pedagogical Guarantee:</strong> Feedback messages are generated via deterministic, rule-verified pedagogical templates rather than uncontrolled generative LLMs, ensuring 100% reproducible advice in academic evaluation settings.
            </span>
          </div>

        </div>
      )}

    </div>
  );
}
