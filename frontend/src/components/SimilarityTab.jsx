import React, { useState } from 'react';
import { GitCompare, Sliders, Play, CheckCircle2, XCircle, Info, Sparkles, Copy, Check } from 'lucide-react';
import { analyzeSimilarity } from '../api';

const PRESETS = [
  {
    name: 'Variable Renaming (Semantic Equivalent)',
    desc: 'Identical logic, different identifier names. Tests semantic invariance to renaming.',
    codeA: `def calculate_sum(numbers_list):
    total_value = 0
    for number in numbers_list:
        total_value += number
    return total_value`,
    codeB: `def calculate_sum(arr):
    s = 0
    for x in arr:
        s += x
    return s`,
  },
  {
    name: 'Loop vs List Comprehension',
    desc: 'Different syntactic structure, identical functional behavior.',
    codeA: `def filter_evens(nums):
    result = []
    for n in nums:
        if n % 2 == 0:
            result.append(n)
    return result`,
    codeB: `def filter_evens(nums):
    return [n for n in nums if n % 2 == 0]`,
  },
  {
    name: 'Recursive vs Iterative Factorial',
    desc: 'Completely different execution paradigms solving the same problem.',
    codeA: `def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)`,
    codeB: `def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result`,
  },
  {
    name: 'Unrelated Algorithms (Bubble Sort vs Binary Search)',
    desc: 'Two completely distinct algorithms with negligible semantic overlap.',
    codeA: `def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr`,
    codeB: `def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1`,
  },
];

export default function SimilarityTab({ onHistoryUpdate }) {
  const [codeA, setCodeA] = useState(PRESETS[0].codeA);
  const [codeB, setCodeB] = useState(PRESETS[0].codeB);
  const [threshold, setThreshold] = useState(0.80);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleApplyPreset = (preset) => {
    setCodeA(preset.codeA);
    setCodeB(preset.codeB);
    setResult(null);
    setError(null);
  };

  const handleRunAnalysis = async () => {
    if (!codeA.trim() || !codeB.trim()) {
      setError('Please provide code in both Snippet A and Snippet B.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeSimilarity(codeA, codeB, threshold);
      setResult(data);
      if (onHistoryUpdate) onHistoryUpdate();
    } catch (err) {
      setError(err.message || 'Similarity analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Preset Selector */}
      <div className="glass-panel" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>
            Quick Comparison Presets:
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)' }}>
            Click to load standard benchmark comparison pairs
          </span>
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {PRESETS.map((p, idx) => (
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

      {/* Code Editors Side-by-Side */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px' }}>
        
        {/* Editor A */}
        <div className="code-editor-wrapper">
          <div className="code-editor-header">
            <span>Snippet A (Python)</span>
            <span style={{ color: 'var(--text-faint)' }}>{codeA.split('\n').length} lines</span>
          </div>
          <textarea
            className="code-textarea"
            value={codeA}
            onChange={(e) => setCodeA(e.target.value)}
            placeholder="# Enter Python function A..."
            spellCheck={false}
          />
        </div>

        {/* Editor B */}
        <div className="code-editor-wrapper">
          <div className="code-editor-header">
            <span>Snippet B (Python)</span>
            <span style={{ color: 'var(--text-faint)' }}>{codeB.split('\n').length} lines</span>
          </div>
          <textarea
            className="code-textarea"
            value={codeB}
            onChange={(e) => setCodeB(e.target.value)}
            placeholder="# Enter Python function B..."
            spellCheck={false}
          />
        </div>

      </div>

      {/* Controls Bar: Threshold Slider & Action */}
      <div className="glass-panel" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', minWidth: '320px' }}>
          <Sliders size={18} color="var(--primary-light)" />
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Similarity Decision Threshold:</span>
              <strong style={{ color: '#ffffff', fontFamily: 'var(--font-mono)' }}>{threshold.toFixed(2)}</strong>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.99"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--primary)', cursor: 'pointer' }}
            />
          </div>
        </div>

        <button
          onClick={handleRunAnalysis}
          disabled={loading}
          className="btn btn-primary"
          style={{ minWidth: '180px', padding: '12px 24px' }}
        >
          {loading ? (
            <span>Computing Embeddings...</span>
          ) : (
            <>
              <Play size={16} fill="currentColor" />
              Analyze Similarity
            </>
          )}
        </button>

      </div>

      {/* Error Message */}
      {error && (
        <div className="glass-panel" style={{ padding: '16px 20px', borderLeft: '4px solid var(--accent-rose)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#fb7185' }}>
            <XCircle size={18} />
            <span style={{ fontSize: '0.875rem' }}>{error}</span>
          </div>
        </div>
      )}

      {/* Analysis Results View */}
      {result && (
        <div className="glass-panel" style={{ padding: '28px', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 800 }}>Similarity Verdict</h3>
                {result.is_similar ? (
                  <span className="badge badge-emerald" style={{ fontSize: '0.85rem', padding: '6px 14px' }}>
                    <CheckCircle2 size={14} /> Semantically Equivalent (Score ≥ {result.threshold})
                  </span>
                ) : (
                  <span className="badge badge-rose" style={{ fontSize: '0.85rem', padding: '6px 14px' }}>
                    <XCircle size={14} /> Dissimilar (Score &lt; {result.threshold})
                  </span>
                )}
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Evaluated with decision model: <strong style={{ color: '#ffffff', textTransform: 'uppercase' }}>{result.decision_model}</strong>
              </p>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <span className={`badge ${result.code_a_valid ? 'badge-emerald' : 'badge-rose'}`}>
                Snippet A AST: {result.code_a_valid ? 'Valid' : 'Syntax Error'}
              </span>
              <span className={`badge ${result.code_b_valid ? 'badge-emerald' : 'badge-rose'}`}>
                Snippet B AST: {result.code_b_valid ? 'Valid' : 'Syntax Error'}
              </span>
            </div>
          </div>

          {/* Comparative Metrics Bars */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '24px' }}>
            
            {/* CodeBERT Card */}
            <div style={{ padding: '20px', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sparkles size={16} color="var(--primary-light)" />
                  <strong style={{ fontSize: '0.95rem' }}>CodeBERT Cosine Similarity</strong>
                </div>
                <span style={{ fontSize: '1.25rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: result.codebert_score !== null ? 'var(--primary-light)' : 'var(--text-faint)' }}>
                  {result.codebert_score !== null ? result.codebert_score.toFixed(4) : 'N/A'}
                </span>
              </div>
              
              {/* Progress bar */}
              <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden', marginBottom: '10px' }}>
                <div style={{
                  width: `${Math.max(0, Math.min(100, (result.codebert_score || 0) * 100))}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #6366f1, #a855f7)',
                  transition: 'width 0.6s ease',
                }} />
              </div>
              <p style={{ fontSize: '0.775rem', color: 'var(--text-muted)' }}>
                Dense 768-d contextual embeddings capturing syntax structure and token dependencies.
              </p>
            </div>

            {/* TF-IDF Baseline Card */}
            <div style={{ padding: '20px', borderRadius: 'var(--radius-md)', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <GitCompare size={16} color="var(--accent-cyan)" />
                  <strong style={{ fontSize: '0.95rem' }}>TF-IDF Baseline Similarity</strong>
                </div>
                <span style={{ fontSize: '1.25rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: result.tfidf_score !== null ? 'var(--accent-cyan)' : 'var(--text-faint)' }}>
                  {result.tfidf_score !== null ? result.tfidf_score.toFixed(4) : 'N/A'}
                </span>
              </div>
              
              {/* Progress bar */}
              <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden', marginBottom: '10px' }}>
                <div style={{
                  width: `${Math.max(0, Math.min(100, (result.tfidf_score || 0) * 100))}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #06b6d4, #10b981)',
                  transition: 'width 0.6s ease',
                }} />
              </div>
              <p style={{ fontSize: '0.775rem', color: 'var(--text-muted)' }}>
                Lexical n-gram frequency baseline. Susceptible to identifier and variable renamings.
              </p>
            </div>

          </div>

          {/* Explanation Box */}
          <div style={{ padding: '16px 20px', borderRadius: 'var(--radius-md)', background: 'rgba(99, 102, 241, 0.06)', border: '1px solid rgba(99, 102, 241, 0.2)', marginBottom: '18px' }}>
            <p style={{ fontSize: '0.875rem', color: '#e0e7ff', lineHeight: 1.6 }}>
              {result.explanation}
            </p>
          </div>

          {/* Academic Disclaimer Callout */}
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', fontSize: '0.775rem', color: 'var(--text-faint)', borderTop: '1px solid var(--border-subtle)', paddingTop: '16px' }}>
            <Info size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
            <span>
              <strong>Scientific Validity Note:</strong> Code similarity computed via embedding cosine proximity assesses representation alignment in latent space. 
              It does NOT guarantee mathematical functional equivalence across all arbitrary inputs (Rice's Theorem).
            </span>
          </div>

        </div>
      )}

    </div>
  );
}
