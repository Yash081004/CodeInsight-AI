import React from 'react';
import { Layers, Brain, GitCompare, ShieldCheck, HelpCircle, ArrowRight, Zap, Target, BookOpen } from 'lucide-react';

export default function OverviewTab({ onNavigateTab }) {
  const errorCategories = [
    {
      id: 'correct',
      name: 'Correct Code',
      badge: 'badge-emerald',
      desc: 'Ground truth program passing all functional and boundary unit tests.',
      example: 'def sum_n(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total'
    },
    {
      id: 'wrong_initialization',
      name: 'Wrong Initialization',
      badge: 'badge-amber',
      desc: 'Variable or accumulator initialized to an erroneous starting value (e.g. 1 instead of 0, or 0 instead of 1 for product).',
      example: 'def product(lst):\n    res = 0  # <--- Bug: should be 1\n    for x in lst: res *= x\n    return res'
    },
    {
      id: 'off_by_one_loop',
      name: 'Off-by-One Loop',
      badge: 'badge-rose',
      desc: 'Loop range omitted boundary index or iterated one element too far (e.g., range(1, n) instead of range(1, n + 1)).',
      example: 'def sum_to_n(n):\n    s = 0\n    for i in range(n):  # <--- Bug: misses n\n        s += i\n    return s'
    },
    {
      id: 'wrong_operator',
      name: 'Wrong Operator',
      badge: 'badge-amber',
      desc: 'Inverted or swapped arithmetic/comparison operator (e.g., using < instead of > in maximum search, or + instead of -).',
      example: 'def find_max(a, b):\n    if a < b:  # <--- Bug: inverted logic\n        return a\n    return b'
    },
    {
      id: 'wrong_condition',
      name: 'Wrong Condition',
      badge: 'badge-rose',
      desc: 'Flawed boolean predicate, missing edge case check, or inverted branch decision.',
      example: 'def is_even(n):\n    if n % 2 == 1: # <--- Bug: returning True for odd\n        return True\n    return False'
    },
  ];

  const vivaQuestions = [
    {
      q: "Why is CodeBERT preferred over traditional AST or TF-IDF?",
      a: "TF-IDF treats code purely as a bag-of-words and is blind to syntax order and semantic flow. CodeBERT is a multi-layer transformer pre-trained on programming and natural language (CodeSearchNet) that captures bimodal representations and deep contextual code semantics."
    },
    {
      q: "How does CodeBERT compute code similarity?",
      a: "Each code snippet is tokenized and passed through CodeBERT. We perform mean-pooling across non-padding token embeddings from the final hidden layer to generate a dense 768-dimensional embedding vector, and then compute cosine similarity between the two vectors."
    },
    {
      q: "Why is semantic similarity cosine score NOT a calibrated probability?",
      a: "Cosine similarity ranges between [-1, 1] based on high-dimensional directional alignment in embedding space. It reflects semantic proximity in representation space, not the mathematical probability that two functions produce identical I/O for all inputs."
    },
    {
      q: "How was the dataset generated and validated?",
      a: "We developed a programmatic mutation engine on 21 base algorithmic problem templates. Each mutation injects precisely one error category and is strictly validated via Python AST parsing and tested against functional test suites to ensure non-equivalence."
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      
      {/* Hero Banner */}
      <div className="glass-panel" style={{
        padding: '36px',
        background: 'linear-gradient(135deg, rgba(21, 27, 45, 0.9) 0%, rgba(30, 41, 75, 0.5) 100%)',
        border: '1px solid rgba(99, 102, 241, 0.25)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute',
          top: '-60px',
          right: '-40px',
          width: '260px',
          height: '260px',
          background: 'radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        <div style={{ maxWidth: '850px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <span className="badge badge-primary">Deep Learning Research Project</span>
            <span className="badge badge-cyan">Transformer NLP for Source Code</span>
          </div>
          
          <h2 style={{ fontSize: '2.1rem', fontWeight: 800, lineHeight: 1.25, letterSpacing: '-0.02em', marginBottom: '16px' }}>
            Semantic Code Similarity & Logic Error Classification
          </h2>
          
          <p style={{ fontSize: '1rem', color: 'var(--text-muted)', lineHeight: 1.7, marginBottom: '24px' }}>
            CodeInsight AI investigates the effectiveness of transformer-based representation learning (<strong style={{ color: '#ffffff' }}>microsoft/codebert-base</strong>) 
            versus traditional frequency baselines (<strong style={{ color: '#ffffff' }}>TF-IDF + Logistic Regression</strong>) in detecting semantic code equivalence 
            and diagnosing 5 fundamental categories of beginner algorithmic logic bugs.
          </p>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button
              onClick={() => onNavigateTab('similarity')}
              className="btn btn-primary"
            >
              <GitCompare size={16} />
              Try Code Similarity
              <ArrowRight size={14} />
            </button>
            <button
              onClick={() => onNavigateTab('classifier')}
              className="btn btn-secondary"
            >
              <Brain size={16} />
              Test Logic Error Classifier
            </button>
            <button
              onClick={() => onNavigateTab('evaluation')}
              className="btn btn-secondary"
            >
              <Layers size={16} />
              View Benchmark Metrics
            </button>
          </div>
        </div>
      </div>

      {/* Architecture Pipeline Cards */}
      <div>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers size={20} color="var(--primary-light)" />
          System Architecture & Workflow
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          
          <div className="glass-panel glow-card" style={{ padding: '24px' }}>
            <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '14px' }}>
              <Zap size={20} color="var(--primary-light)" />
            </div>
            <h4 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '8px' }}>1. AST & Preprocessing</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Source code snippets are validated using Python's native <code style={{ color: 'var(--accent-cyan)' }}>ast.parse</code>. 
              Whitespace, token limits, and comment normalization are strictly performed.
            </p>
          </div>

          <div className="glass-panel glow-card" style={{ padding: '24px' }}>
            <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '14px' }}>
              <Brain size={20} color="var(--accent-cyan)" />
            </div>
            <h4 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '8px' }}>2. CodeBERT Encoding</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Byte-pair encoding tokenization generates attention masks and input IDs. Mean pooling over non-padding hidden states produces 768-d contextual embeddings.
            </p>
          </div>

          <div className="glass-panel glow-card" style={{ padding: '24px' }}>
            <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '14px' }}>
              <GitCompare size={20} color="var(--accent-emerald)" />
            </div>
            <h4 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '8px' }}>3. Dual Similarity Engine</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Calculates CodeBERT cosine similarity alongside the fitted TF-IDF baseline. A configurable threshold compares semantic versus lexical alignments.
            </p>
          </div>

          <div className="glass-panel glow-card" style={{ padding: '24px' }}>
            <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: 'rgba(168, 85, 247, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '14px' }}>
              <ShieldCheck size={20} color="var(--accent-purple)" />
            </div>
            <h4 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '8px' }}>4. Logic Error Diagnosis</h4>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Fine-tuned sequence classification head predicts softmax probabilities across 5 classes, returning deterministic pedagogical hints.
            </p>
          </div>

        </div>
      </div>

      {/* 5 Error Taxonomy */}
      <div>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Target size={20} color="var(--accent-cyan)" />
          Taxonomy of Logic Error Mutations
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
          {errorCategories.map(cat => (
            <div key={cat.id} className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 700 }}>{cat.name}</h4>
                  <span className={`badge ${cat.badge}`}>{cat.id}</span>
                </div>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', marginBottom: '12px', lineHeight: 1.5 }}>
                  {cat.desc}
                </p>
              </div>
              <pre style={{
                background: 'var(--bg-code)',
                padding: '10px',
                borderRadius: 'var(--radius-sm)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                color: '#93c5fd',
                overflowX: 'auto',
                border: '1px solid var(--border-subtle)',
              }}>
                {cat.example}
              </pre>
            </div>
          ))}
        </div>
      </div>

      {/* Viva / Academic FAQ Section */}
      <div className="glass-panel" style={{ padding: '28px' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <HelpCircle size={20} color="var(--accent-amber)" />
          University Demonstration & Viva Q&A Guide
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '16px' }}>
          {vivaQuestions.map((item, idx) => (
            <div key={idx} style={{
              padding: '16px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid var(--border-subtle)',
            }}>
              <p style={{ fontSize: '0.9rem', fontWeight: 700, color: '#ffffff', marginBottom: '8px', display: 'flex', gap: '8px' }}>
                <span style={{ color: 'var(--primary-light)' }}>Q:</span>
                {item.q}
              </p>
              <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.6, display: 'flex', gap: '8px' }}>
                <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>A:</span>
                {item.a}
              </p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
