import React, { useEffect, useState } from 'react';

import { GitCompare, Bug, ArrowRight } from 'lucide-react';

import { fetchEvaluationSummary } from '../api';

import LabelInfoGrid from './ui/LabelInfoGrid';



export default function OverviewTab({ onNavigateTab, health }) {

  const [bench, setBench] = useState(null);



  useEffect(() => {

    let cancelled = false;

    fetchEvaluationSummary('production')

      .then((sum) => { if (!cancelled) setBench(sum); })

      .catch(() => {});

    return () => { cancelled = true; };

  }, []);



  const cbF1 = bench?.codebert_metrics?.macro_f1 ?? bench?.codebert_classification?.macro_f1;

  const blF1 = bench?.baseline_metrics?.macro_f1 ?? bench?.baseline_classification?.macro_f1;



  return (

    <div className="workspace">

      <section className="overview-hero">

        <h2 className="display-title" style={{ fontSize: 'clamp(2.1rem, 4.5vw, 3rem)' }}>

          Code similarity & logic diagnosis

        </h2>

        <p className="page-subtitle" style={{ marginTop: 10, maxWidth: '36rem' }}>

          Compare Python with CodeBERT, classify logic-error hypotheses, and inspect saved benchmark metrics—always labeled as model output, not verified truth.

        </p>

        <div className="overview-actions">

          <button type="button" onClick={() => onNavigateTab('similarity')} className="btn btn-primary btn-lg">

            <GitCompare size={18} aria-hidden />

            Compare code

            <ArrowRight size={16} aria-hidden />

          </button>

          <button type="button" onClick={() => onNavigateTab('classifier')} className="btn btn-secondary btn-lg">

            <Bug size={18} aria-hidden />

            Analyze logic

          </button>

        </div>

      </section>



      <div className="overview-status-strip">

        {health ? (

          <>

            <span>API <strong>{health.status === 'ok' ? 'online' : health.status || 'unknown'}</strong></span>

            <span>Classifier <strong>{health.classifier_available ? 'CodeBERT' : health.baseline_available ? 'TF-IDF' : 'unavailable'}</strong></span>

            <span>Device <strong>{health.device || '—'}</strong></span>

          </>

        ) : (

          <span>API <strong>checking…</strong></span>

        )}

        {cbF1 != null && (

          <span>

            CodeBERT test macro F1 <strong>{(cbF1 * 100).toFixed(1)}%</strong>

          </span>

        )}

        {blF1 != null && (

          <span>

            Baseline test macro F1 <strong>{(blF1 * 100).toFixed(1)}%</strong>

          </span>

        )}

      </div>



      <p style={{ fontSize: '0.78rem', color: 'var(--ochre)', fontFamily: 'var(--font-mono)' }}>

        Metrics from held-out artifacts; synthetic labels not fully human-reviewed.

      </p>



      <details className="expand-section">

        <summary>Five-class taxonomy reference</summary>

        <div className="expand-section__body">

          <LabelInfoGrid />

        </div>

      </details>

    </div>

  );

}

