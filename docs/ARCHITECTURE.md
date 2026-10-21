# CodeInsight AI — Architecture

```mermaid
flowchart TB
  subgraph UI [React 19 + Vite]
    Sim[SimilarityTab]
    Clf[ClassifierTab]
    Eval[EvaluationTab]
  end

  subgraph API [FastAPI]
    H[/health /ready/]
    S[/similarity/analyze/]
    C[/classification/analyze/]
    E[/evaluation/* /]
  end

  subgraph ML [Inference]
    Enc[CodeBERTEncoder base]
    Prod[ErrorClassifier production checkpoint]
    BL[TF-IDF + LogReg joblibs]
  end

  Sim --> S
  Clf --> C
  Eval --> E

  S --> Enc
  S --> BL
  C --> Prod
  C --> BL
  C --> XAI[explainability_service AST hints]

  Prod --> PathProd[(models/codebert_classifier)]
  BL --> PathBase[(models/baselines)]
  E --> Art[(artifacts/metrics)]
```

## Classification path

1. `POST /api/classification/analyze` → `analyze_code()`
2. AST validation (`code_validator`) — no execution of user code
3. Load **`settings.classifier_model_path`** → `models/codebert_classifier/`
4. Fallback: baseline joblibs
5. Response includes `probabilities`, deterministic `feedback`, and `diagnostics`

## Similarity path

1. `POST /api/similarity/analyze` → `analyze_similarity()`
2. CodeBERT mean-pooled embeddings + cosine similarity
3. TF-IDF cosine baseline
4. Threshold comparison — scores are **not** calibrated probabilities

## Evaluation artifacts

| Run | Metrics directory |
|-----|-------------------|
| Production | `artifacts/metrics/codebert_classification_metrics.json` |
| Experiment | `artifacts/metrics/experiments/<run_id>/` |

API inference always uses **production** checkpoint unless configuration is changed.
