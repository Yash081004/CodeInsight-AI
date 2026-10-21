# Academic Project Report — Suggested Outline

1. **Abstract** — Two tasks (similarity + five-class logic errors); honest limitations (small N, preliminary similarity test).
2. **Introduction** — Motivation for student Python code analysis.
3. **Related work** — CodeBERT, semantic code similarity, educational error taxonomies.
4. **Dataset** — 91 samples, 21 templates, grouped splits (seed 42), checksum; similarity 193 pairs / 7 test evaluable.
5. **Methods** — CodeBERT fine-tuning vs TF-IDF+LogReg; embedding cosine vs TF-IDF cosine; validation-only thresholds.
6. **Explainability** — Softmax confidence, static AST heuristics, deterministic feedback (not attention saliency).
7. **Results** — Baseline macro F1 27.17% vs experiment CodeBERT 28.00% (N=16, inconclusive); similarity preliminary.
8. **System design** — FastAPI + React; production vs experiment artifacts.
9. **Ethics & safety** — No arbitrary code execution on server; privacy-preserving history hashes.
10. **Limitations & future work** — Data scale, GPU training, external benchmarks.
11. **Conclusion**
12. **Appendix** — Reproduction commands from `README.md`
