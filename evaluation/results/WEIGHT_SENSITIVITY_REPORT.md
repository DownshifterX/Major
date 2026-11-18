# Weight Sensitivity Analysis Report

> **[PROXY LABELS]** All metrics use keyword/statutory proxy relevance.

## Weight Configurations

| Config | Semantic | Lexical | Statute | Issue | Citation | Court | MRR | P@5 | NDCG@5 |
|---|---|---|---|---|---|---|---|---|---|
| config_A_default | 0.35 | 0.2 | 0.2 | 0.1 | 0.08 | 0.07 | 1.0 | 0.92 | 0.837 |
| config_B_high_semantic | 0.55 | 0.15 | 0.15 | 0.07 | 0.05 | 0.03 | 1.0 | 0.92 | 0.837 |
| config_C_high_statutory | 0.25 | 0.15 | 0.4 | 0.1 | 0.06 | 0.04 | 1.0 | 0.92 | 0.8012 |
| config_D_high_citation | 0.28 | 0.17 | 0.17 | 0.08 | 0.25 | 0.05 | 1.0 | 0.92 | 0.837 |
| config_E_balanced | 0.167 | 0.167 | 0.167 | 0.167 | 0.166 | 0.166 | 1.0 | 0.92 | 0.8012 |

## Interpretation

Results show how sensitive the reranker is to weight changes.
The current default weights (config_A) serve as the reference.

**Do NOT declare a 'best' configuration without human-annotated labels.**
Measured differences here are relative to proxy labels only.
