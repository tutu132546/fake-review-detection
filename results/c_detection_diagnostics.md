# C: LOF / iForest diagnostic run

Preliminary transductive diagnostics, not final detection or DCFO results.

- Population: native; rows: 37,763; features: 8.
- Source SHA256: `9b39675b006464e8a6126863aa56091040a6fdb6a7638feec56fe26828e897de`.
- Largest identical-feature group: 583 accounts.
- Duplicate rows beyond first: 26179.
- Labels excluded from fitting; supplied scaling preserved; no rows deduplicated or perturbed.
- P@50 ties: ascending account ID; min/max over boundary ties also reported.
- All predeclared k values are reported; no best model selected using these labels.

| Method | k | AUC | P@50 | tie range | max score | zero k-distance rows | Top20 overlap with first k | warnings |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| LOF | 20 | 0.470931 | 0.20 | 0.20–0.20 | 7.24031e+09 | 22395 | 20 | 1 |
| LOF | 50 | 0.480585 | 0.10 | 0.10–0.10 | 2.99326e+09 | 16000 | 0 | 1 |
| LOF | 100 | 0.485508 | 0.04 | 0.04–0.04 | 3.8527e+09 | 8103 | 0 | 1 |
| LOF | 600 | 0.498626 | 0.14 | 0.14–0.14 | 123.148 | 0 | 0 | 0 |
| iForest | — | 0.416999 | 0.14 | 0.14–0.14 | 0.762272 | — | — | 0 |

## Runtime and warnings

- LOF k=20: 0.55s. Duplicate values are leading to incorrect results. Increase the number of neighbors for more accurate results.
- LOF k=50: 0.71s. Duplicate values are leading to incorrect results. Increase the number of neighbors for more accurate results.
- LOF k=100: 0.84s. Duplicate values are leading to incorrect results. Increase the number of neighbors for more accurate results.
- LOF k=600: 2.91s. No warning.
- iForest k=NA: 0.43s. No warning.

## Reproducibility

Python 3.14.5; numpy=2.5.1, pandas=3.0.5, scipy=1.18.1, scikit-learn=1.9.1

Use the supplied feature_table.csv and the README command in docs/c_detection.md.
Native evaluation excludes 300 selected injected accounts. It is not the complete pre-injection clean population.
Threshold 1.5 is diagnostic only. Absence of warnings does not establish detection validity or CF feasibility.
