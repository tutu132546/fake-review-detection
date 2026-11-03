# C: LOF / iForest diagnostic run

Preliminary transductive diagnostics, not final detection or DCFO results.

- Population: native; rows: 37,763; features: 8.
- Source SHA256: `0051cdd29478d119a54440194ac4c61093a7692a99b20f832093e66c0446f39f`.
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

- LOF k=20: 2.06s. Duplicate values are leading to incorrect results. Increase the number of neighbors for more accurate results.
- LOF k=50: 2.23s. Duplicate values are leading to incorrect results. Increase the number of neighbors for more accurate results.
- LOF k=100: 3.36s. Duplicate values are leading to incorrect results. Increase the number of neighbors for more accurate results.
- LOF k=600: 10.96s. No warning.
- iForest k=NA: 1.81s. No warning.

## Reproducibility

Python 3.13.7; numpy=2.2.6, pandas=2.3.3, scipy=1.16.2, scikit-learn=1.7.2

Use the supplied feature_table.csv and the README command in docs/c_detection.md.
Native evaluation excludes 300 selected injected accounts. It is not the complete pre-injection clean population.
Threshold 1.5 is diagnostic only. Absence of warnings does not establish detection validity or CF feasibility.
