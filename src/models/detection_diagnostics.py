"""Reproducible LOF/iForest diagnostics on the B-member handoff.

No label is used in fitting. No deduplication, jitter, rescaling or score flipping.
This is transductive evaluation, not a held-out test. Account-level outputs stay
in data/processed; only aggregate reports should be committed.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

RAW_FEATURES = ["n_reviews", "active_days", "daily_avg_reviews", "max_daily_reviews",
                "burstiness", "rating_deviation", "extreme_ratio"]
FEATURES = [f"{f}_z" for f in RAW_FEATURES] + ["burstiness_missing"]
LABELS = ["user_id", "account_is_fake", "is_injected"]
POPULATIONS = ['native', 'native_ge3', 'native_ge3_observed', 'augmented']


def load_features(path, population="native"):
    frame = pd.read_csv(path)
    missing = set(FEATURES + LABELS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.user_id.isna().any() or not frame.user_id.is_unique:
        raise ValueError("user_id must be nonmissing and unique")
    for col in ["account_is_fake", "is_injected", "burstiness_missing"]:
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be binary and nonmissing")
    X = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Model features contain nonfinite values")
    if population in ['native_ge3', 'native_ge3_observed']:
        if 'n_reviews' not in frame:
            raise ValueError('Cohort selection requires raw n_reviews')
        counts = pd.to_numeric(frame.n_reviews, errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts < 1).any() or (counts != np.floor(counts)).any():
            raise ValueError('n_reviews must contain positive integer counts')
        frame = frame.loc[frame.is_injected.eq(0) & (counts >= 3)].copy()
        if population == 'native_ge3_observed':
            frame = frame.loc[frame.burstiness_missing.eq(0)].copy()
    elif population == "native":
        frame = frame.loc[frame.is_injected.eq(0)].copy()
    elif population != "augmented":
        raise ValueError("Unknown population")
    # Explicit tie policy: equal anomaly scores are ordered by ascending user_id.
    frame = frame.sort_values("user_id", kind="stable").reset_index(drop=True)
    return frame, frame[FEATURES].to_numpy(dtype=float)


def evaluate(y, scores, top_k=50):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid score vector")
    if len(y) < top_k or top_k < 1:
        raise ValueError("top_k must be between 1 and population size")
    order = np.argsort(-scores, kind="stable")
    cutoff = scores[order[top_k - 1]]
    higher, tied = scores > cutoff, scores == cutoff
    slots = top_k - int(higher.sum())
    positives = int(y[tied].sum())
    negatives = int(tied.sum()) - positives
    known = int(y[higher].sum())
    return {
        "auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "p_at_k": float(y[order[:top_k]].mean()), "top_k": top_k,
        "p_at_k_tie_min": (known + max(0, slots - negatives)) / top_k,
        "p_at_k_tie_max": (known + min(slots, positives)) / top_k,
        "boundary_tie_size": int(tied.sum()),
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "score_above_million": int((scores > 1e6).sum()),
    }


def run(args):
    if len(set(args.neighbors)) != len(args.neighbors) or any(k < 2 for k in args.neighbors):
        raise ValueError("Use distinct neighbor counts >= 2")
    frame, X = load_features(args.input, args.population)
    if len(frame) <= max(args.neighbors):
        raise ValueError("Every k must be smaller than the selected population")
    if len(frame) < 50:
        raise ValueError("P@50 requires at least 50 rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = frame.groupby(FEATURES, dropna=False).user_id.transform("size").to_numpy()
    report = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "population": args.population, "rows": len(frame), "features": FEATURES,
        "evaluation": "transductive, fixed supplied scaling; not held-out",
        "selection": "all specified k values reported; no automatic winner selection",
        "tie_policy": "score descending, then user_id ascending; tie bounds also reported",
        "duplicate_feature_rows_beyond_first": int(frame[FEATURES].duplicated().sum()),
        "largest_identical_group": int(counts.max()),
        "positive_labels": int(frame.account_is_fake.sum()),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
        "runs": [],
    }
    saved = frame[LABELS].copy()
    first_scores, first_top20 = None, None
    for k in args.neighbors:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LocalOutlierFactor(n_neighbors=k, algorithm="kd_tree", n_jobs=2)
            model.fit(X)
        scores = -model.negative_outlier_factor_.copy()
        row = evaluate(frame.account_is_fake, scores)
        row.update(model="LOF", k=k, seconds=time.perf_counter()-start,
                   exact_zero_k_distance_rows=int((counts > k).sum()),
                   warnings=[str(w.message) for w in caught])
        top20 = set(np.argsort(-scores, kind="stable")[:20])
        if first_scores is None:
            first_scores, first_top20 = scores.copy(), top20
        row["top20_overlap_with_first_k"] = len(top20 & first_top20)
        correlation = spearmanr(first_scores, scores).statistic
        row["spearman_with_first_k"] = float(correlation) if np.isfinite(correlation) else None
        row["above_threshold_1_5"] = int((scores > 1.5).sum())
        saved[f"lof_k{k}"] = scores
        report["runs"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del model
        gc.collect()
    start = time.perf_counter()
    model = IsolationForest(n_estimators=200, max_samples=256, random_state=42, n_jobs=2)
    model.fit(X)
    scores = -model.score_samples(X)
    row = evaluate(frame.account_is_fake, scores)
    row.update(model="iForest", n_estimators=200, max_samples=256, random_state=42,
               seconds=time.perf_counter()-start, warnings=[])
    saved['iforest'] = scores
    report["runs"].append(row)
    saved.to_csv(args.output_dir/'scores.csv', index=False)
    (args.output_dir/'diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# C: LOF / iForest diagnostic run", "",
                 "Preliminary transductive diagnostics, not final detection or DCFO results.", "",
                 f"- Population: {args.population}; rows: {len(frame):,}; features: {len(FEATURES)}.",
                 f"- Source SHA256: `{report['input_sha256']}`.",
                 f"- Largest identical-feature group: {report['largest_identical_group']} accounts.",
                 f"- Duplicate rows beyond first: {report['duplicate_feature_rows_beyond_first']}.",
                 "- Labels excluded from fitting; supplied scaling preserved; no rows deduplicated or perturbed.",
                 "- P@50 ties: ascending account ID; min/max over boundary ties also reported.",
                 "- All predeclared k values are reported; no best model selected using these labels.", "",
                 "| Method | k | AUC | P@50 | tie range | max score | zero k-distance rows | Top20 overlap with first k | warnings |",
                 "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
        for r in report['runs']:
            auc = f"{r['auc']:.6f}" if r['auc'] is not None else "NA"
            lines.append(f"| {r['model']} | {r.get('k','—')} | {auc} | {r['p_at_k']:.2f} | {r['p_at_k_tie_min']:.2f}–{r['p_at_k_tie_max']:.2f} | {r['score_max']:.6g} | {r.get('exact_zero_k_distance_rows','—')} | {r.get('top20_overlap_with_first_k','—')} | {len(r['warnings'])} |")
        lines += ["", "## Runtime and warnings", ""]
        for r in report['runs']:
            lines.append(f"- {r['model']} k={r.get('k','NA')}: {r['seconds']:.2f}s. " + ("; ".join(r['warnings']) or "No warning."))
        lines += ["", "## Reproducibility", "", f"Python {report['python']}; " + ", ".join(f"{p}={v}" for p,v in report['versions'].items()), "",
                  "Use the supplied feature_table.csv and the README command in docs/c_detection.md.",
                  "Native evaluation excludes 300 selected injected accounts. It is not the complete pre-injection clean population.",
                  "Threshold 1.5 is diagnostic only. Absence of warnings does not establish detection validity or CF feasibility."]
        args.report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--population', choices=POPULATIONS, default='native')
    parser.add_argument('--neighbors', type=int, nargs='+', default=[20,50,100,600])
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/c_diagnostics'))
    parser.add_argument('--report', type=Path)
    run(parser.parse_args())
