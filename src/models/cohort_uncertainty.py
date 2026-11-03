"""Audit saved cohort scores; conditional, paired, stratified bootstrap AUC."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from detection_diagnostics import load_features

MODELS = ['lof_k20', 'lof_k50', 'lof_k100', 'lof_k600', 'iforest']
COHORTS = ['native', 'native_ge3', 'native_ge3_observed']


def grouped_auc(groups, labels, weights):
    """Weighted AUC, crediting half for ties, using ascending score groups."""
    positive = np.bincount(groups, weights=weights * labels)
    negative = np.bincount(groups, weights=weights * (1-labels), minlength=len(positive))
    return float(np.sum(positive * (np.cumsum(negative)-negative/2)) /
                 (positive.sum()*negative.sum()))


def bootstrap_auc(labels, scores, repeats=2000, seed=42):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if repeats < 100 or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Need two label classes and at least 100 repetitions')
    if scores.ndim != 2 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError('Invalid scores')
    groups = [np.unique(scores[:, j], return_inverse=True)[1] for j in range(scores.shape[1])]
    indices = [np.flatnonzero(labels == label) for label in [0, 1]]
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, scores.shape[1]))
    weights = np.zeros(len(labels))
    for b in range(repeats):
        # Same sampled accounts for all models: paired comparisons remain paired.
        for idx in indices:
            weights[idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)))
        for j, group in enumerate(groups):
            samples[b, j] = grouped_auc(group, labels, weights)
    return samples


def profile(frame):
    return [len(frame), float(frame.account_is_fake.mean()),
            float(frame.n_reviews.mean()), float(frame.n_reviews.quantile(.75)),
            float(frame.active_days.mean()), float(frame.active_days.quantile(.75)),
            float(frame.n_reviews.ge(3).mean())]


def main(args):
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    lines = ['# C：PR #5 审查补充：lift、区间与账号画像', '',
             f'输入 SHA256：`{source_hash}`', '',
             '使用已保存的账号分数，不重新训练或选择参数。Lift@50 = P@50 / 本人群标签1比例，使用未舍入的比例计算；1倍是随机抽取的期望，不代表已验证的业务收益。', '',
             f'AUC区间：按标签分层、账号有放回抽样，{args.repeats}次，随机种子{args.seed}，百分位法95%区间。同一次重采样用于所有模型，模型差值使用配对区间。',
             '**范围限制：这是固定模型分数下的条件区间，不重新拟合模型，不包含特征生成、训练、参数选择或人群选择的不确定性；账号独立假设可能受共同商家/团伙影响。不是独立测试集上的泛化区间，也未校正多重比较。**', '',
             '| 人群 | 模型 | AUC | 条件95%区间 | P@50 | 标签1比例 | Lift@50 |',
             '|---|---|---:|---|---:|---:|---:|']
    comparisons, profiles, output = [], [], []
    for cohort in COHORTS:
        directory = args.scores_dir/cohort
        metadata = json.loads((directory/'diagnostics.json').read_text(encoding='utf-8'))
        if metadata['input_sha256'] != source_hash or metadata['population'] != cohort:
            raise ValueError('Saved scores belong to a different input or population')
        frame, _ = load_features(args.input, cohort)
        saved = pd.read_csv(directory/'scores.csv')
        if saved.user_id.duplicated().any() or set(saved.user_id) != set(frame.user_id):
            raise ValueError('Account IDs do not match')
        frame = frame.set_index('user_id').loc[saved.user_id].reset_index()
        if not np.array_equal(frame.account_is_fake, saved.account_is_fake):
            raise ValueError('Labels do not match')
        y = saved.account_is_fake.to_numpy()
        matrix = saved[MODELS].to_numpy()
        draws = bootstrap_auc(y, matrix, args.repeats, args.seed)
        aucs = np.array([roc_auc_score(y, matrix[:, j]) for j in range(len(MODELS))])
        entries = []
        for j, model in enumerate(MODELS):
            order = saved.sort_values([model, 'user_id'], ascending=[False, True]).index[:50]
            precision = float(y[order].mean())
            low, high = np.quantile(draws[:, j], [.025, .975])
            lift = precision / y.mean()
            lines.append(f'| {cohort} | {model} | {aucs[j]:.6f} | [{low:.4f}, {high:.4f}] | {precision:.2f} | {y.mean():.4%} | {lift:.3f} |')
            entries.append(dict(model=model, auc=aucs[j], ci=[low, high], p_at_50=precision, lift=lift))
            profiles.append((cohort, f'{model} Top-50', profile(frame.iloc[order])))
        for label in [0, 1]:
            profiles.append((cohort, f'全部标签{label}', profile(frame.loc[frame.account_is_fake.eq(label)])))
        differences = []
        for a, b in [(4, 0), (4, 1), (4, 2), (4, 3), (1, 0), (2, 0), (3, 0)]:
            low, high = np.quantile(draws[:, a]-draws[:, b], [.025, .975])
            comparisons.append(f'| {cohort} | {MODELS[a]} − {MODELS[b]} | {aucs[a]-aucs[b]:.4f} | [{low:.4f}, {high:.4f}] |')
            differences.append(dict(comparison=f'{MODELS[a]} - {MODELS[b]}', difference=aucs[a]-aucs[b], ci=[low, high]))
        output.append(dict(population=cohort, metrics=entries, differences=differences))
    lines += ['', '## 配对AUC差值', '',
              '区间包含0时，不能凭点估计断言排序优劣；单模型区间是否重叠不能替代配对差值检验。以下为探索性比较，不能据此挑选最佳参数。', '',
              '| 人群 | 差值方向 | AUC差值 | 条件95%区间 |', '|---|---|---:|---|'] + comparisons
    lines += ['', '## 标签与Top-50并列画像', '',
              '活跃跨度指首末评论跨度加一天。Top-50经过异常分数筛选，以下仅描述其构成，不据此证明模型偏好或因果关系。均值对极端值敏感，故同时列出75%分位数与评论数≥3比例。', '',
              '| 人群 | 账号组 | 人数 | 标签1比例 | 评论均值 | 评论Q75 | 跨度均值/天 | 跨度Q75/天 | 评论≥3比例 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort, name, (n, rate, rm, rq, dm, dq, ge3) in profiles:
        lines.append(f'| {cohort} | {name} | {n} | {rate:.2%} | {rm:.2f} | {rq:.2f} | {dm:.2f} | {dq:.2f} | {ge3:.2%} |')
    lines += ['', '## 复核方式', '',
              '`python -m unittest discover -s tests -v` 不需要真实数据。',
              '`python src/models/cohort_analysis.py --input data/processed/feature_table.csv` 先生成分数。',
              '`python src/models/cohort_uncertainty.py --input data/processed/feature_table.csv` 再生成本报告。',
              '依赖版本见分人群报告。真实数据须另行取得；审阅者只看报告不等于已复跑。P@50的并列范围见原报告，本报告不提供P@50或lift的bootstrap区间。']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (args.scores_dir/'uncertainty.json').write_text(json.dumps(dict(input_sha256=source_hash, repeats=args.repeats, seed=args.seed, results=output), ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--scores-dir', type=Path, default=Path('data/processed/c_cohorts'))
    parser.add_argument('--report', type=Path, default=Path('results/c_cohort_uncertainty.md'))
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
