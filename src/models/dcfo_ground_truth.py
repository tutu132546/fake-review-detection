# -*- coding: utf-8 -*-
"""C2：DCFO 反事实解释的 ground truth 对齐（v2 日志 feature_delta_sigma）。

用重跑后的 injection_log_v2.json（唯一真源）做三件事：
  1. 逐账号把日志里的 7 维 feature_delta_sigma 与「augmented − clean」表算出的
     (Δ/σ) 逐位比对，确认日志不是手写/过期数据；
  2. 按模板汇总 dz 均值，与 data_dictionary 第五节 §7 的实测排序对照；
  3. 校验 A/C 的 burstiness dz > 0、B 的 burstiness 保持接近 0（Issue #1 的可证伪判据）。

只读 clean / augmented 表与日志，不重跑注入、不改表。
运行：python src/models/dcfo_ground_truth.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PROJECT_ROOT = _SRC.parent
FEATURES = ['n_reviews', 'active_days', 'daily_avg_reviews', 'max_daily_reviews',
            'burstiness', 'rating_deviation', 'extreme_ratio']
TEMPLATES = ('A_Time_Burst', 'B_Rating_Polarization', 'C_Stealth_Mix')

# data_dictionary 第五节 §7 记录的模板主导缺陷——本脚本用于复核该表。
# n_reviews / daily_avg_reviews 是全模板公共效应（§9），不参与"模板特异性"排序。
COMMON_EFFECT = ['n_reviews', 'daily_avg_reviews']
EXPECTED_ORDER = {
    'A_Time_Burst': ['max_daily_reviews', 'burstiness', 'extreme_ratio'],
    'B_Rating_Polarization': ['extreme_ratio', 'rating_deviation', 'max_daily_reviews'],
    'C_Stealth_Mix': ['max_daily_reviews', 'extreme_ratio', 'rating_deviation'],
}
TIE_TOL = 0.05  # 日志保留两位小数展示时的近似并列容差


def _dz_table(log, clean, aug, scaler):
    """按日志账号重算 (aug−clean)/σ，返回 DataFrame（index=user_id）。"""
    rows = []
    for rec in log:
        uid = int(rec['user_id'])
        c = clean.loc[uid, FEATURES]
        a = aug.loc[uid, FEATURES]
        dz = {f: (a[f] - c[f]) / scaler[f]['std'] if not np.isnan(a[f]) and not np.isnan(c[f])
              else (0.0 if np.isnan(a[f]) and np.isnan(c[f]) else np.nan)
              for f in FEATURES}
        rows.append({'user_id': uid, 'template': rec['injection_template'], **dz})
    return pd.DataFrame(rows).set_index('user_id')


def _log_dz_table(log):
    rows = []
    for rec in log:
        rows.append({'user_id': int(rec['user_id']),
                     'template': rec['injection_template'],
                     **{f: rec['feature_delta_sigma'][f] for f in FEATURES}})
    return pd.DataFrame(rows).set_index('user_id')


def main(args):
    clean = pd.read_parquet(args.clean).set_index('user_id')
    aug = pd.read_parquet(args.augmented).set_index('user_id')
    with open(args.log, encoding='utf-8') as f:
        log = json.load(f)
    with open(args.scaler, encoding='utf-8') as f:
        scaler = json.load(f)

    recomputed = _dz_table(log, clean, aug, scaler)
    logged = _log_dz_table(log)

    # 1) 逐账号逐特征比对（日志保留 4 位小数，容差 1e-3）
    both = recomputed[FEATURES] - logged[FEATURES]
    abs_diff = both.abs()
    max_diff = float(np.nanmax(abs_diff.to_numpy()))
    mismatch_cells = int((abs_diff > 1e-3).sum().sum())
    nan_mismatch = int((recomputed[FEATURES].isna() != logged[FEATURES].isna()).sum().sum())

    # 2) 模板级 dz 均值
    mean_by_tpl = logged.groupby('template')[FEATURES].mean()
    mean_by_tpl = mean_by_tpl.loc[list(TEMPLATES)]

    # 3) 排序与方向判据（公共效应特征不参与模板特异性排序；≤TIE_TOL 视为并列）
    ranking_checks = {}
    for tpl in TEMPLATES:
        vals = mean_by_tpl.loc[tpl]
        expected = EXPECTED_ORDER[tpl]
        ok = all(vals[expected[i]] >= vals[expected[i + 1]] - TIE_TOL
                 for i in range(len(expected) - 1))
        actual = vals.drop(index=COMMON_EFFECT).sort_values(ascending=False).index.tolist()[:3]
        ranking_checks[tpl] = (ok, actual, vals) 
    burst = mean_by_tpl['burstiness']
    dir_checks = {
        'A burstiness dz > 0': bool(burst['A_Time_Burst'] > 0),
        'C burstiness dz > 0': bool(burst['C_Stealth_Mix'] > 0),
        'B burstiness 近似未动（|dz| < 0.5）': bool(abs(burst['B_Rating_Polarization']) < 0.5),
        '全模板 active_days dz = 0': bool((mean_by_tpl['active_days'].abs() < 1e-9).all()),
    }

    # 4) modified_features 与重算增量的存在性一致（NaN 转换也算"被修改"）
    fp = fn = 0
    for rec in log:
        uid = int(rec['user_id'])
        mod = rec['modified_features']
        for f in FEATURES:
            c, a = clean.loc[uid, f], aug.loc[uid, f]
            changed = bool((not np.isnan(a) and not np.isnan(c) and abs(a - c) > 1e-12)
                           or (np.isnan(a) != np.isnan(c)))
            if changed and f not in mod:
                fn += 1
            if not changed and f in mod:
                fp += 1

    hard_ok = (mismatch_cells == 0 and nan_mismatch == 0 and fp == 0 and fn == 0
               and all(v[0] for v in ranking_checks.values()) and all(dir_checks.values()))

    lines = [
        '# C2：DCFO ground truth 对齐（v2 日志 `feature_delta_sigma`）',
        '',
        '> 数据源：`data/processed/injection_log_v2.json`（重跑 step3 后的唯一真源，'
        f'{len(log)} 个账号 / {sum(len(r["events_added"]) for r in log):,} 条注入事件）。',
        '> 反事实解释的 ground truth 是"注入前后 7 维特征的 σ 单位增量"，'
        '本报告校验它与磁盘表逐位自洽，并复核模板主导缺陷排序。',
        '',
        '## 1. 逐账号重算比对',
        '',
        '| 检查 | 结果 |',
        '|---|---:|',
        f'| 比对单元（300 账号 × 7 特征） | {len(log) * len(FEATURES):,} |',
        f'| 最大 \\|日志 − 重算\\| | {max_diff:.6f} |',
        f'| 超出 1e-3 的单元 | {mismatch_cells} |',
        f'| NaN 状态不一致的单元 | {nan_mismatch} |',
        f'| `modified_features` 漏记（表变但日志未记） | {fn} |',
        f'| `modified_features` 多记（表未变但日志列出） | {fp} |',
        '',
        '日志的 `delta_sigma` 落盘保留 4 位小数，重算值为全精度；'
        f'最大差 {max_diff:.2e} 落在舍入误差内，判定 **{"一致" if mismatch_cells == 0 else "不一致"}**。',
        '',
        '## 2. 模板级 dz 均值（与 data_dictionary §7 对照）',
        '',
        '| 特征 | A_Time_Burst | B_Rating_Polarization | C_Stealth_Mix |',
        '|---|---:|---:|---:|',
    ]
    for f in FEATURES:
        lines.append(f'| {f} | {mean_by_tpl.loc["A_Time_Burst", f]:+.2f} | '
                     f'{mean_by_tpl.loc["B_Rating_Polarization", f]:+.2f} | '
                     f'{mean_by_tpl.loc["C_Stealth_Mix", f]:+.2f} |')

    lines += ['', f'模板特异性排序（已剔除公共效应 {"/".join(COMMON_EFFECT)}；'
                  f'相差 ≤{TIE_TOL:.2f} 视为并列）：', '',
              '| 模板 | 期望顺序（§7） | 实际数值 | 判定 |', '|---|---|---|---|']
    for tpl in TEMPLATES:
        ok, actual, vals = ranking_checks[tpl]
        shown = ' > '.join(f'{f} {vals[f]:+.2f}' for f in actual)
        lines.append(f'| {tpl} | {" > ".join(EXPECTED_ORDER[tpl])} | {shown} | '
                     f'{"PASS" if ok else "FAIL"} |')

    lines += ['', '## 3. 方向判据（Issue #1 可证伪项）', '', '| 判据 | 结果 |', '|---|---|']
    for k, v in dir_checks.items():
        lines.append(f'| {k} | {"PASS" if v else "FAIL"} |')

    lines += [
        '',
        f'- 总体判定：**{"PASS" if hard_ok else "FAIL"}**'
        '（逐账号一致 ∧ 排序一致 ∧ 方向判据全过）。',
        '- 用途：解释方法（如 DCFO / SHAP 类反事实）应恢复出与上表同量级的 dz；'
        '"是否恢复出主导缺陷" 是可证伪指标，不以"解释看起来合理"代替。',
        '',
        '## 复现',
        '',
        '```bash',
        'python src/models/dcfo_ground_truth.py',
        '```',
        '',
        f'环境：Python {sys.version.split()[0]}；numpy={np.__version__}；pandas={pd.__version__}。',
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'max_abs_diff': max_diff, 'mismatch_cells': mismatch_cells,
                      'nan_mismatch': nan_mismatch, 'modified_false_negative': fn,
                      'modified_false_positive': fp,
                      'ranking_ok': {k: v[0] for k, v in ranking_checks.items()},
                      'direction_ok': dir_checks, 'pass': hard_ok}, ensure_ascii=False))
    print(f'C2 report: {args.report}')
    if not hard_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=Path('data/processed/injection_log_v2.json'))
    parser.add_argument('--clean', type=Path, default=Path('data/processed/clean_accounts.parquet'))
    parser.add_argument('--augmented', type=Path,
                        default=Path('data/processed/augmented_accounts_v2.parquet'))
    parser.add_argument('--scaler', type=Path, default=Path('data/processed/scaler_params.json'))
    parser.add_argument('--report', type=Path,
                        default=Path('results/c_dcfo_ground_truth.md'))
    main(parser.parse_args())
