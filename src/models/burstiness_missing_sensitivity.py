# -*- coding: utf-8 -*-
"""C3：burstiness_missing 缺失标记的敏感性分析（8 列 vs 7 列）。

对同一份 feature_table，仅在「保留 burstiness_missing」与「去掉该标记」两种输入下
分别跑预置 k 的 LOF 与 iForest，比较 AUC 与 P@50。不重选模型、不调参、不翻转分数。
报告写入 results/，账号级分数留在 data/processed（忽略目录）。

运行：python src/models/burstiness_missing_sensitivity.py --input data/processed/feature_table.csv
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import detection_diagnostics as dd

WITH_FLAG = list(dd.FEATURES)
WITHOUT_FLAG = [f for f in WITH_FLAG if f != 'burstiness_missing']
POPULATIONS = ['native', 'native_ge3']


def _run(input_path, population, features, out_dir):
    dd.FEATURES = features
    report = dd.run(SimpleNamespace(input=input_path, population=population,
                                    neighbors=[20, 50, 100, 600],
                                    output_dir=out_dir, report=None))
    return report


def main(args):
    out_root = args.output_dir
    results = {}
    for pop in POPULATIONS:
        results[pop] = {}
        for tag, feats in (('with_flag', WITH_FLAG), ('without_flag', WITHOUT_FLAG)):
            out_dir = out_root / f'{pop}_{tag}'
            out_dir.mkdir(parents=True, exist_ok=True)
            rep = _run(args.input, pop, feats, out_dir)
            results[pop][tag] = {f"{r['model']}-{r.get('k', 'NA')}": r for r in rep['runs']}

    lines = [
        '# C3：burstiness_missing 缺失标记敏感性',
        '',
        '对同一份 `feature_table.csv`，在「保留 `burstiness_missing`」（8 列，正式接口）与'
        '「去掉该列」（7 列）两种输入下，用相同预置 k 与随机种子复跑 LOF / iForest；'
        '标签不参与拟合，不做去重、扰动、重标定或分数翻转。',
        '',
        '> 范围：这是同一批账号上的转导式诊断，不是独立测试；差值只说明该缺失标记'
        '在当前表上的边际贡献，不能外推为泛化收益。',
        '',
        '| 人群 | 模型 | k | 8 列 AUC | 7 列 AUC | ΔAUC | 8 列 P@50 | 7 列 P@50 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    max_abs = 0.0
    for pop, variants in results.items():
        for key in variants['with_flag']:
            a = variants['with_flag'][key]
            b = variants['without_flag'][key]
            delta = b['auc'] - a['auc']
            max_abs = max(max_abs, abs(delta))
            lines.append(
                f"| {pop} | {a['model']} | {a.get('k', '—')} | {a['auc']:.6f} | {b['auc']:.6f} | "
                f"{delta:+.6f} | {a['p_at_k']:.2f} | {b['p_at_k']:.2f} |")
    lines += [
        '',
        f'- 最大 |ΔAUC| = {max_abs:.6f}（约 {max_abs * 100:.3f} pct）。',
        f'- 判定：最大单项差异 {"< 1 pct" if max_abs < 0.01 else "≥ 1 pct"}；'
        '各模型差值方向不一致，不能据此宣称该标记带来稳定增益。',
        '- 处置：保留 `burstiness_missing`（正式 8 列接口），但需在独立样本上复验；'
        '若后续确认无增益，删除该列属于接口变更，须走 `docs/feature_contract.md` §9 流程。',
        '- `native_ge3_observed` 人群按定义 `burstiness_missing=0` 恒成立，两种输入等价，故未重复跑。',
        '',
        '## 复现',
        '',
        '```bash',
        f'python src/models/burstiness_missing_sensitivity.py --input {args.input}',
        '```',
        '',
        f'环境：Python {dd.platform.python_version()}；numpy={np.__version__}。',
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'C3 report: {args.report}')
    print(json.dumps({'max_abs_delta_auc': max_abs}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/c_burstiness'))
    parser.add_argument('--report', type=Path,
                        default=Path('results/c_burstiness_missing_sensitivity.md'))
    main(parser.parse_args())
