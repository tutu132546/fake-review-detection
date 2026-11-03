# -*- coding: utf-8 -*-
"""B2：FraudSquad×3 拆分预置校验（每账号生成评论=2 / 全 5 星 / 5 天窗口 / 三份无交集）。

数据未到位时输出 BLOCKED 并给出缺失路径，不生成任何替代数据。

用法：
  python src/features/qa_fraudsquad.py
  python src/features/qa_fraudsquad.py --input-dir data/processed/unified \
      --report results/qa_fraudsquad_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from features.dataset_config import UNIFIED_DIR

RESULTS = _SRC.parent / 'results'
DEFAULT_IDS = ['fraudsquad_llama3', 'fraudsquad_qwen2', 'fraudsquad_qwen_dsr1']
EXPECTED_PER_ACCOUNT = 2
WINDOW_DAYS = 5


def _check_one(df, dataset_id):
    errors = []
    if not {'user_id', 'item_id', 'rating', 'date', 'label'} <= set(df.columns):
        return [f'{dataset_id}: 缺少统一 schema 字段'], {}
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    gen = df[df['label'] == 1]
    real = df[df['label'] == 0]

    counts = gen.groupby('user_id').size()
    bad_counts = counts[counts != EXPECTED_PER_ACCOUNT]
    if len(bad_counts):
        errors.append(f'{dataset_id}: {len(bad_counts)} 个账号生成评论数 != {EXPECTED_PER_ACCOUNT}'
                      f'（分布 {counts.value_counts().sort_index().to_dict()}）')

    if len(gen) and not (gen['rating'] == 5).all():
        errors.append(f'{dataset_id}: 生成评论非 5 星 {int((gen["rating"] != 5).sum())} 行')

    if len(real):
        first_real = real.groupby('item_id')['date'].min()
        merged = gen.join(first_real.rename('first_real'), on='item_id')
        missing_first = int(merged['first_real'].isna().sum())
        late = merged['date'] > merged['first_real'] + pd.Timedelta(days=WINDOW_DAYS)
        if missing_first:
            errors.append(f'{dataset_id}: {missing_first} 行生成评论的商品无真实评论，无法验证窗口')
        if int(late.sum()):
            errors.append(f'{dataset_id}: {int(late.sum())} 行生成评论晚于商品首评 +{WINDOW_DAYS} 天')
    else:
        errors.append(f'{dataset_id}: 缺少真实评论（label=0）')
    stats = {'rows': int(len(df)), 'accounts': int(df['user_id'].nunique()),
             'generated': int(len(gen)), 'count_dist': {str(k): int(v)
                                                        for k, v in counts.value_counts().sort_index().items()}}
    return errors, stats


def run(input_dir=None, ids=None, report_path=None):
    input_dir = Path(input_dir) if input_dir else UNIFIED_DIR
    ids = ids or DEFAULT_IDS
    report_path = Path(report_path) if report_path else RESULTS / 'qa_fraudsquad_report.md'
    lines = ['# FraudSquad×3 拆分验收报告', '']
    payload = {'status': 'blocked', 'datasets': {}, 'errors': []}
    frames, missing = {}, []
    for did in ids:
        path = input_dir / f'{did}.parquet'
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_parquet(path)
        errs, stats = _check_one(df, did)
        payload['datasets'][did] = stats
        payload['errors'] += errs
        frames[did] = set(df['user_id'].astype(str))
        lines += [f'## {did}', f'- 行数 {stats["rows"]} / 账号 {stats["accounts"]} / '
                  f'生成评论 {stats["generated"]}', f'- 每账号生成数分布 {stats["count_dist"]}']
        lines += [f'- [FAIL] {e}' for e in errs] or ['- [PASS] 四项检查通过']

    if missing:
        payload['status'] = 'blocked'
        payload['missing'] = missing
        lines += ['', '**状态：BLOCKED** —— 以下文件缺失，未生成任何表：']
        lines += [f'- `{m}`' for m in missing]
    elif frames:
        # 三份账号集合必须互不重叠
        overlap = set()
        keys = list(frames)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                overlap |= frames[keys[i]] & frames[keys[j]]
        if overlap:
            payload['errors'].append(f'三份数据账号集合重叠 {len(overlap)} 个')
            lines += ['', f'- [FAIL] 三份账号集合重叠 {len(overlap)} 个']
        else:
            lines += ['', '- [PASS] 三份账号集合无交集']
        payload['status'] = 'pass' if not payload['errors'] else 'fail'

    lines += ['', '> 验收口径：每账号恰好 2 条生成评论、生成评论全 5 星、'
              '日期在商品首条真实评论后 5 天内、三份账号集合互斥。']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description='FraudSquad×3 拆分验收')
    parser.add_argument('--input-dir', default=None)
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    run(args.input_dir, args.datasets, args.report)


if __name__ == '__main__':
    main()
