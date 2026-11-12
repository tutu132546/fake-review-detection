# -*- coding: utf-8 -*-
"""B3：数据集行数三段审计（原始物理行数 / 解析后行数 / 去重后行数）。

用途：定位"官方行数 vs 本地行数"差异属于「文件本身少」还是「读取/解析丢行」。
默认审计 YelpZip；也可用 --raw 指向任意 CSV 做自测（如 YelpChi）。

用法：
  python src/features/audit_yelpzip_rows.py
  python src/features/audit_yelpzip_rows.py --raw data/raw/YelpChi_reviews.csv --official 67395 \
      --report results/yelpchi_row_audit_selftest.md
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
from features.manifest import sha256_file

RESULTS = _SRC.parent / 'results'
OFFICIAL_YELPZIP_ROWS = 608598   # 叶歆澜提供的官方口径；未确认前仅作对照


def _physical_rows(path):
    """文件物理数据行数：CSV 按行计（减表头），parquet 读元数据。"""
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    with open(path, 'rb') as f:
        return max(sum(1 for _ in f) - 1, 0)


def _parsed_rows(path):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == '.parquet' else \
        pd.read_csv(path, low_memory=False)
    return df


def audit(raw_path=None, unified_path=None, official_rows=None, report_path=None):
    raw_path = Path(raw_path) if raw_path else None
    unified_path = Path(unified_path) if unified_path else UNIFIED_DIR / 'yelpzip.parquet'
    unified_path = Path(unified_path)
    report_path = Path(report_path) if report_path else RESULTS / 'yelpzip_row_audit.md'

    lines = ['# 行数三段审计报告', '']
    payload = {'status': 'blocked', 'reason': '原始文件与统一表均缺失'}

    target = raw_path if raw_path and Path(raw_path).exists() else unified_path
    if not Path(target).exists():
        lines += ['**状态：BLOCKED** —— 未找到可审计文件。',
                  '', f'- 期望原始文件：{raw_path or "data/raw/yelpzip.csv"}',
                  f'- 期望统一文件：{unified_path}',
                  '- 需要：YelpZip 原始文件 + 官方行数口径（责任人：叶歆澜）',
                  '- 脚本自测：`python src/features/audit_yelpzip_rows.py '
                  '--raw data/raw/YelpChi_reviews.csv --official 67395`']
        _write(report_path, lines)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    target = Path(target)
    stage1 = _physical_rows(target)
    df = _parsed_rows(target)
    stage2 = int(len(df))
    dedup_cols = [c for c in ('user_id', 'item_id', 'rating', 'date') if c in df.columns]
    stage3 = int(len(df.drop_duplicates(subset=dedup_cols))) if dedup_cols else stage2
    payload = {
        'status': 'ok',
        'file': str(target),
        'sha256': sha256_file(target),
        'physical_rows': stage1,
        'parsed_rows': stage2,
        'dedup_rows': stage3,
        'parse_loss': stage1 - stage2,
        'duplicate_rows': stage2 - stage3,
        'official_rows': official_rows,
        'official_diff': (official_rows - stage1) if official_rows else None,
    }
    if official_rows:
        verdict = ('文件本身少行（物理行数即低于官方口径，属来源问题）'
                   if stage1 < official_rows else '物理行数达标，差异来自解析/去重环节')
        payload['verdict'] = verdict
    lines += [f'- 文件：`{target}`', f'- sha256：`{payload["sha256"]}`', '',
              '| 阶段 | 行数 | 与上一阶段差值 |', '|---|---:|---:|',
              f'| 1 原始物理行数 | {stage1} | - |',
              f'| 2 解析后行数 | {stage2} | {payload["parse_loss"]} |',
              f'| 3 去重后行数 | {stage3} | {payload["duplicate_rows"]} |', '']
    if official_rows:
        lines += [f'官方口径：{official_rows} 行；差值 {payload["official_diff"]}。',
                  f'判定：{payload["verdict"]}。']
    lines += ['', '> 说明：物理行数按 CSV 行计（减表头）/ parquet 元数据；'
              '解析差值>0 说明存在坏行被丢弃，需回查 parser。']
    _write(report_path, lines)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='数据集行数三段审计')
    parser.add_argument('--raw', default=None, help='原始文件（CSV/parquet）')
    parser.add_argument('--unified', default=None, help='统一表 parquet（默认 yelpzip）')
    parser.add_argument('--official', type=int, default=OFFICIAL_YELPZIP_ROWS,
                        help='官方行数口径（默认 YelpZip 608,598）')
    parser.add_argument('--report', default=None, help='报告输出路径')
    args = parser.parse_args()
    audit(args.raw, args.unified, args.official, args.report)


if __name__ == '__main__':
    main()
