# -*- coding: utf-8 -*-
"""B 成员批量入口：一次构建 5 张特征表（+ YelpNYC 缺失占位），并支持 manifest 复现校验。

用法：
  python src/features/build_all.py                          # 构建全部可得数据集
  python src/features/build_all.py --datasets yelpchi
  python src/features/build_all.py --verify-manifest        # 重建后与已有 manifest 比对
缺失数据不造假表：写 status=blocked_by_data / missing 的占位 JSON 并说明原因。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from features.build_feature_table import build_dataset, _write_missing
from features.dataset_config import DATASET_IDS, ensure_dirs, get_spec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def _write_blocked(spec, error):
    ensure_dirs()
    payload = {
        'dataset_id': spec.dataset_id,
        'status': 'blocked_by_data',
        'reason': str(error),
        'expected_input': str(spec.input_path) if spec.input_path else None,
        'text_available': int(spec.text_available),
        # B1：方向未声明时即使数据缺失也留痕，数据到位后 build_dataset 会写完整 manifest
        'label_direction': ('declared_in_spec' if spec.label_direction_verified
                            else 'label_direction_unverified'),
        'fake_raw_values': list(spec.fake_raw_values) if spec.fake_raw_values else None,
        'real_raw_values': list(spec.real_raw_values) if spec.real_raw_values else None,
        'expected_fake_rate': spec.expected_fake_rate,
    }
    spec.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec.manifest_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.warning(f"{spec.dataset_id} blocked_by_data：{error}")


def build_all(dataset_ids, verify=False):
    results = {}
    for did in dataset_ids:
        spec = get_spec(did)
        old = None
        if verify and spec.manifest_path.exists():
            with open(spec.manifest_path, encoding='utf-8') as f:
                old = json.load(f)
        try:
            if spec.mode == 'missing':
                _write_missing(spec)
                results[did] = 'missing'
                continue
            build_dataset(spec)
        except FileNotFoundError as e:
            _write_blocked(spec, e)
            results[did] = 'blocked_by_data'
            continue
        if verify:
            results[did] = _compare_manifest(spec, old)
        else:
            results[did] = 'built'
    return results


def _compare_manifest(spec, old):
    if old is None:
        return 'FAIL(no previous manifest)'
    with open(spec.manifest_path, encoding='utf-8') as f:
        new = json.load(f)
    keys = ('rows', 'accounts', 'label_counts', 'cohort_counts', 'missing_rates')
    diffs = [k for k in keys if old.get(k) != new.get(k)]
    if old.get('table', {}).get('sha256') != new.get('table', {}).get('sha256'):
        diffs.append('table_sha256')
    return 'PASS' if not diffs else f'FAIL({", ".join(diffs)})'


def main():
    parser = argparse.ArgumentParser(description="批量构建特征表并校验 manifest")
    parser.add_argument('--datasets', nargs='+', default=DATASET_IDS,
                        help=f"默认全部：{', '.join(DATASET_IDS)}")
    parser.add_argument('--verify-manifest', action='store_true',
                        help="重建并与已有 manifest 比对 sha256 / 行数 / 标签分布")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(DATASET_IDS)
    if unknown:
        parser.error(f"未知 dataset：{sorted(unknown)}")
    results = build_all(args.datasets, verify=args.verify_manifest)
    logging.info("汇总：" + ", ".join(f"{k}={v}" for k, v in results.items()))
    if any(str(v).startswith('FAIL') for v in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
