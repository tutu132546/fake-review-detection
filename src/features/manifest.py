# -*- coding: utf-8 -*-
"""特征表 sidecar manifest 与 QA 报告生成（方案 F.5 / F.9）。

每张特征表配一份 JSON manifest：任何人拿到表都能反查数据文件、sha256、commit、scaler、
标签分布、缺失率与自检结果；不提交大表本身，只提交 manifest。
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PROJECT_ROOT

FEATURE_VERSION = '1.0.0'


def sha256_file(path, chunk_size=1 << 20):
    """文件 SHA256；文件不存在时返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    """当前 HEAD 的 40 位 commit；仓库不可用时返回 'unknown'。"""
    try:
        out = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return 'unknown'


def env_info():
    import sklearn
    return {
        'python': platform.python_version(),
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit_learn': sklearn.__version__,
        'PYTHONHASHSEED': '0',
    }


def _rel(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(spec, table, scaler, missing_rates, source_files, *,
                   fit_population, label_encoding_check='passed',
                   qa=None, known_issues=None, unified_manifest_sha256=None,
                   extra=None):
    """组装 manifest 字典（字段对齐方案 F.5）。"""
    label_counts = {str(k): int(v) for k, v in table['account_is_fake'].value_counts().sort_index().items()}
    cohort_counts = {str(k): int(v) for k, v in table['cohort'].value_counts().items()}
    manifest = {
        'feature_version': FEATURE_VERSION,
        'dataset_id': spec.dataset_id,
        'built_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'git_commit': git_commit(),
        'unified_manifest_sha256': unified_manifest_sha256,
        'source_files': source_files,
        'rows': int(len(table)),
        'accounts': int(table['global_user_id'].nunique()),
        'label_counts': label_counts,
        'cohort_counts': cohort_counts,
        'text_available': int(spec.text_available),
        'feature_list': list(scaler['features'].keys()),
        'scaler': {
            'path': _rel(spec.scaler_path),
            'sha256': sha256_file(spec.scaler_path),
            'fit_population': fit_population,
        },
        'missing_rates': {k: round(float(v), 4) for k, v in missing_rates.items()},
        'label_encoding_check': label_encoding_check,
        'qa': qa or {},
        'known_issues': known_issues or [],
        'env': env_info(),
        'python': sys.executable,
        'table': {'path': _rel(spec.output_path), 'sha256': sha256_file(spec.output_path)},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def write_qa_report(path, manifest, spec):
    """人工可读的 QA 报告（F.1 / F.9 第 8 项）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    qa = manifest.get('qa', {})
    lines = [
        f"# 特征表 QA：{spec.dataset_id}",
        '',
        f"- 数据版本：feature_version={manifest['feature_version']}，"
        f"built_at={manifest['built_at']}，commit={manifest['git_commit']}",
        f"- 行数：{manifest['rows']}；账号数：{manifest['accounts']}；"
        f"标签分布：{manifest['label_counts']}；人群分布：{manifest['cohort_counts']}",
        f"- 拟合人群：{manifest['scaler']['fit_population']}；scaler：{manifest['scaler']['path']}",
        f"- 文本可用：{manifest['text_available']}（评估时据此屏蔽文本方法）",
        f"- 标签编码检查：{manifest['label_encoding_check']}",
        '',
        '## 缺失率',
        '',
        '| 项 | 比例 |',
        '|---|---:|',
    ]
    for k, v in manifest['missing_rates'].items():
        lines.append(f"| {k} | {v:.2%} |")
    lines += [
        '',
        '## 自检结果',
        '',
        '| 检查 | 结果 |',
        '|---|---|',
    ]
    for k, v in qa.items():
        lines.append(f"| {k} | {v} |")
    lines += ['', '## 已知问题', '']
    if manifest['known_issues']:
        lines += [f"- {x}" for x in manifest['known_issues']]
    else:
        lines.append('- 无')
    lines += ['', '## 数据来源（sha256 可追溯）', '']
    for src in manifest['source_files']:
        lines.append(f"- `{src['path']}`（rows={src.get('rows')}, sha256={src.get('sha256')}）")
    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path
