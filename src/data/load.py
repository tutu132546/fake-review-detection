"""下游（特征 / 检测 / 图 / 评估）读取统一数据的公共入口。

用法::

    from data import load_dataset, load_all, load_accounts
    yelpchi = load_dataset("yelpchi")             # 只读单个数据集（评论级）
    both    = load_all(["yelpchi", "yelpzip"])    # 同 schema 直接 concat
    acc     = load_accounts("fraudsquad_llama3")  # 账号级视图

约定：返回值一定是 `schema.CANONICAL_COLUMNS` 的列序，按 (date, review_id) 升序，
      只读不写。要改数据请改构建脚本后重跑，不要在内存里就地修补。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from .paths import ACCOUNTS_DIR, UNIFIED_DIR
from .schema import ACCOUNT_COLUMNS, CANONICAL_COLUMNS, DATASET_SPECS

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def manifest_path() -> Path:
    return UNIFIED_DIR / MANIFEST_NAME


def read_manifest() -> dict:
    """读取上一次构建的 manifest；不存在时返回空 dict（不抛错，方便首次运行）。"""
    p = manifest_path()
    if not p.exists():
        logger.warning("尚未构建统一数据（%s 不存在），请先运行 python src/build_unified_data.py build", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def available_datasets() -> List[str]:
    """磁盘上真正存在 parquet 的数据集名。"""
    if not UNIFIED_DIR.exists():
        return []
    return sorted(p.stem for p in UNIFIED_DIR.glob("*.parquet") if p.stem != "all_reviews")


def list_datasets() -> pd.DataFrame:
    """列出数据集及其规模/标签分布，供队友一眼确认"我该读哪个"。

    `role` 列区分用途：`paper` = 论文口径的独立数据集；`view` = 只作总览的合并视图
    （不要拿去做对比实验）；`self_built` = 本项目自建的注入数据集。
    """
    from .schema import PUBLIC_DATASETS, VIEW_ONLY_DATASETS

    manifest = read_manifest()
    rows = []
    for name, spec in DATASET_SPECS.items():
        path = UNIFIED_DIR / f"{name}.parquet"
        info = manifest.get("datasets", {}).get(name, {})
        stats = info.get("stats") or {}
        if name in VIEW_ONLY_DATASETS:
            role = "view"
        elif name in PUBLIC_DATASETS:
            role = "paper"
        else:
            role = "self_built"
        rows.append(
            {
                "dataset": name,
                "display_name": spec.display_name,
                "role": role,
                "built": path.exists(),
                "rows": stats.get("rows"),
                "users": stats.get("users"),
                "items": stats.get("items"),
                "fake_ratio": stats.get("fake_ratio"),
                "has_text": spec.has_text,
                "label_source": spec.label_semantics,
                "status": info.get("status", "missing"),
            }
        )
    return pd.DataFrame(rows)


def _resolve_path(name: str) -> Path:
    if name not in DATASET_SPECS:
        raise KeyError(f"未知数据集 {name}；可选: {sorted(DATASET_SPECS)}")
    path = UNIFIED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{name} 的统一表不存在：{path}\n"
            "请先运行 `python src/build_unified_data.py build`（或 `--dataset " + name + "`）。"
        )
    return path


def load_dataset(
    name: str,
    *,
    columns: Optional[Sequence[str]] = None,
    with_text: bool = True,
) -> pd.DataFrame:
    """读取单个数据集的统一表。

    `with_text=False` 时丢弃 `text` 列，可显著降低内存占用（400MB 级文本可省下数 GB）。
    """
    path = _resolve_path(name)
    cols = list(columns) if columns else list(CANONICAL_COLUMNS)
    if not with_text and columns is None:
        cols = [c for c in cols if c != "text"]
    df = pd.read_parquet(path, columns=cols)
    return df.sort_values(["date", "review_id"], kind="mergesort").reset_index(drop=True)


def load_all(
    datasets: Optional[Iterable[str]] = None,
    *,
    with_text: bool = True,
) -> pd.DataFrame:
    """把多个数据集拼成一张表；`dataset` 列保证来源可辨。"""
    names = list(datasets) if datasets is not None else available_datasets()
    if not names:
        raise FileNotFoundError("没有可用的统一表，请先构建")
    frames = [load_dataset(n, with_text=with_text) for n in names]
    out = pd.concat(frames, ignore_index=True)
    logger.info("合并 %d 个数据集，共 %d 行", len(frames), len(out))
    return out


def iter_reviews(name: str, *, batch_size: int = 100_000, with_text: bool = True) -> Iterator[pd.DataFrame]:
    """按行批读取（parquet 列式存储支持），用于 60 万行级别不想全量进内存的场景。"""
    import pyarrow.parquet as pq

    path = _resolve_path(name)
    cols = [c for c in CANONICAL_COLUMNS if with_text or c != "text"]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield batch.to_pandas()


# --------------------------------------------------------------------------
# 账号级视图
# --------------------------------------------------------------------------

def load_accounts(name: str) -> pd.DataFrame:
    """读取账号级表（`build_unified_data.py accounts` 产出）。

    列见 `schema.ACCOUNT_COLUMNS`。`account_label=1` 表示该账号是水军/被接管账号，
    但不是"它所有评论都是假的"——账号级与评论级是两个任务，不要混用标签。
    """
    path = ACCOUNTS_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{name} 的账号级表不存在：{path}\n"
            "请先运行 `python src/build_unified_data.py accounts`（需先 build 评论级表）。"
        )
    df = pd.read_parquet(path)
    return df[[c for c in ACCOUNT_COLUMNS if c in df.columns]]


def list_accounts() -> pd.DataFrame:
    """账号级表的规模与正例比例一览。"""
    manifest = read_manifest()
    rows = (manifest.get("accounts") or [])
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["dataset", "accounts", "positive_accounts"])
