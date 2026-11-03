"""把各数据集的原始文件解析成统一评论级表。

每个数据集的文件形态都不一样（空格分隔无表头、逗号分隔带 BOM、zip 内多文件、
生成集与真实集分开存放），所以这里的原则是：**先探测，再映射，最后校验**。
探测失败的字段一律报错并给出实际表头，不猜。

各数据集的标签语义差异只在本文件出现一次：
- Yelp 系列：原始 1=推荐（真实）、-1 或 0=过滤（虚假）→ 统一翻转
- FraudSquad：真实基线 = 0；LLM 生成 = 1
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import (
    CANONICAL_COLUMNS,
    SchemaError,
    coerce_dtypes,
    label_from_yelp,
)

logger = logging.getLogger(__name__)

#: 列名别名表：不同镜像/再加工版本的叫法不同，统一映射到规范名。
#: 同一组内**顺序即优先级**（例如 FraudSquad 同时有 product 与 product_child，取前者）。
ALIASES: Dict[str, Tuple[str, ...]] = {
    "user_id": ("user_id", "userid", "user", "reviewer_id", "uid", "author", "reviewerid"),
    "item_id": ("prod_id", "product_id", "product", "parent_asin", "asin", "item_id", "business_id", "product_child", "shop_id", "gmap_id"),
    "rating": ("rating", "star", "stars", "overall", "score"),
    "label": ("label", "spam", "is_fake", "fake", "fake_review", "y"),
    "date": ("date", "time", "timestamp", "reviewtime", "review_date", "review_time", "unixreviewtime", "reviewdate"),
    "text": ("text", "review", "reviewtext", "review_text", "content", "review_body", "body", "comment"),
}

_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "\\n"}


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def _read_head(path: Path, n_chars: int = 1 << 16) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read(n_chars)


def sniff_separator(sample: str, candidates: Sequence[str] = (",", "\t", ";", "|")) -> str:
    """用 csv.Sniffer 判断分隔符，失败时退化为"在样本首行里出现最多的候选"。

    Sniffer 在只有一行、或文本里逗号极多时会给出误判，所以再做一次多数表决。
    """
    if not sample.strip():
        raise SchemaError("文件为空，无法判断分隔符")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
        return dialect.delimiter
    except csv.Error:
        pass
    first_line = sample.splitlines()[0]
    counts = {sep: first_line.count(sep) for sep in candidates}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return " "  # 空格分隔（YelpChi metadata.txt 就是这种）
    return best


def _header_matches_aliases(header: Sequence[str]) -> bool:
    known = {a for group in ALIASES.values() for a in group}
    hits = sum(1 for h in header if str(h).strip().lower() in known)
    return hits >= 2


def map_columns(raw_columns: Iterable[str]) -> Dict[str, str]:
    """把原始列名映射成规范列名；返回 {原始列名: 规范列名}。"""
    mapping: Dict[str, str] = {}
    lowered = {str(c).strip().lower(): c for c in raw_columns}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[lowered[alias]] = canonical
                break
    return mapping


def _parse_dates(series: pd.Series) -> pd.Series:
    """兼容 '2014-11-16'、'2014-11-16 08:30:00' 与 10 位/13 位 Unix 时间戳。"""
    numeric = pd.to_numeric(series, errors="coerce")
    looks_unix = numeric.notna() & (numeric > 1e9) & (numeric < 1e11)  # 2001-2286 年
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if looks_unix.any():
        out.loc[looks_unix] = pd.to_datetime(numeric[looks_unix], unit="s", errors="coerce", utc=True).dt.tz_localize(None)
    rest = ~looks_unix
    if rest.any():
        parsed = pd.to_datetime(series[rest], errors="coerce", utc=False, format="mixed")
        try:
            parsed = parsed.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        out.loc[rest] = parsed
    return out


def _clean_text(series: pd.Series) -> pd.Series:
    """去首尾空白、把缺失/占位符统一成空串；保留正文原样（供 TF-IDF 使用）。"""
    s = series.astype("object").where(series.notna(), "")
    s = s.map(lambda v: v if isinstance(v, str) else str(v))
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.str.lower().isin(_MISSING_TOKENS), "")


def _finalize(
    df: pd.DataFrame,
    dataset: str,
    *,
    label_source: str,
    is_injected: int = 0,
    text_available: bool = True,
    drop_na_date: bool = True,
) -> pd.DataFrame:
    """补齐 ID、标签、文本可用标记等派生列，输出规范 schema（未排序）。"""
    n = len(df)
    out = pd.DataFrame(
        {
            "dataset": dataset,
            "review_id": [f"{dataset}-{i:09d}" for i in range(n)],
            "user_id": df["user_id"].astype(str).str.strip(),
            "item_id": df["item_id"].astype(str).str.strip(),
            "rating": pd.to_numeric(df.get("rating"), errors="coerce").astype("float32"),
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "label": pd.to_numeric(df["label"], errors="coerce").astype("int8"),
            "label_source": label_source,
            "is_injected": np.int8(is_injected),
            "text_available": np.int8(1 if text_available else 0),
            "text": _clean_text(df["text"]) if text_available and "text" in df else "",
        }
    )
    out["global_user_id"] = out["dataset"] + ":" + out["user_id"]

    if drop_na_date:
        bad = out["date"].isna()
        if bad.any():
            logger.warning("%s: %d 行时间无法解析，已剔除（占 %.3f%%）", dataset, int(bad.sum()), 100 * bad.mean())
            out = out.loc[~bad].reset_index(drop=True)

    return coerce_dtypes(out)


# --------------------------------------------------------------------------
# 通用 CSV/TXT 读取（带分块，避免 400MB 文本一次性进内存）
# --------------------------------------------------------------------------

def read_tabular(
    path: Path,
    *,
    sep: Optional[str] = None,
    header: Optional[bool] = None,
    chunksize: int = 200_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """读取任意分隔文本，自动判断分隔符与是否有表头。"""
    sample = _read_head(path)
    if sep is None:
        sep = sniff_separator(sample)
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    fields = next(csv.reader(io.StringIO(first_line), delimiter=sep)) if sep != " " else first_line.split()
    if header is None:
        header = _header_matches_aliases(fields)

    read_kwargs = dict(
        sep=sep,
        header=0 if header else None,
        chunksize=chunksize,
        dtype=str,          # 先全按字符串读，避免大整数 uid 被转成 float 丢精度
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8-sig",
        engine="c" if sep in (",", "\t", ";", "|") else "python",
        quoting=csv.QUOTE_MINIMAL if sep == "," else 0,
    )
    if sep == " ":
        read_kwargs.update(sep=r"\s+", engine="python", quoting=0)

    frames: List[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, **read_kwargs):
        frames.append(chunk)
        total += len(chunk)
        if max_rows is not None and total >= max_rows:
            break
    if not frames:
        raise SchemaError(f"{path.name} 读取为空")
    df = pd.concat(frames, ignore_index=True)
    if not header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df if max_rows is None else df.head(max_rows)


# --------------------------------------------------------------------------
# Yelp 系列
# --------------------------------------------------------------------------

#: YelpChi metadata.txt 是无表头的空格分隔 5 列，顺序由 Rayana 数据集固定
_YELPCHI_FIXED_COLUMNS = ["user_id", "item_id", "rating", "label", "date"]


def read_yelpchi(path: Path) -> pd.DataFrame:
    """读取 YelpChi metadata（user / product / rating / label / date）。

    该版本不含评论文本，因此 text_available=0；文本版需向作者申请，见 provenance。
    """
    df = read_tabular(path, sep=None)
    if all(c.startswith("col_") for c in df.columns):
        if df.shape[1] != 5:
            raise SchemaError(f"YelpChi metadata 期望 5 列，实际 {df.shape[1]} 列: {list(df.columns)}")
        df.columns = _YELPCHI_FIXED_COLUMNS
    else:
        mapping = map_columns(df.columns)
        df = df.rename(columns=mapping)
        missing = {"user_id", "item_id", "rating", "label", "date"} - set(df.columns)
        if missing:
            raise SchemaError(f"YelpChi 缺少列 {sorted(missing)}；实际表头: {list(df.columns)}")

    df = df.assign(date=_parse_dates(df["date"]), label=label_from_yelp(df["label"]))
    return _finalize(df, "yelpchi", label_source="yelp_filter", text_available=False)


def read_yelp_family(path: Path, dataset: str, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """读取 YelpNYC / YelpZip 这类"带文本 + 带标签"的评论表。

    兼容：逗号/制表符分隔、有/无表头、带 BOM、rating 为 1.0 浮点、label 为 ±1。
    """
    df = read_tabular(path, max_rows=max_rows)
    mapping = map_columns(df.columns)
    df = df.rename(columns=mapping)
    missing = {"user_id", "item_id", "rating", "label", "date"} - set(df.columns)
    if missing:
        raise SchemaError(f"{dataset} 缺少列 {sorted(missing)}；实际表头: {list(df.columns)}")

    text_available = "text" in df.columns
    if not text_available:
        logger.warning("%s 未找到文本列，text_available 将记为 0", dataset)

    df = df.assign(date=_parse_dates(df["date"]), label=label_from_yelp(df["label"]))
    out = _finalize(df, dataset, label_source="yelp_filter", text_available=text_available)
    if not text_available:
        out["text"] = ""
    return out


# --------------------------------------------------------------------------
# FraudSquad（LLM 生成的水军评论 + Amazon 真实基线）
# --------------------------------------------------------------------------

def read_fraudsquad_amazon(real_csv: Path, generated: Dict[str, Path]) -> pd.DataFrame:
    """把真实基线 + 各模型生成的假评论拼成一份统一表。

    `generated` 形如 {"llama3_8b": path, "qwen_72b": path, ...}，键名会写进
    `label_source` 之外的辅助统计（真实的模型名以文件名记入 manifest）。
    生成评论的 label 恒为 1，真实基线 label 恒为 0——这是"构造即标签"，不是模型预测。
    """
    frames: List[pd.DataFrame] = []

    real = read_tabular(real_csv)
    real_map = map_columns(real.columns)
    real = real.rename(columns=real_map)
    need = {"user_id", "item_id", "rating", "date", "text"}
    missing = need - set(real.columns)
    if missing:
        raise SchemaError(
            f"FraudSquad 真实基线缺少列 {sorted(missing)}；实际表头: {list(real.columns)}。"
            "若是全新版本的数据，请先在此处补充字段映射。"
        )
    real = real.assign(label=0, date=_parse_dates(real["date"]))
    frames.append(_finalize(real, "fraudsquad_amazon", label_source="real_baseline"))

    total_gen = 0
    for model_name, gpath in generated.items():
        gen = read_tabular(gpath)
        gen = gen.rename(columns=map_columns(gen.columns))
        missing = {"user_id", "item_id", "rating", "date", "text"} - set(gen.columns)
        if missing:
            raise SchemaError(f"FraudSquad 生成集 {model_name} 缺少列 {sorted(missing)}；实际表头: {list(gen.columns)}")
        gen = gen.assign(label=1, date=_parse_dates(gen["date"]))
        gen_frame = _finalize(gen, "fraudsquad_amazon", label_source="llm_generated")
        logger.info("FraudSquad 生成集 %s: %d 条", model_name, len(gen_frame))
        total_gen += len(gen_frame)
        frames.append(gen_frame)

    out = pd.concat(frames, ignore_index=True)
    # 重新编号，保证 review_id 在全数据集内唯一
    out["review_id"] = [f"fraudsquad_amazon-{i:09d}" for i in range(len(out))]
    logger.info("FraudSquad 合计 %d 条（真实 %d / 生成 %d）", len(out), len(out) - total_gen, total_gen)
    return coerce_dtypes(out)


def unzip_if_needed(zip_path: Path, target_dir: Path, *, expect: Optional[str] = None) -> Path:
    """解压 zip 并返回解压出的目标文件（用于 amazon_2022_5core.csv.zip 这类单文件包）。"""
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if expect:
            hit = next((n for n in names if n.endswith(expect)), None)
            if hit is None:
                raise SchemaError(f"{zip_path.name} 中未找到 {expect}，实际内容: {names[:10]}")
            dest = target_dir / Path(hit).name
            if not dest.exists() or dest.stat().st_size != zf.getinfo(hit).file_size:
                zf.extract(hit, target_dir)
                # 若 zip 内带目录层级，挪到 target_dir 根下
                extracted = target_dir / hit
                if extracted != dest:
                    extracted.replace(dest)
            return dest
        zf.extractall(target_dir)
    return target_dir
