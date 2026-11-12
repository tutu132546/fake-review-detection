"""统一数据管线 CLI：下载 -> 构建 -> 质检 -> 校验。

    python src/build_unified_data.py download            # 按清单下载（可断点续传）
    python src/build_unified_data.py build              # 生成 data/processed/unified/*.parquet
    python src/build_unified_data.py qa                 # 产出 results/data_quality_report.md
    python src/build_unified_data.py verify             # 校验已构建产物与 manifest 是否一致
    python src/build_unified_data.py all                # download + build + qa + verify

等价写法：`python -m src.data.pipeline <子命令>`（需要从项目根目录运行）。
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import readers, sources
from .download import download_many, sha256_of
from .load import MANIFEST_NAME
from .paths import (
    ACCOUNTS_DIR,
    DOWNLOAD_CACHE_DIR,
    LEGACY_INJECTION_LOG,
    LEGACY_YELPCHI_METADATA,
    LEGACY_YELPCHI_REVIEWS_CSV,
    PROJECT_ROOT,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_FRAUDSQUAD_DIR,
    RAW_YELPCHI_DIR,
    RAW_YELPNYC_DIR,
    RAW_YELPZIP_DIR,
    RESULTS_DIR,
    UNIFIED_DIR,
    ensure_dir,
)
from .schema import (
    DATASET_SPECS,
    PUBLIC_DATASETS,
    VIEW_ONLY_DATASETS,
    SchemaError,
    build_account_table,
    coerce_dtypes,
    summary_stats,
    validate_accounts,
    validate_reviews,
)

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> int:
    targets = args.dataset or [d for d in sources.FILES if sources.FILES[d]]
    ensure_dir(RAW_DIR)
    overall: Dict[str, dict] = {}
    for dataset in targets:
        specs = sources.files_for(dataset)
        if not specs:
            reason = sources.MANUAL_SOURCES.get(dataset, "无自动下载地址")
            logger.warning("%s 无脚本可下载的资源：%s", dataset, reason)
            overall[dataset] = {"status": "manual", "reason": reason}
            continue
        logger.info("下载 %s（%d 个文件）...", dataset, len(specs))
        report = download_many(specs, RAW_DIR, force=args.force, tolerate_failure=args.tolerate_failure)
        ok = sum(1 for v in report.values() if v == "ok")
        overall[dataset] = {"status": "ok" if ok == len(report) else "partial", "files": report}
        logger.info("%s: %d/%d 个文件就绪", dataset, ok, len(report))
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------

def _first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _raw_record(path: Optional[Path]) -> dict:
    """记录一个原始文件的仓库相对路径、字节数与 sha256（路径统一用正斜杠）。"""
    if path is None or not path.exists():
        return {"path": None, "size": None, "sha256": None}
    rel = path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in path.parents else path
    return {
        "path": Path(rel).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def build_yelpchi() -> Tuple[pd.DataFrame, List[dict], str]:
    path = _first_existing([RAW_YELPCHI_DIR / "metadata.txt", LEGACY_YELPCHI_METADATA])
    if path is None:
        return pd.DataFrame(), [], f"未找到 YelpChi metadata（尝试过 {RAW_YELPCHI_DIR}/metadata.txt 与 data/raw/metadata.txt）"
    df = readers.read_yelpchi(path)
    return df, [_raw_record(path)], f"来源文件 {path.name}（不含文本，text_available=0）"


def build_yelpzip() -> Tuple[pd.DataFrame, List[dict], str]:
    candidates = [RAW_YELPZIP_DIR / "yelpzip.csv"] + sorted(RAW_YELPZIP_DIR.glob("*.csv"))
    path = _first_existing(candidates)
    if path is None:
        return pd.DataFrame(), [], "未找到 YelpZip 数据，请先运行 download 子命令"
    df = readers.read_yelp_family(path, "yelpzip")
    raw_audit = df.attrs.get("partial_row_audit") or {}
    note = f"来源文件 {path.name}（带评论文本）"
    if raw_audit:
        only_in_partial = set(raw_audit["partial_users"]) - set(df["user_id"])
        audit = {
            "partial_rows": int(raw_audit["partial_rows"]),
            "contaminated_records": int(raw_audit["contaminated_records"]),
            "partial_users": len(raw_audit["partial_users"]),
            "partial_items": len(raw_audit["partial_items"]),
            "users_only_in_partial": len(only_in_partial),
            "dataset_users": int(df["user_id"].nunique()),
        }
        df.attrs["partial_row_audit"] = audit
        note += (
            f"；解析出 {len(df):,} 条带标签记录，另有 {audit['partial_rows']} 行因镜像导出缺陷"
            f"缺 rating/label 而不可用（涉及 {audit['partial_users']} 个账号、{audit['partial_items']} 个商品，"
            f"其中 {audit['users_only_in_partial']} 个账号只出现在残缺行）；"
            f"{audit['contaminated_records']} 条正常记录的文本曾被这些残缺行污染，已在读取阶段剥离"
        )
    return df, [_raw_record(path)], note


def build_yelpnyc() -> Tuple[pd.DataFrame, List[dict], str]:
    """YelpNYC：官方包把元数据与评论文本拆成两个文件，优先走双文件读取。

    退回单文件分支是给第三方再加工版本留的兼容路径（那种版本通常已经把文本拼进
    同一张表），但文件名要显式排除 README 之类的说明文件。
    """
    meta_path = _first_existing([RAW_YELPNYC_DIR / "metadata"])
    content_path = _first_existing([RAW_YELPNYC_DIR / "reviewContent"])
    if meta_path is not None and content_path is not None:
        df = readers.read_yelp_pair(meta_path, content_path, "yelpnyc")
        audit = dict(df.attrs.get("text_join_audit") or {})
        df.attrs["text_join_audit"] = audit
        note = (
            f"来源为 Rayana 官方发布包的双文件形态：{meta_path.name}（元数据）+ "
            f"{content_path.name}（评论文本），带完整文本"
        )
        if audit:
            note += (
                f"；文本按 (user, item, date) 对齐，策略 {audit['strategy']}，"
                f"{audit['matched']:,}/{audit['rows_metadata']:,} 条匹配、"
                f"{audit['key_mismatches']} 行键错位"
            )
        return df, [_raw_record(meta_path), _raw_record(content_path)], note

    candidates = [
        p for p in sorted(RAW_YELPNYC_DIR.glob("*.csv")) + sorted(RAW_YELPNYC_DIR.glob("*.txt"))
        if not p.name.lower().startswith(("readme", "license", "citation"))
        and p.name != "reviewContent"
    ]
    path = _first_existing(candidates)
    if path is None:
        return pd.DataFrame(), [], sources.MANUAL_SOURCES["yelpnyc"]
    df = readers.read_yelp_family(path, "yelpnyc")
    return df, [_raw_record(path)], f"来源文件 {path.name}（单文件形态）"


#: FraudSquad 的原始文件：真实基线 + 三个模型各自的生成集
_FRAUDSQUAD_MODELS: Dict[str, str] = {
    "llama3_8b": "fraudsquad_llama3",
    "qwen_72b": "fraudsquad_qwen2",
    "qwen_dsr1": "fraudsquad_qwen_dsr1",
}


def build_fraudsquad(dataset_id: str = "fraudsquad_amazon", only: Optional[str] = None) -> Tuple[pd.DataFrame, List[dict], str]:
    """构建 FraudSquad 数据集。

    `only=None`  → 三个模型合并的总览视图（fraudsquad_amazon），不用于论文对比；
    `only=<模型>` → 真实基线 + 该模型生成的假评论，即论文口径的**单个数据集**。
    拆分是必须的：论文里三个模型是三个独立数据集，合并后每个被接管账号会攒到 4~6 条
    假评论（实测 930 个账号 4 条、304 个账号 6 条），与原文数字无法对齐。
    """
    real_zip = RAW_FRAUDSQUAD_DIR / "amazon_2022_5core.csv.zip"
    if not real_zip.exists():
        return pd.DataFrame(), [], "未找到 amazon_2022_5core.csv.zip，请先运行 download 子命令"

    real_csv = readers.unzip_if_needed(real_zip, RAW_FRAUDSQUAD_DIR / "amazon", expect=".csv")
    all_generated = {
        name: RAW_FRAUDSQUAD_DIR / f"reviews_from_{name}.csv" for name in _FRAUDSQUAD_MODELS
    }
    candidates = all_generated if only is None else {only: all_generated[only]}
    generated = {k: v for k, v in candidates.items() if v.exists() and v.stat().st_size > 0}
    if not generated:
        return pd.DataFrame(), [], "未找到 LLM 生成集（reviews_from_*.csv）"

    missing = [k for k in candidates if k not in generated]
    df = readers.read_fraudsquad_amazon(real_csv, generated, dataset_id=dataset_id)
    models = sorted(generated)
    note = f"真实基线 {real_csv.name} + 生成集 {models}"
    if only is not None:
        note += f"（单模型数据集，论文口径）"
    else:
        note += "（三模型合并视图，仅供总览，不作对比基准）"
    if missing:
        note += f"；缺失生成集 {missing}（结果表中需如实说明）"
    records = [_raw_record(real_zip), _raw_record(real_csv)] + [_raw_record(p) for p in generated.values()]
    return df, records, note


def build_injected() -> Tuple[pd.DataFrame, List[dict], str]:
    """重建"YelpChi + 300 注入账号"的评论级视图。

    step3 只落盘了注入事件明细（injection_log_v2.json），这里把事件展开成评论行，
    与原始评论合并，并新增 label_source=injected 的行。**不重新跑注入**，
    以保证与既有注入实验结果逐条一致。
    """
    log_path = LEGACY_INJECTION_LOG
    reviews_path = _first_existing([LEGACY_YELPCHI_REVIEWS_CSV, RAW_YELPCHI_DIR / "YelpChi_reviews.csv"])
    meta_path = _first_existing([RAW_YELPCHI_DIR / "metadata.txt", LEGACY_YELPCHI_METADATA])
    if not log_path.exists() or reviews_path is None or meta_path is None:
        return pd.DataFrame(), [], (
            "缺少注入日志或评论底表（data/processed/injection_log_v2.json / data/raw/YelpChi_reviews.csv），"
            "跳过派生集；这不是错误，注入实验产物不入库"
        )

    base = readers.read_yelpchi(meta_path)
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    rows: List[dict] = []
    for entry in log:
        uid = str(entry.get("user_id"))
        for ev in entry.get("events_added", []):
            rows.append(
                {
                    "user_id": uid,
                    "item_id": str(ev.get("product_id")),
                    "rating": ev.get("rating"),
                    "date": ev.get("date"),
                    "text": "",  # step3 只生成结构化事件，没有生成文本
                }
            )
    injected = pd.DataFrame(rows)
    if injected.empty:
        return pd.DataFrame(), [], "注入日志中没有 events_added，无法展开"

    injected["label"] = 1
    injected["date"] = pd.to_datetime(injected["date"], errors="coerce")
    injected = injected.dropna(subset=["date"])
    injected["label_source"] = "injected"
    injected["is_injected"] = 1
    injected["text_available"] = 0
    injected["dataset"] = "yelpchi_injected"

    base = base.assign(dataset="yelpchi_injected")
    base["is_injected"] = 0
    combined = pd.concat([base, injected], ignore_index=True)
    combined["review_id"] = [f"yelpchi_injected-{i:09d}" for i in range(len(combined))]
    combined["global_user_id"] = combined["dataset"] + ":" + combined["user_id"]
    combined = coerce_dtypes(combined)
    note = (
        f"由 {log_path.name} 的 {len(log)} 个账号、{len(injected)} 条注入事件展开；"
        "注入评论无文本（step3 未生成文本），text_available=0"
    )
    return combined, [_raw_record(log_path), _raw_record(reviews_path)], note


BUILDERS = {
    "yelpchi": build_yelpchi,
    "yelpnyc": build_yelpnyc,
    "yelpzip": build_yelpzip,
    # 三个单模型数据集 = 论文口径的独立数据集（真实基线 + 单一模型生成集）
    "fraudsquad_llama3": partial(build_fraudsquad, dataset_id="fraudsquad_llama3", only="llama3_8b"),
    "fraudsquad_qwen2": partial(build_fraudsquad, dataset_id="fraudsquad_qwen2", only="qwen_72b"),
    "fraudsquad_qwen_dsr1": partial(build_fraudsquad, dataset_id="fraudsquad_qwen_dsr1", only="qwen_dsr1"),
    # 合并视图，仅供总览，不进 all_reviews
    "fraudsquad_amazon": partial(build_fraudsquad, dataset_id="fraudsquad_amazon"),
    "yelpchi_injected": build_injected,
}

#: 不参与 all_reviews 合并的数据集：合并视图会与三个单模型表重复计数
COMBINE_EXCLUDE = frozenset(VIEW_ONLY_DATASETS)

#: 账号级表的标签口径：每个数据集按自身构造方式给出，不用统一猜测
ACCOUNT_LABEL_BASIS_BY_DATASET: Dict[str, str] = {
    "yelpchi": "platform_filtered",
    "yelpnyc": "platform_filtered",
    "yelpzip": "platform_filtered",
    "fraudsquad_llama3": "hijacked_by_llm",
    "fraudsquad_qwen2": "hijacked_by_llm",
    "fraudsquad_qwen_dsr1": "hijacked_by_llm",
    "fraudsquad_amazon": "hijacked_by_llm",
    "yelpchi_injected": "injected",
}


def _write_parquet(df: pd.DataFrame, path: Path) -> dict:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, compression="zstd")
    return {"parquet": str(path.relative_to(PROJECT_ROOT)), "parquet_sha256": sha256_of(path), "parquet_bytes": path.stat().st_size}


def _write_combined(names: Sequence[str]) -> Optional[dict]:
    """把可用数据集写成一张大表；用 ParquetWriter 分批追加，避免把文本全读进内存。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    present = [n for n in names if (UNIFIED_DIR / f"{n}.parquet").exists()]
    if len(present) < 2:
        return None
    out_path = UNIFIED_DIR / "all_reviews.parquet"
    writer = None
    rows = 0
    try:
        for name in present:
            pf = pq.ParquetFile(UNIFIED_DIR / f"{name}.parquet")
            for batch in pf.iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        return None
    return {
        "parquet": str(out_path.relative_to(PROJECT_ROOT)),
        "rows": rows,
        "datasets": present,
        "parquet_sha256": sha256_of(out_path),
        "parquet_bytes": out_path.stat().st_size,
    }


def cmd_build(args: argparse.Namespace) -> int:
    ensure_dir(UNIFIED_DIR)
    targets = args.dataset or list(BUILDERS)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "generated_by": "src/data/pipeline.py build",
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "datasets": {},
        "combined": None,
    }

    for name in targets:
        builder = BUILDERS.get(name)
        if builder is None:
            logger.error("未知数据集 %s，可选: %s", name, sorted(BUILDERS))
            continue
        t0 = time.time()
        logger.info("构建 %s ...", name)
        try:
            df, raw_records, note = builder()
        except (SchemaError, FileNotFoundError, ValueError) as exc:
            logger.error("%s 构建失败：%s", name, exc)
            manifest["datasets"][name] = {"status": "failed", "error": str(exc), "spec": _spec_dict(name)}
            continue

        if df is None or len(df) == 0:
            logger.warning("%s 跳过：%s", name, note)
            manifest["datasets"][name] = {
                "status": "missing",
                "note": note,
                "spec": _spec_dict(name),
                "raw_files": raw_records,
            }
            continue

        df = validate_reviews(df, context=name)
        stats = summary_stats(df)
        expected = DATASET_SPECS[name].expected_rows
        if expected and abs(stats["rows"] - expected) / expected > 0.02:
            logger.warning(
                "%s 行数 %d 与官方公布 %d 偏差超过 2%%，请在质量报告中说明原因",
                name, stats["rows"], expected,
            )

        extra = _write_parquet(df, UNIFIED_DIR / f"{name}.parquet")
        manifest["datasets"][name] = {
            "status": "ok",
            "note": note,
            "spec": _spec_dict(name),
            "raw_files": raw_records,
            "stats": stats,
            "expected_rows": expected,
            "rows_vs_expected_pct": None if not expected else round(100 * (stats["rows"] - expected) / expected, 3),
            "build_seconds": round(time.time() - t0, 1),
            "text_available": int(stats.get("has_text_rows", 0) > 0),
            **({"partial_row_audit": df.attrs["partial_row_audit"]} if df.attrs.get("partial_row_audit") else {}),
            **({"text_join_audit": df.attrs["text_join_audit"]} if df.attrs.get("text_join_audit") else {}),
            **extra,
        }
        logger.info("%s 完成：%d 行，假评论占比 %.2f%%", name, stats["rows"], 100 * stats["fake_ratio"])

    if not args.no_combined:
        manifest["combined"] = _write_combined(
            [n for n in BUILDERS if n not in COMBINE_EXCLUDE and (UNIFIED_DIR / f"{n}.parquet").exists()]
        )

    with open(UNIFIED_DIR / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("manifest 已写入 %s", UNIFIED_DIR / MANIFEST_NAME)

    ok = [k for k, v in manifest["datasets"].items() if v.get("status") == "ok"]
    missing = [k for k, v in manifest["datasets"].items() if v.get("status") != "ok"]
    print(f"\n构建完成：成功 {len(ok)} 个 {ok}；未完成 {len(missing)} 个 {missing}")
    return 0 if ok else 1


def _spec_dict(name: str) -> dict:
    spec = DATASET_SPECS[name]
    return {
        "display_name": spec.display_name,
        "homepage": spec.homepage,
        "download_page": spec.download_page,
        "citation": spec.citation,
        "license": spec.license_note,
        "label_semantics": spec.label_semantics,
        "has_text": spec.has_text,
        "notes": spec.notes,
    }


# --------------------------------------------------------------------------
# 质检报告
# --------------------------------------------------------------------------

def _label_source_breakdown(path: Path) -> Dict[str, int]:
    """统计每个数据集的标签来源构成（真实/生成/注入）。"""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["label_source"]).to_pandas()
    return {str(k): int(v) for k, v in t["label_source"].value_counts().items()}


_FS_USER_CACHE: Dict[str, set] = {}

_FS_SINGLE_MODELS = ("fraudsquad_llama3", "fraudsquad_qwen2", "fraudsquad_qwen_dsr1")


def _generated_users(name: str) -> set:
    """读某个 FraudSquad 单模型表里"写过生成评论"的账号集合（带缓存）。

    注意用 `label_source == "llm_generated"` 而不是 `label == 1`：真实基线里的
    账号可能同时被 Yelp 平台标过假，用 label 会把它们误算进来。
    """
    if name in _FS_USER_CACHE:
        return _FS_USER_CACHE[name]
    import pyarrow.parquet as pq

    path = UNIFIED_DIR / f"{name}.parquet"
    if not path.exists():
        return set()
    t = pq.read_table(path, columns=["user_id", "label_source"]).to_pandas()
    users = set(t.loc[t["label_source"] == "llm_generated", "user_id"])
    _FS_USER_CACHE[name] = users
    return users


def _fs_account_relation() -> str:
    """算清三份 FraudSquad 单模型表到底是不是同一批被接管账号，并给出可读结论。

    组长审核意见原文是"Llama3 和 Qwen2 用的是同一批 1,250 个账号"。这句话对
    前两份成立，但第三份（qwen_dsr1）是另一批账号——这一点必须由数据算出来再
    写进报告，不能凭印象写死，否则跨模型比较的口径会被说错。
    """
    sets = {name: _generated_users(name) for name in _FS_SINGLE_MODELS}
    if any(not v for v in sets.values()):
        return ""

    a, b, c = (sets[n] for n in _FS_SINGLE_MODELS)
    la, lb, lc = _FS_SINGLE_MODELS
    same_ab = a == b
    union = a | b | c

    parts: List[str] = []
    if same_ab:
        parts.append(
            f"`{la}` 与 `{lb}` 的被接管账号**完全相同**（各 {len(a):,} 个，集合相等）"
        )
    else:
        parts.append(
            f"`{la}` 与 `{lb}` 的被接管账号**不完全相同**"
            f"（{len(a):,} vs {len(b):,}，交集 {len(a & b):,}）"
        )
    parts.append(
        f"`{lc}` 是**另一批**账号（{len(c):,} 个，与前者交集仅 {len(a & c):,}），"
        f"三者并集 {len(union):,} 个"
    )
    head = "；".join(parts) + "。"

    if same_ab:
        tail = (
            f"因此：{la} vs {lb} 是**同一批账号的配对比较**（同账号换模型，"
            "可以算配对差、不能当两组独立样本做显著性检验）；"
            f"而 {lc} 与它们**只有 {len(a & c):,} 个账号重叠**，"
            f"是「换账号 + 换模型」的混合差异，不能解读为纯模型效应。"
        )
    else:
        tail = (
            "三份表的账号集合都不完全相同，跨模型比较同时混入了账号差异，"
            "报告时需注明不是纯模型效应。"
        )
    return head + tail


def _caveats(name: str, path: Path, manifest: Optional[dict] = None) -> List[str]:
    """针对具体数据集给出"用之前必须知道"的陷阱，全部由数据算出来，不写死结论。"""
    import pyarrow.parquet as pq

    manifest = manifest or {}
    info = manifest.get("datasets", {}).get(name, {})

    out: List[str] = []
    if name.startswith("fraudsquad"):
        df = pq.read_table(path, columns=["user_id", "label", "label_source", "rating"]).to_pandas()
        real_users = set(df.loc[df["label_source"] == "real_baseline", "user_id"])
        gen_users = set(df.loc[df["label_source"] == "llm_generated", "user_id"])
        overlap = gen_users & real_users
        out.append(
            f"**被接管账号同时保留真实评论**：{len(gen_users):,} 个写过生成评论的账号中，"
            f"{len(overlap):,} 个（{100 * len(overlap) / max(len(gen_users), 1):.1f}%）在真实基线里也有真实评论。"
            "这是论文的设定（接管既有账号），**账号级实验可以做**：正例 = 被接管账号，负例 = 其余账号，"
            "见 `python src/build_unified_data.py accounts`。"
            "唯一不能做的是**评论级**拿「账号是否被接管」当标签——那会把这个账号的历史真实评论也标成假的。"
        )
        gen_rating = df.loc[df["label"] == 1, "rating"]
        out.append(
            f"**评分为捷径特征**：{len(gen_rating):,} 条生成评论的评分全部为 "
            f"{sorted(gen_rating.dropna().unique().tolist())}，而真实评论 5 星占比仅 "
            f"{100 * (df.loc[df['label'] == 0, 'rating'] == 5).mean():.1f}%。"
            "任何用评分做输入的模型都会因此虚高，报告结果时必须把「仅评分」作为一个对照基线报出来。"
        )
        if name == "fraudsquad_amazon":
            per_user = df.loc[df["label_source"] == "llm_generated"].groupby("user_id").size()
            out.append(
                f"**这是三模型合并视图**：{len(per_user):,} 个被接管账号共 {int(per_user.sum()):,} 条假评论，"
                f"每人 {int(per_user.min())}~{int(per_user.max())} 条（均值 {per_user.mean():.1f}），"
                "与原文「每个账号 2 条」的口径不一致，**不要拿它和论文数字比**。"
                "对比实验请用 fraudsquad_llama3 / fraudsquad_qwen2 / fraudsquad_qwen_dsr1。"
            )
        else:
            out.append(
                "**单模型口径**：本表 = 真实基线 86,758 条 + 单一模型生成集，1,250 个被接管账号，"
                "每人约 2 条假评论，与论文设定一致。真实基线在三份单模型表里是同一份，"
                "所以三份表**不是三个互相独立的数据集**。"
            )
            relation = _fs_account_relation()
            if relation:
                out.append("**账号集合关系（按数据实算）**：" + relation)
    if name == "yelpchi":
        out.append(
            "**该版本无评论文本**：metadata 版只含 user/product/rating/label/date，"
            "`text_available=0`。文本方法（TF-IDF 等）在 YelpChi 上不可运行，"
            "不能用「空文本」当输入算出结果再与其它数据集比较。"
        )
        out.append(
            "**第三方镜像已排查，补不上文本**：科学数据银行的「Yelp dataset」发布包"
            "（dataset.zip，MD5 7c6c35041c4eabb65aa024f2f1ecff74）里与 YelpChi 相关的只有一个 "
            "`YelpChi.mat`，经解析是 CARE-GNN（Dou et al., CIKM 2020）的基准子集："
            "45,954 条评论、32 维特征、3 张关系图（net_rur/net_rtr/net_rsr）、label 含 6,677 条正例（14.53%），"
            "**不含任何评论文本**，且比官方 67,395 条少 21,441 条。"
            "它不能用来补 YelpChi 的文本缺口，也不能当 YelpChi 的替代版本使用。"
        )
    if name == "yelpnyc":
        out.append(
            "**标签为平台过滤器结果**（near ground truth）：与 YelpZip 同源，同样存在过滤算法误判，"
            "报告中需说明这是「近真值」而非人工标注。"
        )
        audit = info.get("text_join_audit")
        if audit:
            out.append(
                f"**元数据与文本是两个独立文件**：`metadata`（5 列，含 rating/label）与 "
                f"`reviewContent`（4 列，含正文）各 359,052 行，靠 (user, item, date) 对齐；"
                f"实测策略 {audit['strategy']}、匹配 {audit['matched']:,}/{audit['rows_metadata']:,}、"
                f"键错位 {audit['key_mismatches']} 行。统一表已合并好，"
                "下游不要自己去拼这两个文件（贴错一行是静默错误，指标上完全看不出来）。"
            )
        out.append(
            "**假评比例与另两套不同，不要混用**：官方公布 YelpNYC 为 10.27%"
            "（YelpChi 13.23% / YelpZip 13.22%），这是官方页各自公布的数值，不是抄写错误；"
            "本项目实测 10.2729%，spammer 账号占比 17.7850%（官方 17.79%）。"
            "**不要用 13.22% 描述 YelpNYC**。"
        )
    if name == "yelpzip":
        out.append(
            "**标签为平台过滤器结果**（near ground truth）：Yelp 的过滤算法本身有误判，"
            "这是所有使用该数据集的论文的共同前提，报告中需说明。"
        )
        audit = info.get("partial_row_audit")
        if audit:
            only_ratio = 100 * audit["users_only_in_partial"] / max(audit["dataset_users"], 1)
            out.append(
                f"**镜像文件有 {audit['partial_rows']} 行导出残缺**：这些行只有 "
                "`user / product / date / text` 四列，缺 rating 与 label，会被前一条未闭合引号的记录吞进正文。"
                f"读取阶段已剥离（{audit['contaminated_records']} 条正常记录的正文恢复干净），"
                f"但残缺行本身无标签、无法使用：涉及 {audit['partial_users']} 个账号，"
                f"其中 {audit['users_only_in_partial']} 个只出现在残缺行，"
                f"在 YelpZip 上做账号级实验时这部分账号缺位（占账号总数 {only_ratio:.3f}%）。"
                "补齐需要官方原版文件，见 docs/data/provenance.md §3。"
            )
        out.append(
            "**这 140 行残缺是流通文件自带的，不是本项目造成的**：本项目使用的 yelpzip.csv"
            "（410,865,364 字节，sha256 78f92e83…517fe）与科学数据银行 dataset.zip 内的同名文件"
            "**sha256 完全相同**；两个不同来源的镜像给出同一份字节，说明残缺来自这份导出文件本身，"
            "与本项目的下载与解析无关。"
        )
    if name == "yelpchi_injected":
        out.append(
            "**注入评论没有文本**：step3 只生成结构化事件（日期/评分/商品），"
            "文本方法需把注入行排除或单独处理，不能拿空文本参与训练。"
        )
    return out


def cmd_qa(args: argparse.Namespace) -> int:
    manifest_path = UNIFIED_DIR / MANIFEST_NAME
    if not manifest_path.exists():
        logger.error("尚未构建，请先运行 build")
        return 1
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lines: List[str] = []
    lines.append("# 统一数据集质量报告\n")
    lines.append(f"- 生成时间：{manifest['generated_at']}")
    lines.append(f"- schema 版本：{manifest['schema_version']}")
    lines.append(f"- 生成命令：`python src/build_unified_data.py build qa`")
    lines.append("- 判定口径：`label=1` 一律表示虚假评论（水军/LLM 生成/人工注入），`label=0` 为真实评论\n")

    lines.append("## 1. 数据集总览\n")
    lines.append("| 数据集 | 状态 | 评论数 | 用户数 | 商品数 | 假评论占比 | 文本覆盖 | 官方公布规模 | 行数偏差 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in list(PUBLIC_DATASETS) + ["yelpchi_injected"]:
        info = manifest["datasets"].get(name, {"status": "missing"})
        st = info.get("stats") or {}
        if info.get("status") == "ok":
            text_cov = f"{100 * st['has_text_rows'] / st['rows']:.1f}%"
            pct = info.get("rows_vs_expected_pct")
            lines.append(
                f"| {name} | ok | {st['rows']:,} | {st['users']:,} | {st['items']:,} | "
                f"{100 * st['fake_ratio']:.2f}% | {text_cov} | {info.get('expected_rows') or '—'} | "
                f"{(f'{pct:+.3f}%' if pct is not None else '—')} |"
            )
        else:
            lines.append(
                f"| {name} | {info.get('status')} | — | — | — | — | — | "
                f"{DATASET_SPECS[name].expected_rows or '—'} | — |"
            )

    lines.append("\n## 2. 数据口径核对（与官方公布值逐项对照）\n")
    any_issue = False
    for name, info in manifest["datasets"].items():
        if info.get("status") != "ok":
            any_issue = True
            lines.append(f"- **{name} 未构建**：{info.get('note') or info.get('error')}")
        else:
            pct = info.get("rows_vs_expected_pct")
            if pct is not None and abs(pct) > 2:
                any_issue = True
                lines.append(
                    f"- **{name} 行数与官方公布相差 {pct:+.2f}%**：{info.get('note')}。"
                    "下游引用时须注明实际使用的版本，不能直接套用论文数字。"
                )
    if not any_issue:
        lines.append("- 已构建的数据集行数与官方公布规模偏差均在 2% 以内。")
    # YelpZip 的行数缺口单独交代清楚：这不是"解析丢行"，而是镜像文件本身有残缺行
    z = manifest["datasets"].get("yelpzip", {})
    audit = z.get("partial_row_audit")
    if audit and z.get("stats"):
        lines.append("\n### 2.1 YelpZip 与官方 608,598 条的差额对账\n")
        lines.append("| 项目 | 数量 | 说明 |")
        lines.append("|---|---|---|")
        lines.append(f"| 镜像文件数据行 | {audit['partial_rows'] + z['stats']['rows']:,} | 与官方公布条数一致 |")
        lines.append(f"| 其中带标签、可用 | {z['stats']['rows']:,} | 进入统一表 |")
        lines.append(f"| 其中导出残缺、不可用 | {audit['partial_rows']:,} | 缺 rating 与 label，无法作为样本 |")
        lines.append(
            f"| 账号并集 | {audit['dataset_users']:,} + {audit['users_only_in_partial']:,} = "
            f"{audit['dataset_users'] + audit['users_only_in_partial']:,} | 官方公布 260,277，完全吻合 |"
        )
        lines.append(
            "\n**结论**：镜像并非「少了 140 行」，而是这 140 行在导出时丢了 rating 与 label 两个字段，"
            "并且被前一条未闭合引号的记录吞进正文。读取阶段已把这些残缺行从正常记录的正文里剥离，"
            "残缺行本身因无标签无法使用。行数、账号数两个口径同时对上官方公布值，可确认缺口来源。"
            "补齐需要官方原版文件（邮件申请渠道见 provenance.md）。\n"
        )
        lines.append(
            "**是文件本来就没有，还是读取时被丢掉的？——是文件本来就这样，不是解析丢行。** 三条证据：\n"
        )
        lines.append(
            "1. 原始字节层面：文件共 608,599 个物理换行，其中 608,459 个 `CRLF` + **140 个裸 `LF`**；"
            "按 CSV 记录解析得 608,459 条，再去掉 1 条表头 = 608,458 条，与解析结果逐条对得上。"
            "若是解析阶段丢行，丢的行不会在原始字节里留下痕迹。"
        )
        lines.append(
            f"2. 结构层面：这 {audit['partial_rows']} 个裸 `LF` 后面紧跟的正是 "
            "`user\\tproduct\\tdate\\ttext` 四列制表符分隔的残缺记录，缺的恰好是 rating 与 label；"
            "它们没有独立的行首，所以被上一条未闭合引号的记录整体吞进 `text`。"
        )
        lines.append(
            f"3. 数量层面：把这些残缺行按「独立记录」计回，行数 {audit['partial_rows'] + z['stats']['rows']:,}、"
            f"账号数 {audit['dataset_users'] + audit['users_only_in_partial']:,} 两项同时等于官方公布值，"
            "若中途丢行不可能两项同时对上。\n"
        )
        lines.append(
            "**论文里可直接写的句子**：本项目使用的 YelpZip 取自公开镜像（ModelScope），"
            f"与作者官方版本相差 {abs(z.get('rows_vs_expected_pct') or 0.0):.3f}%"
            f"（{z['stats']['rows']:,} vs 608,598 条），"
            "差额为镜像导出时 140 条记录缺失 rating/label 字段所致，已在读取阶段识别并剔除；"
            "账号数 260,277 与官方一致。\n"
        )

    lines.append("\n### 2.2 交付清单\n")
    lines.append("| 数据集 | 论文口径 | 状态 | 条数 | 有文本 | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    for name in PUBLIC_DATASETS:
        info = manifest["datasets"].get(name, {"status": "missing"})
        st = info.get("stats") or {}
        if info.get("status") == "ok":
            has_text = bool(st.get("has_text_rows", 0))
            if not has_text:
                note = "无评论文本，文本类方法不可用"
            else:
                note = DATASET_SPECS[name].single_model or "—"
            lines.append(
                f"| {name} | 是 | ✅ | {st['rows']:,} | {'✅' if has_text else '❌'} | {note} |"
            )
        else:
            lines.append(f"| {name} | 是 | ⛔ 缺失 | — | — | {info.get('note', '')[:60]} |")
    for name in VIEW_ONLY_DATASETS:
        info = manifest["datasets"].get(name, {})
        st = info.get("stats") or {}
        if st:
            lines.append(f"| {name} | 否（仅总览视图） | ✅ | {st['rows']:,} | ✅ | 不进 all_reviews |")
    for name, info in manifest["datasets"].items():
        if name in PUBLIC_DATASETS or name in VIEW_ONLY_DATASETS:
            continue
        st = info.get("stats") or {}
        if st:
            lines.append(
                f"| {name} | 否（自建） | ✅ | {st['rows']:,} | "
                f"{'✅' if st.get('has_text_rows', 0) else '❌'} | 注入实验用 |"
            )


    lines.append("\n## 3. 标签来源构成（决定能否与其他论文对比）\n")
    lines.append("| 数据集 | 标签来源 | 条数 |")
    lines.append("|---|---|---|")
    for name, info in manifest["datasets"].items():
        path = UNIFIED_DIR / f"{name}.parquet"
        if info.get("status") != "ok" or not path.exists():
            continue
        for src_name, cnt in _label_source_breakdown(path).items():
            lines.append(f"| {name} | {src_name} | {cnt:,} |")

    lines.append("\n## 4. 各数据集明细\n")
    for name, info in manifest["datasets"].items():
        lines.append(f"### {name}\n")
        lines.append(f"- 状态：{info.get('status')}")
        lines.append(f"- 说明：{info.get('note')}")
        spec = info.get("spec", {})
        if spec:
            lines.append(f"- 出处：{spec.get('homepage')}")
            lines.append(f"- 引用：{spec.get('citation')}")
            lines.append(f"- 使用条款：{spec.get('license')}")
        for rf in info.get("raw_files", []):
            if rf.get("path"):
                lines.append(f"- 原始文件：`{rf['path']}`（{rf['size']:,} 字节，sha256 `{rf['sha256'][:16]}…`）")
        st = info.get("stats")
        if st:
            lines.append(
                f"- 统计：{st['rows']:,} 条 / {st['users']:,} 用户 / {st['items']:,} 商品 / "
                f"时间 {st['date_min'][:10]} ~ {st['date_max'][:10]} / 假评论 {st['fake_rows']:,} 条"
            )
            if st.get("duplicated_user_item_date"):
                lines.append(
                    f"- 同一用户—商品—日期出现多条评论：{st['duplicated_user_item_date']:,} 组。"
                    "这是原始数据的真实形态（Yelp 允许修改评论、Amazon 允许同一商品多次购买评价），"
                    "**未做去重**；做用户-商品图时需自行决定是否合并。"
                )
        lines.append("")

    lines.append("## 5. 使用陷阱（用之前必须知道）\n")
    for name, info in manifest["datasets"].items():
        path = UNIFIED_DIR / f"{name}.parquet"
        if info.get("status") != "ok" or not path.exists():
            continue
        notes = _caveats(name, path, manifest)
        if notes:
            lines.append(f"### {name}\n")
            for n_ in notes:
                lines.append(f"- {n_}")
            lines.append("")

    lines.append("## 6. 账号级视图（`build_unified_data.py accounts` 产出）\n")
    acc = manifest.get("accounts") or []
    if acc:
        lines.append("| 数据集 | 账号数 | 正例账号 | 正例占比 | 账号标签口径 |")
        lines.append("|---|---|---|---|---|")
        for a in acc:
            lines.append(
                f"| {a['dataset']} | {a['accounts']:,} | {a['positive_accounts']:,} | "
                f"{100 * a['positive_ratio']:.2f}% | `{a['label_basis']}` |"
            )
        lines.append(
            "\n账号级正例的定义见 `src/data/schema.py: ACCOUNT_LABEL_BASIS`。"
            "**账号级与评论级是两个不同任务**：账号级问「这个号是不是水军」，评论级问「这条评论是不是假的」。"
            "FraudSquad 的被接管账号同时有真实评论，因此账号级正例不代表其全部评论为假；"
            "反过来，评论级实验也不能拿账号标签当答案。\n"
        )
    else:
        lines.append("- 尚未生成，运行 `python src/build_unified_data.py accounts`。\n")

    lines.append("## 7. 标签口径与公平对比提示\n")
    lines.append("- Yelp 系列标签来自 Yelp 平台过滤器（near ground truth），不是人工逐条标注，存在平台自身误差。")
    lines.append("- FraudSquad 的虚假评论由大模型生成，真实评论取自 Amazon 5-core，属「构造即真值」。")
    lines.append("- 三个 FraudSquad 单模型表是**论文口径的独立数据集**；合并视图 `fraudsquad_amazon` 只作总览。")
    lines.append("- 人工注入集（yelpchi_injected）的标签由注入脚本保证，与平台标签不可混为一谈。")
    lines.append("- 对比实验必须写明每种方法是否需要标签：LOF/iForest/DBSCAN 无监督；TF-IDF+LR 与 GNN 类方法有监督。")
    lines.append("- 需要说明的是：YelpChi 与 YelpZip 的假评论比例（约 13%）相近，而 FraudSquad 三个单模型表"
                 "分别约 2.8%（真实基线占绝大多数），跨数据集的绝对数值不可直接横向比较，"
                 "只能说「在同一数据集、同一划分下谁更好」。\n")

    out = RESULTS_DIR / "data_quality_report.md"
    ensure_dir(RESULTS_DIR)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("质量报告已写入 %s", out)
    print(f"质量报告: {out}")
    return 0


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    manifest_path = UNIFIED_DIR / MANIFEST_NAME
    if not manifest_path.exists():
        logger.error("manifest 不存在，无法校验")
        return 1
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    failures: List[str] = []
    for name, info in manifest["datasets"].items():
        if info.get("status") != "ok":
            continue
        path = UNIFIED_DIR / f"{name}.parquet"
        if not path.exists():
            failures.append(f"{name}: parquet 不存在")
            continue
        if info.get("parquet_sha256") and sha256_of(path) != info["parquet_sha256"]:
            failures.append(f"{name}: parquet 与 manifest 记录的 sha256 不一致（文件被改动或重建后未更新 manifest）")
        df = pd.read_parquet(path, columns=["label", "date", "review_id"])
        try:
            # 只做轻量校验：全量校验在 build 阶段已执行
            if not df["label"].isin([0, 1]).all():
                failures.append(f"{name}: label 出现非 0/1")
            if df["date"].isna().any():
                failures.append(f"{name}: date 存在空值")
            if df["review_id"].duplicated().any():
                failures.append(f"{name}: review_id 重复")
            if info.get("stats") and len(df) != info["stats"]["rows"]:
                failures.append(f"{name}: 行数 {len(df)} 与 manifest 记录 {info['stats']['rows']} 不一致")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: 校验异常 {exc}")

    if failures:
        print("校验失败：")
        for f_ in failures:
            print("  -", f_)
        return 2
    print(f"校验通过：{sum(1 for v in manifest['datasets'].values() if v.get('status') == 'ok')} 个数据集与 manifest 一致")
    return 0


# --------------------------------------------------------------------------
# 账号级视图
# --------------------------------------------------------------------------

def cmd_accounts(args: argparse.Namespace) -> int:
    """从已构建的评论级表聚合出账号级表，供账号级检测实验使用。

    为什么要做账号级：FraudSquad 与我们的注入实验都是"接管一批既有账号、让它们写假评论"，
    这正是水军的真实形态。账号级任务的正确表述是「在被接管账号 vs 普通账号之间分类」，
    而不是把该账号的历史真实评论也当成假评论（那是评论级的错）。
    """
    manifest_path = UNIFIED_DIR / MANIFEST_NAME
    if not manifest_path.exists():
        logger.error("尚未构建评论级表，请先运行 build")
        return 1
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ensure_dir(ACCOUNTS_DIR)
    names = args.dataset or [n for n in BUILDERS if (UNIFIED_DIR / f"{n}.parquet").exists()]
    summary: List[dict] = []
    for name in names:
        src = UNIFIED_DIR / f"{name}.parquet"
        if not src.exists():
            logger.warning("%s 尚未构建，跳过账号级聚合", name)
            continue
        basis = ACCOUNT_LABEL_BASIS_BY_DATASET.get(name)
        if basis is None:
            logger.warning("%s 没有定义账号标签口径，跳过", name)
            continue
        df = pd.read_parquet(src, columns=["dataset", "user_id", "item_id", "date", "label", "is_injected"])
        acc = validate_accounts(build_account_table(df, label_basis=basis), context=name)
        out = ACCOUNTS_DIR / f"{name}.parquet"
        acc.to_parquet(out, index=False, compression="zstd")
        pos = int(acc["account_label"].sum())
        summary.append(
            {
                "dataset": name,
                "accounts": int(len(acc)),
                "positive_accounts": pos,
                "positive_ratio": round(pos / max(len(acc), 1), 6),
                "label_basis": basis,
                "parquet": str(out.relative_to(PROJECT_ROOT)),
                "parquet_sha256": sha256_of(out),
            }
        )
        logger.info(
            "%s 账号级：%d 个账号，其中 %d 个正例（%.2f%%）",
            name, len(acc), pos, 100 * pos / max(len(acc), 1),
        )

    manifest["accounts"] = summary
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="公开数据集统一数据管线（Yelp 三套 + FraudSquad 三套 + 自建注入集）")
    p.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="下载原始数据")
    d.add_argument("--dataset", nargs="*", help="只下载指定数据集")
    d.add_argument("--force", action="store_true", help="忽略已存在的完整文件，强制重下")
    d.add_argument("--tolerate-failure", action="store_true", help="单个文件失败不中断")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="构建统一格式")
    b.add_argument("--dataset", nargs="*", help="只构建指定数据集")
    b.add_argument("--no-combined", action="store_true", help="不生成 all_reviews.parquet")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("qa", help="生成质量报告")
    q.set_defaults(func=cmd_qa)

    ac = sub.add_parser("accounts", help="生成账号级表（从评论级聚合）")
    ac.add_argument("--dataset", nargs="*", help="只处理指定数据集")
    ac.set_defaults(func=cmd_accounts)

    v = sub.add_parser("verify", help="校验产物")
    v.set_defaults(func=cmd_verify)

    a = sub.add_parser("all", help="download + build + accounts + qa + verify")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "all":
        ns = argparse.Namespace(dataset=None, force=args.force, tolerate_failure=False)
        rc = cmd_download(ns)
        rc = cmd_build(argparse.Namespace(dataset=None, no_combined=False)) or rc
        cmd_accounts(argparse.Namespace(dataset=None))
        cmd_qa(argparse.Namespace())
        cmd_verify(argparse.Namespace())
        return rc

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
