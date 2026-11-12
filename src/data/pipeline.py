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
    SchemaError,
    coerce_dtypes,
    summary_stats,
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
    if path is None or not path.exists():
        return {"path": None, "size": None, "sha256": None}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in path.parents else str(path),
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
    return df, [_raw_record(path)], f"来源文件 {path.name}（带评论文本）"


def build_yelpnyc() -> Tuple[pd.DataFrame, List[dict], str]:
    candidates = sorted(RAW_YELPNYC_DIR.glob("*.csv")) + sorted(RAW_YELPNYC_DIR.glob("*.txt"))
    path = _first_existing(candidates)
    if path is None:
        return pd.DataFrame(), [], sources.MANUAL_SOURCES["yelpnyc"]
    df = readers.read_yelp_family(path, "yelpnyc")
    return df, [_raw_record(path)], f"来源文件 {path.name}"


def build_fraudsquad(dataset_id: str = "fraudsquad_amazon",
                     only: Optional[str] = None) -> Tuple[pd.DataFrame, List[dict], str]:
    """FraudSquad：默认合并三份生成集；A7 支持按单份生成集构建独立数据集。"""
    real_zip = RAW_FRAUDSQUAD_DIR / "amazon_2022_5core.csv.zip"
    if not real_zip.exists():
        return pd.DataFrame(), [], "未找到 amazon_2022_5core.csv.zip，请先运行 download 子命令"

    real_csv = readers.unzip_if_needed(real_zip, RAW_FRAUDSQUAD_DIR / "amazon", expect=".csv")
    all_generated = {
        "llama3_8b": RAW_FRAUDSQUAD_DIR / "reviews_from_llama3_8b.csv",
        "qwen_72b": RAW_FRAUDSQUAD_DIR / "reviews_from_qwen_72b.csv",
        "qwen_dsr1": RAW_FRAUDSQUAD_DIR / "reviews_from_qwen_dsr1.csv",
    }
    candidates = all_generated if only is None else {only: all_generated[only]}
    generated = {name: p for name, p in candidates.items() if p.exists() and p.stat().st_size > 0}
    if not generated:
        return pd.DataFrame(), [], "未找到任何 LLM 生成集（reviews_from_*.csv）"
    missing = [k for k in candidates if k not in generated]
    df = readers.read_fraudsquad_amazon(real_csv, generated, dataset_id=dataset_id)
    note = f"真实基线 {real_csv.name}；生成集 {sorted(generated)}"
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
    # A8：行数守卫（70,186 ±2%）——v2 日志为唯一真源，重跑 step3 后应命中
    expected_total = 70186
    if abs(len(combined) - expected_total) / expected_total > 0.02:
        raise ValueError(
            f"yelpchi_injected 行数 {len(combined)} 偏离期望 {expected_total} 超过 ±2%"
            f"（注入事件 {len(injected)} 条），先检查 injection_log_v2.json 与评论底表是否同版")
    note = (
        f"由 {log_path.name} 的 {len(log)} 个账号、{len(injected)} 条注入事件展开；"
        "注入评论无文本（step3 未生成文本），text_available=0"
    )
    return combined, [_raw_record(log_path), _raw_record(reviews_path)], note


BUILDERS = {
    "yelpchi": build_yelpchi,
    "yelpnyc": build_yelpnyc,
    "yelpzip": build_yelpzip,
    "fraudsquad_amazon": build_fraudsquad,
    # A7：三份生成集各自成表（共用真实基线，便于同基线公平对比）
    "fraudsquad_llama3": partial(build_fraudsquad, dataset_id="fraudsquad_llama3", only="llama3_8b"),
    "fraudsquad_qwen2": partial(build_fraudsquad, dataset_id="fraudsquad_qwen2", only="qwen_72b"),
    "fraudsquad_qwen_dsr1": partial(build_fraudsquad, dataset_id="fraudsquad_qwen_dsr1", only="qwen_dsr1"),
    "yelpchi_injected": build_injected,
}

# 合并大表时排除 FraudSquad 单份生成集，避免与 fraudsquad_amazon 重复计数
COMBINE_EXCLUDE = {"fraudsquad_llama3", "fraudsquad_qwen2", "fraudsquad_qwen_dsr1"}


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
            **extra,
        }
        logger.info("%s 完成：%d 行，假评论占比 %.2f%%", name, stats["rows"], 100 * stats["fake_ratio"])

    if not args.no_combined:
        manifest["combined"] = _write_combined(
            [n for n in BUILDERS if n not in COMBINE_EXCLUDE and (UNIFIED_DIR / f"{n}.parquet").exists()])

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


def _caveats(name: str, path: Path) -> List[str]:
    """针对具体数据集给出"用之前必须知道"的陷阱，全部由数据算出来，不写死结论。"""
    import pyarrow.parquet as pq

    out: List[str] = []
    if name == "fraudsquad_amazon":
        df = pq.read_table(path, columns=["user_id", "label", "label_source", "rating"]).to_pandas()
        real_users = set(df.loc[df["label_source"] == "real_baseline", "user_id"])
        gen_users = set(df.loc[df["label_source"] == "llm_generated", "user_id"])
        overlap = gen_users & real_users
        out.append(
            f"**生成评论复用了真实账号**：{len(gen_users):,} 个发布过生成评论的用户中，"
            f"{len(overlap):,} 个（{100 * len(overlap) / max(len(gen_users), 1):.1f}%）在真实基线里也有真实评论。"
            "因此**不能**用「该用户是否发过生成评论」直接当账号级标签——那会把正常用户的账号误判为水军。"
            "账号级分析请改用评论级标签 + 人工核对，或把混合账号单独标记。"
        )
        gen_rating = df.loc[df["label"] == 1, "rating"]
        out.append(
            f"**评分为捷径特征**：{len(gen_rating):,} 条生成评论的评分全部为 "
            f"{sorted(gen_rating.dropna().unique().tolist())}，而真实评论 5 星占比仅 "
            f"{100 * (df.loc[df['label'] == 0, 'rating'] == 5).mean():.1f}%。"
            "任何用评分做输入的模型都会因此虚高，报告结果时必须把「仅评分」作为一个对照基线报出来。"
        )
    if name == "yelpchi":
        out.append(
            "**该版本无评论文本**：metadata 版只含 user/product/rating/label/date，"
            "`text_available=0`。文本方法（TF-IDF 等）在 YelpChi 上不可运行，"
            "不能用「空文本」当输入算出结果再与其它数据集比较。"
        )
    if name == "yelpzip":
        out.append(
            "**标签为平台过滤器结果**（near ground truth）：Yelp 的过滤算法本身有误判，"
            "这是所有使用该数据集的论文的共同前提，报告中需说明。"
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

    lines.append("\n## 2. 缺失与偏差说明（如实记录）\n")
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
    lines.append(
        "- **YelpNYC 建议的补齐方式**：向 srayana@cs.stonybrook.edu 邮件申请（注明课程用途）取得后，"
        "把文件放到 `data/raw/yelpnyc/` 再跑一次 `build`，管线会自动识别并纳入，无需改代码。"
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
        notes = _caveats(name, path)
        if notes:
            lines.append(f"### {name}\n")
            for n_ in notes:
                lines.append(f"- {n_}")
            lines.append("")

    lines.append("## 6. 标签口径与公平对比提示\n")
    lines.append("- Yelp 系列标签来自 Yelp 平台过滤器（near ground truth），不是人工逐条标注，存在平台自身误差。")
    lines.append("- FraudSquad 的虚假评论由大模型生成，真实评论取自 Amazon 5-core，属「构造即真值」。")
    lines.append("- 人工注入集（yelpchi_injected）的标签由注入脚本保证，与平台标签不可混为一谈。")
    lines.append("- 对比实验必须写明每种方法是否需要标签：LOF/iForest/DBSCAN 无监督；TF-IDF+LR 与 GNN 类方法有监督。")
    lines.append("- 需要说明的是：YelpChi 与 YelpZip 的假评论比例（约 13%）相近，而 FraudSquad 为 7.94%，"
                 "跨数据集的绝对数值不可直接横向比较，只能说「在同一数据集上谁更好」。\n")

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
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="四数据集统一数据管线")
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

    v = sub.add_parser("verify", help="校验产物")
    v.set_defaults(func=cmd_verify)

    a = sub.add_parser("all", help="download + build + qa + verify")
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
        cmd_qa(argparse.Namespace())
        cmd_verify(argparse.Namespace())
        return rc

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
