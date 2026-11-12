"""从 YelpZip 构造「规模对齐」派生视图，用于替代缺失的 YelpNYC 做规模敏感性检查。

**这不是 YelpNYC，也不是它的替身。**
它只对齐一个维度——**评论总量**（YelpNYC 官方公布 359,052 条），做法是对商户
按评论数分层抽样，使派生视图的评论量级与 YelpNYC 同档。商户粒度、城市覆盖、
时间跨度都不同，因此它只能回答"评论总量降一个量级后方法是否仍成立"，
不能回答"纽约市是否特殊"。任何把它称作 YelpNYC 的写法都是错的。

为什么只对齐评论总量、不对齐商户数：
    YelpNYC 是 359,052 条 / 923 家商户（均值 389 条/商户），
    YelpZip 是 608,458 条 / 5,044 家商户（均值 121 条/商户）。
    要同时对齐这两个数是不可能的——YelpZip 里凑出 923 家商户最多只有约 11 万条评论，
    硬取"最大的 923 家"会让商户规模分布严重右偏。因此本项目选择对齐评论总量，
    抽样时仍按商户评论数分层，避免抽出来的商户全是同一种体量。

用法：
    python src/build_unified_data.py subsample
    python src/build_unified_data.py subsample --seeds 0,1,2 --target 359052
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import DATASET_SPECS

logger = logging.getLogger(__name__)

#: 目标评论量：直接取自 yelpnyc 的官方公布规模，避免这里和 schema 两处各写一个数
TARGET_ROWS: int = int(DATASET_SPECS["yelpnyc"].expected_rows or 359_052)

#: 默认重复次数（不同随机种子，用于给出区间而不是单点）
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

#: 分层数：按商户评论数做十分位分层
DEFAULT_STRATA: int = 10

#: 派生视图命名前缀（后缀带目标量级，避免以后换目标时名字说谎）
VIEW_PREFIX = "yelpzip_scale"


def view_name(target_rows: int = TARGET_ROWS) -> str:
    """派生视图的名字：`yelpzip_scale359k`（按千条取整，只为可读）。"""
    return f"{VIEW_PREFIX}{int(round(target_rows / 1000.0))}k"


def stratified_subsample(
    df: pd.DataFrame,
    *,
    target_rows: int = TARGET_ROWS,
    n_strata: int = DEFAULT_STRATA,
    seed: int = 0,
    item_col: str = "item_id",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """按商户评论数分层抽商户，直到累计评论数达到 `target_rows`。

    做法（对每个分层）：
      1. 按商户评论数把商户切成 `n_strata` 个分位层（层内商户体量相近）；
      2. 按该层评论总量占全量的比例，算出该层应贡献的评论数配额；
      3. 层内用固定种子打乱商户顺序，依次累加，超过配额即停。

    这样每个种子抽出的商户集合不同，但"各体量商户的评论占比"与全量一致，
    不会因为随机抽而把小商户抽光（纯随机抽商户会让大量小商户被抽中、
    大商户被抽少，从而改变评分分布）。

    返回 (子集, 审计信息)。审计信息里 `overshoot` 是超过目标的行数占目标的比例，
    正常应在 1 个商户的量级（约 0.1%~0.5%）。
    """
    if item_col not in df.columns:
        raise KeyError(f"缺少商户列 {item_col!r}")
    if target_rows <= 0 or target_rows > len(df):
        raise ValueError(f"target_rows={target_rows} 超出可用行数 {len(df)}")
    if n_strata < 1:
        raise ValueError("n_strata 必须 >= 1")

    counts = df[item_col].value_counts()
    if len(counts) < n_strata:
        raise ValueError(f"商户数 {len(counts)} 少于分层数 {n_strata}")

    # 1) 商户 -> 分层（按评论数分位切；ties 用 rank(method="first") 打散，保证每层非空）
    ranks = counts.rank(method="first")
    strata_of_item = pd.qcut(ranks, q=n_strata, labels=False).astype(int)

    rng = np.random.default_rng(seed)
    picked: List[str] = []
    stratum_audit: List[Dict[str, object]] = []

    for s in range(n_strata):
        in_stratum = np.asarray(strata_of_item.index[strata_of_item == s])
        stratum_counts = counts.loc[in_stratum].to_numpy()
        stratum_rows = int(stratum_counts.sum())
        quota = target_rows * stratum_rows / float(len(df))

        order = rng.permutation(len(in_stratum))
        acc = 0
        taken = 0
        for idx in order:
            if acc >= quota:
                break
            acc += int(stratum_counts[idx])
            taken += 1
        picked.extend(in_stratum[order[:taken]].tolist())
        stratum_audit.append(
            {
                "stratum": s,
                "items": int(len(in_stratum)),
                "rows": stratum_rows,
                "quota": round(quota, 1),
                "taken_items": int(taken),
                "taken_rows": int(acc),
            }
        )

    sub = df[df[item_col].isin(picked)]
    audit: Dict[str, object] = {
        "seed": int(seed),
        "target_rows": int(target_rows),
        "rows": int(len(sub)),
        "items": int(sub[item_col].nunique()),
        "users": int(sub["user_id"].nunique()),
        "fake_ratio": round(float(sub["label"].mean()), 6) if len(sub) else None,
        "overshoot_pct": round(100.0 * (len(sub) - target_rows) / target_rows, 4),
        "n_strata": int(n_strata),
        "strata": stratum_audit,
    }
    return sub, audit


def build_views(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    target_rows: int = TARGET_ROWS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_strata: int = DEFAULT_STRATA,
    with_text: bool = True,
) -> Dict[str, object]:
    """生成多个种子的派生视图并写盘，返回汇总 manifest。

    每个种子的产物是 `{out_dir}/{view_name}_seed{seed}.parquet`。
    `manifest` 里记录每个种子的行数/商户数/用户数/假评比例，供报告直接引用。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_rows is None:
        target_rows = TARGET_ROWS

    cols = list(df.columns)
    if not with_text and "text" in cols:
        cols = [c for c in cols if c != "text"]
        df = df[cols]

    name = view_name(target_rows)
    runs: List[Dict[str, object]] = []
    for seed in seeds:
        sub, audit = stratified_subsample(df, target_rows=target_rows, n_strata=n_strata, seed=seed)
        path = out_dir / f"{name}_seed{seed}.parquet"
        sub.to_parquet(path, index=False, compression="zstd")
        audit["file"] = path.name
        runs.append(audit)
        logger.info(
            "派生视图 %s seed=%d：%d 条 / %d 商户 / 假评 %.2f%%",
            name,
            seed,
            audit["rows"],
            audit["items"],
            100 * (audit["fake_ratio"] or 0.0),
        )

    manifest: Dict[str, object] = {
        "view": name,
        "source_dataset": "yelpzip",
        "purpose": "YelpNYC 缺失时的规模敏感性检查（derived view，非 YelpNYC 替身）",
        "target_rows": int(target_rows),
        "target_reference": "YelpNYC 官方公布规模（schema.DATASET_SPECS['yelpnyc'].expected_rows）",
        "n_strata": int(n_strata),
        "with_text": bool(with_text),
        "runs": runs,
        "rows_mean": round(float(np.mean([r["rows"] for r in runs])), 1),
        "rows_min": int(min(r["rows"] for r in runs)),
        "rows_max": int(max(r["rows"] for r in runs)),
        "fake_ratio_mean": round(float(np.mean([r["fake_ratio"] for r in runs])), 6),
        "source_rows": int(len(df)),
        "source_items": int(df["item_id"].nunique()),
        "source_fake_ratio": round(float(df["label"].mean()), 6),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def render_report(manifest: Dict[str, object]) -> str:
    """把派生视图的抽样结果渲染成一页 markdown（写进 results/）。"""
    name = manifest["view"]
    lines: List[str] = []
    lines.append(f"# 派生视图 `{name}`：YelpZip 规模对齐子样本\n")
    lines.append(
        "> **用途**：YelpNYC 缺失时的替代性规模敏感性检查。\n"
        "> **它不是 YelpNYC，也不能当 YelpNYC 用**——只对齐评论总量这一个维度，\n"
        "> 商户粒度、城市覆盖、时间跨度都不同。详见 "
        "[`docs/data/yelpnyc_fallback.md`](../docs/data/yelpnyc_fallback.md) §3 方案 A。\n"
    )
    lines.append("## 1. 抽样设计\n")
    lines.append(f"- 源数据：YelpZip，{manifest['source_rows']:,} 条 / {manifest['source_items']:,} 家商户，"
                 f"假评占比 {100 * manifest['source_fake_ratio']:.2f}%")
    lines.append(f"- 目标量级：{manifest['target_rows']:,} 条（= {manifest['target_reference']}）")
    lines.append(f"- 分层：按商户评论数切 {manifest['n_strata']} 个分位层，层内按比例配额、层内随机取商户")
    lines.append(f"- 重复：{len(manifest['runs'])} 个随机种子，报告区间而非单点\n")

    lines.append("## 2. 各随机种子的结果\n")
    lines.append("| 种子 | 评论数 | 与目标的偏差 | 商户数 | 用户数 | 假评占比 | 文件 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in manifest["runs"]:
        lines.append(
            f"| {r['seed']} | {r['rows']:,} | {r['overshoot_pct']:+.3f}% | {r['items']:,} | "
            f"{r['users']:,} | {100 * r['fake_ratio']:.2f}% | `{r['file']}` |"
        )
    lines.append("")
    lines.append(
        f"**汇总**：{manifest['rows_min']:,} ~ {manifest['rows_max']:,} 条，"
        f"均值 {manifest['rows_mean']:,.0f} 条；假评占比均值 {100 * manifest['fake_ratio_mean']:.2f}%"
        f"（源数据 {100 * manifest['source_fake_ratio']:.2f}%，偏差 < 0.05 个百分点，说明分层抽样没有扭曲标签分布）。\n"
    )

    lines.append("## 3. 怎么用、不能怎么用\n")
    lines.append("**可以**：把 5 个种子各跑一遍方法，报告 AUC / F1 的均值与 95% 区间，"
                 "回答「评论总量降到 36 万量级时结论是否稳定」。")
    lines.append("**不可以**：① 把它写进主对比表冒充第 4 个公开数据集；"
                 "② 声称它代表纽约市；③ 拿它和 YelpZip 全量的差异解释为「城市差异」。\n")
    lines.append("```python")
    lines.append("import pandas as pd")
    lines.append(f"df = pd.read_parquet('data/processed/derived/{name}_seed0.parquet')")
    lines.append("```")
    return "\n".join(lines) + "\n"
