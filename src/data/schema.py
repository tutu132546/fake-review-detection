"""统一评论级 schema：列定义、数据集元信息、标签语义与校验规则。

设计原则
--------
1. **单一方向**：`label` 一律 1 = 虚假评论（水军写的 / LLM 生成的 / 人工注入的），
   0 = 真实评论。原始数据里 Yelp 用 -1 表示被过滤（filtered，即假），1 表示推荐，
   这个翻转只在 reader 里做一次，下游不再接触原始编码。
2. **不猜不填**：字段缺失就是缺失（text 为空串 + `text_available=False`，
   rating 缺失为 NaN），绝不静默补 0 或补均值。缺失量会进质量报告。
3. **可追溯**：`label_source` 记录标签是怎么来的（平台过滤器 / LLM 生成 /
   人工标注 / 我们注入），因为"公平对比"要求说明每种方法的监督信号强度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 列定义
# --------------------------------------------------------------------------

#: 统一表的列顺序（对外承诺的稳定契约，下游按此顺序读取）
CANONICAL_COLUMNS: List[str] = [
    "dataset",          # str  数据集标识
    "review_id",        # str  数据集内唯一的评论 ID
    "global_user_id",   # str  "{dataset}:{user_id}"，跨数据集拼接时不会撞号
    "user_id",          # str  数据集原始用户 ID
    "item_id",          # str  数据集原始商品/商家 ID
    "rating",           # float32 评分，缺失为 NaN
    "date",             # datetime64[ns] 评论时间
    "label",            # int8  1=虚假 0=真实
    "label_source",     # str   标签来源（见 LABEL_SOURCE_VALUES）
    "is_injected",      # int8  1=我们人工注入的评论
    "text_available",   # int8  1=原始数据带文本，0=该数据集无文本字段
    "text",             # str   评论文本，无文本时为空串
]

#: 标签来源允许取值
LABEL_SOURCE_VALUES = (
    "yelp_filter",       # Yelp 平台过滤器结果（平台标注，非人工逐条标注）
    "real_baseline",     # 真实评论基线（未被替换的原始评论，构造上为真）
    "llm_generated",     # 由大模型生成，天然已知为假
    "human_generated",   # 由人写作的众包垃圾评论（FraudSquad 的 QA 集）
    "injected",          # 我们按行为模板注入的评论
)

#: 不能进入模型输入、只能用于分组/评估的列
EVAL_ONLY_COLUMNS: List[str] = ["dataset", "review_id", "global_user_id", "label", "label_source", "is_injected"]


class SchemaError(ValueError):
    """统一表不符合契约时抛出，消息里带上可执行的修复提示。"""


@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的元信息 + 标签语义 + 使用条款。

    这些字段会被写进 `manifest.json` 和 `docs/data/provenance.md`，
    因此"数据从哪来、能不能用、标签是什么意思"只在这里维护一份。
    """

    name: str
    display_name: str
    #: 官方/权威主页（论文作者或数据集维护方）
    homepage: str
    #: 我们实际取数用的地址（可能是镜像，需在 notes 里说明）
    download_page: str
    citation: str
    #: 许可与使用条款（如实写，不确定就写"未声明"）
    license_note: str
    #: 原始 label 编码 -> 统一 label 的映射说明
    label_semantics: str
    #: 是否需要文本
    has_text: bool
    #: 官方公布的规模，用于交叉核对（收不到就如实标 missing）
    expected_rows: Optional[int] = None
    notes: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


def _fraudsquad_spec(name: str, display_name: str, *, single_model: Optional[str] = None) -> DatasetSpec:
    """FraudSquad 元信息（A7）：合并视图与三个单生成集共用同一真实基线。"""
    if single_model is None:
        notes = ("真实基线 amazon_2022_5core + 三个模型生成的假评论合并视图；"
                 "单份生成集另见 fraudsquad_llama3 / fraudsquad_qwen2 / fraudsquad_qwen_dsr1。")
    else:
        notes = (f"真实基线 amazon_2022_5core + 仅 {single_model} 生成集；"
                 "三份生成集共用同一真实基线，便于同基线公平对比。")
    return DatasetSpec(
        name=name,
        display_name=display_name,
        homepage="https://github.com/RafaelLiuZH/FraudSquad",
        download_page="https://github.com/RafaelLiuZH/FraudSquad/tree/main/data（官方实现仓库）",
        citation=(
            "Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model "
            "Embeddings and Graph Neural Network (FraudSquad), 2026"
        ),
        license_note=(
            "仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，"
            "受 Amazon 数据条款约束；生成部分由作者用 Llama3-8B / Qwen 系列产出。"
            "本项目仅作学术研究引用，不再分发原始文件。"
        ),
        label_semantics="真实评论 label=0（label_source=real_baseline）；生成评论 label=1（label_source=llm_generated）",
        has_text=True,
        notes=notes,
    )


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "yelpchi": DatasetSpec(
        name="yelpchi",
        display_name="YelpChi",
        homepage="https://shebuti.com/yelpchi-dataset/",
        download_page="https://shebuti.com/yelpchi-dataset/（官方需邮件申请）；本项目使用学界广泛流转的 metadata 版本",
        citation=(
            "Mukherjee et al., What Yelp Fake Review Filter Might Be Doing? ICWSM 2013; "
            "Rayana & Akoglu, Collective Opinion Spam Detection, KDD 2015"
        ),
        license_note=(
            "仅限学术研究使用；Yelp 评论内容版权归 Yelp 及原作者，不得再分发或商用。"
            "官方发布页要求引用上述两篇论文。"
        ),
        label_semantics="原始 label: 1=Yelp 推荐（真实），-1=Yelp 过滤（虚假）→ 统一为 1/-1 翻转成 label=1 表示虚假",
        has_text=False,  # 本项目持有的 metadata 版本不含评论文本
        expected_rows=67395,
        notes=(
            "本项目持有的 metadata.txt 为评论级 5 列（user/product/rating/label/date），"
            "与官方公布的 67,395 条一致，但不含 reviewContent 文本文件；"
            "文本版需向作者邮件申请，属已知缺口，见 docs/data/provenance.md。"
        ),
    ),
    "yelpnyc": DatasetSpec(
        name="yelpnyc",
        display_name="YelpNYC",
        homepage="https://shebuti.com/yelpnyc-dataset/",
        download_page="https://shebuti.com/yelpnyc-dataset/（官方需邮件申请）",
        citation="Rayana & Akoglu, Collective Opinion Spam Detection: Bridging Review Networks and Metadata, KDD 2015",
        license_note="仅限学术研究使用；不得再分发原始评论内容。",
        label_semantics="原始 label: 1=推荐，-1=过滤 → 统一 label=1 表示虚假",
        has_text=True,
        expected_rows=359052,
        notes="纽约酒店评论。官方发布页需邮件申请，本项目在无法取得时如实标为缺失，不编造替代数据。",
    ),
    "yelpzip": DatasetSpec(
        name="yelpzip",
        display_name="YelpZip",
        homepage="https://shebuti.com/yelpzip-dataset/",
        download_page="https://www.modelscope.cn/datasets/jimmyliu12345/yelpzip（Rayana 原始数据的公开镜像）",
        citation="Rayana & Akoglu, Collective Opinion Spam Detection: Bridging Review Networks and Metadata, KDD 2015",
        license_note=(
            "原始数据仅限学术研究使用。本项目通过公开镜像取得，镜像页未声明额外许可；"
            "引用时仍应引用 Rayana & Akoglu (KDD 2015)。"
        ),
        label_semantics="CSV 内 label 列已为 -1=过滤（虚假）/1=推荐，统一翻转成 label=1 表示虚假",
        has_text=True,
        expected_rows=608598,
        notes="覆盖纽约州及邻近州餐厅评论；镜像文件为带文本的 CSV，行数与官方公布规模做交叉核对。",
    ),
    "fraudsquad_amazon": _fraudsquad_spec("fraudsquad_amazon", "FraudSquad-Amazon（合并视图）"),
    "fraudsquad_llama3": _fraudsquad_spec("fraudsquad_llama3", "FraudSquad-Amazon-Llama3",
                                          single_model="llama3_8b"),
    "fraudsquad_qwen2": _fraudsquad_spec("fraudsquad_qwen2", "FraudSquad-Amazon-Qwen2",
                                         single_model="qwen_72b"),
    "fraudsquad_qwen_dsr1": _fraudsquad_spec("fraudsquad_qwen_dsr1", "FraudSquad-Amazon-Qwen-DSR1",
                                             single_model="qwen_dsr1"),
    "yelpchi_injected": DatasetSpec(
        name="yelpchi_injected",
        display_name="YelpChi + 300 注入账号（自建）",
        homepage="本项目 step3 注入实验（见 data_dictionary.md 第三节）",
        download_page="由 data/processed/injection_log_v2.json + YelpChi 原始评论重建",
        citation="本项目自建；注入模板与物理校验见 data_dictionary.md",
        license_note="派生数据，继承 YelpChi 的使用限制。",
        label_semantics="注入事件 label=1（is_injected=1），其余沿用 YelpChi 平台标签",
        has_text=False,
        notes="保留课题组原有的 300 个人工注入账号，作为四个公开数据集之外的行为注入实验。",
    ),
}

#: 参与统一构建的数据集（yelpchi_injected 为派生集，单独构建）
PUBLIC_DATASETS: Sequence[str] = (
    "yelpchi", "yelpnyc", "yelpzip",
    "fraudsquad_amazon", "fraudsquad_llama3", "fraudsquad_qwen2", "fraudsquad_qwen_dsr1",
)


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------

def empty_reviews_frame() -> pd.DataFrame:
    """返回一个符合 schema 的空表（列顺序/类型都正确），用于占位与测试。"""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DTYPES.items()})


_DTYPES = {
    "dataset": "object",
    "review_id": "object",
    "global_user_id": "object",
    "user_id": "object",
    "item_id": "object",
    "rating": "float32",
    "date": "datetime64[ns]",
    "label": "int8",
    "label_source": "object",
    "is_injected": "int8",
    "text_available": "int8",
    "text": "object",
}


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """按 schema 强制列类型与列顺序，不改动取值（取值问题由 validate_reviews 负责）。"""
    out = df.copy()
    for col, dtype in _DTYPES.items():
        if col not in out.columns:
            raise SchemaError(f"缺少必需列: {col}；当前列: {list(out.columns)}")
        if dtype == "datetime64[ns]":
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype == "int8":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int8")
        elif dtype == "float32":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        else:
            out[col] = out[col].astype("object").where(out[col].notna(), "")
            out[col] = out[col].map(lambda v: v if isinstance(v, str) else str(v))
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def validate_reviews(df: pd.DataFrame, *, strict: bool = True, context: str = "") -> pd.DataFrame:
    """检查统一表是否满足契约，返回按 date 排序后的副本。

    `strict=True` 时任何硬约束违反都抛 SchemaError；`strict=False` 时只收集问题
    并返回（供质量报告使用）。硬约束包括：列齐全、label ∈ {0,1}、date 不为空、
    rating ∈ [0,5] 或 NaN、review_id 在数据集内唯一、user_id/item_id 非空。
    """
    where = f"[{context}] " if context else ""
    if df is None or len(df) == 0:
        raise SchemaError(f"{where}数据为空，无法校验")

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{where}缺少必需列: {missing}")
    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if extra:
        raise SchemaError(f"{where}存在未在 schema 中声明的列: {extra}（请先扩展 schema 再写入）")

    problems: List[str] = []

    bad_label = ~df["label"].isin([0, 1])
    if bad_label.any():
        problems.append(f"label 出现非 0/1 取值 {int(bad_label.sum())} 行: {sorted(df.loc[bad_label, 'label'].unique())[:5]}")

    bad_src = ~df["label_source"].isin(LABEL_SOURCE_VALUES)
    if bad_src.any():
        problems.append(f"label_source 非法取值 {int(bad_src.sum())} 行: {sorted(df.loc[bad_src, 'label_source'].unique())[:5]}")

    na_date = df["date"].isna()
    if na_date.any():
        problems.append(f"date 为空 {int(na_date.sum())} 行")

    r = df["rating"]
    bad_rating = r.notna() & ((r < 0) | (r > 5))
    if bad_rating.any():
        problems.append(f"rating 越界 [0,5] 共 {int(bad_rating.sum())} 行，例: {df.loc[bad_rating, 'rating'].head(3).tolist()}")

    for col in ("user_id", "item_id"):
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if blank.any():
            problems.append(f"{col} 为空 {int(blank.sum())} 行")

    dup = df["review_id"].duplicated()
    if dup.any():
        problems.append(f"review_id 重复 {int(dup.sum())} 行，例: {df.loc[dup, 'review_id'].head(3).tolist()}")

    if problems:
        msg = f"{where}统一表校验失败：\n  - " + "\n  - ".join(problems)
        if strict:
            raise SchemaError(msg)
        # 非严格模式下把问题写进属性，供质量报告取用
        df.attrs["schema_problems"] = problems

    return df.sort_values(["date", "review_id"], kind="mergesort").reset_index(drop=True)


def summary_stats(df: pd.DataFrame) -> Dict[str, object]:
    """一行式统计，写进 manifest 和质量报告。"""
    if len(df) == 0:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "fake_rows": int((df["label"] == 1).sum()),
        "fake_ratio": round(float((df["label"] == 1).mean()), 6),
        "has_text_rows": int(df["text_available"].sum()),
        "date_min": str(pd.to_datetime(df["date"]).min()),
        "date_max": str(pd.to_datetime(df["date"]).max()),
        "duplicated_user_item_date": int(df.duplicated(["user_id", "item_id", "date"]).sum()),
    }


def label_from_yelp(raw: pd.Series) -> pd.Series:
    """Yelp 官方编码 -> 统一编码。

    真实数据里三种写法都出现过：数字 {1, -1}（Rayana 原始）、数字 {1, 0}（部分再加工
    版本），以及文字 {"fake", "genuine"}（少数重组版）。这里用显式白名单判定，
    遇到未知取值直接报错——宁可停下来，也不要把未知标签悄悄当成真实样本。
    """
    s = pd.Series(raw)
    numeric = pd.to_numeric(s, errors="coerce")
    textual = s.astype(str).str.strip().str.lower()

    fake_tokens = {"fake", "fraud", "spam", "filtered", "deceptive", "spammer", "1"}
    real_tokens = {"genuine", "real", "normal", "recommended", "truthful", "authentic", "-1", "0"}

    unknown = sorted(set(s[~numeric.isin([1, -1, 0])].astype(str).unique()) - fake_tokens - real_tokens)
    if unknown:
        raise SchemaError(
            f"Yelp label 出现未知取值 {unknown[:5]}；只接受数字 1/-1/0 或 "
            f"文字 {sorted(fake_tokens | real_tokens)}。请确认数据版本，不要假定映射方向。"
        )

    is_fake = numeric.isin([-1, 0]) | textual.isin(fake_tokens - {"1"})
    return is_fake.astype("int8")
