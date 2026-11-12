"""统一评论级 schema：列定义、数据集元信息、标签语义与校验规则。

设计原则
--------
1. **单一方向**：`label` 一律 1 = 虚假评论（水军写的 / LLM 生成的 / 人工注入的），
   0 = 真实评论。原始编码（Yelp 的 -1/1、0/1 等）到统一编码的翻转**只在本文件
   的 `label_from_yelp()` 里发生一次**，且必须显式声明编码方案——见该函数。
2. **不猜不填**：字段缺失就是缺失（text 为空串 + `text_available=False`，
   rating 缺失为 NaN），绝不静默补 0 或补均值。缺失量会进质量报告。
3. **不猜方向**：`{0,1}` 这种编码在"1=假"与"1=真"两种约定下都出现过，单看数据
   无法判断。遇到它一律抛 `SchemaError` 要求显式指定 scheme，绝不默认取一种。
4. **可追溯**：`label_source` 记录标签是怎么来的（平台过滤器 / LLM 生成 /
   众包写作 / 我们注入），因为"公平对比"要求说明每种方法的监督信号强度。
5. **评论级 + 账号级双视图**：主表是评论级（唯一稳定的公共粒度）；账号级由
   `build_account_table()` 从评论级聚合得到，标签口径见 `account_label_from_reviews()`。
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
    "human_generated",   # 匿名众包/社区问答平台上的真人协同刷分（FraudSquad 的 QA 集，
                         # 实为 THUIR WWW'17 中文社区问答数据，与 Yelp 无关）
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
    #: 是否自带评论文本
    has_text: bool
    #: 官方公布的规模，用于交叉核对（收不到就如实标 missing）
    expected_rows: Optional[int] = None
    notes: str = ""
    #: 派生数据集的父数据集（如 FraudSquad 单模型表派生自合并前同一批原始文件）
    derived_from: Optional[str] = None
    #: 只含某一个生成模型时，记录模型标识（写入 manifest 便于对齐论文表格）
    single_model: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


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
        download_page="https://shebuti.com/yelpnyc-dataset/（官方仅邮件申请，无直链）",
        citation="Rayana & Akoglu, Collective Opinion Spam Detection: Bridging Review Networks and Metadata, KDD 2015",
        license_note="仅限学术研究使用；不得再分发原始评论内容。",
        label_semantics="原始 label: 1=推荐，-1=过滤 → 统一 label=1 表示虚假",
        has_text=True,
        expected_rows=359052,
        notes=(
            "纽约市的餐厅（restaurant）评论，官方公布 359,052 条 / 160,264 用户 / 923 家商户，13.22% 被过滤。"
            "注意不是酒店数据——项目早期文档误写作「酒店」，已订正。"
            "官方发布页只提供邮件申请渠道（srayana@cs.stonybrook.edu），本项目申请未获回复，"
            "如实标为缺失，不用其他数据顶替；备选方案见 docs/data/yelpnyc_fallback.md。"
        ),
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
        notes=(
            "覆盖纽约市及周边郊区的餐厅评论（官方公布 608,598 条 / 260,277 用户 / 5,044 家商户）。"
            "所用镜像文件共 608,598 个数据行，其中 140 行是导出残缺行（Tab 分隔、缺 rating 与 label），"
            "可用带标签记录为 608,458 条；差额已定位并写入质量报告第 2 节，不是解析丢行。"
        ),
    ),
    "fraudsquad_amazon": DatasetSpec(
        name="fraudsquad_amazon",
        display_name="FraudSquad-Amazon（三模型合并视图）",
        homepage="https://github.com/RafaelLiuZH/FraudSquad",
        download_page="https://github.com/RafaelLiuZH/FraudSquad/tree/main/data（官方实现仓库）",
        citation=(
            "Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model "
            "Embeddings and Graph Neural Network (FraudSquad), 2025"
        ),
        license_note=(
            "仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，"
            "受 Amazon 数据条款约束；生成部分由作者用 Llama3-8B / Qwen 系列产出。"
            "本项目仅作学术研究引用，不再分发原始文件。"
        ),
        label_semantics="真实评论 label=0（label_source=real_baseline）；生成评论 label=1（label_source=llm_generated）",
        has_text=True,
        expected_rows=94242,
        notes=(
            "三个模型生成的假评论合并成的**视图**，仅为便于总览，**不作为论文对比基准**："
            "论文里三个模型是三个独立数据集，合并后无法与原文数字对齐。"
            "正式结果请用 fraudsquad_llama3 / fraudsquad_qwen2 / fraudsquad_qwen_dsr1。"
            "合并视图不进入 all_reviews.parquet，避免与三个单模型表重复计数。"
        ),
    ),
    "fraudsquad_llama3": DatasetSpec(
        name="fraudsquad_llama3",
        display_name="FraudSquad-Amazon-Llama3",
        homepage="https://github.com/RafaelLiuZH/FraudSquad",
        download_page="https://github.com/RafaelLiuZH/FraudSquad/tree/main/data（官方实现仓库）",
        citation=(
            "Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model "
            "Embeddings and Graph Neural Network (FraudSquad), 2025"
        ),
        license_note=(
            "仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，"
            "受 Amazon 数据条款约束；生成部分由作者用 Llama3-8B 产出。"
            "本项目仅作学术研究引用，不再分发原始文件。"
        ),
        label_semantics="真实评论 label=0；Llama3-8B 生成的假评论 label=1",
        has_text=True,
        expected_rows=89247,
        derived_from="fraudsquad_amazon",
        single_model="llama3_8b",
        notes=(
            "真实基线 amazon_2022_5core（86,758 条 / 8,092 个账号）+ 仅 Llama3-8B 生成的 2,489 条。"
            "结构对齐论文：1,250 个既有账号被接管，每个账号写 2 条假评论（少数 1 条）。"
        ),
    ),
    "fraudsquad_qwen2": DatasetSpec(
        name="fraudsquad_qwen2",
        display_name="FraudSquad-Amazon-Qwen2",
        homepage="https://github.com/RafaelLiuZH/FraudSquad",
        download_page="https://github.com/RafaelLiuZH/FraudSquad/tree/main/data（官方实现仓库）",
        citation=(
            "Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model "
            "Embeddings and Graph Neural Network (FraudSquad), 2025"
        ),
        license_note=(
            "仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，"
            "受 Amazon 数据条款约束；生成部分由作者用 Qwen-72B 产出。"
            "本项目仅作学术研究引用，不再分发原始文件。"
        ),
        label_semantics="真实评论 label=0；Qwen-72B 生成的假评论 label=1",
        has_text=True,
        expected_rows=89253,
        derived_from="fraudsquad_amazon",
        single_model="qwen_72b",
        notes=(
            "真实基线 86,758 条 + 仅 Qwen-72B 生成的 2,495 条；1,250 个被接管账号。"
            "**这 1,250 个账号与 fraudsquad_llama3 完全相同**（但 qwen_dsr1 是另一批，交集仅 309）："
            "与 llama3 之间属同账号配对比较，可减少账号侧混杂，但也不能当独立样本做显著性检验。"
        ),
    ),
    "fraudsquad_qwen_dsr1": DatasetSpec(
        name="fraudsquad_qwen_dsr1",
        display_name="FraudSquad-Amazon-Qwen-DSR1",
        homepage="https://github.com/RafaelLiuZH/FraudSquad",
        download_page="https://github.com/RafaelLiuZH/FraudSquad/tree/main/data（官方实现仓库）",
        citation=(
            "Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model "
            "Embeddings and Graph Neural Network (FraudSquad), 2025"
        ),
        license_note=(
            "仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，"
            "受 Amazon 数据条款约束；生成部分由作者用 Qwen-DeepSeek-R1 产出。"
            "本项目仅作学术研究引用，不再分发原始文件。"
        ),
        label_semantics="真实评论 label=0；Qwen-DeepSeek-R1 生成的假评论 label=1",
        has_text=True,
        expected_rows=89258,
        derived_from="fraudsquad_amazon",
        single_model="qwen_dsr1",
        notes=(
            "真实基线 86,758 条 + 仅 Qwen-DeepSeek-R1 生成的 2,500 条；1,250 个被接管账号，每人恰好 2 条。"
            "**这批账号与 llama3 / qwen2 那批不是同一批**（交集仅 309 个），因此与它们之间只能做独立样本比较，"
            "不能做配对比较；三者并集 2,191 个账号，正是合并视图的正例数。"
        ),
    ),
    "yelpchi_injected": DatasetSpec(
        name="yelpchi_injected",
        display_name="YelpChi + 300 注入账号（自建）",
        homepage="本项目 step3 注入实验（见 data_dictionary.md 第三节）",
        download_page="由 data/processed/injection_log_v2.json + YelpChi 原始评论重建",
        citation="本项目自建；注入模板与物理校验见 data_dictionary.md",
        license_note="派生数据，继承 YelpChi 的使用限制。",
        label_semantics="注入事件 label=1（is_injected=1），其余沿用 YelpChi 平台标签",
        has_text=False,
        expected_rows=70186,
        derived_from="yelpchi",
        notes=(
            "保留课题组原有的 300 个人工注入账号（展开为 2,791 条注入评论），"
            "与 FraudSquad 的「接管既有账号」结构同源，可作为方法迁移性的对照。"
        ),
    ),
}

#: 参与统一构建的公开数据集（yelpchi_injected 为派生集，单独构建）
PUBLIC_DATASETS: Sequence[str] = (
    "yelpchi",
    "yelpnyc",
    "yelpzip",
    "fraudsquad_llama3",
    "fraudsquad_qwen2",
    "fraudsquad_qwen_dsr1",
)

#: 只作总览、不进入合并大表的视图（否则与三个单模型表重复计数）
VIEW_ONLY_DATASETS: Sequence[str] = ("fraudsquad_amazon",)



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


# --------------------------------------------------------------------------
# 原始标签 -> 统一标签（方向只在这里翻转一次）
# --------------------------------------------------------------------------

#: 允许的原始编码方案。键名即调用方要显式声明的 scheme。
#: 每一组都必须给出**完整**的映射表：取到表外的值一律报错，不做默认猜测。
YELP_LABEL_SCHEMES: Dict[str, Dict[int, int]] = {
    # Rayana & Akoglu 原始发布版本：1 = 推荐（真实），-1 = 被 Yelp 过滤（虚假）
    "signed_1_real_minus1_fake": {1: 0, -1: 1},
    # 部分再加工版本用 0/1：1 = 虚假，0 = 真实
    "binary_1_fake": {1: 1, 0: 0},
    # 另一种再加工版本用 0/1：0 = 虚假，1 = 真实
    "binary_0_fake": {0: 1, 1: 0},
}

#: 文字标签（少数重组版把 label 写成字符串）。注意这里**不含** "1"/"0"/"-1"：
#: 数字形式的取值走数值通道，混进文字通道会造成方向冲突。
YELP_LABEL_TOKENS: Dict[str, int] = {
    "fake": 1, "fraud": 1, "spam": 1, "filtered": 1, "deceptive": 1, "spammer": 1,
    "genuine": 0, "real": 0, "normal": 0, "recommended": 0, "truthful": 0, "authentic": 0,
}


def guess_yelp_label_scheme(raw: pd.Series) -> str:
    """在**能确定**的前提下推断编码方案，否则抛错。

    只有 `{-1, 1}` 是无歧义的：Yelp 官方就是「1 推荐 / -1 过滤」，不存在反过来的约定，
    因此可以安全地自动识别。而 `{0, 1}` 在文献里两种方向都出现过（有的版本 1=假，
    有的版本 1=真），单看数据无法判断——这种情况必须由调用方显式指定 scheme，
    否则整张表的标签方向可能整体颠倒，而这类错误在下游极难发现。
    """
    s = pd.Series(raw)
    numeric = pd.to_numeric(s, errors="coerce")
    leftover = s[numeric.isna()].astype(str).str.strip().str.lower()
    values = set(numeric.dropna().astype(int).unique().tolist())
    unknown_text = sorted(set(leftover) - set(YELP_LABEL_TOKENS))

    if unknown_text:
        raise SchemaError(
            f"Yelp label 出现无法识别的文字取值 {unknown_text[:5]}；"
            f"已知文字标签: {sorted(YELP_LABEL_TOKENS)}。请确认数据版本后显式指定 scheme。"
        )
    if not values:
        if leftover.empty:
            raise SchemaError("Yelp label 全为空，无法判定。")
        # 全为文字标签时不需要数值方案，直接走文字映射
        return "text_only"
    if values <= {-1, 1}:
        return "signed_1_real_minus1_fake"
    if values <= {0, 1}:
        raise SchemaError(
            f"Yelp label 取值为 {sorted(values)}（0/1 编码），无法从数据判断方向："
            "`1=虚假, 0=真实`（scheme='binary_1_fake'）与 `0=虚假, 1=真实`（scheme='binary_0_fake'）"
            "在公开版本中都出现过。为避免整表标签反向，请显式传入 scheme，不要依赖自动推断。"
        )
    raise SchemaError(
        f"Yelp label 出现未知取值 {sorted(values)}；只接受 {{-1,1}}（官方原版）或 {{0,1}}（需显式声明 scheme）。"
        "请确认数据版本，不要假定映射方向。"
    )


def label_from_yelp(raw: pd.Series, *, scheme: str = "auto", context: str = "") -> pd.Series:
    """Yelp 原始编码 -> 统一编码（1 = 虚假，0 = 真实）。

    `scheme="auto"` 只在 `{-1,1}` 时自动判定；`{0,1}` 会抛 `SchemaError` 要求显式指定，
    可选值见 `YELP_LABEL_SCHEMES`。显式指定后，若数据里出现该方案未覆盖的取值也会报错，
    避免"悄悄按真实处理"。
    """
    where = f"[{context}] " if context else ""
    s = pd.Series(raw).reset_index(drop=True)
    numeric = pd.to_numeric(s, errors="coerce")
    resolved_scheme = guess_yelp_label_scheme(s) if scheme == "auto" else scheme

    # 全为文字标签时没有数值方案可套，只做文字映射
    text_only = resolved_scheme == "text_only"
    if not text_only and resolved_scheme not in YELP_LABEL_SCHEMES:
        raise SchemaError(
            f"{where}未知的 label scheme={resolved_scheme!r}；可选: {sorted(YELP_LABEL_SCHEMES)} 或 'auto'"
        )
    table = {} if text_only else YELP_LABEL_SCHEMES[resolved_scheme]

    out = pd.Series(pd.NA, index=s.index, dtype="Int8")
    num_mask = numeric.notna()
    if text_only and num_mask.any():
        raise SchemaError(f"{where}scheme='text_only' 但出现了 {int(num_mask.sum())} 行数值标签，请改用具体 scheme")
    if num_mask.any():
        ints = numeric[num_mask].astype(int)
        bad = sorted(set(ints.unique().tolist()) - set(table))
        if bad:
            raise SchemaError(
                f"{where}label 取值 {bad} 不在 scheme={resolved_scheme!r} 的定义域 {sorted(table)} 内；"
                "请确认 scheme 与数据版本是否匹配。"
            )
        out.loc[num_mask] = ints.map(table).astype("Int8")

    txt_mask = ~num_mask
    if txt_mask.any():
        toks = s[txt_mask].astype(str).str.strip().str.lower()
        bad_tok = sorted(set(toks) - set(YELP_LABEL_TOKENS))
        if bad_tok:
            raise SchemaError(f"{where}无法识别的文字标签 {bad_tok[:5]}；已知: {sorted(YELP_LABEL_TOKENS)}")
        out.loc[txt_mask] = toks.map(YELP_LABEL_TOKENS).astype("Int8")

    if out.isna().any():
        raise SchemaError(f"{where}有 {int(out.isna().sum())} 行标签无法解析")
    return out.astype("int8")


# --------------------------------------------------------------------------
# 账号级视图（从评论级聚合，标签口径单独说明）
# --------------------------------------------------------------------------

#: 账号级表的列契约
ACCOUNT_COLUMNS: List[str] = [
    "dataset",            # str  数据集标识
    "account_id",         # str  数据集内账号 ID（= 评论表的 user_id）
    "global_account_id",  # str  "{dataset}:{account_id}"
    "n_reviews",          # int32 该账号在本数据集内的评论总数（含假评论）
    "n_fake_reviews",     # int32 其中被标为虚假的条数
    "n_injected",         # int32 其中由我们注入脚本写入的条数（无注入时为 0）
    "fake_ratio",         # float32 n_fake / n_reviews
    "n_items",            # int32 涉及的商家/商品数
    "n_fake_items",       # int32 虚假评论涉及的商家数
    "first_date",         # datetime64 首条评论时间
    "last_date",          # datetime64 末条评论时间
    "span_days",          # float32  首末间隔天数
    "account_label",      # int8  1=水军/被接管账号，0=普通账号
    "label_basis",        # str   账号标签的依据（见 ACCOUNT_LABEL_BASIS）
]


#: 账号标签的依据，必须逐条写清，因为"账号级是否算水军"本身是个口径选择
ACCOUNT_LABEL_BASIS: Dict[str, str] = {
    "hijacked_by_llm": (
        "FraudSquad 单模型表：该账号至少写过 1 条由该模型生成的假评论，即论文设定的「被接管账号」。"
        "注意这类账号同时保留其历史真实评论，属混合账号。"
    ),
    "platform_filtered": (
        "Yelp 系列：该账号至少有 1 条评论被平台过滤器判为虚假。阈值取 1 是最宽松口径，"
        "下游若改用比例阈值，请直接用 n_fake_reviews / fake_ratio 重算，不要复用本列。"
    ),
    "injected": "本项目注入实验：该账号被注入脚本写入过评论（用 n_injected 判，不用 label）。",
}


def build_account_table(df: pd.DataFrame, *, label_basis: str, min_fake_reviews: int = 1) -> pd.DataFrame:
    """把评论级表聚合成账号级表。

    `label_basis` 说明账号标签从哪来（必须是 `ACCOUNT_LABEL_BASIS` 的键）。
    账号被标为 1 的条件是「虚假评论条数 >= min_fake_reviews」，默认 1 条。

    **为什么可以做账号级实验**：FraudSquad 的构造方式就是"接管一批既有账号、让它们写假评论"，
    与我们要研究的水军行为（同一个号突然集中写好评）结构一致。账号级实验的正确姿势是
    "在被接管账号 vs 普通账号之间做二分类"，而不是"把账号的所有评论都当假评论"。
    唯一不能做的是**评论级**用「账号是否被接管」当标签——那会把该账号的历史真实评论也标记成假的。
    """
    if label_basis not in ACCOUNT_LABEL_BASIS:
        raise SchemaError(f"未知 label_basis={label_basis!r}；可选: {sorted(ACCOUNT_LABEL_BASIS)}")
    if df is None or len(df) == 0:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _ACCOUNT_DTYPES.items()})[ACCOUNT_COLUMNS]

    src = df[["dataset", "user_id", "item_id", "date", "label"]].copy()
    src["is_fake"] = (src["label"] == 1).astype("int8")
    src["is_injected"] = (
        pd.to_numeric(df["is_injected"], errors="coerce").fillna(0).astype("int8")
        if "is_injected" in df.columns
        else pd.Series(0, index=df.index, dtype="int8")
    )

    keys = ["dataset", "user_id"]
    g = src.groupby(keys, sort=False)
    out = pd.DataFrame(
        {
            "n_reviews": g.size(),
            "n_fake_reviews": g["is_fake"].sum(),
            "n_injected": g["is_injected"].sum(),
            "n_items": g["item_id"].nunique(),
            "first_date": g["date"].min(),
            "last_date": g["date"].max(),
        }
    ).reset_index().rename(columns={"user_id": "account_id"})

    # 虚假评论涉及的商家数：只在假评论子集上数，再用左连接对齐（比 groupby.apply 快一个量级）
    fake_items = (
        src.loc[src["is_fake"] == 1]
        .groupby(keys, sort=False)["item_id"]
        .nunique()
        .rename("n_fake_items")
        .reset_index()
        .rename(columns={"user_id": "account_id"})
    )
    out = out.merge(fake_items, on=["dataset", "account_id"], how="left")
    out["n_fake_items"] = out["n_fake_items"].fillna(0).astype("int32")

    out["global_account_id"] = out["dataset"] + ":" + out["account_id"]
    out["fake_ratio"] = (out["n_fake_reviews"] / out["n_reviews"]).astype("float32")
    out["span_days"] = ((out["last_date"] - out["first_date"]).dt.total_seconds() / 86400.0).astype("float32")
    # 账号标签按 basis 取不同的分子：注入集问"我们注过这个号吗"，
    # 而不是"这个号有没有被平台过滤过"——后者是 Yelp 的标签，两件事不能混。
    driver = out["n_injected"] if label_basis == "injected" else out["n_fake_reviews"]
    out["account_label"] = (driver >= min_fake_reviews).astype("int8")
    out["label_basis"] = label_basis
    for c in ("first_date", "last_date"):
        out[c] = pd.to_datetime(out[c])
    for c in ("n_reviews", "n_fake_reviews", "n_injected", "n_items", "n_fake_items"):
        out[c] = out[c].astype("int32")
    out["fake_ratio"] = out["fake_ratio"].astype("float32")
    out["span_days"] = out["span_days"].astype("float32")
    return out[ACCOUNT_COLUMNS].reset_index(drop=True)


_ACCOUNT_DTYPES = {
    "dataset": "object",
    "account_id": "object",
    "global_account_id": "object",
    "n_reviews": "int32",
    "n_fake_reviews": "int32",
    "n_injected": "int32",
    "fake_ratio": "float32",
    "n_items": "int32",
    "n_fake_items": "int32",
    "first_date": "datetime64[ns]",
    "last_date": "datetime64[ns]",
    "span_days": "float32",
    "account_label": "int8",
    "label_basis": "object",
}


def validate_accounts(df: pd.DataFrame, *, context: str = "") -> pd.DataFrame:
    """账号级表的最小校验：列齐全、label ∈ {0,1}、n_fake <= n_reviews、ID 唯一。"""
    where = f"[{context}] " if context else ""
    missing = [c for c in ACCOUNT_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{where}账号表缺少列 {missing}")
    problems: List[str] = []
    if not df["account_label"].isin([0, 1]).all():
        problems.append("account_label 出现非 0/1")
    bad = df["n_fake_reviews"] > df["n_reviews"]
    if bad.any():
        problems.append(f"有 {int(bad.sum())} 个账号的假评论数大于评论总数")
    bad_inj = df["n_injected"] > df["n_reviews"]
    if bad_inj.any():
        problems.append(f"有 {int(bad_inj.sum())} 个账号的注入评论数大于评论总数")
    dup = df["global_account_id"].duplicated()
    if dup.any():
        problems.append(f"global_account_id 重复 {int(dup.sum())} 个")
    if problems:
        raise SchemaError(f"{where}账号表校验失败：" + "；".join(problems))
    return df

