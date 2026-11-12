"""数据集清单：每个数据集的官方出处、候选下载地址、使用条款。

本文件是"数据从哪来"的唯一事实来源。README、manifest、provenance 文档都从这里取值，
避免同一份信息在三处出现不同版本。

关于镜像：YelpChi / YelpNYC / YelpZip 的官方发布页（shebuti.com）要求邮件申请，
不提供直链。本项目实际取数路径见每个 RemoteFile 的 note，且会在
`docs/data/provenance.md` 里逐条写清"官方源 / 实际取得的镜像 / 校验方式"。
"""

from __future__ import annotations

from typing import Dict, List

from .download import RemoteFile
from .schema import DATASET_SPECS

#: GitHub API 的 raw 内容接口（直连 raw.githubusercontent 被网络策略拦截时的回退）
_GH_API_RAW = {"Accept": "application/vnd.github.raw", "X-GitHub-Api-Version": "2022-11-28"}

_FRAUDSQUAD_REPO = "RafaelLiuZH/FraudSquad"
_FRAUDSQUAD_BRANCH = "main"
_FRAUDSQUAD_RAW = f"https://raw.githubusercontent.com/{_FRAUDSQUAD_REPO}/{_FRAUDSQUAD_BRANCH}"
_FRAUDSQUAD_BASE_API = f"https://api.github.com/repos/{_FRAUDSQUAD_REPO}/contents"


def _fraudsquad_file(relpath: str, size: int, note: str = "") -> RemoteFile:
    """FraudSquad 仓库里的一个文件，按可用性排序给出多个镜像。

    顺序说明（实测于本项目所在的受限网络环境）：
    1. `ghfast.top`：raw 代理，不限 20MB，实测 ~300KB/s；
    2. `cdn.jsdmirror.com`：jsDelivr 的国内镜像，速度相近；
    3. `cdn.jsdelivr.net`：官方 CDN，单文件上限 20MB；
    4. `api.github.com`：官方内容 API，能下但限速严重，仅作最后兜底。

    落盘位置统一扁平化到 `data/raw/fraudsquad/<文件名>`，不保留仓库内的多级目录。
    """
    name = relpath.rsplit("/", 1)[-1]
    return RemoteFile(
        relpath=f"fraudsquad/{name}",
        urls=[
            f"https://ghfast.top/{_FRAUDSQUAD_RAW}/{relpath}",
            f"https://cdn.jsdmirror.com/gh/{_FRAUDSQUAD_REPO}@{_FRAUDSQUAD_BRANCH}/{relpath}",
            f"https://cdn.jsdelivr.net/gh/{_FRAUDSQUAD_REPO}@{_FRAUDSQUAD_BRANCH}/{relpath}",
            f"{_FRAUDSQUAD_BASE_API}/{relpath}",
        ],
        size=size,
        note=note or "FraudSquad 官方实现仓库（RafaelLiuZH/FraudSquad）",
    )


#: 按数据集分组的下载清单
FILES: Dict[str, List[RemoteFile]] = {
    # YelpChi：metadata 已在仓库工作副本中（data/raw/metadata.txt），不重复下载。
    # 若本地缺失，只能走官方邮件申请渠道，不做静默替代。
    "yelpchi": [],

    # YelpNYC：官方仅邮件发布；无可靠直链时不下载，构建阶段会标记为缺失。
    "yelpnyc": [],

    "yelpzip": [
        RemoteFile(
            relpath="yelpzip/yelpzip.csv",
            urls=[
                "https://www.modelscope.cn/api/v1/datasets/jimmyliu12345/yelpzip/repo"
                "?Revision=master&FilePath=yelpzip.csv"
            ],
            size=410_865_364,
            note=(
                "Rayana & Akoglu (KDD 2015) 的 YelpZip 带文本版本，经 ModelScope 公开镜像取得；"
                "官方发布页 shebuti.com 仅提供邮件申请。"
            ),
        )
    ],

    "fraudsquad_amazon": [
        _fraudsquad_file(
            "data/Amazon/amazon_2022_5core.csv.zip",
            18_773_569,
            "真实评论基线：Amazon Reviews 2023 的 5-core 子集（FraudSquad 预处理后）",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_llama3_8b.csv",
            1_433_276,
            "Llama3-8B 生成的虚假评论（大模型水军的实证材料）",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_qwen_72b.csv",
            1_468_340,
            "Qwen-72B 生成的虚假评论",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_qwen_dsr1.csv",
            1_520_028,
            "Qwen-DeepSeek-R1 生成的虚假评论",
        ),
        # 以下为 FraudSquad 论文中"人类撰写的垃圾评论"对照集及其索引文件
        _fraudsquad_file("data/QA.zip", 21_535_547, "FraudSquad 的 Yelp QA 人类垃圾评论集"),
        _fraudsquad_file(
            "data/review_yelpHotelData_NRYRcleaned.txt", 5_062_002, "Yelp 酒店评论原文（FraudSquad 预处理）"
        ),
        _fraudsquad_file(
            "data/label_yelpHotelData_NRYRcleaned.txt", 11_708, "上述评论的标签文件"
        ),
        _fraudsquad_file(
            "data/meta_yelpHotelData_NRYRcleaned.txt", 504_477, "上述评论的元信息（用户/商家/评分/时间）"
        ),
    ],
}


#: 需要人工准备、无法脚本下载的数据集及原因（构建时如实写入报告）
MANUAL_SOURCES: Dict[str, str] = {
    "yelpchi": (
        "官方页 https://shebuti.com/yelpchi-dataset/ 要求邮件申请（srayana@cs.stonybrook.edu）。"
        "本项目使用已随仓库流转的 metadata.txt（67,395 条，与官方公布规模一致），"
        "该版本不含评论文本，故 text_available=0。"
    ),
    "yelpnyc": (
        "官方页 https://shebuti.com/yelpnyc-dataset/ 要求邮件申请，未提供直链；"
        "本项目在无法取得时如实标记为缺失，不用其他数据顶替。"
    ),
}


def files_for(dataset: str) -> List[RemoteFile]:
    if dataset not in FILES:
        raise KeyError(f"未知数据集 {dataset}；可选: {sorted(FILES)}")
    return FILES[dataset]


def license_table() -> List[dict]:
    """生成给 README / provenance 用的许可与引用表。"""
    rows = []
    for name, spec in DATASET_SPECS.items():
        rows.append(
            {
                "dataset": name,
                "display_name": spec.display_name,
                "homepage": spec.homepage,
                "download_page": spec.download_page,
                "citation": spec.citation,
                "license": spec.license_note,
                "label_semantics": spec.label_semantics,
                "has_text": spec.has_text,
                "expected_rows": spec.expected_rows,
                "notes": spec.notes,
            }
        )
    return rows
