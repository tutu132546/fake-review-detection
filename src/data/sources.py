"""数据集清单：每个数据集的官方出处、候选下载地址、校验码与使用条款。

本文件是"数据从哪来"的唯一事实来源。README、manifest、provenance 文档都从这里取值，
避免同一份信息在三处出现不同版本。

关于校验码（`sha256`）
----------------------
所有远端文件的 sha256 都**写死在代码里**，下载完成后逐个比对，不匹配就重新下载并在
多次失败后报错。因为本项目只能经第三方代理取 GitHub 文件（直连被网络策略拦截），
"代理有没有改过文件"必须能自证，不能只看大小。

这些 sha256 是本地对文件逐字节算出来的，可信度来自两级核对：
1. `OFFICIAL_GIT_BLOBS` 记录 GitHub 官方仓库 API 返回的 git blob sha1；
2. `tests/test_data_sources.py` 会把本地文件按 git 对象格式重算 sha1 与之比对。
两者一致 ⇒ 本地字节 == 官方仓库字节 ⇒ 由它算出的 sha256 就是官方文件的 sha256。

关于镜像
--------
YelpChi / YelpNYC / YelpZip 的官方发布页（shebuti.com）要求邮件申请，不提供直链。
实际取数路径见每个 RemoteFile 的 note，逐条核对记录写在 `docs/data/provenance.md`。
"""

from __future__ import annotations

from typing import Dict, List

from .download import RemoteFile
from .schema import DATASET_SPECS

_FRAUDSQUAD_REPO = "RafaelLiuZH/FraudSquad"
_FRAUDSQUAD_BRANCH = "main"
_FRAUDSQUAD_RAW = f"https://raw.githubusercontent.com/{_FRAUDSQUAD_REPO}/{_FRAUDSQUAD_BRANCH}"
_FRAUDSQUAD_BASE_API = f"https://api.github.com/repos/{_FRAUDSQUAD_REPO}/contents"


def _fraudsquad_file(relpath: str, size: int, sha256: str, note: str = "") -> RemoteFile:
    """FraudSquad 仓库里的一个文件，按可用性排序给出多个镜像。

    顺序说明（实测于本项目所在的受限网络环境）：
    1. `ghfast.top`：raw 代理，不限 20MB，实测 ~300KB/s；
    2. `cdn.jsdmirror.com`：jsDelivr 的国内镜像，速度相近；
    3. `cdn.jsdelivr.net`：官方 CDN，单文件上限 20MB；
    4. `api.github.com`：官方内容 API，能下但限速严重，仅作最后兜底。

    落盘位置统一扁平化到 `data/raw/fraudsquad/<文件名>`，不保留仓库内的多级目录。
    `sha256` 必填：四个镜像里任何一个被中间层改动过，校验都会当场失败。
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
        sha256=sha256,
        note=note or "FraudSquad 官方实现仓库（RafaelLiuZH/FraudSquad）",
    )


#: 官方仓库中 data/ 下每个文件的 git blob sha1（GitHub Trees API 返回）
#: 用途：证明"本地文件 = 官方文件"，从而让写死的 sha256 站得住
OFFICIAL_GIT_BLOBS: Dict[str, str] = {
    "amazon_2022_5core.csv.zip": "267fd8efdeec07a29de79615453bdb9a7b364f6b",
    "reviews_from_llama3_8b.csv": "b7f3527f5dd4363c891b88af67dd6510d577e033",
    "reviews_from_qwen_72b.csv": "742792c762e3d1fc47af05d9a4083505101cf051",
    "reviews_from_qwen_dsr1.csv": "5d97703d89e5b1aa976b16e44f0f26c4d91f3c15",
    "QA.zip": "6d4acfb1606321e82565e300e1cc2b71f77adc6b",
    "label_yelpHotelData_NRYRcleaned.txt": "a63e068462defeca390c7810da0a711d1e199aea",
    "meta_yelpHotelData_NRYRcleaned.txt": "3ac333da53dcf9e8222b180d777f58a0d979bf32",
    "review_yelpHotelData_NRYRcleaned.txt": "f7ab0974a6e1664f1aef3c3272baff4be7aa7f7c",
    "yelpHotelData_NRYRcleaned.rtf": "3621bbb147dda662c1c9bfb5d7e91aa9ae7f9855",
}

#: YelpZip 镜像文件的校验码。镜像本身不提供官方摘要，只能以"文件大小 + 内部结构核对"
#: 佐证（见 results/data_quality_report.md 第 2 节：608,598 行、5,044 商户、用户并集
#: 与官方公布的 260,277 完全一致）。sha256 在此固定下来，用于发现"换了一次镜像、
#: 内容变了"的情况——它是本项目的**版本指纹**，不等于官方发布的摘要。
#: 佐证 2：科学数据银行「Yelp dataset」（dataset.zip，MD5 7c6c35041c4eabb65aa024f2f1ecff74）
#: 内含的 yelpzip.csv 与本文件 sha256 完全相同 —— 两个不同来源的镜像字节一致，
#: 说明文件里的 140 行残缺是这份流通导出自带的，不是本项目下载或解析引入的。
_YELPZIP_MIRROR_SHA256 = "78f92e83574208f896328c5db99ac3d18644bb34eb457054bda235b2a8e517fe"
_YELPZIP_MIRROR_SIZE = 410_865_364


#: 按数据集分组的下载清单
FILES: Dict[str, List[RemoteFile]] = {
    # YelpChi：metadata 已在工作副本中（data/raw/yelpchi/metadata.txt），不重复下载。
    # 带评论文本的完整版只能走官方邮件申请渠道，不做静默替代。
    "yelpchi": [],

    # YelpNYC：来自作者 Dropbox 发布包（不在脚本下载清单内），
    # 需手工把 metadata 与 reviewContent 放进 data/raw/yelpnyc/，见 provenance.md §2.2
    "yelpnyc": [],

    "yelpzip": [
        RemoteFile(
            relpath="yelpzip/yelpzip.csv",
            urls=[
                "https://www.modelscope.cn/api/v1/datasets/jimmyliu12345/yelpzip/repo"
                "?Revision=master&FilePath=yelpzip.csv"
            ],
            size=_YELPZIP_MIRROR_SIZE,
            sha256=_YELPZIP_MIRROR_SHA256,
            note=(
                "Rayana & Akoglu (KDD 2015) 的 YelpZip 带文本版本，经 ModelScope 公开镜像取得；"
                "官方发布页 shebuti.com 仅提供邮件申请。该镜像文件有 140 行导出残缺"
                "（缺 rating/label），已在读取阶段识别并隔离，详见质量报告第 2 节。"
            ),
        )
    ],

    "fraudsquad_llama3": [
        _fraudsquad_file(
            "data/Amazon/amazon_2022_5core.csv.zip",
            18_773_569,
            "858b1f2f51d5fd78c5b084779b1cf2d659f53f92278dee745d1dd2151b58b130",
            "真实评论基线：Amazon Reviews 2023 的 5-core 子集（FraudSquad 预处理后）",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_llama3_8b.csv",
            1_433_276,
            "650d5fa18c527b2a202f2d0c576ec73c4a0399991e8554b59a6d9a452870fbf6",
            "Llama3-8B 生成的虚假评论（大模型水军的实证材料）",
        ),
    ],
    "fraudsquad_qwen2": [
        _fraudsquad_file(
            "data/Amazon/amazon_2022_5core.csv.zip",
            18_773_569,
            "858b1f2f51d5fd78c5b084779b1cf2d659f53f92278dee745d1dd2151b58b130",
            "真实评论基线（与 llama3 表同一份，便于同基线对比）",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_qwen_72b.csv",
            1_468_340,
            "71cd6f81dfdb617687b92e29cca3a3664f7792971cd29c6baf1c532381db9e32",
            "Qwen-72B 生成的虚假评论",
        ),
    ],
    "fraudsquad_qwen_dsr1": [
        _fraudsquad_file(
            "data/Amazon/amazon_2022_5core.csv.zip",
            18_773_569,
            "858b1f2f51d5fd78c5b084779b1cf2d659f53f92278dee745d1dd2151b58b130",
            "真实评论基线（与 llama3 表同一份，便于同基线对比）",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_qwen_dsr1.csv",
            1_520_028,
            "e706b4341bd9d1491110575b1187bfbc13cea643d47417dcada91e84eedbea4e",
            "Qwen-DeepSeek-R1 生成的虚假评论",
        ),
    ],

    # 合并视图沿用三个单模型的全部文件，故单独列一份（download 会去重跳过已存在的）
    "fraudsquad_amazon": [
        _fraudsquad_file(
            "data/Amazon/amazon_2022_5core.csv.zip",
            18_773_569,
            "858b1f2f51d5fd78c5b084779b1cf2d659f53f92278dee745d1dd2151b58b130",
            "真实评论基线",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_llama3_8b.csv",
            1_433_276,
            "650d5fa18c527b2a202f2d0c576ec73c4a0399991e8554b59a6d9a452870fbf6",
            "Llama3-8B 生成的虚假评论",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_qwen_72b.csv",
            1_468_340,
            "71cd6f81dfdb617687b92e29cca3a3664f7792971cd29c6baf1c532381db9e32",
            "Qwen-72B 生成的虚假评论",
        ),
        _fraudsquad_file(
            "data/Amazon/reviews_from_qwen_dsr1.csv",
            1_520_028,
            "e706b4341bd9d1491110575b1187bfbc13cea643d47417dcada91e84eedbea4e",
            "Qwen-DeepSeek-R1 生成的虚假评论",
        ),
        # 以下两组是 FraudSquad 论文里"人类撰写的垃圾评论"对照材料，**不参与本项目建模**，
        # 仅作为来源留存，便于日后复核论文设置。注意它们与 Yelp 无关：
        # QA.zip 是 THUIR WWW'17 的中文社区问答（CQA）协同刷分数据，论文中称 ChineseQA。
        _fraudsquad_file(
            "data/QA.zip",
            21_535_547,
            "e43276455d9cb2757abde563b5978703c6cdaac794a8ccd3f532402d1504488c",
            "中文社区问答（CQA）协同刷分数据（FraudSquad 论文中称 ChineseQA），"
            "来源 Liu et al., WWW 2017《Detecting Collusive Spamming Activities in "
            "Community Question Answering》（THUIR）；**不是 Yelp 数据**",
        ),
        _fraudsquad_file(
            "data/review_yelpHotelData_NRYRcleaned.txt",
            5_062_002,
            "268502eb5f926c37a9299db68a135c5c3d76e64a5a0f514a43c16876c994c798",
            "Yelp 酒店评论原文（Rayana 三数据集之外的第四个 Yelp 切片，5,854 条，"
            "仅约为官方 YelpNYC 的 1.6%，**不可当作 YelpNYC**）",
        ),
        _fraudsquad_file(
            "data/label_yelpHotelData_NRYRcleaned.txt",
            11_708,
            "ff8f6633cde64869fef71b27bebdcb79c57de9e4c1f14f1b4f4d9b0d8af92421",
            "上述 Yelp 酒店评论的标签文件",
        ),
        _fraudsquad_file(
            "data/meta_yelpHotelData_NRYRcleaned.txt",
            504_477,
            "8fddb66e457585472ae7cb9ab994f7a85d465d0c703e4418ef83a047a3bfc706",
            "上述 Yelp 酒店评论的元信息（用户/商家/评分/时间）",
        ),
    ],
}


#: 需要人工准备、无法脚本下载的数据集及原因（构建时如实写入报告）
MANUAL_SOURCES: Dict[str, str] = {
    "yelpchi": (
        "官方页 https://shebuti.com/yelpchi-dataset/ 要求邮件申请（srayana@cs.stonybrook.edu）。"
        "本项目使用已随项目流转的 metadata.txt（67,395 条，与官方公布规模完全一致），"
        "该版本不含评论文本，故 text_available=0；文本版申请未获回复，缺口见 provenance.md。"
    ),
    "yelpnyc": (
        "**已取得**。官方页 https://shebuti.com/yelpnyc-dataset/ 只提供邮件申请；邮件未获回复后，"
        "改用官方页链接的作者 Dropbox 共享目录 "
        "https://www.dropbox.com/sh/iqcuj0363zcj3go/AAAvbZVR_PSNyJX8AXUXpBqea?dl=0 ，"
        "下载到发布包 15-collective_opinion_spam_detection.zip（sha256 "
        "f16f20032a8efe9c456a470037b14c3d3c1d5498ef5dab9321c82ecc65fa6a5e），"
        "内含 metadata 与 reviewContent 两个文件，需手工放入 data/raw/yelpnyc/。"
        "该包同时带 SpEagle 的 MATLAB 代码与 SentiWordNet 词表，本项目只用其中两个数据文件。"
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


def all_pinned_checksums() -> Dict[str, str]:
    """所有被写死校验码的文件：{落盘相对路径: sha256}，供测试与文档引用。"""
    out: Dict[str, str] = {}
    for specs in FILES.values():
        for spec in specs:
            if spec.sha256:
                out[spec.relpath] = spec.sha256
    return out
