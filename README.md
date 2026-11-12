# fake-review-detection

虚假评论（水军）检测课程项目：结合无监督异常检测、文本相似度与用户-商品网络分析，识别电商/点评平台上的虚假评论及刷单账号。

## 目录结构

```
fake-review-detection/
├── README.md              # 项目介绍、分工、运行说明
├── data_dictionary.md     # 数据字典（含第十一节：多数据集统一格式）
├── requirements.txt
├── data/
│   ├── raw/               # 原始数据（已加入 .gitignore，不提交到仓库）
│   │   ├── yelpchi/       #   YelpChi metadata
│   │   ├── yelpzip/       #   YelpZip 带文本 CSV
│   │   ├── yelpnyc/       #   YelpNYC（待补）
│   │   └── fraudsquad/    #   FraudSquad-Amazon 真实基线与 LLM 生成集
│   └── processed/
│       └── unified/       # 统一格式 parquet + manifest.json（构建产物）
├── docs/
│   └── data/provenance.md # 数据集来源、核验与使用条款
├── notebooks/             # 探索性分析，每人一个，命名如 01_eda_zhang.ipynb
├── src/
│   ├── build_unified_data.py  # 数据管线入口（download / build / qa / verify）
│   ├── data/              # 统一数据层：下载、解析、schema、加载
│   ├── crawler/           # 爬虫
│   ├── features/          # 特征工程
│   ├── models/            # 各类检测模型
│   └── viz/               # 可视化
├── results/               # 图表、指标输出
├── tests/                 # 单元测试（不依赖真实数据）
└── paper/                 # 论文（LaTeX / docx），含参考文献 .bib
```

`data/raw/` 已写入 `.gitignore`，请勿提交大体积原始数据；数据集较大时把网盘链接贴在下方「数据集」部分。

## 数据集

四份公开数据集 + 一份自建注入集，全部整理成同一套评论级格式（`user / item / rating / date / label / text`）。

| 数据集 | 规模 | 文本 | 标签来源 | 状态 |
|---|---|---|---|---|
| **YelpChi** | 67,395 条 / 38,063 用户 / 201 商户 | 无 | Yelp 过滤器 | 已构建 |
| **YelpNYC** | 官方 359,052 条 | 有 | Yelp 过滤器 | 未取得（需邮件申请） |
| **YelpZip** | 608,458 条 / 260,239 用户 / 5,044 商户（官方 608,598，实测 −140 条 / −0.023%） | 有 | Yelp 过滤器 | 已构建 |
| **FraudSquad-Amazon** | 94,242 条（真实基线 86,758 + LLM 生成 7,484）；三份生成集：Llama3 2,489 / Qwen2 2,495 / Qwen-DSR1 2,500，共用真实基线 | 有 | 真实=基线，生成=构造真值 | 已构建 |
| **YelpChi + 300 注入账号**（自建） | 70,186 条（注入 2,791） | 无 | 注入脚本 | 已构建 |

来源、核验过程与使用条款见 **[docs/data/provenance.md](docs/data/provenance.md)**；
字段定义见 [data_dictionary.md 第十一节](data_dictionary.md)；实测质量见
[results/data_quality_report.md](results/data_quality_report.md)。

各数据集的引用要求：

- YelpChi / YelpNYC / YelpZip：Rayana & Akoglu, *Collective Opinion Spam Detection*, KDD 2015
  （YelpChi 还需引 Mukherjee et al., ICWSM 2013）
- FraudSquad：Liu et al., *Detecting LLM-Generated Spam Reviews by Integrating Language Model
  Embeddings and Graph Neural Network*, 2026

**使用条款**：以上数据仅限学术研究，不得再分发或商用。原始文件已列入 `.gitignore`，
仓库只保留下载脚本与校验值。

> 关于 YelpNYC：官方发布页只提供邮件申请渠道（srayana@cs.stonybrook.edu），无直链。
> 拿到文件后放进 `data/raw/yelpnyc/` 重跑构建即可自动纳入，无需改代码。

## 分工

| 角色 | 代码任务 | 论文章节 |
|---|---|---|
| **A 数据工程** | 爬虫 + 清洗 + 数据字典 | 第 3 章 数据来源与预处理 |
| **B 特征工程** | 行为特征（发评时间、频率、评分偏离）、用户画像特征 | 第 4 章 特征体系设计 |
| **C 异常检测** | Isolation Forest / LOF / DBSCAN，无监督主线 | 第 5.1 节 无监督方法 |
| **D 文本 + 图** | 评论文本相似度（TF-IDF/SimHash）、用户-商品二部图、社区发现 | 第 5.2 节 文本与网络方法 |
| **E 评估 + 统筹** | 指标对比、可视化、仓库管理、论文合稿 | 第 1、2、6 章 + 摘要 |

分工姓名待组内确认（BLOCKED）；代码任务以左列为准。

## 协作规范

- 分支命名 `feat/crawler`、`feat/iforest` 等，禁止直接 push `main`
- Commit message 用 `feat:` / `fix:` / `docs:` 前缀
- 每周固定一次同步，用 GitHub Issues 建任务、指派到人
- 用 GitHub Projects 看板跟踪进度

## 运行说明

```bash
pip install -r requirements.txt
```

### 第一步：准备统一格式的数据（四份公开数据集）

```bash
python src/build_unified_data.py download   # 按清单下载（断点续传 + 多镜像回退）
python src/build_unified_data.py build      # 解析 -> 校验 -> 写 parquet + manifest
python src/build_unified_data.py qa         # 生成 results/data_quality_report.md
python src/build_unified_data.py verify     # 校验产物与 manifest 一致
# 一条命令跑完
python src/build_unified_data.py all
```

产物在 `data/processed/unified/`（已 gitignore）：

| 文件 | 内容 |
|---|---|
| `yelpchi.parquet` / `yelpzip.parquet` / `fraudsquad_amazon.parquet` / `yelpchi_injected.parquet` | 各数据集的统一表 |
| `fraudsquad_llama3.parquet` / `fraudsquad_qwen2.parquet` / `fraudsquad_qwen_dsr1.parquet` | FraudSquad 三份生成集各自成表（共用真实基线，便于同基线对比） |
| `all_reviews.parquet` | 上述合并表（84 万条），带 `dataset` 列区分来源 |
| `manifest.json` | 构建时间、原始文件 sha256、各集统计、官方规模对照 |

下游读数据统一走 `src/data/load.py`：

```python
import sys; sys.path.insert(0, "src")
from data import load_dataset, load_all, list_datasets

print(list_datasets())                    # 哪些数据集已构建、规模多少
df = load_dataset("yelpzip")              # 单集
df = load_dataset("yelpzip", with_text=False)   # 不要文本时省内存
both = load_all(["yelpchi", "fraudsquad_amazon"])
```

统一表字段：`dataset, review_id, global_user_id, user_id, item_id, rating, date, label,
label_source, is_injected, text_available, text`。其中 **`label=1` 恒表示虚假评论**，
`label=0` 为真实评论；`global_user_id` 形如 `yelpzip:12345`，跨数据集拼接不会撞号。

### 第二步：生成历史链路产物

由于数据文件体积较大，Git 仓库中不包含 `data/processed/` 下的文件。请按以下顺序在本地运行脚本生成：
1. `python src/step0_convert_metadata.py` -> 生成 `YelpChi_reviews.csv`
2. `python src/step1_build_account_base.py` -> 生成 `clean_accounts.parquet` 和 `scaler_params.json`
3. `python src/step2_sample_1000.py` -> 生成 `yelpchi_1000_accounts.csv`
4. `python src/step3_inject_water_army_pro.py` -> 生成 `augmented_accounts_v2.parquet` 和 `injection_log_v2.json`
   - 有 `data/raw/YelpChi_reviews.csv` 时按真实时间戳重算 burstiness 并落 `events_added`；
   - 缺 `data/raw/YelpChi_reviews.csv` 时脚本直接失败（无 CLI 开关），不会伪造事件
5. `python src/features/build_feature_table.py --dataset yelpchi` -> 生成
   `data/processed/features/feature_table_yelpchi.csv`、`scaler_yelpchi.json`、
   `results/feature_manifests/feature_table_yelpchi.json` 与 `results/feature_qa_yelpchi.md`
   （B 交付的建模交接表：burstiness 中位数填充 + 缺失标记列 + 每数据集独立 `_z` + 标签守卫，详见 `data_dictionary.md` 第五节；
   同时刷新 `data/processed/feature_table.csv` 兼容旧下游）
6. `python src/features/build_all.py` -> 批量构建多数据集特征表（YelpChi / YelpZip / FraudSquad×3 共 5 张已构建）；YelpNYC 写 `MISSING` 占位；缺数据时写 `blocked_by_data`，不造假表
7. `python src/viz/feature_eda.py` -> 生成 EDA 图件并输出模板级 dz 方向性判定
8. `python -m unittest discover -s tests -v` -> 回归测试（Issue #3 / Issue #6 / 标签守卫 / schema 类型）

## C：LOF / iForest 初步检测与诊断

接入 B 交付特征表后的运行方式、评价范围和 DCFO 验证状态见 [检测说明](docs/c_detection.md)。
当前 [诊断结果](results/c_detection_diagnostics.md) 记录了重复特征对 LOF 的影响；这些是初步实验，不是最终检测或规避攻击结果。

PR #4 审查后的 [分人群分析](results/c_cohort_analysis.md) 对照全部未注入账号、评论数至少 3 条及爆发性非缺失的人群。各人群分别拟合，不能将指标差异称为全体账号上的性能提升。此前 DCFO 玩具验证仅在本地完成，本仓库暂不可复现，详见检测说明。

```bash
python src/models/detection_diagnostics.py --input data/processed/feature_table.csv --neighbors 20 50 100 600 --report results/c_detection_diagnostics.md
python src/models/cohort_analysis.py --input data/processed/feature_table.csv
python src/models/cohort_uncertainty.py --input data/processed/feature_table.csv
python -m unittest discover -s tests -p "test_detection_diagnostics.py"
```

PR #5 的 [审查补充报告](results/c_cohort_uncertainty.md) 提供 Lift@50、固定分数下的 AUC bootstrap 区间、配对模型差值及标签/Top-50 并列画像。全部单元测试可用 `python -m unittest discover -s tests -v` 运行，不需要真实数据；完整实验仍需取得交接特征表。

## D：用户-商品二部图与社区发现（图算法）

本模块通过构建用户-商品二部图并投影为用户关系网，使用 Louvain 算法进行社区发现（团伙检测），输出账号图可疑分数与团伙网络可视化图。

### 运行说明

```bash
# 1. 运行二部图构建与社区发现，生成全量账号分数表
python src/models/graph_community_detection.py

# 2. 运行可视化脚本（同时输出 Top30 核心网络和随机对照组网络）
python src/viz/plot_group_network.py

# 3. 运行阈值敏感性分析（生成 results/sensitivity_analysis.md）
python src/models/sensitivity_analysis.py

# 4. 运行 AUC 验证脚本（输出与真实标签的对比结果）
python src/models/evaluate_auc.py
```

### 交付产物
- `data/processed/graph_scores.csv`：包含所有账号的图可疑分数，供后续融合评估。
  - `user_id`：账号唯一标识。
  - `group_id`：所属团伙编号。
  - `graph_score`：图可疑分（计算方式为：节点在团伙内的度数 / 团伙总人数，并进行全局 0~1 归一化）。实测核对发现，最高分（1.0）确实出现在人数较多且极度抱团的大团伙核心（如 392 人团伙中的核心节点），边缘节点分数极低，单人团伙为 0。
- `results/graph_group_network_top30.png`：高可疑候选团伙（核心与Top30邻居）网络结构图。
- `results/graph_group_network_random.png`：同规模随机抽取节点网络图（作为对照组）。
- `results/sensitivity_analysis.md`：图算法阈值敏感性分析结果表。

**关于阈值 500 的说明**：根据敏感性分析结果，阈值设为 500 时，最大连通子图人数（2841人）占有边账号总数（约2880人）的约99%，即约 99% 的有边账号连成了一整块。因此，阈值 500 下的“团伙”实际上是 Louvain 算法对这块巨大连通网进行切割的结果，非自然形成的独立团伙。详见 `results/sensitivity_analysis.md`。
