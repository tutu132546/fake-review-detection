# fake-review-detection

虚假评论（水军）检测课程项目：结合无监督异常检测、文本相似度与用户-商品网络分析，识别电商/点评平台上的虚假评论及刷单账号。

## 目录结构

```
fake-review-detection/
├── README.md              # 项目介绍、分工、运行说明
├── requirements.txt
├── data/
│   ├── raw/               # 原始数据（已加入 .gitignore，不提交到仓库）
│   └── processed/         # 清洗/处理后的数据
├── notebooks/             # 探索性分析，每人一个，命名如 01_eda_zhang.ipynb
├── src/
│   ├── crawler/           # 爬虫
│   ├── features/          # 特征工程
│   ├── models/            # 各类检测模型
│   └── viz/                # 可视化
├── results/                # 图表、指标输出
└── paper/                  # 论文（LaTeX / docx），含参考文献 .bib
```

`data/raw/` 已写入 `.gitignore`，请勿提交大体积原始数据；数据集较大时把网盘链接贴在下方「数据集」部分。

## 数据集

主线使用带标注的公开数据集做量化验证，附加实验再用自建爬虫数据做案例分析：

- **YelpChi**：Yelp 过滤器标注过的虚假评论数据集，学界标准 benchmark，带 label
- **Amazon Review Fraud**（DGL 库自带）：带标注
- 自建数据：豆瓣 / Steam（Steam 官方 API 相对友好，豆瓣反爬较凶，可作为 B 计划）

原始数据网盘链接：`TODO`

## 分工

| 角色 | 代码任务 | 论文章节 |
|---|---|---|
| **A 数据工程** | 爬虫 + 清洗 + 数据字典 | 第 3 章 数据来源与预处理 |
| **B 特征工程** | 行为特征（发评时间、频率、评分偏离）、用户画像特征 | 第 4 章 特征体系设计 |
| **C 异常检测** | Isolation Forest / LOF / DBSCAN，无监督主线 | 第 5.1 节 无监督方法 |
| **D 文本 + 图** | 评论文本相似度（TF-IDF/SimHash）、用户-商品二部图、社区发现 | 第 5.2 节 文本与网络方法 |
| **E 评估 + 统筹** | 指标对比、可视化、仓库管理、论文合稿 | 第 1、2、6 章 + 摘要 |

具体姓名待补充：`TODO`

## 协作规范

- 分支命名 `feat/crawler`、`feat/iforest` 等，禁止直接 push `main`
- Commit message 用 `feat:` / `fix:` / `docs:` 前缀
- 每周固定一次同步，用 GitHub Issues 建任务、指派到人
- 用 GitHub Projects 看板跟踪进度

## 运行说明

```bash
pip install -r requirements.txt
```

`TODO`：补充数据准备、特征生成、模型训练/评估的具体运行步骤。

# 数据产物说明
由于数据文件体积较大，Git 仓库中不包含 `data/processed/` 下的文件。请按以下顺序在本地运行脚本生成：
1. `python src/step0_convert_metadata.py` -> 生成 `YelpChi_reviews.csv`
2. `python src/step1_build_account_base.py` -> 生成 `clean_accounts.parquet` 和 `scaler_params.json`
3. `python src/step2_sample_1000.py` -> 生成 `yelpchi_1000_accounts.csv`
4. `python src/step3_inject_water_army_pro.py` -> 生成 `augmented_accounts_v2.parquet` 和 `injection_log_v2.json`
5. `python src/features/build_feature_table.py` -> 生成 `feature_table.csv`（B 交付的建模交接表：burstiness 中位数填充 + 缺失标记列 + `_z` 统一重算，详见 `data_dictionary.md` 第五节）

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

# 2. 运行可视化脚本，自动提取人数最多的团伙并绘图
python src/viz/plot_group_network.py
```

### 交付产物
- `results/graph_scores.csv`：包含所有账号的图可疑分数，供后续融合评估。
  - `user_id`：账号唯一标识。
  - `group_id`：所属团伙编号。
  - `graph_score`：图可疑分（计算方式为：团伙人数 / 20，封顶为 1.0。即团伙人数达到 20 人即得满分）。
- `results/graph_group_network.png`：高可疑候选团伙的网络结构可视化图。