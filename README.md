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
