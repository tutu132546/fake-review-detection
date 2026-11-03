# 统一数据集质量报告

- 生成时间：2026-11-02T14:11:18+08:00
- schema 版本：1.0.0
- 生成命令：`python src/build_unified_data.py build qa`
- 判定口径：`label=1` 一律表示虚假评论（水军/LLM 生成/人工注入），`label=0` 为真实评论

## 1. 数据集总览

| 数据集 | 状态 | 评论数 | 用户数 | 商品数 | 假评论占比 | 文本覆盖 | 官方公布规模 | 行数偏差 |
|---|---|---|---|---|---|---|---|---|
| yelpchi | ok | 67,395 | 38,063 | 201 | 13.23% | 0.0% | 67395 | +0.000% |
| yelpnyc | missing | — | — | — | — | — | 359052 | — |
| yelpzip | ok | 608,458 | 260,239 | 5,044 | 13.22% | 100.0% | 608598 | -0.023% |
| fraudsquad_amazon | ok | 94,242 | 8,092 | 7,617 | 7.94% | 100.0% | — | — |
| yelpchi_injected | ok | 70,186 | 38,063 | 201 | 16.68% | 0.0% | — | — |

## 2. 缺失与偏差说明（如实记录）

- **yelpnyc 未构建**：官方页 https://shebuti.com/yelpnyc-dataset/ 要求邮件申请，未提供直链；本项目在无法取得时如实标记为缺失，不用其他数据顶替。
- **YelpNYC 建议的补齐方式**：向 srayana@cs.stonybrook.edu 邮件申请（注明课程用途）取得后，把文件放到 `data/raw/yelpnyc/` 再跑一次 `build`，管线会自动识别并纳入，无需改代码。

## 3. 标签来源构成（决定能否与其他论文对比）

| 数据集 | 标签来源 | 条数 |
|---|---|---|
| yelpchi | yelp_filter | 67,395 |
| yelpzip | yelp_filter | 608,458 |
| fraudsquad_amazon | real_baseline | 86,758 |
| fraudsquad_amazon | llm_generated | 7,484 |
| yelpchi_injected | yelp_filter | 67,395 |
| yelpchi_injected | injected | 2,791 |

## 4. 各数据集明细

### yelpchi

- 状态：ok
- 说明：来源文件 metadata.txt（不含文本，text_available=0）
- 出处：https://shebuti.com/yelpchi-dataset/
- 引用：Mukherjee et al., What Yelp Fake Review Filter Might Be Doing? ICWSM 2013; Rayana & Akoglu, Collective Opinion Spam Detection, KDD 2015
- 使用条款：仅限学术研究使用；Yelp 评论内容版权归 Yelp 及原作者，不得再分发或商用。官方发布页要求引用上述两篇论文。
- 原始文件：`data\raw\yelpchi\metadata.txt`（1,776,502 字节，sha256 `c674adf244460b48…`）
- 统计：67,395 条 / 38,063 用户 / 201 商品 / 时间 2004-10-12 ~ 2012-10-08 / 假评论 8,919 条

### yelpnyc

- 状态：missing
- 说明：官方页 https://shebuti.com/yelpnyc-dataset/ 要求邮件申请，未提供直链；本项目在无法取得时如实标记为缺失，不用其他数据顶替。
- 出处：https://shebuti.com/yelpnyc-dataset/
- 引用：Rayana & Akoglu, Collective Opinion Spam Detection: Bridging Review Networks and Metadata, KDD 2015
- 使用条款：仅限学术研究使用；不得再分发原始评论内容。

### yelpzip

- 状态：ok
- 说明：来源文件 yelpzip.csv（带评论文本）
- 出处：https://shebuti.com/yelpzip-dataset/
- 引用：Rayana & Akoglu, Collective Opinion Spam Detection: Bridging Review Networks and Metadata, KDD 2015
- 使用条款：原始数据仅限学术研究使用。本项目通过公开镜像取得，镜像页未声明额外许可；引用时仍应引用 Rayana & Akoglu (KDD 2015)。
- 原始文件：`data\raw\yelpzip\yelpzip.csv`（410,865,364 字节，sha256 `78f92e83574208f8…`）
- 统计：608,458 条 / 260,239 用户 / 5,044 商品 / 时间 2004-10-20 ~ 2015-01-10 / 假评论 80,439 条

### fraudsquad_amazon

- 状态：ok
- 说明：真实基线 amazon_2022_5core.csv；生成集 ['llama3_8b', 'qwen_72b', 'qwen_dsr1']
- 出处：https://github.com/RafaelLiuZH/FraudSquad
- 引用：Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network (FraudSquad), 2026
- 使用条款：仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，受 Amazon 数据条款约束；生成部分由作者用 Llama3-8B / Qwen 系列产出。本项目仅作学术研究引用，不再分发原始文件。
- 原始文件：`data\raw\fraudsquad\amazon_2022_5core.csv.zip`（18,773,569 字节，sha256 `858b1f2f51d5fd78…`）
- 原始文件：`data\raw\fraudsquad\amazon\amazon_2022_5core.csv`（67,012,970 字节，sha256 `ea618dbb180e6090…`）
- 原始文件：`data\raw\fraudsquad\reviews_from_llama3_8b.csv`（1,433,276 字节，sha256 `650d5fa18c527b2a…`）
- 原始文件：`data\raw\fraudsquad\reviews_from_qwen_72b.csv`（1,468,340 字节，sha256 `71cd6f81dfdb6176…`）
- 原始文件：`data\raw\fraudsquad\reviews_from_qwen_dsr1.csv`（1,520,028 字节，sha256 `e706b4341bd9d149…`）
- 统计：94,242 条 / 8,092 用户 / 7,617 商品 / 时间 2021-12-31 ~ 2023-01-03 / 假评论 7,484 条
- 同一用户—商品—日期出现多条评论：2,097 组。这是原始数据的真实形态（Yelp 允许修改评论、Amazon 允许同一商品多次购买评价），**未做去重**；做用户-商品图时需自行决定是否合并。

### yelpchi_injected

- 状态：ok
- 说明：由 injection_log_v2.json 的 300 个账号、2791 条注入事件展开；注入评论无文本（step3 未生成文本），text_available=0
- 出处：本项目 step3 注入实验（见 data_dictionary.md 第三节）
- 引用：本项目自建；注入模板与物理校验见 data_dictionary.md
- 使用条款：派生数据，继承 YelpChi 的使用限制。
- 原始文件：`data\processed\injection_log_v2.json`（877,795 字节，sha256 `28a923a526571568…`）
- 原始文件：`data\raw\YelpChi_reviews.csv`（1,843,935 字节，sha256 `4ee1f48e571fae82…`）
- 统计：70,186 条 / 38,063 用户 / 201 商品 / 时间 2004-10-12 ~ 2012-10-08 / 假评论 11,710 条

## 5. 使用陷阱（用之前必须知道）

### yelpchi

- **该版本无评论文本**：metadata 版只含 user/product/rating/label/date，`text_available=0`。文本方法（TF-IDF 等）在 YelpChi 上不可运行，不能用「空文本」当输入算出结果再与其它数据集比较。

### yelpzip

- **标签为平台过滤器结果**（near ground truth）：Yelp 的过滤算法本身有误判，这是所有使用该数据集的论文的共同前提，报告中需说明。

### fraudsquad_amazon

- **生成评论复用了真实账号**：2,191 个发布过生成评论的用户中，2,191 个（100.0%）在真实基线里也有真实评论。因此**不能**用「该用户是否发过生成评论」直接当账号级标签——那会把正常用户的账号误判为水军。账号级分析请改用评论级标签 + 人工核对，或把混合账号单独标记。
- **评分为捷径特征**：7,484 条生成评论的评分全部为 [5.0]，而真实评论 5 星占比仅 69.6%。任何用评分做输入的模型都会因此虚高，报告结果时必须把「仅评分」作为一个对照基线报出来。

### yelpchi_injected

- **注入评论没有文本**：step3 只生成结构化事件（日期/评分/商品），文本方法需把注入行排除或单独处理，不能拿空文本参与训练。

## 6. 标签口径与公平对比提示

- Yelp 系列标签来自 Yelp 平台过滤器（near ground truth），不是人工逐条标注，存在平台自身误差。
- FraudSquad 的虚假评论由大模型生成，真实评论取自 Amazon 5-core，属「构造即真值」。
- 人工注入集（yelpchi_injected）的标签由注入脚本保证，与平台标签不可混为一谈。
- 对比实验必须写明每种方法是否需要标签：LOF/iForest/DBSCAN 无监督；TF-IDF+LR 与 GNN 类方法有监督。
- 需要说明的是：YelpChi 与 YelpZip 的假评论比例（约 13%）相近，而 FraudSquad 为 7.94%，跨数据集的绝对数值不可直接横向比较，只能说「在同一数据集上谁更好」。
