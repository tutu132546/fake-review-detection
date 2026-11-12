# 统一数据集质量报告

- 生成时间：2026-11-12T21:06:03+08:00
- schema 版本：1.0.0
- 生成命令：`python src/build_unified_data.py build qa`
- 判定口径：`label=1` 一律表示虚假评论（水军/LLM 生成/人工注入），`label=0` 为真实评论

## 1. 数据集总览

| 数据集 | 状态 | 评论数 | 用户数 | 商品数 | 假评论占比 | 文本覆盖 | 官方公布规模 | 行数偏差 |
|---|---|---|---|---|---|---|---|---|
| yelpchi | ok | 67,395 | 38,063 | 201 | 13.23% | 0.0% | 67395 | +0.000% |
| yelpnyc | ok | 359,052 | 160,225 | 923 | 10.27% | 100.0% | 359052 | +0.000% |
| yelpzip | ok | 608,458 | 260,239 | 5,044 | 13.22% | 100.0% | 608598 | -0.023% |
| fraudsquad_llama3 | ok | 89,247 | 8,092 | 7,617 | 2.79% | 100.0% | 89247 | +0.000% |
| fraudsquad_qwen2 | ok | 89,253 | 8,092 | 7,617 | 2.80% | 100.0% | 89253 | +0.000% |
| fraudsquad_qwen_dsr1 | ok | 89,258 | 8,092 | 7,617 | 2.80% | 100.0% | 89258 | +0.000% |
| yelpchi_injected | ok | 70,186 | 38,063 | 201 | 16.68% | 0.0% | 70186 | +0.000% |

## 2. 数据口径核对（与官方公布值逐项对照）

- 已构建的数据集行数与官方公布规模偏差均在 2% 以内。

### 2.1 YelpZip 与官方 608,598 条的差额对账

| 项目 | 数量 | 说明 |
|---|---|---|
| 镜像文件数据行 | 608,598 | 与官方公布条数一致 |
| 其中带标签、可用 | 608,458 | 进入统一表 |
| 其中导出残缺、不可用 | 140 | 缺 rating 与 label，无法作为样本 |
| 账号并集 | 260,239 + 38 = 260,277 | 官方公布 260,277，完全吻合 |

**结论**：镜像并非「少了 140 行」，而是这 140 行在导出时丢了 rating 与 label 两个字段，并且被前一条未闭合引号的记录吞进正文。读取阶段已把这些残缺行从正常记录的正文里剥离，残缺行本身因无标签无法使用。行数、账号数两个口径同时对上官方公布值，可确认缺口来源。补齐需要官方原版文件（邮件申请渠道见 provenance.md）。

**是文件本来就没有，还是读取时被丢掉的？——是文件本来就这样，不是解析丢行。** 三条证据：

1. 原始字节层面：文件共 608,599 个物理换行，其中 608,459 个 `CRLF` + **140 个裸 `LF`**；按 CSV 记录解析得 608,459 条，再去掉 1 条表头 = 608,458 条，与解析结果逐条对得上。若是解析阶段丢行，丢的行不会在原始字节里留下痕迹。
2. 结构层面：这 140 个裸 `LF` 后面紧跟的正是 `user\tproduct\tdate\ttext` 四列制表符分隔的残缺记录，缺的恰好是 rating 与 label；它们没有独立的行首，所以被上一条未闭合引号的记录整体吞进 `text`。
3. 数量层面：把这些残缺行按「独立记录」计回，行数 608,598、账号数 260,277 两项同时等于官方公布值，若中途丢行不可能两项同时对上。

**论文里可直接写的句子**：本项目使用的 YelpZip 取自公开镜像（ModelScope），与作者官方版本相差 0.023%（608,458 vs 608,598 条），差额为镜像导出时 140 条记录缺失 rating/label 字段所致，已在读取阶段识别并剔除；账号数 260,277 与官方一致。


### 2.2 交付清单

| 数据集 | 论文口径 | 状态 | 条数 | 有文本 | 备注 |
|---|---|---|---|---|---|
| yelpchi | 是 | ✅ | 67,395 | ❌ | 无评论文本，文本类方法不可用 |
| yelpnyc | 是 | ✅ | 359,052 | ✅ | — |
| yelpzip | 是 | ✅ | 608,458 | ✅ | — |
| fraudsquad_llama3 | 是 | ✅ | 89,247 | ✅ | llama3_8b |
| fraudsquad_qwen2 | 是 | ✅ | 89,253 | ✅ | qwen_72b |
| fraudsquad_qwen_dsr1 | 是 | ✅ | 89,258 | ✅ | qwen_dsr1 |
| fraudsquad_amazon | 否（仅总览视图） | ✅ | 94,242 | ✅ | 不进 all_reviews |
| yelpchi_injected | 否（自建） | ✅ | 70,186 | ❌ | 注入实验用 |

## 3. 标签来源构成（决定能否与其他论文对比）

| 数据集 | 标签来源 | 条数 |
|---|---|---|
| yelpchi | yelp_filter | 67,395 |
| yelpnyc | yelp_filter | 359,052 |
| yelpzip | yelp_filter | 608,458 |
| fraudsquad_llama3 | real_baseline | 86,758 |
| fraudsquad_llama3 | llm_generated | 2,489 |
| fraudsquad_qwen2 | real_baseline | 86,758 |
| fraudsquad_qwen2 | llm_generated | 2,495 |
| fraudsquad_qwen_dsr1 | real_baseline | 86,758 |
| fraudsquad_qwen_dsr1 | llm_generated | 2,500 |
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
- 原始文件：`data/raw/yelpchi/metadata.txt`（1,776,502 字节，sha256 `c674adf244460b48…`）
- 统计：67,395 条 / 38,063 用户 / 201 商品 / 时间 2004-10-12 ~ 2012-10-08 / 假评论 8,919 条

### yelpnyc

- 状态：ok
- 说明：来源为 Rayana 官方发布包的双文件形态：metadata（元数据）+ reviewContent（评论文本），带完整文本；文本按 (user, item, date) 对齐，策略 positional，359,052/359,052 条匹配、0 行键错位
- 出处：https://shebuti.com/yelpnyc-dataset/
- 引用：Rayana & Akoglu, Collective Opinion Spam Detection: Bridging Review Networks and Metadata, KDD 2015
- 使用条款：仅限学术研究使用；不得再分发原始评论内容。
- 原始文件：`data/raw/yelpnyc/metadata`（9,687,857 字节，sha256 `772df667e98341cb…`）
- 原始文件：`data/raw/yelpnyc/reviewContent`（236,566,411 字节，sha256 `dca37a1aa2d8bc11…`）
- 统计：359,052 条 / 160,225 用户 / 923 商品 / 时间 2004-10-20 ~ 2015-01-07 / 假评论 36,885 条

### yelpzip

- 状态：ok
- 说明：来源文件 yelpzip.csv（带评论文本）；解析出 608,458 条带标签记录，另有 140 行因镜像导出缺陷缺 rating/label 而不可用（涉及 139 个账号、18 个商品，其中 38 个账号只出现在残缺行）；18 条正常记录的文本曾被这些残缺行污染，已在读取阶段剥离
- 出处：https://shebuti.com/yelpzip-dataset/
- 引用：Rayana & Akoglu, Collective Opinion Spam Detection: Bridging Review Networks and Metadata, KDD 2015
- 使用条款：原始数据仅限学术研究使用。本项目通过公开镜像取得，镜像页未声明额外许可；引用时仍应引用 Rayana & Akoglu (KDD 2015)。
- 原始文件：`data/raw/yelpzip/yelpzip.csv`（410,865,364 字节，sha256 `78f92e83574208f8…`）
- 统计：608,458 条 / 260,239 用户 / 5,044 商品 / 时间 2004-10-20 ~ 2015-01-10 / 假评论 80,439 条

### fraudsquad_llama3

- 状态：ok
- 说明：真实基线 amazon_2022_5core.csv + 生成集 ['llama3_8b']（单模型数据集，论文口径）
- 出处：https://github.com/RafaelLiuZH/FraudSquad
- 引用：Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network (FraudSquad), 2025
- 使用条款：仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，受 Amazon 数据条款约束；生成部分由作者用 Llama3-8B 产出。本项目仅作学术研究引用，不再分发原始文件。
- 原始文件：`data/raw/fraudsquad/amazon_2022_5core.csv.zip`（18,773,569 字节，sha256 `858b1f2f51d5fd78…`）
- 原始文件：`data/raw/fraudsquad/amazon/amazon_2022_5core.csv`（67,012,970 字节，sha256 `ea618dbb180e6090…`）
- 原始文件：`data/raw/fraudsquad/reviews_from_llama3_8b.csv`（1,433,276 字节，sha256 `650d5fa18c527b2a…`）
- 统计：89,247 条 / 8,092 用户 / 7,617 商品 / 时间 2021-12-31 ~ 2023-01-03 / 假评论 2,489 条
- 同一用户—商品—日期出现多条评论：2,096 组。这是原始数据的真实形态（Yelp 允许修改评论、Amazon 允许同一商品多次购买评价），**未做去重**；做用户-商品图时需自行决定是否合并。

### fraudsquad_qwen2

- 状态：ok
- 说明：真实基线 amazon_2022_5core.csv + 生成集 ['qwen_72b']（单模型数据集，论文口径）
- 出处：https://github.com/RafaelLiuZH/FraudSquad
- 引用：Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network (FraudSquad), 2025
- 使用条款：仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，受 Amazon 数据条款约束；生成部分由作者用 Qwen-72B 产出。本项目仅作学术研究引用，不再分发原始文件。
- 原始文件：`data/raw/fraudsquad/amazon_2022_5core.csv.zip`（18,773,569 字节，sha256 `858b1f2f51d5fd78…`）
- 原始文件：`data/raw/fraudsquad/amazon/amazon_2022_5core.csv`（67,012,970 字节，sha256 `ea618dbb180e6090…`）
- 原始文件：`data/raw/fraudsquad/reviews_from_qwen_72b.csv`（1,468,340 字节，sha256 `71cd6f81dfdb6176…`）
- 统计：89,253 条 / 8,092 用户 / 7,617 商品 / 时间 2021-12-31 ~ 2023-01-02 / 假评论 2,495 条
- 同一用户—商品—日期出现多条评论：2,096 组。这是原始数据的真实形态（Yelp 允许修改评论、Amazon 允许同一商品多次购买评价），**未做去重**；做用户-商品图时需自行决定是否合并。

### fraudsquad_qwen_dsr1

- 状态：ok
- 说明：真实基线 amazon_2022_5core.csv + 生成集 ['qwen_dsr1']（单模型数据集，论文口径）
- 出处：https://github.com/RafaelLiuZH/FraudSquad
- 引用：Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network (FraudSquad), 2025
- 使用条款：仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，受 Amazon 数据条款约束；生成部分由作者用 Qwen-DeepSeek-R1 产出。本项目仅作学术研究引用，不再分发原始文件。
- 原始文件：`data/raw/fraudsquad/amazon_2022_5core.csv.zip`（18,773,569 字节，sha256 `858b1f2f51d5fd78…`）
- 原始文件：`data/raw/fraudsquad/amazon/amazon_2022_5core.csv`（67,012,970 字节，sha256 `ea618dbb180e6090…`）
- 原始文件：`data/raw/fraudsquad/reviews_from_qwen_dsr1.csv`（1,520,028 字节，sha256 `e706b4341bd9d149…`）
- 统计：89,258 条 / 8,092 用户 / 7,617 商品 / 时间 2021-12-31 ~ 2023-01-02 / 假评论 2,500 条
- 同一用户—商品—日期出现多条评论：2,096 组。这是原始数据的真实形态（Yelp 允许修改评论、Amazon 允许同一商品多次购买评价），**未做去重**；做用户-商品图时需自行决定是否合并。

### fraudsquad_amazon

- 状态：ok
- 说明：真实基线 amazon_2022_5core.csv + 生成集 ['llama3_8b', 'qwen_72b', 'qwen_dsr1']（三模型合并视图，仅供总览，不作对比基准）
- 出处：https://github.com/RafaelLiuZH/FraudSquad
- 引用：Liu et al., Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network (FraudSquad), 2025
- 使用条款：仓库未声明 LICENSE 文件；真实评论部分来自 Amazon Reviews 2023 公开数据集，受 Amazon 数据条款约束；生成部分由作者用 Llama3-8B / Qwen 系列产出。本项目仅作学术研究引用，不再分发原始文件。
- 原始文件：`data/raw/fraudsquad/amazon_2022_5core.csv.zip`（18,773,569 字节，sha256 `858b1f2f51d5fd78…`）
- 原始文件：`data/raw/fraudsquad/amazon/amazon_2022_5core.csv`（67,012,970 字节，sha256 `ea618dbb180e6090…`）
- 原始文件：`data/raw/fraudsquad/reviews_from_llama3_8b.csv`（1,433,276 字节，sha256 `650d5fa18c527b2a…`）
- 原始文件：`data/raw/fraudsquad/reviews_from_qwen_72b.csv`（1,468,340 字节，sha256 `71cd6f81dfdb6176…`）
- 原始文件：`data/raw/fraudsquad/reviews_from_qwen_dsr1.csv`（1,520,028 字节，sha256 `e706b4341bd9d149…`）
- 统计：94,242 条 / 8,092 用户 / 7,617 商品 / 时间 2021-12-31 ~ 2023-01-03 / 假评论 7,484 条
- 同一用户—商品—日期出现多条评论：2,097 组。这是原始数据的真实形态（Yelp 允许修改评论、Amazon 允许同一商品多次购买评价），**未做去重**；做用户-商品图时需自行决定是否合并。

### yelpchi_injected

- 状态：ok
- 说明：由 injection_log_v2.json 的 300 个账号、2791 条注入事件展开；注入评论无文本（step3 未生成文本），text_available=0
- 出处：本项目 step3 注入实验（见 data_dictionary.md 第三节）
- 引用：本项目自建；注入模板与物理校验见 data_dictionary.md
- 使用条款：派生数据，继承 YelpChi 的使用限制。
- 原始文件：`data/processed/injection_log_v2.json`（877,795 字节，sha256 `28a923a526571568…`）
- 原始文件：`data/raw/YelpChi_reviews.csv`（1,843,935 字节，sha256 `4ee1f48e571fae82…`）
- 统计：70,186 条 / 38,063 用户 / 201 商品 / 时间 2004-10-12 ~ 2012-10-08 / 假评论 11,710 条

## 5. 使用陷阱（用之前必须知道）

### yelpchi

- **该版本无评论文本**：metadata 版只含 user/product/rating/label/date，`text_available=0`。文本方法（TF-IDF 等）在 YelpChi 上不可运行，不能用「空文本」当输入算出结果再与其它数据集比较。

### yelpnyc

- **标签为平台过滤器结果**（near ground truth）：与 YelpZip 同源，同样存在过滤算法误判，报告中需说明这是「近真值」而非人工标注。
- **元数据与文本是两个独立文件**：`metadata`（5 列，含 rating/label）与 `reviewContent`（4 列，含正文）各 359,052 行，靠 (user, item, date) 对齐；实测策略 positional、匹配 359,052/359,052、键错位 0 行。统一表已合并好，下游不要自己去拼这两个文件（贴错一行是静默错误，指标上完全看不出来）。
- **假评比例与另两套不同，不要混用**：官方公布 YelpNYC 为 10.27%（YelpChi 13.23% / YelpZip 13.22%），这是官方页各自公布的数值，不是抄写错误；本项目实测 10.2729%，spammer 账号占比 17.7850%（官方 17.79%）。**不要用 13.22% 描述 YelpNYC**。

### yelpzip

- **标签为平台过滤器结果**（near ground truth）：Yelp 的过滤算法本身有误判，这是所有使用该数据集的论文的共同前提，报告中需说明。
- **镜像文件有 140 行导出残缺**：这些行只有 `user / product / date / text` 四列，缺 rating 与 label，会被前一条未闭合引号的记录吞进正文。读取阶段已剥离（18 条正常记录的正文恢复干净），但残缺行本身无标签、无法使用：涉及 139 个账号，其中 38 个只出现在残缺行，在 YelpZip 上做账号级实验时这部分账号缺位（占账号总数 0.015%）。补齐需要官方原版文件，见 docs/data/provenance.md §3。

### fraudsquad_llama3

- **被接管账号同时保留真实评论**：1,250 个写过生成评论的账号中，1,250 个（100.0%）在真实基线里也有真实评论。这是论文的设定（接管既有账号），**账号级实验可以做**：正例 = 被接管账号，负例 = 其余账号，见 `python src/build_unified_data.py accounts`。唯一不能做的是**评论级**拿「账号是否被接管」当标签——那会把这个账号的历史真实评论也标成假的。
- **评分为捷径特征**：2,489 条生成评论的评分全部为 [5.0]，而真实评论 5 星占比仅 69.6%。任何用评分做输入的模型都会因此虚高，报告结果时必须把「仅评分」作为一个对照基线报出来。
- **单模型口径**：本表 = 真实基线 86,758 条 + 单一模型生成集，1,250 个被接管账号，每人约 2 条假评论，与论文设定一致。真实基线在三份单模型表里是同一份，所以三份表**不是三个互相独立的数据集**。
- **账号集合关系（按数据实算）**：`fraudsquad_llama3` 与 `fraudsquad_qwen2` 的被接管账号**完全相同**（各 1,250 个，集合相等）；`fraudsquad_qwen_dsr1` 是**另一批**账号（1,250 个，与前者交集仅 309），三者并集 2,191 个。因此：fraudsquad_llama3 vs fraudsquad_qwen2 是**同一批账号的配对比较**（同账号换模型，可以算配对差、不能当两组独立样本做显著性检验）；而 fraudsquad_qwen_dsr1 与它们**只有 309 个账号重叠**，是「换账号 + 换模型」的混合差异，不能解读为纯模型效应。

### fraudsquad_qwen2

- **被接管账号同时保留真实评论**：1,250 个写过生成评论的账号中，1,250 个（100.0%）在真实基线里也有真实评论。这是论文的设定（接管既有账号），**账号级实验可以做**：正例 = 被接管账号，负例 = 其余账号，见 `python src/build_unified_data.py accounts`。唯一不能做的是**评论级**拿「账号是否被接管」当标签——那会把这个账号的历史真实评论也标成假的。
- **评分为捷径特征**：2,495 条生成评论的评分全部为 [5.0]，而真实评论 5 星占比仅 69.6%。任何用评分做输入的模型都会因此虚高，报告结果时必须把「仅评分」作为一个对照基线报出来。
- **单模型口径**：本表 = 真实基线 86,758 条 + 单一模型生成集，1,250 个被接管账号，每人约 2 条假评论，与论文设定一致。真实基线在三份单模型表里是同一份，所以三份表**不是三个互相独立的数据集**。
- **账号集合关系（按数据实算）**：`fraudsquad_llama3` 与 `fraudsquad_qwen2` 的被接管账号**完全相同**（各 1,250 个，集合相等）；`fraudsquad_qwen_dsr1` 是**另一批**账号（1,250 个，与前者交集仅 309），三者并集 2,191 个。因此：fraudsquad_llama3 vs fraudsquad_qwen2 是**同一批账号的配对比较**（同账号换模型，可以算配对差、不能当两组独立样本做显著性检验）；而 fraudsquad_qwen_dsr1 与它们**只有 309 个账号重叠**，是「换账号 + 换模型」的混合差异，不能解读为纯模型效应。

### fraudsquad_qwen_dsr1

- **被接管账号同时保留真实评论**：1,250 个写过生成评论的账号中，1,250 个（100.0%）在真实基线里也有真实评论。这是论文的设定（接管既有账号），**账号级实验可以做**：正例 = 被接管账号，负例 = 其余账号，见 `python src/build_unified_data.py accounts`。唯一不能做的是**评论级**拿「账号是否被接管」当标签——那会把这个账号的历史真实评论也标成假的。
- **评分为捷径特征**：2,500 条生成评论的评分全部为 [5.0]，而真实评论 5 星占比仅 69.6%。任何用评分做输入的模型都会因此虚高，报告结果时必须把「仅评分」作为一个对照基线报出来。
- **单模型口径**：本表 = 真实基线 86,758 条 + 单一模型生成集，1,250 个被接管账号，每人约 2 条假评论，与论文设定一致。真实基线在三份单模型表里是同一份，所以三份表**不是三个互相独立的数据集**。
- **账号集合关系（按数据实算）**：`fraudsquad_llama3` 与 `fraudsquad_qwen2` 的被接管账号**完全相同**（各 1,250 个，集合相等）；`fraudsquad_qwen_dsr1` 是**另一批**账号（1,250 个，与前者交集仅 309），三者并集 2,191 个。因此：fraudsquad_llama3 vs fraudsquad_qwen2 是**同一批账号的配对比较**（同账号换模型，可以算配对差、不能当两组独立样本做显著性检验）；而 fraudsquad_qwen_dsr1 与它们**只有 309 个账号重叠**，是「换账号 + 换模型」的混合差异，不能解读为纯模型效应。

### fraudsquad_amazon

- **被接管账号同时保留真实评论**：2,191 个写过生成评论的账号中，2,191 个（100.0%）在真实基线里也有真实评论。这是论文的设定（接管既有账号），**账号级实验可以做**：正例 = 被接管账号，负例 = 其余账号，见 `python src/build_unified_data.py accounts`。唯一不能做的是**评论级**拿「账号是否被接管」当标签——那会把这个账号的历史真实评论也标成假的。
- **评分为捷径特征**：7,484 条生成评论的评分全部为 [5.0]，而真实评论 5 星占比仅 69.6%。任何用评分做输入的模型都会因此虚高，报告结果时必须把「仅评分」作为一个对照基线报出来。
- **这是三模型合并视图**：2,191 个被接管账号共 7,484 条假评论，每人 2~6 条（均值 3.4），与原文「每个账号 2 条」的口径不一致，**不要拿它和论文数字比**。对比实验请用 fraudsquad_llama3 / fraudsquad_qwen2 / fraudsquad_qwen_dsr1。

### yelpchi_injected

- **注入评论没有文本**：step3 只生成结构化事件（日期/评分/商品），文本方法需把注入行排除或单独处理，不能拿空文本参与训练。

## 6. 账号级视图（`build_unified_data.py accounts` 产出）

| 数据集 | 账号数 | 正例账号 | 正例占比 | 账号标签口径 |
|---|---|---|---|---|
| yelpchi | 38,063 | 7,739 | 20.33% | `platform_filtered` |
| yelpnyc | 160,225 | 28,496 | 17.79% | `platform_filtered` |
| yelpzip | 260,239 | 62,214 | 23.91% | `platform_filtered` |
| fraudsquad_llama3 | 8,092 | 1,250 | 15.45% | `hijacked_by_llm` |
| fraudsquad_qwen2 | 8,092 | 1,250 | 15.45% | `hijacked_by_llm` |
| fraudsquad_qwen_dsr1 | 8,092 | 1,250 | 15.45% | `hijacked_by_llm` |
| fraudsquad_amazon | 8,092 | 2,191 | 27.08% | `hijacked_by_llm` |
| yelpchi_injected | 38,063 | 300 | 0.79% | `injected` |

账号级正例的定义见 `src/data/schema.py: ACCOUNT_LABEL_BASIS`。**账号级与评论级是两个不同任务**：账号级问「这个号是不是水军」，评论级问「这条评论是不是假的」。FraudSquad 的被接管账号同时有真实评论，因此账号级正例不代表其全部评论为假；反过来，评论级实验也不能拿账号标签当答案。

## 7. 标签口径与公平对比提示

- Yelp 系列标签来自 Yelp 平台过滤器（near ground truth），不是人工逐条标注，存在平台自身误差。
- FraudSquad 的虚假评论由大模型生成，真实评论取自 Amazon 5-core，属「构造即真值」。
- 三个 FraudSquad 单模型表是**论文口径的独立数据集**；合并视图 `fraudsquad_amazon` 只作总览。
- 人工注入集（yelpchi_injected）的标签由注入脚本保证，与平台标签不可混为一谈。
- 对比实验必须写明每种方法是否需要标签：LOF/iForest/DBSCAN 无监督；TF-IDF+LR 与 GNN 类方法有监督。
- 需要说明的是：YelpChi 与 YelpZip 的假评论比例（约 13%）相近，而 FraudSquad 三个单模型表分别约 2.8%（真实基线占绝大多数），跨数据集的绝对数值不可直接横向比较，只能说「在同一数据集、同一划分下谁更好」。
