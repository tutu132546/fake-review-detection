# 数据集来源、核验与使用条款

> 本文档回答三个问题：**数据从哪来**、**我们怎么确认拿到的是对的**、**能怎么用、不能怎么用**。
> 生成统一格式的命令与字段说明见 [`data_dictionary.md` 第十一节](../../data_dictionary.md)，
> 数据质量实测结果见 [`results/data_quality_report.md`](../../results/data_quality_report.md)。

## 0. 一页结论

| 数据集 | 论文口径 | 状态 | 评论数 | 带文本 | 标签来源 | 主要用途 |
|---|---|---|---|---|---|---|
| YelpChi | 是 | ✅ 已构建 | 67,395 | ❌ | Yelp 过滤器 | 图方法、无监督检测主基准 |
| YelpNYC | 是 | ✅ 已构建 | 359,052 | ✅ | Yelp 过滤器 | 单城小规模 + 文本基线 |
| YelpZip | 是 | ✅ 已构建 | 608,458（另有 140 行残缺不可用） | ✅ | Yelp 过滤器 | 大规模 + 文本基线 |
| FraudSquad-Llama3 | 是 | ✅ 已构建 | 89,247（生成 2,489） | ✅ | 构造即真值 | 大模型水军实验 |
| FraudSquad-Qwen2 | 是 | ✅ 已构建 | 89,253（生成 2,495） | ✅ | 构造即真值 | 同上，换模型 |
| FraudSquad-Qwen-DSR1 | 是 | ✅ 已构建 | 89,258（生成 2,500） | ✅ | 构造即真值 | 同上，换模型 |
| FraudSquad-Amazon（合并视图） | **否** | ✅ 已构建 | 94,242（生成 7,484） | ✅ | 构造即真值 | 仅总览，不作对比基准 |
| YelpChi + 300 注入账号 | 否（自建） | ✅ 已构建 | 70,186（注入 2,791） | ❌ | 注入脚本保证 | 解释方法（DCFO）与规避实验 |

**论文口径的 6 张表全部到位**，无缺失数据集。唯一剩余的数据缺口是 **YelpChi 缺评论文本**
（持有的是 `metadata.txt` 5 列版本，没有 `reviewContent`），影响面是"文本类方法不能在 YelpChi 上跑"，
其余图方法/元数据方法不受影响。

> **FraudSquad 为什么拆成三张表**：论文里三个模型是三个独立数据集（真实基线 + 单一模型生成的
> 假评论，每个被接管账号约 2 条假评论）。合并成一份后，同一个账号会攒到 4~6 条假评论
> （实测 930 个账号 4 条、304 个账号 6 条），与原文数字无法对齐，也就失去了"在别人用过的数据上复现"
> 的意义。合并视图保留只为总览，**不进入 `all_reviews.parquet`**，避免与三张单模型表重复计数。

三条必须遵守的纪律：

1. **不编造**。拿不到的就说"未取得"，不拿别的小数据集改名顶替。
2. **不静默**。任何缺失、偏差、口径变化都写进 `results/data_quality_report.md`，随代码一起更新。
3. **不越权再分发**。原始评论内容受 Yelp / Amazon 服务条款约束，`data/` 已列入 `.gitignore`，
   仓库里只有下载脚本和校验值，不放原始文件。

## 1. 统一取数与校验机制

所有下载都走 `src/data/sources.py` 的清单，落盘前做三项校验（`src/data/download.py`）：

| 校验 | 作用 | 失败时 |
|---|---|---|
| 文件大小 | 识别被代理截断的大文件 | 报错并换下一个镜像 |
| **sha256（写死在代码里）** | 识别内容被篡改/损坏 | 报错，不用坏文件 |
| 行数 vs 官方公布 | 识别"下错了版本" | 偏差 > 2% 时在质量报告里强制提示 |

下载支持断点续传（HTTP Range），中断后重跑不会从头再来。同一个文件配多个镜像，
按可用性排序自动回退——本项目所在网络到 GitHub 直连不稳定，实测可用的是
`ghfast.top`、`cdn.jsdmirror.com`、`cdn.jsdelivr.net`、GitHub 内容 API 四条通道。

```bash
python src/build_unified_data.py download      # 下载（断点续传 + 多镜像回退 + sha256 校验）
python src/build_unified_data.py build         # 解析 + 校验 + 写 parquet + 写 manifest
python src/build_unified_data.py accounts      # 从评论级聚合出账号级表
python src/build_unified_data.py qa            # 生成 results/data_quality_report.md
python src/build_unified_data.py verify        # 校验产物与 manifest 一致（含 sha256）
```

### 1.1 校验码以及"为什么这些 sha256 可信"

FraudSquad 的文件只能经第三方代理取得（`ghfast.top` 等），因此必须能自证"代理没改过文件"。
做法是两级核对：

1. `src/data/sources.py` 里为每个文件写死 `sha256`，下载后逐个比对；
2. 同一文件另存 GitHub 官方 Trees API 返回的 **git blob sha1**（`OFFICIAL_GIT_BLOBS`），
   单元测试会把本地文件按 git 对象格式（`sha1("blob <size>\0" + 内容)`）重算并与官方值比对。

两者一致 ⇒ 本地字节 == 官方仓库字节 ⇒ 由它算出的 sha256 就是官方文件的 sha256。

| 文件 | 字节 | sha256（前 16 位） | 与官方 blob 一致 |
|---|---|---|---|
| `amazon_2022_5core.csv.zip` | 18,773,569 | `858b1f2f51d5fd78` | ✅ |
| `reviews_from_llama3_8b.csv` | 1,433,276 | `650d5fa18c527b2a` | ✅ |
| `reviews_from_qwen_72b.csv` | 1,468,340 | `71cd6f81dfdb6176` | ✅ |
| `reviews_from_qwen_dsr1.csv` | 1,520,028 | `e706b4341bd9d149` | ✅ |
| `QA.zip` | 21,535,547 | `e43276455d9cb275` | ✅ |
| `review_yelpHotelData_NRYRcleaned.txt` | 5,062,002 | `268502eb5f926c37` | ✅ |
| `label_yelpHotelData_NRYRcleaned.txt` | 11,708 | `ff8f6633cde64869` | ✅ |
| `meta_yelpHotelData_NRYRcleaned.txt` | 504,477 | `8fddb66e45758547` | ✅ |

YelpZip 例外：镜像站不提供官方摘要，只能以"文件大小 + 内部结构核对"佐证（见 §2.3），
写死的 sha256 `78f92e83574208f8…` 是**本项目的版本指纹**，不等于官方发布摘要——
它的作用是"下次换镜像/重下时能立刻发现内容变了"。

YelpNYC 同样没有官方摘要（作者只给 Dropbox 发布包），但它的正确性可以**正面证明**：
不是靠哈希，而是靠官方页公布的两个比例指标同时复现（见 §2.2 的核验表）。
两个文件的本项目指纹：

| 文件 | 字节 | sha256（前 16 位） | 来源 |
|---|---|---|---|
| `15-collective_opinion_spam_detection.zip` | 264,938,873 | `f16f20032a8efe9c` | 作者 Dropbox 发布包（整体） |
| `metadata` | 9,687,857 | `772df667e98341cb` | 从上述 zip 逐字节解出 |
| `reviewContent` | 236,566,411 | `dca37a1aa2d8bc11` | 从上述 zip 逐字节解出 |

## 2. 逐数据集说明

### 2.1 YelpChi

- **官方出处**：<https://shebuti.com/yelpchi-dataset/>（Shebuti Rayana，Stony Brook）
- **引用**：
  - Mukherjee, Venkataraman, Liu, Glance. *What Yelp Fake Review Filter Might Be Doing?* ICWSM 2013.
  - Rayana, Akoglu. *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* KDD 2015.
- **官方规模**：67,395 条评论 / 38,063 名用户 / 201 家商户（酒店与餐厅，芝加哥地区），13.23% 被过滤。
- **我们实际用的文件**：`data/raw/yelpchi/metadata.txt`（随项目流转的 5 列版本：
  `user_id product_id rating label date`，空格分隔、无表头）。
- **核验结果**：行数 67,395、用户 38,063、商户 201、假评率 13.23%，**四项与官方公布完全一致**。
- **已知缺口**：该版本**不含评论文本**（官方原始包里的 `reviewContent` 文件缺失）。
  文本方法（TF-IDF 等）在 YelpChi 上**跑不了**，这是硬约束，不能靠造空文本绕过。
  带文本版本与 YelpNYC 是同一封邮件申请（见 §2.2），至今未获回复。
- **替代来源排查（2026-11-12）**：已实测科学数据银行的「Yelp dataset」发布包
  （`dataset.zip`，265.29 MB，MD5 `7c6c35041c4eabb65aa024f2f1ecff74`）。
  包内与 YelpChi 相关的只有一个 `YelpChi.mat`，**不含文本**，且不是官方 67,395 条版本——
  详见 §2.6。结论：YelpChi 的评论文本目前**没有任何公开可得的来源**，只能等作者邮件，
  或改用在有文本的数据集上做文本类实验（YelpNYC / YelpZip / FraudSquad 都有文本）。
- **使用条款**：仅限学术研究；评论内容版权属 Yelp 及原作者，不得再分发或商用；
  论文中必须引用上述两篇文献。

### 2.2 YelpNYC（已取得，带完整文本）

- **官方出处**：<https://shebuti.com/yelpnyc-dataset/>
- **引用**：Rayana, Akoglu. KDD 2015.
- **官方规模**：359,052 条评论 / 160,225 名用户 / 923 家商户，**纽约市的餐厅（restaurant）评论**，
  **10.27% 被过滤、17.79% 的用户是 spammer**。
  > 订正两处早期错误：① YelpNYC 曾被写成"酒店评论"，实际是餐厅（YelpChi 才是酒店+餐厅混合）；
  > ② 假评比例曾被误抄为 13.22%——那是 **YelpZip** 的值，YelpNYC 官方公布就是 **10.27%**。
- **取得经过**：官方页只给邮件申请渠道，2026-11-02 发信未获回复。2026-11-12 改用官方页
  "here" 链接指向的作者 Dropbox 共享目录（<https://www.dropbox.com/sh/iqcuj0363zcj3go/AAAvbZVR_PSNyJX8AXUXpBqea?dl=0>），
  下载到发布包 `15-collective_opinion_spam_detection.zip`（264,938,873 字节）。
- **包内构成**：除 SpEagle 的 MATLAB 源码与 SentiWordNet 3.0 词表外，数据文件只有两个，
  两者靠 `(user, product, date)` 对齐：

  | 文件 | 字节 | sha256（前 16 位） | 列 |
  |---|---|---|---|
  | `metadata` | 9,687,857 | `772df667e98341cb` | `user / product / rating / label / date`（5 列，无表头） |
  | `reviewContent` | 236,566,411 | `dca37a1aa2d8bc11` | `user / product / date / text`（4 列，无表头） |

  本项目只把这两个文件放进 `data/raw/yelpnyc/`（另留 `README.txt` 留档），MATLAB 代码与
  SentiWordNet 词表未纳入。`label` 取值为 `1`（推荐）/ `-1`（过滤），已按 §3 的 scheme 统一翻转。

- **核验结果**：官方页没有提供任何摘要，所以正确性不是靠哈希证明的，而是靠**两个独立指标同时复现**：

  | 指标 | 官方公布 | 本项目实测 | 结论 |
  |---|---|---|---|
  | 评论数 | 359,052 | 359,052 | ✅ |
  | 用户数 | 160,225 | 160,225 | ✅ |
  | 商户数 | 923 | 923（product id 0~922 连续） | ✅ |
  | 过滤评论占比 | 10.27% | 36,885 / 359,052 = **10.2729%** | ✅ |
  | spammer 账号占比 | 17.79% | 28,496 / 160,225 = **17.7850%** | ✅ |

  五个数字同时吻合，其中比例是**独立的第二重证据**：如果 `label` 列被错解或错位，
  两个比例不可能同时对上（假评比例对上而 spammer 比例对不上，就说明评论文本贴错了人）。
  三套 Yelp 数据的双指标对账见 §2.3 末尾。

- **文本对齐（易静默出错的一步）**：`metadata` 与 `reviewContent` 是两个独立文件，
  贴错一行就意味着"某条评论挂着别人的标签"，且任何指标都看不出来。本项目做三层校验：
  ① 行数相等（359,052 = 359,052）；② 逐行 `(user, item, date)` 全等（实测 **0 行错位**）；
  ③ 任一不通过则退回键连接并要求匹配率 ≥ 99.5%，否则抛错拒绝生成。
  实测走的是最快路径（按位置贴）且已逐行验证。审计结果写进 `manifest.json` 的
  `text_join_audit`（`strategy=positional`、`matched=359,052`、`key_mismatches=0`）。

- **文本质量**：359,052 条全部有文本列，其中 359,049 条为非空正文；另 3 条原文件正文就是一个
  `-`（占位符），按缺失处理成空串。正文长度中位 470 字符、均值 637、最长 5,356 字符，
  无内嵌换行、无 Tab 溢出（解析器按 Tab 最多切 4 段，第四段之后原样保留）。

- **使用条款**：仅限学术研究，不得再分发原始评论内容；论文中必须引用 Rayana & Akoglu (KDD 2015)。

> 说明：FraudSquad 仓库里另有一个 `review_yelpHotelData_NRYRcleaned.txt`（5,854 条 Yelp 酒店评论），
> 名字看起来像 YelpNYC，但规模只有官方 YelpNYC 的 1.6%，是那篇论文为人类垃圾评论对照实验
> 采样的子集，**不能当作 YelpNYC 使用**。本项目仅把它作为辅助材料下载，没有拿它冒充。

### 2.3 YelpZip

- **官方出处**：<https://shebuti.com/yelpzip-dataset/>
- **引用**：Rayana, Akoglu. KDD 2015.
- **官方规模**：608,598 条评论 / 260,277 名用户 / 5,044 家餐厅，13.22% 被过滤。
- **我们实际用的文件**：`data/raw/yelpzip/yelpzip.csv`（410,865,364 字节，
  sha256 `78f92e83574208f8…`），通过 ModelScope 公开镜像
  <https://www.modelscope.cn/datasets/jimmyliu12345/yelpzip> 取得（官方页无直链）。
- **与官方 608,598 条的差额对账**（这是本次重点查清的问题）：

  | 项目 | 数量 | 说明 |
  |---|---|---|
  | 镜像文件数据行 | **608,598** | 与官方公布条数**完全一致** |
  | 其中带标签、可用 | 608,458 | 进入统一表 |
  | 其中导出残缺、不可用 | **140** | 只有 `user / product / date / text` 四列，缺 rating 与 label |
  | 用户并集 | 260,239 + 38 = **260,277** | 与官方公布用户数**完全一致** |
  | 商户数 | 5,044 | 完全一致 |
  | 假评率 | 13.22% | 实测 13.2201%，完全一致 |
  | spammer 账号占比 | 23.91% | 实测 62,214 / 260,239 = 23.9065%，一致 |

  **结论：不是解析丢行，是镜像文件本身有 140 行导出残缺。**
  这 140 行在导出时丢了 rating 与 label 两个字段、分隔符从逗号变成制表符，
  并且紧跟在某条"引号未闭合"的记录后面，被 CSV 解析器当成上一条评论文本吞了进去。
  后果有两个：① 140 条评论无标签、不可用；② 18 条正常评论的正文被污染成
  "原文（截断）+ 一串残缺行 + 另一条评论的尾巴"。
  统一数据层在读取阶段把这些残缺行从正文里剥离（`src/data/readers.py: strip_partial_rows`），
  18 条记录的正文已恢复干净，残缺行则计入审计（`manifest.json` 的 `partial_row_audit`）。
  行数与用户数两个口径同时对上官方值，可确认缺口来源；补齐需要官方原版文件。

  > 对下游的影响：`yelpzip` 评论级实验不受影响（缺的是 140 条无标签记录，占 0.023%）；
  > 账号级实验有 38 个账号因此缺位（占 260,277 的 0.015%），做账号级结论时需在报告中注明。

- **使用条款**：原始数据仅限学术研究；镜像页未声明额外许可，引用仍以 Rayana & Akoglu (KDD 2015) 为准。

#### 三套 Yelp 数据的双指标对账

官方页对每套数据都公布两个比例：**被过滤评论占全部评论的比例**、以及**写过过滤评论的用户占全部用户的比例**。
本项目三套全部实测复现，这是"数据没下错、标签没解错、文本没贴错"的最强证据：

| 数据集 | 评论数（官方/实测） | 用户数（官方/实测） | 过滤评论占比（官方/实测） | spammer 占比（官方/实测） |
|---|---|---|---|---|
| YelpChi | 67,395 / 67,395 ✅ | 38,063 / 38,063 ✅ | 13.23% / **13.2339%** ✅ | 20.33% / **20.3321%** ✅ |
| YelpNYC | 359,052 / 359,052 ✅ | 160,225 / 160,225 ✅ | 10.27% / **10.2729%** ✅ | 17.79% / **17.7850%** ✅ |
| YelpZip | 608,598 / 608,458（差 140 行残缺） | 260,277 / 260,239（差 38） | 13.22% / **13.2201%** ✅ | 23.91% / **23.9065%** ✅ |

> 三套的过滤比例**并不相同**（13.23% / 10.27% / 13.22%），这是官方页各自公布的数值，
> 不是抄写错误。YelpNYC 之所以低约 3 个百分点，官方未作说明；本项目按原始文件的 `label`
> 列为准，不引用论文表格里的近似值。
> 顺序也有意思：过滤评论比例 YelpChi ≈ YelpZip > YelpNYC，而 spammer 比例恰好相反
> （YelpZip 23.91% > YelpChi 20.33% > YelpNYC 17.79%）——说明 YelpNYC 的 spammer 平均
> 每人写的假评论更少（实测 YelpNYC 有 84.69% 的 spammer 只写过 1 条假评论）。
> 做账号级实验时这是一个可用的差异点。


### 2.4 FraudSquad-Amazon（大模型水军，拆成三张表）

- **官方出处**：<https://github.com/RafaelLiuZH/FraudSquad>（作者实现仓库）
- **引用**：Liu et al. *Detecting LLM-Generated Spam Reviews by Integrating Language Model
  Embeddings and Graph Neural Network*（FraudSquad）。
- **文件与规模**：

  | 文件 | 条数 | 账号数 | 说明 |
  |---|---|---|---|
  | `amazon_2022_5core.csv` | 86,758 | 8,092 | 真实评论基线（Amazon Reviews 2023 的 5-core 子集），7,617 个商品 |
  | `reviews_from_llama3_8b.csv` | 2,489 | 1,250 | Llama3-8B 生成，全部 5 星，499 个商品 |
  | `reviews_from_qwen_72b.csv` | 2,495 | 1,250 | Qwen-72B 生成，全部 5 星，499 个商品 |
  | `reviews_from_qwen_dsr1.csv` | 2,500 | 1,250 | Qwen-DeepSeek-R1 生成，全部 5 星，500 个商品 |
  | `QA.zip` | 37,998 + 24,953 行 | — | **中文社区问答（CQA）协同刷分数据**（论文中称 ChineseQA），来源 Liu et al., WWW 2017，**不是 Yelp** |
  | `review_yelpHotelData_NRYRcleaned.txt` | 5,854 | — | Yelp 酒店评论切片，辅助材料，**不可当作 YelpNYC** |

  > 订正：早期文档把 `QA.zip` 写成"FraudSquad 的 Yelp QA 人类垃圾评论集"，这是错的。
  > 它的 readme 写明来自 `thuir.cn/.../WWW2017.zip`，对应论文 Yuli Liu, Yiqun Liu, Ke Zhou,
  > Min Zhang, Shaoping Ma，*Detecting Collusive Spamming Activities in Community Question
  > Answering*，WWW 2017（DOI 10.1145/3038912.3052594）；内容是百度知道类的**中文问答**协同刷分，
  > 与 Yelp、与酒店评论都无关。本项目**没有**用它做任何建模。

- **三张表怎么构成**：`真实基线 86,758` + `单一模型生成集` → 89,247 / 89,253 / 89,258。
  每张表都是 8,092 个账号，其中 1,250 个是**被接管账号**（论文设定：接管既有账号写假评论）。
- **核验结果**：
  - 三个生成集的账号与商品 **100%** 落在真实基线内（可建图）；
  - 每账号假评论数：Llama3 为 1,239 个 2 条 + 11 个 1 条；Qwen2 为 1,245 个 2 条 + 5 个 1 条；
    DSR1 为 1,250 个全部 2 条 —— 与论文"每账号约 2 条"一致；
  - 三个模型的 1,250 个账号**是同一批**（集合完全相同），跨模型比较属配对比较。
- **两个必须知道的陷阱**（详见质量报告 §5）：
  1. **被接管账号同时保留真实评论**：1,250 个账号 **100%** 在真实基线里也有真实评论（中位数 10 条）。
     这正是论文的设定，所以**账号级实验可以做**：正例 = 被接管账号，负例 = 其余 6,842 个账号，
     用 `python src/build_unified_data.py accounts` 生成。
     但**评论级**不能用"账号是否被接管"当标签——那会把这些账号的历史真实评论也标成假的。
     （此处曾写成"不能做账号级实验"，是错的，已订正。）
  2. **评分是捷径特征**：生成评论全部 5 星，而真实评论 5 星占比 69.6%。任何吃评分的模型都会虚高，
     必须把"仅用评分"作为对照基线一起报出来，否则结果不可信。
- **使用条款**：仓库**未声明 LICENSE**；真实评论部分来自 Amazon Reviews 2023 公开数据集，
  受 Amazon 数据条款约束。本项目仅作学术研究引用，不再分发原始文件。

### 2.5 YelpChi + 300 注入账号（自建）

- **来源**：本项目 `src/step3_inject_water_army_pro.py` 的事件层注入（v3.0，2026-10-24），
  注入对象为 `n_reviews >= 5` 的正常账号中随机 300 个，每个账号新增 5~14 条评论，
  三种行为模板各 100 个（`A_Time_Burst` / `B_Rating_Polarization` / `C_Stealth_Mix`）。
- **本次做的事**：把 `data/processed/injection_log_v2.json` 里的 `events_added` 展开成评论行，
  与 YelpChi 原始评论合并成评论级视图（**不重新跑注入**，保证与既有实验结果逐条一致）。
  结果：67,395 + 2,791 = 70,186 条，用户 38,063 个（注入不新增账号，只给已有账号加评论）。
- **账号级标签**：口径为 `injected`，正例 = 被注入过评论的 **300** 个账号（用 `n_injected` 判定，
  不用 `label`——`label` 里还混着 Yelp 平台过滤的假评论，两者不能混）。
- **已知缺口**：注入评论**没有文本**（step3 只生成日期/评分/商品的结构化事件）。
  文本方法需把注入行排除。
- **使用条款**：派生数据，继承 YelpChi 的全部使用限制。

### 2.6 已排查但未采用的第三方镜像（科学数据银行）

**为什么要查**：YelpChi 的 `reviewContent` 一直拿不到，需要确认"是否真有公开的带文本版本"。

**查了什么**：科学数据银行（SciDB）上有一份题名 *Yelp dataset* 的重发布镜像，
说明写"including the YelpChi, YelpNYC, and YelpZIP datasets"，标称 265.29 MB。
下载后 MD5 与页面公布值 `7c6c35041c4eabb65aa024f2f1ecff74` **一致**（文件完整）。

**包里到底有什么**（解开后逐项核对，不是看标题下结论）：

| 包内文件 | 实际内容 | 本项目是否需要 |
|---|---|---|
| `YelpNYC.zip` | 两个 Excel：`Yelp Metadata.xlsx` 359,052×5、`Yelp dataset.xlsx` 359,052×4 | ❌ 已有官方 raw 版（metadata + reviewContent），Excel 版是二手重排，反而有截断/转义风险 |
| `YelpChi.mat` | 45,954×32 稀疏特征 + `net_rur` / `net_rtr` / `net_rsr` 三张 45,954² 关系图 + `label`（6,677 正例，14.53%） | ❌ 这是 CARE-GNN（Dou et al., CIKM 2020）的**基准子集**，无文本，且比官方 67,395 条少 21,441 条 |
| `yelpzip.csv` | 410,865,364 字节，带 text 列 | ⚠️ 与本项目已有文件 **sha256 完全相同**（`78f92e83…517fe`），无新增信息 |

**三条结论**：

1. **YelpChi 的评论文本仍然没有**。`YelpChi.mat` 里只有特征矩阵与关系图，`whosmat` 只列出
   `homo` / `net_rur` / `net_rtr` / `net_rsr` / `features` / `label` 六个变量，没有任何字符串变量。
   谁把这份 `.mat` 当"YelpChi 带文本版"，谁就错了。
2. **YelpZip 的 140 行残缺不是本项目造成的**。两个不同来源的镜像给出**逐字节相同**的文件，
   说明残缺来自这份流传的导出文件本身。这条是本项目 `results/data_quality_report.md` §2.1
   第三条证据（数量层面对账）之外的**独立旁证**。
3. **不适合拿它当备选数据集**。它没有提供任何我们没有的东西；把它当"第 4 个数据集"是重复计数。

> 备查：若后续需要 CARE-GNN 那套基准矩阵（图方法组做对比实验可能用得上），
> 直接取 <https://github.com/YingtongDou/CARE-GNN> 的 `data/YelpChi.zip`（很小）即可，
> 不要用这里的 207 MB `.mat`；两者内容等价，但后者体积大且来源不可追溯。

## 3. 引用格式（论文参考文献可直接用）

```bibtex
@inproceedings{rayana2015collective,
  title={Collective opinion spam detection: Bridging review networks and metadata},
  author={Rayana, Shebuti and Akoglu, Leman},
  booktitle={Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={985--994},
  year={2015}
}

@inproceedings{mukherjee2013yelp,
  title={What yelp fake review filter might be doing?},
  author={Mukherjee, Arjun and Venkataraman, Vivek and Liu, Bing and Glance, Natalie},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={7},
  number={1},
  pages={409--418},
  year={2013}
}

@article{liu2025fraudsquad,
  title={Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network},
  author={Liu, Rafael and others},
  year={2025},
  note={arXiv 首版 2025-10，2026-04 修订；Code: https://github.com/RafaelLiuZH/FraudSquad}
}
```

> 年份口径：按组长审核意见写 **2025**（arXiv 首版年份），不写 2026。
> `schema.py` 里四个 FraudSquad 规格的 `citation` 字段已统一为 2025。

## 4. 合规自查清单（提交论文前逐条打勾）

- [ ] 数据来源小节写明了每个数据集的获取途径，镜像渠道也写了
- [ ] 引用了 Rayana & Akoglu (KDD 2015)；用了 YelpChi 还引 Mukherjee et al. (ICWSM 2013)
- [ ] 写明了 Yelp 标签是"平台过滤器结果"而非人工标注
- [ ] 写明了 YelpNYC 的取得途径（作者 Dropbox 发布包）与其 10.27% 的过滤比例
      ——**不要沿用 13.22%**，那是 YelpZip 的值
- [ ] 写明了 YelpChi 持的是无文本的 `metadata` 版本，文本类方法未在 YelpChi 上运行
- [ ] 写明了 FraudSquad 是三个独立数据集而非一份合并数据
- [ ] 写明了 YelpZip 实际使用 608,458 条（另有 140 行导出残缺不可用），不直接套用 608,598
- [ ] 未在仓库或论文附录中附带原始评论数据文件
- [ ] 对比表标注了每种方法是否需要标签
