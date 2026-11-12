# 数据集来源、核验与使用条款

> 本文档回答三个问题：**数据从哪来**、**我们怎么确认拿到的是对的**、**能怎么用、不能怎么用**。
> 生成统一格式的命令与字段说明见 [`data_dictionary.md` 第十一节](../../data_dictionary.md)，
> 数据质量实测结果见 [`results/data_quality_report.md`](../../results/data_quality_report.md)，
> YelpNYC 未取得的完整处理方案见 [`yelpnyc_fallback.md`](yelpnyc_fallback.md)。

## 0. 一页结论

| 数据集 | 论文口径 | 状态 | 评论数 | 带文本 | 标签来源 | 主要用途 |
|---|---|---|---|---|---|---|
| YelpChi | 是 | ✅ 已构建 | 67,395 | ❌ | Yelp 过滤器 | 图方法、无监督检测主基准 |
| YelpNYC | 是 | ⚠️ **未取得** | — | — | — | 邮件申请未回复，见 `yelpnyc_fallback.md` |
| YelpZip | 是 | ✅ 已构建 | 608,458（另有 140 行残缺不可用） | ✅ | Yelp 过滤器 | 大规模 + 文本基线 |
| FraudSquad-Llama3 | 是 | ✅ 已构建 | 89,247（生成 2,489） | ✅ | 构造即真值 | 大模型水军实验 |
| FraudSquad-Qwen2 | 是 | ✅ 已构建 | 89,253（生成 2,495） | ✅ | 构造即真值 | 同上，换模型 |
| FraudSquad-Qwen-DSR1 | 是 | ✅ 已构建 | 89,258（生成 2,500） | ✅ | 构造即真值 | 同上，换模型 |
| FraudSquad-Amazon（合并视图） | **否** | ✅ 已构建 | 94,242（生成 7,484） | ✅ | 构造即真值 | 仅总览，不作对比基准 |
| YelpChi + 300 注入账号 | 否（自建） | ✅ 已构建 | 70,186（注入 2,791） | ❌ | 注入脚本保证 | 解释方法（DCFO）与规避实验 |

> **FraudSquad 为什么拆成三张表**：论文里三个模型是三个独立数据集（真实基线 + 单一模型生成的
> 假评论，每个被接管账号约 2 条假评论）。合并成一份后，同一个账号会攒到 4~6 条假评论
> （实测 930 个账号 4 条、304 个账号 6 条），与原文数字无法对齐，也就失去了"在别人用过的数据上复现"
> 的意义。合并视图保留只为总览，**不进入 `all_reviews.parquet`**，避免与三张单模型表重复计数。

三条必须遵守的纪律：

1. **不编造**。YelpNYC 拿不到就写"未取得"，不拿别的小数据集改名顶替。
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
- **使用条款**：仅限学术研究；评论内容版权属 Yelp 及原作者，不得再分发或商用；
  论文中必须引用上述两篇文献。

### 2.2 YelpNYC（未取得）

- **官方出处**：<https://shebuti.com/yelpnyc-dataset/>
- **引用**：Rayana, Akoglu. KDD 2015.
- **官方规模**：359,052 条评论 / 160,264 名用户 / 923 家商户，**纽约市的餐厅（restaurant）评论**，
  13.22% 被过滤。
  > 订正：项目早期文档把 YelpNYC 写成"酒店评论"，这是错的。YelpChi 才是酒店+餐厅混合，
  > YelpNYC / YelpZip 都是餐厅。
- **状态**：**未取得**。发布页明确写着 "To get the datasets with ground truth please email:
  srayana@cs.stonybrook.edu"，没有任何直链。
- **申请经过**：2026-11-02 前后用学校邮箱发信申请（YelpNYC + YelpChi 带文本版 + YelpZip 官方版），
  至 2026-11-12 未获回复；期间复查过 ModelScope、HuggingFace 镜像、Kaggle 公开 API、
  Science Data Bank、figshare、GitHub 各图方法仓库、Gitee/GitCode、Wayback，均无官方版本。
  详细渠道清单与判断见 [`yelpnyc_fallback.md`](yelpnyc_fallback.md)。
- **一条还没试完的路**：官方页的 "here" 链接指向作者 Dropbox 共享目录
  <https://www.dropbox.com/sh/iqcuj0363zcj3go/AAAvbZVR_PSNyJX8AXUXpBqea?dl=0>，
  本机 DNS 被污染不可达，**换校园网/热点用浏览器打开**是最短路径。
- **补齐方式**：拿到后把文件放进 `data/raw/yelpnyc/`，重跑 `build --dataset yelpnyc` 即自动纳入，
  **不需要改代码**（`build_yelpnyc()` 会扫描该目录下的 `*.csv` / `*.txt`）。
  注意：`reviewContent` 的解析器尚未实现（见 `yelpnyc_fallback.md` §5 的提醒）。
- **使用条款**：同上，仅限学术研究，不得再分发。

> 说明：FraudSquad 仓库里有一个 `review_yelpHotelData_NRYRcleaned.txt`（5,854 条 Yelp 酒店评论），
> 名字看起来像 YelpNYC，但规模只有官方 YelpNYC 的 1.6%，是论文作者为人类垃圾评论对照实验
> 采样的子集，**不能当作 YelpNYC 使用**。我们已下载它作为辅助材料，但没有冒充 YelpNYC。

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
  | 假评率 | 13.22% | 完全一致 |

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
- [ ] 写明了 YelpNYC 的实际状态（未取得），未在对比表里编造数字
- [ ] 写明了 FraudSquad 是三个独立数据集而非一份合并数据
- [ ] 写明了 YelpZip 实际使用 608,458 条（另有 140 行导出残缺不可用），不直接套用 608,598
- [ ] 未在仓库或论文附录中附带原始评论数据文件
- [ ] 对比表标注了每种方法是否需要标签
