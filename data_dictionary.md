# YelpChi 行为特征与注入数据集 - 数据字典

本数据字典适用于基于 YelpChi 真实评论数据（`metadata.txt`）构建的完整数据管线。彻底摒弃了图结构数据的错配问题，确保特征计算的物理严谨性。

## 一、 交付文件清单与生成链路
1. **`YelpChi_reviews.csv`**: 评论级原始数据（约 6.7 万条）。由 `step0` 生成，含真实时间戳与评分。
2. **`clean_accounts.parquet`**: 账号级干净底表。由 `step1` 生成，包含真实标签与所有行为特征。
3. **`yelpchi_1000_accounts.csv`**: 账号级基准样本（1000 条）。由 `step2` 生成，供基线训练。
4. **`augmented_accounts_v2.parquet`**: 注入水军后的全量底表。由 `step3` 生成。
5. **`injection_log_v2.json`**: 水军注入 Ground Truth 明细。由 `step3` 生成。
6. **`scaler_params.json`**: 全量特征的 Z-score 缩放参数。由 `step1` 生成。

## 二、 字段说明 (以 `clean_accounts.parquet` 为准)
### 1. 基础字段
*   `user_id`: 账号唯一标识。
*   `n_reviews`: 该账号的真实评论总数。
*   `account_is_fake`: 账号级标签（**1 = 水军**，**0 = 正常**。只要该用户发过一条 `label=-1` 的评论，即标为 1）。
*   `is_injected`: 仅存在于注入表中（1 = 人工注入，0 = 原生）。

### 2. 行为特征字段
*   `active_days`: 注册活跃天数（最后一次评论与第一次评论的时间差）。
*   `daily_avg_reviews`: 日均发评数（总评论数 / 活跃天数）。
*   `max_daily_reviews`: 单日最大发评数。
*   `burstiness`: 爆发性（相邻评论时间间隔的变异系数）。**注：评论数 < 3 的账号该值为 NaN，代表数据不足以计算。**
*   `rating_deviation`: 评分偏离度（打分减去商家均分），范围严格限制在 [-4, 4]。
*   `extreme_ratio`: 极端评分比例（1星或5星比例），范围在 [0, 1] 之间。
*   `[特征名]_z`: 上述特征对应的全局 Z-score 标准化值。

## 三、 水军注入实验说明 (事件层行为模拟，v3.0 / 2026-10-24)

*   **注入对象**：`clean_accounts.parquet` 中 `n_reviews >= 5` 的正常账号，`random_state=42` 抽 300 个（与原版同一批账号，便于和历史结果对照），每号新增 5~14 条评论。
*   **注入原理（已从"改特征值"改为"造事件"）**：为每个账号生成 (日期, 评分, 商品) 三元组追加进真实流水，再用与 `step1` **逐位相同**的公式重算全部 7 个特征。因此"单日发评 > 总评论数""日均 != 总数/活跃天数"这类矛盾由构造消除，不再靠手工同步。
    *   **时间**：三条模板都只落在账号原活跃区间 `[首条, 末条]` 内，`active_days` 严格不变（避免"沉寂多年后突然复活"的痕迹）。
        *   `A_Time_Burst`：全部压在最末活跃日 → 单日爆发，`max_daily` 与 `burstiness` 显著上升；
        *   `B_Rating_Polarization`：只撒在活跃期的**空闲日**（每天 1 条）→ 单日峰值不变，时间维度影响最小；
        *   `C_Stealth_Mix`：40% 压末活跃日 + 60% 撒开 → 介于两者之间。
    *   **评分**：新增评论 90~95% 为 5 星、其余 1 星（全部是极端分）。
    *   **商品**：每条一个新商品，且避开本人已评商品（YelpChi 中一个用户对同一商品只有一条评论）。
*   **爆发性方向已修正**：`burstiness` 由真实时间戳按 `calc_burstiness` 重算（间隔 std/mean，<3 条或全部同日为 NaN）。集中爆发会混入大量 0 间隔，`std/mean` 上升——方向由公式保证，不再依赖假设。原本为 NaN 的账号保持 NaN。
*   **可控边界（如实声明，不掩盖）**：真实历史不可编辑，`n_reviews` 只能增不能减。当账号活跃期只有 1~2 天、没有任何空闲日时（本批 300 个里有 2 个：uid 5683、9296），新增评论只能并入原唯一活跃日，`max_daily_reviews` 必然随总数上调。这是物理必然，日志中完整记录，且 `n_reviews <= active_days * max_daily_reviews` 由构造保证成立。

## 四、 注入日志 (`injection_log_v2.json`)
顶层仍是**记录数组**（`feature_eda.py` 按 `user_id` / `injection_template` 读取，格式未变），每条记录包含：
*   `user_id` / `injection_template`：账号 ID 与模板名（A_Time_Burst / B_Rating_Polarization / C_Stealth_Mix），与旧版字段一致。
*   `policy`：本次注入的行为参数——`n_add`、`temporal`（burst_last_day / spread_free_days / cluster+spread）、`cluster_size`、`free_days_used`、`p_five_star`。用于回答"这次注入模拟了什么行为"。
*   `events_added`：逐条 `{date, rating, product_id}`，即真实注入的事件明细（可复核、可重放）。
*   `modified_features`：只列**实际发生变化的**特征，形如 `max_daily_reviews: {original: 1, mutated: 12, delta_sigma: 22.8}`。
*   `feature_delta_sigma`：全部 7 个特征的 σ 单位增量（DCFO 反事实解释的 ground truth 对齐基准，含未变化项=0）。
*   `n_real` / `n_fake`：注入前真实条数、本次新增条数。
*   `physical_checks`：`passed`（注入前后 9 项断言全部通过才落盘）。

# 五、特征交接说明（B 成员交付，面向建模）

## 1. 交接文件
`feature_table.csv`（38,063 行 × 18 列），生成命令：`python src/features/build_feature_table.py`（脚本仅写入 `data/processed/`；项目根目录的共享副本由队伍同步时手动复制，不进 git）。
该脚本是特征口径的**唯一出口**：burstiness 填充、缺失标记、`_z` 统一重算都在此完成，下游（LOF / iForest / DCFO）一律以本表为准。

## 2. 列结构与使用纪律
| 列组 | 列 | 用途 |
|---|---|---|
| 标识/评估列 | `user_id`, `account_is_fake`, `is_injected` | 仅用于评估与实验分组，**严禁进入无监督模型输入** |
| raw 特征列（7） | 7 个原始量纲特征 | 供 DCFO 反事实解释回译为原始单位（如"单日发评降到 3 条以下"） |
| `_z` 列（7） | 7 个标准化特征 | **无监督模型输入** |
| 缺失标记列 | `burstiness_missing` | 86.5% 账号的 burstiness 为中位数填充值，此标记区分"真实值/填充值"，**进入模型输入** |

**约定：C 的模型输入 = 7 个 `_z` 列 + `burstiness_missing`，共 8 列**（`burstiness_missing` 属于模型输入，不属评估列，与上表一致）。

## 3. 特征定义与水军暴露逻辑
| 特征 | 公式 | 单位/范围 | 暴露逻辑 |
|---|---|---|---|
| n_reviews | 评论总数 | 条 | 刷评直接推高总量 |
| active_days | 首末评论间隔天数 +1 | 天 | 突击型账号活跃期短 |
| daily_avg_reviews | n_reviews / active_days | 条/天 | 突击刷评推高日均 |
| max_daily_reviews | 单日最大发评数 | 条 | 定时任务式刷评单日爆发 |
| burstiness | 相邻评论间隔的变异系数 (std/mean) | 无量纲，≥0 | 集中爆发会在间隔序列混入大量近 0 间隔，波动增大，取值应**偏高**；评论数<3 或全部同日发出时无法定义（NaN）。注：当前 step3 实现为降低，方向待修（见 §9 注） |
| rating_deviation | mean(评分 − 商家均分) | 星，[-4,4] | 集中打极端分，偏离商家均分 |
| extreme_ratio | (1星或5星评论数) / 总数 | [0,1] | 刷评多为极端评分 |

## 4. 标准化口径
`_z = (x − mean) / std`，参数统一来自 `scaler_params.json`（clean 全量底表统计，NaN 不参与统计）。
链路：step1 生成参数 → step3 按同参数重算注入行 → 本脚本对全表再统一重算一遍，全程唯一口径。
**已声明局限**（声明而非修复）：`rating_deviation` 的商家均分在含假评的全量评论上计算，存在轻微混叠；无监督场景可接受，汇报时如实说明。

## 5. burstiness 缺失处理
NaN 共 32,918/38,063（86.5%）：其中 32,666 个为评论数 < 3（数据不足），3,252 个为评论数 ≥ 3 但全部评论同日发出（间隔均值为 0）。
策略：以 **clean 底表观测值中位数 0.9436** 填充（与 scaler 参数同源、不受注入分布污染），并保留 `burstiness_missing` 标记列。
建议 C 做敏感性分析：带/不带 `burstiness_missing` 各跑一遍，AUC 差异 < 1pct 即认为填充无副作用。
注入行仅 2/300 为 NaN（注入池要求 n_reviews ≥ 5），该特征对注入组信息量充足。

## 6. 组间比较陷阱（重要）
注入组 `active_days_z` 组均值偏高（+2.24）是**选池偏差**（注入池只选 n_reviews ≥ 5 的正常号），不是注入效应——active_days 严格未动（dz=0）。
评估注入效应必须用**配对差值 dz**（同一账号注入前后 `_z` 变化）。v3.0 实测基准（2026-10-24，n=100/模板）：

| 特征 | A_Time_Burst | B_Rating_Polarization | C_Stealth_Mix |
|---|---|---|---|
| n_reviews | +4.35 | +4.60 | +4.32 |
| active_days | 0 | 0 | 0 |
| daily_avg_reviews | +0.48 | +0.32 | +0.14 |
| max_daily_reviews | +18.61 | +0.37 | +6.71 |
| burstiness | **+2.34** | −0.21 | **+0.35** |
| rating_deviation | +0.48 | +0.36 | +0.41 |
| extreme_ratio | +0.76 | +0.78 | +0.73 |

读法：A 的主导缺陷在时间维度（max_daily/burstiness），B 几乎不动时间维度、主导缺陷在评分维度，C 介于两者。`daily_avg_reviews` 的模板间差异来自活跃期极短的少数账号（分母小，同样新增条数带来的 dz 极大），不代表模板设计差异。

**选池偏差对评估的三条纪律**（同样适用于 AUC，不仅是组均值）：
1. 论文主结论只用原生标签（account_is_fake）结果；注入组仅用于验证解释方法（DCFO）。
2. 注入号选自 n_reviews≥5 的更活跃正常号，含注入的 AUC 天然偏高，不得作为检测性能证据。
3. 需要与注入组对比时，对照组必须用同条件（n_reviews≥5）的未注入正常号，而非全体正常号。

## 7. 特征 ↔ 注入模板对照（DCFO 验证的 ground truth）
| 模板 | 数量 | 主导缺陷（设计意图） | 实测 dz 排序（前 3） |
|---|---|---|---|
| A_Time_Burst | 100 | 时间爆发 | max_daily +18.6 ≫ burstiness +2.3 > extreme_ratio +0.8 |
| B_Rating_Polarization | 100 | 评分极化 | extreme_ratio +0.8 > rating_deviation +0.4 ≫ max_daily +0.4 |
| C_Stealth_Mix | 100 | 混合 | max_daily +6.7 > extreme_ratio +0.7 > rating_deviation +0.4 |

**重要变化（v3.0）**：事件层注入下，模板不再能做到"只改某几个特征"。新增一条评论必然同时改变 `n_reviews`、`daily_avg`、评分类特征，并可能改变 `max_daily`/`burstiness`——这是"特征由事件推导"的必然结果，也正是旧版出现造假痕迹的原因。模板之间的真正差异是**各类特征变化的主导程度**（见上表），而不是严格的特征隔离。

因此 DCFO 验证的 ground truth 应以日志里的 `feature_delta_sigma`（7 维全量）为准，而不是"模板名义上要改哪几个特征"。个别账号"被修改但数值不变"也如实记录（如原本全 5 星账号的 extreme_ratio 仍为 1.0），日志 original=mutated 属预期。

## 8. 单特征区分度（ROC-AUC vs account_is_fake）
| 特征 | 全表（含注入） | 仅原生（YelpChi 真实标签） |
|---|---|---|
| n_reviews_z | 0.414 | 0.391 |
| active_days_z | 0.415 | 0.396 |
| daily_avg_reviews_z | 0.579 | 0.593 |
| max_daily_reviews_z | 0.489 | 0.473 |
| burstiness_z | 0.494 | 0.504 |
| rating_deviation_z | 0.506 | 0.499 |
| extreme_ratio_z | 0.573 | 0.571 |
| burstiness_missing | 0.546 | 0.564 |

解读：单特征区分度普遍偏弱（0.39–0.59）。原生水军呈"**少评、短活、高日均、极端评分**"画像（n_reviews / active_days 为低值方向，daily_avg_reviews / extreme_ratio 为高值方向）——这正是需要多变量无监督方法（LOF/iForest）与多信号融合的依据，而非单特征阈值。注入会推高 n_reviews（隐蔽性设计），故全表与原生方向一致但数值不同。

## 9. EDA 图件与方向性判定（复现命令：`python src/viz/feature_eda.py`）
| 图件 | 内容 |
|---|---|
| `results/feature_eda_group_box.png` | 原生正常 / 原生水军 / 注入水军 三组 7 特征 `_z` 箱线图（burstiness 按未填充值绘制，NaN 不参与；箱体隐藏离群点） |
| `results/feature_eda_template_dz.png` | 注入前后配对 dz 的模板级均值对比（A/B/C 三色柱状图），消除选池偏差 |

v3.0 实测（2026-10-24）：9 项 PASS + 3 项由硬判据降级为观测项（原因见下）。
- 全模板公共效应：n_reviews dz>0（300/300，均值 +4.42）、daily_avg_reviews dz>0（300/300，均值 +0.31）、extreme_ratio dz>=0（296/300 上升，其余 4 个本就全 5 星，数值不变）；active_days dz=0（300/300，未动）。
- 模板特异性：A/C 的 max_daily_reviews dz>0（+18.6 / +6.7）；B/C 的 rating_deviation dz>0（+0.36 / +0.41）。
- **burstiness 方向已修复（Issue1 已关闭）**：A +2.34、C +0.35，均为正。机制：新增评论压在同一活跃日会引入大量 0 间隔，`std/mean` 上升；数值由 `calc_burstiness` 直接对合并后的时间戳算出，不再乘系数、不再依赖假设。
- **B 的 burstiness / max_daily 不再"恒为 0"，降级为观测项**：B 只撒空闲日，`max_daily` 的 +0.37 全部来自 2 个没有任何空闲日的账号（uid 5683、9296 —— 正是 Issue2 点名的那两个），其余 98 个账号该值严格不变；`burstiness` −0.21 是因为往账号的安静期插入评论会让间隔序列趋于均匀（变异系数向 1 收敛），属"插入时间点"的数学后果，不是模板效应。
- **A 的 rating_deviation 不再"恒为 0"，降级为观测项**：A 新增评论都是 5 星，必然抬高评分偏离（+0.48）。旧版为 0 是因为它根本没生成评分，只改了几个数值。
- 另：`feature_eda.py` 里 3 条 `dz==0` 的硬判据（B 的 max_daily、B 的 burstiness、A 的 rating_deviation）在"事件层注入"下不可能成立——除非继续造假。需由 B 改为容差判定或观测项，具体建议见 PR 说明。
读图提示：图 1 中"原生水军"的 daily_avg_reviews 箱体退化为一条线（约 z=0.6），是因为大量原生假号只有 1 条评论（1/1 天），取值高度集中——这本身是 YelpChi 假号"少评短活"画像的证据。

## 10. D 模块产出：图算法社区发现
*   **`data/processed/graph_scores.csv`**：由 `src/models/graph_community_detection.py` 生成，包含图算法对全量账号的社区划分与打分。
    *   `user_id`: 账号唯一标识。
    *   `group_id`: Louvain 图算法划分的社区/团伙 ID。
    *   `graph_score`: 账号级图可疑分数，范围 `[0, 1]`。
        *   **计算口径**：`(该账号在团伙内的度数 / 团伙总人数)`，随后进行全局最大值归一化。
        *   **解读**：分数越高，说明该账号在团伙内的连接越密集，越可能是“核心节点”。经实测核对前 10 名高分账号，最高分（1.0）出自 392 人的大团伙核心，印证了公式在极度抱团的大团伙中依然能准确识别出核心。单人孤立团伙分数为 0。

# 十一、多数据集统一格式（数据组交付，2026-11）

现有公开数据整理成**评论级**同一套 schema：YelpChi、**YelpNYC**、YelpZip、FraudSquad 的三个
单模型数据集，外加一份自建注入集——**论文口径的 6 张表全部到位，没有缺失数据集**。
唯一剩余的数据缺口是 YelpChi 缺评论文本（持有的 `metadata` 版本不含 `reviewContent`），
影响面仅为「文本类方法不能在 YelpChi 上跑」。
生成命令、来源核验与使用条款见 [`docs/data/provenance.md`](docs/data/provenance.md)，
实测质量见 [`results/data_quality_report.md`](results/data_quality_report.md)。

## 1. 为什么是评论级，以及为什么要再做账号级

评论级是唯一稳定的公共粒度：它天然带 user / item / rating / date / label / text，
四个数据集的账号口径互不相同（Yelp 按"发过被过滤评论"、FraudSquad 按"账号被接管"、
我们的注入集按"是否被注入过"），直接拼账号级表会把口径混在一起。

但**账号级实验本身是可以做的**（这一点在首版里写错了，已订正）：FraudSquad 的构造方式就是
"接管一批既有账号、让它们写假评论"，与我们要研究的水军形态一致。正确做法是
"账号级分类（被接管 vs 普通）"与"评论级分类（这条评论真假）"分开做，
**不能**把账号标签下放到评论级——那会把这个账号的历史真实评论也标成假的。
账号级表由 `python src/build_unified_data.py accounts` 从评论级聚合得到。

## 2. 交付文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `data/processed/unified/yelpchi.parquet` | 67,395 | YelpChi，**无文本** |
| `data/processed/unified/yelpnyc.parquet` | 359,052 | YelpNYC，带文本（官方 `metadata` + `reviewContent` 双文件按 `(user, item, date)` 合并） |
| `data/processed/unified/yelpzip.parquet` | 608,458 | YelpZip，带文本（另有 140 行导出残缺不可用，见 §5.1） |
| `data/processed/unified/fraudsquad_llama3.parquet` | 89,247 | 真实 86,758 + Llama3-8B 生成 2,489 |
| `data/processed/unified/fraudsquad_qwen2.parquet` | 89,253 | 真实 86,758 + Qwen-72B 生成 2,495 |
| `data/processed/unified/fraudsquad_qwen_dsr1.parquet` | 89,258 | 真实 86,758 + Qwen-DSR1 生成 2,500 |
| `data/processed/unified/fraudsquad_amazon.parquet` | 94,242 | **合并视图，仅供总览**，不作对比基准 |
| `data/processed/unified/yelpchi_injected.parquet` | 70,186 | YelpChi 67,395 + 注入 2,791 |
| `data/processed/unified/all_reviews.parquet` | 1,372,849 | 除合并视图外的 7 张表合并（`dataset` 列区分） |
| `data/processed/accounts/<dataset>.parquet` | 见 §6 | 账号级表，8 个数据集各一份 |
| `data/processed/unified/manifest.json` | — | 构建时间、原始文件 sha256、统计、官方规模对照、文本对齐与残缺行审计 |
| `results/data_quality_report.md` | — | 人读的质量报告（含"使用陷阱"与差额对账） |

> ⚠️ `all_reviews.parquet` 里 **86,758 条真实基线出现三次**（三张单模型表各带一份，这是"同基线公平对比"
> 的必然结果）。要统计"真实评论总量"请不要直接用这张大表，按 `dataset` 分开算。
> 合并视图 `fraudsquad_amazon` 已排除在外，否则会重复第四次。

## 3. 字段定义（评论级）

| 列 | 类型 | 说明 |
|---|---|---|
| `dataset` | str | 数据集标识：`yelpchi` / `yelpnyc` / `yelpzip` / `fraudsquad_llama3` / `fraudsquad_qwen2` / `fraudsquad_qwen_dsr1` / `fraudsquad_amazon` / `yelpchi_injected` |
| `review_id` | str | 数据集内唯一评论 ID，形如 `yelpzip-000123456`（按原始行序，稳定可复现） |
| `global_user_id` | str | `{dataset}:{user_id}`，**跨数据集拼接时用这一列**，避免不同数据集用户 ID 撞号 |
| `user_id` | str | 数据集原始用户 ID（统一转字符串，防止 19 位大整数丢精度） |
| `item_id` | str | 数据集原始商品/商户 ID |
| `rating` | float32 | 1~5 星；缺失为 NaN（**不填 0，不填均值**） |
| `date` | datetime64 | 评论时间，**naive**（不带时区，按原始数据的本地时间存） |
| `label` | int8 | **1 = 虚假评论，0 = 真实评论**（全数据集统一方向，见 §4） |
| `label_source` | str | `yelp_filter` / `real_baseline` / `llm_generated` / `human_generated` / `injected` |
| `is_injected` | int8 | 1 = 我们 step3 人工注入的评论 |
| `text_available` | int8 | 1 = 该数据集原本就带文本；0 = 无文本字段（当前为 YelpChi 两版） |
| `text` | str | 评论文本；无文本时为空串（**不是 NaN**），判断有无请看 `text_available` |

`label_source` 让"这个标签是平台给的、模型生成的、还是我们造出来的"在数据层面可区分，
对比表里说明监督信号强度时直接引用它即可。

## 4. 标签方向的统一规则（**最容易出错的地方**）

首版把数字 `0` 一律当成"虚假"，遇到用 `0/1` 编码的数据版本会整表反向。现已改为**显式编码方案**：
`label_from_yelp(raw, scheme=...)`，可选值见 `src/data/schema.py: YELP_LABEL_SCHEMES`。

| 原始数据 | 原始编码 | 应当使用的 scheme | 统一后 `label` |
|---|---|---|---|
| YelpChi / YelpZip（官方原版） | `1` = 推荐（真实）、`-1` = 过滤（虚假） | `signed_1_real_minus1_fake`（`auto` 可自动识别） | `-1 → 1`，`1 → 0` |
| 0/1 再加工版本，1 = 虚假 | `1` = 虚假、`0` = 真实 | `binary_1_fake` | 原样 |
| 0/1 再加工版本，0 = 虚假 | `0` = 虚假、`1` = 真实 | `binary_0_fake` | `0 → 1`，`1 → 0` |
| FraudSquad 真实基线 | 无标签列 | — | 全部 `0`，`label_source=real_baseline` |
| FraudSquad 生成集 | 文件里 `label` 全为 1 | — | 全部 `1`，`label_source=llm_generated` |
| 我们注入的评论 | `injection_log_v2.json` 的 `events_added` | — | 全部 `1`，`label_source=injected` |

**为什么 `{0,1}` 必须显式指定**：`1=虚假` 与 `1=真实` 两种约定在公开版本里都出现过，
单看数据无法判断。`scheme="auto"` 只在 `{-1,1}` 时自动判定（这个方向的约定唯一），
遇到 `{0,1}` 直接抛 `SchemaError` 并提示两个候选方案，**绝不默认猜一个**。
显式指定后，若出现该方案定义域之外的取值同样报错，不会把未知标签悄悄当真实样本。

## 5. 数据集之间的口径差异（写论文时必须说明）

| 维度 | YelpChi | YelpNYC | YelpZip | FraudSquad ×3（单模型） | 注入集 |
|---|---|---|---|---|---|
| 对象 | 芝加哥酒店/餐厅 | 纽约市餐厅 | 纽约市及周边郊区餐厅 | Amazon 商品（5 个类目） | 芝加哥酒店/餐厅 |
| 评论数 | 67,395 | 359,052 | 608,458 | 86,758 + 2,489~2,500 | 70,186 |
| 商户/商品数 | 201 | 923 | 5,044 | 7,617 | 201 |
| 假评比例 | 13.23% | **10.27%** | 13.22% | 2.79% / 2.80% / 2.80% | 16.68% |
| spammer 账号占比 | 20.33% | **17.79%** | 23.91% | 1,250 / 8,092 = 15.45% | 300 / 38,063 = 0.79% |
| 标签来源 | 平台过滤器（near ground truth） | 同左 | 同左 | 构造即真值 | 注入脚本 |
| 账号结构 | 38,063 用户 | 160,225 用户 | 260,239 用户（官方 260,277） | 8,092 账号，其中 1,250 被接管 | 38,063 用户，300 被注入 |
| 时间跨度 | 2004-10 ~ 2012-10 | 2004-10 ~ 2015-01 | 2004-10 ~ 2015-01 | 2021-12 ~ 2023-01 | 2004-10 ~ 2012-10 |
| 文本 | 无 | 有 | 有 | 有 | 无 |

**推论**：跨数据集的绝对 AUC 不可直接横向比较（假评比例、领域、时间全不同），
只能在同一数据集内比较不同方法。对比表应写出"同一数据集、同一划分、同一指标"。

关于三张 FraudSquad 表的可比性，**不要笼统说"共用同一批账号"**（实测不是）：

| 比较 | 账号是否同一批 | 真实基线是否同一份 | 结论 |
|---|---|---|---|
| llama3 vs qwen2 | ✅ 完全相同（各 1,250，集合相等） | ✅ 同一份 86,758 | **严格配对比较**，可做配对检验 |
| llama3/qwen2 vs qwen_dsr1 | ❌ 另一批（交集仅 309，三者并集 2,191） | ✅ 同一份 86,758 | 同时换了账号与模型，**不能当配对比较** |

> 上表由 `results/data_quality_report.md` §5 的账号集合关系段实时算出，
> 换数据版本后请以报告为准。

### 5.1 YelpZip 的 140 行缺口（已查清）

| 项目 | 数量 | 说明 |
|---|---|---|
| 镜像文件数据行 | 608,598 | 与官方公布条数一致 |
| 其中带标签、可用 | 608,458 | 进入统一表 |
| 其中导出残缺、不可用 | 140 | 只有 `user/product/date/text` 四列，缺 rating 与 label |
| 用户并集 | 260,239 + 38 = 260,277 | 与官方公布用户数一致 |

不是解析丢行：这 140 行在导出时丢了 rating 与 label、分隔符变成制表符，并被前一条未闭合引号的
记录吞进正文（连带污染了 18 条正常评论的文本）。读取阶段已剥离（`readers.strip_partial_rows`），
残缺行因无标签不可用。账号级实验时这 38 个账号缺位（占 0.015%），做账号级结论需注明。

## 6. 账号级表（`data/processed/accounts/<dataset>.parquet`）

| 列 | 类型 | 说明 |
|---|---|---|
| `dataset` | str | 数据集标识 |
| `account_id` | str | 账号 ID（= 评论表的 `user_id`） |
| `global_account_id` | str | `{dataset}:{account_id}` |
| `n_reviews` | int32 | 该账号在本数据集内的评论总数 |
| `n_fake_reviews` | int32 | 其中被标为虚假的条数 |
| `n_injected` | int32 | 其中由我们注入脚本写入的条数（无注入时为 0） |
| `fake_ratio` | float32 | `n_fake_reviews / n_reviews` |
| `n_items` / `n_fake_items` | int32 | 涉及的商户数 / 虚假评论涉及的商户数 |
| `first_date` / `last_date` / `span_days` | datetime / float32 | 首末评论时间与跨度（天） |
| `account_label` | int8 | **1 = 水军/被接管账号，0 = 普通账号** |
| `label_basis` | str | 账号标签依据：`hijacked_by_llm` / `platform_filtered` / `injected` |

当前各数据集的正例口径与数量：

| 数据集 | 账号数 | 正例 | 口径 |
|---|---|---|---|
| yelpchi | 38,063 | 7,739（20.33%） | `platform_filtered`：有 ≥1 条被平台过滤的评论 |
| yelpzip | 260,239 | 62,214（23.91%） | 同上 |
| fraudsquad_llama3 / qwen2 / qwen_dsr1 | 各 8,092 | 各 1,250（15.45%） | `hijacked_by_llm`：写过该模型的生成评论 |
| yelpchi_injected | 38,063 | 300（0.79%） | `injected`：被注入脚本写过（用 `n_injected` 判） |

`platform_filtered` 取"至少 1 条"是最宽松口径；若要改用比例阈值，请直接用
`n_fake_reviews / fake_ratio` 重算，**不要复用 `account_label`**。

## 7. 四条使用纪律（血泪版）

1. **账号级标签不要下放到评论级**。FraudSquad 的 1,250 个被接管账号 **100%** 在真实基线里也有
   真实评论（中位数 10 条）。账号级问"这个号是不是被接管"，评论级问"这条评论是不是假的"，
   混用会把这些账号的全部历史评论误判为假。反过来，账号级实验是可以做的（见 §1）。
2. **不要用空文本跑文本模型**。YelpChi 与注入集 `text_available=0`，拿空串喂 TF-IDF 得到的分数
   没有意义，更不能拿去和别的数据集比。
3. **不要忽略"评分捷径"**。FraudSquad 的生成评论 100% 是 5 星，而真实评论 5 星占 69.6%。
   任何吃评分的模型都会虚高，必须把"仅用评分"作为对照基线一起报。
4. **不要拿合并视图比论文数字**。`fraudsquad_amazon` 里每个被接管账号攒到 4~6 条假评论
   （930 个账号 4 条、304 个账号 6 条），与原文"每账号 2 条"不一致。对比请用三张单模型表。

## 8. 读取方式

```python
import sys; sys.path.insert(0, "src")
from data import (load_dataset, load_all, load_accounts, list_datasets,
                  list_accounts, iter_reviews)

list_datasets()                                     # 数据集 + 规模 + role（paper/view/self_built）
df  = load_dataset("fraudsquad_llama3")             # 单集，按 date 升序
df  = load_dataset("yelpzip", with_text=False)      # 不要文本（省几个 GB 内存）
sub = load_all(["yelpchi", "yelpchi_injected"])     # 多集合并
acc = load_accounts("fraudsquad_llama3")            # 账号级表（8,092 行，1,250 正例）
for batch in iter_reviews("yelpzip", batch_size=50_000):   # 60 万行级别流式处理
    ...
```

配套校验：`python src/build_unified_data.py verify` 会比对 parquet 与 `manifest.json` 的
sha256 与行数，防止"本地改过数据但没更新记录"。
