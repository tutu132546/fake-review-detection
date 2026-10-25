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

## 三、 水军注入实验说明 (物理行为模拟)
*   **注入对象**：从 `clean_accounts.parquet` 中抽取 `n_reviews >= 5` 的正常账号 300 个。
*   **注入原理**：模拟水军突击新增 5~15 条极端评价的行为。
*   **特征联动更新**：
    *   评论数 `n_reviews`、单日最大发评 `max_daily_reviews` 对应增加。
    *   极端评分比例 `extreme_ratio` 按 `(旧极端数 + 新增数) / 新总数` 精确重新计算。
    *   爆发性 `burstiness`：物理上集中爆发会混入近 0 间隔、波动增大，应升高；当前 step3 实现为降低，方向待修（见 §9 注）。
*   **防漏机制**：所有修改后的特征，均同步通过 `scaler_params.json` 重新计算对应的 `_z` 列，确保数据分布完全自洽。

## 四、 注入日志 (`injection_log_v2.json`)
*   `user_id`: 被注入的账号 ID。
*   `injection_template`: 使用的模板名称（A_Time_Burst / B_Rating_Polarization / C_Stealth_Mix）。
*   `modified_features`: 字典结构，记录特征变化（如 `max_daily_reviews: {original: 2, mutated: 12}`）。用于后续验证 DCFO 的因果解释。**已修复为完整记录**：实际被修改的特征全部入账（含 n_reviews 与 daily_avg_reviews），日志数值与 aug 表核对一致率 1500/1500。

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
评估注入效应必须用**配对差值 dz**（同一账号注入前后 `_z` 变化）。修复后实测 dz 基准：
n_reviews +4.61 ｜ max_daily_reviews +14.09（模板 A/C）｜ burstiness −1.01（A/C，实现产物，物理方向应为正，见 §9 注）｜ rating_deviation +0.61（B/C）｜ extreme_ratio +0.77 ｜ daily_avg_reviews +0.29 ｜ active_days 0（未动）。

**选池偏差对评估的三条纪律**（同样适用于 AUC，不仅是组均值）：
1. 论文主结论只用原生标签（account_is_fake）结果；注入组仅用于验证解释方法（DCFO）。
2. 注入号选自 n_reviews≥5 的更活跃正常号，含注入的 AUC 天然偏高，不得作为检测性能证据。
3. 需要与注入组对比时，对照组必须用同条件（n_reviews≥5）的未注入正常号，而非全体正常号。

## 7. 特征 ↔ 注入模板对照（DCFO 验证的 ground truth）
| 模板 | 数量 | 实际修改的特征 |
|---|---|---|
| A_Time_Burst | 100 | n_reviews, max_daily_reviews, burstiness, extreme_ratio, daily_avg_reviews |
| B_Rating_Polarization | 100 | n_reviews, rating_deviation, extreme_ratio, daily_avg_reviews |
| C_Stealth_Mix | 100 | n_reviews, max_daily_reviews, burstiness, rating_deviation, extreme_ratio, daily_avg_reviews |

注：个别账号"被修改但数值不变"也如实记录（如 4 个原本全 5 星账号的 extreme_ratio 仍为 1.0、1 个全同日账号的 burstiness 仍为 NaN），日志 original=mutated 属预期。

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

可证伪判定 11 项全部 PASS（2026-10-22 实测）+ 1 项描述性观测：
- 全模板公共效应：n_reviews dz>0（300/300，均值 +4.61）、daily_avg_reviews dz>0（300/300，均值 +0.29，修复前恒为 0）、extreme_ratio dz>=0（296/300 上升，其余 4 个本就全 5 星，数值不变）；active_days dz=0（未动）。
- 模板特异性：A/C 的 max_daily_reviews dz>0（+21.5 / +20.8）；B/C 的 rating_deviation dz>0（+0.86 / +0.97）。
- burstiness 方向 [待测]（即第三节与 §3"见 §9 注"所指）：物理上集中爆发会混入近 0 间隔、波动增大，dz 应为正；当前 step3 实现为 dz<0（−1.44 / −1.59），源于"间隔规律化"的错误假设，仅证明"代码按写的改了"。修复方案（构造新增评论时间戳后用 calc_burstiness 重算）已立项，修复前该项降级为描述性观测，不作为方向性 PASS 依据。
- 未动特征恒为 0：B 的 max_daily_reviews 与 burstiness、A 的 rating_deviation。
读图提示：图 1 中"原生水军"的 daily_avg_reviews 箱体退化为一条线（约 z=0.6），是因为大量原生假号只有 1 条评论（1/1 天），取值高度集中——这本身是 YelpChi 假号"少评短活"画像的证据。

## 10. D 模块产出：图算法社区发现
*   **`data/processed/graph_scores.csv`**：由 `src/models/graph_community_detection.py` 生成，包含图算法对全量账号的社区划分与打分。
    *   `user_id`: 账号唯一标识。
    *   `group_id`: Louvain 图算法划分的社区/团伙 ID。
    *   `graph_score`: 账号级图可疑分数，范围 `[0, 1]`。
        *   **计算口径**：`(该账号在团伙内的度数 / 团伙总人数)`，随后进行全局最大值归一化。
        *   **解读**：分数越高，说明该账号在团伙内的连接越密集，越可能是“核心节点”。经实测核对前 10 名高分账号，最高分（1.0）出自 392 人的大团伙核心，印证了公式在极度抱团的大团伙中依然能准确识别出核心。单人孤立团伙分数为 0。