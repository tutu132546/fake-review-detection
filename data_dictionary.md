# YelpChi 行为特征与注入数据集 - 数据字典

本数据字典适用于基于 YelpChi 真实评论数据（`metadata.txt`）构建的完整数据管线。彻底摒弃了图结构数据的错配问题，确保特征计算的物理严谨性。

## 一、 交付文件清单与生成链路
1. **`YelpChi_reviews.csv`**: 评论级原始数据（约 6.7 万条）。由 `step0` 生成，含真实时间戳与评分。
2. **`clean_accounts.parquet`**: 账号级干净底表。由 `step1` 生成，包含真实标签与所有行为特征。
3. **`yelpchi_1000_accounts.csv`**: 账号级基准样本（1000 条）。由 `step2` 生成，供基线训练。
4. **`augmented_accounts_v2.parquet`**: 注入水军后的全量底表。由 `step3` 生成。
5. **`injection_log_v3.json`**: 水军注入 Ground Truth 明细（含 `add_n` / `events_added` / `issue3_status`）。由 `step3` 生成。
6. **`scaler_params.json`**: YelpChi 全量特征的 Z-score 缩放参数（clean 底表统计）。由 `step1` 生成；多数据集分别见 `features/scaler_<dataset_id>.json`。
7. **`features/feature_table_yelpchi.csv`**: 建模用账号级特征表（38,063 行 × 22 列）。由 `src/features/build_feature_table.py --dataset yelpchi` 生成；`data/processed/feature_table.csv` 为兼容旧下游的共享副本。
8. **`results/feature_manifests/feature_table_yelpchi.json`**: sidecar manifest（数据文件 sha256、commit、标签分布、缺失率、自检结果）。由同一条命令生成。

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

### 3. 多数据集字段可用性矩阵（骨架，B5）

| 数据集 | user_id | item_id | rating | date | text | label | 缺失率 |
|---|---|---|---|---|---|---|---|
| YelpChi（account_base） | 可用 | 可用（聚合前） | 可用 | 可用 | **不适用**（`text_available=0`，metadata 无正文） | 可用（评论级 -1/1 → 账号级 0/1） | 见 manifest |
| YelpZip（unified） | 可用 | 可用 | 可用 | 可用 | 可用 | 可用（`fake_raw_values=(-1)`） | TBD（待表） |
| FraudSquad-Llama3 | 可用 | 可用 | 可用 | 可用 | 可用 | 可用（需 `label_source`） | TBD（待表） |
| FraudSquad-Qwen2 | 可用 | 可用 | 可用 | 可用 | 可用 | 可用（需 `label_source`） | TBD（待表） |
| FraudSquad-Qwen-DSR1 | 可用 | 可用 | 可用 | 可用 | 可用 | 可用（需 `label_source`） | TBD（待表） |
| YelpNYC | 不适用 | 不适用 | 不适用 | 不适用 | 可用 | 不适用 | MISSING（官方数据不可得） |

- 每格"缺失率"由构建时写入的 `results/feature_manifests/feature_table_<id>.json` 的 `missing_rates` 回填；标 `TBD` 表示该表尚未产出，禁止手填估算值。
- `text` 列指评论文本正文；`text_available=0` 的数据集必须屏蔽文本/图文本方法。
- 标签方向：YelpZip 已声明 `fake=-1 / real=1`；FraudSquad×3 未声明，manifest 标 `label_direction_unverified`，需上游确认原始 `label` 语义后才能启用方向校验。


## 三、 水军注入实验说明 (物理行为模拟)

*   **注入对象**：从 `clean_accounts.parquet` 中抽取 `n_reviews >= 5` 的正常账号 300 个。
*   **注入原理**：模拟水军突击新增 5~15 条极端评价的行为。
*   **特征联动更新**：
    *   评论数 `n_reviews` 对应增加；单日最大发评 `max_daily_reviews` 在模板 A/C 下由合并时间序列直接重算（实测 200/200 变化），模板 B 因刻意不改时间戳，若原有值不足以容纳新增评论则按 `ceil(new_n / active_days)` 补齐物理约束（Issue #6，实测 2 个账号）。
    *   活跃期 `active_days` 一般不变，仅当新增评论落在原活跃期外时扩展（实测 2 个账号，均在模板 C）。
    *   极端评分比例 `extreme_ratio` 按 `(旧极端数 + 新增数) / 新总数` 精确重新计算。
    *   爆发性 `burstiness`：**仅在提供 review 级时间戳（`YelpChi_reviews.csv`）时**，为新增评论构造日期并合并新旧时间序列后用 `calc_burstiness` 重算（物理上集中爆发会混入近 0 间隔、波动增大，应升高）；原本不可定义（NaN）的账号保持 NaN。缺少 review 级数据时逐条标记 `issue3_status=blocked_by_review_level_data`，保留原实现值但**不作方向性结论**，禁止用乘系数冒充修复（见 §9 注）。
*   **防漏机制**：所有修改后的特征，均同步通过 `scaler_params.json` 重新计算对应的 `_z` 列，确保数据分布完全自洽。

## 四、 注入日志 (`injection_log_v3.json`)
*   `user_id`: 被注入的账号 ID。
*   `injection_template`: 使用的模板名称（A_Time_Burst / B_Rating_Polarization / C_Stealth_Mix）。
*   `add_n`: 本次新增评论数（300 条记录合计 2,910，与 aug 表 `n_reviews` 增量一致）。
*   `events_added`: 新增评论逐条落账（`product_id` / `rating` / `date`）；提供 review 级数据时 300 条记录合计 2,910（与 `add_n` 合计一致）。模板 B 的事件仅作台账，不参与时间特征重算（B 为评分侧模板）；无 review 级时间戳时为 `[]`（不伪造）。
*   `issue3_status`: `recomputed`（按真实时间戳重算）或 `blocked_by_review_level_data`（缺数据，方向待验证）。
*   `modified_features`: 字典结构，记录特征变化（如 `max_daily_reviews: {original: 2, mutated: 12}`）。用于后续验证 DCFO 的因果解释。**已修复为完整记录**：实际被修改的特征全部入账（含 n_reviews 与 daily_avg_reviews），日志数值与 aug 表核对 1502/1502 项一致（4 位小数精度）。

# 五、特征交接说明（B 成员交付，面向建模）

## 1. 交接文件
`data/processed/features/feature_table_yelpchi.csv`（38,063 行 × 22 列），生成命令：
`python src/features/build_feature_table.py --dataset yelpchi`（或 `python -m src.features.build_feature_table --dataset yelpchi`）。
同一条命令同时产出 `features/scaler_yelpchi.json`、`results/feature_manifests/feature_table_yelpchi.json` 与 `results/feature_qa_yelpchi.md`；`data/processed/feature_table.csv` 为兼容旧下游的共享副本。
多数据集：`python src/features/build_all.py`（YelpZip / FraudSquad×3 缺数据时写 `blocked_by_data`，YelpNYC 写 `MISSING` 占位，均不造假表）。
该脚本是特征口径的**唯一出口**：burstiness 填充、缺失标记、每数据集独立 `_z` 重算、标签守卫（label 必须 ∈{0,1}）都在此完成，下游（LOF / iForest / DCFO）一律以本表为准。

## 2. 列结构与使用纪律
| 列组 | 列 | 用途 |
|---|---|---|
| 标识/评估列 | `dataset`, `global_user_id`, `user_id`, `account_is_fake`, `cohort`, `is_injected`, `text_available` | 仅用于评估与实验分组，**严禁进入无监督模型输入** |
| raw 特征列（7） | 7 个原始量纲特征 | 供 DCFO 反事实解释回译为原始单位（如"单日发评降到 3 条以下"） |
| `_z` 列（7） | 7 个标准化特征 | **无监督模型输入** |
| 缺失标记列 | `burstiness_missing` | 86.5% 账号的 burstiness 为中位数填充值，此标记区分"真实值/填充值"，**进入模型输入** |

`global_user_id = <dataset_id>:<user_id>`，用于 5 张表合并时消歧；`cohort` 取值 `native` / `self_injected`（YelpChi 自注入）/ `hijacked_llm`（FraudSquad）；`text_available` 为 0 时该数据集必须屏蔽文本方法。

**约定：C 的模型输入 = 7 个 `_z` 列 + `burstiness_missing`，共 8 列**（`burstiness_missing` 属于模型输入，不属评估列，与上表一致）。

## 3. 特征定义与水军暴露逻辑
| 特征 | 公式 | 单位/范围 | 暴露逻辑 |
|---|---|---|---|
| n_reviews | 评论总数 | 条 | 刷评直接推高总量 |
| active_days | 首末评论间隔天数 +1 | 天 | 突击型账号活跃期短 |
| daily_avg_reviews | n_reviews / active_days | 条/天 | 突击刷评推高日均 |
| max_daily_reviews | 单日最大发评数 | 条 | 定时任务式刷评单日爆发 |
| burstiness | 相邻评论间隔的变异系数 (std/mean) | 无量纲，≥0 | 集中爆发会在间隔序列混入大量近 0 间隔，波动增大，取值应**偏高**；评论数<3 或全部同日发出时无法定义（NaN）。有 review 级时间戳时按合并时间序列重算；缺数据时该列方向待验（见 §9 注） |
| rating_deviation | mean(评分 − 商家均分) | 星，[-4,4] | 集中打极端分，偏离商家均分 |
| extreme_ratio | (1星或5星评论数) / 总数 | [0,1] | 刷评多为极端评分 |

## 4. 标准化口径
`_z = (x − mean) / std`，**每个数据集单独 fit**（`features/scaler_<dataset_id>.json`），禁止跨数据集联合 fit；`std < 1e-12` 时置 1.0。
YelpChi 的拟合人群是注入前 `clean_accounts.parquet` 全体账号（含原生真假、不按标签筛），故与旧 `scaler_params.json` 逐项一致（实测 7/7 无差异，已算）；其他数据集用该份全部账号。注入行/被接管行的 `_z` 用同一组参数换算，不参与 fit。
**已声明局限**（声明而非修复）：`rating_deviation` 的商家均分在含假评的全量评论上计算，存在轻微混叠；无监督场景可接受，汇报时如实说明。

## 5. burstiness 缺失处理
NaN 共 32,918/38,063（86.5%）：其中 32,666 个为评论数 < 3（数据不足），252 个为评论数 ≥ 3 但全部评论同日发出（间隔均值为 0）。
（口径修正：此前误记为 3,252；按 step1 实现 `(max-min).days+1`，同日账号 active_days=1、n≥3，NaN 可完全由"间隔均值为 0"解释，实测 252，已算。）
策略：以 **clean 底表观测值中位数 0.9436** 填充（与 scaler 参数同源、不受注入分布污染），并保留 `burstiness_missing` 标记列。
建议 C 做敏感性分析：带/不带 `burstiness_missing` 各跑一遍，AUC 差异 < 1pct 即认为填充无副作用。
注入行仅 2/300 为 NaN（注入池要求 n_reviews ≥ 5；按合并时间序列重算后 NaN 数量不变，原本不可定义者保持 NaN）。

## 6. 组间比较陷阱（重要）
注入组 `active_days_z` 组均值偏高（+2.24）是**选池偏差**（注入池只选 n_reviews ≥ 5 的正常号），不是注入效应——active_days 基本未动（300 个账号中仅 2 个因新增评论落在原活跃期外而扩展，均值 dz 在 0.001 量级）。
评估注入效应必须用**配对差值 dz**（同一账号注入前后 `_z` 变化）。修复后实测 dz 基准（2026-11-03，300 账号，`issue3_status=recomputed`）：
n_reviews +4.61（A +4.68 / B +4.65 / C +4.52）｜ max_daily_reviews +10.01（A +15.37 / C +14.43；B 仅 2 个账号补齐约束，均值 +0.22）｜ burstiness +1.33（A +1.90 / C +2.08；Issue #3 已按真实时间序列重算，方向转为正向）｜ rating_deviation +0.61（B +0.86 / C +0.97）｜ extreme_ratio +0.77 ｜ daily_avg_reviews +0.29 ｜ active_days 0.00（仅 2 个 C 模板账号扩展）。

**选池偏差对评估的三条纪律**（同样适用于 AUC，不仅是组均值）：
1. 论文主结论只用原生标签（account_is_fake）结果；注入组仅用于验证解释方法（DCFO）。
2. 注入号选自 n_reviews≥5 的更活跃正常号，含注入的 AUC 天然偏高，不得作为检测性能证据。
3. 需要与注入组对比时，对照组必须用同条件（n_reviews≥5）的未注入正常号，而非全体正常号。

## 7. 特征 ↔ 注入模板对照（DCFO 验证的 ground truth）
| 模板 | 数量 | 实际修改的特征 |
|---|---|---|
| A_Time_Burst | 100 | n_reviews, max_daily_reviews, burstiness, extreme_ratio, daily_avg_reviews |
| B_Rating_Polarization | 100 | n_reviews, rating_deviation, extreme_ratio, daily_avg_reviews；其中 2 个账号按 Issue #6 补齐 max_daily_reviews |
| C_Stealth_Mix | 100 | n_reviews, max_daily_reviews, burstiness, rating_deviation, extreme_ratio, daily_avg_reviews |

注：个别账号"被修改但数值不变"也如实记录（如 4 个原本全 5 星账号的 extreme_ratio 仍为 1.0、1 个全同日账号的 burstiness 仍为 NaN），日志 original=mutated 属预期。

## 8. 单特征区分度（ROC-AUC vs account_is_fake）
| 特征 | 全表（含注入） | 仅原生（YelpChi 真实标签） |
|---|---|---|
| n_reviews_z | 0.414 | 0.391 |
| active_days_z | 0.415 | 0.396 |
| daily_avg_reviews_z | 0.579 | 0.593 |
| max_daily_reviews_z | 0.489 | 0.473 |
| burstiness_z | 0.517 | 0.504 |
| rating_deviation_z | 0.506 | 0.499 |
| extreme_ratio_z | 0.573 | 0.571 |
| burstiness_missing | 0.546 | 0.564 |

解读：单特征区分度普遍偏弱（0.39–0.59）。原生水军呈"**少评、短活、高日均、极端评分**"画像（n_reviews / active_days 为低值方向，daily_avg_reviews / extreme_ratio 为高值方向）——这正是需要多变量无监督方法（LOF/iForest）与多信号融合的依据，而非单特征阈值。注入会推高 n_reviews（隐蔽性设计），故全表与原生方向一致但数值不同。

## 9. EDA 图件与方向性判定（复现命令：`python src/viz/feature_eda.py`）
| 图件 | 内容 |
|---|---|
| `results/feature_eda_group_box.png` | 原生正常 / 原生水军 / 注入水军 三组 7 特征 `_z` 箱线图（burstiness 按未填充值绘制，NaN 不参与；箱体隐藏离群点） |
| `results/feature_eda_template_dz.png` | 注入前后配对 dz 的模板级均值对比（A/B/C 三色柱状图），消除选池偏差 |

可证伪判定 13 项全部 PASS（2026-11-03 实测，`issue3_status=recomputed`）：
- 全模板公共效应：n_reviews dz>0（300/300，均值 +4.61）、daily_avg_reviews dz>0（300/300，均值 +0.29，修复前恒为 0）、extreme_ratio dz>=0（296/300 上升，其余 4 个本就全 5 星，数值不变）；active_days 基本未动（均值 dz=0.000，仅 2/200 个 A/C 账号扩展活跃期，判定为 dz>=0）。
- 模板特异性：A/C 的 max_daily_reviews dz>0（+15.37 / +14.43，200/200 变化）；A/C 的 burstiness dz>0（+1.90 / +2.08，199/200 重算，1 个原本全同日账号保持 NaN）；B/C 的 rating_deviation dz>0（+0.86 / +0.97）。
- Issue #6 修复校验：模板 B 的 max_daily_reviews 由"恒为 0"改为"约束校验"——dz>=0 且 dz>0 的账号数（2/100）与日志补齐数一致，全表 `n_reviews <= active_days * max_daily_reviews` 违规 0 行（已算）。
- Issue #3 修复校验（已 PASS）：`issue3_status=recomputed`，`events_added` 合计 2,910 与 `add_n` 合计一致；模板 A/C 的 burstiness dz 由修复前的 −1.44 / −1.59 转为 **+1.90 / +2.08**，方向与"集中爆发→间隔波动增大"的物理预期一致（[已算]，300 账号配对 dz）。
- 未动特征恒为 0：B 的 burstiness、A 的 rating_deviation。
读图提示：图 1 中"原生水军"的 daily_avg_reviews 箱体退化为一条线（约 z=0.6），是因为大量原生假号只有 1 条评论（1/1 天），取值高度集中——这本身是 YelpChi 假号"少评短活"画像的证据。