# 特征表契约（feature contract）

> 适用：`src/features/build_feature_table.py` 与 `src/features/build_all.py` 产出的所有
> `feature_table_<dataset>.csv`。本文是建模侧与数据侧之间的接口约定（A9）。
> 单一真源：`src/features/dataset_config.py`（列名 / 顺序）与 `src/features/schema.py`（类型校验）。

## 1. 输出列（22 列，顺序固定，不得按数据集增减）

| 段 | 列 | 类型 | 说明 |
|---|---|---|---|
| 标识 | `dataset` | str | 数据集 id（如 `yelpchi`）；统一层来源另见 manifest 的 `source_dataset` |
| 标识 | `global_user_id` | str | `{dataset}:{user_id}` |
| 标识 | `user_id` | str | 原始账号 id，读回必须是字符串（防 19 位整数丢精度） |
| 标签 | `account_is_fake` | int8 | 1 = 水军账号，0 = 正常账号；**只用于评估，禁止进特征** |
| 人群 | `cohort` | str | `native` / `self_injected` / `hijacked_llm`；由标签与注入列聚合，**禁止进特征** |
| 注入 | `is_injected` | int8 | 1 = 我们 step3 注入的账号；来自日志/统一层 `is_injected` 列，**禁止进特征** |
| 文本 | `text_available` | int8 | 该数据集是否自带文本；**禁止进特征** |
| 缺失 | `burstiness_missing` | int8 | burstiness 因评论数不足无法计算（已按拟合人群中位数填充） |
| 原始 | `n_reviews`, `active_days`, `daily_avg_reviews`, `max_daily_reviews`, `burstiness`, `rating_deviation`, `extreme_ratio` | int/float | 7 个行为特征原值（burstiness 为填充后值） |
| 标准化 | 上述 7 列的 `<col>_z` | float | 每数据集独立 scaler 的 Z 值 |

## 2. 模型输入白名单（8 列）

`MODEL_INPUT_COLUMNS = [n_reviews_z, active_days_z, daily_avg_reviews_z, max_daily_reviews_z,
burstiness_z, rating_deviation_z, extreme_ratio_z, burstiness_missing]`

- 建模代码只允许读这 8 列；新增特征必须改 `dataset_config.FEATURE_LIST` 并重跑全部数据集，
  不允许在建模脚本里临时拼列。
- `burstiness_missing` 是**缺失标记**，不是特征值本身；`native_ge3_observed` 人群要求它为 0。

## 3. 禁入列（硬性）

`account_is_fake` / `cohort` / `is_injected` / `text_available` / `dataset` / `global_user_id` /
`user_id`，以及任何由标签派生的字段。

原因：这些列直接或间接编码 ground truth（尤其 `is_injected` 与 `cohort`），一旦进入特征
就是标签泄漏；`user_id` 进入特征会让模型记住账号而非行为。

## 4. 统一评论级 12 列 → 账号级 22 列映射

输入（`data/processed/unified/<dataset>.parquet`）：`dataset, review_id, global_user_id, user_id,
item_id, rating, date, label, label_source, is_injected, text_available, text`。

| 账号级输出 | 由评论级如何聚合 |
|---|---|
| `account_is_fake` | `groupby(user_id).label.max()`（发过一条假评即账号为假） |
| `is_injected` | `groupby(user_id).is_injected.max()`（A5；A6 起 rating 缺失不丢行） |
| `cohort` | 注入账号 → `self_injected`；否则 `label==1` → 数据集人群名（FraudSquad=`hijacked_llm`），其余 `native` |
| `n_reviews` / `active_days` / `daily_avg_reviews` / `max_daily_reviews` | 评论计数与日期跨度 |
| `burstiness` | 评论间隔的 burstiness；不足 3 条或全同日 → NaN → `burstiness_missing=1` |
| `rating_deviation` / `extreme_ratio` | 相对商品均值的评分偏差 / 1 星或 5 星占比 |
| `*_z` | 用该数据集拟合人群的 scaler 变换；**unified 路径的拟合人群排除注入行**（A5） |

## 5. `expected_rows` 双层语义（同名不同义，勿混用）

| 位置 | 值 | 语义 |
|---|---|---|
| `src/data/schema.py` 的 `DatasetSpec.expected_rows` | 67,395 | **评论级**行数（统一表） |
| `src/features/dataset_config.py` 的 `DatasetSpec.expected_rows` | 38,063 | **账号级**行数（特征表） |

YelpChi 的账号表与评论表由不同链路产出（`clean_accounts.parquet` vs `unified/yelpchi.parquet`），
两个数字都正确，只是统计对象不同。新增数据集时两处都要填，且必须写明是账号行还是评论行。

## 6. `label_source` 五值枚举（与 `src/data/schema.py` 对齐）

`yelp_filter`（平台过滤器）/ `real_baseline`（真实基线）/ `llm_generated`（模型生成）/
`human_generated`（人工生成）/ `injected`（本项目 step3 注入）。

入口守卫：`adapters._assert_label_encoding` 拒绝枚举外的值与 `label ∉ {0,1}`。
构造类数据集（FraudSquad）必须显式提供 `label_source`。

## 7. 注入事件的唯一真源

`data/processed/injection_log_v2.json`（A2）。必须同时满足三件套（A3）：

1. 每条记录含 `policy`；
2. `events_added` 非空，且 `len(events_added) == policy.n_add`；
3. `physical_checks == "passed"`。

`injection_log_v3.json` 仅作历史归档，任何代码不得读取。`build_injected()` 只读 v2；
`yelpchi_injected` 行数守卫为 **70,186 ± 2%**（`src/data/pipeline.py`，A8）。

## 8. Manifest 溯源字段

`results/feature_manifests/feature_table_<dataset>.json` 必须包含：

- `source_files`：输入文件路径 + sha256（YelpChi 含 `injection_log_v2.json`）；
- `source_dataset`：统一层来源（`yelpchi` → `yelpchi_injected`）；
- `injection_log`：注入日志的路径 / sha256 / 行数（仅 account_base 路径）；
- `missing_rates`：7 个特征缺失率；unified 路径额外含 `review_rating`（A6）；
- `label_encoding_check` / `known_issues` / `qa` 三项自检结果。

## 9. 变更流程

改列名、改顺序、改枚举或改聚合口径，必须同时：

1. 更新 `dataset_config.OUTPUT_COLUMNS` / `schema` 校验；
2. 更新本文件与 `data_dictionary.md`；
3. 重跑 `build_all` 并附新旧 manifest diff；
4. 跑 `python -m unittest discover -s tests -v`，回归测试不得出现"假绿"（缺数据必须显式 skip）。
