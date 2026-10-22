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
    *   爆发性 `burstiness` 因时间集中而合理降低。
*   **防漏机制**：所有修改后的特征，均同步通过 `scaler_params.json` 重新计算对应的 `_z` 列，确保数据分布完全自洽。

## 四、 注入日志 (`injection_log_v2.json`)
*   `user_id`: 被注入的账号 ID。
*   `injection_template`: 使用的模板名称（A_Time_Burst / B_Rating_Polarization / C_Stealth_Mix）。
*   `modified_features`: 字典结构，记录特征变化（如 `max_daily_reviews: {original: 2, mutated: 12}`）。用于后续验证 DCFO 的因果解释。