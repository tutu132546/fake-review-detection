# YelpChi 行为特征与注入数据集 - 数据字典

本数据字典适用于基于 YelpChi 原始评论数据（`metadata.txt`）重构的完整数据管线。所有数据产物由 `src/` 目录下的三个核心脚本生成。

## 一、 交付文件清单
1. **`YelpChi_reviews.csv`**: 原始评论级数据（约 6.7 万条）。含真实的时间戳与评分，供队友计算底层行为特征。
2. **`yelpchi_1000_accounts.csv`**: 账号级基准样本（1000 条），供前期模型训练。
3. **`clean_accounts.parquet`**: 账号级干净底表（全量），包含真实标签与所有计算出的行为特征。
4. **`augmented_accounts_v2.parquet`**: 注入水军后的全量底表。
5. **`injection_log_v2.json`**: 水军注入 Ground Truth 明细。
6. **`scaler_params.json`**: 全量特征的 Z-score 缩放参数。

## 二、 字段说明 (以 `clean_accounts.parquet` 为准)
### 1. 基础字段
*   `user_id`: 账号唯一标识。
*   `n_reviews`: 该账号的真实评论总数。
*   `account_is_fake`: 账号级标签（**1 = 水军**，**0 = 正常**。只要该用户发过一条 `label=-1` 的评论，即标为 1）。
*   `is_injected`: 仅在注入表中存在（1 = 人工注入，0 = 原生）。

### 2. 行为特征字段
*   `active_days`: 注册活跃天数（最后一次评论与第一次评论的时间差）。
*   `daily_avg_reviews`: 日均发评数（总评论数 / 活跃天数）。
*   `max_daily_reviews`: 单日最大发评数（捕捉突发性刷评）。
*   `burstiness`: 爆发性（相邻评论时间间隔的变异系数，越低代表发帖越集中）。
*   `rating_deviation`: 评分偏离度（该用户的打分减去该商家的全局平均分）。
*   `extreme_ratio`: 极端评分比例（该用户打出 1 星或 5 星的比例）。
*   `[特征名]_z`: 上述特征对应的全局 Z-score 标准化值（均值为0，标准差为1）。

## 三、 水军注入实验说明
*   **注入对象**：从 `clean_accounts.parquet` 的 0 号（正常）样本中随机抽取 300 个账号。
*   **模板分离设计**：
    *   **模板 A (Time_Burst)**：修改 `max_daily_reviews`（增加）和 `burstiness`（降低），模拟爆发式刷评。
    *   **模板 B (Rating_Polarization)**：修改 `rating_deviation`（增加）和 `extreme_ratio`（增加），模拟恶意差评或刷好评。
    *   **模板 C (Stealth_Mix)**：小幅修改时间与评分特征，模拟高隐蔽性水军。
*   **物理防漏机制**：注入值严格受限于物理现实（如单日最大发评绝不超出总发评数）。
*   **数据防漏机制**：`_z` 列同步复用 `scaler_params.json` 中的参数更新，绝对不产生分布泄漏。

## 四、 注入日志 (`injection_log_v2.json`)
*   `user_id`: 被注入的账号 ID。
*   `injection_template`: 使用的模板名称（A/B/C）。
*   `modified_features`: 字典结构，详细记录被篡改特征的 `original`（原值）与 `mutated`（修改后值）。用于后续验证反事实解释（DCFO）的准确性。