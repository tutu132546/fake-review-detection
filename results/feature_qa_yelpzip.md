# 特征表 QA：yelpzip

- 数据版本：feature_version=1.0.0，built_at=2026-11-03T19:01:57+08:00，commit=3c82a2bda77290e3647e27eaf31e67c0143b3253
- 行数：260239；账号数：260239；标签分布：{'0': 198025, '1': 62214}；人群分布：{'native': 260239}
- 拟合人群：all_accounts；scaler：data/processed/features/scaler_yelpzip.json
- 文本可用：1（评估时据此屏蔽文本方法）
- 标签编码检查：passed

## 缺失率

| 项 | 比例 |
|---|---:|
| n_reviews | 0.00% |
| active_days | 0.00% |
| daily_avg_reviews | 0.00% |
| max_daily_reviews | 0.00% |
| burstiness | 82.21% |
| rating_deviation | 0.00% |
| extreme_ratio | 0.00% |
| review_rating | 0.00% |

## 自检结果

| 检查 | 结果 |
|---|---|
| schema_violations | 0 |
| constraint_violations | 0 |
| nan_feature_cells | 0 |
| daily_avg_max_abs_diff | 0.0 |
| z_mean_max_abs | 0.0 |
| z_std_max_dev | 0.0 |
| degenerate_features | [] |
| rows | 260239 |

## 已知问题

- 无

## 数据来源（sha256 可追溯）

- `C:\Users\31039\Desktop\work\挖掘机\fake-review-detection\data\processed\unified\yelpzip.parquet`（rows=608458, sha256=3dc80622c697bfd33c3e4ef8865ef5341a0256b310388b153d99f14c785d31a3）
