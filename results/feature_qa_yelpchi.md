# 特征表 QA：yelpchi

- 数据版本：feature_version=1.0.0，built_at=2026-11-03T13:17:33+08:00，commit=9fa966489f5aa0131e105a92d140322b88ecee40
- 行数：38063；账号数：38063；标签分布：{'0': 30024, '1': 8039}；人群分布：{'native': 37763, 'self_injected': 300}
- 拟合人群：clean_native；scaler：data/processed/features/scaler_yelpchi.json
- 文本可用：0（评估时据此屏蔽文本方法）
- 标签编码检查：passed

## 缺失率

| 项 | 比例 |
|---|---:|
| n_reviews | 0.00% |
| active_days | 0.00% |
| daily_avg_reviews | 0.00% |
| max_daily_reviews | 0.00% |
| burstiness | 86.48% |
| rating_deviation | 0.00% |
| extreme_ratio | 0.00% |

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
| rows | 38063 |

## 已知问题

- 无

## 数据来源（sha256 可追溯）

- `C:\Users\31039\Desktop\work\挖掘机\fake-review-detection\data\processed\augmented_accounts_v2.parquet`（rows=38063, sha256=2cea17607a32cc19833d81b14e2ba9b6dce21e008b6575b1fa52fce9f935553d）
- `C:\Users\31039\Desktop\work\挖掘机\fake-review-detection\data\processed\clean_accounts.parquet`（rows=38063, sha256=1f3da4b46dfb62d684a792201743c2218d71ed32c9f1a6d3bd32d11dfa60fb0d）
- `C:\Users\31039\Desktop\work\挖掘机\fake-review-detection\data\processed\injection_log_v3.json`（rows=300, sha256=8cf5cdefaaca993d356d6c46cbaa4d342bd3ffff35eeab6d8233b9c296061369）
