# 行数三段审计报告

**状态：BLOCKED** —— 未找到可审计文件。

- 期望原始文件：data/raw/yelpzip.csv
- 期望统一文件：C:\Users\31039\Desktop\work\挖掘机\fake-review-detection\data\processed\unified\yelpzip.parquet
- 需要：YelpZip 原始文件 + 官方行数口径（责任人：叶歆澜）
- 脚本自测：`python src/features/audit_yelpzip_rows.py --raw data/raw/YelpChi_reviews.csv --official 67395`
