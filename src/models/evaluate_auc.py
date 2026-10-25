import pandas as pd
from sklearn.metrics import roc_auc_score

print("正在加载数据...")
# 读取真实标签
labels_df = pd.read_parquet('data/processed/clean_accounts.parquet')[['user_id', 'account_is_fake', 'n_reviews']]
# 读取你的图分数
graph_df = pd.read_csv('data/processed/graph_scores.csv')

# 合并两张表
merged = pd.merge(labels_df, graph_df, on='user_id', how='inner')

# 1. 总体 AUC
auc_all = roc_auc_score(merged['account_is_fake'], merged['graph_score'])
print(f"=== 总体 AUC: {auc_all:.4f} ===")

# 2. 评论数 >= 3 的活跃用户 AUC
merged_3 = merged[merged['n_reviews'] >= 3]
if len(merged_3) > 0:
    auc_3 = roc_auc_score(merged_3['account_is_fake'], merged_3['graph_score'])
    print(f"=== 评论数 >= 3 的人群 AUC: {auc_3:.4f} ===")

# 3. 被图算法抓出来的人里，有多少是真的？
caught = merged[merged['graph_score'] > 0]
print(f"\n全量账号的真实水军基准占比: {merged['account_is_fake'].mean() * 100:.2f}%")
print(f"被图算法抓出的 {len(caught)} 个团伙成员中，真实水军占比: {caught['account_is_fake'].mean() * 100:.2f}%")