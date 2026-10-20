import pandas as pd
import numpy as np
from config import ACCOUNT_PARQUET, SAMPLE_CSV

def main():
    print(f"读取 {ACCOUNT_PARQUET}")
    df = pd.read_parquet(ACCOUNT_PARQUET)

    # 1. 分箱
    df["review_bin"] = pd.qcut(df["n_reviews"], q=5, duplicates="drop")

    # 2. 计算每个分层应该抽多少条
    group_sizes = df.groupby(["account_is_fake", "review_bin"], observed=True).size()
    sample_counts = (group_sizes / len(df) * 1000).round().astype(int)
    
    # 修复取整造成的总数偏差
    diff = 1000 - sample_counts.sum()
    if len(sample_counts) > 0:
        sample_counts.iloc[0] += diff

    # 3. 按计算好的数量逐层抽样并拼接
    sampled_dfs = []
    for (fake, bin_), n in sample_counts.items():
        group_df = df[(df['account_is_fake'] == fake) & (df['review_bin'] == bin_)]
        if n > 0:
            sampled_dfs.append(group_df.sample(n=min(n, len(group_df)), random_state=42))
    
    sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # 4. 打乱并截断
    sampled = sampled.sample(frac=1, random_state=42).head(1000).reset_index(drop=True)
    if "review_bin" in sampled.columns:
        sampled = sampled.drop(columns=["review_bin"])

    # 5. 保存
    sampled.to_csv(SAMPLE_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 已保存交付物: {SAMPLE_CSV}")
    print(f"抽样总数: {len(sampled)}")
    print(f"水军号: {sampled['account_is_fake'].sum()} 条")
    print(f"正常号: {(sampled['account_is_fake']==0).sum()} 条")

if __name__ == "__main__":
    main()