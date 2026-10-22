import pandas as pd
import numpy as np
import json
import logging
from config import RAW_CSV_PATH, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_account_level_base():
    logging.info("1. 读取原始评论级数据...")
    df = pd.read_csv(RAW_CSV_PATH)
    df['is_fake_review'] = (df['label'] == -1).astype(int)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    logging.info("2. 聚合账号基础特征...")
    g = df.groupby('user_id')
    acc = pd.DataFrame()
    acc['n_reviews'] = g.size()
    acc['account_is_fake'] = g['is_fake_review'].max()
    acc['active_days'] = g['date'].apply(lambda s: (s.max() - s.min()).days + 1)
    acc['daily_avg_reviews'] = acc['n_reviews'] / acc['active_days']
    acc['max_daily_reviews'] = g.apply(lambda x: x.groupby(x['date'].dt.date).size().max())
    
    # 【核心修复】评论 < 3 条的 burstiness 保留为 NaN，不要设为 0
    def calc_burstiness(ts):
        if len(ts) < 3: return np.nan 
        ts = np.sort(ts.values)
        gaps = np.diff(ts).astype('timedelta64[s]').astype(float)
        return gaps.std() / gaps.mean() if gaps.mean() > 0 else np.nan
    
    acc['burstiness'] = g['date'].apply(calc_burstiness)
    
    # 评分特征
    prod_avg = df.groupby('product_id')['rating'].mean().rename('prod_avg_rating')
    df = df.merge(prod_avg, on='product_id', how='left')
    acc['rating_deviation'] = df.groupby('user_id').apply(lambda x: (x['rating'] - x['prod_avg_rating']).mean())
    acc['extreme_ratio'] = g['rating'].apply(lambda s: ((s == 1) | (s == 5)).mean())
    
    logging.info("3. 计算 Z-score 参数...")
    # 处理 NaN 的 Z-score 计算（忽略 NaN 计算均值和方差）
    feature_cols = ['n_reviews', 'active_days', 'daily_avg_reviews', 'max_daily_reviews', 'burstiness', 'rating_deviation', 'extreme_ratio']
    scaler_params = {}
    for col in feature_cols:
        mean_val = float(acc[col].mean())
        std_val = float(acc[col].std()) if acc[col].std() > 0 else 1.0
        scaler_params[col] = {"mean": mean_val, "std": std_val}
        acc[f'{col}_z'] = (acc[col] - mean_val) / std_val
        
    with open(PROCESSED_DIR / "scaler_params.json", 'w') as f:
        json.dump(scaler_params, f, indent=4)

    acc = acc.reset_index()
    n_fake = acc['account_is_fake'].sum()
    logging.info(f"✅ 底表生成完毕！总账号 {len(acc)}，水军 {n_fake}，正常 {len(acc)-n_fake}")
    acc.to_parquet(PROCESSED_DIR / "clean_accounts.parquet", index=False)

if __name__ == "__main__":
    build_account_level_base()