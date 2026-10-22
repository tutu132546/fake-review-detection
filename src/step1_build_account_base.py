import pandas as pd
import numpy as np
import json
import logging
from config import RAW_CSV_PATH, PROCESSED_DIR

# 配置日志，让终端输出清晰易读
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_account_level_base():
    logging.info("1. 读取原始评论级数据...")
    # 假设你的 CSV 表头是: user_id, product_id, rating, label, date
    df = pd.read_csv(RAW_CSV_PATH)
    
    # --- 核心防呆：严格处理标签 ---
    # 原数据中 1 为正常，-1 为水军。某用户只要有一条 -1 评论，即标为水军(1)
    logging.info("2. 转换标签逻辑 (label=-1 为水军)...")
    df['is_fake_review'] = (df['label'] == -1).astype(int)
    
    # 时间格式转换，出错则强制报错（ToNaT）
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']) # 丢弃时间解析失败的脏数据
    
    logging.info("3. 计算每个账号的复杂行为特征...")
    g = df.groupby('user_id')
    
    # 基础属性
    acc = pd.DataFrame()
    acc['n_reviews'] = g.size()
    acc['account_is_fake'] = g['is_fake_review'].max() # 只要发过一次假评论，就算假号
    
    # (1) 注册时长 (天) = 最后一次评论 - 第一次评论
    acc['active_days'] = g['date'].apply(lambda s: (s.max() - s.min()).days + 1)
    
    # (2) 日均发评数 = 总评论数 / 活跃天数
    acc['daily_avg_reviews'] = acc['n_reviews'] / acc['active_days']
    
    # (3) 单日最大发评数 (水军爆发的最直接指标)
    acc['max_daily_reviews'] = g.apply(
        lambda x: x.groupby(x['date'].dt.date).size().max()
    )
    
    # (4) 爆发性 (Burstiness) = 相邻评论时间间隔的变异系数 (标准差/均值)
    # 水军往往集中爆发，该值会异常小或异常大
    def calc_burstiness(ts):
        if len(ts) < 3: return 0.0  # 评论太少无法计算时间间隔
        ts = np.sort(ts.values)
        gaps = np.diff(ts).astype('timedelta64[s]').astype(float)
        mean_gap = gaps.mean()
        if mean_gap == 0: return 0.0
        return gaps.std() / mean_gap
    acc['burstiness'] = g['date'].apply(calc_burstiness)
    
    # (5) 评分偏离度 = 用户的打分 - 该商品的真实平均分
    # 计算商品全局均分并合并
    prod_avg = df.groupby('product_id')['rating'].mean().rename('prod_avg_rating')
    df = df.merge(prod_avg, on='product_id', how='left')
    df['rating_dev'] = df['rating'] - df['prod_avg_rating']
    acc['rating_deviation'] = df.groupby('user_id')['rating_dev'].mean()
    
    # (6) 极端评分比例 = (1星 + 5星) / 总评论数
    acc['extreme_ratio'] = g['rating'].apply(lambda s: ((s == 1) | (s == 5)).mean())
    
    logging.info("4. 计算全量 Z-score 标准化 (保存参数供后续注入复用)...")
    feature_cols = ['n_reviews', 'active_days', 'daily_avg_reviews', 
                    'max_daily_reviews', 'burstiness', 'rating_deviation', 'extreme_ratio']
    
    scaler_params = {}
    for col in feature_cols:
        mean_val = float(acc[col].mean())
        std_val = float(acc[col].std())
        # 防止除以 0 的极端情况
        if std_val == 0: std_val = 1.0 
        
        scaler_params[col] = {"mean": mean_val, "std": std_val}
        acc[f'{col}_z'] = (acc[col] - mean_val) / std_val
        
    # 保存缩放参数，为后续的水军注入实验做防泄漏准备
    with open(PROCESSED_DIR / "scaler_params.json", 'w') as f:
        json.dump(scaler_params, f, indent=4)

    logging.info("5. 保存账号级底表...")
    acc = acc.reset_index()
    
    # 检查并打印数据分布，确认标签没反
    n_fake = acc['account_is_fake'].sum()
    n_real = len(acc) - n_fake
    logging.info(f"✅ 底表生成完毕！总计 {len(acc)} 个账号。")
    logging.info(f"👉 真实性校验: 水军号 {n_fake} 个 ({n_fake/len(acc)*100:.1f}%)，正常号 {n_real} 个 ({n_real/len(acc)*100:.1f}%)")
    
    acc.to_parquet(PROCESSED_DIR / "clean_accounts.parquet", index=False)
    logging.info(f"✅ 文件已保存至: data/processed/clean_accounts.parquet")

if __name__ == "__main__":
    build_account_level_base()