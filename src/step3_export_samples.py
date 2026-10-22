import pandas as pd
import logging
from config import RAW_DIR, PROCESSED_DIR, RAW_METADATA_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_data():
    logging.info("1. 处理原始评论级数据 (给队友)...")
    # 将无表头的 txt 转为带表头的标准 csv
    df_rev = pd.read_csv(RAW_METADATA_PATH, sep=r'\s+', header=None, 
                         names=['user_id', 'product_id', 'rating', 'label', 'date'])
    raw_csv_path = PROCESSED_DIR / "YelpChi_reviews.csv"
    df_rev.to_csv(raw_csv_path, index=False)
    logging.info(f"✅ 原始评论级数据已导出: {raw_csv_path} (共 {len(df_rev)} 条)")

    logging.info("2. 从干净底表分层抽取账号样本...")
    df_clean = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    
    # 分离正常号和欺诈号池子
    fake_pool = df_clean[df_clean['account_is_fake'] == 1]
    real_pool = df_clean[df_clean['account_is_fake'] == 0]
    
    logging.info(f"📊 当前数据池现状 -> 水军号: {len(fake_pool)} 个, 正常号: {len(real_pool)} 个")
    
    # 目标数量 (如果池子不够，就用池子最大量)
    target_total = 1000
    target_fake = 200
    
    n_fake = min(target_fake, len(fake_pool))
    n_real = min(target_total - n_fake, len(real_pool))
    
    logging.info(f"🎯 计划抽样 -> 水军号: {n_fake} 个, 正常号: {n_real} 个")
    
    if n_fake == 0:
        logging.warning("⚠️ 严重警告：数据集中没有任何水军号！请检查 step1 的合并逻辑。")
    
    # 安全抽样
    fake_samples = fake_pool.sample(n=n_fake, random_state=42) if n_fake > 0 else pd.DataFrame()
    real_samples = real_pool.sample(n=n_real, random_state=42) if n_real > 0 else pd.DataFrame()
    
    sampled = pd.concat([fake_samples, real_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    sample_csv_path = PROCESSED_DIR / "yelpchi_1000_accounts.csv"
    sampled.to_csv(sample_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"✅ 账号级样本已导出: {sample_csv_path} (实际共 {len(sampled)} 条)")

if __name__ == "__main__":
    export_data()