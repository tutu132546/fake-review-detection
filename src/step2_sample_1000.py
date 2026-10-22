import pandas as pd
import logging
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def sample_1000_accounts():
    logging.info("1. 读取干净账号底表...")
    base_path = PROCESSED_DIR / "clean_accounts.parquet"
    df = pd.read_parquet(base_path)
    
    # 分离数据池
    fake_pool = df[df['account_is_fake'] == 1]
    real_pool = df[df['account_is_fake'] == 0]
    
    logging.info(f"👉 当前数据池现状: 水军 {len(fake_pool)} 个, 正常 {len(real_pool)} 个")
    
    # 设定目标抽样数量 (为了后续模型训练，水军和正常号的比例定为 1:4)
    target_fake = 200
    target_real = 800
    
    # 健壮性防护：如果池子里的数量不够目标数量，就用池子里全部的数量
    n_fake = min(target_fake, len(fake_pool))
    n_real = min(target_real, len(real_pool))
    
    logging.info(f"🎯 计划抽取: 水军 {n_fake} 个, 正常 {n_real} 个 (共 {n_fake + n_real} 条)")
    
    # 执行抽样
    fake_samples = fake_pool.sample(n=n_fake, random_state=42) if n_fake > 0 else pd.DataFrame()
    real_samples = real_pool.sample(n=n_real, random_state=42) if n_real > 0 else pd.DataFrame()
    
    # 合并并打乱顺序（打乱可以避免训练时数据按标签聚集，影响模型收敛）
    sampled = pd.concat([fake_samples, real_samples])
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存为 CSV (使用 utf-8-sig 防止队友用 Excel 打开乱码)
    out_path = PROCESSED_DIR / "yelpchi_1000_accounts.csv"
    sampled.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    logging.info(f"✅ 抽样完成！文件已保存至: {out_path}")
    logging.info(f"最终样本确认: 水军 {len(sampled[sampled['account_is_fake']==1])} 条, 正常 {len(sampled[sampled['account_is_fake']==0])} 条。")

if __name__ == "__main__":
    sample_1000_accounts()