import pandas as pd
import numpy as np
import json
import logging
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def inject_water_army_pro():
    logging.info("加载干净底表与缩放参数...")
    df = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    
    # 读取第一步保存的 Z-score 参数，确保数据绝对可比
    with open(PROCESSED_DIR / "scaler_params.json", "r") as f:
        scaler = json.load(f)

    # 抽取 300 个正常人进行注入
    normal_idx = df[df['account_is_fake'] == 0].sample(n=300, random_state=42).index
    rng = np.random.default_rng(seed=42)
    
    df['is_injected'] = 0
    logs = []
    
    logging.info("开始水军注入 (严格分离 A, B, C 模板，附带物理约束)...")

    for i, idx in enumerate(normal_idx):
        original = df.loc[idx].copy()
        
        # 按 1:1:1 分配模板 A, B, C
        if i < 100:
            template = "A_Time_Burst"
            # 模板 A：拉高单日最大发评，改变爆发性
            new_max_daily = original['max_daily_reviews'] + rng.integers(3, 8)
            # 【物理约束】单日发评不能超过总发评
            new_max_daily = min(new_max_daily, original['n_reviews'])
            
            # 爆发性变异系数变小（发帖更集中），但不能小于 0
            new_burst = max(original['burstiness'] * rng.uniform(0.2, 0.5), 0.01)
            
            df.loc[idx, 'max_daily_reviews'] = new_max_daily
            df.loc[idx, 'burstiness'] = new_burst
            
            mods = {
                "max_daily_reviews": {"original": int(original['max_daily_reviews']), "mutated": int(new_max_daily)},
                "burstiness": {"original": round(float(original['burstiness']), 4), "mutated": round(float(new_burst), 4)}
            }
            
        elif i < 200:
            template = "B_Rating_Polarization"
            # 模板 B：评分偏离变大（刷好评/差评），极端评分比例拉满
            new_dev = original['rating_deviation'] + rng.uniform(1.5, 3.0)
            new_extreme = min(original['extreme_ratio'] + rng.uniform(0.2, 0.4), 1.0) # 比例不能超过 1
            
            df.loc[idx, 'rating_deviation'] = new_dev
            df.loc[idx, 'extreme_ratio'] = new_extreme
            
            mods = {
                "rating_deviation": {"original": round(float(original['rating_deviation']), 4), "mutated": round(float(new_dev), 4)},
                "extreme_ratio": {"original": round(float(original['extreme_ratio']), 4), "mutated": round(float(new_extreme), 4)}
            }
            
        else:
            template = "C_Stealth_Mix"
            # 模板 C：隐蔽伪装，各项微调
            new_burst = max(original['burstiness'] * rng.uniform(0.5, 0.8), 0.01)
            new_extreme = min(original['extreme_ratio'] + rng.uniform(0.1, 0.2), 1.0)
            
            df.loc[idx, 'burstiness'] = new_burst
            df.loc[idx, 'extreme_ratio'] = new_extreme
            
            mods = {
                "burstiness": {"original": round(float(original['burstiness']), 4), "mutated": round(float(new_burst), 4)},
                "extreme_ratio": {"original": round(float(original['extreme_ratio']), 4), "mutated": round(float(new_extreme), 4)}
            }

        # 修改标签
        df.loc[idx, 'account_is_fake'] = 1
        df.loc[idx, 'is_injected'] = 1
        
        # 【核心修复】动态同步更新被修改特征的 Z-score
        for feat_name in mods.keys():
            mutated_val = mods[feat_name]["mutated"]
            mean_val = scaler[feat_name]["mean"]
            std_val = scaler[feat_name]["std"]
            df.loc[idx, f'{feat_name}_z'] = (mutated_val - mean_val) / std_val

        # 记录严谨的注入日志
        logs.append({
            "user_id": int(original['user_id']),
            "injection_template": template,
            "modified_features": mods
        })

    # 保存注入后的数据集
    out_parquet_path = PROCESSED_DIR / "augmented_accounts_v2.parquet"
    df.to_parquet(out_parquet_path, index=False)
    
    # 保存注入日志
    log_path = PROCESSED_DIR / "injection_log_v2.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    logging.info(f"✅ 注入完成！数据集: {out_parquet_path}")
    logging.info(f"✅ 注入日志: {log_path} (包含 {len(logs)} 条记录)")
    logging.info(f"👉 分布检查: 注入后水军总数 = {df['account_is_fake'].sum()}")

if __name__ == "__main__":
    inject_water_army_pro()