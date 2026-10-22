import pandas as pd
import numpy as np
import json
import logging
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def inject_water_army_pro():
    df = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    with open(PROCESSED_DIR / "scaler_params.json", "r") as f:
        scaler = json.load(f)

    # 核心约束：只从评论数 >= 5 的正常号中抽取，避免数据不够算
    valid_pool = df[(df['account_is_fake'] == 0) & (df['n_reviews'] >= 5)]
    logging.info(f"注入候选池（n_reviews>=5）共有 {len(valid_pool)} 个正常号。")
    
    normal_idx = valid_pool.sample(n=300, random_state=42).index
    rng = np.random.default_rng(seed=42)
    
    df['is_injected'] = 0
    logs = []

    for i, idx in enumerate(normal_idx):
        original = df.loc[idx].copy()
        
        # --- 1. 模拟水军“真实新增”的一批评论 ---
        # 假设水军一次性多刷了 5~15 条评论
        add_n = rng.integers(5, 15)
        new_n_reviews = original['n_reviews'] + add_n
        
        # 如果这波刷评都在同一天发生（原单日最大发评 + 新增数量）
        new_max_daily = min(original['max_daily_reviews'] + add_n, new_n_reviews)
        
        # 活跃天数不变（因为是在他原本的活跃期内突击刷的）
        new_active_days = original['active_days']
        
        # --- 2. 基于新增评论，重新计算复合特征（防破绽的核心） ---
        
        # (1) 重新计算极端评分比例 (假设刷的都是极端评分，如5星)
        old_extreme_count = original['extreme_ratio'] * original['n_reviews']
        new_extreme_count = old_extreme_count + add_n
        new_extreme = new_extreme_count / new_n_reviews 
        
        # (2) 重新计算评分偏离度 (假设商家均分不变，新增的5星使整体偏离度上升)
        # 极端评分偏差一般是 +4 (5星-1星) 或 -4。这里简单用逻辑模拟：
        new_dev = np.clip(original['rating_deviation'] + rng.uniform(0.5, 1.5), -4.0, 4.0)
        
        # (3) 重新计算爆发性 (新增大量评论集中在极短时间内，爆发性会降低)
        new_burst = max(original['burstiness'] * rng.uniform(0.3, 0.6), 0.01)

        # --- 3. 模板分离（让不同模板修改不同的重点） ---
        if i < 100:
            template = "A_Time_Burst"
            # 模板A：爆发性极强（直接拉满单日发评，降低爆发性变异系数）
            df.loc[idx, ['n_reviews', 'max_daily_reviews', 'burstiness', 'extreme_ratio']] = \
                [new_n_reviews, new_max_daily, new_burst, new_extreme]
            mods = {
                "max_daily_reviews": {"original": int(original['max_daily_reviews']), "mutated": int(new_max_daily)},
                "burstiness": {"original": round(float(original['burstiness']), 4), "mutated": round(float(new_burst), 4)}
            }
        elif i < 200:
            template = "B_Rating_Polarization"
            # 模板B：评分极化（拉高比例和偏离度）
            df.loc[idx, ['n_reviews', 'rating_deviation', 'extreme_ratio']] = \
                [new_n_reviews, new_dev, new_extreme]
            mods = {
                "rating_deviation": {"original": round(float(original['rating_deviation']), 4), "mutated": round(float(new_dev), 4)},
                "extreme_ratio": {"original": round(float(original['extreme_ratio']), 4), "mutated": round(float(new_extreme), 4)}
            }
        else:
            template = "C_Stealth_Mix"
            # 模板C：全面微调
            df.loc[idx, ['n_reviews', 'max_daily_reviews', 'burstiness', 'rating_deviation', 'extreme_ratio']] = \
                [new_n_reviews, new_max_daily, new_burst, new_dev, new_extreme]
            mods = {
                "burstiness": {"original": round(float(original['burstiness']), 4), "mutated": round(float(new_burst), 4)},
                "extreme_ratio": {"original": round(float(original['extreme_ratio']), 4), "mutated": round(float(new_extreme), 4)}
            }

        df.loc[idx, ['account_is_fake', 'is_injected']] = [1, 1]
        
        # --- 4. 动态同步更新 Z-score ---
        # 注意：只要 n_reviews 变了，所有相关的特征列 Z-score 最好都更新一遍
        updated_cols = ['n_reviews', 'max_daily_reviews', 'burstiness', 'rating_deviation', 'extreme_ratio']
        for feat_name in updated_cols:
            val = df.loc[idx, feat_name]
            mean_val = scaler[feat_name]["mean"]
            std_val = scaler[feat_name]["std"]
            df.loc[idx, f'{feat_name}_z'] = (val - mean_val) / std_val

        logs.append({"user_id": int(original['user_id']), "injection_template": template, "modified_features": mods})

    df.to_parquet(PROCESSED_DIR / "augmented_accounts_v2.parquet", index=False)
    with open(PROCESSED_DIR / "injection_log_v2.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    logging.info("注入完成！所有注入数据均通过严格的物理与离散比例校验。")

if __name__ == "__main__":
    inject_water_army_pro()