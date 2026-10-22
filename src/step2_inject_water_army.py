import pandas as pd
import numpy as np
import json
import logging
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inject_water_army():
    logging.info("加载干净底表...")
    df = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    
    # 【核心修复1】强转数据类型，防止 Pandas 严格类型检查报错
    cols_to_float = ['max_cc_size', 'degree_max', 'f0_mean', 'f2_std']
    for col in cols_to_float:
        df[col] = df[col].astype(float)
    
    # 提取原始正常账号的缩放参数（为了注入后复用，保证实验的可比性）
    scaler_params = {}
    for col in cols_to_float:
        scaler_params[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std() + 1e-9)
        }
    with open(PROCESSED_DIR / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=4)
    
    # 从正常账号中抽取 300 个
    normal_df = df[df['account_is_fake'] == 0]
    inject_idx = normal_df.sample(n=300, random_state=42).index
    
    rng = np.random.default_rng(seed=42)
    logs = []
    df['is_injected'] = 0
    
    logging.info("开始专业级注入（模板 A: 隐蔽突发, B: 团伙抱团）...")
    for idx in inject_idx:
        original = df.loc[idx].copy()
        
        # --- 模板 A: 隐蔽突发 (均值偏移，方差压缩) ---
        # 提升均值，模拟刷好评；压缩方差，模拟评论高度雷同
        new_f0_mean = original['f0_mean'] + rng.uniform(1.5, 3.0)
        new_f2_std = original['f2_std'] * rng.uniform(0.1, 0.3)
        
        # --- 模板 B: 团伙抱团 (图结构扩张) ---
        new_cc_size = original['max_cc_size'] * rng.uniform(1.5, 2.5) + rng.integers(5, 15)
        new_degree_max = original['degree_max'] + rng.integers(10, 30)

        # 写入 DataFrame
        df.loc[idx, 'f0_mean'] = new_f0_mean
        df.loc[idx, 'f2_std'] = new_f2_std
        df.loc[idx, 'max_cc_size'] = new_cc_size
        df.loc[idx, 'degree_max'] = new_degree_max
        
        # 【核心修复2】更新注入后的 Z-score（严格复用注入前的均值和方差）
        df.loc[idx, 'f0_mean_z'] = (new_f0_mean - scaler_params['f0_mean']['mean']) / scaler_params['f0_mean']['std']
        df.loc[idx, 'f2_std_z'] = (new_f2_std - scaler_params['f2_std']['mean']) / scaler_params['f2_std']['std']
        df.loc[idx, 'max_cc_size_z'] = (new_cc_size - scaler_params['max_cc_size']['mean']) / scaler_params['max_cc_size']['std']
        
        # 修改标签
        df.loc[idx, 'account_is_fake'] = 1
        df.loc[idx, 'is_injected'] = 1
        
        # 记录详细日志，供 DCFO 验证
        logs.append({
            "user_id": int(original['user_id']),
            "injection_reason": "Template_A_and_B_Mix",
            "modified_features": {
                "f0_mean": {"original": round(float(original['f0_mean']), 4), "mutated": round(float(new_f0_mean), 4)},
                "f2_std": {"original": round(float(original['f2_std']), 4), "mutated": round(float(new_f2_std), 4)},
                "max_cc_size": {"original": round(float(original['max_cc_size']), 4), "mutated": round(float(new_cc_size), 4)},
                "degree_max": {"original": round(float(original['degree_max']), 4), "mutated": round(float(new_degree_max), 4)}
            }
        })

    # 保存注入后的完整数据集
    out_path = PROCESSED_DIR / "augmented_accounts.parquet"
    df.to_parquet(out_path, index=False)
    
    # 保存日志
    with open(PROCESSED_DIR / "injection_log.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
        
    logging.info(f"✅ 水军注入完成: {out_path}")
    logging.info(f"✅ 特征缩放参数与注入日志已保存。")

if __name__ == "__main__":
    inject_water_army()