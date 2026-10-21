import pandas as pd
import numpy as np
import json
from config import ACCOUNT_PARQUET, PROCESSED_DIR

def generate_stealthy_burst_profile(original_row, rng):
    mutated_row = original_row.copy()
    mods = {}
    
    feat = 'f0_mean'
    delta = rng.uniform(1.5, 3.0) 
    mutated_row[feat] += delta
    mods[feat] = {"original": round(float(original_row[feat]), 4), "mutated": round(float(mutated_row[feat]), 4)}
    
    feat_std = 'f1_std'
    mutated_row[feat_std] *= rng.uniform(0.1, 0.3)
    mods[feat_std] = {"original": round(float(original_row[feat_std]), 4), "mutated": round(float(mutated_row[feat_std]), 4)}

    mutated_row['max_cc_size'] += rng.integers(5, 15)
    mutated_row['degree_max'] += rng.integers(10, 30)
    mods['max_cc_size'] = {"original": round(float(original_row['max_cc_size']), 4), "mutated": round(float(mutated_row['max_cc_size']), 4)}
    mods['degree_max'] = {"original": round(float(original_row['degree_max']), 4), "mutated": round(float(mutated_row['degree_max']), 4)}

    return mutated_row, "Template_A_Stealthy_Burst", mods

def generate_flooding_profile(original_row, rng):
    mutated_row = original_row.copy()
    mods = {}
    
    for feat in ['f2_mean', 'f3_mean']:
        delta = rng.uniform(3.0, 5.0) 
        mutated_row[feat] += delta
        mods[feat] = {"original": round(float(original_row[feat]), 4), "mutated": round(float(mutated_row[feat]), 4)}
        
    mutated_row['max_cc_size'] = original_row['max_cc_size'] * rng.uniform(2.0, 3.0)
    mutated_row['degree_mean'] = original_row['degree_mean'] + rng.uniform(5.0, 10.0)
    mods['max_cc_size'] = {"original": round(float(original_row['max_cc_size']), 4), "mutated": round(float(mutated_row['max_cc_size']), 4)}
    mods['degree_mean'] = {"original": round(float(original_row['degree_mean']), 4), "mutated": round(float(mutated_row['degree_mean']), 4)}

    return mutated_row, "Template_B_Flooding", mods

def generate_camouflage_profile(original_row, rng):
    mutated_row = original_row.copy()
    mods = {}
    
    feat = 'f5_max'
    mutated_row[feat] *= rng.uniform(1.05, 1.15) 
    mods[feat] = {"original": round(float(original_row[feat]), 4), "mutated": round(float(mutated_row[feat]), 4)}
    
    mutated_row['n_components'] = original_row['n_components'] + rng.integers(1, 3)
    mods['n_components'] = {"original": round(float(original_row['n_components']), 4), "mutated": round(float(mutated_row['n_components']), 4)}

    return mutated_row, "Template_C_Camouflage", mods

def main():
    print("加载账号级数据...")
    df = pd.read_parquet(ACCOUNT_PARQUET)
    
    # 🌟 核心修复点：提前将可能被加上小数的图结构列转为 float 类型
    graph_cols = ['max_cc_size', 'degree_max', 'degree_mean', 'n_components']
    for col in graph_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    normal_df = df[df["account_is_fake"] == 0].copy()
    
    num_inject = 300
    if len(normal_df) < num_inject:
        raise ValueError(f"正常账号数量不足 ({len(normal_df)})，无法进行注入！")
        
    injected_indices = normal_df.sample(n=num_inject, random_state=42).index
    
    rng = np.random.default_rng(seed=42)
    injection_logs = []
    
    print(f"开始注入 {num_inject} 个水军账号（模拟 3 种高级攻击模板）...")
    
    for idx in injected_indices:
        original_row = df.loc[idx].copy()
        
        template_choice = rng.choice(['A', 'B', 'C'])
        
        if template_choice == 'A':
            mutated_row, reason, mods = generate_stealthy_burst_profile(original_row, rng)
        elif template_choice == 'B':
            mutated_row, reason, mods = generate_flooding_profile(original_row, rng)
        else:
            mutated_row, reason, mods = generate_camouflage_profile(original_row, rng)
            
        for col in mutated_row.index:
            if col not in ['user_id', 'account_is_fake', 'is_injected']:
                df.loc[idx, col] = mutated_row[col]
                
        df.loc[idx, "account_is_fake"] = 1
        df.loc[idx, "is_injected"] = 1
        
        injection_logs.append({
            "user_id": int(original_row["user_id"]),
            "injection_reason": reason,
            "modified_features": mods
        })
        
    df["is_injected"] = df["is_injected"].fillna(0).astype(int)
    
    augmented_path = PROCESSED_DIR / "augmented_accounts.parquet"
    df.to_parquet(augmented_path, index=False)
    
    log_path = PROCESSED_DIR / "injection_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(injection_logs, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 注入完成！注入后完整数据集已保存至:\n   {augmented_path}")
    print(f"✅ 详细的注入行为记录 (JSON) 已保存至:\n   {log_path}")
    print(f"现在数据集总量: {len(df)}，其中人工注入水军: {num_inject} 个。")

if __name__ == "__main__":
    main()