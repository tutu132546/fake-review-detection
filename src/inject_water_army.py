import pandas as pd
import numpy as np
import json
from config import ACCOUNT_PARQUET, PROCESSED_DIR

def main():
    print("加载账号级数据...")
    df = pd.read_parquet(ACCOUNT_PARQUET)
    
    # 抽 200 个正常人进行水军注入
    normal_df = df[df["account_is_fake"] == 0].sample(n=200, random_state=42).copy()
    injection_logs = []
    
    for i, idx in enumerate(normal_df.index):
        account_id = normal_df.loc[idx, "user_id"]
        
        # 模拟注入：偏移特征，加大团伙连通度
        if i < 70:
            normal_df.loc[idx, "f0_mean"] += 3.5
            normal_df.loc[idx, "f1_mean"] += 2.0
            normal_df.loc[idx, "max_cc_size"] *= 2
            reason = "coordinated_burst_on_features_f0_f1"
        elif i < 140:
            normal_df.loc[idx, "f2_std"] *= 0.1
            normal_df.loc[idx, "f3_std"] *= 0.1
            reason = "low_variance_on_features_f2_f3"
        else:
            normal_df.loc[idx, "max_cc_size"] *= 5
            normal_df.loc[idx, "degree_max"] *= 3
            reason = "topological_anomaly_high_degree"
            
        normal_df.loc[idx, "account_is_fake"] = 1
        normal_df.loc[idx, "is_injected"] = 1
        
        # 记录：到底注入了什么（这是验证后续 DCFO 解释准不准的唯一标准！）
        injection_logs.append({
            "user_id": int(account_id),
            "injection_reason": reason
        })
        
    untouched_df = df.drop(normal_df.index)
    untouched_df["is_injected"] = 0
    
    augmented_df = pd.concat([untouched_df, normal_df])
    
    augmented_path = PROCESSED_DIR / "augmented_accounts.parquet"
    augmented_df.to_parquet(augmented_path, index=False)
    
    log_path = PROCESSED_DIR / "injection_log.json"
    with open(log_path, "w") as f:
        json.dump(injection_logs, f, indent=4)
        
    print(f"✅ 注入完成！数据集已保存至 {augmented_path}")
    print(f"✅ 注入记录已保存至 {log_path}")

if __name__ == "__main__":
    main()