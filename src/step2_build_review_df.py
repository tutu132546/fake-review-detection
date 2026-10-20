import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from config import MAT_PATH, REVIEW_PARQUET

def get_user_labels(mat):
    # 找显式字段
    for key in ["user_ids", "user_id", "review_user", "users"]:
        if key in mat:
            print(f"使用显式字段 {key} 作为 user_id")
            return np.asarray(mat[key]).flatten().astype(np.int64)
            
    # 找不到就反推
    print("未找到显式 user_id，使用 net_rur 连通分量反推...")
    net_rur = mat["net_rur"]
    n_components, labels = connected_components(net_rur, directed=False)
    return labels.astype(np.int64)

def main():
    print(f"加载文件: {MAT_PATH}")
    mat = sio.loadmat(str(MAT_PATH))
    n_reviews = mat["label"].flatten().shape[0]

    # 建立基础 DF
    df = pd.DataFrame({
        "review_id": np.arange(n_reviews),
        "user_id": get_user_labels(mat),
        "label": mat["label"].flatten().astype(int),
    })

    # 智能抓取时间和评分（兼容不同字段名，没有就跳过）
    for src_key, col_name in [
        ("ratings", "rating"), ("rating", "rating"),
        ("timestamps", "timestamp"), ("timestamp", "timestamp"), 
        ("business_ids", "business_id"), ("business_id", "business_id")
    ]:
        if src_key in mat and col_name not in df.columns:
            df[col_name] = np.asarray(mat[src_key]).flatten()
            print(f"拼入 {src_key} -> {col_name}")

    # 拼 32 维评论级特征
    if "features" in mat:
        feats = mat["features"]
        
        # 🌟 核心修复：处理稀疏矩阵或 numpy.matrix 格式 🌟
        if hasattr(feats, "toarray"):
            feats = feats.toarray()
        feats = np.array(feats, dtype=np.float64) 
        
        if feats.shape[0] == n_reviews:
            feat_cols = [f"f{i}" for i in range(feats.shape[1])]
            df_feat = pd.DataFrame(feats, columns=feat_cols)
            df = pd.concat([df, df_feat], axis=1)
            print(f"拼入 {feats.shape[1]} 维评论级特征")

    # 保存为 Parquet
    df.to_parquet(REVIEW_PARQUET, index=False)
    print(f"\n✅ 已保存 {REVIEW_PARQUET}，形状 {df.shape}")
    print(f"唯一用户数: {df['user_id'].nunique()}")

if __name__ == "__main__":
    main()