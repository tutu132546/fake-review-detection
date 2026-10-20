import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.sparse.csgraph import connected_components
from config import REVIEW_PARQUET, ACCOUNT_PARQUET, MAT_PATH

def main():
    print(f"读取 {REVIEW_PARQUET}")
    df = pd.read_parquet(REVIEW_PARQUET)
    
    # 1. 基础统计
    g = df.groupby("user_id")
    acc = pd.DataFrame(index=g.size().index)
    acc["n_reviews"] = g.size()
    # 只要该用户有任何一条评论被标为假，该账号视为水军嫌疑号
    acc["account_is_fake"] = g["label"].apply(lambda s: int((s == 1).any()))

    # 2. 提取 32 维行为特征的高阶统计量 (反映行为分布)
    feat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if feat_cols:
        print("正在聚合并提取 32 维行为特征...")
        acc = acc.join(g[feat_cols].mean().add_suffix("_mean"))
        acc = acc.join(g[feat_cols].std().fillna(0).add_suffix("_std"))
        acc = acc.join(g[feat_cols].max().add_suffix("_max"))

    # 3. 提取图结构特征 (极为关键！水军抱团特征)
    print("正在从 net_rur 提取图结构特征...")
    mat = sio.loadmat(str(MAT_PATH))
    net_rur = mat["net_rur"]
    
    # 3.1 连通分量（团伙大小）
    _, cc_labels = connected_components(net_rur, directed=False)
    df_cc = pd.DataFrame({"user_id": df["user_id"], "cc_label": cc_labels})
    # 统计每个账号下的评论，在图里聚集成了几个子团
    cc_group = df_cc.groupby("user_id")["cc_label"].agg(
        n_components="nunique",          # 分裂成几个连通块
        max_cc_size=lambda x: max(x.value_counts()) # 最大团伙的评论数
    )
    acc = acc.join(cc_group)
    
    # 3.2 节点度数（评论之间的交互密集度）
    degrees = np.asarray(net_rur.sum(axis=1)).flatten()
    df_degree = pd.DataFrame({"user_id": df["user_id"], "degree": degrees})
    degree_group = df_degree.groupby("user_id")["degree"].agg(
        degree_mean="mean",
        degree_std="std",
        degree_max="max"
    ).fillna(0)
    acc = acc.join(degree_group)

    # 4. 收尾保存
    acc = acc.reset_index()
    acc.to_parquet(ACCOUNT_PARQUET, index=False)
    
    print(f"\n✅ 已保存 {ACCOUNT_PARQUET}，形状 {acc.shape}")
    print(f"账号数: {len(acc)}，其中水军嫌疑号: {acc['account_is_fake'].sum()} ({acc['account_is_fake'].mean()*100:.2f}%)")
    print(f"特征数: {acc.shape[1]} 列（包含32维行为聚合与图拓扑特征）")

if __name__ == "__main__":
    main()