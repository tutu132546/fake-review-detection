import pandas as pd
import numpy as np
import scipy.io as sio
import logging
from scipy.sparse.csgraph import connected_components
from config import RAW_DIR, PROCESSED_DIR, MAT_PATH, RAW_METADATA_PATH

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_base():
    logging.info("1. 解析 metadata.txt 真实账号标签...")
    # 字段: [user_id, product_id, rating, label, date]
    df_rev = pd.read_csv(RAW_METADATA_PATH, sep=r'\s+', header=None, 
                         names=['user_id', 'product_id', 'rating', 'label', 'date'])
    
    # 聚合账号级标签和总数
    g_real = df_rev.groupby('user_id')
    acc_real = pd.DataFrame({
        'n_reviews': g_real.size(),
        'account_is_fake': g_real['label'].apply(lambda s: int((s == -1).any()))
    })

    logging.info("2. 解析 YelpChi.mat 提取图特征与 32 维特征...")
    mat = sio.loadmat(str(MAT_PATH))
    
    # 提取 32 维特征 (处理稀疏矩阵)
    feats = mat["features"]
    if hasattr(feats, "toarray"): 
        feats = feats.toarray()
    
    # 用连通分量反推 .mat 中每个节点对应的 user_id
    _, cc_labels = connected_components(mat["net_rur"], directed=False)
    degrees = np.asarray(mat["net_rur"].sum(axis=1)).flatten()

    df_mat = pd.DataFrame({
        'user_id': cc_labels,
        'degree': degrees
    })
    for i in range(32):
        df_mat[f'f{i}'] = feats[:, i]

    logging.info("3. 聚合特征并进行内连接...")
    agg_dict = {
        'max_cc_size': ('user_id', 'size'), # 连通团伙大小
        'degree_max': ('degree', 'max')     # 最大节点度
    }
    for i in range(32):
        agg_dict[f'f{i}_mean'] = (f'f{i}', 'mean')
        agg_dict[f'f{i}_std'] = (f'f{i}', 'std')
        agg_dict[f'f{i}_max'] = (f'f{i}', 'max')

    struct_group = df_mat.groupby('user_id').agg(**agg_dict).fillna(0)
    
    # 合并真实标签与图/匿名特征
    final_df = acc_real.join(struct_group, how='inner').reset_index()
    
    # 全量 Z-score 标准化（为后续注入对比打基础）
    cols_to_scale = [c for c in final_df.columns if c.startswith('f') or c in ['max_cc_size', 'degree_max']]
    for c in cols_to_scale:
        final_df[f'{c}_z'] = (final_df[c] - final_df[c].mean()) / (final_df[c].std() + 1e-9)

    out_path = PROCESSED_DIR / "clean_accounts.parquet"
    final_df.to_parquet(out_path, index=False)
    logging.info(f"✅ 账号级底表生成完毕: {out_path}, 形状: {final_df.shape}")

if __name__ == "__main__":
    build_base()