import pandas as pd
import numpy as np
from config import ACCOUNT_PARQUET, PROCESSED_DIR

def main():
    df = pd.read_parquet(ACCOUNT_PARQUET)
    
    # 1. 定义32维特征列
    feat_cols = [f'f{i}_mean' for i in range(32)]
    
    # 2. 定义行为代理变量（与时间/爆发性高度相关）
    proxy_vars = ['n_reviews', 'max_cc_size', 'degree_max']
    
    # 3. 计算每个f特征与代理变量的最大绝对相关性
    corr_results = {}
    for f in feat_cols:
        max_corr = 0.0
        for proxy in proxy_vars:
            if proxy in df.columns:
                corr = abs(df[f].corr(df[proxy]))
                if corr > max_corr:
                    max_corr = corr
        corr_results[f] = max_corr
    
    # 4. 按相关性降序排列，选出Top 8作为高潜行为特征
    sorted_feats = sorted(corr_results.items(), key=lambda x: x[1], reverse=True)
    top_feats = [feat for feat, _ in sorted_feats[:8]]
    
    print("=== 高潜行为特征（与评论数/图规模强相关） ===")
    for feat in top_feats:
        print(f"  {feat}: 最大相关系数 = {corr_results[feat]:.4f}")
    
    # 5. 语义映射表（基于SpEagle论文分类，标注为近似映射）
    # 注意：此映射基于统计推断与论文语义，非官方索引词典
    semantic_map = {
        # 行为类映射（优先级最高）
        'MNR':   {'feature': top_feats[0], 'note': '单日最大评论数（行为密度代理）'},
        'BST':   {'feature': top_feats[1], 'note': '爆发性（时间跨度代理）'},
        'avgRD': {'feature': top_feats[2], 'note': '平均评分偏差（评分偏离代理）'},
        'EXT':   {'feature': top_feats[3], 'note': '极端评分比例（评分极端性代理）'},
        # 图结构映射
        'graph_max_cc_size': {'feature': 'max_cc_size', 'note': '最大抱团规模'},
        'graph_degree_max':  {'feature': 'degree_max', 'note': '最大交互度数'},
    }
    
    # 6. 构建输出表
    out_df = pd.DataFrame()
    out_df['user_id'] = df['user_id']
    out_df['account_is_fake'] = df['account_is_fake']
    out_df['n_reviews'] = df['n_reviews']
    
    # 输出筛选后的行为特征（保留原始列名并附Z-score）
    for feat in top_feats:
        out_df[feat] = df[feat]
        out_df[f'{feat}_zscore'] = (df[feat] - df[feat].mean()) / (df[feat].std() + 1e-9)
    
    # 输出语义映射列（便于队友直接使用）
    for sem_name, info in semantic_map.items():
        raw_col = info['feature']
        if raw_col in df.columns:
            out_df[f'semantic_{sem_name}'] = df[raw_col]
            out_df[f'semantic_{sem_name}_zscore'] = (
                (df[raw_col] - df[raw_col].mean()) / (df[raw_col].std() + 1e-9)
            )
    
    # 7. 保存
    out_path = PROCESSED_DIR / "teammate_features_professional.csv"
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 专业版特征表已生成: {out_path}")
    print(f"包含 {len(out_df)} 个账号，{len(out_df.columns)} 列")
    print("\n=== 语义映射说明（交付给队友时附上） ===")
    for sem_name, info in semantic_map.items():
        print(f"  semantic_{sem_name} ← {info['feature']} ({info['note']})")

if __name__ == "__main__":
    main()