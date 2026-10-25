"""
模块名称: graph_community_detection.py
功能说明: 
    基于用户-商品评论数据，构建二部图并投影为用户关系网络。
    通过过滤超级节点（热门商品）和弱连接关系进行降噪。
    使用 Louvain 算法进行社区发现（团伙检测），并为每个账号输出图可疑分数。
"""

import networkx as nx
import community as community_louvain
from networkx.algorithms import bipartite
import pandas as pd
import os

# ================= 配置参数区 =================
# 数据源路径与列名映射（根据实际数据字典调整）
FILE_PATH = 'data/raw/YelpChi_reviews.csv' 
USER_COL = 'user_id'      
ITEM_COL = 'product_id'  

# 算法超参数
MAX_ITEM_DEGREE = 500  # 放宽阈值，只剔除极少数超级爆款
MIN_WEIGHT = 2        # 边权重阈值：两个用户至少共同评论过2个商品才建立连接
# GROUP_DIVISOR = 20.0  # 分数归一化分母：团伙人数达到20人即得满分1.0
# ==============================================

def run_graph_analysis():
    # 1. 读取清洗后的评论明细数据
    print(f"正在读取真实数据: {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)

    # 2. 构建用户-商品二部图
    print("正在构建二部图...")
    B = nx.Graph()
    # 利用 zip 快速将用户列和商品列配对，批量添加边
    B.add_edges_from(zip(df[USER_COL], df[ITEM_COL]))

    # 3. 降噪处理：过滤超级节点（网红商品）
    # 目的：防止大量普通用户因为偶然评价了同一家热门店铺而形成虚假的“巨型社区”
    items = set(df[ITEM_COL])
    print("正在过滤超级热门商品...")
    super_nodes = [node for node in items if B.degree(node) > MAX_ITEM_DEGREE]
    print(f"发现 {len(super_nodes)} 个超级热门商品，已将其移除。")
    B.remove_nodes_from(super_nodes)

    # 打印剩下的商品节点数，证明图没被删空
    print(f"过滤后剩余商品节点数: {len([n for n in B.nodes if n in items])}")

    # 4. 二部图投影：将 用户-商品 图转换为 用户-用户 图
    # 筛选出过滤超级节点后，仍在图中保留的用户
    users = [u for u in df[USER_COL].unique() if B.has_node(u)]
    
    print("正在投影用户关系网...")
    # 投影规则：如果两个用户共同评价了同一个商品，则他们之间连一条边，权重为共同评价的商品数
    G_user = bipartite.weighted_projected_graph(B, users)

    # 5. 进一步降噪：移除弱连接（偶然交集）
    edges_to_remove = [(u, v) for u, v, d in G_user.edges(data=True) if d['weight'] < MIN_WEIGHT]
    G_user.remove_edges_from(edges_to_remove)

    print(f"=== 过滤后，用户关系网边数: {G_user.number_of_edges()} ===")

    # 6. 社区发现（团伙检测）与打分
    if G_user.number_of_edges() > 0:
        print("正在运行 Louvain 算法找团伙...")
        # best_partition 会返回一个字典 {user_id: group_id}
        partition = community_louvain.best_partition(G_user, random_state=42)
        
        # 将结果转为 DataFrame 便于后续处理
        result_df = pd.DataFrame(list(partition.items()), columns=['user_id', 'group_id'])
        
        # 统计每个团伙的规模（人数）
        group_sizes = result_df['group_id'].value_counts().to_dict()
        
        # 计算账号级的连续指标（节点度数 / 团伙总人数）
        # 获取每个账号在全局图中的度数（连接了多少人）
        user_degrees = dict(G_user.degree())
        # 映射到表格中
        result_df['degree'] = result_df['user_id'].map(user_degrees).fillna(0)
        result_df['group_size'] = result_df['group_id'].map(group_sizes).fillna(1)
        
        # 计算分数：你在团伙里认识的人越多，分数越高。
        # 核心头目（度数最大）接近 1.0，边缘小弟（只有1条边）分数极低，单人团伙为 0。
        result_df['graph_score'] = result_df['degree'] / result_df['group_size']

        # 把所有分数缩放到 0~1 之间，解决大于 1 的问题！
        result_df['graph_score'] = result_df['graph_score'] / result_df['graph_score'].max()
        
        # 清理不需要的中间列
        result_df = result_df.drop(columns=['degree', 'group_size'])

        # 按分数降序排列，最可疑的排在最前
        result_df = result_df.sort_values(by='graph_score', ascending=False)

        
        # 7. 保存交付结果
        os.makedirs('data/processed', exist_ok=True)
        save_path = 'data/processed/graph_scores.csv' 
        result_df.to_csv(save_path, index=False)
        
        print(f"\n✅ 团伙划分完毕！共发现 {len(group_sizes)} 个团伙。")
        print(f"✅ 账号图分数已保存至: {save_path}")
        
        # 8. 打印排行前 5 的核心团伙供观察
        print("\n--- 规模最大的前 5 个团伙 ---")
        top_5_groups = result_df['group_id'].value_counts().head(5)
        for gid, size in top_5_groups.items():
            print(f"团伙 {gid}: 共有 {size} 人")
    else:
        print("数据太稀疏，过滤后无法形成有效团伙。")

if __name__ == "__main__":
    run_graph_analysis()