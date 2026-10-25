"""
模块名称: plot_group_network.py
功能说明: 
    读取社区发现结果 (graph_scores.csv)，提取特定团伙的账号。
    对这些目标账号重建局部二部图，并绘制网络可视化图，用于论文展示与案例分析。
"""

import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']  # 增加字体回退列表，兼容 Mac 和非 Windows 系统
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ================= 配置参数区 =================
FILE_PATH = 'data/raw/YelpChi_reviews.csv' 
SCORES_PATH = 'data/processed/graph_scores.csv'
MIN_EDGE_WEIGHT = 2    # 绘图时仅保留共同评论数>=2的强关系边

# 全局读取一次数据
print("正在读取数据与分数表...")
df = pd.read_csv(FILE_PATH)
scores_df = pd.read_csv(SCORES_PATH)

# 自动寻找人数最多的团伙
group_sizes = scores_df['group_id'].value_counts()
TARGET_GROUP_ID = group_sizes.index[0]  
# ==============================================

def draw_target_group():
    # 1. 提取目标团伙的嫌疑账号集合
    target_users = set(scores_df[scores_df['group_id'] == TARGET_GROUP_ID]['user_id'])
    print(f"正在提取团伙 {TARGET_GROUP_ID} 的网络结构，共 {len(target_users)} 个账号...")

    # 2. 构建局部子图
    # 仅筛选出这些目标用户的评论记录，避免加载全局大图导致卡顿
    df_target = df[df['user_id'].isin(target_users)]
    
    B = nx.Graph()
    B.add_edges_from(zip(df_target['user_id'], df_target['product_id']))

    # 投影成局部用户网络
    G_user = bipartite.weighted_projected_graph(B, list(target_users))

    # 3. 过滤细枝末节，保留强关联边用于美观展示
    edges_to_keep = [(u, v) for u, v, d in G_user.edges(data=True) if d['weight'] >= MIN_EDGE_WEIGHT]
    G_final = G_user.edge_subgraph(edges_to_keep).copy()

    # 从大网络中，提取最核心的“头目”及其关联最紧密的 Top 30 邻居
    if len(G_final.nodes) > 0:
        core_node = max(dict(G_final.degree()).items(), key=lambda x: x[1])[0]
        neighbors = list(G_final.neighbors(core_node))
        
        # 1. 提取 Top 30 紧密邻居
        neighbors_sorted = sorted(neighbors, key=lambda x: G_final[core_node][x]['weight'], reverse=True)
        top_neighbors = neighbors_sorted[:30] 
        nodes_top = [core_node] + top_neighbors
        G_top = G_final.subgraph(nodes_top).copy()
        
        # 2. 公平对照组：随机选一个“非核心”节点，取它权重最高的 Top 30 邻居
        import random
        random.seed(42)
        all_nodes = list(G_final.nodes)
        # 排除掉核心节点
        non_core_nodes = [n for n in all_nodes if n != core_node]
        
        if len(non_core_nodes) > 0:
            random_node = random.choice(non_core_nodes)
            random_neighbors = list(G_final.neighbors(random_node))
            # 按权重给邻居排序
            random_neighbors_sorted = sorted(random_neighbors, key=lambda x: G_final[random_node][x]['weight'], reverse=True)
            top_random_neighbors = random_neighbors_sorted[:30]
            
            nodes_random = [random_node] + top_random_neighbors
            G_random = G_final.subgraph(nodes_random).copy()
        else:
            G_random = G_final.copy() # 兜底
        
        # 我们把原来的 G_final 替换成 G_top 画第一张图
        G_final = G_top

    # ---------------- 图1：绘制 Top 30 核心邻居网络 ----------------
    pos_top = nx.spring_layout(G_top, k=1.0, seed=42)
    plt.figure(figsize=(10, 8))
    plt.title(f"高可疑候选团伙 (核心与Top30邻居) 网络结构", fontsize=16, fontweight='bold')

    degrees = dict(G_top.degree())
    max_degree = max(degrees.values()) 
    
    node_sizes = [1200 if deg == max_degree else 300 for deg in degrees.values()]
    node_colors = ['red' if deg == max_degree else '#4da6ff' for deg in degrees.values()]

    nx.draw_networkx_nodes(G_top, pos_top, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G_top, pos_top, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G_top, pos_top, font_size=8)

    plt.axis('off')
    
    # 图例
    center_node = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=12, label='核心节点')
    regular_node = mlines.Line2D([], [], color='#4da6ff', marker='o', linestyle='None', markersize=8, label='边缘节点')
    edge_line = mlines.Line2D([], [], color='gray', linewidth=1.5, label='强关联 (>= 2 个商品)') 
    
    plt.legend(handles=[center_node, regular_node, edge_line], loc='lower right', bbox_to_anchor=(1.1, 0.0), fontsize=12)

    save_img_path_top = 'results/graph_group_network_top30.png'
    plt.savefig(save_img_path_top, dpi=300, bbox_inches='tight')
    print(f"✅ Top30网络图已保存至: {save_img_path_top}")
    plt.close() # 画完必须关掉，不然两张图会重叠

    # ---------------- 图2：绘制随机对照组网络 ----------------
    pos_rand = nx.spring_layout(G_random, k=1.0, seed=42)
    plt.figure(figsize=(10, 8))
    plt.title(f"随机对照 (非核心节点及其Top30邻居) 网络结构", fontsize=16, fontweight='bold')

    # 随机图没有明显的中心节点，所以全是蓝色
    nx.draw_networkx_nodes(G_random, pos_rand, node_color='#4da6ff', node_size=300, alpha=0.8)
    nx.draw_networkx_edges(G_random, pos_rand, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G_random, pos_rand, font_size=8)

    plt.axis('off')
    plt.legend(handles=[regular_node, edge_line], loc='lower right', bbox_to_anchor=(1.1, 0.0), fontsize=12)

    save_img_path_rand = 'results/graph_group_network_random.png'
    plt.savefig(save_img_path_rand, dpi=300, bbox_inches='tight')
    print(f"✅ 随机对照组网络图已保存至: {save_img_path_rand}")
    plt.close()

if __name__ == "__main__":
    draw_target_group()