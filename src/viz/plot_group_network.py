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
SCORES_PATH = 'results/graph_scores.csv'
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
        
        # 🌟 新增逻辑：按边权重排序，只取前 30 个最紧密的邻居
        neighbors_sorted = sorted(neighbors, key=lambda x: G_final[core_node][x]['weight'], reverse=True)
        top_neighbors = neighbors_sorted[:30] 
        
        nodes_to_keep = [core_node] + top_neighbors
        G_final = G_final.subgraph(nodes_to_keep).copy()
        # 更新一下标题提示（可选）
        print(f"为了美观展示，只提取了核心节点及其 Top {len(top_neighbors)} 个最紧密邻居。")
        
    # 重新计算位置
    pos = nx.spring_layout(G_final, k=1.0, seed=42) # 把 k 调大一点，让点散开

    # 4. 网络图可视化绘制
    plt.figure(figsize=(10, 8))
    plt.title(f"高可疑候选团伙 {TARGET_GROUP_ID} 网络结构 (团伙总人数: {len(target_users)})", fontsize=16, fontweight='bold')

    # 根据节点的连接数（度数）动态分配大小和颜色
    degrees = dict(G_final.degree())
    max_degree = max(degrees.values()) # 找到连接数最多的那个节点（也就是中心头目）
    
    # 计算大小：中心节点 1200，边缘节点 300
    node_sizes = [1200 if deg == max_degree else 300 for deg in degrees.values()]
    # 计算颜色：中心节点是红色（red），边缘节点是浅蓝色（#4da6ff）
    node_colors = ['red' if deg == max_degree else '#4da6ff' for deg in degrees.values()]

    # 绘制节点（使用上面算好的大小和颜色）
    nx.draw_networkx_nodes(G_final, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    # 绘制边（灰色线条）
    nx.draw_networkx_edges(G_final, pos, edge_color='gray', alpha=0.5)
    # 绘制节点标签（账号ID）
    nx.draw_networkx_labels(G_final, pos, font_size=8)

    # 隐藏坐标轴边框
    plt.axis('off')

    # 添加图例 (Legend)
    # 创建中心红点的图例
    center_node = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                              markersize=12, label='核心节点 (连接数最多)')  # 改这里
    # 创建边缘蓝点的图例
    regular_node = mlines.Line2D([], [], color='#4da6ff', marker='o', linestyle='None',
                                markersize=8, label='边缘节点')  # 改这里
    # 创建连线的图例（把 >= 换成普通符号，防乱码）
    edge_line = mlines.Line2D([], [], color='gray', linewidth=1.5,
                             label='强关联 (共同评论 >= 2 个商品)') 
    
    # 图例放回右下角，稍微留一点边距
    plt.legend(handles=[center_node, regular_node, edge_line],
           loc='lower right',
           bbox_to_anchor=(1.1, 0.0), # 把锚点往右移出画布一点点
           fontsize=12)

    # 6. 保存高清图片至 results 文件夹
    save_img_path = 'results/graph_group_network.png'
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    print(f"✅ 团伙网络图已成功保存至: {save_img_path}")

if __name__ == "__main__":
    draw_target_group()