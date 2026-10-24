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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，防止中文显示为方块
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ================= 配置参数区 =================
FILE_PATH = 'data/raw/YelpChi_reviews.csv' 
SCORES_PATH = 'results/graph_scores.csv'
TARGET_GROUP_ID = 168  # 指定需要可视化的目标团伙ID（通常选择规模较大、结构清晰的团伙）
MIN_EDGE_WEIGHT = 2    # 绘图时仅保留共同评论数>=2的强关系边
# ==============================================

def draw_target_group():
    # 1. 读取全量评论数据与算法打分结果
    df = pd.read_csv(FILE_PATH)
    scores_df = pd.read_csv(SCORES_PATH)

    # 2. 提取目标团伙的嫌疑账号集合
    target_users = set(scores_df[scores_df['group_id'] == TARGET_GROUP_ID]['user_id'])
    print(f"正在提取团伙 {TARGET_GROUP_ID} 的网络结构，共 {len(target_users)} 个账号...")

    # 3. 构建局部子图
    # 仅筛选出这些目标用户的评论记录，避免加载全局大图导致卡顿
    df_target = df[df['user_id'].isin(target_users)]
    
    B = nx.Graph()
    B.add_edges_from(zip(df_target['user_id'], df_target['product_id']))

    # 投影成局部用户网络
    G_user = bipartite.weighted_projected_graph(B, list(target_users))

    # 4. 过滤细枝末节，保留强关联边用于美观展示
    edges_to_keep = [(u, v) for u, v, d in G_user.edges(data=True) if d['weight'] >= MIN_EDGE_WEIGHT]
    G_final = G_user.edge_subgraph(edges_to_keep).copy()

    # 5. 网络图可视化绘制
    plt.figure(figsize=(10, 8))
    plt.title(f"水军团伙 {TARGET_GROUP_ID} 网络结构 (团伙总人数: {len(target_users)})", fontsize=16, fontweight='bold')

    # 使用弹簧布局算法（spring_layout）计算节点位置
    # 原理：像弹簧一样根据连接权重将节点拉开，使图形分布更加自然（seed=42 保证每次生成的形状一致，便于复现）
    pos = nx.spring_layout(G_final, k=0.5, seed=42)

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
                              markersize=12, label='核心成员 (连接数最多)')  # 改这里
    # 创建边缘蓝点的图例
    regular_node = mlines.Line2D([], [], color='#4da6ff', marker='o', linestyle='None',
                                markersize=8, label='普通成员')  # 改这里
    # 创建连线的图例（把 >= 换成普通符号，防乱码）
    edge_line = mlines.Line2D([], [], color='gray', linewidth=1.5,
                             label='强关联 (共同评论 >= 2 个商品)') 
    
    plt.legend(handles=[center_node, regular_node, edge_line], loc='lower right', fontsize=12)

    # 6. 保存高清图片至 results 文件夹
    save_img_path = 'results/graph_group_network.png'
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    print(f"✅ 团伙网络图已成功保存至: {save_img_path}")

if __name__ == "__main__":
    draw_target_group()