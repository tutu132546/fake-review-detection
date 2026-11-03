import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd

# 配置
FILE_PATH = 'data/raw/YelpChi_reviews.csv'
USER_COL = 'user_id'
ITEM_COL = 'product_id'
MIN_WEIGHT = 2
thresholds = [50, 100, 200, 500, 999999] # 999999代表不过滤

print("正在读取数据并构建初始二部图...")
df = pd.read_csv(FILE_PATH)
B_all = nx.Graph()
B_all.add_edges_from(zip(df[USER_COL], df[ITEM_COL]))
items = set(df[ITEM_COL])
all_users = set(df[USER_COL])

results = []

for MAX_ITEM_DEGREE in thresholds:
    print(f"正在测试阈值: {MAX_ITEM_DEGREE if MAX_ITEM_DEGREE < 999999 else '不过滤'}...")
    
    # 过滤商品
    super_nodes = [node for node in items if B_all.degree(node) > MAX_ITEM_DEGREE]
    B = B_all.copy()
    B.remove_nodes_from(super_nodes)
    
    # 投影用户网络
    valid_users = [u for u in all_users if B.has_node(u)]
    G_user = bipartite.weighted_projected_graph(B, valid_users)
    
    # 过滤弱连接
    edges_to_remove = [(u, v) for u, v, d in G_user.edges(data=True) if d['weight'] < MIN_WEIGHT]
    G_user.remove_edges_from(edges_to_remove)
    
    # 统计指标
    removed_items_count = len(super_nodes)
    # 计算有边相连的用户数量
    users_with_edges = set(u for edge in G_user.edges() for u in edge)
    edge_ratio = len(users_with_edges) / len(all_users)
    
    # 找最大连通分量（团伙）大小
    if len(G_user.nodes) > 0:
        components = list(nx.connected_components(G_user))
        max_component_size = max(len(c) for c in components)
    else:
        max_component_size = 0
        
    results.append({
        '阈值 (MAX_ITEM_DEGREE)': MAX_ITEM_DEGREE if MAX_ITEM_DEGREE < 999999 else '不过滤',
        '被剔除商品数': removed_items_count,
        '有连线的账号占比': f"{edge_ratio * 100:.2f}%",
        '最大连通子图人数': max_component_size
    })

# 输出为 Markdown 表格
result_df = pd.DataFrame(results)
md_table = result_df.to_markdown(index=False)
print("\n=== 阈值敏感性分析结果 ===")
print(md_table)

# 保存为 Markdown 文件
with open('results/sensitivity_analysis.md', 'w', encoding='utf-8') as f:
    f.write("# 图算法阈值敏感性分析\n\n")
    f.write("为了评估商品过滤阈值 (`MAX_ITEM_DEGREE`) 对图网络结构的影响，我们测试了不同阈值下的网络状态。结果如下：\n\n")
    f.write(md_table)
print("\n✅ 表格已保存至: results/sensitivity_analysis.md")