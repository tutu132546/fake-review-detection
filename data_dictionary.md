# YelpChi 账号级数据集说明（Data Dictionary）

## 1. 基础字段
* `user_id`: 匿名化处理后的用户唯一 ID
* `n_reviews`: 该账号在数据集中的评论总数
* `account_is_fake`: 标签（1 表示曾被判定为发布欺诈评论，0 表示正常）
* `is_injected`: 标记是否为人工注入的水军（1=注入，0=原生数据）

## 2. 行为特征 (32维原始特征聚合)
包含 `f0` 到 `f31` 的均值 (`_mean`)、标准差 (`_std`) 和最大值 (`_max`)。
* `f*_mean`: 反映账号评论行为的整体倾向（如打分分布）。
* `f*_std`: 反映行为波动性（水军往往具有高度一致的重复行为）。
* `f*_max`: 反映极端行为的峰值。

## 3. 图拓扑特征 (基于 R-U-R 同用户网络)
* `n_components`: 该账号的评论在图中的连通分量数量（反映账号评论的碎片化程度）。
* `max_cc_size`: 最大连通分量包含的节点数（**水军团伙抱团的核心指标**）。
* `degree_mean` / `degree_std` / `degree_max`: 评论节点的连接度数统计（反映社交交互的密集度）。

## 4. 注入实验日志说明 (`injection_log.json`)
该文件记录了所有注入水军的真实 Ground Truth。
* `user_id`: 被注入的账号 ID。
* `injection_reason`: 注入的攻击模板（A: 隐蔽突发, B: 暴力刷榜, C: 深度伪装）。
* `modified_features`: 记录被篡改特征的具体变化，包含 `original`（原始值）与 `mutated`（注入后值）。这是验证 DCFO 反事实解释是否准确指向真实原因的核心依据。