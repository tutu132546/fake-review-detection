\# YelpChi 可解释水军检测：数据字典与实验说明



\## 一、 交付文件清单

1\. \*\*`YelpChi\_reviews.csv`\*\*: 真实评论级原始数据（共 67,395 条）。包含时间戳和评分，供队友计算“日均发评数”等真实特征。

2\. \*\*`yelpchi\_1000\_accounts.csv`\*\*: 账号级基准样本（共 1000 条）。用于模型基线训练与初步评估。

3\. \*\*`clean\_accounts.parquet`\*\*: 全量干净账号底表。完美融合了真实标签、图结构和匿名特征。

4\. \*\*`augmented\_accounts.parquet`\*\*: 水军注入后的全量底表（含 300 个注水账号）。

5\. \*\*`injection\_log.json`\*\*: 水军注入 Ground Truth 明细。

6\. \*\*`scaler\_params.json`\*\*: 注入前计算的全量特征均值和标准差，用于 Z-score 严格对齐。



\## 二、 字段说明

\### 1. 基础字段 (`clean\_accounts.parquet`)

\* `user\_id`: 账号唯一标识。

\* `n\_reviews`: 该账号的真实评论总数（基于 metadata 统计）。

\* `account\_is\_fake`: 账号级标签（1 = 曾发过水军评论，0 = 正常账号）。

\* `is\_injected`: 仅存在于 augmented 文件中（1 = 人工注入，0 = 原生）。



\### 2. 图拓扑特征 (基于 `net\_rur` 提取)

\* `max\_cc\_size`: 最大连通分量大小（衡量抱团攻击规模）。

\* `degree\_max`: 最大节点度数（衡量交互密集度）。



\### 3. 32 维聚合特征

\* `f0\_mean` \~ `f31\_mean`: SpEagle 元数据特征的账号级均值。

\* `f0\_std` \~ `f31\_std`: 32 维特征标准差。

\* `f0\_max` \~ `f31\_max`: 32 维特征最大值。

\* `f\*\_z`: 对应的全局 Z-score 标准化值。



\### 4. 注入日志 (`injection\_log.json`)

\* `user\_id`: 被注入账号 ID。

\* `injection\_reason`: 注入模板（A：隐蔽突发 / B：团伙抱团）。

\* `modified\_features`: 包含 `original`（原值）和 `mutated`（注入后值）。用于检验 DCFO 解释的准确性。



\## 三、 水军注入与实验设计

\* \*\*注入规模\*\*：从正常号中抽取 300 个，占比约 1%。

\* \*\*注入逻辑\*\*：

&#x20; \* \*\*模板 A（隐蔽突发）\*\*：`f0\_mean` 增加 1.5\~3.0，`f2\_std` 压缩至原来的 10%\~30%（模拟评论高度同质化）。

&#x20; \* \*\*模板 B（团伙抱团）\*\*：`max\_cc\_size` 乘 1.5\~2.5 并加随机数，`degree\_max` 增加 10\~30（模拟拉帮结派）。

\* \*\*实验闭环\*\*：任何模型若判定注入号异常，其给出的特征重要性或反事实解释，均可直接与 `injection\_log.json` 中的 `original/mutated` 进行比对，验证解释的真实性。



\## 四、 Z-score 说明

所有 32 维及图特征的标准化均基于 `clean\_accounts.parquet` 全量计算。`augmented\_accounts.parquet` 中的 `\_z` 列严格复用了 `scaler\_params.json` 的均值与方差，确保注入前后特征分布处于同一尺度空间。

