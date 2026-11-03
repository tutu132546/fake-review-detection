# 水军行为定义与特征映射（B 成员交付，完成度约 80%）

> 范围：7 个账号级行为特征、5 类水军行为、YelpChi 侧可观测性。
> 未含：FraudSquad 三种 AI 生成方式的可观测性实测——依赖叶歆澜拆分后的 3 份统一数据，统一标 `[待核]`（见 §5）。
> 标注：`[已核]` 有公开文献出处；`[待核]` 需人工核对原文或需数据实测。

## 1. 判定口径

- "行为"必须能落到一个**可计算特征 + 方向**，否则只算描述，不进第 4 章。
- 方向（升高/降低）必须可在 YelpChi 上实测；不能用"一般来说会这样"代替。
- 每个断言要么给文献，要么给本仓库实测数字；两者都没有的标 `[待核]`。

## 2. 七个行为特征

| # | 特征 | 定义/公式 | 水军行为假设 | 预期方向 | 出处 |
|---|---|---|---|---|---|
| 1 | `n_reviews` | 账号评论总数 | 刷评会推高账号产出量；但"养号"型水军反而偏低 | 双向（原生 YelpChi 假号偏低） | Jindal & Liu, *Opinion Spam and Analysis*, WSDM 2008 `[已核]` |
| 2 | `active_days` | `(末次日期 − 首次日期).days + 1` | 突击型账号活跃期短，"养号"型活跃期长 | 原生假号偏低 | Mukherjee et al., *Spotting Fake Reviewer Groups in Consumer Reviews*, WWW 2013 `[已核]` |
| 3 | `daily_avg_reviews` | `n_reviews / active_days` | 集中投放推高日均 | 升高 | Lim et al., *Detecting Product Review Spammers Using Rating Behaviors*, CIKM 2010 `[已核]` |
| 4 | `max_daily_reviews` | 单日最大评论数 | 脚本化刷评呈现单日爆发 | 升高 | Xie et al., *Towards Online Review Spam Detection*, WWW 2012 `[已核]` |
| 5 | `burstiness` | 相邻评论间隔的变异系数 `std(gaps)/mean(gaps)`；n<3 或全部同日为 NaN | 集中爆发使间隔序列混入近 0 间隔，波动增大 | 升高（已实测 +1.90 / +2.08 dz） | Goh & Barabási, *Burstiness and Memory in Complex Systems*, EPL 2008 `[已核]`（定义）；评论场景应用 `[待核]` |
| 6 | `rating_deviation` | `mean(评分 − 商品均分)`，∈[-4,4] | 水军集中打极端分，偏离商品均分 | 升高 | Lim et al. 2010 `[已核]`；Rayana & Akoglu, *Collective Opinion Spam Detection*, KDD 2015 `[已核]` |
| 7 | `extreme_ratio` | `(1 星或 5 星评论数)/总数`，∈[0,1] | 刷评偏好极端评分 | 升高 | Mukherjee et al. 2013 `[已核]` |

### 2.1 YelpChi 实测（原生标签，38,063 账号）

| 特征 | 单特征 AUC（仅原生） | 解读 |
|---|---:|---|
| n_reviews_z | 0.391 | 原生假号"少评" |
| active_days_z | 0.396 | 原生假号"短活" |
| daily_avg_reviews_z | 0.593 | 高日均 |
| max_daily_reviews_z | 0.473 | 弱信号 |
| burstiness_z | 0.504 | 原生上几乎无区分度 |
| rating_deviation_z | 0.499 | 原生上近随机 |
| extreme_ratio_z | 0.571 | 极端评分倾向 |
| burstiness_missing | 0.564 | 86.5% 缺失本身即信号（多为低评论数账号） |

结论：单特征 0.39–0.59，**不足以做阈值检测**，必须多变量（LOF/iForest）+ 多信号融合。`[已算]`

## 3. 五类水军行为

| # | 行为 | 定义 | 可观测特征（本仓库） | YelpChi 侧证据 | 判定 |
|---|---|---|---|---|---|
| B1 | 单日爆发型 | 短时间内集中投放大量评论 | `max_daily_reviews↑`、`daily_avg_reviews↑`、`burstiness↑` | 注入模板 A 实测 `max_daily +15.37z` | `[已核]`（注入侧） |
| B2 | 评分极化型 | 只打 1 星或 5 星，系统性偏离商家均分 | `extreme_ratio↑`、`rating_deviation↑` | 注入模板 B 实测 `rating_deviation +0.86z` | `[已核]`（注入侧） |
| B3 | 隐蔽混评型 | 混入正常评论、节奏接近真实用户，仅评分偏移 | `burstiness` 与常人接近、`extreme_ratio` 小幅升高 | 注入模板 C 实测 `burstiness +2.08z`、`rating_deviation +0.97z` | `[已核]`（注入侧） |
| B4 | 账号接管型（被 LLM 接管） | 原真实账号被接管后追加 AI 生成评论 | 账号历史正常 + 近期突变（需时间切分） | 本仓库无此标签，FraudSquad 提供 | `[待核]` |
| B5 | AI 批量生成型 | 同一模型批量生成评论，风格/结构高度一致 | 文本层面（本仓库无文本特征）；账号层面仅见 `n_reviews↑` | FraudSquad×3 提供 | `[待核]` |

### 3.1 与注入模板的对应

| 模板 | 模拟行为 | 账户数 | 实际修改特征 |
|---|---|---:|---|
| A_Time_Burst | B1 单日爆发 | 100 | n_reviews, max_daily_reviews, burstiness, extreme_ratio, daily_avg_reviews |
| B_Rating_Polarization | B2 评分极化 | 100 | n_reviews, rating_deviation, extreme_ratio, daily_avg_reviews（+2 个账号补 max_daily） |
| C_Stealth_Mix | B3 隐蔽混评 | 100 | n_reviews, max_daily_reviews, burstiness, rating_deviation, extreme_ratio, daily_avg_reviews |

## 4. 特征与行为的多对多关系（防止误读）

- `burstiness↑` 不是"水军专属"：真实用户中的旅行/活动密集期也会升高；只能用**同账号前后对比**（配对 dz）判定，不能用组间均值。
- `n_reviews↑` 在注入实验中必然升高（注入定义），但原生假号反而偏低——**两个结论方向相反，不能混用**。
- `active_days` 在多数注入中不动，任何"注入组 active_days 偏高"的组间差异都是**选池偏差**（注入池要求 `n_reviews ≥ 5`）。

## 5. `[待核]` 清单（FraudSquad 剩余 20%）

| 待核项 | 为什么现在不能定 | 依赖 | 核对人 | 截止 |
|---|---|---|---|---|
| 每账号生成评论恰好 2 条 | 需拆分后的 3 份统一数据实测 | 叶歆澜 `unified/fraudsquad_*.parquet` | 杨蔚艳 | 11.6 |
| 生成评论全 5 星 | 同上 | 同上 | 杨蔚艳 | 11.6 |
| 生成评论在商品首评后 5 天内 | 同上 | 同上 | 杨蔚艳 | 11.6 |
| 三种生成方式（Llama3/Qwen2/Qwen-DSR1）在 7 特征上的可观测差异 | 需分模型建表后对比 | `build_all.py` 产出 3 表 | 杨蔚艳 | 11.7 |
| "burstiness 用于评论水军"的原始文献页码 | 只确认了定义文献（Goh & Barabási 2008），评论场景应用需回查 | 文献 PDF | 杨蔚艳 | 11.6 |

> 完成标准：上表 5 项全部有结论后，本文档补入实测段并删除 `[待核]` 标记，即达 100%。
