# 派生视图 `yelpzip_scale359k`：YelpZip 规模对齐子样本

> **用途**：YelpNYC 缺失时的替代性规模敏感性检查。
> **它不是 YelpNYC，也不能当 YelpNYC 用**——只对齐评论总量这一个维度，
> 商户粒度、城市覆盖、时间跨度都不同。详见 [`docs/data/yelpnyc_fallback.md`](../docs/data/yelpnyc_fallback.md) §3 方案 A。

## 1. 抽样设计

- 源数据：YelpZip，608,458 条 / 5,044 家商户，假评占比 13.22%
- 目标量级：359,052 条（= YelpNYC 官方公布规模（schema.DATASET_SPECS['yelpnyc'].expected_rows））
- 分层：按商户评论数切 10 个分位层，层内按比例配额、层内随机取商户
- 重复：5 个随机种子，报告区间而非单点

## 2. 各随机种子的结果

| 种子 | 评论数 | 与目标的偏差 | 商户数 | 用户数 | 假评占比 | 文件 |
|---|---|---|---|---|---|---|
| 0 | 359,725 | +0.187% | 3,002 | 180,495 | 13.09% | `yelpzip_scale359k_seed0.parquet` |
| 1 | 359,431 | +0.106% | 2,972 | 183,070 | 13.30% | `yelpzip_scale359k_seed1.parquet` |
| 2 | 360,160 | +0.309% | 2,997 | 183,601 | 13.28% | `yelpzip_scale359k_seed2.parquet` |
| 3 | 360,242 | +0.331% | 2,995 | 183,502 | 13.37% | `yelpzip_scale359k_seed3.parquet` |
| 4 | 359,457 | +0.113% | 2,971 | 180,402 | 13.28% | `yelpzip_scale359k_seed4.parquet` |

**汇总**：359,431 ~ 360,242 条，均值 359,803 条；假评占比均值 13.26%（源数据 13.22%，偏差 < 0.05 个百分点，说明分层抽样没有扭曲标签分布）。

## 3. 怎么用、不能怎么用

**可以**：把 5 个种子各跑一遍方法，报告 AUC / F1 的均值与 95% 区间，回答「评论总量降到 36 万量级时结论是否稳定」。
**不可以**：① 把它写进主对比表冒充第 4 个公开数据集；② 声称它代表纽约市；③ 拿它和 YelpZip 全量的差异解释为「城市差异」。

```python
import pandas as pd
df = pd.read_parquet('data/processed/derived/yelpzip_scale359k_seed0.parquet')
```
