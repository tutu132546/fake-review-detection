# C2：DCFO ground truth 对齐（v2 日志 `feature_delta_sigma`）

> 数据源：`data/processed/injection_log_v2.json`（重跑 step3 后的唯一真源，300 个账号 / 2,791 条注入事件）。
> 反事实解释的 ground truth 是"注入前后 7 维特征的 σ 单位增量"，本报告校验它与磁盘表逐位自洽，并复核模板主导缺陷排序。

## 1. 逐账号重算比对

| 检查 | 结果 |
|---|---:|
| 比对单元（300 账号 × 7 特征） | 2,100 |
| 最大 \|日志 − 重算\| | 0.000050 |
| 超出 1e-3 的单元 | 0 |
| NaN 状态不一致的单元 | 0 |
| `modified_features` 漏记（表变但日志未记） | 0 |
| `modified_features` 多记（表未变但日志列出） | 0 |

日志的 `delta_sigma` 落盘保留 4 位小数，重算值为全精度；最大差 5.00e-05 落在舍入误差内，判定 **一致**。

## 2. 模板级 dz 均值（与 data_dictionary §7 对照）

| 特征 | A_Time_Burst | B_Rating_Polarization | C_Stealth_Mix |
|---|---:|---:|---:|
| n_reviews | +4.35 | +4.60 | +4.32 |
| active_days | +0.00 | +0.00 | +0.00 |
| daily_avg_reviews | +0.48 | +0.32 | +0.14 |
| max_daily_reviews | +18.61 | +0.37 | +6.71 |
| burstiness | +2.34 | -0.21 | +0.35 |
| rating_deviation | +0.48 | +0.36 | +0.41 |
| extreme_ratio | +0.76 | +0.78 | +0.73 |

模板特异性排序（已剔除公共效应 n_reviews/daily_avg_reviews；相差 ≤0.05 视为并列）：

| 模板 | 期望顺序（§7） | 实际数值 | 判定 |
|---|---|---|---|
| A_Time_Burst | max_daily_reviews > burstiness > extreme_ratio | max_daily_reviews +18.61 > burstiness +2.34 > extreme_ratio +0.76 | PASS |
| B_Rating_Polarization | extreme_ratio > rating_deviation > max_daily_reviews | extreme_ratio +0.78 > max_daily_reviews +0.37 > rating_deviation +0.36 | PASS |
| C_Stealth_Mix | max_daily_reviews > extreme_ratio > rating_deviation | max_daily_reviews +6.71 > extreme_ratio +0.73 > rating_deviation +0.41 | PASS |

## 3. 方向判据（Issue #1 可证伪项）

| 判据 | 结果 |
|---|---|
| A burstiness dz > 0 | PASS |
| C burstiness dz > 0 | PASS |
| B burstiness 近似未动（|dz| < 0.5） | PASS |
| 全模板 active_days dz = 0 | PASS |

- 总体判定：**PASS**（逐账号一致 ∧ 排序一致 ∧ 方向判据全过）。
- 用途：解释方法（如 DCFO / SHAP 类反事实）应恢复出与上表同量级的 dz；"是否恢复出主导缺陷" 是可证伪指标，不以"解释看起来合理"代替。

## 复现

```bash
python src/models/dcfo_ground_truth.py
```

环境：Python 3.14.5；numpy=2.5.1；pandas=3.0.5。
