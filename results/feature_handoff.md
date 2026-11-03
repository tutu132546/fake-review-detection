# 特征表交接说明（B → C/D/E）

> 状态：**5 张表全部交付**（YelpChi + YelpZip + FraudSquad×3；YelpNYC 保持 `MISSING` 占位）。
> 生成日期：2026-11-03（B1 重跑 step3 后重算，v2 日志 2,791 事件 / 70,186 行）。

## 1. 已交付表（[已算]，规模取自各 manifest）

| 数据集 | 路径 | 账号数 × 22 列 | 人群 | burstiness 缺失率 | manifest |
|---|---|---:|---|---:|---|
| YelpChi | `data/processed/features/feature_table_yelpchi.csv` | 38,063 | native 37,763 / self_injected 300 | 86.48% | `results/feature_manifests/feature_table_yelpchi.json` |
| YelpZip | `data/processed/features/feature_table_yelpzip.csv` | 260,239 | native 260,239 | 82.21% | `results/feature_manifests/feature_table_yelpzip.json` |
| FraudSquad-Llama3 | `data/processed/features/feature_table_fraudsquad_llama3.csv` | 8,092 | native 6,842 / hijacked_llm 1,250 | 2.14% | `results/feature_manifests/feature_table_fraudsquad_llama3.json` |
| FraudSquad-Qwen2 | `data/processed/features/feature_table_fraudsquad_qwen2.csv` | 8,092 | native 6,842 / hijacked_llm 1,250 | 2.14% | `results/feature_manifests/feature_table_fraudsquad_qwen2.json` |
| FraudSquad-Qwen-DSR1 | `data/processed/features/feature_table_fraudsquad_qwen_dsr1.csv` | 8,092 | native 6,842 / hijacked_llm 1,250 | 2.20% | `results/feature_manifests/feature_table_fraudsquad_qwen_dsr1.json` |

- YelpChi 兼容副本：`data/processed/feature_table.csv`（与 yelpchi 表同内容，供旧下游）。
- YelpNYC：`results/feature_manifests/feature_table_yelpnyc.MISSING.json`（官方数据需邮件申请，未取得，不造假表）。

## 2. 生成与复现

```bash
# 1) 统一层（7 个 parquet + manifest）
python src/build_unified_data.py build
# 2) 5 张特征表 + sidecar manifest + QA（YelpChi 另写 feature_qa_yelpchi.md）
python src/features/build_all.py
# 3) 复现校验（sha / 行数 / 标签守卫）
python src/features/build_all.py --verify-manifest
```

## 3. 给建模方（刘庭佳）的接口

- **模型输入（8 列）**：7 个 `_z` 列 + `burstiness_missing`。
- **禁用列**：`dataset, global_user_id, user_id, account_is_fake, cohort, is_injected, text_available`（仅评估/分组用）。
- **读取纪律**：必须用 `src.features.build_feature_table.read_feature_table(path)` 读回，否则纯数字 `user_id` 会被推断为 int64。
- **标准化**：每数据集独立 fit（`features/scaler_<dataset_id>.json`）；YelpChi 走 account_base 分支，其 scaler 与 `data/processed/scaler_params.json` 7/7 一致；unified 路径拟合时会排除 `is_injected==1` 行，但当前 4 张 unified 表的 `is_injected` 全为 0（自注入只存在于 YelpChi），该分支实为空操作，已由单测覆盖合成场景。
- **burstiness**：中位数 0.9436 填充，`burstiness_missing` 标记区分真值/填充，进模型输入。
- **C3 敏感性**：带/不带该标记的最大 |ΔAUC| 1.98 pct 且方向不一致，不能宣称稳定增益，详见 `results/c_burstiness_missing_sensitivity.md`。

## 4. 给文本/图方法（韩婧雨、周飞奕）的接口

- YelpChi 的 `text_available=0`（metadata 版无正文），**必须屏蔽文本/图文本方法**；二部图/社区发现可回到 `data/raw/YelpChi_reviews.csv` 取边。
- YelpZip 与 FraudSquad×3 的 `text_available=1`；FraudSquad 的生成/被接管账号标记为 `cohort=hijacked_llm`。
- FraudSquad×3 共用同一份真实基线（8,092 个账号中的 6,842 个 native），llama3 与 qwen2 共用同一批 1,250 个生成目标账号，跨集对比时注意账号重叠（详见 `results/qa_fraudsquad_report.md`）。

## 5. 已知问题与口径

1. **注入组不可用于检测性能结论**：注入池只选 `n_reviews ≥ 5` 的正常号，AUC 天然偏高；主结论只用原生标签。
2. **`rating_deviation` 轻微混叠**：商家均分在含假评的全量评论上计算，已声明未修复。
3. **Issue #3 已修复**（v2 实测）：A/C burstiness 平均 dz **+2.34 / +0.35**；B 为 **−0.21**（时间维度近似未动，属观测项，不再断言"恒为 0"）。
4. **注入数量已统一**：v2 = 2,791 事件 / 70,186 行，与 PR 口径一致；v3（2,910）仅归档，详见 `results/injection_count_audit.md`。
5. **FraudSquad 标签方向未声明**：manifest 标 `label_direction_unverified`，需上游确认原始 `label` 语义后才能启用方向校验。
6. **`active_days` 口径修正（2026-11-03）**：统一层为秒级时间戳，`active_days` 改为"首末评论自然日跨度"（先 `normalize()` 再相减），修复了 `n_reviews <= active_days × max_daily_reviews` 在 59 个 FraudSquad 账号上的假违规；YelpChi 为日粒度，数值不变。

## 6. 就绪消息（发给刘庭佳 / 韩婧雨 / 周飞奕）

> 5 张特征表已就绪（38,063 / 260,239 / 3×8,092 行，22 列）：
> `data/processed/features/feature_table_<dataset_id>.csv`，
> manifest 在 `results/feature_manifests/`，YelpChi QA 在 `results/feature_qa_yelpchi.md`。
> 模型输入 = 7 个 `_z` + `burstiness_missing`（8 列）；读表请用 `read_feature_table()`。
> YelpChi 无正文（`text_available=0`），文本方法只在 YelpZip / FraudSquad 上跑。
