# 特征表交接说明（B → C/D/E）

> 状态：YelpChi 已可交付；其余 4 表 `BLOCKED`（等待 `data/processed/unified/`）。
> 生成日期：2026-11-03。

## 1. 已交付表

| 数据集 | 路径 | 规模 | manifest | QA |
|---|---|---|---|---|
| YelpChi | `data/processed/features/feature_table_yelpchi.csv` | 38,063 行 × 22 列 | `results/feature_manifests/feature_table_yelpchi.json` | `results/feature_qa_yelpchi.md` |
| YelpChi（兼容副本） | `data/processed/feature_table.csv` | 同上 | 同上 | 同上 |

生成命令：

```bash
python src/features/build_feature_table.py --dataset yelpchi
# 等价：python -m src.features.build_feature_table --dataset yelpchi
```

复现校验：

```bash
python src/features/build_all.py --verify-manifest
```

## 2. 待交付表（BLOCKED）

| 数据集 | 阻塞原因 | 需要的文件 | 责任人 |
|---|---|---|---|
| YelpZip | `data/processed/unified/` 不存在 | `unified/yelpzip.parquet` + sha256 + 官方行数口径 | 叶歆澜 |
| FraudSquad-Llama3 | 同上，且需按模型拆分 | `unified/fraudsquad_llama3.parquet` | 叶歆澜 |
| FraudSquad-Qwen2 | 同上 | `unified/fraudsquad_qwen2.parquet` | 叶歆澜 |
| FraudSquad-Qwen-DSR1 | 同上 | `unified/fraudsquad_qwen_dsr1.parquet` | 叶歆澜 |
| YelpNYC | 无统一数据 | `unified/yelpnyc.parquet`（MISSING 占位已生成） | 叶歆澜 |

数据到位后一条命令构建：

```bash
python src/features/build_all.py
python src/features/build_all.py --verify-manifest
```

## 3. 给建模方（刘庭佳）的接口

- **模型输入（8 列）**：7 个 `_z` 列 + `burstiness_missing`。
- **禁用列**：`dataset, global_user_id, user_id, account_is_fake, cohort, is_injected, text_available`（仅评估/分组用）。
- **读取纪律**：必须用 `src.features.build_feature_table.read_feature_table(path)` 读回，否则纯数字 `user_id` 会被推断为 int64。
- **标准化**：每数据集独立 fit；YelpChi 的 scaler 与 `data/processed/scaler_params.json` 7/7 一致。
- **burstiness**：中位数 0.9436 填充，`burstiness_missing` 标记区分真值/填充，进模型输入。

## 4. 给文本/图方法（韩婧雨、周飞奕）的接口

- YelpChi 有 `text_available=1`，可做 TF-IDF / SimHash / 二部图。
- FraudSquad 表到位后 `text_available=1`、`cohort=hijacked_llm`；`text_available=0` 的数据集必须屏蔽文本方法。
- 账号级表不含单条评论文本，文本方法请回到 `YelpChi_reviews.csv` 按 `user_id` 关联。

## 5. 已知问题

1. **注入组不可用于检测性能结论**：注入池只选 `n_reviews ≥ 5` 的正常号，AUC 天然偏高；主结论只用原生标签。
2. **`rating_deviation` 轻微混叠**：商家均分在含假评的全量评论上计算，已声明未修复。
3. **Issue #3 已修复**：`issue3_status=recomputed`，A/C burstiness dz +1.90 / +2.08；B 保持 0。
4. **2,791 vs 2,910**：PR 口径待确认，详见 `results/injection_count_audit.md`。
5. **FraudSquad 标签方向未声明**：manifest 标 `label_direction_unverified`，需叶歆澜确认原始 `label` 语义。

## 6. 就绪消息（发给刘庭佳 / 韩婧雨 / 周飞奕，待发送）

> YelpChi 特征表 v2 已就绪：`data/processed/features/feature_table_yelpchi.csv`（38,063×22），
> manifest 在 `results/feature_manifests/feature_table_yelpchi.json`，QA 在 `results/feature_qa_yelpchi.md`。
> 模型输入 = 7 个 `_z` + `burstiness_missing`（8 列）；读表请用 `read_feature_table()`。
> YelpZip / FraudSquad 表在等统一数据，到位后我会发路径与 manifest。
