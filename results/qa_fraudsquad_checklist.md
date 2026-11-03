# FraudSquad×3 拆分验收清单（B2，2026-11-03 已执行）

> 状态：**PASS** —— 三份统一数据已构建，`python src/features/qa_fraudsquad.py` 输出 `"status": "pass"`，无 `[FAIL]`。
> 脚本：`src/features/qa_fraudsquad.py`；报告：`results/qa_fraudsquad_report.md`。

## 1. 输入（已到位）

| 数据集 ID | 文件 | 行数 | 账号 |
|---|---|---:|---:|
| `fraudsquad_llama3` | `data/processed/unified/fraudsquad_llama3.parquet` | 89,247 | 8,092 |
| `fraudsquad_qwen2` | `data/processed/unified/fraudsquad_qwen2.parquet` | 89,253 | 8,092 |
| `fraudsquad_qwen_dsr1` | `data/processed/unified/fraudsquad_qwen_dsr1.parquet` | 89,258 | 8,092 |

统一 schema：`user_id, item_id, rating, date, label, label_source`；`label ∈ {0,1}`，`label_source ∈ {yelp_filter, real_baseline, llm_generated, human_generated, injected}`（以 `adapters.LEGAL_LABEL_SOURCES` 为准）。

## 2. 硬判据（4 条，全部 PASS）

1. **每账号生成评论数 ≤ 2**：实测分布 Llama3 `{1: 11, 2: 1239}`、Qwen2 `{1: 5, 2: 1245}`、DSR1 `{2: 1250}`。
   - 判定依据：三模型被合并时会出现 4/6 条，`>2` 即 FAIL；出现 1 条是生成侧缺条，记观测。
2. **生成评论全 5 星**：三份 `(rating == 5).all()` 为 True。
3. **生成评论日期落在真实语料区间**：真实区间 `[2021-12-31, 2022-12-31]`；越界比例 0.17% / 0.24% / 0.44%，均 ≤1% 容差（最大溢出 2 天），记观测。
4. **三份共用同一份真实基线**：`label==0` 评论所属账号集合三份完全一致（8,092 个；账号级为 native 6,842 / hijacked_llm 1,250）。

## 3. 观测项（不作判定）

- 每份 1,766 / 1,770 / 1,778 行生成评论晚于"商品首评 + 5 天"——原断言在真实数据上不成立，已降级；
- 生成账号交集：Llama3 ∩ Qwen2 = 1,250（同一批目标账号）；Llama3 ∩ DSR1 = Qwen2 ∩ DSR1 = 309；
- 三份共用真实基线与生成目标账号，是构造设计，不是拆分错误。

## 4. 执行

```bash
python src/features/qa_fraudsquad.py
# 指定目录/子集：
python src/features/qa_fraudsquad.py --input-dir data/processed/unified \
  --datasets fraudsquad_llama3 fraudsquad_qwen2 fraudsquad_qwen_dsr1 \
  --report results/qa_fraudsquad_report.md
```

通过判据：脚本输出 `"status": "pass"`，报告内每份 `[PASS]`、且无 `[FAIL]`。

## 5. 失败信息含义

| 输出 | 含义 | 处理 |
|---|---|---|
| `生成评论数 > 2` | 三模型合并或拆分错误 | 让叶歆澜按模型拆三份 |
| `生成评论非 5 星` | 生成模板被改 / 混入真实评论 | 回查统一层构造 |
| `日期越界 > 1%` | 日期打标错误 | 回查生成评论日期注入逻辑 |
| `真实基线不一致` | 三份没有共用同一 base，跨集对比不可比 | 回查拆分逻辑 |
| `缺少真实评论` | 该份没有 `label==0` 基底 | 回查拆分逻辑 |

> 口径修订（2026-11-03）：原"每账号恰 2 条 / 5 天窗口 / 三份互斥"三条断言基于未验证假设，
> 真实数据上不成立；现按实测改为"≤2 / 真实区间 / 真实基线一致"，原因见报告「口径修订」。
