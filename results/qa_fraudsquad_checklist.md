# FraudSquad×3 拆分验收清单（B2）

> 状态：**BLOCKED** —— 统一数据未到位，脚本与断言已就绪，未生成任何替代数据（[待测]）。
> 脚本：`src/features/qa_fraudsquad.py`；报告：`results/qa_fraudsquad_report.md`。

## 1. 输入要求

| 数据集 ID | 文件 | 要求 |
|---|---|---|
| `fraudsquad_llama3` | `data/processed/unified/fraudsquad_llama3.parquet` | 真实评论 + 单一模型生成评论 |
| `fraudsquad_qwen2` | `data/processed/unified/fraudsquad_qwen2.parquet` | 同上 |
| `fraudsquad_qwen_dsr1` | `data/processed/unified/fraudsquad_qwen_dsr1.parquet` | 同上 |

统一 schema：`user_id, item_id, rating, date, label, label_source`；`label ∈ {0,1}`，`label_source ∈ {platform_filter, llm-generated, llm-generated?}`（以 `adapters.LEGAL_LABEL_SOURCES` 为准）。

## 2. 四条硬断言

1. **每账号生成评论数 = 2**：`label==1` 按 `user_id` 计数，任何 ≠2 都判 FAIL。
   - 已知风险：930 个账号 4 条、304 个账号 6 条——若出现即说明三模型被合并，必须先拆分。
2. **生成评论全 5 星**：`(rating == 5).all()`。
3. **日期窗口**：生成评论日期 ≤ 商品首条真实评论日期 + 5 天。
4. **三份互斥**：三个数据集的 `user_id` 集合交集为空。

## 3. 执行

```bash
python src/features/qa_fraudsquad.py
# 指定目录/子集：
python src/features/qa_fraudsquad.py --input-dir data/processed/unified \
  --datasets fraudsquad_llama3 fraudsquad_qwen2 fraudsquad_qwen_dsr1 \
  --report results/qa_fraudsquad_report.md
```

通过判据：脚本输出 `"status": "pass"`，报告内每份 `[PASS]`、且无 `[FAIL]`。

## 4. 失败信息含义

| 输出 | 含义 | 处理 |
|---|---|---|
| `生成评论数 != 2` | 三模型合并或拆分错误 | 让叶歆澜按模型拆三份 |
| `生成评论非 5 星` | 生成模板被改 / 混入真实评论 | 回查统一层构造 |
| `晚于商品首评 +5 天` | 日期打标错误 | 回查生成评论日期注入逻辑 |
| `账号集合重叠` | 同一账号被多份接管 | 拆分时按模型互斥分配 |
| `缺少真实评论` | 该份没有 `label==0` 基底 | 回查拆分逻辑 |
