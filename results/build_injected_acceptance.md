# `build_injected` 验收用例（B7）

> 背景：PR #10 的 `build_injected` 仍可能读取 `injection_log_v2.json`（无事件），导致注入表为空。
> 本文件只定义验收断言，不改 PR 代码；PR 负责人把读取路径切到 v3 后按下列命令执行即可判定。

## 1. 前置条件

- `data/processed/injection_log_v3.json` 存在，且每条记录含 `events_added` 数组。
- `build_injected` 读取的是 **v3**（`LOG_PATH = PROCESSED_DIR / "injection_log_v3.json"`）。

## 2. 验收断言（全部为硬断言）

| # | 断言 | 期望值（本地实测） | 判定方式 |
|---|---|---:|---|
| 1 | 注入事件表非空 | > 0 | `len(df) > 0` |
| 2 | 事件总行数 = 日志 `events_added` 总数 | **2,910** | 与日志求和比对 |
| 3 | 模板 A 事件数 | **983** | 按 `user_id` 映射模板后计数 |
| 4 | 模板 B 事件数 | **977** | 同上 |
| 5 | 模板 C 事件数 | **950** | 同上 |
| 6 | 必备字段 | `user_id, item_id, rating, date` | 列名集合检查 |
| 7 | `rating` 全部为 5 | 100% | `(df.rating == 5).all()` |
| 8 | `date` 可解析且无缺失 | 100% | `to_datetime(errors='coerce').notna().all()` |

> 修正说明：清单原写"3 模板各 970 条"为估算；实测为 **983 / 977 / 950**（合计 2,910），以实测为准 [已算]。
> 若 PR 输出的模板分布不同于上表，先查是否按 `user_id` 正确关联日志，而不是直接改期望值。

## 3. 执行命令（PR 改完后）

```bash
python src/step3_inject_water_army_pro.py          # 确保日志为 v3 且 300 条 recomputed
python -m unittest tests.test_features.InjectionRegressionTests -v
```

验收通过判据：断言 1–8 全 True；`InjectionRegressionTests` 全绿。

## 4. 失败时的定位顺序

1. 事件表为空 → 检查 `build_injected` 读取路径是否仍为 v2；
2. 行数 ≠ 2,910 → 对比日志 `add_n` 与 `events_added`（当前两者相等）；
3. 字段缺失 → 检查 v3 事件字段名（`product_id/rating/date`，按统一 schema 需映射为 `item_id`）；
4. 模板分布不符 → 检查日志 `user_id` 类型（int）与表内 `user_id`（str）关联是否失败。
