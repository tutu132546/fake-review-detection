# `build_injected` 验收用例（B7，2026-11-03 重跑后）

> 当前状态：`injection_log_v2.json` 为唯一真源（B1 已重跑），`build_injected` 实测
> **2,791 条事件 / 70,186 行**，行数守卫 70,186 ± 2% 命中。
> `injection_log_v3.json`（2,910 条）已退役，仅作历史归档。

## 1. 前置条件（当前）

- `data/processed/injection_log_v2.json` 存在，且每条记录满足三件套：
  `policy`、`events_added` 非空、`physical_checks == "passed"`（见 `docs/feature_contract.md` §7）。
- `data/raw/YelpChi_reviews.csv` 存在（step3 需要真实时间戳；缺失时脚本直接失败，不会伪造事件）。
- `build_injected` 读取 `injection_log_v2.json`，并断言注入后总行数 70,186 ± 2%。

## 2. 验收断言（全部为硬断言，数值为本地实测 [已算]）

| # | 断言 | 期望值 | 判定方式 |
|---|---|---:|---|
| 1 | 注入事件表非空 | > 0 | `len(df) > 0` |
| 2 | 事件总行数 = 日志 `events_added` 总数 | **2,791** | 与日志求和比对 |
| 3 | 模板 A 事件数 | **915** | 按 `user_id` 映射模板后计数 |
| 4 | 模板 B 事件数 | **967** | 同上 |
| 5 | 模板 C 事件数 | **909** | 同上 |
| 6 | 注入后总行数 | **70,186**（守卫 ±2%） | `len(clean) + 事件数` 与 `yelpchi_injected` 表行数 |
| 7 | 必备字段 | `user_id, item_id, rating, date` 无缺失 | 列名集合 + `isna().sum()` |
| 8 | `rating` 取值 | 仅 `{1, 5}`；5 星占比 **91.54%** | `isin([1,5]).all()` + 占比在 90–95% |
| 9 | `date` 可解析 | 100% | `to_datetime(errors='coerce').notna().all()` |

> 口径说明：旧用例写"rating 全部为 5"，与注入策略（90–95% 五星、其余 1 星）不符，已改为
> "仅取 {1,5} 且五星占比落在 90–95%"。实测 2,555 条 5 星 + 236 条 1 星 = 91.54%。
> 旧用例的 2,910 / 983 / 977 / 950 为 v3 历史值，已全部替换为 v2 实测值。

## 3. 执行命令

```bash
python src/step3_inject_water_army_pro.py                # 生成 v2（2,791 事件 / 70,186 行）
python src/build_unified_data.py build                   # 构建 unified/yelpchi_injected.parquet
python -m unittest tests.test_features.InjectionRegressionTests -v
```

验收通过判据：断言 1–9 全 True；`InjectionRegressionTests` 实际执行且全绿（不再 skip）。

## 4. 失败时的定位顺序

1. 事件表为空 → 检查 `build_injected` 读取路径是否为 v2，且 v2 的 `events_added` 非空；
2. 行数 ≠ 2,791 → 对比日志 `policy.n_add` 与 `events_added`（当前两者严格相等）；
3. 总行数偏离 70,186 超过 2% → 先确认评论底表与日志是否同版（`YelpChi_reviews.csv` 行数 67,395）；
4. 字段缺失 → 检查日志事件字段 `product_id/rating/date` 是否正确映射为统一 schema 的 `item_id`；
5. 模板分布不符 → 检查日志 `user_id` 类型（int）与表内 `user_id`（str）关联是否失败。
