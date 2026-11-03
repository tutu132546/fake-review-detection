# 注入数量口径差异说明（2,791 vs 2,910）

> 生成日期：2026-11-03；数据源 `data/processed/injection_log_v3.json`（`issue3_status=recomputed`）。
> 结论性质：本地口径 [已算]；对方口径 [待确认]。

## 1. 本地可复算口径

| 指标 | 数值 | 复算方式 |
|---|---:|---|
| 被注入账号数 | 300 | `len(log)` |
| `add_n` 合计 | **2,910** | `sum(r['add_n'])` |
| `events_added` 合计 | **2,910** | `sum(len(r['events_added']))` |
| A_Time_Burst | 983 | 按模板求和 |
| B_Rating_Polarization | 977 | 按模板求和 |
| C_Stealth_Mix | 950 | 按模板求和 |
| aug 表 `n_reviews` 增量 | 2,910 | aug − clean 逐账号差值求和 |

复算命令（无需额外脚本）：

```bash
python -c "import json,collections;log=json.load(open('data/processed/injection_log_v3.json',encoding='utf-8'));print(sum(r['add_n'] for r in log),sum(len(r['events_added']) for r in log))"
```

## 2. 差异结论

- 本地日志 `add_n` 与 `events_added` **严格相等，均为 2,910**；注入表增量亦为 2,910，三方自洽。
- PR 描述中的 **2,791** 比本地少 **119** 条。当前无任何本地文件能复算出 2,791，故只能判定为**口径不同**，不能判定谁错。
- 最可能的三种口径（均未验证，[待确认]）：
  1. 仅统计"有效日期事件"：剔除落在原活跃期外或无法构造日期的评论；
  2. 旧版随机流（日期构造与 `add_n` 共用一条流）产生过 2,796 的中间结果，与 2,791 相差 5 条，可能是更早一版；
  3. 模板 B 不计入（977 条）或只计 A+C（1,933 条）——数值均不等于 2,791，可排除。
- 需叶歆澜/日志作者回复：2,791 的统计口径（字段、模板范围、是否含无效日期）。回复前本文档不修改本地结论。

## 3. 验收纪律

- 下游一律以 `injection_log_v3.json` 的 `add_n` / `events_added`（2,910）为准。
- 若对方确认 2,791 为正确口径，则需同步修改 step3 的日期构造逻辑并重跑全链路，不能在文档层"抹平"119 条。
