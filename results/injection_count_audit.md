# 注入数量口径：v2 = 2,791（已统一）

> 2026-11-03：B1 已重跑她版 `step3_inject_water_army_pro.py`，`injection_log_v2.json` 成为唯一真源，
> 本地实测 **2,791 条事件 / 70,186 行注入表**，与 PR 描述的 2,791 完全一致。
> 历史文件 `injection_log_v3.json`（2,910 条）仅作归档，代码与文档均不再引用。

## 1. 当前可复算口径（v2，[已算]）

| 指标 | 数值 | 复算方式 |
|---|---:|---|
| 被注入账号数 | 300 | `len(log)` |
| `policy.n_add` 合计 | **2,791** | `sum(r['policy']['n_add'] for r in log)` |
| `events_added` 合计 | **2,791** | `sum(len(r['events_added']) for r in log)` |
| A_Time_Burst | 915 | 按模板求和 |
| B_Rating_Polarization | 967 | 按模板求和 |
| C_Stealth_Mix | 909 | 按模板求和 |
| aug 表 `n_reviews` 增量 | 2,791 | aug − clean 逐账号差值求和 |
| 注入后总行数 | **70,186** | 67,395（clean）+ 2,791 |
| 统一层 `yelpchi_injected` 行数 | **70,186** | `build_unified_data.py build` 实测 |

复算命令：

```bash
python -c "import json;log=json.load(open('data/processed/injection_log_v2.json',encoding='utf-8'));print(len(log), sum(r['policy']['n_add'] for r in log), sum(len(r['events_added']) for r in log))"
```

三个口径（`n_add` / `events_added` / 表增量 / 统一层行数）逐位自洽，且与 PR 的 2,791 一致 [已算]。

## 2. 与历史 v3（2,910）的差异

| 口径 | 事件数 | 说明 |
|---|---:|---|
| v2（当前，重跑后） | 2,791 | 唯一真源；模板 A/B/C = 915 / 967 / 909 |
| v3（归档） | 2,910 | 旧脚本产出的历史文件（生成代码已不在仓库），模板 A/B/C = 983 / 977 / 950 |

- 差异 **119 条**，但分布不是"少数账号贴边"：**264/300 个账号的条数在 v2 与 v3 间不同**，
  逐账号差值落在 **−9 ~ +9**，合计 +119；两版每号都仍在 5~14 条预算内 [已算]。
- 这与"同一 300 账号、同一模板分配、重新抽样"的形态一致（模板分配完全相同，仅条数重抽），
  但仓库里没有 v3 的生成代码（v3 由旧脚本产出，已被替换），**无法逐行证明原因** → [待确认]。
  不影响结论：2,791 是当前代码可复现、三方自洽的口径，v3 仅归档。

## 3. 验收纪律

- 下游只读 `injection_log_v2.json`（`adapters.py` / `feature_eda.py` / `pipeline.build_injected` 均已切换）。
- 三件套校验：每条记录必须有 `policy`、`events_added` 非空、`physical_checks == "passed"`；
  `InjectionRegressionTests` 已从 skip 转为实际执行并通过。
- `yelpchi_injected` 行数守卫 70,186 ± 2% 已在 `pipeline.build_injected` 中生效并命中 [已算]。
