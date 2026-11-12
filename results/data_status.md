# 数据到位状态（一页纸）

> 更新：2026-11-12 ｜ 负责人：叶歆澜（数据组）
> 完整来源与条款见 [`docs/data/provenance.md`](../docs/data/provenance.md)，
> YelpNYC 方案见 [`docs/data/yelpnyc_fallback.md`](../docs/data/yelpnyc_fallback.md)，
> 实测质量与使用陷阱见 [`data_quality_report.md`](data_quality_report.md)。

## 1. 现在有什么、还缺什么

| 数据集 | 论文口径 | 状态 | 评论数 | 账号数 | 带文本 | 文件 |
|---|---|---|---|---|---|---|
| YelpChi | ✅ | 已构建 | 67,395 | 38,063 | ❌ | `unified/yelpchi.parquet` |
| YelpZip | ✅ | 已构建 | 608,458 | 260,239 | ✅ | `unified/yelpzip.parquet` |
| FraudSquad-Llama3 | ✅ | 已构建 | 89,247 | 8,092（1,250 被接管） | ✅ | `unified/fraudsquad_llama3.parquet` |
| FraudSquad-Qwen2 | ✅ | 已构建 | 89,253 | 同上 | ✅ | `unified/fraudsquad_qwen2.parquet` |
| FraudSquad-Qwen-DSR1 | ✅ | 已构建 | 89,258 | 同上 | ✅ | `unified/fraudsquad_qwen_dsr1.parquet` |
| **YelpNYC** | ✅ | ⛔ **缺失** | — | — | — | — |
| FraudSquad 合并视图 | ❌ 仅总览 | 已构建 | 94,242 | 8,092（2,191 被接管） | ✅ | `unified/fraudsquad_amazon.parquet` |
| YelpChi + 300 注入 | ❌ 自建 | 已构建 | 70,186 | 38,063（300 被注入） | ❌ | `unified/yelpchi_injected.parquet` |

**账号级表**（`data/processed/accounts/<dataset>.parquet`）：7 份，口径见数据字典 §6。

**派生视图**（`data/processed/derived/`）：`yelpzip_scale359k_seed0~4.parquet` ——
YelpZip 规模对齐子样本，**只用于 YelpNYC 缺失时的稳健性检查，不是 YelpNYC，也不进主对比表**。
实测见 [`yelpzip_scale359k.md`](yelpzip_scale359k.md)。

**结论**：公开数据到位 3 个来源 / 5 张论文口径表，缺 1 个（YelpNYC）。
已满足"3–4 个公开数据集"的要求；YelpNYC 的处理与论证见 `yelpnyc_fallback.md`。

## 2. 给下游三人的一句话交接

| 谁 | 用什么 | 注意 |
|---|---|---|
| 杨蔚艳（特征） | 上述 5 张表 + 账号级表 | 三张 FraudSquad 表共用同一批账号与同一份真实基线，特征表要按数据集分开算，不要合并 |
| 韩婧雨（检测/解释） | `load_dataset()` / `load_accounts()` | 账号级与评论级是两个任务，别混标签；FraudSquad 必须用单模型表而非合并视图 |
| 周飞奕（对比/图） | `load_dataset(name, with_text=False)` 省内存 | TF-IDF 基线在 YelpChi 上**跑不了**（无文本），请用 YelpZip 与 FraudSquad |
| 刘庭佳（评估） | `list_datasets()` 看 `role` 列；规模敏感性用 `data/processed/derived/yelpzip_scale359k_seed{0..4}.parquet` | `role=view` 的表与派生视图都不要进主对比表；派生视图只回答"评论总量变小后是否稳" |

## 3. 本次针对审核意见的整改（逐条对应）

| # | 审核意见 | 处理 | 证据 |
|---|---|---|---|
| 1 | **FraudSquad 三套数据被合成一份**（必须改） | 拆成 `fraudsquad_llama3` / `fraudsquad_qwen2` / `fraudsquad_qwen_dsr1`，每张 = 真实基线 + 单一模型生成集，1,250 个被接管账号、每人约 2 条假评论；合并视图保留但排除出 `all_reviews` | 质量报告 §1；`pipeline.py: BUILDERS / COMBINE_EXCLUDE` |
| 2 | 账号级标签的结论有误 | 已订正：**账号级实验可以做**，正例 = 被接管账号；新增 `accounts` 子命令与 `ACCOUNT_LABEL_BASIS` 说明。仍保留的提醒是"账号标签不能下放到评论级" | 数据字典 §6；质量报告 §6 |
| 3 | 下载校验：把校验码写死 | 8 个 FraudSquad 文件的 sha256 全部写进 `sources.py`，另存官方 git blob sha1 做交叉核对（8/8 逐字节一致） | `sources.py: OFFICIAL_GIT_BLOBS`；provenance §1.1 |
| 4 | YelpZip 少 140 行，要说明原因 | 已查清：镜像文件本身有 140 行导出残缺（缺 rating/label、制表符分隔），被前一条未闭合引号的记录吞进正文，连带污染 18 条正常评论。现已剥离并审计。**行数 608,458+140=608,598、用户 260,239+38=260,277，两个口径同时对上官方** | 质量报告 §2.1；`readers.strip_partial_rows` |
| 5 | 标签编码：0 被当成虚假 | 改为显式 scheme（`signed` / `binary_1_fake` / `binary_0_fake`），`{0,1}` 一律抛错要求显式声明，不再默认猜方向 | 数据字典 §4；`schema.label_from_yelp` + 单测 |
| 6 | 文字错误：YelpNYC 写成"酒店"、QA 写成 Yelp 的 | YelpNYC/YelpZip 订正为**餐厅**；`QA.zip` 订正为 **THUIR WWW'17 中文社区问答协同刷分数据**（37,998 + 24,953 行），与 Yelp 无关，本项目未用它建模 | `schema.py` / `sources.py` / `provenance.md` 已同步 |

> 另有一件事不在审核意见里、但属于本次交付：**YelpNYC 缺口**。
> 申请无回复 → 如实标 `missing`，并给出经论证的备选方案，其中方案 A（YelpZip 规模对齐子样本）
> **已实现跑通**，5 个种子的产物与实测数字见 [`yelpzip_scale359k.md`](yelpzip_scale359k.md)，
> 论证过程见 [`docs/data/yelpnyc_fallback.md`](../docs/data/yelpnyc_fallback.md)。

## 4. 复现命令（一条链路）

```bash
python src/build_unified_data.py download      # 断点续传 + 多镜像回退 + sha256 校验
python src/build_unified_data.py build         # 7 张评论级表 + manifest
python src/build_unified_data.py accounts      # 7 张账号级表
python src/build_unified_data.py subsample     # YelpZip 规模对齐派生视图（5 个种子）
python src/build_unified_data.py qa            # 质量报告
python src/build_unified_data.py verify        # 产物与 manifest 一致性校验
```

预估耗时：下载（视网络）2–10 分钟，构建约 1 分钟，账号级约 10 秒，派生视图约 1 分钟。
（`all` 子命令已包含上述全部步骤。）

## 5. 仍待外部输入的事项

| 事项 | 阻塞谁 | 现状 |
|---|---|---|
| YelpNYC 授权 | 刘庭佳的对比表要标 N/A | 邮件未回复；方案见 `yelpnyc_fallback.md`（推荐走 YelpZip 规模分层子样本） |
| YelpChi 带文本版 | 周飞奕的 TF-IDF 在 YelpChi 上跑不了 | 同一封邮件申请，未回复；已在数据里用 `text_available=0` 标死 |
