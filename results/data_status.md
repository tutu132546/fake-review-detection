# 数据到位状态（一页纸）

> 更新：2026-11-12（**第二次更新：YelpNYC 已取得并接入**） ｜ 负责人：叶歆澜（数据组）
> 完整来源与条款见 [`docs/data/provenance.md`](../docs/data/provenance.md)，
> 实测质量与使用陷阱见 [`data_quality_report.md`](data_quality_report.md)。

## 1. 现在有什么、还缺什么

| 数据集 | 论文口径 | 状态 | 评论数 | 账号数 | 带文本 | 文件 |
|---|---|---|---|---|---|---|
| YelpChi | ✅ | 已构建 | 67,395 | 38,063 | ❌ | `unified/yelpchi.parquet` |
| **YelpNYC** | ✅ | **已构建（新）** | **359,052** | **160,225** | **✅** | `unified/yelpnyc.parquet` |
| YelpZip | ✅ | 已构建 | 608,458 | 260,239 | ✅ | `unified/yelpzip.parquet` |
| FraudSquad-Llama3 | ✅ | 已构建 | 89,247 | 8,092（1,250 被接管） | ✅ | `unified/fraudsquad_llama3.parquet` |
| FraudSquad-Qwen2 | ✅ | 已构建 | 89,253 | 同上 | ✅ | `unified/fraudsquad_qwen2.parquet` |
| FraudSquad-Qwen-DSR1 | ✅ | 已构建 | 89,258 | 同上 | ✅ | `unified/fraudsquad_qwen_dsr1.parquet` |
| FraudSquad 合并视图 | ❌ 仅总览 | 已构建 | 94,242 | 8,092（2,191 被接管） | ✅ | `unified/fraudsquad_amazon.parquet` |
| YelpChi + 300 注入 | ❌ 自建 | 已构建 | 70,186 | 38,063（300 被注入） | ❌ | `unified/yelpchi_injected.parquet` |

**账号级表**（`data/processed/accounts/<dataset>.parquet`）：**8 份**，口径见数据字典 §6。

**结论**：**论文口径的 6 张表全部到位，没有缺失数据集。**
唯一剩余缺口是 **YelpChi 缺评论文本**（持的是 `metadata` 版本，无 `reviewContent`），
影响面仅为「文本类方法不能在 YelpChi 上跑」，图方法与元数据方法不受影响。

### 1.1 YelpNYC 是怎么拿到的、凭什么说它对

- 官方页只给邮件申请渠道，发信未获回复；改从官方页 "here" 链接指向的**作者 Dropbox 共享目录**
  取得发布包 `15-collective_opinion_spam_detection.zip`（264,938,873 字节）。
- 包内数据文件是**两个**：`metadata`（user/product/rating/label/date）与
  `reviewContent`（user/product/date/text），靠 `(user, product, date)` 对齐。
- 正确性靠**五个数字 + 两个独立比例**同时复现，不是靠哈希：

  | 指标 | 官方公布 | 本项目实测 |
  |---|---|---|
  | 评论数 | 359,052 | 359,052 ✅ |
  | 用户数 | 160,225 | 160,225 ✅ |
  | 商户数 | 923 | 923 ✅ |
  | 过滤评论占比 | 10.27% | 36,885 / 359,052 = **10.2729%** ✅ |
  | spammer 账号占比 | 17.79% | 28,496 / 160,225 = **17.7850%** ✅ |

- 文本对齐：`metadata` 与 `reviewContent` 各 359,052 行，逐行 `(user, item, date)` 比对，
  **0 行错位**；审计写进 `manifest.json` 的 `text_join_audit`。

> ⚠️ **一处必须知道的口径**：YelpNYC 的假评比例是 **10.27%**，不是 13.22%。
> 三套 Yelp 数据的官方值分别是 13.23%（YelpChi）/ **10.27%（YelpNYC）** / 13.22%（YelpZip）。
> 早期文档把 YelpNYC 写成 13.22% 是抄错了（那是 YelpZip 的值），已在全部文档订正。

## 2. 给下游三人的一句话交接

| 谁 | 用什么 | 注意 |
|---|---|---|
| 杨蔚艳（特征） | 上述 6 张论文口径表 + 账号级表 | 三张 FraudSquad 表共享同一份真实基线，但**账号集合不完全相同**：llama3 与 qwen2 的 1,250 个账号完全一致，qwen_dsr1 是另一批（交集 309）。特征表按数据集分开算，不要合并 |
| 韩婧雨（检测/解释） | `load_dataset()` / `load_accounts()` | 账号级与评论级是两个任务，别混标签；FraudSquad 必须用单模型表而非合并视图 |
| 周飞奕（对比/图） | `load_dataset(name, with_text=False)` 省内存 | 文本基线现在有 **YelpNYC + YelpZip + FraudSquad** 三处可跑；只有 YelpChi 无文本（`text_available=0`），别在那里跑 TF-IDF |
| 刘庭佳（评估） | `list_datasets()` 看 `role` 列 | `role=view`（合并视图）与 `role=self_built`（注入集）都不要进主对比表；YelpNYC 可以进，它是论文口径的独立数据集 |

## 3. 本次针对审核意见的整改（逐条对应）

| # | 审核意见 | 处理 | 证据 |
|---|---|---|---|
| 1 | **FraudSquad 三套数据被合成一份**（必须改） | 拆成 `fraudsquad_llama3` / `fraudsquad_qwen2` / `fraudsquad_qwen_dsr1`，每张 = 真实基线 + 单一模型生成集，1,250 个被接管账号、每人约 2 条假评论；合并视图保留但排除出 `all_reviews` | 质量报告 §1；`pipeline.py: BUILDERS / COMBINE_EXCLUDE` |
| 2 | 账号级标签的结论有误 | 已订正：**账号级实验可以做**，正例 = 被接管账号；新增 `accounts` 子命令与 `ACCOUNT_LABEL_BASIS` 说明。仍保留的提醒是"账号标签不能下放到评论级" | 数据字典 §6；质量报告 §6 |
| 3 | 下载校验：把校验码写死 | 8 个 FraudSquad 文件的 sha256 全部写进 `sources.py`，另存官方 git blob sha1 做交叉核对（8/8 逐字节一致） | `sources.py: OFFICIAL_GIT_BLOBS`；provenance §1.1 |
| 4 | YelpZip 少 140 行，要说明原因 | 已查清：镜像文件本身有 140 行导出残缺（缺 rating/label、制表符分隔），被前一条未闭合引号的记录吞进正文，连带污染 18 条正常评论。现已剥离并审计。**行数 608,458+140=608,598、用户 260,239+38=260,277，两个口径同时对上官方** | 质量报告 §2.1；`readers.strip_partial_rows` |
| 5 | 标签编码：0 被当成虚假 | 改为显式 scheme（`signed` / `binary_1_fake` / `binary_0_fake`），`{0,1}` 一律抛错要求显式声明，不再默认猜方向 | 数据字典 §4；`schema.label_from_yelp` + 单测 |
| 6 | 文字错误：YelpNYC 写成"酒店"、QA 写成 Yelp 的 | YelpNYC/YelpZip 订正为**餐厅**；`QA.zip` 订正为 **THUIR WWW'17 中文社区问答协同刷分数据**（37,998 + 24,953 行），与 Yelp 无关，本项目未用它建模 | `schema.py` / `sources.py` / `provenance.md` 已同步 |

### 3.1 本次追加（不在原审核意见里）

| 事项 | 结果 |
|---|---|
| **YelpNYC 补齐** | 从作者 Dropbox 官方发布包取得，359,052 条 / 160,225 用户 / 923 商户、带完整文本，五项官方指标全部复现 |
| **订正 YelpNYC 假评比例** | 13.22% → **10.27%**（13.22% 是 YelpZip 的值，早期抄错）；用户数 160,264 → 160,225 |
| **订正 FraudSquad 账号关系** | 原写"三表共用同一批 1,250 个账号"是错的：llama3 = qwen2（相同），qwen_dsr1 是另一批（交集 309、并集 2,191） |
| **补齐双指标对账** | 三套 Yelp 数据的"过滤评论占比 + spammer 占比"全部与官方吻合，见 provenance §2.3 末尾 |
| **清理备选方案** | YelpNYC 到手后，规模对齐子样本（`subsample` 子命令、`results/yelpzip_scale359k.md`、`docs/data/yelpnyc_fallback.md`）已全部删除，不再保留无用代码与文档 |

## 4. 复现命令（一条链路）

```bash
python src/build_unified_data.py download      # 断点续传 + 多镜像回退 + sha256 校验
python src/build_unified_data.py build         # 8 张评论级表 + manifest
python src/build_unified_data.py accounts      # 8 张账号级表
python src/build_unified_data.py qa            # 质量报告
python src/build_unified_data.py verify        # 产物与 manifest 一致性校验
```

预估耗时：下载（视网络）2–10 分钟，构建约 2 分钟，账号级约 15 秒。
（`all` 子命令已包含上述全部步骤。）

> YelpNYC 的两个原始文件来自作者 Dropbox，不在脚本下载清单里，需手工放入
> `data/raw/yelpnyc/`（`metadata` 与 `reviewContent`），详见 provenance §2.2。

## 5. 仍待外部输入的事项

| 事项 | 阻塞谁 | 现状 |
|---|---|---|
| YelpChi 带文本版 | 周飞奕的 TF-IDF 在 YelpChi 上跑不了 | 邮件申请未回复；已在数据里用 `text_available=0` 标死，其余数据集都有文本，基线实验不受影响 |
