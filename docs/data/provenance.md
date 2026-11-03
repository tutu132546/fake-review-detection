# 数据集来源、核验与使用条款

> 本文档回答三个问题：**数据从哪来**、**我们怎么确认拿到的是对的**、**能怎么用、不能怎么用**。
> 生成统一格式的命令与字段说明见 [`data_dictionary.md` 第十一节](../../data_dictionary.md)，
> 数据质量实测结果见 [`results/data_quality_report.md`](../../results/data_quality_report.md)。

## 0. 一页结论

| 数据集 | 状态 | 评论数 | 带文本 | 标签来源 | 主要用途 |
|---|---|---|---|---|---|
| YelpChi | ✅ 已构建 | 67,395 | ❌ | Yelp 过滤器 | 图方法、无监督检测主基准 |
| YelpNYC | ⚠️ **未取得** | — | — | — | 待邮件申请补齐（见 §2.2） |
| YelpZip | ✅ 已构建 | 608,458 | ✅ | Yelp 过滤器 | 大规模 + 文本基线 |
| FraudSquad-Amazon | ✅ 已构建 | 94,242（其中 LLM 生成 7,484） | ✅ | 构造即真值 | 大模型水军实验 |
| YelpChi + 300 注入账号 | ✅ 已构建 | 70,186（其中注入 2,791） | ❌ | 注入脚本保证 | 解释方法（DCFO）与规避实验 |

三条必须遵守的纪律：

1. **不编造**。YelpNYC 拿不到就写"未取得"，不拿别的小数据集改名顶替。
2. **不静默**。任何缺失、偏差、口径变化都写进 `results/data_quality_report.md`，随代码一起更新。
3. **不越权再分发**。原始评论内容受 Yelp / Amazon 服务条款约束，`data/` 已列入 `.gitignore`，
   仓库里只有下载脚本和校验值，不放原始文件。

## 1. 统一取数与校验机制

所有下载都走 `src/data/sources.py` 的清单，落盘前做三项校验（`src/data/download.py`）：

| 校验 | 作用 | 失败时 |
|---|---|---|
| 文件大小 | 识别被代理截断的大文件 | 报错并换下一个镜像 |
| sha256 | 识别内容被篡改/损坏 | 报错，不用坏文件 |
| 行数 vs 官方公布 | 识别"下错了版本" | 偏差 > 2% 时在质量报告里强制提示 |

下载支持断点续传（HTTP Range），中断后重跑不会从头再来。同一个文件配多个镜像，
按可用性排序自动回退——本项目所在网络到 GitHub 直连不稳定，实测可用的是
`ghfast.top`、`cdn.jsdmirror.com`、`cdn.jsdelivr.net`、GitHub 内容 API 四条通道。

```bash
python src/build_unified_data.py download      # 下载（可断点续传、多镜像回退）
python src/build_unified_data.py build         # 解析 + 校验 + 写 parquet + 写 manifest
python src/build_unified_data.py qa            # 生成 results/data_quality_report.md
python src/build_unified_data.py verify        # 校验产物与 manifest 一致（含 sha256）
```

## 2. 逐数据集说明

### 2.1 YelpChi

- **官方出处**：<https://shebuti.com/yelpchi-dataset/>（Shebuti Rayana，Stony Brook）
- **引用**：
  - Mukherjee, Venkataraman, Liu, Glance. *What Yelp Fake Review Filter Might Be Doing?* ICWSM 2013.
  - Rayana, Akoglu. *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* KDD 2015.
- **官方规模**：67,395 条评论 / 38,063 名用户 / 201 家商户（酒店与餐厅，芝加哥地区），13.23% 被过滤。
- **我们实际用的文件**：`data/raw/yelpchi/metadata.txt`（随项目流转的 5 列版本：
  `user_id product_id rating label date`，空格分隔、无表头）。
- **核验结果**：行数 67,395、用户 38,063、商户 201、假评率 13.23%，**四项与官方公布完全一致**。
- **已知缺口**：该版本**不含评论文本**（官方原始包里的 `reviewContent` 文件缺失）。
  文本方法（TF-IDF 等）在 YelpChi 上**跑不了**，这是硬约束，不能靠造空文本绕过。
  补齐方式同 §2.2（同一作者、同一封邮件可一并申请）。
- **使用条款**：仅限学术研究；评论内容版权属 Yelp 及原作者，不得再分发或商用；
  论文中必须引用上述两篇文献。

### 2.2 YelpNYC（未取得）

- **官方出处**：<https://shebuti.com/yelpnyc-dataset/>
- **引用**：Rayana, Akoglu. KDD 2015.
- **官方规模**：359,052 条评论 / 160,264 名用户 / 923 家商户，纽约酒店评论，13.22% 被过滤。
- **状态**：**未取得**。发布页明确写着 "To get the datasets with ground truth please email:
  srayana@cs.stonybrook.edu"，没有任何直链；我们尝试过的公开渠道（ModelScope、HuggingFace 镜像、
  Kaggle 公开 API、Science Data Bank、figshare）要么没有该数据集，要么需登录/邮件授权。
- **补齐方式（10 分钟可完成）**：

  ```text
  1. 用学校邮箱发信到 srayana@cs.stonybrook.edu，说明：课程项目、仅用于学术研究、
     需要 YelpNYC 与 YelpChi（含 reviewContent 文本）；
  2. 收到后把压缩包解压，任意 .csv/.txt 放进 data/raw/yelpnyc/ 与 data/raw/yelpchi/；
  3. 重跑 python src/build_unified_data.py build qa。
  ```

  管线会自动识别（`src/data/pipeline.py: build_yelpnyc` 扫描该目录下的 `*.csv` / `*.txt`），
  **不需要改代码**。若文件列名不同，`src/data/readers.py` 的 `ALIASES` 表已覆盖常见写法。
- **使用条款**：同上，仅限学术研究，不得再分发。

> 说明：FraudSquad 仓库里有一个 `review_yelpHotelData_NRYRcleaned.txt`（5,854 条 Yelp 酒店评论），
> 名字看起来像 YelpNYC，但规模只有官方 YelpNYC 的 1.6%，是论文作者为人类垃圾评论对照实验
> 采样的子集，**不能当作 YelpNYC 使用**。我们已下载它作为辅助材料，但没有冒充 YelpNYC。

### 2.3 YelpZip

- **官方出处**：<https://shebuti.com/yelpzip-dataset/>
- **引用**：Rayana, Akoglu. KDD 2015.
- **官方规模**：608,598 条评论 / 260,277 名用户 / 5,044 家餐厅，13.22% 被过滤。
- **我们实际用的文件**：`data/raw/yelpzip/yelpzip.csv`（410,865,364 字节，
  sha256 `78f92e83…`），通过 ModelScope 公开镜像
  <https://www.modelscope.cn/datasets/jimmyliu12345/yelpzip> 取得（官方页无直链）。
- **核验结果**：

  | 指标 | 官方 | 本仓库 | 判断 |
  |---|---|---|---|
  | 评论数 | 608,598 | 608,458 | −140 条（−0.023%），可接受 |
  | 商户数 | 5,044 | 5,044 | **完全一致** |
  | 用户数 | 260,277 | 260,239 | −38（0.015%） |
  | 假评率 | 13.22% | 13.22% | **完全一致** |

  商户数完全一致、假评率完全一致，说明这是同一数据源的完整副本；少的 140 条推测来自
  上游打包时的去重。引用数字时以"本仓库实际 608,458 条"为准，不要直接写论文的 608,598。
- **使用条款**：原始数据仅限学术研究；镜像页未声明额外许可，引用仍以 Rayana & Akoglu (KDD 2015) 为准。

### 2.4 FraudSquad-Amazon（大模型水军）

- **官方出处**：<https://github.com/RafaelLiuZH/FraudSquad>（作者实现仓库）
- **引用**：Liu et al. *Detecting LLM-Generated Spam Reviews by Integrating Language Model
  Embeddings and Graph Neural Network*（FraudSquad）。
- **包含内容**：

  | 文件 | 条数 | 说明 |
  |---|---|---|
  | `amazon_2022_5core.csv` | 86,758 | 真实评论基线（Amazon Reviews 2023 的 5-core 子集） |
  | `reviews_from_llama3_8b.csv` | 2,489 | Llama3-8B 生成，全部 5 星 |
  | `reviews_from_qwen_72b.csv` | 2,495 | Qwen-72B 生成，全部 5 星 |
  | `reviews_from_qwen_dsr1.csv` | 2,500 | Qwen-DeepSeek-R1 生成，全部 5 星 |
  | `QA.zip` | 5,854×2 | 作者的人类垃圾评论对照集（**中文**百度知道问答，非 Yelp 评论） |

- **核验结果**：三个生成集的商品全部落在真实基线的商品集合内（499/499/500 全部命中），
  说明生成评论被挂到了真实商品上，图的连通性成立。
- **两个必须知道的陷阱**（详见质量报告 §5）：
  1. **生成评论复用了真实账号**：2,191 个发布过生成评论的用户，**100%** 在真实基线里也有真实评论。
     因此**不能**用"该用户是否发过生成评论"当账号级标签，否则会把正常用户误判为水军。
     需要账号级标签时，应显式标注"混合账号"并单独报告。
  2. **评分是捷径特征**：生成评论全部 5 星，而真实评论 5 星占比 69.6%。任何吃评分的模型都会虚高，
     必须把"仅用评分"作为对照基线一起报出来，否则结果不可信。
- **使用条款**：仓库**未声明 LICENSE**；真实评论部分来自 Amazon Reviews 2023 公开数据集，
  受 Amazon 数据条款约束。本项目仅作学术研究引用，不再分发原始文件。

### 2.5 YelpChi + 300 注入账号（自建）

- **来源**：本项目 `src/step3_inject_water_army_pro.py` 的事件层注入（v3.0，2026-10-24），
  注入对象为 `n_reviews >= 5` 的正常账号中随机 300 个，每个账号新增 5~14 条评论，
  三种行为模板各 100 个（`A_Time_Burst` / `B_Rating_Polarization` / `C_Stealth_Mix`）。
- **本次做的事**：把 `data/processed/injection_log_v2.json` 里的 `events_added` 展开成评论行，
  与 YelpChi 原始评论合并成评论级视图（**不重新跑注入**，保证与既有实验结果逐条一致）。
  结果：67,395 + 2,791 = 70,186 条，用户 38,063 个（注入不新增账号，只给已有账号加评论）。
- **已知缺口**：注入评论**没有文本**（step3 只生成日期/评分/商品的结构化事件）。
  文本方法需把注入行排除。
- **使用条款**：派生数据，继承 YelpChi 的全部使用限制。

## 3. 引用格式（论文参考文献可直接用）

```bibtex
@inproceedings{rayana2015collective,
  title={Collective opinion spam detection: Bridging review networks and metadata},
  author={Rayana, Shebuti and Akoglu, Leman},
  booktitle={Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={985--994},
  year={2015}
}

@inproceedings{mukherjee2013yelp,
  title={What yelp fake review filter might be doing?},
  author={Mukherjee, Arjun and Venkataraman, Vivek and Liu, Bing and Glance, Natalie},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={7},
  number={1},
  pages={409--418},
  year={2013}
}

@article{liu2026fraudsquad,
  title={Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network},
  author={Liu, Rafael and others},
  year={2026},
  note={Code: https://github.com/RafaelLiuZH/FraudSquad}
}
```

## 4. 合规自查清单（提交论文前逐条打勾）

- [ ] 数据来源小节写明了每个数据集的获取途径，镜像渠道也写了
- [ ] 引用了 Rayana & Akoglu (KDD 2015)；用了 YelpChi 还引 Mukherjee et al. (ICWSM 2013)
- [ ] 写明了 Yelp 标签是"平台过滤器结果"而非人工标注
- [ ] 写明了 YelpNYC 的实际状态（已取得 / 未取得），未取得时未在对比表里编造数字
- [ ] 未在仓库或论文附录中附带原始评论数据文件
- [ ] 对比表标注了每种方法是否需要标签
