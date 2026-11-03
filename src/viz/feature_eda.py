# -*- coding: utf-8 -*-
"""B 成员交付：特征 EDA 与注入方向性验证

产出（results/）：
  1. feature_eda_group_box.png   原生正常 / 原生水军 / 注入水军 三组 7 特征 _z 箱线图
  2. feature_eda_template_dz.png 注入前后配对 dz（_z 变化）的模板级均值对比

同时打印可证伪判定标准（对应 data_dictionary.md 第五节 §6/§7 的量化验证）。
注：箱线图基于未填充底表（burstiness 的 NaN 不参与绘制）；建模填充口径见 feature_table.csv。

用法：python src/viz/feature_eda.py （任意工作目录均可）
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC',
                                   'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

FEATURES = ['n_reviews', 'active_days', 'daily_avg_reviews', 'max_daily_reviews',
            'burstiness', 'rating_deviation', 'extreme_ratio']
ZC = [f + '_z' for f in FEATURES]
TEMPLATES = ['A', 'B', 'C']
TPL_SHORT = {'A_Time_Burst': 'A', 'B_Rating_Polarization': 'B', 'C_Stealth_Mix': 'C'}
COLORS = ['#4C72B0', '#DD8452', '#55A868']


def run_eda():
    aug = pd.read_parquet(PROCESSED_DIR / "augmented_accounts_v2.parquet")
    clean = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    # A2：注入日志唯一真源为 v2（v3 仅历史归档）
    log_path = PROCESSED_DIR / "injection_log_v2.json"
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    log_by = {r['user_id']: r for r in log}
    issue3_fixed = sum(len(r.get('events_added') or []) for r in log) > 0

    inj_mask = aug['is_injected'] == 1
    inj_ids = aug.loc[inj_mask, 'user_id']
    cm = clean.set_index('user_id')
    augm = aug.set_index('user_id')

    # ---------- 图 1：三组箱线图 ----------
    groups = {
        '原生正常': aug.loc[~inj_mask & (aug['account_is_fake'] == 0)],
        '原生水军': aug.loc[~inj_mask & (aug['account_is_fake'] == 1)],
        '注入水军': aug.loc[inj_mask],
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        data, labels = [], []
        for (gname, g), c in zip(groups.items(), COLORS):
            vals = g[f'{feat}_z'].dropna()  # burstiness 未填充，NaN 不参与
            data.append(vals.values)
            labels.append(f'{gname}\n(n={len(vals):,})')
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black'))
        for patch, c in zip(bp['boxes'], COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_xticks(range(1, 4))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(feat, fontsize=12)
        ax.set_ylabel('z 分数')
    axes[7].axis('off')
    axes[7].text(0.0, 0.97,
                 '说明\n'
                 '· burstiness 按未填充原始值绘制，\n  NaN 不参与（86.5% 账号无定义）\n'
                 '· 建模表（data/processed/features/\n  feature_table_yelpchi.csv）以\n  中位数 0.9436 填充 + missing 标记\n'
                 '· 箱线图隐藏离群点以突出分布主体\n'
                 '· 注入组箱体整体偏离 0 即联动\n  生效；方向见判定清单',
                 va='top', fontsize=11)
    fig.suptitle('三组账号行为特征对比（Z-score，YelpChi 全量 38,063 账号）', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = RESULTS_DIR / 'feature_eda_group_box.png'
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    logging.info(f"图 1 已保存: {p1}")

    # ---------- 图 2：模板级配对 dz ----------
    dz = augm.loc[inj_ids, ZC] - cm.loc[inj_ids, ZC]
    dz['template'] = [TPL_SHORT[log_by[int(u)]['injection_template']] for u in inj_ids]
    m = dz.groupby('template')[ZC].mean()

    x = range(len(FEATURES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 6))
    for j, t in enumerate(TEMPLATES):
        ax.bar([xi + (j - 1) * width for xi in x],
               m.loc[t, ZC].values, width, label=f'模板 {t}（n=100）')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(FEATURES, fontsize=10)
    ax.set_ylabel('注入前后 dz 均值（z 分数）')
    ax.set_title('注入效应的配对 dz（同账号注入前后对比，消除选池偏差）', fontsize=13)
    ax.legend()
    fig.tight_layout()
    p2 = RESULTS_DIR / 'feature_eda_template_dz.png'
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    logging.info(f"图 2 已保存: {p2}")

    # ---------- 可证伪判定 ----------
    logging.info("可证伪判定（对应 data_dictionary 第五节 §6/§7）：PASS/FAIL 为硬判据；"
                 "[观测] 为事件层注入下必然不成立的项，只记录不判定")

    def chk(name, cond, detail):
        logging.info(f"   [{'PASS' if cond else 'FAIL'}] {name} | {detail}")

    def obs(name, detail):
        logging.info(f"   [观测] {name} | {detail}")

    chk('全模板 n_reviews dz>0（300/300）',
        bool((dz['n_reviews_z'] > 1e-12).all()),
        f"dz均值 {dz['n_reviews_z'].mean():+.3f}，上升 {int((dz['n_reviews_z'] > 1e-12).sum())}/{len(dz)}")
    chk('全模板 daily_avg_reviews dz>0（300/300）',
        bool((dz['daily_avg_reviews_z'] > 1e-12).all()),
        f"dz均值 {dz['daily_avg_reviews_z'].mean():+.3f}（修复前恒为 0）")
    n_ext_up = int((dz['extreme_ratio_z'] > 1e-12).sum())
    chk('全模板 extreme_ratio dz>=0（其余为原本全 5 星，数值不变）',
        bool((dz['extreme_ratio_z'] >= -1e-12).all()),
        f"上升 {n_ext_up}/{len(dz)}，其余 {len(dz) - n_ext_up} 个本就全 5 星")

    dz_ac = dz[dz['template'].isin(['A', 'C'])]
    if issue3_fixed:
        chk('模板A/C active_days dz>=0（新增评论可能扩展活跃期）',
            bool((dz_ac['active_days_z'] >= -1e-12).all()),
            f"变化 {int((dz_ac['active_days_z'] > 1e-12).sum())}/{len(dz_ac)}")
    else:
        chk('全模板 active_days dz=0（无 review 级时间戳，未动该特征）',
            bool((dz['active_days_z'].abs() < 1e-12).all()), f"{len(dz)}/{len(dz)} 恒为 0")

    for t in ['A', 'C']:
        chk(f'模板{t} max_daily_reviews dz>0',
            bool(m.loc[t, 'max_daily_reviews_z'] > 0),
            f"dz均值 {m.loc[t, 'max_daily_reviews_z']:+.3f}")
        if issue3_fixed:
            chk(f'模板{t} burstiness dz>0（按真实时间序列重算，方向可证伪）',
                bool(m.loc[t, 'burstiness_z'] > 0),
                f"dz均值 {m.loc[t, 'burstiness_z']:+.3f}")
        else:
            logging.info(f"   [描述] 模板{t} burstiness dz均值 {m.loc[t, 'burstiness_z']:+.3f} | "
                         f"注入日志 v2 无 review 级事件（events_added 为空），"
                         f"不作方向性判定；补 review 级时间戳后恢复 dz>0 检验")
    for t in ['B', 'C']:
        chk(f'模板{t} rating_deviation dz>0',
            bool(m.loc[t, 'rating_deviation_z'] > 0),
            f"dz均值 {m.loc[t, 'rating_deviation_z']:+.3f}")

    b_fixed = [r for r in log if r['injection_template'] == 'B_Rating_Polarization'
               and 'max_daily_reviews' in r['modified_features']]
    dz_b = dz[dz['template'] == 'B']
    n_b_pos = int((dz_b['max_daily_reviews_z'] > 1e-12).sum())
    chk('模板B max_daily_reviews 满足 n<=active_days*max_daily（dz>=0，且与日志补齐数一致）',
        bool((dz_b['max_daily_reviews_z'] >= -1e-12).all() and n_b_pos == len(b_fixed)),
        f"dz>0 {n_b_pos}/{len(dz_b)}，日志补齐 {len(b_fixed)} 个账号（Issue #6）")
    obs('模板B burstiness（只撒空闲日，间隔序列趋于均匀，非"未动"）',
        f"dz均值 {m.loc['B', 'burstiness_z']:+.3f}")
    obs('模板A rating_deviation（新增评论全 5 星，必然抬高偏离）',
        f"dz均值 {m.loc['A', 'rating_deviation_z']:+.3f}")

    logging.info("EDA 完成")


if __name__ == "__main__":
    run_eda()
