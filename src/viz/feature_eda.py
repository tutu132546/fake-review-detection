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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
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
    with open(PROCESSED_DIR / "injection_log_v2.json", encoding="utf-8") as f:
        log = json.load(f)
    log_by = {r['user_id']: r for r in log}

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
                 '· 建模表（feature_table.csv）以\n  中位数 0.9436 填充 + missing 标记\n'
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
    logging.info("可证伪判定（对应 data_dictionary 第五节 §6/§7；任一 FAIL 即 step3 联动有错）:")

    def chk(name, cond, detail):
        logging.info(f"   [{'PASS' if cond else 'FAIL'}] {name} | {detail}")

    chk('全模板 n_reviews dz>0（300/300）',
        bool((dz['n_reviews_z'] > 1e-12).all()),
        f"dz均值 {dz['n_reviews_z'].mean():+.3f}，上升 {int((dz['n_reviews_z'] > 1e-12).sum())}/300")
    chk('全模板 daily_avg_reviews dz>0（300/300）',
        bool((dz['daily_avg_reviews_z'] > 1e-12).all()),
        f"dz均值 {dz['daily_avg_reviews_z'].mean():+.3f}（修复前恒为 0）")
    chk('全模板 extreme_ratio dz>=0（296/300 上升）',
        bool((dz['extreme_ratio_z'] >= -1e-12).all()) and int((dz['extreme_ratio_z'] > 1e-12).sum()) == 296,
        f"上升 {int((dz['extreme_ratio_z'] > 1e-12).sum())}/300，其余 4 个本就全 5 星")
    chk('全模板 active_days dz=0（未动特征）',
        bool((dz['active_days_z'].abs() < 1e-12).all()), '300/300 恒为 0')
    for t in ['A', 'C']:
        chk(f'模板{t} max_daily_reviews dz>0',
            bool(m.loc[t, 'max_daily_reviews_z'] > 0),
            f"dz均值 {m.loc[t, 'max_daily_reviews_z']:+.3f}")
        chk(f'模板{t} burstiness dz<0',
            bool(m.loc[t, 'burstiness_z'] < 0),
            f"dz均值 {m.loc[t, 'burstiness_z']:+.3f}")
    for t in ['B', 'C']:
        chk(f'模板{t} rating_deviation dz>0',
            bool(m.loc[t, 'rating_deviation_z'] > 0),
            f"dz均值 {m.loc[t, 'rating_deviation_z']:+.3f}")
    chk('模板B max_daily_reviews dz=0（未动特征）',
        bool(abs(m.loc['B', 'max_daily_reviews_z']) < 1e-12), '恒为 0')
    chk('模板B burstiness dz=0（未动特征）',
        bool(abs(m.loc['B', 'burstiness_z']) < 1e-12), '恒为 0')
    chk('模板A rating_deviation dz=0（未动特征）',
        bool(abs(m.loc['A', 'rating_deviation_z']) < 1e-12), '恒为 0')

    logging.info("EDA 完成")


if __name__ == "__main__":
    run_eda()
