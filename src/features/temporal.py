# -*- coding: utf-8 -*-
"""时间维度行为特征的公共实现（step1 与 step3 共用，避免两处口径漂移）。

本模块只接收时间戳序列，不接触 label / is_injected / cohort 等评估列。
缺失规则（与 data_dictionary 第五节一致）：
  - burstiness: 评论数 < 3，或相邻间隔均值 = 0（全部同日）→ NaN；
  - active_days: 单条评论 = 1；
  - max_daily_reviews: 同一自然日最大评论数。
"""
import numpy as np
import pandas as pd


def calc_burstiness(timestamps):
    """相邻评论间隔的变异系数 std(gaps)/mean(gaps)，单位为秒（无量纲结果）。

    timestamps: 单账号的时间戳序列（datetime64 / pandas Series / list）
    n < 3 或间隔均值为 0 → np.nan
    """
    ts = np.sort(pd.to_datetime(pd.Series(list(timestamps))).values)
    if len(ts) < 3:
        return np.nan
    gaps = np.diff(ts).astype('timedelta64[s]').astype(float)
    return float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else np.nan


def active_days(dates):
    """账号活跃跨度：(max(date) - min(date)).days + 1；单条评论为 1。"""
    s = pd.to_datetime(pd.Series(list(dates)))
    return int((s.max() - s.min()).days) + 1


def max_daily_reviews(dates):
    """同一自然日的最大评论数。"""
    s = pd.to_datetime(pd.Series(list(dates)))
    return int(s.dt.normalize().value_counts().max())


def time_features_by_user(reviews):
    """按 user_id 汇总时间特征。

    参数 reviews: DataFrame，至少含 user_id, date
    返回 DataFrame（index=user_id）：active_days / max_daily_reviews / burstiness
    """
    df = reviews[['user_id', 'date']].copy()
    df['date'] = pd.to_datetime(df['date'])
    g = df.groupby('user_id', sort=True)
    out = pd.DataFrame(index=g.size().index)
    out['active_days'] = g['date'].apply(active_days).astype('int32')
    out['max_daily_reviews'] = g['date'].apply(max_daily_reviews).astype('int32')
    out['burstiness'] = g['date'].apply(calc_burstiness).astype('float64')
    return out
