"""Run predeclared cohorts with unchanged features, scaling and parameters.

Each model is refitted within its cohort. This changes the density reference and
population together; cross-cohort AUC is not an estimate of model improvement.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from detection_diagnostics import FEATURES, load_features, run

COHORTS = ['native', 'native_ge3', 'native_ge3_observed']


def describe(frame):
    return {
        'rows': len(frame),
        'positive_count': int(frame.account_is_fake.sum()),
        'positive_rate': float(frame.account_is_fake.mean()),
        'missing_burstiness': int(frame.burstiness_missing.sum()),
        'reviews_median': float(frame.n_reviews.median()),
        'active_days_median': float(frame.active_days.median()),
        'reviews_under3_rate': float(frame.n_reviews.lt(3).mean()),
        'duplicate_rows_beyond_first': int(frame[FEATURES].duplicated().sum()),
        'largest_identical_group': int(frame.groupby(FEATURES).size().max()),
    }


def main(args):
    # These choices are fixed before inspecting cohort performance.
    reports = []
    first_hash = None
    native_ids = None
    for cohort in COHORTS:
        out = args.output_dir/cohort
        frame, _ = load_features(args.input, cohort)
        if native_ids is None:
            native_ids = set(frame.user_id)
        elif not set(frame.user_id).issubset(native_ids):
            raise ValueError('Subcohort is not contained in native population')
        report = run(SimpleNamespace(input=args.input, population=cohort,
                     neighbors=[20,50,100,600], output_dir=out, report=None))
        if first_hash is None:
            first_hash = report['input_sha256']
        elif report['input_sha256'] != first_hash:
            raise ValueError('Input changed during cohort analysis')
        report['cohort_summary'] = describe(frame)
        report['label_profiles'] = {str(int(label)): describe(group)
                                    for label,group in frame.groupby('account_is_fake')}
        scores = pd.read_csv(out/'scores.csv')
        aligned = frame.set_index('user_id').loc[scores.user_id].reset_index()
        if not np.array_equal(aligned.account_is_fake, scores.account_is_fake):
            raise ValueError('Label/score alignment mismatch')
        report['top50_profiles'] = {}
        for col in ['lof_k20','lof_k50','lof_k100','lof_k600','iforest']:
            order = np.argsort(-scores[col].to_numpy(),kind='stable')[:50]
            report['top50_profiles'][col] = describe(aligned.iloc[order])
        reports.append(report)
    (args.output_dir/'cohorts.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# C：分人群分析（PR #4 审查后续）','',
        '固定输入列、原有标准化参数、LOF k=20/50/100/600、iForest 200树/max_samples=256/seed=42。',
        '所有模型均在人群内重新拟合和评价；不是全体模型的切片评估，也不是独立测试。没有删除人群内的重复账号、没有加噪声、没有翻转分数或选择最优k。',
        '三个人群改变了样本组成与邻居背景，因此跨人群指标差异不能直接称为算法提升。',
        '',f"输入 SHA256：`{first_hash}`",'',
        'native：全部未注入账号；native_ge3：未注入且评论数≥3；native_ge3_observed：再要求burstiness_missing=0。',
        '最后一组保留原8列接口，其缺失标记恒为0，不重新标准化。', '',
        '| 人群 | 账号数 | 标签1数 | 标签1比例 | 爆发性缺失数 | 重复特征行（除首条） | 最大同值组 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        s=r['cohort_summary']
        lines.append(f"| {r['population']} | {s['rows']} | {s['positive_count']} | {s['positive_rate']:.4%} | {s['missing_burstiness']} | {s['duplicate_rows_beyond_first']} | {s['largest_identical_group']} |")
    lines += ['', '## 固定参数的检测结果','',
        '| 人群 | 方法 | k | AUC | P@50 | 并列边界P@50范围 | 最大分数 | 告警数 |',
        '|---|---|---:|---:|---:|---|---:|---:|']
    for report in reports:
        for r in report['runs']:
            auc=f"{r['auc']:.6f}" if r['auc'] is not None else 'NA'
            lines.append(f"| {report['population']} | {r['model']} | {r.get('k','—')} | {auc} | {r['p_at_k']:.2f} | {r['p_at_k_tie_min']:.2f}–{r['p_at_k_tie_max']:.2f} | {r['score_max']:.6g} | {len(r['warnings'])} |")
    lines += ['', '## 全部原生账号的标签画像','',
        '以下是描述性对照，不能证明因果或将Yelp代理标签视为人工核验真值。','',
        '| 标签 | 人数 | 评论数中位数 | 活跃跨度中位数（天） | 评论不足3条比例 |',
        '|---|---:|---:|---:|---:|']
    for label,s in reports[0]['label_profiles'].items():
        lines.append(f"| {label} | {s['rows']} | {s['reviews_median']:.2f} | {s['active_days_median']:.2f} | {s['reviews_under3_rate']:.2%} |")
    lines += ['', '## 全部原生账号的高异常分数组（Top-50）','',
        '| 模型 | 标签1比例 | 评论数中位数 | 活跃跨度中位数（天） | 评论不足3条比例 |',
        '|---|---:|---:|---:|---:|']
    for model,s in reports[0]['top50_profiles'].items():
        lines.append(f"| {model} | {s['positive_rate']:.2%} | {s['reviews_median']:.2f} | {s['active_days_median']:.2f} | {s['reviews_under3_rate']:.2%} |")
    lines += ['', '## 复现环境与局限','',
        f"Python {reports[0]['python']}; "+', '.join(f'{k}={v}' for k,v in reports[0]['versions'].items()),
        '运行：python src/models/cohort_analysis.py --input data/processed/feature_table.csv',
        '明细与完整JSON保存在忽略目录data/processed/c_cohorts，不提交账号ID。原PR #4报告保持不变。',
        '同分按账号ID升序；P@50并列范围也已报告。尚未做不确定性区间或独立样本验证，不能声称统计显著。',
        'AUC<0.5记录为该配置下异常分数与代理标签整体排序偏反的负面发现；不自动将异常分数取负以提高成绩。',
        '人群内重新拟合改变LOF参照邻域；不能把任一人群的结果套用为全体账号检测性能。']
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Cohort report:', args.report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('data/processed/c_cohorts'))
    p.add_argument('--report',type=Path,default=Path('results/c_cohort_analysis.md'))
    main(p.parse_args())
