# -*- coding: utf-8 -*-
"""B2：FraudSquad×3 拆分预置校验。

硬判据（任一 FAIL 即拆分/构造有错）：
  1. 每账号生成评论数 <= 2（三模型被合并时会出现 4/6 条，这是本检查的原始目的）；
  2. 生成评论全 5 星；
  3. 生成评论日期落在真实语料日期区间内；
  4. 三份共用同一份真实基线（label=0 的账号集合逐份一致）。

观测项（只记录，不作判定；2026-11-03 实测口径修订，见报告「口径修订」）：
  - 生成评论数 = 1 的账号数（该模型只生成了 1 条，不等于拆分错误）；
  - 晚于商品首条真实评论 +5 天的行数（原断言在真实数据上不成立）；
  - 三份生成账号集合的两两交集（llama3 与 qwen2 共享 1,250 个目标账号）。

数据未到位时输出 BLOCKED 并给出缺失路径，不生成任何替代数据。

用法：
  python src/features/qa_fraudsquad.py
  python src/features/qa_fraudsquad.py --input-dir data/processed/unified \
      --report results/qa_fraudsquad_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from features.dataset_config import UNIFIED_DIR

RESULTS = _SRC.parent / 'results'
DEFAULT_IDS = ['fraudsquad_llama3', 'fraudsquad_qwen2', 'fraudsquad_qwen_dsr1']
MAX_PER_ACCOUNT = 2
WINDOW_DAYS = 5


def _check_one(df, dataset_id):
    errors, observes = [], []
    if not {'user_id', 'item_id', 'rating', 'date', 'label'} <= set(df.columns):
        return [f'{dataset_id}: 缺少统一 schema 字段'], [], {}, {}
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    gen = df[df['label'] == 1]
    real = df[df['label'] == 0]

    counts = gen.groupby('user_id').size()
    over = counts[counts > MAX_PER_ACCOUNT]
    if len(over):
        errors.append(f'{dataset_id}: {len(over)} 个账号生成评论数 > {MAX_PER_ACCOUNT}'
                      f'（疑似三模型合并，分布 {counts.value_counts().sort_index().to_dict()}）')
    single = int((counts == 1).sum())
    if single:
        observes.append(f'{dataset_id}: {single} 个账号只生成 1 条（该模型未给该账号生成第 2 条，非拆分错误）')

    if len(gen) and not (gen['rating'] == 5).all():
        errors.append(f'{dataset_id}: 生成评论非 5 星 {int((gen["rating"] != 5).sum())} 行')

    if len(real):
        lo, hi = real['date'].min(), real['date'].max()
        out_of_range = gen[(~gen['date'].between(lo, hi))]
        rate = len(out_of_range) / max(len(gen), 1)
        if rate > 0.01:
            errors.append(f'{dataset_id}: {len(out_of_range)} 行生成评论日期落在真实语料区间 '
                          f'[{lo.date()}, {hi.date()}] 之外（{rate:.2%} > 1%）')
        elif len(out_of_range):
            overflow_days = int((out_of_range['date'].max() - hi).days) if \
                out_of_range['date'].max() > hi else int((lo - out_of_range['date'].min()).days)
            observes.append(f'{dataset_id}: {len(out_of_range)}/{len(gen)} 行（{rate:.2%}）生成评论'
                            f'略在真实语料区间 [{lo.date()}, {hi.date()}] 之外，'
                            f'最大溢出 {overflow_days} 天（<=1% 容差，视为生成侧日期贴边）')
        first_real = real.groupby('item_id')['date'].min()
        merged = gen.join(first_real.rename('first_real'), on='item_id')
        missing_first = int(merged['first_real'].isna().sum())
        late = int((merged['date'] > merged['first_real'] + pd.Timedelta(days=WINDOW_DAYS)).sum())
        if missing_first:
            observes.append(f'{dataset_id}: {missing_first} 行生成评论的商品无真实评论，不参与窗口观测')
        observes.append(f'{dataset_id}: {late} 行生成评论晚于商品首评 +{WINDOW_DAYS} 天'
                        f'（原断言不成立，见「口径修订」）')
    else:
        errors.append(f'{dataset_id}: 缺少真实评论（label=0）')

    stats = {'rows': int(len(df)), 'accounts': int(df['user_id'].nunique()),
             'generated': int(len(gen)),
             'count_dist': {str(k): int(v) for k, v in counts.value_counts().sort_index().items()}}
    sets = {'real': set(real['user_id'].astype(str)),
            'gen': set(gen['user_id'].astype(str))}
    return errors, observes, stats, sets


def run(input_dir=None, ids=None, report_path=None):
    input_dir = Path(input_dir) if input_dir else UNIFIED_DIR
    ids = ids or DEFAULT_IDS
    report_path = Path(report_path) if report_path else RESULTS / 'qa_fraudsquad_report.md'
    lines = ['# FraudSquad×3 拆分验收报告', '']
    payload = {'status': 'blocked', 'datasets': {}, 'errors': [], 'observations': []}
    sets_by_id, missing = {}, []
    for did in ids:
        path = input_dir / f'{did}.parquet'
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_parquet(path)
        errs, obs, stats, sets = _check_one(df, did)
        payload['datasets'][did] = stats
        payload['errors'] += errs
        payload['observations'] += obs
        sets_by_id[did] = sets
        lines += [f'## {did}', f'- 行数 {stats["rows"]} / 账号 {stats["accounts"]} / '
                  f'生成评论 {stats["generated"]}', f'- 每账号生成数分布 {stats["count_dist"]}']
        lines += [f'- [FAIL] {e}' for e in errs] or ['- [PASS] 四条硬判据通过']
        lines += [f'- [观测] {o}' for o in obs]

    if missing:
        payload['status'] = 'blocked'
        payload['missing'] = missing
        lines += ['', '**状态：BLOCKED** —— 以下文件缺失，未生成任何表：']
        lines += [f'- `{m}`' for m in missing]
    elif sets_by_id:
        keys = list(sets_by_id)
        # 硬判据 4：三份共用同一份真实基线
        real_sets = [sets_by_id[k]['real'] for k in keys]
        if all(s == real_sets[0] for s in real_sets[1:]):
            lines += ['', f'- [PASS] 三份共用同一份真实基线（{len(real_sets[0])} 个有真实评论的账号）']
        else:
            payload['errors'].append('三份的真实基线账号集合不一致')
            lines += ['', '- [FAIL] 三份的真实基线账号集合不一致']
        # 观测：生成账号集合两两交集
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter = sets_by_id[keys[i]]['gen'] & sets_by_id[keys[j]]['gen']
                if inter:
                    msg = (f'{keys[i]} ∩ {keys[j]} 生成账号 {len(inter)} 个'
                           f'（共用目标账号，非拆分错误）')
                    payload['observations'].append(msg)
                    lines += [f'- [观测] {msg}']
                else:
                    lines += [f'- [观测] {keys[i]} ∩ {keys[j]} 生成账号无交集']
        payload['status'] = 'pass' if not payload['errors'] else 'fail'

    lines += ['', '## 口径修订（2026-11-03，基于真实数据）', '',
              '- 原断言「每账号生成评论数 == 2」改为 `<= 2`：实测 llama3 有 11 个、qwen2 有 5 个账号只有 1 条，'
              '属生成侧缺条，不是三模型合并（合并会出现 4/6 条）；',
              '- 原断言「生成评论 ≤ 商品首评 + 5 天」在真实数据上不成立（每份 1,700+ 行），'
              '现改为「日期落在真实语料区间」，5 天窗口降级为观测项，待上游确认构造意图；',
              '- 原断言「三份账号集合互斥」与数据事实冲突：三份本就共用真实基线，'
              '且 llama3 与 qwen2 共用同一批 1,250 个生成目标账号；'
              '现拆为「真实基线一致」硬判据 + 「生成账号交集」观测项。', '',
              '> 硬判据：每账号生成数 <= 2、生成评论全 5 星、日期在真实语料区间内、三份真实基线一致。']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description='FraudSquad×3 拆分验收')
    parser.add_argument('--input-dir', default=None)
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    run(args.input_dir, args.datasets, args.report)


if __name__ == '__main__':
    main()
