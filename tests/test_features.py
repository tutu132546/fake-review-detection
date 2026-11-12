# -*- coding: utf-8 -*-
"""B 成员回归测试（A5）：Issue #3 / Issue #6 / 标签守卫 / schema 类型校验。

运行：python -m unittest discover -s tests -v
依赖本地 data/processed 产物；缺数据时相关用例自动 skip 并给出原因，不会伪造结果。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.adapters import (_assert_label_encoding, aggregate_account_labels,
                               check_label_direction, injection_log_issues, load_unified)
from features.build_feature_table import check_schema, read_feature_table
from features.dataset_config import get_spec
from features.temporal import calc_burstiness

PROCESSED = ROOT / 'data' / 'processed'
AUG = PROCESSED / 'augmented_accounts_v2.parquet'
CLEAN = PROCESSED / 'clean_accounts.parquet'
LOG_V2 = PROCESSED / 'injection_log_v2.json'
TABLE = PROCESSED / 'features' / 'feature_table_yelpchi.csv'


class TemporalTests(unittest.TestCase):
    def test_burstiness_nan_below_three_reviews(self):
        self.assertTrue(np.isnan(calc_burstiness(['2020-01-01', '2020-01-05'])))

    def test_burstiness_nan_when_all_same_day(self):
        self.assertTrue(np.isnan(calc_burstiness(['2020-01-01'] * 5)))

    def test_burstiness_positive_for_irregular_gaps(self):
        value = calc_burstiness(['2020-01-01', '2020-01-02', '2020-01-20'])
        self.assertGreater(value, 0.0)


class LabelGuardTests(unittest.TestCase):
    def test_binary_label_accepted(self):
        self.assertEqual(_assert_label_encoding(pd.Series([0, 1, 0, 1])), 'passed')

    def test_minus_one_one_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _assert_label_encoding(pd.Series([-1, 1, 1]))
        self.assertIn('标签编码非法', str(ctx.exception))

    def test_illegal_label_source_rejected(self):
        with self.assertRaises(ValueError):
            _assert_label_encoding(pd.Series([0, 1]), ['platform_filter', 'llm-generated'])

    def test_legal_label_sources_accepted(self):
        # A4：与 data_dictionary 的 5 值枚举对齐
        legal = ['yelp_filter', 'real_baseline', 'llm_generated', 'human_generated', 'injected']
        self.assertEqual(_assert_label_encoding(pd.Series([0, 1]), legal), 'passed')

    def test_reversed_raw_direction_rejected(self):
        spec = get_spec('yelpzip')
        with self.assertRaises(ValueError) as ctx:
            check_label_direction(pd.Series([-1] * 90 + [1] * 10), spec)
        self.assertIn('标签方向疑似反转', str(ctx.exception))

    def test_undeclared_direction_marked_unverified(self):
        spec = get_spec('fraudsquad_llama3')
        self.assertEqual(check_label_direction(pd.Series([0, 1]), spec),
                         'label_direction_unverified')


def _make_log(**overrides):
    """构造一条完整的新格式注入日志（policy / events_added / physical_checks 三件套齐全）。"""
    entry = {
        'user_id': 'u1',
        'injection_template': 'A_Time_Burst',
        'add_n': 2,
        'issue3_status': 'recomputed',
        'policy': 'recompute_timestamps',
        'physical_checks': 'passed',
        'events_added': [
            {'product_id': 'p1', 'rating': 5, 'date': '2020-01-01'},
            {'product_id': 'p2', 'rating': 1, 'date': '2020-01-02'},
        ],
        'modified_features': ['burstiness'],
    }
    entry.update(overrides)
    return [entry]


class InjectionLogPolicyTests(unittest.TestCase):
    """A3：日志三件套的结构校验（纯合成数据，不依赖本地产物，改坏必红）。"""

    def test_complete_log_passes(self):
        self.assertEqual(injection_log_issues(_make_log()), [])

    def test_missing_policy_flagged(self):
        log = _make_log()
        log[0].pop('policy')
        self.assertIn('injection_policy_missing', injection_log_issues(log))

    def test_missing_physical_checks_flagged(self):
        log = _make_log()
        log[0].pop('physical_checks')
        self.assertIn('physical_checks_missing', injection_log_issues(log))

    def test_empty_events_flagged(self):
        self.assertIn('review_level_events_missing', injection_log_issues(_make_log(events_added=[])))

    def test_physical_checks_failed_flagged(self):
        self.assertIn('physical_checks_failed', injection_log_issues(_make_log(physical_checks='failed')))

    def test_old_format_flagged(self):
        old = [{'user_id': 1, 'injection_template': 'A_Time_Burst', 'modified_features': {}}]
        self.assertIn('injection_policy_missing', injection_log_issues(old))
        self.assertIn('review_level_events_missing', injection_log_issues(old))


class AccountLabelAggregationTests(unittest.TestCase):
    """A5：is_injected 由评论列聚合，cohort 优先识别注入账号。"""

    def test_injected_and_cohort_derived_correctly(self):
        reviews = pd.DataFrame({
            'user_id': ['u_native', 'u_injected', 'u_hijacked'],
            'label': [0, 1, 1],
            'is_injected': [0, 1, 0],
        })
        out = aggregate_account_labels(reviews, get_spec('fraudsquad_llama3'))
        out = out.loc[['u_native', 'u_injected', 'u_hijacked']]
        self.assertEqual(out['is_injected'].tolist(), [0, 1, 0])
        self.assertEqual(out['cohort'].tolist(), ['native', 'self_injected', 'hijacked_llm'])


class RatingMissingPolicyTests(unittest.TestCase):
    """A6：rating 缺失不再丢行，只记录缺失率。"""

    def test_nan_rating_rows_kept_and_rate_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reviews.csv'
            pd.DataFrame({
                'user_id': ['u1', 'u1', 'u2'],
                'item_id': ['i1', 'i2', 'i3'],
                'rating': [5.0, np.nan, 3.0],
                'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
                'label': [0, 0, 1],
            }).to_csv(path, index=False, encoding='utf-8')
            loaded = load_unified(get_spec('yelpzip', input_path=str(path)))
        self.assertEqual(loaded.label_stats['rows'], 3)
        self.assertEqual(loaded.label_stats['rating_missing_rows'], 1)
        self.assertAlmostEqual(loaded.label_stats['rating_missing_rate'], round(1 / 3, 6))


@unittest.skipUnless(TABLE.exists(), f"缺少特征表 {TABLE}")
class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = read_feature_table(TABLE)

    def test_user_id_read_back_as_string(self):
        self.assertTrue(pd.api.types.is_object_dtype(self.table['user_id']))

    def test_real_table_has_no_schema_violation(self):
        self.assertEqual(check_schema(self.table), [])

    def test_bad_dtype_counted_as_violation(self):
        bad = self.table.copy()
        bad['user_id'] = bad['user_id'].astype('int64')
        violations = check_schema(bad)
        self.assertGreater(len(violations), 0)
        self.assertIn('user_id', violations[0])

    def test_column_order_violation_detected(self):
        bad = self.table[list(reversed(self.table.columns))]
        self.assertGreater(len(check_schema(bad)), 0)


@unittest.skipUnless(AUG.exists() and CLEAN.exists(), "缺少账号表，跳过 Issue #6 回归")
class AccountConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aug = pd.read_parquet(AUG).set_index('user_id')
        cls.clean = pd.read_parquet(CLEAN).set_index('user_id')

    def test_issue6_constraint_holds(self):
        violated = self.aug['n_reviews'] - self.aug['active_days'] * self.aug['max_daily_reviews']
        self.assertEqual(int((violated > 0).sum()), 0)


@unittest.skipUnless(LOG_V2.exists() and AUG.exists() and CLEAN.exists(),
                     "缺少 v2 注入日志 / 账号表，跳过 Issue #3 回归")
class InjectionRegressionTests(unittest.TestCase):
    """A3：唯一真源 v2。旧格式（重跑前）显式 skip，不伪装成通过。"""

    @classmethod
    def setUpClass(cls):
        with open(LOG_V2, encoding='utf-8') as f:
            cls.log = json.load(f)
        if not any('policy' in r for r in cls.log):
            raise unittest.SkipTest(
                "当前 injection_log_v2.json 仍为重跑前的旧格式（无 policy / events_added）；"
                "需先重跑 step3（B1）后再启用本组回归，避免假绿")
        # 新格式则必须整体通过；任一条被改坏都会由下面用例判红
        cls.aug = pd.read_parquet(AUG).set_index('user_id')
        cls.clean = pd.read_parquet(CLEAN).set_index('user_id')
        cls.log_by = {r['user_id']: r for r in cls.log}
        cls.injected = cls.aug.index[cls.aug['is_injected'] == 1]

    def test_events_complete_and_checks_passed(self):
        self.assertEqual(injection_log_issues(self.log), [])
        for r in self.log:
            self.assertEqual(len(r['events_added']), r['policy']['n_add'])
            self.assertEqual(r['physical_checks'], 'passed')

    def test_policy_and_physical_checks_present(self):
        for r in self.log:
            self.assertIn('policy', r)
            self.assertEqual(r['physical_checks'], 'passed')

    def test_b_max_daily_matches_log(self):
        fixed = [r for r in self.log
                 if r['injection_template'] == 'B_Rating_Polarization'
                 and 'max_daily_reviews' in r['modified_features']]
        b_ids = [u for u in self.injected
                 if self.log_by[int(u)]['injection_template'] == 'B_Rating_Polarization']
        changed = int((self.aug.loc[b_ids, 'max_daily_reviews']
                       - self.clean.loc[b_ids, 'max_daily_reviews']).abs().gt(0).sum())
        self.assertEqual(changed, len(fixed))

    def test_issue3_burstiness_dz_positive_for_ac(self):
        """data_dictionary §9：A/C 的 burstiness 平均 dz 为正（A +2.34 / C +0.35），逐账号不保证为正。"""
        for tpl in ('A_Time_Burst', 'C_Stealth_Mix'):
            ids = [u for u in self.injected
                   if self.log_by[int(u)]['injection_template'] == tpl]
            dz = (self.aug.loc[ids, 'burstiness'] - self.clean.loc[ids, 'burstiness']).dropna()
            self.assertGreater(float(dz.mean()), 0.0, f'{tpl} burstiness 平均 dz 应为正')

    def test_untouched_features_stay_zero(self):
        """active_days 严格未动（data_dictionary §6/§9：300/300 dz=0）；B 的 burstiness 允许微调。"""
        diff = (self.aug.loc[self.injected, 'active_days']
                - self.clean.loc[self.injected, 'active_days']).abs()
        self.assertTrue(diff.max() <= 1e-12)


if __name__ == '__main__':
    unittest.main()
