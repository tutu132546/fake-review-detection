# -*- coding: utf-8 -*-
"""B 成员回归测试（A5）：Issue #3 / Issue #6 / 标签守卫 / schema 类型校验。

运行：python -m unittest discover -s tests -v
依赖本地 data/processed 产物；缺数据时相关用例自动 skip 并给出原因，不会伪造结果。
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.adapters import _assert_label_encoding, check_label_direction
from features.build_feature_table import check_schema, read_feature_table
from features.dataset_config import get_spec
from features.temporal import calc_burstiness

PROCESSED = ROOT / 'data' / 'processed'
AUG = PROCESSED / 'augmented_accounts_v2.parquet'
CLEAN = PROCESSED / 'clean_accounts.parquet'
LOG_V3 = PROCESSED / 'injection_log_v3.json'
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

    def test_reversed_raw_direction_rejected(self):
        spec = get_spec('yelpzip')
        with self.assertRaises(ValueError) as ctx:
            check_label_direction(pd.Series([-1] * 90 + [1] * 10), spec)
        self.assertIn('标签方向疑似反转', str(ctx.exception))

    def test_undeclared_direction_marked_unverified(self):
        spec = get_spec('fraudsquad_llama3')
        self.assertEqual(check_label_direction(pd.Series([0, 1]), spec),
                         'label_direction_unverified')


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


@unittest.skipUnless(LOG_V3.exists() and AUG.exists() and CLEAN.exists(),
                     "缺少注入日志 / 账号表，跳过 Issue #3/#6 回归")
class InjectionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(LOG_V3, encoding='utf-8') as f:
            cls.log = json.load(f)
        cls.aug = pd.read_parquet(AUG).set_index('user_id')
        cls.clean = pd.read_parquet(CLEAN).set_index('user_id')
        cls.log_by = {r['user_id']: r for r in cls.log}
        cls.injected = cls.aug.index[cls.aug['is_injected'] == 1]

    def test_issue3_all_recomputed_and_events_complete(self):
        statuses = {r['issue3_status'] for r in self.log}
        self.assertEqual(statuses, {'recomputed'})
        self.assertEqual(sum(len(r['events_added']) for r in self.log),
                         sum(r['add_n'] for r in self.log))

    def test_issue6_constraint_holds(self):
        violated = self.aug['n_reviews'] - self.aug['active_days'] * self.aug['max_daily_reviews']
        self.assertEqual(int((violated > 0).sum()), 0)

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
        ac_ids = [u for u in self.injected
                  if self.log_by[int(u)]['injection_template'] in ('A_Time_Burst', 'C_Stealth_Mix')]
        dz = self.aug.loc[ac_ids, 'burstiness'] - self.clean.loc[ac_ids, 'burstiness']
        dz = dz.dropna()
        self.assertGreater(float(dz.mean()), 0.0)
        self.assertGreater(int((dz > 0).sum()), len(dz) * 0.9)

    def test_untouched_features_stay_zero(self):
        b_ids = [u for u in self.injected
                 if self.log_by[int(u)]['injection_template'] == 'B_Rating_Polarization']
        diff = (self.aug.loc[b_ids, 'burstiness'] - self.clean.loc[b_ids, 'burstiness']).abs()
        self.assertTrue(diff.max() <= 1e-12)


if __name__ == '__main__':
    unittest.main()
