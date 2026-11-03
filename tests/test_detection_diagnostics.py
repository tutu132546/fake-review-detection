import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('diagnostics', Path(__file__).resolve().parents[1]/'src/models/detection_diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticTests(unittest.TestCase):
    def test_tie_bounds_do_not_claim_an_arbitrary_order_is_unique(self):
        result = diag.evaluate([1, 0, 1, 0], [3, 2, 2, 1], top_k=2)
        self.assertEqual(result['p_at_k'], .5)
        self.assertEqual(result['p_at_k_tie_min'], .5)
        self.assertEqual(result['p_at_k_tie_max'], 1)

    def test_larger_anomaly_score_means_more_positive(self):
        self.assertEqual(diag.evaluate([0, 0, 1, 1], [1, 2, 3, 4], 2)['auc'], 1)

    def test_single_class_has_no_auc(self):
        self.assertIsNone(diag.evaluate([0, 0], [1, 2], 1)['auc'])

    def make_frame(self):
        frame = pd.DataFrame({f: [0., 1., 2.] for f in diag.FEATURES})
        frame['burstiness_missing'] = [1, 0, 1]
        frame['user_id'] = [30, 10, 20]
        frame['account_is_fake'] = [0, 1, 1]
        frame['is_injected'] = [0, 1, 0]
        frame['unrelated_numeric_column'] = 999
        return frame

    def test_filters_injections_and_excludes_labels_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'f.csv'
            self.make_frame().to_csv(path,index=False)
            frame, X = diag.load_features(path)
        self.assertEqual(frame.user_id.tolist(), [20, 30])
        self.assertEqual(X.shape, (2,8))
        self.assertFalse(np.any(X == 999))
        self.assertEqual(X[0,0], 2.) # supplied scaling preserved

    def test_rejects_duplicate_ids_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'f.csv'
            frame = self.make_frame()
            frame.loc[1,'user_id'] = 30
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'unique'):
                diag.load_features(path)
            frame = self.make_frame()
            frame.loc[0,diag.FEATURES[0]] = np.inf
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                diag.load_features(path)

    def test_missingness_flag_must_stay_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'f.csv'
            frame = self.make_frame()
            frame['burstiness_missing'] = [0., .3, 1.]
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'binary'):
                diag.load_features(path)

    def test_three_reviews_does_not_imply_observed_burstiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'f.csv'
            frame = self.make_frame()
            frame['n_reviews'] = [3, 4, 5]
            frame['is_injected'] = [0, 1, 0]
            frame['burstiness_missing'] = [1, 0, 0]
            frame.to_csv(path,index=False)
            ge3, X = diag.load_features(path,'native_ge3')
            observed, Xobs = diag.load_features(path,'native_ge3_observed')
        self.assertEqual(ge3.user_id.tolist(),[20,30])
        self.assertEqual(observed.user_id.tolist(),[20])
        self.assertEqual(Xobs.shape,(1,8))
        self.assertEqual(Xobs[0,-1],0)
        self.assertEqual(Xobs[0,0],2.)

    def test_cohort_count_must_be_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'f.csv'
            frame=self.make_frame()
            frame['n_reviews']=[1,2.5,3]
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'integer'):
                diag.load_features(path,'native_ge3')


if __name__ == '__main__':
    unittest.main()
