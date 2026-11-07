import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import h5py
import numpy as np
import pandas as pd
from run_spatial_policy_transfer_v1 import (
    weights, fit_cutoff, apply_cutoff, conditional_keep, hash_uniform, fit_policies, SCORES, infer_h5,
)


class SpatialPolicyTests(unittest.TestCase):
    def test_patient_section_weights(self):
        frame = pd.DataFrame({'analysis_unit': ['a', 'a', 'a', 'b'], 'sample_id': ['x', 'x', 'y', 'z']})
        np.testing.assert_allclose(weights(frame), [.125, .125, .25, .5])

    def test_constant_score_is_random_tie_not_discrimination(self):
        rule = fit_cutoff(np.ones(10), np.ones(10))
        self.assertAlmostEqual(rule['tie_acceptance'], .8)
        np.testing.assert_equal(apply_cutoff(np.ones(3), np.array([.1, .79, .81]), rule), [True, True, False])

    def test_weighted_quantile_expected_mass(self):
        rule = fit_cutoff([0, 1, 1, 2], [.2, .3, .4, .1])
        self.assertEqual(rule['value'], 1)
        self.assertAlmostEqual(rule['below_mass'] + rule['tied_mass'] * rule['tie_acceptance'], .8)

    def test_invalid_input_rejected(self):
        for values, mass in [([np.nan], [1]), ([0], [0]), ([], []), ([0, 1], [1])]:
            with self.assertRaises(ValueError):
                fit_cutoff(values, mass)

    def test_hash_uses_sample_and_barcode_not_labels(self):
        a = hash_uniform('s1', ['a', 'b'])
        np.testing.assert_equal(a, hash_uniform('s1', ['a', 'b']))
        self.assertFalse(np.array_equal(a, hash_uniform('s2', ['a', 'b'])))

    def test_unsupported_dominant_bin_falls_back_to_pooled(self):
        frame = pd.DataFrame({'preparation': ['frozen'] * 4, 'analysis_unit': ['p1', 'p1', 'p2', 'p2'],
                              'sample_id': ['s1', 's1', 's2', 's2'], 'dominant': ['A', 'B', 'A', 'B']})
        for score in SCORES:
            frame[score] = [.1, .4, .6, .9]
        rule = fit_policies(frame)['frozen']['entropy_only']
        self.assertTrue(rule['conditional']['A']['fallback_to_pooled'])
        self.assertEqual(rule['conditional']['A']['rule'], rule['pooled'])
        u = np.zeros(4)
        np.testing.assert_equal(conditional_keep(frame, 'entropy_only', u, rule),
                                apply_cutoff(frame.entropy_only, u, rule['pooled']))

    def test_large_h5_stream_accepts_string_path_and_preserves_spots(self):
        n = 50001
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stream.h5'
            with h5py.File(path, 'w') as f:
                g = f.create_group('matrix')
                g.create_dataset('shape', data=[2, n])
                g.create_dataset('indptr', data=np.arange(n + 1, dtype=np.int64))
                g.create_dataset('indices', data=np.zeros(n, dtype=np.int32))
                g.create_dataset('data', data=np.ones(n, dtype=np.int32))
                g.create_dataset('barcodes', data=np.array([f'b{i}' for i in range(n)], dtype='S'))
                feat = g.create_group('features')
                feat.create_dataset('id', data=np.array(['ENSG1', 'ENSG2'], dtype='S'))
                feat.create_dataset('name', data=np.array(['A', 'B'], dtype='S'))
                feat.create_dataset('feature_type', data=np.array(['Gene Expression'] * 2, dtype='S'))

            def dummy_predict(models, tensor, centroids):
                p = np.full((len(tensor), 7), 1/7)
                return p, np.zeros(len(tensor)), np.zeros(len(tensor)), p

            with patch('run_spatial_policy_transfer_v1.predict', side_effect=dummy_predict):
                frame, audit = infer_h5(str(path), None, None, ['ENSG1', 'ENSG2'],
                                         ['ENSG1', 'ENSG2'], {}, {'labels': list('ABCDEFG')},
                                         {'scales': {'novelty': [0, 1], 'disagreement': [0, 1]}},
                                         'cpu', chunk=10000)
            self.assertEqual(len(frame), n)
            self.assertEqual(audit['input_spots'], n)
            self.assertEqual(audit['zero_common_count_spots'], 0)
            self.assertAlmostEqual(audit['common_count_mass_fraction'], 1)


if __name__ == '__main__':
    unittest.main()
