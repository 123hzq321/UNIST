import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from validate_cross_dataset_joint_v1 import read_h5, group_identity, tls_labels, grouped_summary
from report_cross_dataset_joint_v1 import tls_summary


class CrossDatasetTests(unittest.TestCase):
    def test_breast_uses_author_patient_not_filename(self):
        group, source = group_identity('GSE210616', 'Primary TNBC, patient 9, section 1', '092A')
        self.assertEqual(group, 'patient_9')
        self.assertEqual(source, 'GEO_explicit_patient_title')

    def test_missing_breast_identity_fails(self):
        with self.assertRaises(ValueError):
            group_identity('GSE210616', 'TNBC sample 092', '092A')

    def test_lung_pairs_share_unit(self):
        a = group_identity('GSE300676', 'CRC_mPAP3_A', 'CRC_mPAP3_A')
        b = group_identity('GSE300676', 'CRC_mPAP3_B', 'CRC_mPAP3_B')
        self.assertEqual(a, b)

    def test_unknown_renal_identity_not_upgraded(self):
        self.assertEqual(group_identity('GSE175540', 'ccRCC tumor frozen_c_2', 'frozen_c_2')[1],
                         'section_only_patient_identity_unresolved')

    def test_two_patient_renal_uses_characteristic_not_tumor_title(self):
        group, source = group_identity('GSE210041', 'Tumour 7 (T7)', 'GLMF2', ['patient: Patient 7'])
        self.assertEqual(group, 'patient_7')
        self.assertIn('two_patient_descriptive', source)
        with self.assertRaises(ValueError):
            group_identity('GSE210041', 'Tumour 7 (T7)', 'GLMF2')

    def test_10x_orientation_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.h5'
            expected = np.array([[1, 3, 0], [0, 2, 4]], dtype=np.int32)
            matrix = sparse.csc_matrix(expected.T)
            with h5py.File(path, 'w') as f:
                g = f.create_group('matrix')
                for k, value in [('data', matrix.data), ('indices', matrix.indices),
                                 ('indptr', matrix.indptr), ('shape', matrix.shape)]:
                    g.create_dataset(k, data=value)
                g.create_dataset('barcodes', data=np.array(['a', 'b'], dtype='S'))
                fg = g.create_group('features')
                fg.create_dataset('id', data=np.array(['ENSG001', 'ENSG002', 'ENSG003'], dtype='S'))
                fg.create_dataset('name', data=np.array(['A', 'B', 'C'], dtype='S'))
            actual, features, ids = read_h5(path)
            np.testing.assert_equal(actual.toarray(), expected)
            self.assertEqual(ids.tolist(), ['a', 'b'])
            self.assertEqual(features[0].tolist(), ['ENSG001', 'ENSG002', 'ENSG003'])

    def test_nonbinary_tls_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'labels.csv'
            pd.DataFrame({'Barcode': ['a', 'b'], 'TLS': ['T_AGG', None]}).to_csv(path, index=False)
            labels, role, n = tls_labels(path, pd.Index(['a', 'b']))
            self.assertTrue(np.isnan(labels).all())
            self.assertEqual(n, 0)
            self.assertIn('excluded', role)

    def test_binary_tls_exact_barcode_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'labels.csv'
            pd.DataFrame({'Barcode': ['b', 'a'], 'TLS_2_cat': ['TLS', 'NO_TLS']}).to_csv(path, index=False)
            labels, _, n = tls_labels(path, pd.Index(['a', 'b', 'c']))
            np.testing.assert_equal(labels, [0, 1, np.nan])
            self.assertEqual(n, 2)

    def test_patient_mean_precedes_cohort_mean(self):
        data = pd.DataFrame({
            'cohort': ['GSE210616'] * 3, 'cancer': ['breast'] * 3,
            'preparation': ['frozen'] * 3, 'score': ['entropy_only'] * 3,
            'nominal_coverage': [.8] * 3, 'unit_identity_source': ['GEO_explicit_patient_title'] * 3,
            'analysis_unit': ['p1', 'p1', 'p2'], 'sample_id': ['s1', 's2', 's3'],
            'actual_coverage': [0., 1., 1.]})
        with tempfile.TemporaryDirectory() as tmp:
            grouped_summary(data, Path(tmp))
            result = pd.read_csv(Path(tmp) / 'cohort_coverage_summary.csv').iloc[0]
            self.assertAlmostEqual(result.mean_coverage, .75)
            self.assertEqual(result.n_units, 2)

    def test_unresolved_sections_get_no_population_ci(self):
        data = pd.DataFrame({
            'cohort': ['GSE175540'] * 2, 'cancer': ['renal'] * 2,
            'preparation': ['frozen'] * 2, 'score': ['entropy_only'] * 2,
            'nominal_coverage': [.8] * 2,
            'unit_identity_source': ['section_only_patient_identity_unresolved'] * 2,
            'analysis_unit': ['s1', 's2'], 'sample_id': ['s1', 's2'], 'actual_coverage': [.1, .8]})
        with tempfile.TemporaryDirectory() as tmp:
            grouped_summary(data, Path(tmp))
            result = pd.read_csv(Path(tmp) / 'cohort_coverage_summary.csv').iloc[0]
            self.assertTrue(pd.isna(result.ci_low))
            self.assertTrue(pd.isna(result.ci_high))

    def test_tls_summary_requires_both_classes_and_tracks_missing_auc(self):
        data = pd.DataFrame({
            'preparation': ['frozen'] * 3, 'score': ['fixed_fusion'] * 3,
            'policy': ['frozen80'] * 3, 'n_tls': [10, 0, 10], 'n_non_tls': [10, 10, 10],
            'tls_retention': [.2, np.nan, .4], 'non_tls_retention': [.8, .5, .6],
            'tls_b_cell_fraction_auc_full': [.9, np.nan, .8],
            'tls_b_cell_fraction_auc_retained': [.8, np.nan, np.nan],
            'auc_change': [-.1, np.nan, np.nan]})
        row = tls_summary(data).iloc[0]
        self.assertEqual(row.n_both_class_sections, 2)
        self.assertEqual(row.n_valid_retained_auc_pairs, 1)
        self.assertAlmostEqual(row.mean_tls_retention, .3)
        self.assertAlmostEqual(row.mean_non_tls_retention, .7)
        self.assertAlmostEqual(row.mean_auc_change_among_valid_pairs, -.1)


if __name__ == '__main__':
    unittest.main()
