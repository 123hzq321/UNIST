import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from validate_frozen_joint_spatial_v1 import align_counts, blind_keep, join_metadata, pathology_endpoints, auc, count_stream


class FrozenSpatialTests(unittest.TestCase):
    def test_full_common_denominator_and_duplicate_ids(self):
        x = sparse.csr_matrix([[1, 2, 7, 100]], dtype=np.float32)
        features = pd.DataFrame({0: ['A.1', 'A.2', 'B', 'OUT'],
                                 1: ['a', 'a', 'b', 'out'], 2: ['Gene Expression'] * 4})
        mapped, counts, audit = align_counts(x, features, ['A', 'MISSING'], ['A', 'B'])
        np.testing.assert_allclose(mapped, [[3000, 0]])
        np.testing.assert_allclose(counts, [10])
        self.assertEqual(audit['selected_coverage'], .5)
        self.assertEqual(audit['duplicate_gene_ids'], 1)

    def test_no_available_selected_features(self):
        mapped, _, audit = align_counts(sparse.csr_matrix([[3]]), pd.DataFrame({0: ['B']}), ['A'], ['B'])
        np.testing.assert_equal(mapped, [[0]])
        self.assertEqual(audit['selected_coverage'], 0)

    def test_metadata_join_preserves_count_order(self):
        x = sparse.csr_matrix([[2], [3], [4]])
        barcodes = pd.Index(['b', 'a', 'missing'])
        obs = pd.DataFrame({'v': [10, 20]}, index=['a', 'b'])
        xx, bc, joined, absent = join_metadata(x, barcodes, obs)
        self.assertEqual(joined.v.tolist(), [20, 10])
        self.assertEqual(bc.tolist(), ['b', 'a'])
        self.assertEqual(absent, 1)
        np.testing.assert_equal(xx.toarray(), [[2], [3]])

    def test_ties_blind_to_input_order(self):
        ids = np.array(['a', 'b', 'c', 'd', 'e'])
        values = np.ones(5)
        keep = blind_keep(values, ids)
        reversed_keep = blind_keep(values[::-1], ids[::-1])[::-1]
        np.testing.assert_equal(keep, reversed_keep)
        self.assertEqual(keep.sum(), 4)

    def test_mixed_pathology_is_not_error_truth(self):
        obs = pd.DataFrame({'Classification': ['Invasive cancer + stroma', 'Stroma', 'Artefact', 'Normal duct']})
        y, name, controls = pathology_endpoints(obs, 'breast_wu2021')
        np.testing.assert_equal(y, [1, 0, np.nan, 1])
        self.assertIn('mixed_pathology_not_error', controls)
        self.assertIn('presence', name)

    def test_manual_tls_overrides_unassigned(self):
        obs = pd.DataFrame({'manual_anno': ['TLS', 'TUM', 'NOR', 'UNASSIGNED'],
                            'manual_anno_tls': ['ETLS', 'TUM', 'NOR', 'MTLS']})
        y, _, controls = pathology_endpoints(obs, 'tls_usz2025')
        np.testing.assert_equal(y, [1, 0, np.nan, 1])
        self.assertIn('TLS_complexity_not_error', controls)

    def test_one_class_is_missing_not_zero(self):
        self.assertTrue(np.isnan(auc([1, 1], [.2, .7])))

    def test_plain_file_with_gz_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'features.tsv.gz'
            path.write_bytes(b'AL627309.1\n')
            with count_stream(path) as stream:
                self.assertEqual(stream.read(), b'AL627309.1\n')

    def test_symbol_dot_is_not_an_ensembl_version(self):
        features = pd.DataFrame({0: ['AL627309.1', 'ENSG002.5']})
        mapped, _, _ = align_counts(sparse.csr_matrix([[2, 2]]), features,
                                    ['ENSG001', 'ENSG002'], ['ENSG001', 'ENSG002'],
                                    {'AL627309.1': 'ENSG001'})
        np.testing.assert_equal(mapped, [[5000, 5000]])


if __name__ == '__main__':
    unittest.main()
