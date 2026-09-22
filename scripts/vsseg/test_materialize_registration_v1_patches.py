import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import materialize_registration_v1_patches as mod


class TestMaterializeRegistrationV1Patches(unittest.TestCase):
    def test_component_quotas(self):
        self.assertEqual(mod.component_quotas(32, 2), [16, 16])
        self.assertEqual(mod.component_quotas(5, 2), [3, 2])

    def test_sample_id_is_stable_and_deidentified(self):
        a = mod.sample_id("PAIR_X", 1, 2048, 4096, 1024, 2048)
        b = mod.sample_id("PAIR_X", 1, 2048, 4096, 1024, 2048)
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("PATCH_"))
        self.assertNotIn("PAIR_X", a)

    def test_affine_source_bbox_identity(self):
        matrix = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float)
        self.assertEqual(
            mod.affine_source_bbox(matrix, [100, 200, 300, 400], 10),
            [90, 190, 310, 410],
        )

    def test_roi_integrity_gate(self):
        args = argparse.Namespace(
            roi_integrity_min_edge_corr=0.60,
            roi_integrity_min_gray_corr=0.65,
            roi_integrity_min_mask_iou=0.55,
        )
        self.assertTrue(
            mod.roi_integrity_passes(
                {"edge_corr": 0.7, "gray_corr": 0.8, "mask_iou": 0.2}, args
            )
        )
        self.assertTrue(
            mod.roi_integrity_passes(
                {"edge_corr": 0.7, "gray_corr": 0.3, "mask_iou": 0.8}, args
            )
        )
        self.assertFalse(
            mod.roi_integrity_passes(
                {"edge_corr": 0.4, "gray_corr": 0.9, "mask_iou": 0.9}, args
            )
        )

    def test_grid_centers_stay_context_safe(self):
        centers = mod.grid_centers(
            [0, 0, 5000, 5000], (6000, 6000), context_size=2048, stride=1024
        )
        self.assertTrue(centers)
        for x, y in centers:
            self.assertGreaterEqual(x - 1024, 0)
            self.assertGreaterEqual(y - 1024, 0)
            self.assertLessEqual(x + 1024, 6000)
            self.assertLessEqual(y + 1024, 6000)


class TestPreDhrOverlapGate(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            pre_dhr_min_affine_tissue_fraction=0.10,
            pre_dhr_min_tissue_ratio=0.35,
            pre_dhr_min_tissue_dice=0.60,
            pre_dhr_max_centroid_distance_norm=0.20,
        )

    def test_rejects_missing_affine_tissue(self):
        he = np.full((128, 128, 3), 255, np.uint8)
        aff = np.full_like(he, 255)
        he[16:112, 16:112] = (120, 40, 120)
        metrics = mod.tissue_overlap_metrics(he, aff)
        self.assertFalse(mod.pre_dhr_overlap_passes(metrics, self.args))

    def test_accepts_matching_tissue_support(self):
        he = np.full((128, 128, 3), 255, np.uint8)
        aff = np.full_like(he, 255)
        he[16:112, 16:112] = (120, 40, 120)
        aff[18:114, 18:114] = (160, 100, 40)
        metrics = mod.tissue_overlap_metrics(he, aff)
        self.assertTrue(mod.pre_dhr_overlap_passes(metrics, self.args))


if __name__ == "__main__":
    unittest.main()
