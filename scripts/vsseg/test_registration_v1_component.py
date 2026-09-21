import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registration_utils as utils
import registration_v1_component as mod


class KfbBackend:
    __module__ = "kfb.kfb_slide"


class OpenSlideBackend:
    __module__ = "openslide"


class FakeSlide:
    def __init__(self, backend, fail=False):
        self._osr = backend()
        self.level_downsamples = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        self.dimensions = (20000, 40000)
        self.fail = fail
        self.region_calls = []
        self.fixed_calls = []

    def read_region(self, location, level, size):
        self.region_calls.append((location, level, size))
        if self.fail:
            raise RuntimeError("synthetic read failure")
        width, height = size
        return np.zeros((height, width, 3), dtype=np.uint8)

    def read_fixed_region(self, location, level, size):
        self.fixed_calls.append((location, level, size))
        raise AssertionError("formal v1 must never use read_fixed_region")


def component(component_id, bbox, shape=(200, 200)):
    x, y, w, h = bbox
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    area = int(mask.sum())
    ys, xs = np.where(mask)
    return {
        "id": component_id,
        "area": area,
        "area_fraction_canvas": area / float(shape[0] * shape[1]),
        "area_fraction_tissue": 0.5,
        "bbox": list(bbox),
        "centroid": [float(xs.mean()), float(ys.mean())],
        "aspect": float(w) / h,
        "mask": mask,
    }


class TestRegistrationV1Component(unittest.TestCase):
    def test_kfb_level_location_divides_level0_coordinate(self):
        slide = FakeSlide(KfbBackend)
        self.assertEqual(
            mod.backend_location_from_level0(slide, (5852, 31740), 2),
            (1463, 7935),
        )

    def test_openslide_location_stays_level0(self):
        slide = FakeSlide(OpenSlideBackend)
        self.assertEqual(
            mod.backend_location_from_level0(slide, (5852, 31740), 2),
            (5852, 31740),
        )

    def test_read_uses_read_region_only_and_propagates_failure(self):
        slide = FakeSlide(KfbBackend, fail=True)
        with self.assertRaises(RuntimeError):
            mod.read_level0_bbox_at_level(slide, [5852, 31740, 7900, 33788], 2)
        self.assertEqual(len(slide.region_calls), 1)
        self.assertEqual(slide.fixed_calls, [])

    def test_integrity_gate_accepts_structure_consistent_level(self):
        args = argparse.Namespace(
            kfb_integrity_min_ok_samples=5,
            kfb_integrity_min_edge_corr=0.60,
            kfb_integrity_min_gray_corr=0.65,
            kfb_integrity_min_mask_iou=0.55,
        )
        good = {
            "n_ok": 9,
            "median_edge_corr": 0.73,
            "median_gray_corr": 0.85,
            "median_mask_iou": 0.73,
        }
        bad = {
            "n_ok": 9,
            "median_edge_corr": 0.46,
            "median_gray_corr": 0.16,
            "median_mask_iou": 0.17,
        }
        self.assertTrue(mod.integrity_summary_passes(good, args))
        self.assertFalse(mod.integrity_summary_passes(bad, args))

    def test_nearest_allowed_level_respects_common_downsample(self):
        slide = FakeSlide(KfbBackend)
        level = mod._nearest_level_for_downsample(
            slide, target_downsample=16.0, allowed_levels=[2, 4, 5]
        )
        self.assertEqual(level, 4)

    def test_merge_nearby_fragments_but_not_distant_block(self):
        components = [
            component(0, (10, 10, 30, 30)),
            component(1, (42, 12, 30, 30)),
            component(2, (140, 140, 30, 30)),
        ]
        merged = utils.merge_nearby_components(
            components, (200, 200, 3), max_gap_fraction=0.03
        )
        self.assertEqual(len(merged), 2)
        source_groups = sorted(sorted(c["source_component_ids"]) for c in merged)
        self.assertEqual(source_groups, [[0, 1], [2]])

    def test_component_matching_preserves_layout(self):
        he = [component(0, (10, 10, 40, 40)), component(1, (130, 130, 40, 40))]
        ihc = [component(0, (15, 14, 41, 39)), component(1, (126, 134, 39, 42))]
        matches, missing_he, missing_ihc = utils.match_components(
            he, ihc, (200, 200, 3), (200, 200, 3), max_cost=1.0
        )
        self.assertEqual([(h, i) for h, i, _ in matches], [(0, 0), (1, 1)])
        self.assertEqual(missing_he, [])
        self.assertEqual(missing_ihc, [])


if __name__ == "__main__":
    unittest.main()
