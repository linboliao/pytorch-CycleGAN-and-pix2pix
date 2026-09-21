import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("registration_v1_auto_component.py")
spec = importlib.util.spec_from_file_location("registration_v1_auto_component", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


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


class TestAutoComponentRegistration(unittest.TestCase):
    def test_merge_nearby_fragments_but_not_distant_block(self):
        components = [
            component(0, (10, 10, 30, 30)),
            component(1, (42, 12, 30, 30)),
            component(2, (140, 140, 30, 30)),
        ]
        merged = mod.merge_nearby_components(
            components, (200, 200, 3), max_gap_fraction=0.03
        )
        self.assertEqual(len(merged), 2)
        source_groups = sorted(sorted(c["source_component_ids"]) for c in merged)
        self.assertEqual(source_groups, [[0, 1], [2]])

    def test_component_matching_preserves_layout(self):
        he = [
            component(0, (10, 10, 40, 40)),
            component(1, (130, 130, 40, 40)),
        ]
        ihc = [
            component(0, (15, 14, 41, 39)),
            component(1, (126, 134, 39, 42)),
        ]
        matches, missing_he, missing_ihc = mod.match_components(
            he, ihc, (200, 200, 3), (200, 200, 3), max_cost=1.0
        )
        self.assertEqual([(h, i) for h, i, _ in matches], [(0, 0), (1, 1)])
        self.assertEqual(missing_he, [])
        self.assertEqual(missing_ihc, [])

    def test_thumbnail_to_level0_affine(self):
        thumb_matrix = np.asarray([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]])
        level0 = mod.thumbnail_to_level0_affine(
            thumb_matrix,
            he_sx=0.1,
            he_sy=0.1,
            ihc_sx=0.2,
            ihc_sy=0.2,
        )
        expected = np.asarray([[0.5, 0.0, 50.0], [0.0, 0.5, 100.0]])
        self.assertTrue(np.allclose(level0, expected))

    def test_high_consistency_five_match_case_is_medium(self):
        result = {
            "inlier_count": 5,
            "inlier_ratio": 0.55,
            "inlier_median_residual_px": 0.5,
            "inlier_p95_residual_px": 3.0,
            "spatial_coverage": 0.06,
        }
        self.assertEqual(mod.confidence_level(result), "medium")


if __name__ == "__main__":
    unittest.main()
