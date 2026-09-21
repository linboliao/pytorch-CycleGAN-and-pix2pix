import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registration_v1_1_structure_component as mod


class KfbBackend:
    __module__ = "kfb.kfb_slide"


class OpenSlideBackend:
    __module__ = "openslide"


class FakeSlide:
    def __init__(self, backend, fail_roi=False):
        self._osr = backend()
        self.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        self.fail_roi = fail_roi
        self.region_calls = []
        self.fixed_calls = []

    def read_region(self, location, level, size):
        self.region_calls.append((location, level, size))
        if self.fail_roi:
            raise RuntimeError("synthetic ROI failure")
        w, h = size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def read_fixed_region(self, location, level, size):
        self.fixed_calls.append((location, level, size))
        return np.zeros((256, 256, 3), dtype=np.uint8)


class TestKfbCoordinateHandling(unittest.TestCase):
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

    def test_kfb_read_region_receives_level_coordinates(self):
        slide = FakeSlide(KfbBackend)
        output = mod.read_region_resilient(
            slide, (5852, 31740), 2, (64, 32)
        )
        self.assertEqual(output.shape, (32, 64, 3))
        self.assertEqual(slide.region_calls[0][0], (1463, 7935))

    def test_kfb_fixed_tile_offsets_are_level_coordinates(self):
        slide = FakeSlide(KfbBackend, fail_roi=True)
        output = mod.read_region_resilient(
            slide, (5852, 31740), 2, (300, 300)
        )
        self.assertEqual(output.shape, (300, 300, 3))
        locations = [call[0] for call in slide.fixed_calls]
        self.assertIn((1463, 7935), locations)
        self.assertIn((1719, 7935), locations)
        self.assertIn((1463, 8191), locations)

    def test_openslide_fixed_tile_offsets_remain_level0(self):
        slide = FakeSlide(OpenSlideBackend, fail_roi=True)
        mod.read_region_resilient(slide, (1000, 2000), 2, (300, 300))
        locations = [call[0] for call in slide.fixed_calls]
        self.assertIn((1000, 2000), locations)
        self.assertIn((2024, 2000), locations)
        self.assertIn((1000, 3024), locations)


if __name__ == "__main__":
    unittest.main()
