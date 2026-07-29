"""Regression test: planning must work from saved pixels without generator state."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from puzzle_sim import analyze_camera_frame, generate_camera_frame


class ImageIsolationTest(unittest.TestCase):
    def test_saved_image_is_sufficient_for_planning(self):
        for piece_count in range(1, 5):
            with self.subTest(piece_count=piece_count):
                image = generate_camera_frame(seed=100 + piece_count,
                                              piece_count=piece_count)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "camera.png"
                    self.assertTrue(cv2.imwrite(str(path), image))
                    reloaded = cv2.imread(str(path), cv2.IMREAD_COLOR)

                pieces, transforms, _matches = analyze_camera_frame(reloaded)
                self.assertEqual(len(pieces), piece_count)
                self.assertEqual(len(transforms), piece_count)

    def test_joker_texture_is_detected_from_pixels_only(self):
        for piece_count in range(1, 5):
            with self.subTest(piece_count=piece_count):
                image = generate_camera_frame(
                    seed=200 + piece_count,
                    piece_count=piece_count,
                    material_mode="joker")
                reloaded = cv2.imdecode(
                    cv2.imencode(".png", image)[1], cv2.IMREAD_COLOR)
                pieces, transforms, _matches = analyze_camera_frame(reloaded)
                self.assertEqual(len(pieces), piece_count)
                self.assertEqual(len(transforms), piece_count)
                for transform in transforms:
                    # Each planned pose remains a proper rigid transform.
                    self.assertAlmostEqual(
                        float(np.linalg.det(transform[:2, :2])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
