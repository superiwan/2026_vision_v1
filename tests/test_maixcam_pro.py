"""PC regression: black A4 detection and 1-4 piece assembly."""

import unittest

import cv2
import numpy as np

from maixcam_pro.piece_detector import detect_pieces
from maixcam_pro.puzzle_solver import motion_commands, solve
from puzzle_sim import place_randomly, random_cut


def black_a4_scene(piece_count):
    rng = np.random.default_rng(300 + piece_count)
    source = random_cut(rng, piece_count)
    placed = place_randomly(source, rng)
    frame = np.full((1200, 900, 3), 245, np.uint8)
    cv2.rectangle(frame, (15, 15), (884, 1184), (0, 0, 0), -1)
    cv2.line(frame, (16, 575), (883, 575), (245, 245, 245), 3)
    for polygon in placed:
        points = np.round(polygon).astype(np.int32)
        cv2.fillPoly(frame, [points], (245, 245, 245))
        cv2.polylines(frame, [points], True, (180, 180, 180), 2)
    return frame


class MaixCamProPipelineTest(unittest.TestCase):
    def test_black_a4_pipeline_for_one_to_four_pieces(self):
        for piece_count in range(1, 5):
            with self.subTest(piece_count=piece_count):
                frame = black_a4_scene(piece_count)
                pieces, paper, _ = detect_pieces(frame)
                self.assertEqual(len(pieces), piece_count)
                self.assertTrue(all(3 <= len(piece) <= 5 for piece in pieces))

                transforms, matches, fill_ratio = solve(pieces, paper)
                commands = motion_commands(pieces, transforms)
                self.assertEqual(len(transforms), piece_count)
                self.assertEqual(len(commands), piece_count)
                self.assertGreater(fill_ratio, 0.95)
                for command in commands:
                    self.assertGreaterEqual(command["distance"], 0.0)
                    self.assertEqual(len(command["matrix_3x3"]), 3)


if __name__ == "__main__":
    unittest.main()

