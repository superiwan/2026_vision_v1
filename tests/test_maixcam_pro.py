"""PC regression: black A4 detection and 1-4 piece assembly."""

import math
import unittest

import cv2
import numpy as np

from maixcam_pro import config
from maixcam_pro.piece_detector import (
    A4PieceDetector,
    detect_pieces,
)
from maixcam_pro.puzzle_solver import (
    align_edge,
    apply_h,
    motion_commands,
    rigid,
    solve,
)
from maixcam_pro.puzzle_solver_merge import (
    enumerate_rectangle_assemblies,
    solve as solve_merge,
)
from maixcam_pro.step_view import ReleaseButton, StepView
from maixcam_pro.workflow import PuzzleWorkflow
from puzzle_sim import place_randomly, random_cut


def black_a4_scene(piece_count):
    rng = np.random.default_rng(300 + piece_count)
    source = ([np.float64(((0, 0), (400, 0), (400, 240), (0, 240)))]
              if piece_count == 1 else random_cut(rng, piece_count, "common"))
    placed = place_randomly(source, rng)
    # 900 x 1273 is a true sqrt(2) A4 plane, projected into a realistic camera
    # view with a bright margin so strict ROI/border filtering can accept it.
    paper = np.zeros((1273, 900, 3), np.uint8)
    cv2.line(paper, (0, 575), (899, 575), (95, 95, 95), 3)
    for polygon in placed:
        points = np.round(polygon).astype(np.int32)
        cv2.fillPoly(paper, [points], (245, 245, 245))
        cv2.polylines(paper, [points], True, (180, 180, 180), 2)

    frame = np.full((480, 640, 3), 225, np.uint8)
    source = np.float32(((0, 0), (899, 0), (899, 1272), (0, 1272)))
    quad = np.float32(((155, 24), (472, 42), (535, 455), (82, 430)))
    homography = cv2.getPerspectiveTransform(source, quad)
    projected = cv2.warpPerspective(paper, homography, (640, 480))
    mask = cv2.warpPerspective(
        np.full((1273, 900), 255, np.uint8), homography, (640, 480))
    frame[mask > 0] = projected[mask > 0]
    return frame


def lock_a4(detector, frame):
    result = None
    for stable_count in range(1, config.PAPER_STABLE_FRAMES + 1):
        result = detector.find_a4(frame)
        assert result["paper_stable_count"] == stable_count
        assert result["paper_locked"] == (
            stable_count == config.PAPER_STABLE_FRAMES)
    return result


def reference_perspective_scene():
    """Perspective scene using the same geometry as D:\\26_new self-test."""
    width, height = 420, 594
    paper = np.zeros((height, width, 3), np.uint8)
    target = (
        np.float32(((0, 0), (190, 0), (148, 45), (0, 30))),
        np.float32(((0, 30), (148, 45), (190, 120), (0, 120))),
        np.float32(((190, 0), (190, 120), (148, 45))),
    )
    poses = ((8, (35, 65)), (-12, (185, 235)), (28, (170, -20)))
    for polygon, (angle, translation) in zip(target, poses):
        angle = math.radians(angle)
        rotation = np.float32(((math.cos(angle), -math.sin(angle)),
                               (math.sin(angle), math.cos(angle))))
        placed = polygon.dot(rotation.T) + np.float32(translation)
        cv2.fillPoly(paper, [np.round(placed).astype(np.int32)],
                     (245, 245, 245))

    raw = np.full((480, 640, 3), 205, np.uint8)
    quad = np.float32(((145, 25), (475, 48), (545, 455), (80, 430)))
    destination = np.float32(((0, 0), (width - 1, 0),
                              (width - 1, height - 1), (0, height - 1)))
    inverse = cv2.getPerspectiveTransform(destination, quad)
    projected = cv2.warpPerspective(paper, inverse, (640, 480))
    mask = cv2.warpPerspective(
        np.full((height, width), 255, np.uint8), inverse, (640, 480))
    raw[mask > 0] = projected[mask > 0]
    return raw


def _clip_half_plane(polygon, line_start, line_end, sign):
    direction = line_end - line_start

    def side(point):
        offset = point - line_start
        return sign * (direction[0] * offset[1] - direction[1] * offset[0])

    clipped = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        current_side = side(current)
        previous_side = side(previous)
        if current_side >= 0:
            if previous_side < 0:
                ratio = previous_side / (previous_side - current_side)
                clipped.append(previous + ratio * (current - previous))
            clipped.append(current)
        elif previous_side >= 0:
            ratio = previous_side / (previous_side - current_side)
            clipped.append(previous + ratio * (current - previous))
    return np.asarray(clipped, dtype=np.float64)


def two_full_cut_pieces():
    """Four pieces made by two full straight cuts across one rectangle."""
    fragments = [np.float64(((0, 0), (400, 0), (400, 240), (0, 240)))]
    cuts = (
        (np.float64((165, 0)), np.float64((235, 240))),
        (np.float64((0, 85)), np.float64((400, 165))),
    )
    for start, end in cuts:
        divided = []
        for fragment in fragments:
            divided.extend((
                _clip_half_plane(fragment, start, end, 1.0),
                _clip_half_plane(fragment, start, end, -1.0),
            ))
        fragments = [fragment for fragment in divided if len(fragment) >= 3]
    return fragments


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

    def test_iterative_contour_merge_for_one_to_four_pieces(self):
        for piece_count in range(1, 5):
            with self.subTest(piece_count=piece_count):
                pieces, paper, _ = detect_pieces(black_a4_scene(piece_count))
                candidates = enumerate_rectangle_assemblies(pieces)
                self.assertTrue(candidates)
                self.assertTrue(all(
                    candidate.sources == frozenset(range(piece_count))
                    for candidate in candidates))
                transforms, matches, fill_ratio = solve_merge(pieces, paper)
                self.assertEqual(len(transforms), piece_count)
                self.assertEqual(len(matches), max(0, piece_count - 1))
                self.assertGreater(fill_ratio, 0.95)

    def test_algorithm2_edge_transform_maps_q0_q1_to_p1_p0(self):
        p0 = np.float64((10.0, 20.0))
        p1 = np.float64((70.0, 20.0))
        q0 = np.float64((130.0, 80.0))
        q1 = np.float64((130.0, 140.0))
        transform = align_edge(q0, q1, p1, p0)
        mapped = apply_h(np.asarray((q0, q1)), transform)
        np.testing.assert_allclose(mapped, np.asarray((p1, p0)), atol=1e-8)

    def test_algorithm2_reassembles_four_pieces_from_two_full_cuts(self):
        source = two_full_cut_pieces()
        self.assertEqual(len(source), 4)
        poses = ((17, 80, 60), (-31, 380, 90),
                 (73, 160, 320), (-114, 520, 300))
        placed = [
            apply_h(piece, rigid(math.radians(angle), tx, ty))
            for piece, (angle, tx, ty) in zip(source, poses)
        ]
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))
        transforms, matches, fill_ratio = solve_merge(placed, paper)
        self.assertEqual(len(transforms), 4)
        self.assertEqual(len(matches), 3)
        self.assertGreater(fill_ratio, 0.99)

    def test_upstream_v21_topologies_use_device_solver(self):
        paper = np.int32((((0, 0),), ((419, 0),),
                          ((419, 593),), ((0, 593),)))
        modes = ("common", "boundary_fan", "strips", "equal_rectangles",
                 "t_junction", "corner", "concave")
        for mode in modes:
            with self.subTest(mode=mode):
                rng = np.random.default_rng(7)
                source = random_cut(rng, 4, mode)
                placed = place_randomly(source, rng)
                auto_transforms, _auto_matches, auto_fill_ratio = solve(
                    placed, paper)
                self.assertEqual(len(auto_transforms), 4)
                self.assertGreater(auto_fill_ratio, 0.85)
                transforms, matches, fill_ratio = solve(
                    placed, paper, cut_mode=mode)
                self.assertEqual(len(transforms), 4)
                self.assertGreater(fill_ratio, 0.85)
                if mode == "t_junction":
                    self.assertTrue(any(
                        tuple(match[5:]) != (0.0, 1.0, 0.0, 1.0)
                        for match in matches))

    def test_a4_cache_changes_only_when_find_is_triggered(self):
        frame = black_a4_scene(4)
        detector = A4PieceDetector()
        found = lock_a4(detector, frame)
        self.assertTrue(found["paper_locked"])
        generation = found["paper_generation"]
        homography = found["homography"].copy()

        blank = np.full_like(frame, 245)
        invalid = detector.analyze_cached_a4(blank)
        self.assertEqual(invalid["piece_error"], "A4 cache invalid")
        self.assertTrue(invalid["paper_locked"])

        preview = detector.prepare_preview(blank)
        self.assertTrue(preview["paper_locked"])
        self.assertEqual(preview["paper_generation"], generation)
        np.testing.assert_allclose(preview["homography"], homography)

        refind = detector.find_a4_from_preview()
        self.assertFalse(refind["paper_locked"])
        self.assertEqual(refind["paper_generation"], generation + 1)

    def test_piece_detection_runs_only_on_solve_action(self):
        frame = black_a4_scene(4)
        detector = A4PieceDetector()
        lock_a4(detector, frame)
        analysis = detector.analyze_cached_a4(frame)
        self.assertEqual(len(analysis["pieces"]), 4)
        self.assertGreater(analysis["timings"]["pieces"], 0.0)

        preview = detector.prepare_preview(frame)
        self.assertEqual(len(preview["pieces"]), 4)
        self.assertEqual(preview["timings"]["warp"], 0.0)
        self.assertEqual(preview["timings"]["pieces"], 0.0)

    def test_reference_perspective_detection_with_current_solver(self):
        pieces, paper, _ = detect_pieces(reference_perspective_scene())
        self.assertEqual(len(pieces), 3)
        self.assertEqual([len(piece) for piece in pieces], [4, 4, 3])
        _, _, fill_ratio = solve(pieces, paper)
        self.assertGreater(fill_ratio, 0.95)
        _, _, merge_fill_ratio = solve_merge(pieces, paper)
        self.assertGreater(merge_fill_ratio, 0.95)

    def test_otsu_a4_detector_keeps_largest_a4_with_dark_distractors(self):
        frame = reference_perspective_scene()
        cv2.rectangle(frame, (0, 80), (55, 470), (18, 18, 78), -1)
        cv2.rectangle(frame, (585, 60), (639, 450), (78, 22, 18), -1)
        detector = A4PieceDetector()
        found = lock_a4(detector, frame)
        self.assertTrue(found["paper_locked"])
        self.assertIsNotNone(found["homography"])
        pieces = detector.analyze_cached_a4(frame)
        self.assertEqual(len(pieces["pieces"]), 3)

    def test_one_button_workflow_advances_one_heavy_stage_per_frame(self):
        frame = reference_perspective_scene()
        workflow = PuzzleWorkflow()
        view = StepView(640, 480)

        self.assertEqual(workflow.algorithm, 1)
        self.assertEqual(workflow.stage, workflow.READY)
        self.assertEqual(workflow.action_label, "START")
        workflow.start()
        self.assertEqual(workflow.stage, workflow.LOCATE_A4)
        self.assertIsNone(workflow.action_label)

        for stable_count in range(1, config.PAPER_STABLE_FRAMES + 1):
            workflow.advance(frame)
            if stable_count < config.PAPER_STABLE_FRAMES:
                self.assertEqual(workflow.stage, workflow.LOCATE_A4)
        self.assertEqual(workflow.stage, workflow.DETECT_PIECES)
        self.assertTrue(workflow.result["paper_locked"])

        workflow.advance(frame)
        self.assertEqual(workflow.stage, workflow.SOLVE_PUZZLE)
        self.assertEqual(len(workflow.result["pieces"]), 3)
        detected_screen = view.render(frame.copy(), workflow, 0.0)
        self.assertEqual(detected_screen.shape, (480, 640, 3))

        workflow.advance(frame)
        self.assertEqual(workflow.stage, workflow.COMPLETE)
        self.assertEqual(workflow.action_label, "NEW RUN")
        self.assertEqual(len(workflow.commands), 3)
        solved_screen = view.render(frame.copy(), workflow, 0.0)
        self.assertEqual(solved_screen.shape, (480, 640, 3))

    def test_touch_action_requires_a_complete_press_inside_button(self):
        button = ReleaseButton()
        rect = (20, 400, 600, 64)

        self.assertFalse(button.update(100, 420, True, rect))
        self.assertTrue(button.update(100, 420, False, rect))

        self.assertFalse(button.update(5, 420, True, rect))
        self.assertFalse(button.update(100, 420, False, rect))

        self.assertFalse(button.update(100, 420, True, rect))
        self.assertFalse(button.update(5, 420, True, rect))
        self.assertFalse(button.update(5, 420, False, rect))


if __name__ == "__main__":
    unittest.main()
