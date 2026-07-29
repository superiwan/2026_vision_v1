"""Legacy offline 2x2 diagnostics; the device runtime uses step_view.py."""

import cv2
import numpy as np

try:
    from . import config
    from .puzzle_solver import apply_h
except ImportError:
    import config
    from puzzle_solver import apply_h


def _draw_polygon(image, polygon, color, thickness=2):
    points = np.round(polygon).astype(np.int32)
    cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)


class DebugDashboard:
    """Render an optional offline diagnostic composite for detector tuning."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.half_width = width // 2
        self.half_height = height // 2
        self.right_width = width - self.half_width
        self.bottom_height = height - self.half_height

        self.dashboard = np.empty((height, width, 3), np.uint8)
        self.live_panel = np.empty(
            (self.half_height, self.half_width, 3), np.uint8)
        self.diag_panel = np.empty(
            (self.half_height, self.right_width, 3), np.uint8)
        self.a4_panel = np.empty(
            (self.bottom_height, self.half_width, 3), np.uint8)
        self.solution_panel = np.empty(
            (self.bottom_height, self.right_width, 3), np.uint8)

        left_diag_width = self.right_width // 2
        right_diag_width = self.right_width - left_diag_width
        self.gray_small = np.empty(
            (self.half_height, left_diag_width), np.uint8)
        self.binary_small = np.empty(
            (self.half_height, right_diag_width), np.uint8)
        self.gray_color = np.empty(
            (self.half_height, left_diag_width, 3), np.uint8)
        self.binary_color = np.empty(
            (self.half_height, right_diag_width, 3), np.uint8)

        warp_shape = (config.A4_WARP_HEIGHT, config.A4_WARP_WIDTH, 3)
        self.a4_color = np.empty(warp_shape, np.uint8)
        self.solution_color = np.empty(warp_shape, np.uint8)

    def _annotate_live(self, frame, result, status):
        contour = result["paper_contour"]
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, config.PAPER_COLOR,
                             2, cv2.LINE_AA)
        inverse = result["inverse_homography"]
        if inverse is not None:
            for index, piece in enumerate(result["pieces"]):
                camera_piece = cv2.perspectiveTransform(
                    piece.astype(np.float32).reshape(1, -1, 2), inverse)[0]
                color = config.PIECE_COLORS[index % len(config.PIECE_COLORS)]
                _draw_polygon(frame, camera_piece, color, 2)
                center = np.round(camera_piece.mean(axis=0)).astype(int)
                cv2.putText(frame, "P%d" % index, tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                            cv2.LINE_AA)
        cv2.putText(frame, status, (8, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2,
                    cv2.LINE_AA)

    def _render_diagnostics(self, gray, binary):
        left_width = self.gray_small.shape[1]
        right_width = self.binary_small.shape[1]
        cv2.resize(gray, (left_width, self.half_height),
                   dst=self.gray_small, interpolation=cv2.INTER_AREA)
        cv2.resize(binary, (right_width, self.half_height),
                   dst=self.binary_small, interpolation=cv2.INTER_NEAREST)
        cv2.cvtColor(self.gray_small, cv2.COLOR_GRAY2BGR,
                     dst=self.gray_color)
        cv2.cvtColor(self.binary_small, cv2.COLOR_GRAY2BGR,
                     dst=self.binary_color)
        self.diag_panel[:, :left_width] = self.gray_color
        self.diag_panel[:, left_width:] = self.binary_color

    def _render_a4(self, result, transforms, commands):
        warped = result["warped_gray"]
        if warped is None:
            self.a4_color.fill(20)
            self.solution_color.fill(20)
        else:
            cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR, dst=self.a4_color)
            cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR,
                         dst=self.solution_color)
            for index, piece in enumerate(result["pieces"]):
                color = config.PIECE_COLORS[index % len(config.PIECE_COLORS)]
                _draw_polygon(self.a4_color, piece, color, 2)
                center = np.round(piece.mean(axis=0)).astype(int)
                cv2.putText(self.a4_color, "P%d" % index, tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                            cv2.LINE_AA)

            if transforms is not None:
                target_points = []
                for piece, transform in zip(result["pieces"], transforms):
                    target = apply_h(piece, transform)
                    target_points.append(target)
                    _draw_polygon(self.solution_color, target,
                                  config.TARGET_COLOR, 2)
                all_target = np.vstack(target_points).astype(np.float32)
                rectangle = cv2.boxPoints(
                    cv2.minAreaRect(all_target)).round().astype(np.int32)
                cv2.polylines(self.solution_color, [rectangle], True,
                              config.TARGET_COLOR, 3, cv2.LINE_AA)

                y = 42
                for command in commands or ():
                    text = "P%d R%+.1f D%.0fpx" % (
                        command["piece"], command["rotation_deg"],
                        command["distance"])
                    cv2.putText(self.solution_color, text, (5, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(self.solution_color, text, (5, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    y += 20

        cv2.resize(self.a4_color, (self.half_width, self.bottom_height),
                   dst=self.a4_panel, interpolation=cv2.INTER_AREA)
        cv2.resize(self.solution_color,
                   (self.right_width, self.bottom_height),
                   dst=self.solution_panel, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _title(image, x, y, text):
        cv2.putText(image, text, (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1,
                    cv2.LINE_AA)

    def _draw_buttons(self):
        cv2.rectangle(self.dashboard, (0, 0),
                      (self.half_width - 1, 35), (120, 80, 35), -1)
        cv2.rectangle(self.dashboard, (self.half_width, 0),
                      (self.width - 1, 35), (45, 105, 50), -1)
        cv2.putText(self.dashboard, "FIND A4",
                    (self.half_width // 2 - 50, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(self.dashboard, "SOLVE ONCE",
                    (self.half_width + self.right_width // 2 - 65, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                    cv2.LINE_AA)

    def render(self, frame, result, transforms, commands, status,
               fps, timings):
        self._annotate_live(frame, result, status)
        cv2.resize(frame, (self.half_width, self.half_height),
                   dst=self.live_panel, interpolation=cv2.INTER_AREA)
        self._render_diagnostics(result["gray"], result["paper_binary"])
        self._render_a4(result, transforms, commands)

        self.dashboard[:self.half_height, :self.half_width] = self.live_panel
        self.dashboard[:self.half_height, self.half_width:] = self.diag_panel
        self.dashboard[self.half_height:, :self.half_width] = self.a4_panel
        self.dashboard[self.half_height:, self.half_width:] = self.solution_panel

        self._title(self.dashboard, 0, 36, "LIVE + A4")
        self._title(self.dashboard, self.half_width, 36, "GRAY | A4 BIN")
        self._title(self.dashboard, 0, self.half_height,
                    "RECTIFIED + PIECES")
        self._title(self.dashboard, self.half_width, self.half_height,
                    "ASSEMBLY")

        cv2.rectangle(self.dashboard, (0, self.height - 30),
                      (self.width, self.height), (0, 0, 0), -1)
        line1 = "FPS %.1f  G %.2f B %.2f A4 %.2f W %.2f P %.2f ms" % (
            fps, timings.get("gray", 0.0), timings.get("paper_bin", 0.0),
            timings.get("paper", 0.0), timings.get("warp", 0.0),
            timings.get("pieces", 0.0))
        line2 = "SOLVE %.2f VIEW %.2f SHOW %.2f TOTAL %.2f ms" % (
            timings.get("solve", 0.0), timings.get("view", 0.0),
            timings.get("show", 0.0), timings.get("total", 0.0))
        cv2.putText(self.dashboard, line1, (5, self.height - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(self.dashboard, line2, (5, self.height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1,
                    cv2.LINE_AA)
        self._draw_buttons()
        return self.dashboard
