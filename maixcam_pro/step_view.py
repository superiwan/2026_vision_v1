"""Single-page runtime views for the MaixCAM Pro 640x480 screen."""

import cv2
import numpy as np

try:
    from . import config
    from .puzzle_solver import apply_h
    from .workflow import PuzzleWorkflow
except ImportError:  # MaixVision runs this directory as the project root.
    import config
    from puzzle_solver import apply_h
    from workflow import PuzzleWorkflow


WHITE = (245, 245, 245)
MUTED = (170, 176, 184)
PANEL = (24, 27, 32)
SUCCESS = (68, 210, 105)
WARNING = (0, 190, 255)


def _draw_polygon(target, polygon, color, thickness=2):
    points = np.round(polygon).astype(np.int32)
    cv2.polylines(target, [points], True, color, thickness, cv2.LINE_AA)


class ReleaseButton:
    """Recognize a click only when one press stays inside the active button."""

    def __init__(self):
        self.pressed = False
        self.started_inside = False
        self.last_inside = False

    @staticmethod
    def _inside(x, y, rect):
        left, top, width, height = rect
        return left <= x < left + width and top <= y < top + height

    def update(self, x, y, pressed, rect):
        inside = self._inside(x, y, rect)
        if pressed:
            if not self.pressed:
                self.started_inside = inside
            self.pressed = True
            self.last_inside = inside
            return False
        if not self.pressed:
            return False
        clicked = self.started_inside and self.last_inside
        self.pressed = False
        self.started_inside = False
        self.last_inside = False
        return clicked

    def read(self, touch, rect):
        return self.update(*touch.read(), rect)


class StepView:
    """Render one primary view at a time, with large finger-safe controls."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        margin = config.UI_MARGIN
        self.action_rect = (
            margin,
            height - config.UI_BUTTON_HEIGHT - margin,
            width - margin * 2,
            config.UI_BUTTON_HEIGHT,
        )
        self.canvas = np.empty((height, width, 3), np.uint8)
        self.a4_color = np.empty(
            (config.A4_WARP_HEIGHT, config.A4_WARP_WIDTH, 3), np.uint8)

    @staticmethod
    def _put(target, text, position, scale=0.55, color=WHITE,
             thickness=1):
        cv2.putText(target, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def _draw_header(self, target, workflow, fps):
        cv2.rectangle(target, (0, 0),
                      (self.width, config.UI_HEADER_HEIGHT), PANEL, -1)
        self._put(target, "PUZZLE VISION", (16, 27), 0.62, WHITE, 2)
        self._put(target, "ALG %d  FPS %.1f" % (workflow.algorithm, fps),
                  (self.width - 160, 27), 0.48, SUCCESS, 1)

        gap = 7
        width = (self.width - 32 - gap * 2) // 3
        y = config.UI_HEADER_HEIGHT - 8
        for index in range(3):
            color = SUCCESS if index < workflow.progress_step else (70, 74, 82)
            cv2.rectangle(target, (16 + index * (width + gap), y),
                          (16 + index * (width + gap) + width, y + 3),
                          color, -1)

    def _draw_action(self, target, label, color):
        x, y, width, height = self.action_rect
        cv2.rectangle(target, (x, y), (x + width, y + height), color, -1)
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        self._put(target, label, (text_x, text_y), 0.82, WHITE, 2)

    def _draw_a4_guide(self, target):
        available_top = config.UI_HEADER_HEIGHT + 12
        available_bottom = self.action_rect[1] - 12
        available_height = available_bottom - available_top
        if config.PAPER_GUIDE_LANDSCAPE:
            guide_width = min(
                self.width - config.UI_MARGIN * 4,
                int(available_height * config.PAPER_A4_ASPECT_RATIO),
            )
            guide_height = int(guide_width / config.PAPER_A4_ASPECT_RATIO)
        else:
            guide_height = available_height
            guide_width = int(guide_height / config.PAPER_A4_ASPECT_RATIO)
        left = (self.width - guide_width) // 2
        right = left + guide_width
        top = available_top + (available_height - guide_height) // 2
        bottom = top + guide_height
        length = 28
        for x1, y1, x2, y2 in (
                (left, top, left + length, top),
                (left, top, left, top + length),
                (right, top, right - length, top),
                (right, top, right, top + length),
                (left, bottom, left + length, bottom),
                (left, bottom, left, bottom - length),
                (right, bottom, right - length, bottom),
                (right, bottom, right, bottom - length)):
            cv2.line(target, (x1, y1), (x2, y2), WARNING, 3, cv2.LINE_AA)

    def _render_live(self, frame, workflow, fps):
        self._draw_header(frame, workflow, fps)
        if workflow.stage in (PuzzleWorkflow.READY, PuzzleWorkflow.LOCATE_A4):
            self._draw_a4_guide(frame)

        contour = workflow.result.get("paper_contour")
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, config.PAPER_COLOR,
                             3, cv2.LINE_AA)

        if workflow.stage == PuzzleWorkflow.READY:
            self._put(frame, "ALIGN THE FULL BLACK A4 INSIDE THE GUIDE",
                      (98, self.action_rect[1] - 14), 0.48, WHITE, 1)
            self._draw_action(frame, workflow.action_label, (38, 118, 70))
        elif workflow.stage == PuzzleWorkflow.LOCATE_A4:
            stable = workflow.result.get("paper_stable_count", 0)
            required = workflow.result.get("paper_stable_required", 1)
            self._put(frame, "STEP 1/3  LOCATING A4... %d/%d" % (
                stable, required),
                      (18, self.height - 22), 0.62, WARNING, 2)
        elif workflow.stage == PuzzleWorkflow.DETECT_PIECES:
            self._put(frame, "STEP 1/3 DONE  -  DETECTING PIECES...",
                      (18, self.height - 22), 0.58, SUCCESS, 2)
        else:
            message = (workflow.error or "CAPTURE FAILED")[:58]
            self._put(frame, message, (18, self.action_rect[1] - 14),
                      0.52, config.ERROR_COLOR, 2)
            self._draw_action(frame, workflow.action_label, (55, 65, 155))
        return frame

    def _prepare_a4(self, workflow):
        warped = workflow.result.get("warped_gray")
        if warped is None:
            self.a4_color.fill(18)
        else:
            cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR, dst=self.a4_color)

        for index, piece in enumerate(workflow.result.get("pieces") or ()):
            color = config.PIECE_COLORS[index % len(config.PIECE_COLORS)]
            _draw_polygon(self.a4_color, piece, color, 3)
            center = np.round(piece.mean(axis=0)).astype(int)
            self._put(self.a4_color, "P%d" % index, tuple(center),
                      0.58, color, 2)

        if workflow.transforms is not None:
            target_points = []
            for piece, transform in zip(
                    workflow.result["pieces"], workflow.transforms):
                target = apply_h(piece, transform)
                target_points.append(target)
                _draw_polygon(self.a4_color, target, config.TARGET_COLOR, 3)
            if target_points:
                rectangle = cv2.boxPoints(cv2.minAreaRect(
                    np.vstack(target_points).astype(np.float32)))
                cv2.polylines(self.a4_color,
                              [np.round(rectangle).astype(np.int32)], True,
                              config.TARGET_COLOR, 4, cv2.LINE_AA)

    def _render_a4(self, workflow, fps):
        self.canvas.fill(16)
        self._draw_header(self.canvas, workflow, fps)
        self._prepare_a4(workflow)

        has_action = workflow.action_label is not None
        top = config.UI_HEADER_HEIGHT + 12
        bottom = self.action_rect[1] - 12 if has_action else self.height - 26
        available_height = max(1, bottom - top)
        max_image_width = int(self.width * 0.58)
        scale = min(available_height / config.A4_WARP_HEIGHT,
                    max_image_width / config.A4_WARP_WIDTH)
        draw_width = max(1, int(config.A4_WARP_WIDTH * scale))
        draw_height = max(1, int(config.A4_WARP_HEIGHT * scale))
        left = 22
        image_top = top + (available_height - draw_height) // 2
        target = self.canvas[
            image_top:image_top + draw_height, left:left + draw_width]
        cv2.resize(self.a4_color, (draw_width, draw_height), dst=target,
                   interpolation=cv2.INTER_AREA)
        cv2.rectangle(self.canvas, (left - 1, image_top - 1),
                      (left + draw_width, image_top + draw_height),
                      (90, 95, 105), 1)

        info_x = left + draw_width + 20
        if workflow.stage == PuzzleWorkflow.SOLVE_PUZZLE:
            self._put(self.canvas, "STEP 2/3", (info_x, top + 38),
                      0.72, WARNING, 2)
            self._put(self.canvas, "%d PIECES FOUND" % len(
                workflow.result["pieces"]), (info_x, top + 78),
                0.64, WHITE, 2)
            self._put(self.canvas, "MATCHING EDGES", (info_x, top + 116),
                      0.50, MUTED, 1)
            self._put(self.canvas, "SOLVING...", (info_x, top + 144),
                      0.50, MUTED, 1)
        elif workflow.stage == PuzzleWorkflow.COMPLETE:
            self._put(self.canvas, "DONE", (info_x, top + 42),
                      0.90, SUCCESS, 3)
            self._put(self.canvas, "%d PIECES" % len(workflow.commands or ()),
                      (info_x, top + 86), 0.62, WHITE, 2)
            self._put(self.canvas, "RECT %.1f%%" % (
                (workflow.fill_ratio or 0.0) * 100.0),
                (info_x, top + 120), 0.56, WHITE, 1)
            self._put(self.canvas, "TIME %.0f ms" % workflow.elapsed_ms,
                      (info_x, top + 150), 0.56, WHITE, 1)
            self._put(self.canvas, "POSES IN TERMINAL",
                      (info_x, top + 184), 0.43, MUTED, 1)
            self._draw_action(self.canvas, workflow.action_label,
                              (38, 118, 70))
        else:
            message = (workflow.error or "SOLVE FAILED")[:32]
            self._put(self.canvas, "FAILED", (info_x, top + 42),
                      0.76, config.ERROR_COLOR, 2)
            self._put(self.canvas, message, (info_x, top + 82),
                      0.45, WHITE, 1)
            self._put(self.canvas, "CHECK A4 / PIECES",
                      (info_x, top + 116), 0.45, MUTED, 1)
            self._draw_action(self.canvas, workflow.action_label,
                              (55, 65, 155))
        return self.canvas

    def render(self, frame, workflow, fps):
        """Return the borrowed live frame or the preallocated A4 canvas."""
        if workflow.stage in (
                PuzzleWorkflow.READY,
                PuzzleWorkflow.LOCATE_A4,
                PuzzleWorkflow.DETECT_PIECES):
            return self._render_live(frame, workflow, fps)
        if (workflow.stage == PuzzleWorkflow.ERROR
                and workflow.result.get("warped_gray") is None):
            return self._render_live(frame, workflow, fps)
        return self._render_a4(workflow, fps)
