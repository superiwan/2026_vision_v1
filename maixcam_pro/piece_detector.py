"""Cached A4 location and one-shot piece detection adapted from D:\\26_new."""

import math
import time

import cv2
import numpy as np

try:
    from . import config
except ImportError:  # MaixVision runs this directory as the project root.
    import config


def _ms(start):
    return (time.perf_counter() - start) * 1000.0


def _find_contours(binary):
    result = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def order_quad(points):
    """Return TL, TR, BR, BL and rotate landscape observations to portrait."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    if np.linalg.norm(ordered[1] - ordered[0]) > np.linalg.norm(
            ordered[3] - ordered[0]):
        ordered = np.roll(ordered, 1, axis=0)
    return ordered


def _corner_cosine(previous, current, following):
    first = previous - current
    second = following - current
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-6:
        return 1.0
    return abs(float(np.dot(first, second)) / denominator)


def _line_intersection(first, second):
    point1, direction1 = first
    point2, direction2 = second
    cross = (direction1[0] * direction2[1]
             - direction1[1] * direction2[0])
    if abs(cross) < 0.05:
        return None
    delta = point2 - point1
    distance = (delta[0] * direction2[1]
                - delta[1] * direction2[0]) / cross
    return point1 + direction1 * distance


def _canonical_polygon(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    if cv2.contourArea(polygon.astype(np.float32), oriented=True) < 0:
        polygon = polygon[::-1].copy()
    start = min(range(len(polygon)),
                key=lambda index: (polygon[index][1], polygon[index][0]))
    return np.roll(polygon, -start, axis=0)


def _contour_polygon_quality(contour, polygon):
    contour_points = contour.reshape(-1, 2)
    all_points = np.vstack((contour_points, polygon))
    minimum = np.floor(all_points.min(axis=0) - 2).astype(int)
    maximum = np.ceil(all_points.max(axis=0) + 2).astype(int)
    width, height = maximum - minimum + 1
    contour_mask = np.zeros((int(height), int(width)), np.uint8)
    polygon_mask = np.zeros_like(contour_mask)
    cv2.fillPoly(contour_mask,
                 [np.round(contour_points - minimum).astype(np.int32)], 1)
    cv2.fillPoly(polygon_mask,
                 [np.round(polygon - minimum).astype(np.int32)], 1)
    intersection = int(np.count_nonzero(contour_mask & polygon_mask))
    union = int(np.count_nonzero(contour_mask | polygon_mask))
    contour_area = max(1.0, cv2.contourArea(contour.astype(np.float32)))
    area_error = abs(cv2.contourArea(polygon.astype(np.float32))
                     - contour_area) / contour_area
    return intersection / max(1, union), area_error


def _fit_contour_edge(contour_points, start_index, end_index):
    if end_index >= start_index:
        segment = contour_points[start_index:end_index + 1]
    else:
        segment = np.vstack((contour_points[start_index:],
                             contour_points[:end_index + 1]))
    if len(segment) < 2:
        return None
    if len(segment) >= 10:
        trim = max(1, len(segment) // 12)
        segment = segment[trim:-trim]
    vx, vy, x0, y0 = cv2.fitLine(
        segment.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    return np.float32((x0, y0)), np.float32((vx, vy))


def _intersect_lines(first, second):
    point1, direction1 = first
    point2, direction2 = second
    cross = float(direction1[0] * direction2[1]
                  - direction1[1] * direction2[0])
    if abs(cross) < 0.08:
        return None
    delta = point2 - point1
    distance = float(delta[0] * direction2[1]
                     - delta[1] * direction2[0]) / cross
    return point1 + direction1 * distance


def _refine_polygon_vertices(contour, polygon):
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    indices = [int(np.argmin(np.sum(
        (contour_points - vertex) ** 2, axis=1))) for vertex in polygon]
    lines = []
    for index in range(len(polygon)):
        line = _fit_contour_edge(
            contour_points, indices[index], indices[(index + 1) % len(polygon)])
        if line is None:
            return polygon.copy()
        lines.append(line)

    refined = []
    for index, original in enumerate(polygon):
        intersection = _intersect_lines(lines[index - 1], lines[index])
        previous_length = np.linalg.norm(polygon[index - 1] - original)
        next_length = np.linalg.norm(
            polygon[(index + 1) % len(polygon)] - original)
        max_shift = max(6.0, 0.18 * min(previous_length, next_length))
        if intersection is None or np.linalg.norm(
                intersection - original) > max_shift:
            intersection = original
        refined.append(intersection)
    return np.asarray(refined, dtype=np.float32)


def approximate_piece(contour):
    """Select the best 3-5 point fit, then refine its supporting lines."""
    perimeter = cv2.arcLength(contour, True)
    best = None
    best_score = -float("inf")
    seen = set()
    for ratio in config.PIECE_APPROX_EPSILON_RATIOS:
        approximation = cv2.approxPolyDP(contour, ratio * perimeter, True)
        if not 3 <= len(approximation) <= 5:
            continue
        signature = tuple(int(value) for value in approximation.reshape(-1))
        if signature in seen:
            continue
        seen.add(signature)
        polygon = approximation[:, 0, :].astype(np.float32)
        polygon = _refine_polygon_vertices(contour, polygon)
        iou, area_error = _contour_polygon_quality(contour, polygon)
        # A tiny extra corner caused by glare or a jagged paper edge can break
        # edge matching. Prefer the simpler fit when its pixel fit is nearly
        # identical, while retaining genuine five-sided pieces.
        score = (iou - 0.35 * area_error
                 - config.PIECE_VERTEX_PENALTY * (len(polygon) - 3))
        if score > best_score:
            best, best_score = polygon, score
    return None if best is None else _canonical_polygon(best)


class A4PieceDetector:
    """Run A4 and piece processing only when the workflow requests a stage."""

    def __init__(self):
        self.paper_generation = 0
        self.paper_locked = False
        self.paper_quad = None
        self.paper_contour = None
        self.homography = None
        self.inverse_homography = None
        self.paper_rect = np.array([
            [[0, 0]],
            [[config.A4_WARP_WIDTH - 1, 0]],
            [[config.A4_WARP_WIDTH - 1, config.A4_WARP_HEIGHT - 1]],
            [[0, config.A4_WARP_HEIGHT - 1]],
        ], dtype=np.int32)

        self.gray = None
        self.blurred = None
        self.lab = None
        self.edge_map = None
        self.paper_binary = None
        self.paper_work = None
        self.candidate_mask = None
        self.candidate_work = None
        self.warped_color = np.empty(
            (config.A4_WARP_HEIGHT, config.A4_WARP_WIDTH, 3), np.uint8)
        self.warped_gray = np.empty(
            (config.A4_WARP_HEIGHT, config.A4_WARP_WIDTH), np.uint8)
        self.piece_binary = np.empty_like(self.warped_gray)
        self.piece_work = np.empty_like(self.warped_gray)
        self.paper_kernel = np.ones(
            (config.PAPER_CLOSE_KERNEL, config.PAPER_CLOSE_KERNEL), np.uint8)
        self.paper_open_kernel = np.ones(
            (config.PAPER_OPEN_KERNEL, config.PAPER_OPEN_KERNEL), np.uint8)
        self.piece_kernel = np.ones(
            (config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)
        self.has_analysis = False
        self.pieces = []
        self.piece_error = None
        self.last_timings = {}
        self.paper_candidate_quad = None
        self.paper_stable_count = 0
        self.paper_search_attempts = 0

    def _ensure_frame_buffers(self, frame):
        shape = frame.shape[:2]
        if self.gray is not None and self.gray.shape == shape:
            return
        self.gray = np.empty(shape, np.uint8)
        self.blurred = np.empty(shape, np.uint8)
        self.lab = np.empty(shape + (3,), np.uint8)
        self.edge_map = np.empty(shape, np.uint8)
        self.paper_binary = np.empty(shape, np.uint8)
        self.paper_work = np.empty(shape, np.uint8)
        self.candidate_mask = np.empty(shape, np.uint8)
        self.candidate_work = np.empty(shape, np.uint8)
        self.clear_a4()

    def _clear_analysis(self):
        self.has_analysis = False
        self.pieces = []
        self.piece_error = None

    def clear_a4(self):
        self.paper_locked = False
        self.paper_quad = None
        self.paper_contour = None
        self.homography = None
        self.inverse_homography = None
        self.paper_candidate_quad = None
        self.paper_stable_count = 0
        self.paper_search_attempts = 0
        self._clear_analysis()

    def prepare_preview(self, frame):
        """Build a neutral-black LAB mask inside the configured search ROI."""
        self._ensure_frame_buffers(frame)
        timings = {}
        start = time.perf_counter()
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.gray)
        cv2.GaussianBlur(
            self.gray, (config.PAPER_BLUR_KERNEL, config.PAPER_BLUR_KERNEL),
            0, dst=self.blurred)
        cv2.cvtColor(frame, cv2.COLOR_BGR2LAB, dst=self.lab)
        timings["gray"] = _ms(start)

        start = time.perf_counter()
        lower = np.uint8((
            round(config.PAPER_LAB_L_MIN * 255.0 / 100.0),
            config.PAPER_LAB_A_MIN + 128,
            config.PAPER_LAB_B_MIN + 128,
        ))
        upper = np.uint8((
            round(config.PAPER_LAB_L_MAX * 255.0 / 100.0),
            config.PAPER_LAB_A_MAX + 128,
            config.PAPER_LAB_B_MAX + 128,
        ))
        cv2.inRange(self.lab, lower, upper, dst=self.paper_binary)
        left, top, right, bottom = self._paper_roi()
        self.paper_binary[:top, :] = 0
        self.paper_binary[bottom:, :] = 0
        self.paper_binary[:, :left] = 0
        self.paper_binary[:, right:] = 0
        cv2.morphologyEx(
            self.paper_binary, cv2.MORPH_CLOSE, self.paper_kernel,
            dst=self.paper_work, iterations=config.PAPER_CLOSE_ITERATIONS)
        cv2.morphologyEx(
            self.paper_work, cv2.MORPH_OPEN, self.paper_open_kernel,
            dst=self.paper_work, iterations=config.PAPER_OPEN_ITERATIONS)
        timings["paper_bin"] = _ms(start)
        timings.update({"paper": 0.0, "warp": 0.0, "pieces": 0.0})
        self.last_timings = timings
        return self.result(timings)

    def _paper_roi(self):
        height, width = self.gray.shape
        margin_x = max(1, round(width * config.PAPER_ROI_MARGIN_X_RATIO))
        margin_y = max(1, round(height * config.PAPER_ROI_MARGIN_Y_RATIO))
        return margin_x, margin_y, width - margin_x, height - margin_y

    def _quad_quality(self, quad):
        height, width = self.gray.shape
        frame_area = height * width
        quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        quad_area = abs(cv2.contourArea(quad))
        area_ratio = quad_area / frame_area
        if not (config.PAPER_MIN_AREA_RATIO <= area_ratio
                <= config.PAPER_MAX_AREA_RATIO):
            return None

        sides = [
            float(np.linalg.norm(quad[(index + 1) % 4] - quad[index]))
            for index in range(4)
        ]
        if min(sides) < 4.0:
            return None
        side_a = (sides[0] + sides[2]) * 0.5
        side_b = (sides[1] + sides[3]) * 0.5
        aspect = max(side_a, side_b) / min(side_a, side_b)
        aspect_error = abs(aspect - config.PAPER_A4_ASPECT_RATIO)
        aspect_error /= config.PAPER_A4_ASPECT_RATIO
        if aspect_error > config.PAPER_ASPECT_REL_TOLERANCE:
            return None

        opposite_a = min(sides[0], sides[2]) / max(sides[0], sides[2])
        opposite_b = min(sides[1], sides[3]) / max(sides[1], sides[3])
        if min(opposite_a, opposite_b) < config.PAPER_MIN_OPPOSITE_SIMILARITY:
            return None
        corner_cosine = max(
            _corner_cosine(quad[index - 1], quad[index],
                           quad[(index + 1) % 4])
            for index in range(4)
        )
        if corner_cosine > config.PAPER_MAX_CORNER_COSINE:
            return None

        roi_left, roi_top, roi_right, roi_bottom = self._paper_roi()
        border = config.PAPER_ROI_BORDER_MARGIN_PX
        if (float(quad[:, 0].min()) <= roi_left + border
                or float(quad[:, 0].max()) >= roi_right - border
                or float(quad[:, 1].min()) <= roi_top + border
                or float(quad[:, 1].max()) >= roi_bottom - border):
            return None

        self.candidate_mask.fill(0)
        cv2.fillPoly(self.candidate_mask,
                     [np.round(quad).astype(np.int32)], 255)
        cv2.bitwise_and(self.paper_work, self.candidate_mask,
                        dst=self.candidate_work)
        fill_ratio = min(1.0, cv2.countNonZero(self.candidate_work)
                         / max(quad_area, 1.0))
        if fill_ratio < config.PAPER_MIN_FILL_RATIO:
            return None

        center = quad.mean(axis=0)
        roi_center = np.float32((
            (roi_left + roi_right) * 0.5,
            (roi_top + roi_bottom) * 0.5,
        ))
        center_error = np.linalg.norm(center - roi_center)
        center_error /= max(width, height)
        score = (area_ratio * fill_ratio
                 - aspect_error * 0.20
                 - center_error * 0.05)
        return score

    def _refine_a4_edges(self, quad):
        """Refine only four LAB-approved sides using local contrast edges."""
        if min(self.gray.shape) < config.PAPER_EDGE_REFINE_MIN_SHORT_SIDE:
            return quad
        cv2.Canny(self.blurred, config.PAPER_EDGE_CANNY_LOW,
                  config.PAPER_EDGE_CANNY_HIGH, self.edge_map)
        height, width = self.gray.shape
        band = max(6, round(min(height, width)
                            * config.PAPER_EDGE_REFINE_BAND_RATIO))
        sample_offset = max(2, round(min(height, width)
                                     * config.PAPER_EDGE_SAMPLE_OFFSET_RATIO))
        center = quad.mean(axis=0)
        lines = []
        for index in range(4):
            start = quad[index].astype(np.float64)
            end = quad[(index + 1) % 4].astype(np.float64)
            vector = end - start
            length = float(np.linalg.norm(vector))
            direction = vector / max(length, 1e-9)
            normal = np.float64((-direction[1], direction[0]))
            if np.dot(center - (start + end) * 0.5, normal) < 0:
                normal = -normal

            x0 = max(0, int(math.floor(min(start[0], end[0]) - band)))
            x1 = min(width, int(math.ceil(max(start[0], end[0]) + band + 1)))
            y0 = max(0, int(math.floor(min(start[1], end[1]) - band)))
            y1 = min(height, int(math.ceil(max(start[1], end[1]) + band + 1)))
            rows, columns = np.nonzero(self.edge_map[y0:y1, x0:x1])
            points = np.column_stack((columns + x0, rows + y0)).astype(
                np.float64)
            if len(points):
                offsets = points - start
                projections = offsets @ direction
                distances = (offsets[:, 0] * direction[1]
                             - offsets[:, 1] * direction[0])
                keep = ((projections > length * 0.04)
                        & (projections < length * 0.96)
                        & (np.abs(distances) < band))
                points = points[keep]
            if len(points):
                inside = np.round(points + normal * sample_offset).astype(int)
                outside = np.round(points - normal * sample_offset).astype(int)
                inside[:, 0] = np.clip(inside[:, 0], 0, width - 1)
                inside[:, 1] = np.clip(inside[:, 1], 0, height - 1)
                outside[:, 0] = np.clip(outside[:, 0], 0, width - 1)
                outside[:, 1] = np.clip(outside[:, 1], 0, height - 1)
                contrast = (
                    self.gray[outside[:, 1], outside[:, 0]].astype(int)
                    - self.gray[inside[:, 1], inside[:, 0]].astype(int)
                )
                points = points[contrast >= config.PAPER_EDGE_MIN_CONTRAST]
            minimum_points = max(
                12, round(length * config.PAPER_EDGE_MIN_POINTS_RATIO))
            if len(points) < minimum_points:
                return quad
            vx, vy, x, y = cv2.fitLine(
                points.astype(np.float32), cv2.DIST_HUBER,
                0, 0.01, 0.01).reshape(-1)
            fitted_direction = np.float64((vx, vy))
            fitted_direction /= max(np.linalg.norm(fitted_direction), 1e-9)
            lines.append((np.float64((x, y)), fitted_direction))

        refined = []
        for index in range(4):
            point = _line_intersection(lines[index - 1], lines[index])
            if point is None:
                return quad
            refined.append(point)
        refined = np.asarray(refined, dtype=np.float32)
        if (not np.isfinite(refined).all()
                or np.max(np.linalg.norm(refined - quad, axis=1)) > band * 2.0
                or not cv2.isContourConvex(
                    np.round(refined).astype(np.int32).reshape(-1, 1, 2))
                or self._quad_quality(refined) is None):
            return quad
        return refined

    def _find_a4_quad(self):
        best = None
        best_score = -float("inf")
        for contour in _find_contours(self.paper_work):
            if cv2.contourArea(contour) < (
                    self.gray.size * config.PAPER_MIN_AREA_RATIO * 0.65):
                continue
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for epsilon in config.PAPER_APPROX_EPSILON_RATIOS:
                approximation = cv2.approxPolyDP(
                    hull, epsilon * perimeter, True)
                if (len(approximation) != 4
                        or not cv2.isContourConvex(approximation)):
                    continue
                quad = order_quad(approximation[:, 0, :])
                score = self._quad_quality(quad)
                if score is None:
                    continue
                if score > best_score:
                    best, best_score = quad, score
                break
        return None if best is None else self._refine_a4_edges(best)

    def find_a4_from_preview(self):
        """Confirm one stable LAB/geometry candidate, then cache its warp."""
        start = time.perf_counter()
        quad = self._find_a4_quad()
        self.paper_generation += 1
        self.paper_search_attempts += 1
        self._clear_analysis()
        self.paper_locked = False
        self.homography = None
        self.inverse_homography = None
        if quad is None:
            self.paper_quad = None
            self.paper_contour = None
            self.paper_candidate_quad = None
            self.paper_stable_count = 0
        else:
            stable_threshold = config.PAPER_STABLE_MAX_CORNER_SHIFT_PX
            stable_threshold *= max(1.0, self.gray.shape[1]
                                    / float(config.CAMERA_WIDTH))
            if self.paper_candidate_quad is None:
                self.paper_stable_count = 1
            else:
                movement = np.max(np.linalg.norm(
                    quad - self.paper_candidate_quad, axis=1))
                self.paper_stable_count = (
                    self.paper_stable_count + 1
                    if movement <= stable_threshold else 1
                )
            self.paper_candidate_quad = quad.copy()
            self.paper_quad = quad
            self.paper_contour = np.round(quad).astype(
                np.int32).reshape(-1, 1, 2)
            if self.paper_stable_count >= config.PAPER_STABLE_FRAMES:
                destination = np.float32((
                    (0, 0),
                    (config.A4_WARP_WIDTH - 1, 0),
                    (config.A4_WARP_WIDTH - 1,
                     config.A4_WARP_HEIGHT - 1),
                    (0, config.A4_WARP_HEIGHT - 1),
                ))
                self.homography = cv2.getPerspectiveTransform(
                    quad, destination)
                self.inverse_homography = cv2.getPerspectiveTransform(
                    destination, quad)
                self.paper_locked = True
        timings = dict(self.last_timings)
        timings["paper"] = _ms(start)
        self.last_timings = timings
        return self.result(timings)

    def find_a4(self, frame):
        self.prepare_preview(frame)
        return self.find_a4_from_preview()

    def _cached_a4_is_valid(self):
        border = np.concatenate((
            self.warped_gray[:8, :].ravel(),
            self.warped_gray[-8:, :].ravel(),
            self.warped_gray[:, :8].ravel(),
            self.warped_gray[:, -8:].ravel(),
        ))
        return (float(np.mean(self.warped_gray < config.PIECE_GRAY_MIN)) > 0.55
                and float(np.mean(border < config.PIECE_GRAY_MIN)) > 0.65)

    def _detect_pieces(self, timings):
        start = time.perf_counter()
        cv2.threshold(self.warped_gray, config.PIECE_GRAY_MIN, 255,
                      cv2.THRESH_BINARY, dst=self.piece_binary)
        cv2.morphologyEx(self.piece_binary, cv2.MORPH_CLOSE,
                         self.piece_kernel, dst=self.piece_work, iterations=1)
        self.piece_binary[:, :] = self.piece_work
        inset = config.A4_BORDER_INSET
        self.piece_binary[:inset, :] = 0
        self.piece_binary[-inset:, :] = 0
        self.piece_binary[:, :inset] = 0
        self.piece_binary[:, -inset:] = 0
        timings["binary_morph"] = _ms(start)

        start = time.perf_counter()
        a4_area = config.A4_WARP_WIDTH * config.A4_WARP_HEIGHT
        contours = [
            contour for contour in _find_contours(self.piece_binary)
            if a4_area * config.PIECE_MIN_AREA_RATIO
            <= cv2.contourArea(contour)
            <= a4_area * config.PIECE_MAX_AREA_RATIO
        ]
        contours.sort(key=cv2.contourArea, reverse=True)
        timings["contours"] = _ms(start)

        start = time.perf_counter()
        pieces = []
        for contour in contours:
            polygon = approximate_piece(contour)
            if polygon is not None:
                pieces.append(polygon)
            if len(pieces) == 4:
                break
        timings["approx_poly"] = _ms(start)
        timings["pieces"] = (timings["binary_morph"]
                              + timings["contours"]
                              + timings["approx_poly"])
        return pieces

    def analyze_cached_a4(self, frame):
        """Run the one-shot vision stage; normal preview does no such work."""
        timings = dict(self.last_timings)
        timings.update({"warp": 0.0, "pieces": 0.0})
        if not self.paper_locked:
            self.piece_error = "A4 not found"
            return self.result(timings)

        start = time.perf_counter()
        cv2.warpPerspective(
            frame, self.homography,
            (config.A4_WARP_WIDTH, config.A4_WARP_HEIGHT),
            dst=self.warped_color, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.cvtColor(self.warped_color, cv2.COLOR_BGR2GRAY,
                     dst=self.warped_gray)
        timings["warp"] = _ms(start)
        if not self._cached_a4_is_valid():
            self._clear_analysis()
            self.piece_error = "A4 cache invalid"
            self.last_timings = timings
            return self.result(timings)

        self.pieces = self._detect_pieces(timings)
        self.has_analysis = True
        self.piece_error = None
        if not 1 <= len(self.pieces) <= 4:
            self.piece_error = "detected %d pieces; need 1-4" % len(self.pieces)
        self.last_timings = timings
        return self.result(timings)

    def result(self, timings=None):
        return {
            "gray": self.gray,
            "paper_binary": self.paper_work,
            "paper_locked": self.paper_locked,
            "paper_searched": self.paper_search_attempts > 0,
            "paper_candidate": self.paper_quad is not None,
            "paper_stable_count": self.paper_stable_count,
            "paper_stable_required": config.PAPER_STABLE_FRAMES,
            "paper_search_attempts": self.paper_search_attempts,
            "paper_search_exhausted": (
                not self.paper_locked
                and self.paper_search_attempts >= config.PAPER_SEARCH_MAX_FRAMES
            ),
            "paper_generation": self.paper_generation,
            "paper_contour": self.paper_contour,
            "paper_rect": self.paper_rect,
            "homography": self.homography,
            "inverse_homography": self.inverse_homography,
            "warped_gray": self.warped_gray if self.has_analysis else None,
            "piece_binary": self.piece_binary if self.has_analysis else None,
            "pieces": self.pieces,
            "piece_error": self.piece_error,
            "timings": self.last_timings if timings is None else timings,
        }


def detect_pieces(frame):
    """One-shot PC/test helper using the same FIND then SOLVE vision path."""
    detector = A4PieceDetector()
    result = detector.result()
    for _attempt in range(config.PAPER_SEARCH_MAX_FRAMES):
        result = detector.find_a4(frame)
        if result["paper_locked"] or result["paper_search_exhausted"]:
            break
    if not result["paper_locked"]:
        raise RuntimeError("未找到黑色 A4")
    result = detector.analyze_cached_a4(frame)
    if result["piece_error"]:
        raise RuntimeError(result["piece_error"])
    return result["pieces"], result["paper_rect"], result["piece_binary"]
