"""Algorithm 2: enumerate equal-edge contour merges until one rectangle remains."""

import itertools
import math

import cv2
import numpy as np

try:
    from . import config
    from .puzzle_solver import (
        _assembly_quality,
        _target_transform,
        align_edge,
        apply_h,
        edges,
    )
except ImportError:  # MaixVision runs this directory as the project root.
    import config
    from puzzle_solver import (
        _assembly_quality,
        _target_transform,
        align_edge,
        apply_h,
        edges,
    )


class CompositeContour:
    """One merged boundary plus every original piece's accumulated transform."""

    def __init__(self, polygon, transforms, sources, history=(), score=0.0):
        self.polygon = _canonical_polygon(polygon)
        self.transforms = transforms
        self.sources = frozenset(sources)
        self.history = tuple(history)
        self.score = float(score)


def _canonical_polygon(polygon):
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    kept = []
    for point in polygon:
        if not kept or np.linalg.norm(point - kept[-1]) > 1e-6:
            kept.append(point)
    polygon = np.asarray(kept, dtype=np.float64)
    if len(polygon) > 1 and np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-6:
        polygon = polygon[:-1]
    if cv2.contourArea(polygon.astype(np.float32), oriented=True) < 0:
        polygon = polygon[::-1].copy()
    start = min(range(len(polygon)),
                key=lambda index: (polygon[index][1], polygon[index][0]))
    return np.roll(polygon, -start, axis=0)


def _interior_angle(polygon, index):
    """Return the 0..2pi interior angle for a consistently wound polygon."""
    previous = polygon[index - 1]
    current = polygon[index]
    following = polygon[(index + 1) % len(polygon)]
    incoming = current - previous
    outgoing = following - current
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    turn = math.atan2(cross, float(np.dot(incoming, outgoing)))
    if cv2.contourArea(polygon.astype(np.float32), oriented=True) < 0:
        turn = -turn
    angle = math.pi - turn
    while angle <= 0:
        angle += 2.0 * math.pi
    while angle > 2.0 * math.pi:
        angle -= 2.0 * math.pi
    return angle


def _angle_error(first, second):
    total = first + second
    return min(abs(total - math.pi / 2.0), abs(total - math.pi))


def _simplify_collinear(polygon):
    polygon = _canonical_polygon(polygon)
    tolerance = math.radians(config.MERGE_COLLINEAR_TOLERANCE_DEG)
    changed = True
    while changed and len(polygon) > 3:
        changed = False
        kept = []
        for index, point in enumerate(polygon):
            angle = _interior_angle(polygon, index)
            if abs(angle - math.pi) <= tolerance:
                changed = True
            else:
                kept.append(point)
        if changed:
            if len(kept) < 3:
                return polygon
            polygon = _canonical_polygon(kept)
    return polygon


def _segments_cross(a, b, c, d, epsilon=1e-7):
    def orientation(p, q, r):
        return ((q[0] - p[0]) * (r[1] - p[1])
                - (q[1] - p[1]) * (r[0] - p[0]))

    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return ((first > epsilon and second < -epsilon
             or first < -epsilon and second > epsilon)
            and (third > epsilon and fourth < -epsilon
                 or third < -epsilon and fourth > epsilon))


def _is_simple_polygon(polygon):
    count = len(polygon)
    for first in range(count):
        a = polygon[first]
        b = polygon[(first + 1) % count]
        for second in range(first + 1, count):
            if second in (first, (first + 1) % count):
                continue
            if first == 0 and second == count - 1:
                continue
            c = polygon[second]
            d = polygon[(second + 1) % count]
            if _segments_cross(a, b, c, d):
                return False
    return True


def _overlap_ratio(first, second):
    all_points = np.vstack((first, second))
    minimum = np.floor(all_points.min(axis=0) - 2).astype(int)
    maximum = np.ceil(all_points.max(axis=0) + 2).astype(int)
    width, height = maximum - minimum + 1
    if width <= 0 or height <= 0:
        return 1.0
    masks = []
    for polygon in (first, second):
        mask = np.zeros((int(height), int(width)), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon - minimum).astype(np.int32)], 1)
        masks.append(mask)
    overlap = float(np.count_nonzero(masks[0] & masks[1]))
    smaller = max(1.0, min(np.count_nonzero(masks[0]),
                           np.count_nonzero(masks[1])))
    return overlap / smaller


def _merge_boundary(first, second_world):
    """Cancel every reversed shared edge, then trace the remaining outer cycle."""
    nodes = [point.copy() for point in first]
    first_nodes = list(range(len(first)))
    second_nodes = []
    for point in second_world:
        distances = [np.linalg.norm(point - node) for node in nodes[:len(first)]]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= config.MERGE_ENDPOINT_TOLERANCE_PX:
            second_nodes.append(nearest)
        else:
            second_nodes.append(len(nodes))
            nodes.append(point.copy())

    directed = []
    for polygon_nodes in (first_nodes, second_nodes):
        for index, start in enumerate(polygon_nodes):
            directed.append((start, polygon_nodes[(index + 1) % len(
                polygon_nodes)]))

    remaining = []
    used = [False] * len(directed)
    for index, edge in enumerate(directed):
        if used[index]:
            continue
        reverse_index = None
        for candidate in range(index + 1, len(directed)):
            if not used[candidate] and directed[candidate] == (edge[1], edge[0]):
                reverse_index = candidate
                break
        if reverse_index is None:
            remaining.append(edge)
        else:
            used[reverse_index] = True
        used[index] = True

    outgoing = {}
    incoming = {}
    for start, end in remaining:
        outgoing.setdefault(start, []).append(end)
        incoming.setdefault(end, []).append(start)
    active_nodes = set(outgoing) | set(incoming)
    if (not active_nodes
            or any(len(outgoing.get(node, ())) != 1 for node in active_nodes)
            or any(len(incoming.get(node, ())) != 1 for node in active_nodes)):
        return None

    start = min(active_nodes, key=lambda index: (nodes[index][1], nodes[index][0]))
    ordered = []
    current = start
    for _ in range(len(remaining)):
        if current in ordered:
            return None
        ordered.append(current)
        current = outgoing[current][0]
    if current != start or len(ordered) != len(active_nodes):
        return None
    return _simplify_collinear(np.asarray([nodes[index] for index in ordered]))


def _merge_options(first, second):
    """Enumerate every valid equal-edge and 90/180-degree merge."""
    tolerance = math.radians(config.MERGE_ANGLE_TOLERANCE_DEG)
    options = []
    for edge_first, (p0, p1) in enumerate(edges(first.polygon)):
        length_first = np.linalg.norm(p1 - p0)
        for edge_second, (q0, q1) in enumerate(edges(second.polygon)):
            length_second = np.linalg.norm(q1 - q0)
            relative_error = abs(length_first - length_second) / max(
                length_first, length_second, 1e-9)
            if relative_error >= config.EDGE_LENGTH_TOLERANCE:
                continue

            endpoint_error_0 = _angle_error(
                _interior_angle(first.polygon, edge_first),
                _interior_angle(second.polygon,
                                (edge_second + 1) % len(second.polygon)))
            endpoint_error_1 = _angle_error(
                _interior_angle(first.polygon,
                                (edge_first + 1) % len(first.polygon)),
                _interior_angle(second.polygon, edge_second))
            angle_error = min(endpoint_error_0, endpoint_error_1)
            if angle_error > tolerance:
                continue

            # Reverse the shared-edge direction: Q0 -> P1 and Q1 -> P0.
            transform = align_edge(q0, q1, p1, p0)
            second_world = apply_h(second.polygon, transform)
            overlap = _overlap_ratio(first.polygon, second_world)
            if overlap > config.MERGE_MAX_OVERLAP_RATIO:
                continue

            polygon = _merge_boundary(first.polygon, second_world)
            if (polygon is None or len(polygon) < 3
                    or not _is_simple_polygon(polygon)):
                continue
            expected_area = (abs(cv2.contourArea(first.polygon.astype(np.float32)))
                             + abs(cv2.contourArea(
                                 second_world.astype(np.float32))))
            actual_area = abs(cv2.contourArea(polygon.astype(np.float32)))
            area_error = abs(actual_area - expected_area) / max(
                expected_area, 1.0)
            if area_error > config.MERGE_MAX_AREA_ERROR_RATIO:
                continue

            transforms = dict(first.transforms)
            transforms.update({
                source: transform @ source_transform
                for source, source_transform in second.transforms.items()
            })
            record = (
                float(relative_error),
                min(first.sources),
                edge_first,
                min(second.sources),
                edge_second,
            )
            score = (first.score + second.score
                     + relative_error * 10.0
                     + angle_error / max(tolerance, 1e-9)
                     + area_error * 5.0
                     + overlap * 5.0)
            options.append(CompositeContour(
                polygon,
                transforms,
                first.sources | second.sources,
                first.history + second.history + (record,),
                score,
            ))
    return options


def _polygon_signature(polygon):
    features = []
    for index, (start, end) in enumerate(edges(polygon)):
        length = round(float(np.linalg.norm(end - start)), 1)
        angle = round(math.degrees(_interior_angle(polygon, index)), 1)
        features.append((length, angle))
    rotations = [tuple(features[index:] + features[:index])
                 for index in range(len(features))]
    return min(rotations)


def _state_signature(state):
    return tuple(sorted(
        (tuple(sorted(contour.sources)), _polygon_signature(contour.polygon))
        for contour in state
    ))


def _deduplicate(states):
    unique = {}
    for state in states:
        signature = _state_signature(state)
        score = sum(contour.score for contour in state)
        current = unique.get(signature)
        if current is None or score < current[0]:
            unique[signature] = (score, state)
    ordered = sorted(unique.values(), key=lambda item: item[0])
    return [state for _score, state in ordered[:config.MERGE_MAX_STATES]]


def enumerate_rectangle_assemblies(pieces):
    """Return all retained one-contour candidates after N-1 merge rounds."""
    initial = tuple(
        CompositeContour(
            piece,
            {index: np.eye(3, dtype=np.float64)},
            (index,),
        )
        for index, piece in enumerate(pieces)
    )
    states = [initial]
    for _round in range(max(0, len(pieces) - 1)):
        next_states = []
        for state in states:
            for first_index, second_index in itertools.combinations(
                    range(len(state)), 2):
                first = state[first_index]
                second = state[second_index]
                remaining = [
                    contour for index, contour in enumerate(state)
                    if index not in (first_index, second_index)
                ]
                for merged in _merge_options(first, second):
                    next_states.append(tuple(remaining + [merged]))
        if not next_states:
            return []
        states = _deduplicate(next_states)
    return [state[0] for state in states if len(state) == 1]


def solve(pieces, paper):
    """Return transforms from the best iterative contour-merge rectangle."""
    if not 1 <= len(pieces) <= 4:
        raise ValueError("碎片数量必须为 1～4")

    best = None
    for assembly in enumerate_rectangle_assemblies(pieces):
        if set(assembly.transforms) != set(range(len(pieces))):
            continue
        transforms = [assembly.transforms[index] for index in range(len(pieces))]
        assembled = [apply_h(piece, transform)
                     for piece, transform in zip(pieces, transforms)]
        quality, fill_ratio = _assembly_quality(
            assembled, assembly.history, 0.0)
        rectangle = _simplify_collinear(assembly.polygon)
        if len(rectangle) != 4 or fill_ratio < config.MIN_RECTANGLE_FILL:
            continue
        angle_errors = [
            abs(_interior_angle(rectangle, index) - math.pi / 2.0)
            for index in range(4)
        ]
        if max(angle_errors) > math.radians(
                config.MERGE_RECTANGLE_ANGLE_TOLERANCE_DEG):
            continue
        right_angle_error = sum(angle_errors)
        score = quality + assembly.score * 100.0 + right_angle_error * 100.0
        if best is None or score < best[0]:
            best = (score, fill_ratio, transforms, assembly.history)

    if best is None:
        raise RuntimeError("算法2未找到可连续合并成矩形的轮廓组合")

    _score, fill_ratio, transforms, history = best
    final = _target_transform(pieces, transforms, paper)
    return final, history, fill_ratio
