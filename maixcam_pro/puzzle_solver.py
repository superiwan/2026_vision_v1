"""Geometry-only edge matching, adjacency recovery and rectangle assembly."""

import itertools
import math

import cv2
import numpy as np

try:
    from . import config
except ImportError:  # Run directly in MaixVision with this folder as project root.
    import config


def rigid(angle, tx=0.0, ty=0.0):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([
        [cosine, -sine, tx],
        [sine, cosine, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def apply_h(points, transform):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ transform.T
    return mapped[:, :2] / mapped[:, 2, None]


def edges(polygon):
    return [(polygon[i], polygon[(i + 1) % len(polygon)])
            for i in range(len(polygon))]


def align_edge(src_a, src_b, dst_a, dst_b):
    """Rigid transform mapping src_a/src_b to dst_a/dst_b."""
    source = src_b - src_a
    target = dst_b - dst_a
    angle = (math.atan2(target[1], target[0])
             - math.atan2(source[1], source[0]))
    transform = rigid(angle)
    mapped = apply_h(np.array([src_a]), transform)[0]
    transform[:2, 2] = dst_a - mapped
    return transform


def candidate_matchings(pieces):
    """Shortlist equal-length edges from different pieces."""
    all_edges = {(piece_index, edge_index): edge
                 for piece_index, piece in enumerate(pieces)
                 for edge_index, edge in enumerate(edges(piece))}
    candidates = []
    for (i, ei), (j, ej) in itertools.combinations(all_edges, 2):
        if i == j:
            continue
        a, b = all_edges[(i, ei)]
        c, d = all_edges[(j, ej)]
        length_a = np.linalg.norm(b - a)
        length_b = np.linalg.norm(d - c)
        relative_error = abs(length_a - length_b) / max(length_a, length_b)
        if relative_error < config.EDGE_LENGTH_TOLERANCE:
            candidates.append((relative_error, i, ei, j, ej))
    candidates.sort()
    return candidates[:config.MAX_EDGE_CANDIDATES]


def matching_sets(pieces):
    """Recover the same degree-constrained adjacency graph as the source project."""
    count = len(pieces)
    if count == 1:
        yield ()
        return

    candidates = candidate_matchings(pieces)
    pair_count = 1 if count == 2 else count
    required_degree = [1] * count if count == 2 else [2] * count
    for combination in itertools.combinations(candidates, pair_count):
        used_edges = set()
        degree = [0] * count
        graph = [set() for _ in range(count)]
        valid = True
        for _, i, ei, j, ej in combination:
            if (i, ei) in used_edges or (j, ej) in used_edges:
                valid = False
                break
            used_edges.update(((i, ei), (j, ej)))
            degree[i] += 1
            degree[j] += 1
            graph[i].add(j)
            graph[j].add(i)
        if not valid or degree != required_degree:
            continue

        visited, stack = {0}, [0]
        while stack:
            for neighbour in graph[stack.pop()]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        if len(visited) == count:
            yield combination


def optimize_pose_graph(pieces, matches, initial):
    """Distribute closed-loop endpoint error; copied in simplified form."""
    if len(pieces) < 3:
        return initial

    def pack(poses):
        values = []
        for pose in poses[1:]:
            values.extend((math.atan2(pose[1, 0], pose[0, 0]),
                           pose[0, 2], pose[1, 2]))
        return np.asarray(values, dtype=np.float64)

    def unpack(values):
        poses = [initial[0]]
        for index in range(len(pieces) - 1):
            theta, tx, ty = values[3 * index:3 * index + 3]
            poses.append(rigid(theta, tx, ty))
        return poses

    def residual(values):
        poses = unpack(values)
        result = []
        for _, i, ei, j, ej in matches:
            ia, ib = edges(pieces[i])[ei]
            ja, jb = edges(pieces[j])[ej]
            world_i = apply_h(np.array([ia, ib]), poses[i])
            world_j = apply_h(np.array([jb, ja]), poses[j])
            result.extend((world_i - world_j).ravel())
        return np.asarray(result)

    values = pack(initial)
    for _ in range(15):
        current = residual(values)
        jacobian = np.empty((len(current), len(values)))
        for index in range(len(values)):
            step = 1e-5 if index % 3 == 0 else 1e-3
            shifted = values.copy()
            shifted[index] += step
            jacobian[:, index] = (residual(shifted) - current) / step
        delta, *_ = np.linalg.lstsq(jacobian, -current, rcond=None)
        values += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return unpack(values)


def _assembly_quality(assembled, matches, closure_error):
    all_points = np.vstack(assembled)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    shift = -minimum + 6
    width, height = np.ceil(maximum - minimum + 12).astype(int)
    masks = []
    for polygon in assembled:
        mask = np.zeros((height, width), np.uint8)
        points = np.round(polygon + shift).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask)

    total = sum(masks)
    overlap = float(np.count_nonzero(total > 1))
    union = (total > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    rectangle_area = max(1.0, rect[1][0] * rect[1][1])
    contour_area = cv2.contourArea(contour)
    fill_error = max(0.0, rectangle_area - contour_area)
    fill_ratio = contour_area / rectangle_area
    match_error = sum(match[0] for match in matches) * 5000.0
    score = closure_error * 5.0 + overlap * 3.0 + fill_error + match_error
    return score, fill_ratio


def assemble_from_matches(pieces, matches):
    adjacency = [[] for _ in pieces]
    for _, i, ei, j, ej in matches:
        adjacency[i].append((j, ei, ej))
        adjacency[j].append((i, ej, ei))

    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3)
    stack = [0]
    closure_error = 0.0
    while stack:
        i = stack.pop()
        for j, edge_i, edge_j in adjacency[i]:
            ia, ib = edges(pieces[i])[edge_i]
            ja, jb = edges(pieces[j])[edge_j]
            world_a, world_b = apply_h(
                np.array([ia, ib]), transforms[i])
            proposed = align_edge(ja, jb, world_b, world_a)
            if transforms[j] is None:
                transforms[j] = proposed
                stack.append(j)
            else:
                error = apply_h(pieces[j], proposed) - apply_h(
                    pieces[j], transforms[j])
                closure_error += np.linalg.norm(error, axis=1).mean()

    if any(transform is None for transform in transforms):
        return None
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    score, fill_ratio = _assembly_quality(assembled, matches, closure_error)
    return score, fill_ratio, transforms


def _target_transform(pieces, transforms, paper):
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    all_points = np.vstack(assembled).astype(np.float32)
    _, size, angle_degrees = cv2.minAreaRect(all_points)
    if size[0] < size[1]:
        angle_degrees += 90.0

    normalize = rigid(math.radians(-angle_degrees))
    rotated = apply_h(all_points, normalize)
    minimum, maximum = rotated.min(axis=0), rotated.max(axis=0)
    if (maximum - minimum)[0] < (maximum - minimum)[1]:
        normalize = rigid(math.radians(90.0 - angle_degrees))
        rotated = apply_h(all_points, normalize)
        minimum, maximum = rotated.min(axis=0), rotated.max(axis=0)

    x, y, width, height = cv2.boundingRect(paper.astype(np.int32))
    target_center = np.array([
        x + width * config.TARGET_CENTER_X_RATIO,
        y + height * config.TARGET_CENTER_Y_RATIO,
    ])
    recovered_size = maximum - minimum
    margin = min(width, height) * config.TARGET_MARGIN_RATIO
    target_origin = target_center - recovered_size / 2.0
    target_origin[0] = min(max(target_origin[0], x + margin),
                           x + width - margin - recovered_size[0])
    target_origin[1] = min(max(target_origin[1], y + margin),
                           y + height - margin - recovered_size[1])
    translate = rigid(0.0, *(target_origin - minimum))
    return [translate @ normalize @ transform for transform in transforms]


def solve(pieces, paper):
    """Return final per-piece 3x3 transforms, selected matches and fill ratio."""
    if not 1 <= len(pieces) <= 4:
        raise ValueError("碎片数量必须为 1～4")

    best = None
    for matches in matching_sets(pieces):
        result = assemble_from_matches(pieces, matches)
        if result is not None and (best is None or result[0] < best[0]):
            best = (*result, matches)
    if best is None:
        raise RuntimeError("未找到满足边长和邻接关系的矩形拼接")

    _, fill_ratio, transforms, matches = best
    transforms = optimize_pose_graph(pieces, matches, transforms)
    assembled = [apply_h(piece, transform)
                 for piece, transform in zip(pieces, transforms)]
    _, fill_ratio = _assembly_quality(assembled, matches, 0.0)
    if fill_ratio < config.MIN_RECTANGLE_FILL:
        raise RuntimeError("最佳拼接的矩形填充率仅 %.1f%%" % (fill_ratio * 100.0))
    final = _target_transform(pieces, transforms, paper)
    return final, matches, fill_ratio


def motion_commands(pieces, transforms):
    """Convert transforms to rotation and pixel translation commands."""
    commands = []
    for index, (piece, transform) in enumerate(zip(pieces, transforms)):
        current_center = piece.mean(axis=0)
        target_center = apply_h(np.array([current_center]), transform)[0]
        delta = target_center - current_center
        commands.append({
            "piece": index,
            "rotation_deg": math.degrees(
                math.atan2(transform[1, 0], transform[0, 0])),
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "distance": float(np.linalg.norm(delta)),
            "matrix_3x3": transform.tolist(),
        })
    return commands

