#!/usr/bin/env python3
"""E题拼图装置：随机切割、随机摆放、视觉识别和矩形还原仿真。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np


CANVAS_W, CANVAS_H = 900, 1200
DIVIDER_Y = 575
CARD_W, CARD_H = 420.0, 280.0
PIECE_BGR = (190, 95, 30)
PAPER_BGR = (238, 238, 238)


def random_cut(rng: np.random.Generator) -> list[np.ndarray]:
    """Return four clockwise polygons cut from a CARD_W x CARD_H rectangle."""
    cx = rng.uniform(0.38, 0.62) * CARD_W
    cy = rng.uniform(0.35, 0.65) * CARD_H
    t = np.array([rng.uniform(.25, .75) * CARD_W, 0.0])
    r = np.array([CARD_W, rng.uniform(.25, .75) * CARD_H])
    b = np.array([rng.uniform(.25, .75) * CARD_W, CARD_H])
    l = np.array([0.0, rng.uniform(.25, .75) * CARD_H])
    c = np.array([cx, cy])
    tl, tr = np.array([0., 0.]), np.array([CARD_W, 0.])
    br, bl = np.array([CARD_W, CARD_H]), np.array([0., CARD_H])
    return [
        np.array([tl, t, c, l]),
        np.array([t, tr, r, c]),
        np.array([c, r, br, b]),
        np.array([l, c, b, bl]),
    ]


def rigid(angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def apply_h(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = np.c_[points, np.ones(len(points))] @ h.T
    return q[:, :2] / q[:, 2, None]


def place_randomly(polys: list[np.ndarray], rng: np.random.Generator):
    placed, poses = [], []
    occupancy = np.zeros((DIVIDER_Y - 25, CANVAS_W), np.uint8)
    for poly in polys:
        centroid = poly.mean(axis=0)
        local = poly - centroid
        accepted = False
        for _ in range(2000):
            angle = rng.uniform(-math.pi, math.pi)
            rot = rigid(angle, 0, 0)
            rotated = apply_h(local, rot)
            mn, mx = rotated.min(0), rotated.max(0)
            tx = rng.uniform(25 - mn[0], CANVAS_W - 25 - mx[0])
            ty = rng.uniform(25 - mn[1], DIVIDER_Y - 35 - mx[1])
            scene_poly = rotated + [tx, ty]
            mask = np.zeros_like(occupancy)
            cv2.fillPoly(mask, [np.round(scene_poly).astype(np.int32)], 255)
            dilated = cv2.dilate(mask, np.ones((19, 19), np.uint8))
            if not np.any((dilated > 0) & (occupancy > 0)):
                cv2.fillPoly(occupancy, [np.round(scene_poly).astype(np.int32)], 255)
                # local-to-scene; original-to-scene is local shift followed by this.
                pose = rigid(angle, tx, ty) @ rigid(0, -centroid[0], -centroid[1])
                placed.append(scene_poly)
                poses.append(pose)
                accepted = True
                break
        if not accepted:
            raise RuntimeError("无法在上半区无重叠摆放碎片，请更换随机种子")
    return placed, poses


def render_scene(placed: list[np.ndarray]) -> np.ndarray:
    image = np.full((CANVAS_H, CANVAS_W, 3), PAPER_BGR, np.uint8)
    cv2.line(image, (0, DIVIDER_Y), (CANVAS_W, DIVIDER_Y), (35, 35, 35), 4)
    target_x = int((CANVAS_W - CARD_W) / 2)
    target_y = 770
    cv2.rectangle(image, (target_x, target_y),
                  (int(target_x + CARD_W), int(target_y + CARD_H)), (100, 100, 100), 2)
    for poly in placed:
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(image, [pts], PIECE_BGR, lineType=cv2.LINE_8)
        cv2.polylines(image, [pts], True, (30, 30, 30), 2, cv2.LINE_AA)
    return image


def order_clockwise(vertices: np.ndarray) -> np.ndarray:
    c = vertices.mean(axis=0)
    a = np.arctan2(vertices[:, 1] - c[1], vertices[:, 0] - c[0])
    return vertices[np.argsort(a)]


def detect_pieces(image: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image[:DIVIDER_Y], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 80, 40), (140, 255, 245))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 3000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.012 * peri, True).reshape(-1, 2).astype(float)
        if 3 <= len(approx) <= 5:
            pieces.append(order_clockwise(approx))
    pieces.sort(key=lambda p: (p.mean(0)[1], p.mean(0)[0]))
    if len(pieces) != 4:
        raise RuntimeError(f"视觉检测到 {len(pieces)} 块碎片，预期为 4")
    return pieces


def edges(poly: np.ndarray):
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def align_edge(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """Rigid transform mapping src_a->dst_a and src_b->dst_b."""
    u, v = src_b - src_a, dst_b - dst_a
    angle = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
    r = rigid(angle, 0, 0)
    mapped = apply_h(np.array([src_a]), r)[0]
    r[:2, 2] = dst_a - mapped
    return r


def candidate_matchings(pieces: list[np.ndarray]):
    all_edges = {(i, e): edge for i, p in enumerate(pieces) for e, edge in enumerate(edges(p))}
    candidates = []
    for (i, ei), (j, ej) in itertools.combinations(all_edges, 2):
        if i == j:
            continue
        a, b = all_edges[(i, ei)]
        c, d = all_edges[(j, ej)]
        la, lb = np.linalg.norm(b - a), np.linalg.norm(d - c)
        rel = abs(la - lb) / max(la, lb)
        if rel < 0.045:
            candidates.append((rel, i, ei, j, ej))
    candidates.sort()
    # Keep ambiguity bounded while preserving all near-exact cut-edge matches.
    return candidates[:20]


def matching_sets(pieces: list[np.ndarray]):
    cand = candidate_matchings(pieces)
    for combo in itertools.combinations(cand, 4):
        used, degree = set(), [0] * 4
        ok = True
        graph = [set() for _ in range(4)]
        for _, i, ei, j, ej in combo:
            if (i, ei) in used or (j, ej) in used:
                ok = False
                break
            used |= {(i, ei), (j, ej)}
            degree[i] += 1
            degree[j] += 1
            graph[i].add(j)
            graph[j].add(i)
        if not ok or degree != [2, 2, 2, 2]:
            continue
        seen, stack = {0}, [0]
        while stack:
            for j in graph[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) == 4:
            yield combo


def assemble_from_matches(pieces, matches):
    adjacency = [[] for _ in pieces]
    for _, i, ei, j, ej in matches:
        adjacency[i].append((j, ei, ej))
        adjacency[j].append((i, ej, ei))
    transforms = [None] * 4
    transforms[0] = np.eye(3)
    stack = [0]
    closure_error = 0.0
    while stack:
        i = stack.pop()
        for j, ei, ej in adjacency[i]:
            ia, ib = edges(pieces[i])[ei]
            ja, jb = edges(pieces[j])[ej]
            wa, wb = apply_h(np.array([ia, ib]), transforms[i])
            proposed = align_edge(ja, jb, wb, wa)  # cutting edges meet in reverse order
            if transforms[j] is None:
                transforms[j] = proposed
                stack.append(j)
            else:
                closure_error += np.linalg.norm(apply_h(pieces[j], proposed) -
                                                apply_h(pieces[j], transforms[j]), axis=1).mean()
    assembled = [apply_h(p, h) for p, h in zip(pieces, transforms)]
    # Quality: low overlap, compact rectangular union, and graph closure.
    allp = np.vstack(assembled)
    mn, mx = allp.min(0), allp.max(0)
    scale = 1.0
    shift = -mn + 10
    w, h = np.ceil(mx - mn + 20).astype(int)
    masks = []
    for p in assembled:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(p * scale + shift).astype(np.int32)], 1)
        masks.append(m)
    total = sum(masks)
    overlap = float(np.count_nonzero(total > 1))
    union = (total > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    rect_area = cv2.minAreaRect(cnt)[1][0] * cv2.minAreaRect(cnt)[1][1]
    fill_error = max(0.0, rect_area - cv2.contourArea(cnt))
    match_error = sum(x[0] for x in matches) * 5000
    score = closure_error * 5 + overlap * 3 + fill_error + match_error
    return score, transforms, assembled


def solve(pieces: list[np.ndarray]):
    best = None
    for matches in matching_sets(pieces):
        result = assemble_from_matches(pieces, matches)
        if best is None or result[0] < best[0]:
            best = (*result, matches)
    if best is None:
        raise RuntimeError("未找到满足边长配对与四片环形邻接关系的拼接")
    _, transforms, assembled, matches = best

    # Normalize recovered rectangle to the requested lower-half target.
    allp = np.vstack(assembled).astype(np.float32)
    center, size, angle = cv2.minAreaRect(allp)
    if size[0] < size[1]:
        angle += 90.0
    normalize = rigid(math.radians(-angle), 0, 0)
    rotated = apply_h(allp, normalize)
    mn, mx = rotated.min(0), rotated.max(0)
    if (mx - mn)[0] < (mx - mn)[1]:
        normalize = rigid(math.radians(90) - math.radians(angle), 0, 0)
        rotated = apply_h(allp, normalize)
        mn, mx = rotated.min(0), rotated.max(0)
    target_origin = np.array([(CANVAS_W - CARD_W) / 2, 770.0])
    translate = rigid(0, *(target_origin - mn))
    final = [translate @ normalize @ h for h in transforms]
    return final, matches


def annotate_detection(image, pieces):
    out = image.copy()
    for i, p in enumerate(pieces):
        pts = np.round(p).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 220, 255), 3)
        for k, pt in enumerate(pts):
            cv2.circle(out, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(out, str(k), tuple(pt + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 160), 1, cv2.LINE_AA)
        c = np.round(p.mean(0)).astype(int)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_solution(image, pieces, transforms):
    out = image.copy()
    colors = [(70, 100, 230), (70, 190, 80), (220, 120, 60), (170, 70, 190)]
    for i, (p, h) in enumerate(zip(pieces, transforms)):
        q = np.round(apply_h(p, h)).astype(np.int32)
        cv2.fillPoly(out, [q], colors[i])
        cv2.polylines(out, [q], True, (20, 20, 20), 2, cv2.LINE_AA)
        c = np.round(q.mean(0)).astype(int)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_summary(scene, detected, solution):
    scale = 0.48
    panels = []
    for title, im in [("1 INPUT", scene), ("2 DETECT", detected), ("3 RESTORE", solution)]:
        x = cv2.resize(im, None, fx=scale, fy=scale)
        cv2.rectangle(x, (0, 0), (x.shape[1], 42), (255, 255, 255), -1)
        cv2.putText(x, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .75, (20, 20, 20), 2)
        panels.append(x)
    return np.hstack(panels)


def run_once(seed: int, output: Path, save=True):
    rng = np.random.default_rng(seed)
    source = random_cut(rng)
    placed, _ground_truth = place_randomly(source, rng)
    scene = render_scene(placed)
    detected_pieces = detect_pieces(scene)
    transforms, matches = solve(detected_pieces)

    # Pixel-domain reconstruction metrics.
    restored = [apply_h(p, h) for p, h in zip(detected_pieces, transforms)]
    allp = np.vstack(restored).astype(np.float32)
    rect = cv2.minAreaRect(allp)
    rw, rh = sorted(rect[1], reverse=True)
    dimension_error = max(abs(rw - CARD_W), abs(rh - CARD_H))

    if save:
        output.mkdir(parents=True, exist_ok=True)
        detected = annotate_detection(scene, detected_pieces)
        solution = render_solution(scene, detected_pieces, transforms)
        cv2.imwrite(str(output / "scene.png"), scene)
        cv2.imwrite(str(output / "detected.png"), detected)
        cv2.imwrite(str(output / "solution.png"), solution)
        cv2.imwrite(str(output / "summary.png"), make_summary(scene, detected, solution))
        records = []
        for i, (p, h) in enumerate(zip(detected_pieces, transforms)):
            angle = math.degrees(math.atan2(h[1, 0], h[0, 0]))
            records.append({
                "piece_id": i,
                "detected_center_px": p.mean(0).round(3).tolist(),
                "rotation_deg": round(angle, 6),
                "translation_px": h[:2, 2].round(6).tolist(),
                "matrix_2x3": h[:2].round(9).tolist(),
                "matrix_3x3": h.round(9).tolist(),
            })
        data = {
            "seed": seed,
            "coordinate_system": "image pixels: x right, y down",
            "target_rectangle_px": {"x": 240, "y": 770, "width": CARD_W, "height": CARD_H},
            "matched_cut_edges": [[int(i), int(ei), int(j), int(ej)]
                                  for _, i, ei, j, ej in matches],
            "dimension_error_px": round(float(dimension_error), 4),
            "pieces": records,
        }
        (output / "transforms.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return dimension_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--output", type=Path, default=Path("output/demo"), help="输出目录")
    parser.add_argument("--batch", type=int, default=0, help="批量验证次数（不保存图片）")
    args = parser.parse_args()
    if args.batch:
        errors, failures = [], []
        for seed in range(args.seed, args.seed + args.batch):
            try:
                errors.append(run_once(seed, args.output, save=False))
            except Exception as exc:
                failures.append((seed, str(exc)))
        print(f"批量测试: {args.batch} 次，成功 {len(errors)}，失败 {len(failures)}")
        if errors:
            print(f"矩形尺寸最大误差: {max(errors):.3f} px，平均: {np.mean(errors):.3f} px")
        if failures:
            print("失败样例:", failures[:10])
            raise SystemExit(1)
    else:
        err = run_once(args.seed, args.output, save=True)
        print(f"完成。输出目录: {args.output.resolve()}")
        print(f"还原矩形尺寸误差: {err:.3f} px")
        print(f"位姿矩阵: {(args.output / 'transforms.json').resolve()}")


if __name__ == "__main__":
    main()
