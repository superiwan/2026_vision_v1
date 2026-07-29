"""Detect 1-4 non-black polygon pieces inside the black A4 sheet."""

import cv2
import numpy as np

try:
    from . import config
except ImportError:  # Run directly in MaixVision with this folder as project root.
    import config


def order_clockwise(vertices):
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1],
                        vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)]


def _odd_kernel(size):
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return np.ones((size, size), np.uint8)


def find_black_paper(frame):
    """Return the outer contour of the largest black A4-like region."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, config.PAPER_GRAY_MAX)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, _odd_kernel(config.PAPER_CLOSE_KERNEL))
    contours, _ = cv2.findContours(
        dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    candidates = [c for c in contours
                  if cv2.contourArea(c) >= frame_area * config.PAPER_MIN_AREA_RATIO]
    if not candidates:
        raise RuntimeError("未找到黑色 A4 纸，请检查取景和 PAPER_GRAY_MAX")

    contour = max(candidates, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(
        contour, config.PAPER_APPROX_EPSILON * perimeter, True)
    if len(polygon) != 4:
        polygon = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1, 1, 2)
    return np.round(polygon).astype(np.int32)


def _approx_piece(contour):
    perimeter = cv2.arcLength(contour, True)
    for epsilon in config.PIECE_APPROX_EPSILONS:
        polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if 3 <= len(polygon) <= 5:
            points = polygon.reshape(-1, 2).astype(np.float64)
            return order_clockwise(points)
    return None


def detect_pieces(frame):
    """Return (pieces, paper_contour, binary_mask).

    Every piece is a clockwise float64 array shaped (3..5, 2), in camera pixels.
    """
    if frame is None or frame.ndim != 3:
        raise ValueError("输入必须是 BGR 彩色图像")

    paper = find_black_paper(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    paper_mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(paper_mask, [paper], -1, 255, thickness=-1)
    if config.PAPER_MASK_INSET > 0:
        paper_mask = cv2.erode(
            paper_mask, _odd_kernel(config.PAPER_MASK_INSET))

    background_pixels = gray[paper_mask > 0]
    if background_pixels.size == 0:
        raise RuntimeError("黑色 A4 区域为空")
    background_gray = float(np.percentile(background_pixels, 30))
    threshold = int(max(config.PIECE_GRAY_MIN,
                        background_gray + config.PIECE_BACKGROUND_DELTA))
    threshold = min(threshold, 250)

    piece_mask = cv2.inRange(gray, threshold, 255)
    piece_mask = cv2.bitwise_and(piece_mask, paper_mask)
    kernel = _odd_kernel(config.MORPH_KERNEL)
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, kernel)
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_area = max(1.0, cv2.contourArea(paper))
    pieces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / paper_area
        if not (config.PIECE_MIN_AREA_RATIO <= area_ratio
                <= config.PIECE_MAX_AREA_RATIO):
            continue

        _, _, width, height = cv2.boundingRect(contour)
        short_side = max(1, min(width, height))
        aspect = max(width, height) / short_side
        if aspect > config.THIN_OBJECT_ASPECT_RATIO:
            continue

        polygon = _approx_piece(contour)
        if polygon is not None:
            pieces.append(polygon)

    pieces.sort(key=lambda p: (p.mean(axis=0)[1], p.mean(axis=0)[0]))
    if not 1 <= len(pieces) <= 4:
        raise RuntimeError(
            "检测到 %d 块有效碎片，要求为 1～4 块" % len(pieces))
    return pieces, paper, piece_mask

