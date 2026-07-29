"""Run the exact detector/solver on a still image before device deployment."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from piece_detector import detect_pieces
from puzzle_solver import apply_h, motion_commands, solve


def unused_path(path):
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name("%s_%d%s" % (path.stem, index, path.suffix))
        if not candidate.exists():
            return candidate
    raise RuntimeError("无法为输出文件生成不冲突的名称")


def main():
    parser = argparse.ArgumentParser(description="MaixCAM Pro 拼图 PC 图片测试")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/maixcam_pro_pc"))
    args = parser.parse_args()

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("无法读取图片: %s" % args.image)
    pieces, paper, mask = detect_pieces(frame)

    detected = frame.copy()
    cv2.drawContours(detected, [paper], -1, (255, 180, 0), 3)
    for index, piece in enumerate(pieces):
        points = piece.round().astype(np.int32)
        cv2.polylines(detected, [points], True, (0, 220, 255), 3)
        center = piece.mean(axis=0).round().astype(int)
        cv2.putText(detected, "P%d" % index, tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    args.output.mkdir(parents=True, exist_ok=True)
    detected_path = unused_path(args.output / "detected.png")
    mask_path = unused_path(args.output / "mask.png")
    cv2.imwrite(str(detected_path), detected)
    cv2.imwrite(str(mask_path), mask)

    print("检测到 %d 块碎片: %s" % (
        len(pieces), [len(piece) for piece in pieces]))
    try:
        transforms, matches, fill_ratio = solve(pieces, paper)
    except RuntimeError as error:
        print("检测成功，但几何拼接失败: %s" % error)
        print("检测图: %s" % detected_path.resolve())
        raise SystemExit(2)

    commands = motion_commands(pieces, transforms)
    solved = detected.copy()
    targets = []
    for piece, transform in zip(pieces, transforms):
        target = apply_h(piece, transform)
        targets.append(target)
        cv2.polylines(solved, [target.round().astype(np.int32)], True,
                      (0, 255, 0), 3)
    rectangle = cv2.boxPoints(cv2.minAreaRect(
        np.vstack(targets).astype(np.float32))).round().astype(np.int32)
    cv2.polylines(solved, [rectangle], True, (0, 255, 0), 4)

    solved_path = unused_path(args.output / "solved.png")
    json_path = unused_path(args.output / "motions.json")
    cv2.imwrite(str(solved_path), solved)
    json_path.write_text(json.dumps({
        "rectangle_fill_ratio": fill_ratio,
        "matches": [list(match) for match in matches],
        "commands": commands,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for command in commands:
        print("P%(piece)d: 旋转 %(rotation_deg)+.2f deg, "
              "移动 (%(dx)+.2f, %(dy)+.2f) px, 距离 %(distance).2f px" % command)
    print("矩形填充率: %.2f%%" % (fill_ratio * 100.0))
    print("结果图: %s" % solved_path.resolve())
    print("位姿: %s" % json_path.resolve())


if __name__ == "__main__":
    main()

