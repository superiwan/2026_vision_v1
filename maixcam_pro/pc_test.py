"""Run the exact one-button device workflow on one PC image."""

import argparse
import json
from pathlib import Path

import cv2

try:
    from . import config
    from .step_view import StepView
    from .workflow import PuzzleWorkflow
except ImportError:  # Direct script execution from the repository root.
    import config
    from step_view import StepView
    from workflow import PuzzleWorkflow


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
    parser.add_argument("--output", type=Path,
                        default=Path("output/maixcam_pro_pc"))
    parser.add_argument("--algorithm", type=int, choices=(1, 2),
                        default=config.SOLVER_ALGORITHM)
    args = parser.parse_args()

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("无法读取图片: %s" % args.image)

    args.output.mkdir(parents=True, exist_ok=True)
    renderer = StepView(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    workflow = PuzzleWorkflow(algorithm=args.algorithm)
    workflow.start()
    for _attempt in range(config.PAPER_SEARCH_MAX_FRAMES):
        workflow.advance(frame)
        if workflow.stage != workflow.LOCATE_A4:
            break
    if workflow.stage == workflow.ERROR:
        mask_path = unused_path(args.output / "a4_binary.png")
        error_path = unused_path(args.output / "a4_failed.png")
        if workflow.result.get("paper_binary") is not None:
            cv2.imwrite(str(mask_path), workflow.result["paper_binary"])
        cv2.imwrite(str(error_path), frame)
        print("A4 二值图: %s" % mask_path.resolve())
        print("失败原图: %s" % error_path.resolve())
        raise RuntimeError(workflow.error)
    workflow.advance(frame)
    if workflow.stage == workflow.ERROR:
        raise RuntimeError(workflow.error)

    pieces = workflow.result["pieces"]
    timings = dict(workflow.result["timings"])
    timings["solve"] = 0.0

    detected_path = unused_path(args.output / "detected.png")
    mask_path = unused_path(args.output / "piece_binary.png")
    detected = renderer.render(frame, workflow, 0.0).copy()
    piece_binary = workflow.result["piece_binary"].copy()
    cv2.imwrite(str(detected_path), detected)
    cv2.imwrite(str(mask_path), piece_binary)

    print("检测到 %d 块碎片: %s" % (
        len(pieces), [len(piece) for piece in pieces]))
    workflow.advance(frame)
    if workflow.stage == workflow.ERROR:
        print("检测成功，但几何拼接失败: %s" % workflow.error)
        print("检测图: %s" % detected_path.resolve())
        raise SystemExit(2)

    timings["solve"] = workflow.solve_ms
    timings["algorithm_total"] = sum(timings[key] for key in (
        "gray", "paper_bin", "paper", "warp", "pieces", "solve"))
    timings["pc_workflow_total"] = workflow.elapsed_ms
    solved_path = unused_path(args.output / "solved.png")
    json_path = unused_path(args.output / "motions.json")
    solved = renderer.render(frame, workflow, 0.0).copy()
    cv2.imwrite(str(solved_path), solved)
    json_path.write_text(json.dumps({
        "solver_algorithm": workflow.algorithm,
        "coordinate_system": "rectified A4 pixels",
        "rectangle_fill_ratio": workflow.fill_ratio,
        "matches": [list(match) for match in workflow.matches],
        "timings_ms": timings,
        "commands": workflow.commands,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for command in workflow.commands:
        print("P%(piece)d: 旋转 %(rotation_deg)+.2f deg, "
              "移动 (%(dx)+.2f, %(dy)+.2f) px, 距离 %(distance).2f px" % command)
    print("矩形填充率: %.2f%%" % (workflow.fill_ratio * 100.0))
    print("阶段耗时(ms):", {key: round(value, 3)
                           for key, value in timings.items()})
    print("结果图: %s" % solved_path.resolve())
    print("位姿: %s" % json_path.resolve())


if __name__ == "__main__":
    main()
