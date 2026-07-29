"""Batch-simulate camera scenes and separate geometry from vision failures."""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
    from .puzzle_solver_merge import solve as solve_merge
    from .step_view import StepView
    from .workflow import PuzzleWorkflow
except ImportError:  # Direct script execution from the repository root.
    import config
    from puzzle_solver_merge import solve as solve_merge
    from step_view import StepView
    from workflow import PuzzleWorkflow

try:
    from puzzle_sim import place_randomly, random_cut
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from puzzle_sim import place_randomly, random_cut


PAPER_WIDTH = 900
PAPER_HEIGHT = 1273
DIVIDER_Y = 575
RAW_WIDTH = 640
RAW_HEIGHT = 480

TIERS = {
    "clean": {
        "paper": (3, 18),
        "piece": (238, 255),
        "gradient": 5.0,
        "noise": 0.0,
        "blur_choices": (0,),
        "quad_jitter": 4.0,
        "glare_probability": 0.0,
    },
    "normal": {
        "paper": (5, 38),
        "piece": (205, 255),
        "gradient": 20.0,
        "noise": 2.5,
        "blur_choices": (0, 3, 3),
        "quad_jitter": 13.0,
        "glare_probability": 0.0,
    },
    "harsh": {
        "paper": (12, 72),
        "piece": (165, 235),
        "gradient": 45.0,
        "noise": 6.0,
        "blur_choices": (3, 3, 5),
        "quad_jitter": 24.0,
        "glare_probability": 0.35,
    },
}


def _unused_directory(path):
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name("%s_%d" % (path.name, index))
        if not candidate.exists():
            return candidate
    raise RuntimeError("无法生成不冲突的输出目录")


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


def _two_cut_pieces(rng):
    rectangle = np.float64(((0, 0), (400, 0), (400, 240), (0, 240)))
    first_top = rng.uniform(115, 285)
    first_bottom = rng.uniform(115, 285)
    second_left = rng.uniform(65, 175)
    second_right = rng.uniform(65, 175)
    cuts = (
        (np.float64((first_top, 0)), np.float64((first_bottom, 240))),
        (np.float64((0, second_left)), np.float64((400, second_right))),
    )
    fragments = [rectangle]
    for start, end in cuts:
        divided = []
        for fragment in fragments:
            divided.extend((
                _clip_half_plane(fragment, start, end, 1.0),
                _clip_half_plane(fragment, start, end, -1.0),
            ))
        fragments = [fragment for fragment in divided if len(fragment) >= 3]
    if len(fragments) != 4:
        raise RuntimeError("两刀场景没有生成 4 个碎片")
    return fragments


def _source_pieces(rng, piece_count):
    if piece_count == 4 and rng.random() < 0.5:
        return _two_cut_pieces(rng), "two_full_cuts"
    return random_cut(rng, piece_count), "radial_or_single_cut"


def _random_quad(rng, jitter):
    # Keep the unjittered projection close to a true portrait A4 ratio. The
    # older near-square base unintentionally simulated an unrealistically
    # oblique camera and conflicted with strict A4 geometry filtering.
    base = np.float32(((170, 25), (470, 32), (490, 452), (150, 445)))
    offsets = rng.normal(0.0, jitter, size=(4, 2)).astype(np.float32)
    quad = base + offsets
    quad[:, 0] = np.clip(quad[:, 0], 18, RAW_WIDTH - 18)
    quad[:, 1] = np.clip(quad[:, 1], 12, RAW_HEIGHT - 12)
    return quad


def _apply_lighting(frame, rng, tier):
    height, width = frame.shape[:2]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    direction = rng.uniform(-math.pi, math.pi)
    gradient = (math.cos(direction) * x[None, :]
                + math.sin(direction) * y[:, None])
    gradient *= tier["gradient"]
    adjusted = frame.astype(np.float32) + gradient[:, :, None]

    if rng.random() < tier["glare_probability"]:
        mask = np.zeros((height, width), np.uint8)
        center = (int(rng.uniform(170, 470)), int(rng.uniform(100, 390)))
        axes = (int(rng.uniform(35, 100)), int(rng.uniform(20, 65)))
        cv2.ellipse(mask, center, axes, rng.uniform(0, 180),
                    0, 360, 255, -1, cv2.LINE_AA)
        glare = rng.uniform(45, 95)
        adjusted += (mask.astype(np.float32) / 255.0)[:, :, None] * glare

    blur = int(rng.choice(tier["blur_choices"]))
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    if blur:
        adjusted = cv2.GaussianBlur(adjusted, (blur, blur), 0)
    if tier["noise"] > 0:
        noise = rng.normal(0.0, tier["noise"], adjusted.shape).astype(np.float32)
        adjusted = np.clip(adjusted.astype(np.float32) + noise,
                           0, 255).astype(np.uint8)
    return adjusted


def make_scene(rng, piece_count, tier_name):
    tier = TIERS[tier_name]
    source, cut_type = _source_pieces(rng, piece_count)
    placed = place_randomly(source, rng)
    paper_level = int(rng.integers(tier["paper"][0], tier["paper"][1] + 1))
    piece_level = int(rng.integers(tier["piece"][0], tier["piece"][1] + 1))
    paper = np.full((PAPER_HEIGHT, PAPER_WIDTH, 3), paper_level, np.uint8)
    cv2.line(paper, (0, DIVIDER_Y), (PAPER_WIDTH - 1, DIVIDER_Y),
             (95, 95, 95), 3)
    for polygon in placed:
        points = np.round(polygon).astype(np.int32)
        cv2.fillPoly(paper, [points], (piece_level,) * 3, cv2.LINE_AA)
        edge_level = max(config.PIECE_GRAY_MIN + 4, piece_level - 22)
        cv2.polylines(paper, [points], True, (edge_level,) * 3,
                      2, cv2.LINE_AA)

    background = int(rng.integers(165, 236))
    raw = np.full((RAW_HEIGHT, RAW_WIDTH, 3), background, np.uint8)
    source_quad = np.float32(((0, 0), (PAPER_WIDTH - 1, 0),
                              (PAPER_WIDTH - 1, PAPER_HEIGHT - 1),
                              (0, PAPER_HEIGHT - 1)))
    target_quad = _random_quad(rng, tier["quad_jitter"])
    homography = cv2.getPerspectiveTransform(source_quad, target_quad)
    projected = cv2.warpPerspective(
        paper, homography, (RAW_WIDTH, RAW_HEIGHT), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full((PAPER_HEIGHT, PAPER_WIDTH), 255, np.uint8),
        homography, (RAW_WIDTH, RAW_HEIGHT), flags=cv2.INTER_NEAREST)
    raw[mask > 0] = projected[mask > 0]
    raw = _apply_lighting(raw, rng, tier)
    return raw, placed, cut_type


def _geometry_check(placed):
    paper = np.int32((((0, 0),), ((PAPER_WIDTH - 1, 0),),
                      ((PAPER_WIDTH - 1, PAPER_HEIGHT - 1),),
                      ((0, PAPER_HEIGHT - 1),)))
    started = time.perf_counter()
    try:
        _transforms, _matches, fill_ratio = solve_merge(placed, paper)
        return True, fill_ratio, (time.perf_counter() - started) * 1000.0, ""
    except Exception as error:
        return False, None, (time.perf_counter() - started) * 1000.0, str(error)


def _vision_check(frame):
    workflow = PuzzleWorkflow(algorithm=2)
    workflow.start()
    for _attempt in range(config.PAPER_SEARCH_MAX_FRAMES):
        workflow.advance(frame)
        if workflow.stage != workflow.LOCATE_A4:
            break
    if workflow.stage == workflow.ERROR:
        return workflow, "a4", workflow.error
    workflow.advance(frame)
    if workflow.stage == workflow.ERROR:
        return workflow, "pieces", workflow.error
    workflow.advance(None)
    if workflow.stage == workflow.ERROR:
        return workflow, "solver", workflow.error
    return workflow, "ok", ""


def _save_sample(output, tier, piece_count, case_index, frame, workflow):
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = "%s_%dp_case%03d" % (tier, piece_count, case_index)
    raw_path = sample_dir / (stem + "_raw.png")
    result_path = sample_dir / (stem + "_result.png")
    cv2.imwrite(str(raw_path), frame)
    result = StepView(RAW_WIDTH, RAW_HEIGHT).render(
        frame.copy(), workflow, 0.0).copy()
    cv2.imwrite(str(result_path), result)
    return raw_path, result_path


def _contact_sheet(sample_pairs, output):
    cell_width, cell_height = 640, 240
    sheet = np.full((cell_height * len(TIERS), cell_width * 4, 3),
                    22, np.uint8)
    tier_rows = {tier: index for index, tier in enumerate(TIERS)}
    for (tier, piece_count), (raw_path, result_path) in sample_pairs.items():
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        result = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
        if raw is None or result is None:
            continue
        raw = cv2.resize(raw, (320, 240), interpolation=cv2.INTER_AREA)
        result = cv2.resize(result, (320, 240), interpolation=cv2.INTER_AREA)
        cell = np.hstack((raw, result))
        cv2.putText(cell, "%s / %d pieces" % (tier.upper(), piece_count),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
        row = tier_rows[tier]
        column = piece_count - 1
        sheet[row * cell_height:(row + 1) * cell_height,
              column * cell_width:(column + 1) * cell_width] = cell
    path = output / "contact_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def _summarize(records):
    summary = {}
    for tier in TIERS:
        tier_records = [record for record in records if record["tier"] == tier]
        fills = [record["vision_fill_ratio"] for record in tier_records
                 if record["strict_pass"]
                 and record["vision_fill_ratio"] is not None]
        summary[tier] = {
            "cases": len(tier_records),
            "geometry_pass": sum(record["geometry_pass"] for record in tier_records),
            "a4_pass": sum(record["failure_stage"] != "a4"
                           for record in tier_records),
            "piece_detection_pass": sum(record["detected_count"]
                                        == record["expected_count"]
                                        for record in tier_records),
            "pipeline_pass": sum(record["pipeline_pass"] for record in tier_records),
            "strict_pass": sum(record["strict_pass"] for record in tier_records),
            "mean_fill_ratio": (float(np.mean(fills)) if fills else None),
            "mean_solver_ms": float(np.mean([
                record["vision_solver_ms"] for record in tier_records
                if record["strict_pass"]
            ])) if any(record["strict_pass"] for record in tier_records) else None,
            "diagnostic_categories": dict(Counter(
                record["diagnostic_category"] for record in tier_records
                if record["diagnostic_category"] != "ok"
            )),
            "failure_reasons": dict(Counter(
                record["failure_reason"] for record in tier_records
                if record["failure_reason"]
            )),
        }
    return summary


def _write_report(output, records, summary, contact_sheet, args):
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        "seed": args.seed,
        "cases_per_piece_per_tier": args.cases_per_piece,
        "solver_algorithm": 2,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output / "cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# 算法 2 PC 批量仿真报告",
        "",
        "- 随机种子：`%d`" % args.seed,
        "- 每档、每种碎片数量：`%d` 个场景" % args.cases_per_piece,
        "- 总场景数：`%d`" % len(records),
        "- 每个场景先用真值轮廓验证几何算法，再把图像送入与真机相同的检测和求解链。",
        "",
        "| 环境 | 场景 | 真值几何通过 | A4通过 | 碎片数正确 | 求解器完成 | 严格全链通过 | 平均填充率 | 平均求解ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier, values in summary.items():
        fill = "-" if values["mean_fill_ratio"] is None else "%.2f%%" % (
            values["mean_fill_ratio"] * 100.0)
        solver_ms = "-" if values["mean_solver_ms"] is None else "%.2f" % (
            values["mean_solver_ms"])
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %s | %s |" % (
            tier,
            values["cases"],
            values["geometry_pass"],
            values["a4_pass"],
            values["piece_detection_pass"],
            values["pipeline_pass"],
            values["strict_pass"],
            fill,
            solver_ms,
        ))
    lines.extend((
        "",
        "## 失败归因",
        "",
        "`严格全链通过` 要求检测出的碎片数量正确，并且求解器完成；这样不会把漏检后碰巧形成矩形的结果计为成功。",
        "",
        "| 环境 | A4检测 | 碎片数量 | 检测轮廓几何 | 真值几何算法 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ))
    for tier, values in summary.items():
        categories = values["diagnostic_categories"]
        lines.append("| %s | %d | %d | %d | %d |" % (
            tier,
            categories.get("a4_detection", 0),
            categories.get("piece_count", 0),
            categories.get("detected_polygon_geometry", 0),
            categories.get("geometry_algorithm", 0),
        ))
    lines.extend((
        "",
        "## 判读方法",
        "",
        "- 真值几何通过、但碎片数错误：优先检查实物阈值、反光、阴影、接触/遮挡和轮廓近似。",
        "- 碎片数正确、但 solver 失败：检测出的顶点或边长误差已经超过算法 2 容差。",
        "- clean/normal 全链稳定、实物仍失败：问题更可能在相机画面和实物条件，而不是矩形合并逻辑。",
        "- 本报告不能代替真实 MaixCAM 图像；最有价值的下一步是保存一张真机原始帧再运行 `pc_test.py`。",
        "",
        "## 文件",
        "",
        "- 联系表：`%s`" % contact_sheet.name,
        "- 全部记录：`%s`" % csv_path.name,
        "- 汇总数据：`%s`" % summary_path.name,
    ))
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="算法2 PC 相机全链批量仿真")
    parser.add_argument("--cases-per-piece", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path,
                        default=Path("output/algorithm2_pc_validation"))
    args = parser.parse_args()
    if args.cases_per_piece < 1:
        raise ValueError("--cases-per-piece 必须大于 0")

    output = _unused_directory(args.output)
    output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    records = []
    sample_pairs = {}
    for tier in TIERS:
        for piece_count in range(1, 5):
            for case_index in range(args.cases_per_piece):
                frame, placed, cut_type = make_scene(rng, piece_count, tier)
                geometry_ok, geometry_fill, geometry_ms, geometry_error = (
                    _geometry_check(placed))
                workflow, failure_stage, failure_reason = _vision_check(frame)
                detected_count = len(workflow.result.get("pieces") or ())
                pipeline_ok = workflow.stage == workflow.COMPLETE
                count_ok = detected_count == piece_count
                strict_ok = geometry_ok and count_ok and pipeline_ok
                if not geometry_ok:
                    diagnostic_category = "geometry_algorithm"
                elif failure_stage == "a4":
                    diagnostic_category = "a4_detection"
                elif not count_ok:
                    diagnostic_category = "piece_count"
                elif not pipeline_ok:
                    diagnostic_category = "detected_polygon_geometry"
                else:
                    diagnostic_category = "ok"
                record = {
                    "tier": tier,
                    "piece_count": piece_count,
                    "case_index": case_index,
                    "cut_type": cut_type,
                    "expected_count": piece_count,
                    "detected_count": detected_count,
                    "geometry_pass": geometry_ok,
                    "geometry_fill_ratio": geometry_fill,
                    "geometry_solver_ms": geometry_ms,
                    "pipeline_pass": pipeline_ok,
                    "strict_pass": strict_ok,
                    "vision_fill_ratio": workflow.fill_ratio,
                    "vision_solver_ms": workflow.solve_ms,
                    "failure_stage": failure_stage,
                    "diagnostic_category": diagnostic_category,
                    "failure_reason": failure_reason or geometry_error,
                }
                records.append(record)
                key = (tier, piece_count)
                if strict_ok and key not in sample_pairs:
                    sample_pairs[key] = _save_sample(
                        output, tier, piece_count, case_index, frame, workflow)
                if not strict_ok:
                    failure_dir = output / "failures"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    failure_path = failure_dir / (
                        "%s_%dp_case%03d_%s.png" % (
                            tier, piece_count, case_index, failure_stage))
                    cv2.imwrite(str(failure_path), frame)

    contact_sheet = _contact_sheet(sample_pairs, output)
    summary = _summarize(records)
    report_path, csv_path, summary_path = _write_report(
        output, records, summary, contact_sheet, args)
    print("OUTPUT:", output.resolve())
    print("REPORT:", report_path.resolve())
    print("CONTACT_SHEET:", contact_sheet.resolve())
    print("CASES_CSV:", csv_path.resolve())
    print("SUMMARY_JSON:", summary_path.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
