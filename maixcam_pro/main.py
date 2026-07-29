"""MaixCAM Pro entry: camera -> pieces -> geometry solver -> screen overlay."""

import cv2
import numpy as np
from maix import app, camera, display, image

import config
from piece_detector import detect_pieces
from puzzle_solver import apply_h, motion_commands, solve


def draw_polygon(frame, polygon, color, thickness=2):
    points = polygon.round().astype("int32")
    cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)


def draw_result(frame, pieces, paper, transforms, commands, fill_ratio):
    cv2.drawContours(frame, [paper], -1, config.PAPER_COLOR, 2, cv2.LINE_AA)
    target_points = []
    for index, (piece, transform) in enumerate(zip(pieces, transforms)):
        color = config.PIECE_COLORS[index % len(config.PIECE_COLORS)]
        draw_polygon(frame, piece, color, 2)
        center = piece.mean(axis=0).round().astype(int)
        cv2.putText(frame, "P%d" % index, tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        target = apply_h(piece, transform)
        target_points.append(target)
        draw_polygon(frame, target, config.TARGET_COLOR, 2)

    all_target = np.vstack(target_points).astype("float32")
    rectangle = cv2.boxPoints(cv2.minAreaRect(all_target)).round().astype("int32")
    cv2.polylines(frame, [rectangle], True, config.TARGET_COLOR, 3, cv2.LINE_AA)

    y = 20
    for command in commands:
        text = "P%d R%+.1f deg D(%+.0f,%+.0f) L%.0f px" % (
            command["piece"], command["rotation_deg"], command["dx"],
            command["dy"], command["distance"])
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
        y += 18
    cv2.putText(frame, "rect %.1f%%" % (fill_ratio * 100.0),
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                config.TARGET_COLOR, 1, cv2.LINE_AA)


def run():
    cam = camera.Camera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                        image.Format.FMT_BGR888)
    screen = display.Display()
    cam.skip_frames(config.CAMERA_SKIP_FRAMES)

    cached = None
    last_message = None
    frame_index = 0
    while not app.need_exit():
        maix_frame = cam.read()
        # FMT_BGR888 allows OpenCV to borrow the camera buffer without a copy.
        frame = image.image2cv(maix_frame, ensure_bgr=False, copy=False)

        if frame_index % config.SOLVE_EVERY_N_FRAMES == 0:
            try:
                pieces, paper, _ = detect_pieces(frame)
                transforms, matches, fill_ratio = solve(pieces, paper)
                commands = motion_commands(pieces, transforms)
                cached = (pieces, paper, transforms, commands, fill_ratio)
                message = "; ".join(
                    "P%d: R%+.1f, dx=%+.1f, dy=%+.1f, d=%.1fpx" % (
                        c["piece"], c["rotation_deg"], c["dx"],
                        c["dy"], c["distance"])
                    for c in commands)
                if message != last_message:
                    print(message)
                    last_message = message
            except Exception as error:
                cached = None
                message = "ERROR: %s" % error
                if message != last_message:
                    print(message)
                last_message = message

        if cached is not None:
            draw_result(frame, *cached)
        elif last_message:
            cv2.putText(frame, "ERROR: see MaixVision terminal", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        config.ERROR_COLOR, 2, cv2.LINE_AA)

        screen.show(maix_frame)
        frame_index += 1


if __name__ == "__main__":
    run()
