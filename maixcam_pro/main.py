"""MaixCAM Pro one-button puzzle vision entry."""

from maix import app, camera, display, image, time, touchscreen

try:
    from . import config
    from .step_view import ReleaseButton, StepView
    from .workflow import PuzzleWorkflow
except ImportError:  # MaixVision runs this directory as the project root.
    import config
    from step_view import ReleaseButton, StepView
    from workflow import PuzzleWorkflow


def _print_result(workflow, fps):
    print("SOLVED ALG %d | %.1f%% | FPS %.1f | pipeline %.2f ms | "
          "solve %.2f ms" % (
        workflow.algorithm,
        workflow.fill_ratio * 100.0,
        fps,
        workflow.elapsed_ms,
        workflow.solve_ms,
    ))
    for command in workflow.commands:
        print("P%d: R%+.2f deg, dx=%+.2f, dy=%+.2f, d=%.2f px" % (
            command["piece"], command["rotation_deg"], command["dx"],
            command["dy"], command["distance"]))


def run():
    screen = display.Display()
    screen_size = (screen.width(), screen.height())
    expected_size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    if screen_size != expected_size:
        print("WARNING: expected MaixCAM Pro display %s, got %s" % (
            expected_size, screen_size))

    cam = camera.Camera(
        config.CAMERA_WIDTH,
        config.CAMERA_HEIGHT,
        image.Format.FMT_BGR888,
        fps=config.CAMERA_FPS,
        buff_num=config.CAMERA_BUFFERS,
    )
    touch = touchscreen.TouchScreen()
    cam.skip_frames(config.CAMERA_SKIP_FRAMES)

    workflow = PuzzleWorkflow()
    view = StepView(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    button = ReleaseButton()
    fps = 0.0
    frame_index = 0
    terminal_drawn = False
    time.fps_set_buff_len(10)
    time.fps_start()

    while not app.need_exit():
        started_now = False
        if workflow.stage in workflow.TERMINAL_STAGES and terminal_drawn:
            if button.read(touch, view.action_rect):
                if workflow.stage == workflow.COMPLETE:
                    workflow.reset()
                else:
                    workflow.start()
                    started_now = True
                terminal_drawn = False
                cam.clear_buff()
                time.fps_start()
            else:
                time.sleep_ms(12)
                continue

        cycle_start = time.ticks_us()
        maix_frame = None
        frame = None
        if workflow.stage != workflow.SOLVE_PUZZLE:
            maix_frame = cam.read()
            frame = image.image2cv(
                maix_frame, ensure_bgr=False, copy=False)

        if workflow.stage == workflow.READY:
            if button.read(touch, view.action_rect):
                workflow.start()
                started_now = True

        previous_stage = workflow.stage
        if workflow.stage in workflow.ACTIVE_STAGES and not started_now:
            workflow.advance(frame)
        if previous_stage == workflow.SOLVE_PUZZLE \
                and workflow.stage == workflow.COMPLETE:
            _print_result(workflow, fps)

        output = view.render(frame, workflow, fps)
        show_start = time.ticks_us()
        if maix_frame is not None and output is frame:
            screen.show(maix_frame)
        else:
            screen.show(image.cv2image(output, bgr=True, copy=False))
        show_ms = (time.ticks_us() - show_start) / 1000.0
        total_ms = (time.ticks_us() - cycle_start) / 1000.0
        fps = time.fps()

        terminal_drawn = workflow.stage in workflow.TERMINAL_STAGES
        frame_index += 1
        if frame_index % config.PRINT_TIMING_EVERY_N_FRAMES == 0:
            print("FPS %.1f | frame %.2f ms | show %.2f ms | stage %s" % (
                fps, total_ms, show_ms, workflow.stage))


if __name__ == "__main__":
    run()
