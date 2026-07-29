"""One-button, frame-stepped puzzle workflow independent of Maix hardware."""

import time

try:
    from . import config
    from .piece_detector import A4PieceDetector
    from .puzzle_solver import motion_commands, solve as solve_graph
    from .puzzle_solver_merge import solve as solve_merge
except ImportError:  # MaixVision runs this directory as the project root.
    import config
    from piece_detector import A4PieceDetector
    from puzzle_solver import motion_commands, solve as solve_graph
    from puzzle_solver_merge import solve as solve_merge


class PuzzleWorkflow:
    """Advance at most one expensive operation on every camera frame."""

    READY = "ready"
    LOCATE_A4 = "locate_a4"
    DETECT_PIECES = "detect_pieces"
    SOLVE_PUZZLE = "solve_puzzle"
    COMPLETE = "complete"
    ERROR = "error"

    ACTIVE_STAGES = (LOCATE_A4, DETECT_PIECES, SOLVE_PUZZLE)
    TERMINAL_STAGES = (COMPLETE, ERROR)

    def __init__(self, detector=None, algorithm=None):
        self.detector = detector or A4PieceDetector()
        self.algorithm = (config.SOLVER_ALGORITHM if algorithm is None
                          else int(algorithm))
        if self.algorithm not in (1, 2):
            raise ValueError("SOLVER_ALGORITHM 必须为 1 或 2")
        self.stage = self.READY
        self.result = self.detector.result()
        self.transforms = None
        self.matches = None
        self.commands = None
        self.fill_ratio = None
        self.error = None
        self.elapsed_ms = 0.0
        self.solve_ms = 0.0
        self._started_at = None

    def _solve(self):
        solver = solve_graph if self.algorithm == 1 else solve_merge
        return solver(self.result["pieces"], self.result["paper_rect"])

    def start(self):
        """Start a new automatic run; no further touch is required."""
        self.detector.clear_a4()
        self.stage = self.LOCATE_A4
        self.result = self.detector.result()
        self.transforms = None
        self.matches = None
        self.commands = None
        self.fill_ratio = None
        self.error = None
        self.elapsed_ms = 0.0
        self.solve_ms = 0.0
        self._started_at = time.perf_counter()

    def reset(self):
        """Return to the live alignment screen."""
        self.detector.clear_a4()
        self.stage = self.READY
        self.result = self.detector.result()
        self.transforms = None
        self.matches = None
        self.commands = None
        self.fill_ratio = None
        self.error = None
        self.elapsed_ms = 0.0
        self.solve_ms = 0.0
        self._started_at = None

    def _update_elapsed(self):
        if self._started_at is not None:
            self.elapsed_ms = (time.perf_counter() - self._started_at) * 1000.0

    def _fail(self, message):
        self.error = str(message)
        self.stage = self.ERROR
        self._update_elapsed()
        print("PUZZLE ERROR:", self.error)

    def advance(self, frame):
        """Run exactly one pending stage against an unmodified BGR frame."""
        if self.stage == self.LOCATE_A4:
            self.result = self.detector.find_a4(frame)
            if self.result["paper_locked"]:
                self.stage = self.DETECT_PIECES
            elif self.result["paper_search_exhausted"]:
                self._fail("A4 NOT FOUND")

        elif self.stage == self.DETECT_PIECES:
            self.result = self.detector.analyze_cached_a4(frame)
            if self.result["piece_error"]:
                self._fail(self.result["piece_error"].upper())
            else:
                self.stage = self.SOLVE_PUZZLE

        elif self.stage == self.SOLVE_PUZZLE:
            started = time.perf_counter()
            try:
                self.transforms, self.matches, self.fill_ratio = self._solve()
                self.commands = motion_commands(
                    self.result["pieces"], self.transforms)
                self.solve_ms = (time.perf_counter() - started) * 1000.0
                self.stage = self.COMPLETE
                self._update_elapsed()
            except Exception as error:
                self.solve_ms = (time.perf_counter() - started) * 1000.0
                self._fail(error)
        return self.stage

    @property
    def action_label(self):
        if self.stage == self.READY:
            return "START"
        if self.stage == self.COMPLETE:
            return "NEW RUN"
        if self.stage == self.ERROR:
            return "RETRY"
        return None

    @property
    def progress_step(self):
        return {
            self.READY: 0,
            self.LOCATE_A4: 0,
            self.DETECT_PIECES: 1,
            self.SOLVE_PUZZLE: 2,
            self.COMPLETE: 3,
            self.ERROR: 0,
        }[self.stage]
