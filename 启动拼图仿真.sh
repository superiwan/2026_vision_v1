#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "$SCRIPT_DIR/拼图视觉仿真/拼图视觉仿真" ]]; then
    exec "$SCRIPT_DIR/拼图视觉仿真/拼图视觉仿真"
fi

exec python3 "$SCRIPT_DIR/puzzle_gui.py"
