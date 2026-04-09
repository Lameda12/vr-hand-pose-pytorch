#!/usr/bin/env bash
# Run the Turin webcam hand pose demo.
# Usage: bash scripts/run_demo.sh [--camera 0] [--checkpoint path/to/model.pt]

set -euo pipefail

CAMERA=0
CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera) CAMERA="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CMD="python -m src.demo.webcam_demo --camera $CAMERA"
if [[ -n "$CHECKPOINT" ]]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

echo "Starting Turin demo: $CMD"
exec $CMD
