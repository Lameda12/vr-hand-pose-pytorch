#!/usr/bin/env bash
# Run latency benchmarks on CPU and GPU (if available).
# Outputs results to stdout and saves summary to benchmarks/results.txt

set -euo pipefail

OUTFILE="benchmarks/results_$(date +%Y%m%d_%H%M%S).txt"

echo "=== Turin Latency Benchmark ===" | tee "$OUTFILE"
echo "Date: $(date)" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

echo "--- CPU @720p ---" | tee -a "$OUTFILE"
python benchmarks/latency_benchmark.py --device cpu --resolution 720p --runs 50 | tee -a "$OUTFILE"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "" | tee -a "$OUTFILE"
    echo "--- CUDA @720p FP16 ---" | tee -a "$OUTFILE"
    python benchmarks/latency_benchmark.py --device cuda --fp16 --resolution 720p --runs 100 | tee -a "$OUTFILE"
else
    echo "CUDA not available — skipping GPU benchmark" | tee -a "$OUTFILE"
fi

echo "" | tee -a "$OUTFILE"
echo "Results saved to $OUTFILE"
