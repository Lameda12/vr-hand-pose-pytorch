# Turin — VR Hand Pose PyTorch

Real-time egocentric hand tracking + 3D pose estimation for VR interaction.

Target: **<50ms end-to-end @720p/30fps** on a mid-range CPU. GPU path auto-activates (FP16, ~4ms inference).

---

## Architecture

```
Webcam (BGR)
    │
    ▼
Preprocess (BGR→RGB, normalize)              ~1ms
    │
    ▼
HandPoseNet (MobileNetV3 backbone)
  ├─ DetectionHead   → objectness + bbox     stride-8 feature map
  ├─ HeatmapHead     → 21 keypoints (u,v)   stride-4 heatmaps
  └─ DepthHead       → z_rel per keypoint   [-1,1] relative depth
    │
    ▼
KalmanTracker / SORTTracker                  temporal smoothing, ID persistence
    │
    ▼
GestureState                                 pinch · point · fist · open_palm
    │
    ▼
Visualizer                                   skeleton overlay + gesture UI
```

**Keypoint convention**: MediaPipe / FreiHAND 21-point layout (wrist + 4 joints × 5 fingers).

---

## Project Status

### Completed

| Component | File(s) | Notes |
|---|---|---|
| CLAUDE.md | `CLAUDE.md` | Project config, coding standards, workflow |
| MobileNetV3 backbone | `src/models/backbone.py` | Pretrained, stride-8, 96ch output |
| Lightweight CPU backbone | `src/models/backbone.py` | DW-sep CNN, 64ch, ~2MB params |
| HandPoseNet | `src/models/hand_pose_net.py` | Detection + heatmap + depth heads |
| Loss functions | `src/models/losses.py` | FocalLoss + HeatmapLoss + DepthLoss |
| Kalman tracker | `src/tracker/kalman_tracker.py` | 84-dim state, IoU association |
| SORT tracker | `src/tracker/sort_tracker.py` | Confidence-gated, wraps Kalman |
| Gesture state machine | `src/demo/gesture_state.py` | Pinch/point/fist/palm + drag delta |
| Webcam demo | `src/demo/webcam_demo.py` | Full pipeline, drag-circle UI |
| Visualizer | `src/demo/visualizer.py` | Per-finger colors, gesture overlays |
| FreiHAND dataloader | `src/data/freihand.py` | RGB + 21-kp + depth, train/val/eval splits |
| Gaussian heatmap GT | `src/data/heatmap_utils.py` | σ=2 Gaussian blobs, softargmax, coord scaling |
| Data augmentation | `src/data/augmentation.py` | Flip, rotate, scale, color jitter, occlusion |
| PyTorch Lightning trainer | `src/train.py` | Train/val loop, AdamW + cosine LR, W&B/CSV |
| Evaluation metrics | `src/evaluate.py` | PCK@0.2, AUC, MPJPE, tracking ID switch rate |
| C++ preprocess wrapper | `cpp/preprocess.cpp` | pybind11, AVX2, optional CUDA path |
| C++ build system | `cpp/CMakeLists.txt` | cmake, optional -DUSE_CUDA |
| Unit tests — models | `tests/test_models.py` | Shape, loss, batch consistency |
| Unit tests — tracker | `tests/test_tracker.py` | ID persistence, IoU, SORT gating |
| Unit tests — gesture | `tests/test_gesture.py` | Pinch, drag, extension detection |
| Unit tests — data | `tests/test_data.py` | Heatmaps, augmentation, scaling roundtrip |
| Unit tests — evaluate | `tests/test_evaluate.py` | PCK, MPJPE, ID switch rate, CSV logging |
| Latency benchmark | `benchmarks/latency_benchmark.py` | CPU/GPU, P95/P99, FPS report |
| Model config | `configs/model_config.yaml` | All hyperparams, zero hardcoding |
| Docker | `docker/Dockerfile`, `docker-compose.yml` | CPU + GPU targets |
| Scripts | `scripts/run_demo.sh`, `scripts/run_benchmark.sh` | |
| pyproject.toml | `pyproject.toml` | Deps, dev/train extras |

### In Progress / Next

| Task | Priority | Notes |
|---|---|---|
| SORT Hungarian assignment | `src/tracker/sort_tracker.py` | scipy.linear_sum_assignment, globally optimal |
| HO3D dataloader | `src/data/ho3d.py` | Hand-object occlusion dataset, compatible API |
| Combined dataloader | `src/data/ho3d.py:build_combined_dataloader` | FreiHAND + HO3D concat |
| Checkpoint export | `scripts/export_checkpoint.py` | Lightning .ckpt → plain .pt state dict |
| Torch profiler | `benchmarks/profile_inference.py` | Op-level breakdown, chrome trace output |
| Config: FreiHAND baseline | `configs/train_freihand.yaml` | Single-dataset training config |
| Config: combined fine-tune | `configs/train_combined.yaml` | FreiHAND + HO3D robustness config |

### Remaining / Future

| Task | Priority | Notes |
|---|---|---|
| Open3D 3D visualization | Low | Project (u,v,z_rel) → live 3D skeleton |
| TensorRT/ONNX export | Future | Only after baseline PCK@0.2 >0.6 validated |
| HO3D full integration test | Low | Needs dataset download |

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run Demo (random weights — for pipeline validation only)

```bash
bash scripts/run_demo.sh
# or with a checkpoint:
bash scripts/run_demo.sh --checkpoint checkpoints/best.pt
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Benchmark

```bash
bash scripts/run_benchmark.sh
# or directly:
python benchmarks/latency_benchmark.py --device cpu --resolution 720p
python benchmarks/latency_benchmark.py --device cuda --fp16 --resolution 720p
```

### Profile (find hot ops)

```bash
python benchmarks/profile_inference.py --device cuda --resolution 720p
# → benchmarks/profile_trace.json  (open in chrome://tracing)
# → benchmarks/profile_summary.txt (top-20 ops by CUDA time)
```

### Train

```bash
# FreiHAND baseline
python -m src.train --config configs/train_freihand.yaml --data /data/freihand

# Smoke test (1 batch only)
python -m src.train --config configs/train_freihand.yaml --data /data/freihand --fast-dev-run

# Combined FreiHAND + HO3D (after baseline converges)
python -m src.train --config configs/train_combined.yaml --data /data/freihand \
    --checkpoint checkpoints/best.ckpt

# Export trained model for deployment
python scripts/export_checkpoint.py \
    --ckpt checkpoints/best.ckpt \
    --output checkpoints/best.pt \
    --verify
```

### Build C++ Preprocessing Extension

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Optional: cmake -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# Then in Python:
import sys; sys.path.insert(0, "cpp/build")
import preprocess_cpp
tensor = preprocess_cpp.preprocess_bgr(frame_bgr, out_w=640, out_h=480)
```

### Docker

```bash
# Run tests
docker compose -f docker/docker-compose.yml run test

# CPU benchmark
docker compose -f docker/docker-compose.yml run benchmark-cpu
```

---

## Performance Targets

| Metric | Target | Status |
|---|---|---|
| End-to-end latency @720p CPU | <50ms | Pending real-weight benchmark |
| End-to-end latency @720p GPU FP16 | <15ms | Pending real-weight benchmark |
| Backbone forward @720p GPU | <4ms | Architecture validated |
| Tracking ID switch rate | <5% | Tracker implemented |
| mAP on 100-frame validation set | >0.6 | Training pipeline pending |

---

## Keypoint Layout

```
        4   8   12  16  20
        |   |   |   |   |
        3   7   11  15  19
        |   |   |   |   |
        2   6   10  14  18
        |   |   |   |   |
        1   5   9   13  17
         \  |   |   |  /
              0 (wrist)
```

Indices: 0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Pinky

---

## Gesture Outputs

| Gesture | Trigger | VR Use |
|---|---|---|
| `PINCH` | thumb-index dist < 30px | drag, select |
| `POINT` | index extended, others curled | ray-cast click |
| `OPEN_PALM` | 4+ fingers extended | grab, dismiss |
| `FIST` | all fingers curled | hold, anchor |

`drag_delta`: cumulative (Δu, Δv) displacement from pinch origin — drives drag-circle UI.
`point_ray`: unit (dx, dy) direction from wrist to index tip — maps to VR ray-cast.

---

## References

- FreiHAND dataset: Zimmermann et al., ICCV 2019
- HO3D dataset: Hampali et al., CVPR 2020
- SORT tracker: Bewley et al., arXiv 1602.00763
- Prior work: Eye Draw gaze-tracking (MediaPipe-based, 500+ test sessions)
