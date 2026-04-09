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
| Unit tests — models | `tests/test_models.py` | Shape, loss, batch consistency |
| Unit tests — tracker | `tests/test_tracker.py` | ID persistence, IoU, SORT gating |
| Unit tests — gesture | `tests/test_gesture.py` | Pinch, drag, extension detection |
| Latency benchmark | `benchmarks/latency_benchmark.py` | CPU/GPU, P95/P99, FPS report |
| Model config | `configs/model_config.yaml` | All hyperparams, zero hardcoding |
| Docker | `docker/Dockerfile`, `docker-compose.yml` | CPU + GPU targets |
| Scripts | `scripts/run_demo.sh`, `scripts/run_benchmark.sh` | |
| pyproject.toml | `pyproject.toml` | Deps, dev/train extras |

### In Progress / Next

| Task | Priority | Notes |
|---|---|---|
| FreiHAND dataloader | High | `src/data/freihand.py` — load RGB + 21-kp GT + depth |
| PyTorch Lightning trainer | High | `src/train.py` — train/val loop, mAP eval |
| Gaussian heatmap generation | High | Build GT heatmaps from kp coords (σ=2) |
| mAP evaluation | High | PCK metric on validation set, target >0.6 |
| C++ preprocess wrapper | Medium | `cpp/preprocess.cpp` — BGR→tensor hot path |
| Data augmentation pipeline | Medium | Flip, rotate, scale, color jitter |
| Checkpoint save/load | Medium | Best-by-mAP, resume training |
| SORT Hungarian assignment | Low | Replace greedy IoU with scipy.linear_sum_assignment |
| Open3D 3D visualization | Low | Project (u,v,z_rel) → 3D skeleton |

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
