# Turin — VR Hand Pose PyTorch

Real-time egocentric hand tracking + 3D pose estimation for VR interaction.

**Target**: <50ms end-to-end @720p/30fps on a mid-range CPU. GPU path (FP16) targets <15ms.

[![CI](https://github.com/lameda12/vr-hand-pose-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/lameda12/vr-hand-pose-pytorch/actions/workflows/ci.yml)

---

## Pipeline

```
Webcam (BGR)
    │
    ▼
Preprocess (BGR→RGB, ImageNet normalize)               ~1ms CPU / <0.5ms C++ ext
    │
    ▼
HandPoseNet
  Backbone ──────────────────── [B, C, H/8, W/8]
  ├─ DetectionHead    objectness + bbox delta           stride-8 feature map
  ├─ HeatmapHead      21 keypoint heatmaps (u,v)        stride-4, softargmax
  └─ DepthHead        z_rel per keypoint [-1,1]         global avg pool → FC
    │
    ▼
SORTTracker (Kalman + Hungarian assignment)             ID-persistent multi-hand
    │
    ▼
GestureState                                            pinch · point · fist · open_palm
    │
    ├── 2D Visualizer (OpenCV skeleton + gesture HUD)
    └── HandViz3D (Open3D live 3D skeleton, optional)
```

**Keypoint layout**: MediaPipe / FreiHAND 21-point — wrist(0) + 4 joints × 5 fingers.

---

## Project Status

### Complete

| Component | File | Notes |
|---|---|---|
| CLAUDE.md | `CLAUDE.md` | Project config, standards, workflow |
| **Models** | | |
| MobileNetV3 backbone | `src/models/backbone.py` | ImageNet pretrained, stride-8, 96ch |
| Lightweight CPU backbone | `src/models/backbone.py` | DW-sep CNN, 64ch, ~2MB, no pretrained |
| HandPoseNet | `src/models/hand_pose_net.py` | Detection + heatmap + depth, softargmax infer |
| Loss functions | `src/models/losses.py` | FocalLoss + HeatmapLoss + DepthLoss |
| **Data** | | |
| FreiHAND loader | `src/data/freihand.py` | 3D→2D projection, splits, depth normalization |
| HO3D loader | `src/data/ho3d.py` | Hand-object occlusion, same API as FreiHAND |
| Combined loader | `src/data/ho3d.py` | FreiHAND+HO3D `ConcatDataset` |
| Gaussian heatmap GT | `src/data/heatmap_utils.py` | σ=2 blobs, softargmax2d, coord scaling |
| Augmentation pipeline | `src/data/augmentation.py` | Flip, rotate, scale, color, noise, occlusion |
| **Tracking** | | |
| Kalman tracker | `src/tracker/kalman_tracker.py` | 84-dim state, IoU assoc., predict-only occlusion |
| SORT tracker | `src/tracker/sort_tracker.py` | Hungarian (scipy), confidence gating |
| **Demo** | | |
| Gesture state machine | `src/demo/gesture_state.py` | Pinch/point/fist/palm + drag delta + point ray |
| 2D Visualizer | `src/demo/visualizer.py` | Per-finger skeleton, gesture HUD |
| Webcam demo | `src/demo/webcam_demo.py` | Full pipeline, drag-circle UI, FPS overlay |
| Open3D 3D viz | `src/demo/viz3d.py` | Non-blocking live 3D skeleton, unproject u,v,z_rel |
| **Training** | | |
| PyTorch Lightning trainer | `src/train.py` | AdamW+cosine, W&B+CSV, early stopping |
| Evaluation metrics | `src/evaluate.py` | PCK@0.2, AUC, MPJPE, tracking ID switch rate |
| **Benchmarks** | | |
| Latency benchmark | `benchmarks/latency_benchmark.py` | P95/P99, FPS, PASS/FAIL vs 50ms |
| Accuracy benchmark | `benchmarks/accuracy_benchmark.py` | PCK@0.2 + AUC on val set |
| Torch profiler | `benchmarks/profile_inference.py` | Op-level chrome trace, top-20 ops |
| **C++** | | |
| Preprocessing hot path | `cpp/preprocess.cpp` | pybind11, AVX2 SIMD, GIL release, CUDA opt |
| CMake build | `cpp/CMakeLists.txt` | AVX2 detection, optional -DUSE_CUDA |
| **Tests** | | |
| Unit: models | `tests/test_models.py` | Shape, loss, batch consistency |
| Unit: tracker | `tests/test_tracker.py` | ID persistence, IoU, SORT gating |
| Unit: gesture | `tests/test_gesture.py` | Pinch, drag, extension |
| Unit: data | `tests/test_data.py` | Heatmap peak, augmentation, roundtrip |
| Unit: evaluate | `tests/test_evaluate.py` | PCK, MPJPE, ID switch, CSV write |
| Integration | `tests/test_integration.py` | Full pipeline synthetic, gradient flow |
| Shared fixtures | `tests/conftest.py` | Session-scoped model, frame, kp fixtures |
| **Config** | | |
| Base config | `configs/model_config.yaml` | All hyperparams, zero hardcoding |
| FreiHAND baseline | `configs/train_freihand.yaml` | Single-dataset training |
| Combined fine-tune | `configs/train_combined.yaml` | FreiHAND+HO3D robustness |
| **Ops** | | |
| CI | `.github/workflows/ci.yml` | Unit tests + latency smoke + lint |
| Docker | `docker/Dockerfile` + `docker-compose.yml` | CPU + GPU targets |
| Checkpoint export | `scripts/export_checkpoint.py` | Lightning .ckpt → deploy .pt |
| Data download | `scripts/download_data.sh` | FreiHAND + HO3D setup guide |
| Run demo | `scripts/run_demo.sh` | |
| Run benchmark | `scripts/run_benchmark.sh` | |
| pyproject.toml | `pyproject.toml` | All deps, dev/train extras |

### Remaining

| Task | Note |
|---|---|
| Run training to convergence | Need FreiHAND dataset (130k samples); target PCK@0.2 >0.6 |
| TensorRT/ONNX export | Only after baseline validated — see CLAUDE.md constraint |

---

## Quick Start

### Install

```bash
# CPU (CI/dev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# GPU + training extras
pip install torch torchvision
pip install -e ".[dev,train]"
```

### Run Demo

```bash
bash scripts/run_demo.sh                               # random weights (pipeline check)
bash scripts/run_demo.sh --checkpoint checkpoints/best.pt  # trained model
```

### Test

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Benchmark

```bash
bash scripts/run_benchmark.sh

# Direct:
python benchmarks/latency_benchmark.py --device cpu  --resolution 720p
python benchmarks/latency_benchmark.py --device cuda --fp16 --resolution 720p

# Profile hot ops:
python benchmarks/profile_inference.py --device cuda --resolution 720p
# → benchmarks/profile_trace.json  (open in chrome://tracing)
```

### Train

```bash
# Set up dataset
bash scripts/download_data.sh --freihand /data/freihand

# FreiHAND baseline
python -m src.train --config configs/train_freihand.yaml --data /data/freihand

# Smoke test (1 batch, no dataset needed with synthetic loader)
python -m src.train --config configs/train_freihand.yaml --data /data/freihand --fast-dev-run

# After baseline converges: fine-tune on FreiHAND+HO3D
python -m src.train \
    --config configs/train_combined.yaml \
    --data /data/freihand \
    --checkpoint checkpoints/best.ckpt

# Export for deployment
python scripts/export_checkpoint.py \
    --ckpt checkpoints/best.ckpt \
    --output checkpoints/best.pt \
    --verify
```

### Accuracy Eval

```bash
python benchmarks/accuracy_benchmark.py \
    --checkpoint checkpoints/best.pt \
    --data /data/freihand \
    --split val
```

### C++ Preprocessing Extension

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release      # CPU (AVX2)
# cmake -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release  # + CUDA resize
cmake --build build -j$(nproc)
```

```python
import sys; sys.path.insert(0, "cpp/build")
import preprocess_cpp
tensor_chw = preprocess_cpp.preprocess_bgr(frame_bgr, out_w=640, out_h=480)
# → np.ndarray [3, 480, 640] float32, RGB, ImageNet-normalized
# torch.from_numpy(tensor_chw).unsqueeze(0)  → ready for model
print(f"CUDA path available: {preprocess_cpp.has_cuda}")
```

### Docker

```bash
docker compose -f docker/docker-compose.yml run test
docker compose -f docker/docker-compose.yml run benchmark-cpu
docker compose -f docker/docker-compose.yml run benchmark-gpu   # requires nvidia-docker
```

---

## Performance Targets

| Metric | Target | Status |
|---|---|---|
| End-to-end @720p CPU | <50ms | Arch validated; benchmark pending real weights |
| End-to-end @720p GPU FP16 | <15ms | Arch validated |
| Backbone forward @720p GPU | <4ms | MobileNetV3 validated |
| Tracking ID switch rate | <5% | Kalman+SORT implemented |
| mAP / PCK@0.2 on val set | >0.60 | Training pending dataset |

---

## Gesture API

| Gesture | Trigger condition | VR action |
|---|---|---|
| `PINCH` | thumb–index tip dist < 30px | drag, select |
| `POINT` | index extended, others curled | ray-cast click |
| `OPEN_PALM` | 4+ fingers extended | grab, dismiss |
| `FIST` | all fingers curled | hold, anchor |

**`drag_delta`** — `np.ndarray [2]` cumulative (Δu, Δv) from pinch origin → drives drag-circle UI.

**`point_ray`** — `np.ndarray [2]` unit (dx, dy) from wrist to index tip → maps to VR ray-cast.

---

## Keypoint Layout

```
        4   8   12  16  20    ← fingertips
        |   |   |   |   |
        3   7   11  15  19
        |   |   |   |   |
        2   6   10  14  18
        |   |   |   |   |
        1   5   9   13  17    ← MCPs
         \  |   |   |  /
               0              ← wrist
```

`0`=Wrist · `1-4`=Thumb · `5-8`=Index · `9-12`=Middle · `13-16`=Ring · `17-20`=Pinky

---

## References

- FreiHAND: Zimmermann et al., ICCV 2019
- HO3D: Hampali et al., CVPR 2020
- SORT tracker: Bewley et al., arXiv 1602.00763
- MobileNetV3: Howard et al., ICCV 2019
- Focal Loss: Lin et al., ICCV 2017
- Prior work: Eye Draw gaze-tracking (MediaPipe, 500+ test sessions)
