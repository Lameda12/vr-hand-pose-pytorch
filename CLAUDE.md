# CLAUDE.md - VR Hand Pose PyTorch Project (Turin)

Personal configuration for Claude sessions working on the VR hand tracking system.

---

## Project Identity

**Turin**: Real-time egocentric hand tracking + 3D pose estimation system for VR interaction demos.

**Goals**:
- Sub-50ms inference on CPU/GPU
- Robust under motion/occlusion
- Gesture-based UI control (pinch-drag, point-click)
- Upgrade path from Eye Draw eye-tracking project

**Status**: Active development, baseline implementation phase

---

## Tech Stack & Constraints

**Core Dependencies**:
- PyTorch 2.4+
- OpenCV 4.10+
- NumPy, scikit-image

**Model Architecture**:
- YOLOv8-pose or MediaPipe baseline
- Fine-tune on FreiHAND/HO3D dataset subset

**Tracking**:
- Kalman filter or SORT for temporal smoothing/ID persistence

**Demo Environment**:
- Webcam input → inference → gesture state → virtual UI
- Visualization via Matplotlib or Open3D

**Performance Optimization**:
- C++/OpenCV wrapper for critical hot paths (preprocess/inference)
- Half-precision where stable
- GPU acceleration via cv2.cuda when available

**Deployment Constraints**:
- No TensorRT/ONNX until baseline validated
- No cloud dependencies
- Docker for reproducibility

**Language Distribution**: 70% Python, 20% C++, 10% Bash/scripts

---

## Project Structure

```
vr-hand-pose-pytorch/
├── src/
│   ├── models/          # PyTorch model definitions
│   ├── tracker/         # Kalman/SORT tracking logic
│   └── demo/            # Webcam demo + gesture UI
├── tests/               # Unit + integration tests
├── benchmarks/          # Latency + accuracy evals
├── data/                # Dataset subsets
├── configs/             # Model/tracker configs
└── CLAUDE.md            # This file
```

---

## Coding Standards

**PyTorch**:
- Use `torch.no_grad()` everywhere in inference paths
- Enable half-precision (FP16) where numerically stable
- Explicit shape documentation in docstrings

**OpenCV**:
- Explicit BGR→RGB conversion with comments
- Use `cv2.cuda` module for GPU operations when available
- Validate frame dimensions at pipeline entry

**Naming Conventions**:
- Python: `snake_case` for functions/variables
- C++: `camelCase` for functions, `PascalCase` for classes
- Function prefixes: `detect_*`, `track_*`, `viz_*` for clarity

**Documentation**:
- Every function/class requires:
  - Parameter types and shapes
  - Return types and shapes
  - Performance characteristics (latency, memory)
  - Shape invariants (e.g., "input must be [B, 3, H, W]")

**Testing**:
- Minimum 80% code coverage
- Performance regression suite: latency <50ms @30fps
- Accuracy benchmarks on validation set

---

## Review Criteria (Self-Check Before Commit)

Before finalizing any code, verify:

1. **Latency**: End-to-end <50ms/frame @720p/30fps on mid-range laptop
2. **Accuracy**: mAP >0.6 on 100-frame validation set; tracking ID switch <5%
3. **Robustness**: No crashes on occlusion/full-hand exit; graceful degradation
4. **Modularity**: Swap models/trackers via config files; zero hardcoding
5. **VR-Relevant Outputs**: 3D pose + gesture state (pinch distance, point ray)
6. **Open Questions**: Document unresolved issues as GitHub issues immediately

---

## Experiment Workflow

**Phase Breakdown**:
1. Model baseline → tracker integration → demo → performance tuning

**Implementation Rhythm**:
- Iterative 200-500 line chunks
- Test after each chunk
- No feature stacking without validation

**Evaluation**:
- Run benchmarks after every change
- Log metrics to Weights & Biases or CSV
- Compare against baseline before merging

**Polish**:
- Refactor for readability (code should read like prose)
- Add inline performance notes
- Update benchmarks

---

## CLAUDE Instructions

**When working on this project**:
- Think step-by-step in `<thinking>` tags before implementation
- Output diffs/patches for existing files, not full rewrites
- Verify performance and accuracy claims before finalizing
- Ask clarifying questions if requirements are ambiguous
- Use direct, technical language (this is Valve-caliber work)

**Code Generation**:
- Prefer functional patterns where cleaner
- Include error handling for production paths
- Add performance-critical comments inline
- Validate tensor shapes at function boundaries

**Debugging**:
- Lead with root cause, not symptoms
- Include reproduction steps
- Suggest fix + verification method

---

## Communication Style for This Project

- Skip preamble. State the solution or finding directly.
- Use technical precision: "mAP dropped 0.15 due to occlusion" not "accuracy seems worse"
- When reviewing code: lead with what's broken, then what works
- If uncertain about CV-specific behavior, label as `[Inference]` or `[Unverified]`
- No motivational language. This is engineering work.

---

## Key Prompts for Development Phases

**Scaffold Repo**:
```
Using this CLAUDE.md, scaffold the PyTorch hand-pose repo structure.
Include src/{models,tracker,demo}, pyproject.toml, Docker, tests.
Output folder structure + __init__.py files.
```

**Model Training**:
```
Fine-tune YOLOv8-pose on FreiHAND subset for egocentric hands.
Use PyTorch Lightning. Include dataloader, trainer, validation mAP eval.
Target: 30fps inference. Verify shapes and latency.
```

**Performance Optimization**:
```
Profile this inference loop [paste code].
Optimize GPU/CPU transfers, add torch profiler hooks.
Target <50ms/frame. Output before/after benchmarks.
```

**Tracking Integration**:
```
Implement Kalman filter for hand keypoint smoothing.
Input: per-frame detections. Output: temporally consistent tracks.
Handle occlusion (predict-only mode) and re-identification.
```

**Demo Application**:
```
Build OpenCV+PyTorch webcam hand tracker:
detect → keypoints → skeleton visualization → pinch gesture detection.
Include drag-circle UI interaction. Optimize for <50ms latency.
```

---

## References

- **Eye Draw Project**: Prior work on gaze-tracking (MediaPipe-based), 500+ test sessions
- **Target Standard**: Valve Index controller CV pipeline (gesture recognition, sub-frame latency)
- **Dataset**: FreiHAND (egocentric RGB-D hand poses), HO3D (hand-object interaction)

---

## What NOT to Do

- Don't add TensorRT/ONNX export until baseline is validated
- Don't use cloud APIs or external dependencies
- Don't sacrifice modularity for minor performance gains
- Don't write "tutorial code" with excessive comments explaining basics
- Don't add features without corresponding benchmarks

---

## Project-Specific Context

This builds on the Eye Draw gaze-tracking project, applying CV perception skills to hand pose estimation. The goal is not just accurate detection but **real-time, interaction-ready tracking** suitable for VR controllers or AR gesture interfaces. Think Valve Index finger tracking quality, not demo-quality pose estimation.

Every line of code should answer: "Would this ship in a production VR system?"
