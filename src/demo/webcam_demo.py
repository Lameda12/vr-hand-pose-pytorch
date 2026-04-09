"""
Webcam real-time hand pose demo.

Pipeline:
  BGR frame → preprocess → HandPoseNet.infer() → KalmanTracker.update()
    → GestureState.update() → viz_skeleton() → display

Target: <50ms end-to-end @720p/30fps on mid-range CPU.
GPU path auto-activates when CUDA is available.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, List

import cv2
import numpy as np
import torch

from ..models import HandPoseNet
from ..tracker import KalmanHandTracker
from .gesture_state import GestureState, GestureResult, Gesture
from .visualizer import viz_skeleton, viz_gesture_overlay

logger = logging.getLogger(__name__)


# BGR→RGB is always required; comment marks the intent explicitly
_BGR_TO_RGB = True

# Keypoint skeleton connectivity (MediaPipe/FreiHAND layout)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),            # palm knuckles
]


class WebcamDemo:
    """
    Real-time webcam hand pose demo with gesture UI.

    Args:
        model_path:       path to HandPoseNet checkpoint (.pt); None = random weights
        backbone:         "mobilenetv3" | "lightweight"
        device:           "cuda" | "cpu" | "auto"
        input_size:       (W, H) to resize frames before inference
        det_threshold:    objectness score threshold for detection
        max_hands:        maximum tracked hands (default 2 for VR left+right)
        display_fps:      show FPS overlay
        pinch_threshold:  pixels for pinch gesture activation

    Usage:
        demo = WebcamDemo(model_path="checkpoints/best.pt")
        demo.run()   # blocks until 'q' pressed
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: str = "mobilenetv3",
        device: str = "auto",
        input_size: tuple[int, int] = (640, 480),
        det_threshold: float = 0.5,
        max_hands: int = 2,
        display_fps: bool = True,
        pinch_threshold: float = 30.0,
    ) -> None:
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Running on {self.device}")

        # Model
        self.model = HandPoseNet(backbone_name=backbone, pretrained_backbone=False)
        if model_path is not None:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info(f"Loaded checkpoint: {model_path}")
        self.model.to(self.device)
        self.model.eval()

        # FP16 on CUDA for latency
        self.use_fp16 = self.device.type == "cuda"
        if self.use_fp16:
            self.model = self.model.half()

        self.input_size = input_size       # (W, H)
        self.det_threshold = det_threshold
        self.display_fps = display_fps

        # Tracker + gesture per hand slot
        self.tracker = KalmanHandTracker(max_misses=5, min_hits=2)
        self.gesture_states = [GestureState(pinch_px_threshold=pinch_threshold) for _ in range(max_hands)]

        # Normalize params (ImageNet)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        if self.use_fp16:
            self._mean = self._mean.half()
            self._std = self._std.half()

        # Drag-circle UI state
        self._drag_circle_center: Optional[np.ndarray] = None
        self._drag_circle_radius = 60

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, camera_idx: int = 0) -> None:
        """
        Main loop. Opens webcam, runs pipeline, renders, blocks until 'q'.

        Args:
            camera_idx: cv2.VideoCapture index (0 = default webcam)
        """
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_idx}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)

        logger.info("Demo running — press 'q' to quit")
        frame_times: list[float] = []

        try:
            while True:
                t0 = time.perf_counter()

                ret, frame_bgr = cap.read()
                if not ret:
                    logger.warning("Empty frame; skipping")
                    continue

                # Validate at pipeline entry
                assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, \
                    f"Unexpected frame shape: {frame_bgr.shape}"

                render_frame = frame_bgr.copy()
                gesture_results = self._process_frame(frame_bgr, render_frame)

                # Drag-circle UI
                self._update_drag_ui(gesture_results, render_frame)

                # FPS display
                t1 = time.perf_counter()
                frame_times.append(t1 - t0)
                if len(frame_times) > 30:
                    frame_times.pop(0)

                if self.display_fps:
                    avg_ms = 1000 * sum(frame_times) / len(frame_times)
                    fps = 1000 / (avg_ms + 1e-6)
                    cv2.putText(
                        render_frame,
                        f"{fps:.1f} FPS  {avg_ms:.1f}ms",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Turin Hand Pose", render_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Demo stopped")

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame_bgr: np.ndarray,
        render_target: np.ndarray,
    ) -> list[GestureResult]:
        """
        Full inference + tracking + gesture pipeline for one frame.

        Args:
            frame_bgr:     [H, W, 3] uint8 BGR input
            render_target: [H, W, 3] to draw visualizations on (mutated)

        Returns:
            list of GestureResult per active track
        """
        tensor = self._preprocess(frame_bgr)

        with torch.no_grad():
            preds = self.model.infer(tensor, det_threshold=self.det_threshold)

        # Extract per-hand detections from model output
        detections = self._parse_detections(preds, frame_bgr.shape[:2])

        # Track
        tracks = self.tracker.update(detections)

        # Gesture + visualization
        gesture_results = []
        for i, track in enumerate(tracks[:len(self.gesture_states)]):
            kp2d = track.state[:42].reshape(21, 2)   # [21, 2] pixel coords (heatmap space)
            # Scale keypoints to display resolution
            kp_display = self._scale_keypoints(kp2d, frame_bgr.shape[:2])
            viz_skeleton(render_target, kp_display, SKELETON_EDGES, track.track_id)

            result = self.gesture_states[i].update(kp_display)
            gesture_results.append(result)
            viz_gesture_overlay(render_target, result, kp_display)

        return gesture_results

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        BGR uint8 → normalized float tensor on device.

        Args:
            frame_bgr: [H, W, 3] uint8

        Returns:
            [1, 3, H, W] float32/float16
        """
        # Resize to model input size
        frame_resized = cv2.resize(frame_bgr, self.input_size)
        # BGR → RGB (explicit conversion, always required)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) if _BGR_TO_RGB else frame_resized
        # HWC → CHW, uint8 → float [0,1]
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        if self.use_fp16:
            tensor = tensor.half()
        # ImageNet normalize
        tensor = (tensor - self._mean) / self._std
        return tensor.unsqueeze(0)   # [1, 3, H, W]

    def _parse_detections(
        self,
        preds: dict,
        frame_hw: tuple[int, int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Convert model predictions to tracker-compatible (keypoints, bbox) list.

        Args:
            preds:    dict from HandPoseNet.infer()
            frame_hw: (H, W) of original frame for coordinate scaling

        Returns:
            list of (keypoints[21,2], bbox[4]) in frame pixel coords
        """
        detections = []
        scores = preds["det_scores"][0].cpu().numpy()    # [Hf*Wf]
        keypoints = preds["keypoints"][0].cpu().numpy()  # [21, 2] — heatmap coords

        # Scale keypoints from heatmap space to frame space
        H, W = frame_hw
        Hm, Wm = self.input_size[1] // 4, self.input_size[0] // 4
        kp_frame = keypoints.copy()
        kp_frame[:, 0] *= W / Wm
        kp_frame[:, 1] *= H / Hm

        # Derive bbox from keypoints
        if scores.max() >= self.det_threshold:
            x1, y1 = kp_frame.min(0)
            x2, y2 = kp_frame.max(0)
            pad = 20
            bbox = np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad], dtype=np.float32)
            detections.append((kp_frame.astype(np.float32), bbox))

        return detections

    def _scale_keypoints(
        self,
        kp_heatmap: np.ndarray,
        frame_hw: tuple[int, int],
    ) -> np.ndarray:
        """Scale [21,2] from heatmap to frame resolution."""
        H, W = frame_hw
        Hm, Wm = self.input_size[1] // 4, self.input_size[0] // 4
        kp = kp_heatmap.copy()
        kp[:, 0] *= W / Wm
        kp[:, 1] *= H / Hm
        return kp

    def _update_drag_ui(
        self,
        gesture_results: list[GestureResult],
        frame: np.ndarray,
    ) -> None:
        """
        Render drag-circle UI element driven by pinch gesture.

        A circle appears at pinch start and follows the pinch midpoint.
        """
        for result in gesture_results:
            if result.gesture == Gesture.PINCH and result.drag_delta is not None:
                if self._drag_circle_center is None:
                    # Place circle at screen center on first pinch frame
                    h, w = frame.shape[:2]
                    self._drag_circle_center = np.array([w // 2, h // 2], dtype=np.float32)
                center = (self._drag_circle_center + result.drag_delta).astype(int)
                center = np.clip(center, 0, [frame.shape[1], frame.shape[0]])
                cv2.circle(frame, tuple(center), self._drag_circle_radius, (0, 200, 255), 3)
                cv2.putText(frame, "PINCH DRAG", (center[0] - 40, center[1] - self._drag_circle_radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                return

        self._drag_circle_center = None
