"""
Open3D 3D hand skeleton visualization.

Projects (u, v, z_rel) keypoints into 3D space and renders a live
skeleton using Open3D's non-blocking visualizer.

Coordinate system:
  - x: right  (image u)
  - y: down    (image v)
  - z: depth   (z_rel from DepthHead, scaled by focal_px)

Usage (standalone):
    from src.demo.viz3d import HandViz3D
    viz = HandViz3D()
    viz.open()
    for frame in ...:
        keypoints_uv  = model.infer(frame)["keypoints"][0]   # [21, 2]
        depth_rel     = model.infer(frame)["depth"][0]       # [21]
        K             = camera_intrinsics                    # [3, 3]
        viz.update(keypoints_uv.numpy(), depth_rel.numpy(), K)
    viz.close()

Requires: open3d>=0.18 (not in default deps — add to extras or install manually)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Skeleton edges (MediaPipe 21-kp layout)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Per-finger colors (RGB float)
_FINGER_COLORS = {
    "thumb":  [0.58, 0.08, 1.00],
    "index":  [0.00, 0.00, 1.00],
    "middle": [0.00, 1.00, 0.00],
    "ring":   [1.00, 0.65, 0.00],
    "pinky":  [1.00, 0.00, 0.00],
    "palm":   [0.80, 0.80, 0.80],
}

_EDGE_FINGER_MAP = {
    frozenset({0, 1}): "thumb",  frozenset({1, 2}): "thumb",
    frozenset({2, 3}): "thumb",  frozenset({3, 4}): "thumb",
    frozenset({0, 5}): "index",  frozenset({5, 6}): "index",
    frozenset({6, 7}): "index",  frozenset({7, 8}): "index",
    frozenset({0, 9}): "middle", frozenset({9, 10}): "middle",
    frozenset({10, 11}): "middle", frozenset({11, 12}): "middle",
    frozenset({0, 13}): "ring",  frozenset({13, 14}): "ring",
    frozenset({14, 15}): "ring", frozenset({15, 16}): "ring",
    frozenset({0, 17}): "pinky", frozenset({17, 18}): "pinky",
    frozenset({18, 19}): "pinky", frozenset({19, 20}): "pinky",
    frozenset({5, 9}): "palm",  frozenset({9, 13}): "palm",
    frozenset({13, 17}): "palm",
}

# Scale factor: z_rel [-1,1] → meters (heuristic: typical hand depth range ~0.1m)
_DEPTH_SCALE_M = 0.05


def _unproject(
    kp_uv: np.ndarray,
    depth_rel: np.ndarray,
    K: np.ndarray,
    depth_scale: float = _DEPTH_SCALE_M,
) -> np.ndarray:
    """
    Back-project 2D keypoints + relative depth to 3D points.

    Formula:
        z   = z_anchor + z_rel * depth_scale   (heuristic; improves with real depth)
        x   = (u - cx) * z / fx
        y   = (v - cy) * z / fy

    Args:
        kp_uv:      [21, 2] (u, v) in image pixel coords
        depth_rel:  [21] float in [-1, 1]
        K:          [3, 3] camera intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
        depth_scale:meters per unit of depth_rel

    Returns:
        pts3d: [21, 3] float64 (x, y, z) in meters
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Heuristic anchor: wrist at 0.5m (typical arm-extended distance in VR)
    z_anchor = 0.5
    z_vals = z_anchor + depth_rel * depth_scale   # [21]

    x_vals = (kp_uv[:, 0] - cx) * z_vals / (fx + 1e-6)
    y_vals = (kp_uv[:, 1] - cy) * z_vals / (fy + 1e-6)

    return np.stack([x_vals, y_vals, z_vals], axis=-1).astype(np.float64)


class HandViz3D:
    """
    Non-blocking Open3D visualizer for real-time 3D hand skeleton.

    Updates geometry each frame without blocking the main thread.
    Falls back gracefully with a warning if Open3D is not installed.

    Args:
        window_name: title of the Open3D window
        width/height: window dimensions in pixels
        point_size:   rendered keypoint sphere radius

    Lifecycle:
        viz = HandViz3D()
        viz.open()               # creates window
        viz.update(kp, depth, K) # call each frame
        viz.close()              # destroys window
    """

    def __init__(
        self,
        window_name: str = "Turin — 3D Hand Pose",
        width: int = 800,
        height: int = 600,
        point_size: float = 0.008,
    ) -> None:
        self._window_name = window_name
        self._width = width
        self._height = height
        self._point_size = point_size
        self._vis = None
        self._pcd = None
        self._lines = None
        self._open3d_available = self._check_open3d()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Open the visualizer window.

        Returns:
            True if window opened successfully, False if Open3D unavailable.
        """
        if not self._open3d_available:
            logger.warning("Open3D not installed — 3D visualization unavailable. "
                           "Install with: pip install open3d>=0.18")
            return False

        import open3d as o3d
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(self._window_name, self._width, self._height)

        # Initialize geometry objects (updated in-place each frame)
        self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
        self._pcd.colors = o3d.utility.Vector3dVector(np.ones((21, 3)))

        self._lines = o3d.geometry.LineSet()
        self._lines.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
        self._lines.lines  = o3d.utility.Vector2iVector(
            [[i, j] for i, j in SKELETON_EDGES]
        )
        edge_colors = [
            _FINGER_COLORS.get(_EDGE_FINGER_MAP.get(frozenset({i, j}), "palm"), [0.8, 0.8, 0.8])
            for i, j in SKELETON_EDGES
        ]
        self._lines.colors = o3d.utility.Vector3dVector(edge_colors)

        # Add coordinate frame for orientation reference
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self._vis.add_geometry(frame)
        self._vis.add_geometry(self._pcd)
        self._vis.add_geometry(self._lines)

        # Render options
        opt = self._vis.get_render_option()
        opt.point_size = self._point_size * 1000   # Open3D uses screen-space units
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.line_width = 3.0

        logger.info("Open3D visualizer opened")
        return True

    def update(
        self,
        keypoints_uv: np.ndarray,
        depth_rel: np.ndarray,
        K: np.ndarray,
    ) -> None:
        """
        Update 3D skeleton with new keypoints.

        Args:
            keypoints_uv: [21, 2] float32/64 (u, v) in image pixel coords
            depth_rel:    [21] float32/64 in [-1, 1]
            K:            [3, 3] camera intrinsics

        Call this once per frame at ~30fps. Non-blocking.
        """
        if not self._open3d_available or self._vis is None:
            return

        import open3d as o3d

        pts3d = _unproject(keypoints_uv, depth_rel, K)  # [21, 3]

        self._pcd.points = o3d.utility.Vector3dVector(pts3d)
        # Color joints by finger (use wrist color for joint 0)
        joint_colors = np.array([
            _FINGER_COLORS.get(self._joint_finger(i), [1.0, 1.0, 1.0])
            for i in range(21)
        ])
        self._pcd.colors = o3d.utility.Vector3dVector(joint_colors)

        self._lines.points = o3d.utility.Vector3dVector(pts3d)

        self._vis.update_geometry(self._pcd)
        self._vis.update_geometry(self._lines)
        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        """Destroy the visualizer window."""
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
            logger.info("Open3D visualizer closed")

    def __enter__(self) -> "HandViz3D":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check_open3d() -> bool:
        try:
            import open3d  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _joint_finger(idx: int) -> str:
        if idx == 0:
            return "palm"
        finger = (idx - 1) // 4
        return ["thumb", "index", "middle", "ring", "pinky"][finger]


def project_and_display_once(
    keypoints_uv: np.ndarray,
    depth_rel: np.ndarray,
    K: np.ndarray,
) -> None:
    """
    One-shot static display: project keypoints to 3D and block until window closed.

    Useful for debugging a single frame without setting up the full update loop.

    Args:
        keypoints_uv: [21, 2]
        depth_rel:    [21]
        K:            [3, 3]
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("Open3D not installed — cannot display 3D skeleton.")
        return

    pts3d = _unproject(keypoints_uv, depth_rel, K)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((21, 3)))

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(pts3d)
    lines.lines  = o3d.utility.Vector2iVector([[i, j] for i, j in SKELETON_EDGES])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(
        [pcd, lines, frame],
        window_name="Turin — 3D Hand (static)",
        width=800,
        height=600,
    )
