"""
Unit tests for evaluation metrics.

Run: pytest tests/test_evaluate.py -v
"""

import pytest
import numpy as np
import torch
from src.evaluate import (
    compute_pck,
    compute_pck_curve,
    compute_mpjpe,
    PoseEvaluator,
    TrackingEvaluator,
    EvalMetrics,
    PCK_THRESHOLDS,
    save_metrics_csv,
)


def _perfect_kp() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred, gt, vis) where pred == gt."""
    gt = np.random.rand(21, 2).astype(np.float32) * 200
    return gt.copy(), gt, np.ones(21, dtype=np.float32)


def _bad_kp(offset: float = 200.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred, gt, vis) where pred is far from gt."""
    gt = np.zeros((21, 2), dtype=np.float32) + 100
    pred = gt + offset
    return pred, gt, np.ones(21, dtype=np.float32)


class TestComputePCK:
    def test_perfect_prediction(self):
        pred, gt, vis = _perfect_kp()
        pck = compute_pck(pred, gt, vis, threshold_frac=0.2)
        assert pck == pytest.approx(1.0)

    def test_bad_prediction(self):
        pred, gt, vis = _bad_kp(offset=500.0)
        pck = compute_pck(pred, gt, vis, threshold_frac=0.2)
        assert pck == pytest.approx(0.0)

    def test_range(self):
        pred = np.random.rand(21, 2).astype(np.float32) * 100
        gt = np.random.rand(21, 2).astype(np.float32) * 100
        vis = np.ones(21, dtype=np.float32)
        pck = compute_pck(pred, gt, vis)
        assert 0.0 <= pck <= 1.0

    def test_all_invisible(self):
        pred, gt, _ = _perfect_kp()
        vis = np.zeros(21, dtype=np.float32)
        pck = compute_pck(pred, gt, vis)
        assert pck == 0.0

    def test_partial_visibility(self):
        gt = np.zeros((21, 2), dtype=np.float32)
        gt[0] = [100, 100]
        gt[1] = [200, 100]   # adds bbox size
        pred = gt.copy()
        pred[0] = [100, 100]   # correct
        pred[1] = [200, 100]   # correct
        vis = np.array([1, 1] + [0] * 19, dtype=np.float32)
        pck = compute_pck(pred, gt, vis)
        assert pck == pytest.approx(1.0)


class TestComputePCKCurve:
    def test_shape(self):
        pred, gt, vis = _perfect_kp()
        curve = compute_pck_curve(pred, gt, vis)
        assert curve.shape == PCK_THRESHOLDS.shape

    def test_monotone_increasing(self):
        """PCK curve must be non-decreasing as threshold increases."""
        pred = np.random.rand(21, 2).astype(np.float32) * 100
        gt = np.random.rand(21, 2).astype(np.float32) * 100
        vis = np.ones(21, dtype=np.float32)
        curve = compute_pck_curve(pred, gt, vis)
        diffs = np.diff(curve)
        assert (diffs >= -1e-6).all(), f"Non-monotone PCK curve: {curve}"


class TestComputeMPJPE:
    def test_perfect(self):
        xyz = np.random.rand(21, 3).astype(np.float32)
        vis = np.ones(21, dtype=np.float32)
        assert compute_mpjpe(xyz, xyz.copy(), vis) == pytest.approx(0.0, abs=1e-5)

    def test_known_error(self):
        pred = np.zeros((21, 3), dtype=np.float32)
        gt = np.ones((21, 3), dtype=np.float32)   # distance = sqrt(3) per joint
        vis = np.ones(21, dtype=np.float32)
        expected = float(np.sqrt(3))
        assert compute_mpjpe(pred, gt, vis) == pytest.approx(expected, rel=1e-4)

    def test_invisible_excluded(self):
        pred = np.zeros((21, 3), dtype=np.float32)
        gt = np.ones((21, 3), dtype=np.float32)
        vis = np.zeros(21, dtype=np.float32)
        result = compute_mpjpe(pred, gt, vis)
        assert np.isnan(result)


class TestPoseEvaluator:
    def _make_batch(self, B: int = 4):
        """Fake batch + perfect predictions."""
        kp = torch.rand(B, 21, 2) * 100
        return (
            {"keypoints": kp.clone()},   # pred == gt → perfect
            {
                "keypoints": kp.clone(),
                "kp_mask": torch.ones(B, 21),
            }
        )

    def test_perfect_pck(self):
        ev = PoseEvaluator()
        pred, batch = self._make_batch(4)
        ev.update(pred, batch)
        m = ev.compute()
        assert m.pck_at_02 == pytest.approx(1.0)

    def test_accumulate_multiple_batches(self):
        ev = PoseEvaluator()
        for _ in range(3):
            pred, batch = self._make_batch(4)
            ev.update(pred, batch)
        m = ev.compute()
        assert m.n_samples == 12

    def test_reset(self):
        ev = PoseEvaluator()
        pred, batch = self._make_batch(4)
        ev.update(pred, batch)
        ev.reset()
        m = ev.compute()
        assert m.n_samples == 0

    def test_auc_range(self):
        ev = PoseEvaluator()
        pred, batch = self._make_batch(8)
        ev.update(pred, batch)
        m = ev.compute()
        assert 0.0 <= m.auc <= 1.0


class TestTrackingEvaluator:
    def _make_track(self, track_id: int, u: float = 100.0):
        """Minimal HandTrack-like object."""
        from src.tracker.kalman_tracker import HandTrack
        import numpy as np
        state = np.zeros(84, dtype=np.float32)
        state[0] = u   # wrist u for sorting
        return HandTrack(
            track_id=track_id,
            state=state,
            cov=np.eye(84, dtype=np.float32),
            hits=3,
        )

    def test_no_switches_stable(self):
        te = TrackingEvaluator()
        track = self._make_track(track_id=0, u=100.0)
        for _ in range(5):
            te.update([track])
        assert te.id_switch_rate() == pytest.approx(0.0)

    def test_switch_detected(self):
        te = TrackingEvaluator()
        t0 = self._make_track(track_id=0, u=100.0)
        te.update([t0])
        # Switch: same slot, different ID
        t1 = self._make_track(track_id=1, u=100.0)
        te.update([t1])
        assert te.id_switch_rate() > 0.0

    def test_reset(self):
        te = TrackingEvaluator()
        te.update([self._make_track(0)])
        te.update([self._make_track(1)])
        te.reset()
        assert te.id_switch_rate() == 0.0


class TestSaveMetricsCSV:
    def test_creates_file(self, tmp_path):
        m = EvalMetrics(pck_at_02=0.72, auc=0.81, n_samples=100)
        csv_path = tmp_path / "metrics.csv"
        save_metrics_csv(m, csv_path, epoch=0)
        assert csv_path.exists()

    def test_appends_rows(self, tmp_path):
        m = EvalMetrics(pck_at_02=0.72, auc=0.81, n_samples=100)
        csv_path = tmp_path / "metrics.csv"
        save_metrics_csv(m, csv_path, epoch=0)
        save_metrics_csv(m, csv_path, epoch=1)
        lines = csv_path.read_text().strip().splitlines()
        assert len(lines) == 3   # header + 2 rows

    def test_csv_header(self, tmp_path):
        m = EvalMetrics()
        csv_path = tmp_path / "metrics.csv"
        save_metrics_csv(m, csv_path, epoch=0)
        header = csv_path.read_text().splitlines()[0]
        assert "pck_at_02" in header
        assert "auc" in header
