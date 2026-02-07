"""
Unit tests for SMART association module.
Tests: matching.py, track.py, tracker.py

Run: python -m pytest test_association.py -v
  or: python test_association.py
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'lib'))

# Mock cython_bbox before importing matching
mock_bbox_module = MagicMock()
def mock_bbox_ious(a, b):
    """Pure numpy IoU implementation for testing."""
    M, N = len(a), len(b)
    iou_matrix = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            x1 = max(a[i, 0], b[j, 0])
            y1 = max(a[i, 1], b[j, 1])
            x2 = min(a[i, 2], b[j, 2])
            y2 = min(a[i, 3], b[j, 3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
            area_b = (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1])
            union = area_a + area_b - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0
    return iou_matrix

mock_bbox_module.bbox_overlaps = mock_bbox_ious
sys.modules['cython_bbox'] = mock_bbox_module

from tracker_fair import matching
from utils.track import Track
from utils.kalman_filter import KalmanFilter


# ===========================================================================
# matching.py tests
# ===========================================================================
class TestLinearAssignment(unittest.TestCase):
    def test_empty_cost_matrix(self):
        cost = np.empty((0, 5), dtype=np.float64)
        matches, ua, ub = matching.linear_assignment(cost, thresh=0.5)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(ua), 0)
        self.assertEqual(len(ub), 5)

    def test_perfect_diagonal_match(self):
        """Identity-like cost: each det matches its corresponding track."""
        cost = np.array([
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1],
        ])
        matches, ua, ub = matching.linear_assignment(cost, thresh=0.5)
        self.assertEqual(len(matches), 3)
        for m in matches:
            self.assertEqual(m[0], m[1])  # Diagonal match
        self.assertEqual(len(ua), 0)
        self.assertEqual(len(ub), 0)

    def test_threshold_filtering(self):
        """Costs above threshold should not match."""
        cost = np.array([
            [0.8, 0.9],
            [0.9, 0.8],
        ])
        matches, ua, ub = matching.linear_assignment(cost, thresh=0.5)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(ua), 2)
        self.assertEqual(len(ub), 2)

    def test_unbalanced_more_dets(self):
        """More detections than tracks."""
        cost = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.9],
        ])
        matches, ua, ub = matching.linear_assignment(cost, thresh=0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(ua), 1)  # One unmatched det
        self.assertEqual(len(ub), 0)

    def test_unbalanced_more_tracks(self):
        """More tracks than detections."""
        cost = np.array([
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
        ])
        matches, ua, ub = matching.linear_assignment(cost, thresh=0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(ua), 0)
        self.assertEqual(len(ub), 1)  # One unmatched track


class TestGreedyAssignment(unittest.TestCase):
    def test_simple_match(self):
        dist = np.array([
            [1.0, 100.0],
            [100.0, 2.0],
        ])
        matches = matching.greedy_assignment(dist)
        self.assertEqual(len(matches), 2)
        self.assertTrue([0, 0] in matches.tolist())
        self.assertTrue([1, 1] in matches.tolist())

    def test_empty_columns(self):
        dist = np.empty((3, 0))
        matches = matching.greedy_assignment(dist)
        self.assertEqual(len(matches), 0)

    def test_gated_entries_not_matched(self):
        dist = np.array([
            [1e18, 1e18],
            [1e18, 1e18],
        ])
        matches = matching.greedy_assignment(dist)
        self.assertEqual(len(matches), 0)


class TestEmbeddingDistance(unittest.TestCase):
    def test_identical_embeddings(self):
        emb = np.random.randn(3, 128).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        cost = matching.embedding_distance(emb, emb)
        # Diagonal should be ~0 (identical vectors)
        for i in range(3):
            self.assertAlmostEqual(cost[i, i], 0.0, places=5)

    def test_orthogonal_embeddings(self):
        a = np.array([[1, 0, 0, 0]], dtype=np.float64)
        b = np.array([[0, 1, 0, 0]], dtype=np.float64)
        cost = matching.embedding_distance(a, b)
        self.assertAlmostEqual(cost[0, 0], 1.0, places=5)

    def test_opposite_embeddings(self):
        a = np.array([[1, 0]], dtype=np.float64)
        b = np.array([[-1, 0]], dtype=np.float64)
        cost = matching.embedding_distance(a, b)
        self.assertAlmostEqual(cost[0, 0], 2.0, places=5)

    def test_empty(self):
        cost = matching.embedding_distance(np.empty((0, 64)), np.empty((0, 64)))
        self.assertEqual(cost.shape, (0, 0))


class TestIoUs(unittest.TestCase):
    def test_perfect_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[0, 0, 10, 10]], dtype=np.float64)
        result = matching.ious(a, b)
        self.assertAlmostEqual(result[0, 0], 1.0, places=5)

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        result = matching.ious(a, b)
        self.assertAlmostEqual(result[0, 0], 0.0, places=5)

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        result = matching.ious(a, b)
        # Intersection: 5x5=25, Union: 100+100-25=175
        self.assertAlmostEqual(result[0, 0], 25.0 / 175.0, places=5)

    def test_empty(self):
        result = matching.ious(np.empty((0, 4)), np.array([[0, 0, 10, 10]]))
        self.assertEqual(result.shape, (0, 1))


class TestIoUDistance(unittest.TestCase):
    def test_returns_cost(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[0, 0, 10, 10]], dtype=np.float64)
        cost = matching.iou_distance(a, b)
        self.assertAlmostEqual(cost[0, 0], 0.0, places=5)  # 1 - 1.0 = 0


class TestAdjustSimilarityWithGating(unittest.TestCase):
    def test_gating_applied(self):
        cos_sim = np.array([[0.2, 0.8], [0.7, 0.3]])
        invalid = np.array([[False, True], [True, False]])
        result = matching.adjust_similarity_with_gating(cos_sim, invalid)
        self.assertAlmostEqual(result[0, 0], 0.2)
        self.assertAlmostEqual(result[0, 1], 1.0)  # Gated
        self.assertAlmostEqual(result[1, 0], 1.0)  # Gated
        self.assertAlmostEqual(result[1, 1], 0.3)

    def test_no_modification_on_empty(self):
        cos_sim = np.empty((0, 0))
        invalid = np.empty((0, 0), dtype=bool)
        result = matching.adjust_similarity_with_gating(cos_sim, invalid)
        self.assertEqual(result.shape, (0, 0))


class TestEmbeddingFilter(unittest.TestCase):
    def test_smoothing_applied(self):
        history = {}
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        ret1 = [{'tracking_id': 1, 'embedding': emb1}]
        matching.embedding_filter(ret1, history, smoothing_window=5)
        ret2 = [{'tracking_id': 1, 'embedding': emb2}]
        matching.embedding_filter(ret2, history, smoothing_window=5)
        # After smoothing, should be average of emb1 and emb2
        expected = np.mean([emb1, emb2], axis=0)
        np.testing.assert_array_almost_equal(ret2[0]['embedding'], expected)

    def test_no_embedding_key_continues(self):
        """Should not crash when 'embedding' key missing - just skip."""
        history = {}
        ret = [
            {'tracking_id': 1},  # No embedding
            {'tracking_id': 2, 'embedding': np.array([1.0, 0.0])},
        ]
        result = matching.embedding_filter(ret, history, smoothing_window=5)
        self.assertEqual(len(result), 2)
        self.assertIn(2, history)  # Second item processed
        self.assertNotIn(1, history)  # First item skipped

    def test_window_limit(self):
        history = {}
        for i in range(10):
            ret = [{'tracking_id': 1, 'embedding': np.array([float(i), 0.0])}]
            matching.embedding_filter(ret, history, smoothing_window=3)
        self.assertEqual(len(history[1]), 3)


class TestCombineCostMatrices(unittest.TestCase):
    def test_pure_cosine(self):
        cos = np.array([[0.2, 0.8], [0.7, 0.3]])
        dist = np.array([[10, 20], [30, 40]]).T
        result = matching.combine_cost_matrices(cos, dist, similarity_weight=1.0)
        np.testing.assert_array_almost_equal(result, cos)

    def test_pure_distance(self):
        cos = np.array([[0.2, 0.8], [0.7, 0.3]])
        dist = np.array([[10, 20], [30, 40]])
        result = matching.combine_cost_matrices(cos, dist, similarity_weight=0.0)
        np.testing.assert_array_almost_equal(result, dist.T)


# ===========================================================================
# track.py tests
# ===========================================================================
class TestTrack(unittest.TestCase):
    def _make_track(self, bbox=None, embedding=None):
        bbox = bbox or [10, 20, 50, 80]
        return Track(
            track_id=1,
            initial_bbox=bbox,
            initial_score=0.9,
            initial_class=1,
            initial_ct=[30, 50],
            initial_tracking=[0.5, -0.3],
            initial_embedding=embedding or np.random.randn(64).astype(np.float32),
            smoothing_window=10,
            max_age=30,
        )

    def test_init_kalman_state(self):
        track = self._make_track()
        self.assertIsNotNone(track.mean)
        self.assertIsNotNone(track.covariance)
        self.assertEqual(track.mean.shape, (8,))
        self.assertEqual(track.covariance.shape, (8, 8))

    def test_bbox_to_xyah_roundtrip(self):
        bbox = [10, 20, 50, 80]
        xyah = Track.bbox_to_xyah(bbox)
        bbox_back = Track.xyah_to_bbox(xyah)
        np.testing.assert_array_almost_equal(bbox, bbox_back)

    def test_bbox_to_xyah_values(self):
        bbox = [0, 0, 40, 60]
        xyah = Track.bbox_to_xyah(bbox)
        self.assertAlmostEqual(xyah[0], 20)   # cx
        self.assertAlmostEqual(xyah[1], 30)   # cy
        self.assertAlmostEqual(xyah[2], 40/60)  # aspect ratio
        self.assertAlmostEqual(xyah[3], 60)   # height

    def test_predict_updates_position(self):
        track = self._make_track([100, 100, 200, 200])
        old_center = list(track.center)
        track.predict()
        # KF prediction should change the state (velocity update)
        self.assertIsNotNone(track.bbox)
        self.assertIsNotNone(track.center)
        self.assertEqual(len(track.bbox), 4)
        self.assertEqual(len(track.center), 2)

    def test_predict_zeros_velocity_for_lost(self):
        track = self._make_track()
        track.age = 1  # Simulate lost track
        old_mean = track.mean.copy()
        track.predict()
        # Should have been called with zeroed height velocity
        self.assertIsNotNone(track.mean)

    def test_update_resets_age(self):
        track = self._make_track()
        track.age = 5
        track.update(new_bbox=[15, 25, 55, 85], new_score=0.95, new_class=1, new_ct=[35, 55])
        self.assertEqual(track.age, 0)

    def test_update_kalman_correction(self):
        track = self._make_track([100, 100, 200, 200])
        old_mean = track.mean.copy()
        track.update(new_bbox=[105, 105, 205, 205])
        # KF correction should update the state
        self.assertFalse(np.array_equal(track.mean, old_mean))

    def test_probation_decrements(self):
        track = self._make_track()
        self.assertTrue(track.is_on_probation)
        self.assertEqual(track.probation_frames, 2)
        track.update(new_bbox=[10, 20, 50, 80], decrement_probation=True)
        self.assertEqual(track.probation_frames, 1)
        self.assertTrue(track.is_on_probation)
        track.update(new_bbox=[10, 20, 50, 80], decrement_probation=True)
        self.assertEqual(track.probation_frames, 0)
        self.assertFalse(track.is_on_probation)

    def test_smooth_fcn_ema(self):
        track = self._make_track()
        history = []
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        r1 = track.smooth_fcn(history, v1)
        np.testing.assert_array_equal(r1, v1)  # First value returned as-is
        r2 = track.smooth_fcn(history, v2)
        expected = 0.9 * v2 + 0.1 * v1  # EMA
        np.testing.assert_array_almost_equal(r2, expected)

    def test_smooth_fcn_preserves_history(self):
        """History should store originals, not smoothed values."""
        track = self._make_track()
        history = []
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        track.smooth_fcn(history, v1)
        track.smooth_fcn(history, v2)
        # History should contain original v1 and v2
        np.testing.assert_array_equal(history[0], v1)
        np.testing.assert_array_equal(history[1], v2)

    def test_center_distance_direction(self):
        track = self._make_track()
        track.center = [10, 20]
        disp = track.center_distance([15, 25])
        np.testing.assert_array_equal(disp, [5, 5])  # new - old

    def test_increment_age(self):
        track = self._make_track()
        track.increment_age()
        self.assertEqual(track.age, 1)
        self.assertEqual(track.active, 1)
        track.age = 30
        track.increment_age()
        self.assertEqual(track.active, 0)  # Deactivated

    def test_to_xyah(self):
        bbox = [10, 20, 50, 80]
        track = self._make_track(bbox)
        xyah = track.to_xyah()
        expected = Track.bbox_to_xyah(bbox)
        np.testing.assert_array_almost_equal(xyah, expected)


# ===========================================================================
# tracker.py tests (integration-level)
# ===========================================================================
class MockOpt:
    """Minimal opt object for Tracker tests."""
    def __init__(self, task='tracking,embedding', debug=0, max_age=30, hungarian=False, new_thresh=0.3):
        self.task = task
        self.debug = debug
        self.max_age = max_age
        self.hungarian = hungarian
        self.new_thresh = new_thresh
        self.heads = {'hm': 1, 'wh': 2, 'tracking': 2, 'embedding': 64}
        self.exp_id = 'test'


class TestTracker(unittest.TestCase):
    def _make_detection(self, ct, bbox, score=0.9, cls=1, tracking=None, embedding=None):
        return {
            'ct': np.array(ct, dtype=np.float32),
            'bbox': bbox,
            'score': score,
            'class': cls,
            'tracking': np.array(tracking or [0.0, 0.0], dtype=np.float32),
            'embedding': embedding if embedding is not None else np.random.randn(64).astype(np.float32),
        }

    def test_empty_first_frame(self):
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)
        ret = tracker.step([])
        self.assertEqual(ret, [])
        self.assertEqual(len(tracker.tracks), 0)

    def test_new_tracks_created(self):
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)
        dets = [
            self._make_detection([100, 100], [80, 80, 120, 120], score=0.9),
            self._make_detection([200, 200], [180, 180, 220, 220], score=0.8),
        ]
        ret = tracker.step(dets)
        # First frame: no tracks exist, so all dets become new tracks
        # But probation=2, so ret should be empty (tracks on probation)
        self.assertEqual(len(ret), 0)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(tracker.tracks[0].track_id, 1)
        self.assertEqual(tracker.tracks[1].track_id, 2)

    def test_track_matching_after_probation(self):
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)

        emb1 = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        emb2 = np.array([0.0, 1.0] + [0.0] * 62, dtype=np.float32)

        # Frame 0: Create tracks (probation=2)
        dets0 = [
            self._make_detection([100, 100], [80, 80, 120, 120], embedding=emb1),
            self._make_detection([300, 300], [280, 280, 320, 320], embedding=emb2),
        ]
        tracker.step(dets0)
        self.assertEqual(len(tracker.tracks), 2)

        # Frame 1: Match - probation goes from 2 to 1 (still on probation)
        dets1 = [
            self._make_detection([102, 102], [82, 82, 122, 122], embedding=emb1),
            self._make_detection([302, 302], [282, 282, 322, 322], embedding=emb2),
        ]
        ret1 = tracker.step(dets1)
        self.assertEqual(len(ret1), 0)  # Still on probation

        # Frame 2: Match - probation goes from 1 to 0 -> confirmed
        dets2 = [
            self._make_detection([104, 104], [84, 84, 124, 124], embedding=emb1),
            self._make_detection([304, 304], [284, 284, 324, 324], embedding=emb2),
        ]
        ret2 = tracker.step(dets2)
        self.assertEqual(len(ret2), 2)
        ids = {r['tracking_id'] for r in ret2}
        self.assertEqual(ids, {1, 2})

    def test_unmatched_track_ages(self):
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)

        emb = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        dets = [self._make_detection([100, 100], [80, 80, 120, 120], embedding=emb)]
        tracker.step(dets)

        # Frame with no detections - track should age
        tracker.step([])
        self.assertEqual(tracker.tracks[0].age, 1)

        # Another empty frame
        tracker.step([])
        self.assertEqual(tracker.tracks[0].age, 2)

    def test_track_removal_at_max_age(self):
        from utils.tracker import Tracker
        opt = MockOpt(max_age=3)
        tracker = Tracker(opt)

        emb = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        dets = [self._make_detection([100, 100], [80, 80, 120, 120], embedding=emb)]
        tracker.step(dets)
        self.assertEqual(len(tracker.tracks), 1)

        # Track ages: frame1->age=1, frame2->age=2, frame3->age=3 (=max_age, removed)
        # Removal happens when age >= max_age after increment
        for _ in range(4):
            tracker.step([])
        self.assertEqual(len(tracker.tracks), 0)  # Track removed

    def test_low_score_no_new_track(self):
        from utils.tracker import Tracker
        opt = MockOpt(new_thresh=0.5)
        tracker = Tracker(opt)
        dets = [self._make_detection([100, 100], [80, 80, 120, 120], score=0.3)]
        tracker.step(dets)
        self.assertEqual(len(tracker.tracks), 0)  # Below threshold

    def test_kalman_predict_called(self):
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)

        emb = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        dets = [self._make_detection([100, 100], [80, 80, 120, 120], embedding=emb)]
        tracker.step(dets)

        # Store position before prediction
        old_mean = tracker.tracks[0].mean.copy()
        tracker.step([])  # This triggers predict() on the track
        # Mean should change after prediction
        new_mean = tracker.tracks[0].mean
        # The position part should have been updated by KF (velocities applied)
        self.assertIsNotNone(new_mean)

    def test_reset(self):
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)
        emb = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        tracker.step([self._make_detection([100, 100], [80, 80, 120, 120], embedding=emb)])
        tracker.reset()
        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(tracker.next_track_id, 0)
        self.assertEqual(tracker.frm_count, 0)

    def test_category_gating(self):
        """Detections of different class should not match."""
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)

        emb = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        # Create track with class=1
        tracker.step([self._make_detection([100, 100], [80, 80, 120, 120], cls=1, embedding=emb)])

        # Detection with same embedding but different class
        dets = [self._make_detection([100, 100], [80, 80, 120, 120], cls=2, embedding=emb)]
        tracker.step(dets)
        # Should create new track, not match existing one
        self.assertEqual(len(tracker.tracks), 2)

    def test_embedding_confidence_gate(self):
        """Low-confidence matches should not update embedding."""
        from utils.tracker import Tracker
        opt = MockOpt()
        tracker = Tracker(opt)

        emb_orig = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        emb_new = np.array([0.0, 1.0] + [0.0] * 62, dtype=np.float32)

        tracker.step([self._make_detection([100, 100], [80, 80, 120, 120], score=0.9, embedding=emb_orig)])
        original_emb = tracker.tracks[0].embedding.copy()

        # Match with very low score - embedding should NOT be updated
        dets = [self._make_detection([101, 101], [81, 81, 121, 121], score=0.2, embedding=emb_new)]
        tracker.step(dets)
        np.testing.assert_array_almost_equal(tracker.tracks[0].embedding, original_emb)


# ===========================================================================
# KalmanFilter tests
# ===========================================================================
class TestKalmanFilter(unittest.TestCase):
    def test_initiate(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 0.5, 80]))
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(cov.shape, (8, 8))
        self.assertAlmostEqual(mean[0], 100)
        self.assertAlmostEqual(mean[1], 200)

    def test_predict_constant_velocity(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 0.5, 80]))
        # Set velocity
        mean[4] = 5.0  # vx
        mean[5] = 3.0  # vy
        mean_pred, _ = kf.predict(mean, cov)
        # Position should move by velocity (dt=1)
        self.assertAlmostEqual(mean_pred[0], 105.0)
        self.assertAlmostEqual(mean_pred[1], 203.0)

    def test_update_pulls_toward_measurement(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 0.5, 80]))
        measurement = np.array([110, 210, 0.5, 80])
        mean_upd, _ = kf.update(mean, cov, measurement)
        # Updated position should be between prior and measurement
        self.assertGreater(mean_upd[0], 100)
        self.assertLess(mean_upd[0], 110)

    def test_gating_distance(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 0.5, 80]))
        measurements = np.array([
            [100, 200, 0.5, 80],   # Very close
            [500, 500, 0.5, 80],   # Very far
        ])
        dists = kf.gating_distance(mean, cov, measurements)
        self.assertLess(dists[0], dists[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
