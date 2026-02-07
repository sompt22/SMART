import numpy as np
from utils.kalman_filter import KalmanFilter


class Track(object):
    shared_kalman = KalmanFilter()

    def __init__(self, track_id, initial_bbox, initial_score, initial_class, initial_ct, initial_tracking, initial_embedding=None, smoothing_window=10, max_age=30):
        self.track_id = track_id
        self.bbox = initial_bbox  # [x1, y1, x2, y2]
        self.score = initial_score
        self.class_id = initial_class
        self.center = initial_ct
        self.center_disp_history = np.array((0, 0))
        self.tracking = initial_tracking if initial_tracking is not None else None
        self.tracking_history = [] if initial_tracking is None else [initial_tracking]
        self.embedding = initial_embedding if initial_embedding is not None else None
        self.embeddings_history = [] if initial_embedding is None else [initial_embedding]
        self.age = 0
        self.active = 1
        self.is_on_probation = True
        self.probation_frames = 2  # Number of frames to wait before activating the track
        self.max_age = max_age  # Tracks are considered inactive if not updated for this many frames
        self.smoothing_window = smoothing_window  # Number of embeddings to consider for smoothing

        # Kalman filter state
        self.mean, self.covariance = self.shared_kalman.initiate(self.bbox_to_xyah(self.bbox))

    @staticmethod
    def bbox_to_xyah(bbox):
        """Convert [x1, y1, x2, y2] to [cx, cy, aspect_ratio, height]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        a = w / h if h > 0 else 0
        return np.array([cx, cy, a, h])

    @staticmethod
    def xyah_to_bbox(xyah):
        """Convert [cx, cy, aspect_ratio, height] to [x1, y1, x2, y2]."""
        cx, cy, a, h = xyah
        w = a * h
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    def predict(self):
        """Run Kalman filter prediction step."""
        mean_state = self.mean.copy()
        if self.age > 0:
            # Zero out velocity components for lost tracks (reduce drift)
            mean_state[7] = 0
        self.mean, self.covariance = self.shared_kalman.predict(mean_state, self.covariance)
        # Update bbox and center from KF predicted state
        predicted_xyah = self.mean[:4]
        self.bbox = self.xyah_to_bbox(predicted_xyah)
        self.center = [predicted_xyah[0], predicted_xyah[1]]

    def update(self, new_bbox=None, new_score=None, new_class=None, new_ct=None, new_tracking=None, new_embedding=None, decrement_probation=True):
        """Update the track with new data."""
        if new_bbox is not None:
            self.bbox = new_bbox
            # Kalman filter correction step
            measurement = self.bbox_to_xyah(new_bbox)
            self.mean, self.covariance = self.shared_kalman.update(self.mean, self.covariance, measurement)
        if new_score is not None:
            self.score = new_score
        if new_class is not None:
            self.class_id = new_class
        if new_ct is not None:
            self.center_disp_history = self.center_distance(new_ct)
            self.center = new_ct
        if new_tracking is not None:
            self.tracking = new_tracking
        if new_embedding is not None:
            self.embedding = self.smooth_fcn(self.embeddings_history, new_embedding)
        self.age = 0

        # Decrease probation frames on update
        if self.is_on_probation and decrement_probation:
            self.probation_frames -= 1
            if self.probation_frames <= 0:
                self.is_on_probation = False

    def increment_age(self):
        """Increment the track's age, potentially deactivating it if it gets too old."""
        self.age += 1
        if self.age > self.max_age:
            self.active = 0

    def smooth_fcn(self, history, new_value):
        """Smooth the current vector using exponential moving average (EMA)."""
        alpha = 0.9  # Weight for the new value
        if len(history) == 0:
            history.append(new_value.copy())
            return new_value
        # EMA: emphasize recent embeddings over old ones
        smoothed = alpha * new_value + (1 - alpha) * history[-1]
        history.append(new_value.copy())  # Store original, not smoothed
        if len(history) > self.smoothing_window:
            history.pop(0)
        return smoothed

    def center_distance(self, new_center):
        """Calculate the displacement vector from old center to new center."""
        return np.array(new_center) - np.array(self.center)

    def to_xyah(self):
        """Return current state in [cx, cy, aspect_ratio, height] format."""
        return self.bbox_to_xyah(self.bbox)
