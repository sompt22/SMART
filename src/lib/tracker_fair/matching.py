import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from utils import kalman_filter


def linear_assignment(cost_matrix, thresh):
    """Solve the linear assignment problem using the Jonker-Volgenant algorithm."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def greedy_assignment(dist):
    """Greedy matching: assign each row to the nearest unassigned column."""
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


def merge_matches(m1, m2, shape):
    """Merge two sets of matches via sparse matrix multiplication."""
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))
    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))
    return match, unmatched_O, unmatched_Q


# ==================== Distance / Cost Functions ====================

def embedding_distance(tracks, detections, metric='cosine'):
    """Compute cosine distance between track and detection embedding arrays.

    Args:
        tracks: np.ndarray of shape (M, embedding_dim)
        detections: np.ndarray of shape (N, embedding_dim)
        metric: distance metric for cdist (default: 'cosine')

    Returns:
        cost_matrix: np.ndarray of shape (M, N), values in [0, 1] for cosine
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(0.0, cdist(tracks, detections, metric))
    return cost_matrix


def ious(atlbrs, btlbrs):
    """Compute IoU matrix using cython_bbox (vectorized).

    Args:
        atlbrs: np.ndarray (M, 4) in [x1, y1, x2, y2] format
        btlbrs: np.ndarray (N, 4) in [x1, y1, x2, y2] format

    Returns:
        iou_matrix: np.ndarray (M, N) with IoU values
    """
    iou_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if iou_matrix.size == 0:
        return iou_matrix
    iou_matrix = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )
    return iou_matrix


def iou_distance(atracks, btracks):
    """Compute IoU-based cost matrix (1 - IoU).

    Accepts either Track objects (with .tlbr property) or raw np.ndarray bboxes.
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


def combine_cost_matrices(cosine_cost_matrix, distance_matrix, similarity_weight=0.5):
    """Combine embedding cost and spatial distance into a single cost matrix.

    Args:
        cosine_cost_matrix: np.ndarray (M, N) - cosine distance values
        distance_matrix: np.ndarray (N, M) - spatial distance values (will be transposed)
        similarity_weight: float in [0, 1] - weight for cosine cost

    Returns:
        combined_matrix: np.ndarray (M, N)
    """
    combined_matrix = np.zeros((len(cosine_cost_matrix), len(distance_matrix)), dtype=np.float64)
    if combined_matrix.size == 0:
        return combined_matrix
    similarity_weight = max(0.0, min(1.0, similarity_weight))
    combined_matrix = (1 - similarity_weight) * distance_matrix.T + similarity_weight * cosine_cost_matrix
    return combined_matrix


# ==================== Gating Functions ====================

def adjust_similarity_with_gating(cos_sim, invalid):
    """Apply spatial gating to cosine similarity matrix.

    Sets invalid (gated-out) pairs to maximum cost (1.0).
    """
    adjusted_cos_sim = cos_sim.copy()
    if cos_sim.size > 0:
        adjusted_cos_sim[invalid] = 1.0
    return adjusted_cos_sim


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """Gate cost matrix using Kalman filter Mahalanobis distance.

    Entries exceeding chi-square 95% threshold are set to infinity.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """Fuse appearance cost with Kalman filter motion cost.

    Combined cost = lambda * appearance_cost + (1 - lambda) * motion_cost.
    Entries exceeding gating threshold are set to infinity.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


# ==================== Utility Functions ====================

def calculate_distance_matrix(dets, tracks):
    """Calculate squared L2 distance matrix between detection and track centers.

    Args:
        dets: list of detection dicts with 'ct' and 'tracking' keys
        tracks: list of track dicts with 'ct' key

    Returns:
        distance_matrix: np.ndarray (N, M) of squared distances
    """
    tracks_ = np.array([pre_det['ct'] for pre_det in tracks], np.float32)
    dets_ = np.array([det['ct'] + det['tracking'] for det in dets], np.float32)
    distance_matrix = (((tracks_.reshape(1, -1, 2) - dets_.reshape(-1, 1, 2)) ** 2).sum(axis=2))
    return distance_matrix


def embedding_filter(ret, embedding_history, smoothing_window):
    """Smooth embeddings over temporal window using running average.

    Note: This is used by trackerv1. The main tracker uses Track.smooth_fcn (EMA) instead.
    """
    for i, emb in enumerate(ret):
        if 'embedding' not in emb:
            continue
        tid = emb['tracking_id']
        if tid in embedding_history:
            embedding_history[tid].append(emb['embedding'])
        else:
            embedding_history[tid] = [emb['embedding']]
        if len(embedding_history[tid]) > smoothing_window:
            embedding_history[tid].pop(0)

        if len(embedding_history[tid]) > 0:
            smoothed_embedding = np.mean(embedding_history[tid], axis=0)
            embedding_history[tid][-1] = smoothed_embedding
        else:
            smoothed_embedding = emb['embedding']
        ret[i]['embedding'] = smoothed_embedding
    return ret
