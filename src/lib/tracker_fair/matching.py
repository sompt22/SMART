import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#import seaborn as sns

from cython_bbox import bbox_overlaps as bbox_ious
from utils import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def calculate_iou_matrix(tracks, detections):
    """
    Calculate the Intersection over Union (IoU) value matrix between tracks and detections.

    Args:
    tracks (list): List of track bounding boxes in the format [(x1, y1, x2, y2), ...].
    detections (list): List of detection bounding boxes in the format [(x1, y1, x2, y2), ...].

    Returns:
    numpy.ndarray: IoU value matrix of shape (len(tracks), len(detections)).
    """
    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    iou_matrix_inv = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if iou_matrix.size == 0:
        return iou_matrix, iou_matrix_inv

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            track_bbox = track['bbox']
            detection_bbox = det['bbox']
            #print("track_bbox: ", track_bbox)
            #print("detection_bbox: ", detection_bbox)

            # Calculate the intersection coordinates (top-left and bottom-right)
            x1_i = max(track_bbox[0], detection_bbox[0])
            y1_i = max(track_bbox[1], detection_bbox[1])
            x2_i = min(track_bbox[2], detection_bbox[2])
            y2_i = min(track_bbox[3], detection_bbox[3])

            # Calculate the area of intersection
            intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

            # Calculate the area of each bounding box
            track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
            detection_area = (detection_bbox[2] - detection_bbox[0]) * (detection_bbox[3] - detection_bbox[1])

            # Calculate the IoU value
            iou = intersection_area / (track_area + detection_area - intersection_area)
            iou_matrix[i, j] = iou
    iou_matrix_inv = 1. - iou_matrix        

    return iou_matrix, iou_matrix_inv

def combine_cost_matrices(cosine_similarity_matrix, distance_matrix, similarity_weight=0.5):
    """
    Combine two cost matrices into a single cost matrix using a weighted sum.

    Args:
    cosine_similarity_matrix (numpy.ndarray): Matrix of cosine similarities (values between 0 and 1).
    distance_matrix (numpy.ndarray): Matrix of center point distances (in pixel numbers).
    similarity_weight (float): Weight for the cosine similarity matrix (between 0 and 1).

    Returns:
    numpy.ndarray: Combined cost matrix.
    """
    combined_matrix = np.zeros((len(cosine_similarity_matrix), len(distance_matrix)), dtype=np.float64)
    if combined_matrix.size == 0:
        return combined_matrix 
      
    # Ensure that the weights are in the valid range [0, 1]
    similarity_weight = max(0.0, min(1.0, similarity_weight))

    # Combine the matrices using a weighted sum
    combined_matrix = (1 - similarity_weight) * distance_matrix.T + similarity_weight * cosine_similarity_matrix

    return combined_matrix


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance___(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    """ *****************************!!!!!!!!!!!!!******************************** 
    tracks and detections are numpy arrays that contain embeddings
    these embeddings are normalized to unit length by F.normalize [-1,1]
    dot product of two unit vectors is the cosine similarity
    1 means identical, 0 means orthogonal, -1 means opposite
    used metric is cosine distance
    lap.lapjv solves the linear assignment problem which is a minimization problem
    """
    len_tracks = len(tracks)
    len_detections = len(detections)

    cost_matrix = np.zeros((len_tracks,len_detections), dtype=np.float64)
    cost_matrix_inv = np.zeros((len_tracks, len_detections), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix, cost_matrix_inv
    #det_features = np.asarray([track['embedding'] for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    #track_features = np.asarray([track['embedding'] for track in tracks], dtype=np.float64)
    #cost_matrix = np.maximum(0.0, cdist(tracks, detections, metric))  # Nomalized features
    
    data_is_normalized = True
    if not data_is_normalized:
        a = np.asarray(tracks) / np.linalg.norm(tracks, axis=1, keepdims=True)
        b = np.asarray(detections) / np.linalg.norm(detections, axis=1, keepdims=True)
    else:
        a = np.asarray(tracks)
        b = np.asarray(detections)    
    cost_matrix_inv = 0.5 * (1. - np.dot(a, b.T)) #!!!!!!!!!!!!!!!!!!! NORMALIZE BETWEEN 0 AND 1
    cost_matrix  = np.dot(a, b.T)

    return cost_matrix, cost_matrix_inv

def embedding_distance_(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    cost_matrix_inv = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix, cost_matrix_inv
    #det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    #track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(tracks, detections, metric))  # Nomalized features
    cost_matrix_inv = 1. - cost_matrix
    return cost_matrix, cost_matrix_inv

def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(0.0, cdist(tracks, detections, metric))  # Nomalized features
    return cost_matrix

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
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


    
def embedding_filter(ret, embedding_history, smoothing_window):
    for i, emb in enumerate(ret):
        if 'embedding' in emb:
            if emb['tracking_id'] in embedding_history:
                embedding_history[emb['tracking_id']].append(emb['embedding'])
            else:
                embedding_history[emb['tracking_id']] = [emb['embedding']]   
            if len(embedding_history[emb['tracking_id']]) > smoothing_window:
                embedding_history[emb['tracking_id']].pop(0)
            
            # Calculate the smoothed embedding as the average of embeddings in the window
            if len(embedding_history[emb['tracking_id']]) > 0:
                smoothed_embedding = np.mean(embedding_history[emb['tracking_id']], axis=0)
                embedding_history[emb['tracking_id']][-1] = smoothed_embedding
            else:
                smoothed_embedding = emb['tracking_id']  # If the buffer is empty, use the original embedding         
            ret[i]['embedding'] = smoothed_embedding
        else:
            continue
    return ret

def adjust_similarity_with_gating(cos_sim, invalid):
    # Set a low similarity score for invalid pairs
    # Assuming cos_sim ranges from -1 to 1, we use -1 to denote completely dissimilar (or invalid) pairs
    # If cos_sim ranges from 0 to 1, you might choose 0 or another appropriate low value
    adjusted_cos_sim = cos_sim.copy()  # Avoid modifying the original matrix
    if not cos_sim.size == 0:
        adjusted_cos_sim[invalid] = 1  # Penalize invalid pairs by setting their similarity to 1 (inverse for hungarian algorithm)
    return adjusted_cos_sim


def calculate_distance_matrix(dets, tracks, unmatched_dets, unmatched_tracks):
    """Calculate the distance matrix for unmatched detections and tracks."""
    tracks_ = np.array([pre_det['ct'] for pre_det in tracks], np.float32) # M x 2
    dets_ = np.array([det['ct'] + det['tracking'] for det in dets], np.float32) # N x 2
    lse_dist = (((tracks_.reshape(1, -1, 2) - dets_.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M L2 Norm = Least Square Error (LSE)
    distance_matrix = lse_dist
    
    return distance_matrix

def plot_cost_matrices(cosine_similarity_matrix, adjusted_cosine_similarity_matrix, matched, path):
        # Plotting the Cosine Similarity Matrix
        if cosine_similarity_matrix.size == 0:
            print('Cosine Similarity Matrix is empty')
            return
        if adjusted_cosine_similarity_matrix.size == 0:
            print('Adjusted Cosine Similarity Matrix is empty')
            return

        # Generate labels based on matched indices
        detection_labels = ['Detection {}'.format(i+1) for i in range(cosine_similarity_matrix.shape[0])]
        track_labels = ['Track {}'.format(i+1) for i in range(cosine_similarity_matrix.shape[1])]

        # Create a map of matched detections and tracks for labeling
        matched_detection_labels = [detection_labels[d] for d, t in matched]
        matched_track_labels = [track_labels[t] for d, t in matched]


        print('Plotting the Cosine Similarity Matrix and the Adjusted Cosine Similarity Matrix')
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        sns.heatmap(cosine_similarity_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    xticklabels=matched_track_labels, yticklabels=matched_detection_labels)
        plt.title('Cosine Similarity Matrix')

        # Plotting the Adjusted Cosine Similarity Matrix
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        sns.heatmap(adjusted_cosine_similarity_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    xticklabels=matched_track_labels, yticklabels=matched_detection_labels)
        plt.title('Adjusted Cosine Similarity Matrix')

        plt.tight_layout()
        plt.savefig(path)