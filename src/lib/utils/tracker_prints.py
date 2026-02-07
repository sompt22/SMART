import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment_sk
import copy

from tracker_fair import matching


class Tracker(object):
    """Experimental tracker variant with configurable association method selection.

    This is a dictionary-based tracker (tracks are dicts, not Track objects).
    For the main production tracker, see utils/tracker.py which uses Track objects
    with Kalman filter integration.

    Association methods:
        'dist': Greedy spatial distance matching
        'embedding': Cosine embedding distance with Hungarian assignment
        'fusion': Weighted combination of embedding + spatial distance
    """
    def __init__(self, opt):
        self.opt = opt
        self.reset()
        self.frm_count = 0
        self.embedding_history = dict()
        self.smoothing_window = 10
        self.ass_method = 'dist'  # Configurable: 'dist', 'embedding', 'fusion'

    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                item['embedding'] = []
                if 'ct' not in item:
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)

    def reset(self):
        self.id_count = 0
        self.tracks = []
        self.frm_count = 0

    def step(self, results, public_det=None):
        self.frm_count += 1
        N = len(results)
        M = len(self.tracks)

        if "tracking" in self.opt.heads and 'tracking' in self.opt.task:
            dets = np.array([det['ct'] + det['tracking'] for det in results], np.float32)
        else:
            dets = np.array([det['ct'] for det in results], np.float32)

        track_size = np.array([((t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1])) for t in self.tracks], np.float32)
        track_cat = np.array([t['class'] for t in self.tracks], np.int32)
        item_size = np.array([((it['bbox'][2] - it['bbox'][0]) * (it['bbox'][3] - it['bbox'][1])) for it in results], np.float32)
        item_cat = np.array([it['class'] for it in results], np.int32)
        tracks = np.array([t['ct'] for t in self.tracks], np.float32)

        lse_dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))
        invalid = ((lse_dist > track_size.reshape(1, M)) + (lse_dist > item_size.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
        lse_dist_penalty = lse_dist + invalid * 1e18

        # Distance-based matching (always computed as baseline)
        matched_indices_dist = matching.greedy_assignment(copy.deepcopy(lse_dist_penalty))
        unmatched_dets_dist = [d for d in range(dets.shape[0]) if not (d in matched_indices_dist[:, 0])]
        unmatched_tracks_dist = [d for d in range(tracks.shape[0]) if not (d in matched_indices_dist[:, 1])]

        # Embedding-based matching
        matched_indices_emb, unmatched_dets_emb, unmatched_tracks_emb = np.empty((0, 2), dtype=int), [], []
        cos_sim = None
        if 'embedding' in self.opt.heads and 'embedding' in self.opt.task:
            dets_emb = np.asarray([det['embedding'] for det in results], np.float32)
            tracks_emb = np.asarray([t['embedding'] for t in self.tracks], np.float32)
            cos_sim = matching.embedding_distance(tracks_emb, dets_emb)
            gated_cos_sim = matching.adjust_similarity_with_gating(cos_sim, invalid.T)
            matched_indices_emb, unmatched_tracks_emb, unmatched_dets_emb = matching.linear_assignment(gated_cos_sim, thresh=0.5)

        # Fusion matching
        matched_indices_fus, unmatched_dets_fus, unmatched_tracks_fus = np.empty((0, 2), dtype=int), [], []
        if cos_sim is not None and lse_dist.size > 0:
            max_lse = np.max(lse_dist)
            min_lse = np.min(lse_dist)
            denom = max_lse - min_lse if max_lse != min_lse else 1.0
            lse_normalized = (lse_dist - min_lse) / denom
            lse_normalized_penalty = lse_normalized + invalid * 100
            matrix_fusion = matching.combine_cost_matrices(cos_sim, lse_normalized_penalty, similarity_weight=1)
            matched_indices_fus, unmatched_tracks_fus, unmatched_dets_fus = matching.linear_assignment(matrix_fusion, thresh=0.3)

        # Select association method
        if self.ass_method == 'embedding' and len(matched_indices_emb) > 0:
            matched_indices = matched_indices_emb
            unmatched_dets = unmatched_dets_emb
            unmatched_tracks = unmatched_tracks_emb
        elif self.ass_method == 'fusion' and len(matched_indices_fus) > 0:
            matched_indices = matched_indices_fus
            unmatched_dets = unmatched_dets_fus
            unmatched_tracks = unmatched_tracks_fus
        else:
            matched_indices = matched_indices_dist
            unmatched_dets = unmatched_dets_dist
            unmatched_tracks = unmatched_tracks_dist

        ret = []
        for m in matched_indices:
            if m[0] < len(results) and m[1] < len(self.tracks):
                track = results[m[0]]
                track['tracking_id'] = self.tracks[m[1]]['tracking_id']
                track['age'] = 1
                track['active'] = self.tracks[m[1]]['active'] + 1
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            if track['score'] > self.opt.new_thresh:
                self.id_count += 1
                track['tracking_id'] = self.id_count
                track['age'] = 1
                track['active'] = 1
                ret.append(track)

        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.opt.max_age:
                track['age'] += 1
                track['active'] = 0
                bbox = track['bbox']
                ct = track['ct']
                v = [0, 0]
                track['bbox'] = [bbox[0] + v[0], bbox[1] + v[1], bbox[2] + v[0], bbox[3] + v[1]]
                track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                ret.append(track)

        ret = matching.embedding_filter(ret, self.embedding_history, self.smoothing_window)
        self.tracks = ret
        return ret
