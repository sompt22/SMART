import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment_sk
import copy
import os
import pathlib
from tracker_fair import matching


class Tracker(object):
    """Dictionary-based tracker with file-based debug logging.

    This is an earlier version of the tracker that stores tracks as dicts.
    For the main production tracker with Track objects and Kalman filter,
    see utils/tracker.py.
    """
    def __init__(self, opt):
        self.opt = opt
        self.reset()
        self.frm_count = 0
        self.embedding_history = dict()
        self.smoothing_window = 10
        if self.opt.debug == 4:
            self._initialize_debug_file()

    def _initialize_debug_file(self):
        base_dir = os.path.join(pathlib.Path().resolve(), "..", "exp", self.opt.task, self.opt.exp_id, "debug")
        tasks = self.opt.task.split(",")
        if "tracking" in tasks and "embedding" in tasks:
            filename = "tracking_embedding.txt"
        elif "tracking" in tasks:
            filename = "tracking.txt"
        elif "embedding" in tasks:
            filename = "embedding.txt"
        else:
            filename = "debug.txt"
        full_path = os.path.join(base_dir, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        self.oper = open(full_path, "a")

    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                item.update({'active': 1, 'age': 1, 'tracking_id': self.id_count, 'embedding': []})
                if 'ct' not in item:
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)

    def reset(self):
        self.id_count = 0
        self.tracks = []
        self.frm_count = 0
        self.embedding_history = dict() if "embedding" in self.opt.task else None
        self.smoothing_window = 10 if self.embedding_history is not None else None

    def step(self, results, public_det=None):
        self.frm_count += 1
        N = len(results)
        M = len(self.tracks)

        if 'tracking' in self.opt.task:
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
        lse_dist = lse_dist + invalid * 1e18

        if 'embedding' in self.opt.task and 'tracking' in self.opt.heads:
            # Embedding + tracking: use cosine similarity with gating
            dets_emb = np.asarray([det['embedding'] for det in results], np.float32)
            tracks_emb = np.asarray([t['embedding'] for t in self.tracks], np.float32)
            cos_sim = matching.embedding_distance(dets_emb, tracks_emb)
            adjusted_cos_sim = matching.adjust_similarity_with_gating(cos_sim, invalid)
            matched_indices, unmatched_dets, unmatched_tracks = matching.linear_assignment(adjusted_cos_sim, thresh=0.5)

            if self.opt.debug == 4:
                self.oper.write(f"Frame: {self.frm_count}\n")
                self.oper.write(f"Cosine Similarity Matrix:\n{cos_sim}\n")
                self.oper.write(f"Matched Indices:\n{matched_indices}\n")
                self.oper.write(f"Unmatched Detections: {unmatched_dets}\n")
                self.oper.write(f"Unmatched Tracks: {unmatched_tracks}\n")

        elif 'tracking' in self.opt.heads:
            # Tracking only: spatial distance matching
            if self.opt.hungarian:
                lse_dist[lse_dist > 1e18] = 1e18
                matched_sk = linear_assignment_sk(lse_dist)
                matches = []
                unmatched_dets = []
                unmatched_tracks = []
                for m in zip(*matched_sk):
                    if lse_dist[m[0], m[1]] > 1e16:
                        unmatched_dets.append(m[0])
                        unmatched_tracks.append(m[1])
                    else:
                        matches.append(m)
                matched_indices = np.array(matches).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int)
            else:
                matched_indices = matching.greedy_assignment(copy.deepcopy(lse_dist))
                unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
                unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]

            if self.opt.debug == 4:
                self.oper.write(f"Frame: {self.frm_count}\n")
                self.oper.write(f"Matched Indices:\n{matched_indices}\n")
                self.oper.write(f"Unmatched Detections: {unmatched_dets}\n")
                self.oper.write(f"Unmatched Tracks: {unmatched_tracks}\n")

        elif 'embedding' in self.opt.task:
            # Embedding only: cosine similarity matching
            dets_emb = np.asarray([det['embedding'] for det in results], np.float32)
            tracks_emb = np.asarray([t['embedding'] for t in self.tracks], np.float32)
            cos_sim = matching.embedding_distance(dets_emb, tracks_emb)
            adjusted_cos_sim = matching.adjust_similarity_with_gating(cos_sim, invalid)
            matched_indices, unmatched_dets, unmatched_tracks = matching.linear_assignment(adjusted_cos_sim, thresh=0.5)

            if self.opt.debug == 4:
                self.oper.write(f"Frame: {self.frm_count}\n")
                self.oper.write(f"Cosine Similarity Matrix:\n{cos_sim}\n")
                self.oper.write(f"Matched Indices:\n{matched_indices}\n")
                self.oper.write(f"Unmatched Detections: {unmatched_dets}\n")
                self.oper.write(f"Unmatched Tracks: {unmatched_tracks}\n")

        ret = []
        for m in matched_indices:
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

        if "embedding" in self.opt.heads:
            ret = matching.embedding_filter(ret, self.embedding_history, self.smoothing_window)
        self.tracks = ret
        return ret
