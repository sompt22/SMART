import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment_sk
from numba import jit
import copy
import os
import pathlib
from tracker_fair import matching
from utils.track import Track

class Tracker:
    def __init__(self, opt):
        self.opt = opt
        self.frm_count = 0
        self.tracks = []
        self.next_track_id = 0
        self.max_age = self.opt.max_age  # Tracks are considered inactive if not updated for this many frames
        self.smoothing_window = 20
        self.tracking_task  = True if 'tracking' in self.opt.task else False
        self.embedding_task = True if 'embedding' in self.opt.task else False
        if self.opt.debug == 4: self.initialize_files()

    def initialize_files(self):
    # Construct the base directory path
        base_dir = os.path.join(pathlib.Path().resolve(), "..", "exp", self.opt.task, self.opt.exp_id, "debug")

        # Process the task(s) to determine the filename
        tasks = self.opt.task.split(",")
        if self.tracking_task and self.embedding_task:
            filename = "tracking_embedding.txt"
        elif self.tracking_task:
            filename = "tracking.txt"
        elif self.embedding_task:
            filename = "embedding.txt"
        else:
            filename = "debug.txt"
        full_path = os.path.join(base_dir, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        self.oper = open(full_path, "a")

    def init_track(self, detection):
        for det in detection:
            if det['score'] > self.opt.new_thresh:
                self.create_new_track(det)

    def reset(self):
        self.next_track_id = 0
        self.tracks = []
        self.frm_count = 0

    def step(self, detections, public_det=None):
        N = len(detections) # Number of Detections
        M = len(self.tracks) # Number of Tracks
        if self.opt.debug == 4: self.oper.write("Frame: " + str(self.frm_count) + "\n")

        # === Kalman filter prediction for all tracks ===
        for track in self.tracks:
            track.predict()

        if self.tracking_task:
            dets = np.array([det['ct'] + det['tracking'] for det in detections], np.float32) # N x 2
        else:
            dets = np.array([det['ct'] for det in detections], np.float32) # N x 2

        # Detection bbox area & category
        item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * (item['bbox'][3] - item['bbox'][1])) for item in detections], np.float32) # N
        item_cat = np.array([item['class'] for item in detections], np.int32) # N
        # Track bbox area & category (using KF-predicted positions)
        track_size = np.array([((track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])) for track in self.tracks], np.float32) # M
        track_cat = np.array([track.class_id for track in self.tracks], np.int32) # M
        tracks = np.array([pre_det.center for pre_det in self.tracks], np.float32) # M x 2

        lse_dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M
        if self.opt.debug == 4:
            self.oper.write(f"Least Square Error (LSE) Matrix: {lse_dist.shape} (Detections x Tracks)" + "\n")
            self.oper.write(str(lse_dist) + "\n")

        ## Gating for box size with adaptive threshold
        min_gate = 400.0  # minimum gate value (20px^2 equivalent)
        effective_track_size = np.maximum(track_size, min_gate)
        effective_item_size = np.maximum(item_size, min_gate)
        invalid = ((lse_dist > effective_track_size.reshape(1, M)) + \
                   (lse_dist > effective_item_size.reshape(N, 1)) + \
                   (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
        lse_dist = lse_dist + invalid * 1e18
        if self.opt.debug == 4:
            self.oper.write(f"Invalid Matrix: {invalid.shape} (Detections x Tracks)" + "\n")
            self.oper.write(str(invalid)+ "\n")
            self.oper.write("Gated Least Square Error (LSE) Matrix: \n")
            self.oper.write(str(lse_dist) + "\n")

        # === Stage 1: Embedding-based association ===
        if self.embedding_task:
            dets_emb = np.asarray([det['embedding'] for det in detections], np.float32)      # N x embedding_dim
            tracks_emb = np.asarray([track.embedding for track in self.tracks], np.float32)  # M x embedding_dim
            cos_sim = matching.embedding_distance(dets_emb, tracks_emb)
            gated_cos_sim = cos_sim + invalid * 1.0
            matched_indices, unmatched_dets, unmatched_tracks = matching.linear_assignment(gated_cos_sim, thresh=0.65)
            if self.opt.debug == 4:
                self.oper.write("Cosine Similarity Matrix: \n")
                self.oper.write(str(cos_sim) + "\n")
                self.oper.write("Gated Cosine Similarity Matrix: \n")
                self.oper.write(str(gated_cos_sim) + "\n")

            # === Stage 2: IoU-based fallback for unmatched from Stage 1 ===
            if len(unmatched_dets) > 0 and len(unmatched_tracks) > 0:
                u_detections_bboxes = np.array([detections[i]['bbox'] for i in unmatched_dets], dtype=np.float64)
                u_tracks_bboxes = np.array([self.tracks[i].bbox for i in unmatched_tracks], dtype=np.float64)
                iou_matrix = matching.ious(
                    np.ascontiguousarray(u_detections_bboxes, dtype=np.float64),
                    np.ascontiguousarray(u_tracks_bboxes, dtype=np.float64)
                )
                iou_cost = 1.0 - iou_matrix
                # Apply category gating on the sub-matrix
                for di, d_idx in enumerate(unmatched_dets):
                    for ti, t_idx in enumerate(unmatched_tracks):
                        if item_cat[d_idx] != track_cat[t_idx]:
                            iou_cost[di, ti] = 1.0

                iou_matched, iou_unmatched_dets, iou_unmatched_tracks = matching.linear_assignment(iou_cost, thresh=0.7)

                if self.opt.debug == 4:
                    self.oper.write("Stage 2 IoU Cost Matrix: \n")
                    self.oper.write(str(iou_cost) + "\n")
                    self.oper.write("Stage 2 IoU Matched: \n")
                    self.oper.write(str(iou_matched) + "\n")

                # Remap indices back to original
                iou_matched_remapped = np.array([[unmatched_dets[m[0]], unmatched_tracks[m[1]]] for m in iou_matched], dtype=int).reshape(-1, 2)
                if len(matched_indices) > 0 and len(iou_matched_remapped) > 0:
                    matched_indices = np.concatenate([matched_indices, iou_matched_remapped], axis=0)
                elif len(iou_matched_remapped) > 0:
                    matched_indices = iou_matched_remapped

                unmatched_dets = np.array([unmatched_dets[i] for i in iou_unmatched_dets])
                unmatched_tracks = np.array([unmatched_tracks[i] for i in iou_unmatched_tracks])

        else:
            if self.opt.hungarian:
                item_score = np.array([item['score'] for item in detections], np.float32) # N
                lse_dist[lse_dist > 1e18] = 1e18
                matched_indices = linear_assignment_sk(lse_dist)
            else:
                matched_indices = matching.greedy_assignment(copy.deepcopy(lse_dist))
            unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
            unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
        if self.opt.debug == 4:
            self.oper.write("Matched Indices: \n")
            self.oper.write(str(matched_indices) + "\n")
            self.oper.write("Unmatched Detections: " + str(unmatched_dets) + "\n")
            self.oper.write("Unmatched Tracks: " + str(unmatched_tracks) + "\n")

        ret = []
        for m in matched_indices:
            track = self.tracks[m[1]]
            temp_age = track.age
            new_tracking = detections[m[0]]['tracking'] if self.tracking_task else None
            new_embedding = detections[m[0]]['embedding'] if self.embedding_task else None
            # Only update embedding for high-confidence matches to avoid noisy updates
            if new_embedding is not None and detections[m[0]]['score'] < 0.3:
                new_embedding = None
            track.update(new_bbox=detections[m[0]]['bbox'],\
                        new_score=detections[m[0]]['score'],\
                        new_class=detections[m[0]]['class'],\
                        new_ct=detections[m[0]]['ct'],\
                        new_tracking=new_tracking,\
                        new_embedding=new_embedding,\
                        decrement_probation=track.is_on_probation)
            if not track.is_on_probation:
                matched_track = {'tracking_id': track.track_id, \
                                'bbox': track.bbox, \
                                'ct': track.center, \
                                'score': track.score, \
                                'class': track.class_id}
                if self.tracking_task: matched_track['tracking'] = track.tracking
                if self.embedding_task: matched_track['embedding'] = track.embedding
                ret.append(matched_track)
                if self.opt.debug == 4: self.oper.write(f"Track {track.track_id} matched! \n")
            if self.opt.debug == 4 and temp_age != track.age:
                self.oper.write(f"Track {track.track_id} age (Disappeared): {temp_age}  \n")

        for i in unmatched_dets:
            if self.opt.debug == 4: self.oper.write(f"Unmatched Detection and Score: {i},  {detections[i]['score']} \n")
            if detections[i]['score'] > self.opt.new_thresh:
                self.create_new_track(detections[i])

        # Collect tracks to remove (safe iteration - don't modify list during loop)
        tracks_to_remove = []
        for i in unmatched_tracks:
            if i >= len(self.tracks):
                continue
            track = self.tracks[i]
            if self.opt.debug == 4: self.oper.write(f"Unmatched Track: {track.track_id} \n")
            if track.age < self.max_age:
                track.increment_age()
                # KF already predicted the new position via track.predict() at the top of step()
                # No need for manual bbox/center extrapolation - KF handles motion model
                unmatched_track = {'tracking_id': track.track_id, \
                                'bbox': track.bbox, \
                                'ct': track.center, \
                                'score': track.score, \
                                'class': track.class_id}
                if self.tracking_task: unmatched_track['tracking'] = track.tracking
                if self.embedding_task: unmatched_track['embedding'] = track.embedding
                #ret.append(unmatched_track)
            else:
                if self.opt.debug == 4: self.oper.write(f"Track {track.track_id} removed. \n")
                tracks_to_remove.append(track)
        # Remove dead tracks after iteration completes
        for track in tracks_to_remove:
            self.tracks.remove(track)
        self.frm_count += 1
        return ret

    def create_new_track(self, detection):
        """Create a new track for a detection."""
        init_tracking = detection['tracking'] if 'tracking' in self.opt.task else None
        init_embedding = detection['embedding'] if 'embedding' in self.opt.task else None
        self.next_track_id += 1
        self.tracks.append(Track(track_id=self.next_track_id, \
                                 initial_bbox=detection['bbox'],\
                                 initial_score=detection['score'],\
                                 initial_class=detection['class'], \
                                 initial_ct=detection['ct'],\
                                 initial_tracking=init_tracking,\
                                 initial_embedding=init_embedding,\
                                 max_age=self.max_age,\
                                 smoothing_window=self.smoothing_window))
