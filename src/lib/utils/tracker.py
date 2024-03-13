import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment_sk
from numba import jit
import copy
import os
import pathlib
from tracker_fair import matching

class Tracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.reset()
    self.frm_count = 0
    if "embedding" in self.opt.task and "tracking" in self.opt.task:
      self.embedding_history = dict()
      self.smoothing_window = 10
      if self.opt.debug == 4:
        self.tracking_embedding_match_matrix = os.path.join(pathlib.Path().resolve(), "..", "exp", self.opt.task, self.opt.exp_id, "debug", "tracking_embedding.txt")
        dir_path = os.path.dirname(self.tracking_embedding_match_matrix)
        if not os.path.exists(dir_path):
          os.makedirs(dir_path)
        self.tracking_embedding_matrix = open(self.tracking_embedding_match_matrix, "a")  
    elif "embedding" in self.opt.task:
      self.embedding_history = dict()
      self.smoothing_window = 10
      if self.opt.debug == 4:
        self.embedding_match_matrix = os.path.join(pathlib.Path().resolve(),"..","exp", self.opt.task, self.opt.exp_id, "debug", "embedding.txt")
        dir_path = os.path.dirname(self.embedding_match_matrix)
        if not os.path.exists(dir_path):
          os.makedirs(dir_path)
        self.embedding_matrix = open(self.embedding_match_matrix, "a")  
    elif "tracking" in self.opt.task:
      if self.opt.debug == 4:
        self.tracking_match_matrix = os.path.join(pathlib.Path().resolve(),"..","exp", self.opt.task, self.opt.exp_id, "debug", "tracking.txt")
        dir_path = os.path.dirname(self.tracking_match_matrix)
        if not os.path.exists(dir_path):
          os.makedirs(dir_path)
        self.tracking_matrix = open(self.tracking_match_matrix, "a")
           
  def init_track(self, results):
    for item in results:
      if item['score'] > self.opt.new_thresh:
        self.id_count += 1
        # active and age are never used in the paper
        item['active'] = 1
        item['age'] = 1
        item['tracking_id'] = self.id_count
        item['embedding'] = []
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.tracks.append(item)

  def reset(self):
    self.id_count = 0
    self.tracks = []
    self.frm_count = 0

  def step(self, results, public_det=None):
    self.frm_count += 1 
    N = len(results) # Number of Detections
    M = len(self.tracks) # Number of Tracks
    if 'tracking' in self.opt.task:
      dets = np.array([det['ct'] + det['tracking'] for det in results], np.float32) # N x 2
    else:
      dets = np.array([det['ct'] for det in results], np.float32) # N x 2 

    # Track bbox area & category
    track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * (track['bbox'][3] - track['bbox'][1])) for track in self.tracks], np.float32) # M
    track_cat = np.array([track['class'] for track in self.tracks], np.int32) # M
    # Detection bbox area & category
    item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * (item['bbox'][3] - item['bbox'][1])) for item in results], np.float32) # N
    item_cat = np.array([item['class'] for item in results], np.int32) # N
    tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    lse_dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M L2 Norm = Least Square Error (LSE)

    ## Gating for box size
    invalid = ((lse_dist > track_size.reshape(1, M)) + (lse_dist > item_size.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    lse_dist = lse_dist + invalid * 1e18
    
    if 'embedding' in self.opt.task and 'tracking' in self.opt.heads:
      print("embedding and tracking")
      # Step 1: Calculate Cosine Similarity Matrix
      dets_emb = np.asarray([det['embedding'] for det in results], np.float32)            # N x embedding_dim
      tracks_emb = np.asarray([track['embedding'] for track in self.tracks], np.float32)  # M x embedding_dim
      cos_sim, cos_sim_inv = matching.embedding_distance(tracks_emb, dets_emb)
      invalid_tr = np.transpose(invalid)
      if self.opt.debug == 4:
        self.tracking_embedding_matrix.write("Frame: " + str(self.frm_count) + "\n")
        self.tracking_embedding_matrix.write("Cosine Similarity Matrix: \n")
        self.tracking_embedding_matrix.write(str(cos_sim_inv) + "\n")
        self.tracking_embedding_matrix.write("Invalid Transpose: \n")
        self.tracking_embedding_matrix.write(str(invalid_tr) + "\n")
      
      # Step 2: Apply Gating
      # Assuming `invalid` is your gating matrix based on spatial constraints
      # Adjust the cosine similarity matrix with gating information (invalid pairs set to a low value)
      adjusted_cos_sim = matching.adjust_similarity_with_gating(cos_sim_inv, invalid_tr)  # Use the function from previous discussions
      if self.opt.debug == 4:
        self.tracking_embedding_matrix.write("Adjusted Cosine Similarity Matrix: \n")
        self.tracking_embedding_matrix.write(str(adjusted_cos_sim) + "\n")
      # Step 3: Perform Initial Matching
      matched_indices, unmatched_tracks, unmatched_dets = matching.linear_assignment(adjusted_cos_sim, thresh=0.5)
      if self.opt.debug == 4:
        self.tracking_embedding_matrix.write("Matched Indices: \n")
        self.tracking_embedding_matrix.write(str(matched_indices) + "\n")
        self.tracking_embedding_matrix.write("Unmatched Detections: " + str(unmatched_dets) + "\n")
        self.tracking_embedding_matrix.write("Unmatched Tracks: " + str(unmatched_tracks) + "\n")
        path = os.path.join(pathlib.Path().resolve(), "..", "exp", self.opt.task, self.opt.exp_id, "debug")  
        #matching.plot_cost_matrices(cos_sim_inv, adjusted_cos_sim, matched_indices ,path +f'/{self.frm_count}cosine_similarity_matrices.png')        

      # Step 4: Apply Greedy Matching for Unmatched Detections and Tracks
      # Only consider unmatched detections and tracks for greedy matching
      if len(unmatched_dets) > 0 and len(unmatched_tracks) > 0:
          additional_lse_dist = matching.calculate_distance_matrix(results, self.tracks, unmatched_dets, unmatched_tracks)
          additional_matched_indices = matching.greedy_assignment(copy.deepcopy(additional_lse_dist))
          matched_indices = np.concatenate([matched_indices, additional_matched_indices], axis=0)
          unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
          unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
          if self.opt.debug == 4:    
            self.tracking_embedding_matrix.write("Additional Matched Indices: \n")
            self.tracking_embedding_matrix.write(str(additional_matched_indices) + "\n")
            self.tracking_embedding_matrix.write("Additional Unmatched Detections: " + str(unmatched_dets) + "\n")
            self.tracking_embedding_matrix.write("Additional Unmatched Tracks: " + str(unmatched_tracks) + "\n")
                    
    elif 'tracking' in self.opt.heads:
      print("tracking")
      if self.opt.hungarian:
        item_score = np.array([item['score'] for item in results], np.float32) # N
        lse_dist[lse_dist > 1e18] = 1e18
        matched_indices = linear_assignment_sk(lse_dist)
        matches = []
        for m in matched_indices:
          if dist[m[0], m[1]] > 1e16:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
          else:
            matches.append(m)
        matches = np.array(matches).reshape(-1, 2)	  
        matched_indices = matches	  
      else:
        matched_indices = matching.greedy_assignment(copy.deepcopy(lse_dist))
        unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]

      if self.opt.debug == 4:        
        self.tracking_matrix.write("Frame: " + str(self.frm_count) + "\n")
        self.tracking_matrix.write("Invalid: \n")
        self.tracking_matrix.write(str(invalid) + "\n")
        self.tracking_matrix.write("Matched Indices: \n")
        self.tracking_matrix.write(str(matched_indices) + "\n")
        self.tracking_matrix.write("Unmatched Detections: " + str(unmatched_dets) + "\n")
        self.tracking_matrix.write("Unmatched Tracks: " + str(unmatched_tracks) + "\n")
    elif 'embedding' in self.opt.task:
      print("embedding")
      dets_emb = np.asarray([det['embedding'] for det in results], np.float32)                # N x embedding_dim   
      tracks_emb = np.asarray([pre_det['embedding'] for pre_det in self.tracks], np.float32)  # M x embedding_dim
      cos_sim, cos_sim_inv = matching.embedding_distance(tracks_emb, dets_emb)                # cosine similarity of Detections & Tracks (0 match, 1 mismatch)
      invalid_tr = np.transpose(invalid) 
      adjusted_cos_sim = matching.adjust_similarity_with_gating(cos_sim_inv,invalid_tr)       # Adjust similarity with gating       
      matched_indices, unmatched_tracks, unmatched_dets = matching.linear_assignment(adjusted_cos_sim, thresh=0.5)           
      if self.opt.debug == 4:      
        self.embedding_matrix.write("Frame: " + str(self.frm_count) + "\n")
        self.embedding_matrix.write("Invalid Transpose: \n")   
        self.embedding_matrix.write(str(invalid_tr) + "\n")
        self.embedding_matrix.write("Cosine Similarity Matrix: \n")
        self.embedding_matrix.write(str(cos_sim_inv) + "\n")    
        self.embedding_matrix.write("Adjusted Cosine Similarity Matrix: \n")
        self.embedding_matrix.write(str(adjusted_cos_sim) + "\n")    
        self.embedding_matrix.write("Matched Indices: \n")
        self.embedding_matrix.write(str(matched_indices) + "\n")
        self.embedding_matrix.write("Unmatched Detections: " + str(unmatched_dets) + "\n")
        self.embedding_matrix.write("Unmatched Tracks: " + str(unmatched_tracks) + "\n")
        path = os.path.join(pathlib.Path().resolve(), "..", "exp", self.opt.task, self.opt.exp_id, "debug")  
        matching.plot_cost_matrices(cos_sim_inv, adjusted_cos_sim, matched_indices ,path +f'/{self.frm_count}cosine_similarity_matrices.png')
             
    ret = []
    for m in matched_indices:
      if m[0] < len(results) and m[1] < len(self.tracks):
        track = results[m[0]]
        track['tracking_id'] = self.tracks[m[1]]['tracking_id']
        track['age'] = 1
        track['active'] = self.tracks[m[1]]['active'] + 1
        ret.append(track)

    # Private detection: create tracks for all un-matched detections
    for i in unmatched_dets:
      track = results[i]
      if track['score'] > self.opt.new_thresh:
        self.id_count += 1
        track['tracking_id'] = self.id_count
        track['age'] = 1
        track['active'] =  1
        ret.append(track)
        
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.opt.max_age:
        track['age'] += 1
        track['active'] = 0
        bbox = track['bbox']
        ct = track['ct']
        v = [0, 0]
        track['bbox'] = [
          bbox[0] + v[0], bbox[1] + v[1],
          bbox[2] + v[0], bbox[3] + v[1]]
        track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
        ret.append(track)
    if "embedding" in self.opt.heads:
      ret= matching.embedding_filter(ret, self.embedding_history, self.smoothing_window)          
    self.tracks = ret
    return ret
  
