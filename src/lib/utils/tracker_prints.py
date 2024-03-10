import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment_sk
from numba import jit
import copy

from tracker_fair import matching

class Tracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.reset()
    self.frm_count = 0
    self.embedding_history = dict()
    self.smoothing_window = 10

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
    if "tracking" in self.opt.heads and 'tracking' in self.opt.task:
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
    # Detection and Track center distances matrix
    #print("tracks: ")
    #print(tracks.reshape(1, -1, 2))
    #print("dets: ")
    #print(dets.reshape(-1, 1, 2))
    lse_dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M L2 Norm = Least Square Error (LSE)
    #lse_dist = np.sqrt(((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))

    lse_dist_normalized = lse_dist
    lse_dist_normalized_penalty = lse_dist
    standardized_matrix = lse_dist
    ## Gating for box size
    invalid = ((lse_dist > track_size.reshape(1, M)) + (lse_dist > item_size.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    """
    print("invalid: ")
    print(invalid)
    print("lse_dist: ")
    print(lse_dist)    
    """
    lse_dist_penalty = lse_dist + invalid * 1e18
    matched_indices_dist_raw = matching.greedy_assignment(copy.deepcopy(lse_dist_penalty))
    unmatched_dets_dist_raw = [d for d in range(dets.shape[0]) if not (d in matched_indices_dist_raw[:, 0])]
    unmatched_tracks_dist_raw = [d for d in range(tracks.shape[0]) if not (d in matched_indices_dist_raw[:, 1])]     
    print("lse_dist_penaltyAAAAAAAAAAAAAAA: ")
    print(lse_dist_penalty)     
    print("matched_indices_dist_rawBBBBBBBBBBBBBBBB: ")
    print(matched_indices_dist_raw)
    """
    print("lse_dist_penaltyAAAAAAAAAAAAAAA: ")
    print(lse_dist_penalty) 
    
    matched_indices_dist_raw = matching.greedy_assignment(copy.deepcopy(lse_dist_penalty))
    unmatched_dets_dist_raw = [d for d in range(dets.shape[0]) if not (d in matched_indices_dist_raw[:, 0])]
    unmatched_tracks_dist_raw = [d for d in range(tracks.shape[0]) if not (d in matched_indices_dist_raw[:, 1])] 

    print("matched_indices_dist_rawBBBBBBBBBBBBBBBB: ")
    print(matched_indices_dist_raw)
    """

    if lse_dist.size > 0:
      min_lse_distance = np.min(lse_dist)
      max_lse_distance = np.max(lse_dist)
      lse_dist_normalized = (lse_dist - min_lse_distance) / (max_lse_distance - min_lse_distance) 
      standardized_matrix = (lse_dist - np.mean(lse_dist)) / np.std(lse_dist)
       
      #lse_dist_normalized = lse_dist / max_lse_distance
      #print("lse_dist_normalized: ")
      #print(lse_dist_normalized)        
      lse_dist_normalized_penalty = lse_dist_normalized + invalid * 100
      standardized_matrix_penalty = standardized_matrix + invalid * 100
      matched_indices_dist_penalty, unmatched_tracks_dist_penalty, unmatched_dets_dist_penalty = matching.linear_assignment(standardized_matrix_penalty, thresh=0.6)
      print("standardized_matrix_penaltyCCCCCCCCCCCCCC: ")
      print(standardized_matrix_penalty)
      print("matched_indices_dist_penaltyDDDDDDDDDDDD: ")
      print(matched_indices_dist_penalty)
      """
      matched_indices_dist_penalty, unmatched_tracks_dist_penalty, unmatched_dets_dist_penalty = matching.linear_assignment(lse_dist_normalized_penalty, thresh=0.6)
      print("lse_dist_normalized_penaltyCCCCCCCCCCCCCC: ")
      print(lse_dist_normalized_penalty)
      print("matched_indices_dist_penaltyDDDDDDDDDDDD: ")
      print(matched_indices_dist_penalty)    
      """  
    unmatched_dets = []
    unmatched_tracks = []
    matched_indices = []
  
    #print("Frame: ", self.frm_count)
    if 'embedding' in self.opt.heads and 'embedding' in self.opt.task:
      print("embedding!!!!!!!!!!")
      dets_emb = np.asarray([det['embedding'] for det in results], np.float32) # N x embedding_dim   
      tracks_emb = np.asarray([pre_det['embedding'] for pre_det in self.tracks], np.float32) # M x embedding_dim
      cos_sim, cos_sim_inv = matching.embedding_distance(tracks_emb, dets_emb) # cosine similarity of Detections & Tracks
      #cos_sim_inv_penalty = cos_sim_inv + invalid.T * 10
      matched_indices_cos, unmatched_tracks_cos, unmatched_dets_cos = matching.linear_assignment(cos_sim_inv, thresh=0.5)   
    """  
    print("cos_sim_inv: ") 
    print(cos_sim_inv)           
    print("matched_indices_cos: ")
    print(matched_indices_cos)
    """
        
    #if 'tracking' in self.opt.heads and 'tracking' in self.opt.task:              
    matrix_fusion = matching.combine_cost_matrices(cos_sim_inv, lse_dist_normalized_penalty, similarity_weight=1)
    matched_indices_fus, unmatched_tracks_fus, unmatched_dets_fus = matching.linear_assignment(matrix_fusion, thresh=0.3)
    
    print("matrix_fusion: ")
    print(matrix_fusion)
    print("matched_indices_fus: ")
    print(matched_indices_fus)
    
    """
    matched_indices = matched_indices_dist_raw
    unmatched_dets = unmatched_dets_dist_raw
    unmatched_tracks = unmatched_tracks_dist_raw
     
    matched_indices = matched_indices_fus
    unmatched_dets = unmatched_dets_fus
    unmatched_tracks = unmatched_tracks_fus    
    """    
    
    
    
    
    ass_method = 'dist'
    #A switch case that selects association technique
    match ass_method:
      case 'dist':
          matched_indices = matched_indices_dist_raw
          unmatched_dets = unmatched_dets_dist_raw
          unmatched_tracks = unmatched_tracks_dist_raw
      case 'embedding':
          matched_indices = matched_indices_cos  
          unmatched_dets = unmatched_dets_cos  
          unmatched_tracks = unmatched_tracks_cos 
      case 'fusion':
          matched_indices = matched_indices_fus
          unmatched_dets = unmatched_dets_fus
          unmatched_tracks = unmatched_tracks_fus
      case 'aa':
          matched_indices = matched_indices_dist_penalty
          unmatched_dets = unmatched_dets_dist_penalty
          unmatched_tracks = unmatched_tracks_dist_penalty        
    
            
    """ IOU Matching
    iou_dist,iou_dist_inv = matching.calculate_iou_matrix(self.tracks, results)     
    print("iou_dist: ")
    print(iou_dist)
    print("iou_dist_inv: ")
    print(iou_dist_inv)
    matched_indices, unmatched_tracks, unmatched_dets = matching.linear_assignment(lse_dist_normalized, thresh=0.6) 
    print("-------------------------------------------------------------")      
    """
    
    """
    if matched_indices == []:
      matched_indices = matched_indices_
    else:
      matched_indices = np.concatenate((matched_indices,matched_indices_),axis=0)
    unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
    """

    """
    dists = matching.iou_distance(unmatched_tracks, unmatched_dets)
    _matches, unmatched_tracks, unmatched_dets = matching.linear_assignment(dists, thresh=0.5)

    matches = np.append(matches,_matches)
    """

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
    """
    for i, emb in enumerate(ret):
      if 'embedding' in emb:
        if emb['tracking_id'] in self.embedding_history:
          self.embedding_history[emb['tracking_id']].append(emb['embedding'])
        else:
          self.embedding_history[emb['tracking_id']] = [emb['embedding']]   
        if len(self.embedding_history[emb['tracking_id']]) > self.smoothing_window:
          self.embedding_history[emb['tracking_id']].pop(0)
          
        # Calculate the smoothed embedding as the average of embeddings in the window
        if len(self.embedding_history[emb['tracking_id']]) > 0:
            smoothed_embedding = np.mean(self.embedding_history[emb['tracking_id']], axis=0)
            self.embedding_history[emb['tracking_id']][-1] = smoothed_embedding
        else:
            smoothed_embedding = emb['tracking_id']  # If the buffer is empty, use the original embedding         
        ret[i]['embedding'] = smoothed_embedding     
      """  
    ret = matching.embedding_filter(ret, self.embedding_history, self.smoothing_window)           
    self.tracks = ret
    return ret
  