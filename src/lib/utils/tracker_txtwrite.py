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
    lse_dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M L2 Norm = Least Square Error (LSE)
    unmatched_dets = []
    unmatched_tracks = []
    matched_indices = []
    if 'embedding' in self.opt.heads and 'embedding' in self.opt.task:
      dets_emb = np.asarray([det['embedding'] for det in results], np.float32) # N x embedding_dim   
      tracks_emb = np.asarray([pre_det['embedding'] for pre_det in self.tracks], np.float32) # M x embedding_dim
      cos_sim, cos_sim_inv = matching.embedding_distance(tracks_emb, dets_emb) # cosine similarity of Detections & Tracks
      matched_indices, unmatched_tracks, unmatched_dets = matching.linear_assignment(cos_sim_inv, thresh=0.7)

      
      """
      with open('/home/fatih/phd/FairCenterMOT/results/embeddings.txt', 'a') as f:
        if self.frm_count > 50:
          f.write("Number of detections: " + str(N) + "\n")
          f.write("Number of tracks: " + str(M) + "\n")      
          f.write(f'Frame {str(self.frm_count)} cos_sim: \n' )
          f.write(str(cos_sim) +'\n')
          f.write(f'Frame {str(self.frm_count)} matched_indices: \n')
          f.write(str(matched_indices) +'\n')
          f.write(f'Frame {str(self.frm_count)} unmatched_tracks: ' + str(unmatched_tracks) +'\n')
          f.write(f'Frame {str(self.frm_count)} unmatched_dets: ' + str(unmatched_dets) +'\n') 
          f.write(f'Frame {str(self.frm_count)} track_score: ' + str(track_score) +'\n')
          f.write(f'Frame {str(self.frm_count)} track_age: ' + str(track_age) +'\n')
          f.write(f'Frame {str(self.frm_count)} track_active: ' + str(track_active) +'\n')
          #f.write(f'Frame {str(self.frm_count)} dets_emb:  \n')
          #f.write(str(dets_emb) +'\n')
          f.write(f'Frame {str(self.frm_count)} tracks_emb: \n')
          f.write(str(tracks_emb) +'\n')
          f.write("-------------------------------------------------------------------------------------------------------------------\n")             
        #if self.frm_count == 60:
          #exit()
      #print(matched_indices, unmatched_dets,unmatched_tracks)
      #print('emre')
      """ 
      
      """
      matched_indices = linear_assignment_sk(cos_sim) # LAP optimization of cosine similarities
      matched_indices = np.asarray(matched_indices)
      matched_indices = np.transpose(matched_indices)
      unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
      unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
      """
    """
    if 'tracking' in self.opt.heads and 'tracking' in self.opt.task:
      invalid = ((lse_dist > track_size.reshape(1, M)) + (lse_dist > item_size.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
      lse_dist = lse_dist + invalid * 1e18
      if 'embedding' in self.opt.heads and 'embedding' in self.opt.task:   
        for tr in matched_indices:
          matched_matrix = np.zeros((N,M))
          matched_matrix[:,tr[1]] = 1e18
          #matched_matrix[tr[0],:] = 1e18
          lse_dist = lse_dist + matched_matrix  
      matched_indices_ = greedy_assignment(copy.deepcopy(lse_dist))
      if matched_indices == []:
        matched_indices = matched_indices_
      else:
        matched_indices = np.concatenate((matched_indices,matched_indices_),axis=0)
      unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
      unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]


    
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
        #track['embedding'] = self.tracks[m[1]]['embedding']
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
    #print(unmatched_dets)
    #print('fatih')      
    #print(self.frm_count)      
    #print(unmatched_tracks) 
    for i in unmatched_tracks:
      #print(i)
      #print(self.tracks)
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
    self.tracks = ret
    return ret

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

def greedy_assignment_separate(dist, unmatched_det, unmatched_track, matched_indices):
  if len(unmatched_det) == 0 or len(unmatched_track) == 0:
    return matched_indices
  for i in unmatched_det:
    for j in unmatched_track:
      if dist[i][j] < 1e16:
        dist[:, j] = 1e18
        matched_indices = np.append(matched_indices,[i,j])
  matched_indices = np.array(matched_indices, np.int32).reshape(-1, 2)
  return matched_indices
