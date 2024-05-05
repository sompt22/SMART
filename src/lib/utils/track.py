import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment_sk
from numba import jit
import copy
import os
import pathlib
from tracker_fair import matching

class Track(object):
    def __init__(self, track_id, initial_bbox, initial_score, initial_class, initial_ct, initial_tracking, initial_embedding=None, smoothing_window=10, max_age=30):
        self.track_id = track_id
        self.bbox = initial_bbox  # Assuming bbox is a tuple (x, y, width, height)
        self.score = initial_score
        self.class_id = initial_class
        self.center = initial_ct
        self.center_disp_history = np.array((0,0))
        self.tracking = initial_tracking  if initial_tracking is not None else None
        self.tracking_history = [] if initial_tracking is None else [initial_tracking]
        self.embedding = initial_embedding if initial_embedding is not None else None
        self.embeddings_history = [] if initial_embedding is None else [initial_embedding]
        self.age = 0
        self.active = 1
        self.is_on_probation = True
        self.probation_frames = 1  # Number of frames to wait before activating the track   
        self.max_age = max_age  # Tracks are considered inactive if not updated for this many frames  
        self.smoothing_window = smoothing_window  # Number of embeddings to consider for smoothing    
           
    
    def update(self, new_bbox=None, new_score=None, new_class=None, new_ct=None, new_tracking=None, new_embedding=None, decrement_probation=True):              
        """Update the track with new data."""
        if new_bbox is not None:
            self.bbox = new_bbox
        if new_score is not None:
            self.score = new_score
        if new_class is not None:
            self.class_id = new_class
        if new_ct is not None: 
            self.center_disp_history = self.center_distance(new_ct)           
            self.center = new_ct               
        if new_tracking is not None:
            #self.tracking = self.smooth_fcn(self.tracking_history, new_tracking)
            self.tracking = new_tracking
        if new_embedding is not None:
            self.embedding = self.smooth_fcn(self.embeddings_history, new_embedding)
        self.age = 0
        
        # Decrease probation frames on update
        if self.is_on_probation and decrement_probation:
            self.probation_frames -= 1
            if self.probation_frames <= 0:
                self.is_on_probation = False
                # Additional logic to "confirm" the track can be placed here
                #print(f"Track {self.track_id} exiting probation after confirmation.")        
        
    
    def increment_age(self):
        """Increment the track's age, potentially deactivating it if it gets too old."""
        self.age += 1
        if self.age > self.max_age:
            self.active = 0
                
    def smooth_fcn(self, history, new_value):
        """Smooth the current vector based on the history."""
        history.append(new_value)
        if len(history) > self.smoothing_window:
            history.pop(0)
        smoothed = np.mean(history, axis=0)
        history[-1] = smoothed
        return smoothed
   
    def center_distance(self, new_center):
        """Calculate the distance between the track's center and a new center."""
        return np.array(self.center) - np.array(new_center)
                   
