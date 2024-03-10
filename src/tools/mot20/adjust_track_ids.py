import os
import cv2
import uuid
from tqdm import tqdm
import shutil

def find_min_max_id(annotation_path):    
    # Read in annotations
    min_track_id = float("inf")
    max_track_id = -1
             
    label_names = os.listdir(annotation_path)
    for label_name in tqdm(label_names, desc="Processing labels in sequence"):   
        label_path = os.path.join(annotation_path, label_name)
    
        with open(label_path, "r") as f:
            annotations = f.readlines()
      
        for annotation in annotations:
            _, track_id, _, _, _, _ = annotation.split()
            track_id = int(track_id)
            if track_id == -1:
                continue
            elif track_id < min_track_id:
                min_track_id = track_id
                
            if track_id > max_track_id:
                max_track_id = track_id                 
                                                             
    return min_track_id, max_track_id           
                
def adjust_track_ids(annotation_path, min_id):
    output_label_path = os.path.join(annotation_path, "../reduced_id")
    os.makedirs(output_label_path, exist_ok=True)
    # Subtract minimum track id from all track ids
    label_names = os.listdir(annotation_path)
    for label_name in tqdm(label_names, desc="Processing labels in sequence"):   
        label_path = os.path.join(annotation_path, label_name)
    
        with open(label_path, "r") as f:
            annotations = f.readlines()    
    
        new_annotations = []
        for annotation in annotations:
            class_id, track_id, xcenter, ycenter, bbox_width, bbox_height = annotation.split()
            track_id = int(track_id)
            if track_id == -1:
                new_track_id = -1
            else:    
                new_track_id = track_id - min_id + 1
            new_annotations.append(f"{class_id} {new_track_id} {xcenter} {ycenter} {bbox_width} {bbox_height}\n")

        # Overwrite original annotation file with updated annotations
        with open(os.path.join(output_label_path, label_name), "w") as f:
            f.writelines(new_annotations)


label_path = "/home/fatih/mnt/mscoco/merged_MOT17/labels/train"

min_id, max_id = find_min_max_id(label_path)

print(f"{min_id} min id, {max_id} max id")

adjust_track_ids(label_path,min_id)
