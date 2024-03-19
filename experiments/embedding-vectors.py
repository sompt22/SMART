import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2
import random

"""
 from sklearn.metrics.pairwise import cosine_similarity
 X = [[0, 0, 0], [1, 1, 1]]
 Y = [[1, 0, 0], [1, 1, 0]]
 cosine_similarity(X, Y)
array([[0.     , 0.     ],
       [0.57..., 0.81...]])
"""


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_patches_and_embeddings(annotations, img_dir, output_dir, images_dict, num_tracks=50, max_frames=30):
    track_ids = list(set([ann['track_id'] for ann in annotations]))
    selected_track_ids = random.sample(track_ids, min(num_tracks, len(track_ids)))
    
    track_embeddings = {track_id: [] for track_id in selected_track_ids}
    for ann in annotations:
        if ann['track_id'] in selected_track_ids:
            img_id = ann['image_id']
            if img_id in images_dict:
                img_path = os.path.join(img_dir, images_dict[img_id])
                img = cv2.imread(img_path)
                # Assuming 'bbox' is correctly structured as [x, y, width, height]
                bbox = ann['bbox']
                if bbox[2] <= 5 or bbox[3] <= 5:
                    continue
                if ann['embedding'] is None:
                    continue
                patch = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                track_folder = os.path.join(output_dir, str(ann['track_id']))
                os.makedirs(track_folder, exist_ok=True)
                patch_file_name = os.path.join(track_folder, f"frame_{img_id}.jpg")
                #print('Min and Max value of embedding:', np.min(ann['embedding']), np.max(ann['embedding']))
                cv2.imwrite(patch_file_name, patch)
                track_embeddings[ann['track_id']].append(ann['embedding'])
                #if len(track_embeddings[ann['track_id']]) >= max_frames:
                    #break  # Limit to max_frames per track
    return track_embeddings
def calculate_similarities(track_embeddings):
    intra_similarities = {}
    for track_id, embeddings in track_embeddings.items():
        # Ensure embeddings are a 2D array
        embeddings = np.array(embeddings)
        if embeddings.ndim == 1 and embeddings.size > 0:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.size == 0:  # Skip tracks with no embeddings
            continue
        # Calculate intra-track similarities
        print('Shape of embeddings 1:', embeddings.shape) #Shape of embeddings 1: (1800, 128)
        similarities = cosine_similarity(embeddings)
        intra_similarities[track_id] = similarities[np.triu_indices(len(embeddings), k=1)]

    inter_similarities = []
    for track_id_i in track_embeddings:
        for track_id_j in track_embeddings:
            if track_id_i < track_id_j:
                embeddings_i = np.array(track_embeddings[track_id_i])
                embeddings_j = np.array(track_embeddings[track_id_j])
                if embeddings_i.ndim == 1 and embeddings_i.size > 0:
                    embeddings_i = embeddings_i.reshape(1, -1)
                if embeddings_j.ndim == 1 and embeddings_j.size > 0:
                    embeddings_j = embeddings_j.reshape(1, -1)
                # Skip similarity calculation if either set of embeddings is empty
                if embeddings_i.size == 0 or embeddings_j.size == 0:
                    continue
                print('Shape of embeddings 2:', embeddings_i.shape, embeddings_j.shape) # Shape of embeddings 2: (1800, 128) (722, 128)
                similarities = cosine_similarity(embeddings_i, embeddings_j)
                inter_similarities.extend(similarities.flatten())
    return intra_similarities, inter_similarities

def plot_similarity_matrices(track_embeddings, output_dir):
    for track_id, embeddings in track_embeddings.items():
        # Ensure embeddings are a 2D array
        embeddings = np.array(embeddings)
        if embeddings.ndim == 1 and embeddings.size > 0:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.size == 0:  # Skip tracks with no embeddings
            print(f"Skipping track ID {track_id} due to no embeddings.")
            continue
        print('Shape of embeddings 3:', embeddings.shape) # Shape of embeddings 3: (1800, 128)
        similarity_matrix = cosine_similarity(embeddings)
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Inter-Similarity Matrix for Track ID {track_id}")
        plt.savefig(os.path.join(output_dir, f"inter_similarity_matrix_{track_id}.png"))
        plt.close()


def plot_similarity_histograms(intra_similarities, inter_similarities, output_dir):
    all_intra_similarities = np.concatenate(list(intra_similarities.values()))
    all_inter_similarities = np.array(inter_similarities)
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(all_intra_similarities, bins=20, alpha=0.7, label='Intra-track')
    plt.title('Intra-track Similarities')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(all_inter_similarities, bins=20, alpha=0.7, color='orange', label='Inter-track')
    plt.title('Inter-track Similarities')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_histograms.png"))
    plt.close()

# Path to your 'train.json' file and images directory
file_path = '/home/fatih/phd/SMART/data/sompt22/annotations/train.json'
img_dir = '/home/fatih/phd/SMART/data/sompt22/images/train'
output_dir = './exp/vector/exp2'

data = load_json_data(file_path)
annotations = data['annotations']
images_dict = {image['id']: image['file_name'] for image in data['images']}
track_embeddings = extract_patches_and_embeddings(annotations, img_dir, output_dir,images_dict)
# Assume calculate_similarities function is defined as before
intra_similarities, inter_similarities = calculate_similarities(track_embeddings)
plot_similarity_matrices(track_embeddings, output_dir)
plot_similarity_histograms(intra_similarities, inter_similarities, output_dir)
