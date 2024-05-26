import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2
import random

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
                bbox = ann['bbox']
                if bbox[2] <= 5 or bbox[3] <= 5:
                    continue
                if ann['embedding'] is None:
                    continue
                patch = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                track_folder = os.path.join(output_dir, str(ann['track_id']))
                os.makedirs(track_folder, exist_ok=True)
                patch_file_name = os.path.join(track_folder, f"frame_{img_id}.jpg")
                cv2.imwrite(patch_file_name, patch)
                track_embeddings[ann['track_id']].append(ann['embedding'])
    return track_embeddings

def calculate_similarities(track_embeddings):
    intra_similarities = {}
    for track_id, embeddings in track_embeddings.items():
        embeddings = np.array(embeddings)
        if embeddings.ndim == 1 and embeddings.size > 0:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.size == 0:
            continue
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
                if embeddings_i.size == 0 or embeddings_j.size == 0:
                    continue
                similarities = cosine_similarity(embeddings_i, embeddings_j)
                inter_similarities.extend(similarities.flatten())
    return intra_similarities, inter_similarities

def plot_combined_histogram(intra_similarities, inter_similarities, output_dir):
    all_intra_similarities = np.concatenate(list(intra_similarities.values()))
    all_inter_similarities = np.array(inter_similarities)

    plt.figure(figsize=(10, 8))
    plt.hist(all_intra_similarities, bins=20, range=(0, 1), alpha=0.7, label='Intra-track', density=True)
    plt.hist(all_inter_similarities, bins=20, range=(0, 1), alpha=0.7, label='Inter-track', color='orange', density=True)
    plt.title('Track Similarities')
    plt.xlabel('Similarity Score')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_similarity_histogram_normalized.png"))
    plt.close()

# Path to your 'train.json' file and images directory
file_path = '/home/fatih/phd/SMART/data/sompt22/annotations/val.json'
img_dir = '/home/fatih/phd/SMART/data/sompt22/images/val'
output_dir = './exp/vector/exp4'

data = load_json_data(file_path)
annotations = data['annotations']
images_dict = {image['id']: image['file_name'] for image in data['images']}
track_embeddings = extract_patches_and_embeddings(annotations, img_dir, output_dir, images_dict)
intra_similarities, inter_similarities = calculate_similarities(track_embeddings)
plot_combined_histogram(intra_similarities, inter_similarities, output_dir)
