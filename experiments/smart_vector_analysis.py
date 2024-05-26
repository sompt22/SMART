import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt

def load_association_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_embeddings(data, num_track_ids):
    track_embeddings = defaultdict(list)
    count_tracks = 0
    for sequence in data['Sequences']:
        sequence_id = sequence['SequenceID']
        for frame_tracks in sequence['Tracks']:
            for track in frame_tracks:
                tracking_id = track['tracking_id']
                unique_track_id = (sequence_id, tracking_id)
                if unique_track_id not in track_embeddings and count_tracks >= num_track_ids:
                    continue
                if unique_track_id not in track_embeddings:
                    count_tracks += 1
                embedding = track['embedding']
                track_embeddings[unique_track_id].append(embedding)
                if count_tracks >= num_track_ids:
                    break
            if count_tracks >= num_track_ids:
                break
        if count_tracks >= num_track_ids:
            break
    return track_embeddings

def calculate_cosine_similarities(track_embeddings):
    intra_track_similarities = {}
    inter_track_similarities = defaultdict(list)

    track_ids = list(track_embeddings.keys())
    for track_id in track_ids:
        embeddings = np.array(track_embeddings[track_id])
        if len(embeddings) > 1:
            intra_sim = cosine_similarity(embeddings)
            intra_track_similarities[track_id] = intra_sim[np.triu_indices(len(embeddings), k=1)]

    for i, track_id1 in enumerate(track_ids):
        for j, track_id2 in enumerate(track_ids):
            if i < j:
                embeddings1 = np.array(track_embeddings[track_id1])
                embeddings2 = np.array(track_embeddings[track_id2])
                inter_sim = cosine_similarity(embeddings1, embeddings2)
                inter_track_similarities[(track_id1, track_id2)] = inter_sim.flatten()

    return intra_track_similarities, inter_track_similarities

def plot_cosine_similarities(intra_track_similarities, inter_track_similarities, save_path=None):
    intra_sim_scores = []
    for sims in intra_track_similarities.values():
        intra_sim_scores.extend(sims)

    inter_sim_scores = []
    for sims in inter_track_similarities.values():
        inter_sim_scores.extend(sims)

    plt.figure(figsize=(10, 5))
    plt.hist(intra_sim_scores, bins=50, alpha=0.5, density=True, label='Intra-track Cosine Similarities', range=(0, 1))
    plt.hist(inter_sim_scores, bins=50, alpha=0.5, density=True, label='Inter-track Cosine Similarities', range=(0, 1))
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Histogram of Cosine Similarities')
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main(file_path, num_track_ids, save_path=None):
    data = load_association_data(file_path)
    track_embeddings = extract_embeddings(data, num_track_ids)
    intra_track_similarities, inter_track_similarities = calculate_cosine_similarities(track_embeddings)

    print("Intra-track Cosine Similarities:")
    for track_id, sims in intra_track_similarities.items():
        print(f"Track ID {track_id}: {sims}")

    print("\nInter-track Cosine Similarities:")
    for track_pair, sims in inter_track_similarities.items():
        print(f"Track Pair {track_pair}: {sims}")

    plot_cosine_similarities(intra_track_similarities, inter_track_similarities, save_path)

if __name__ == "__main__":
    file_path = '/home/fatih/phd/SMART/exp/vector/exp3/sompt22-vector_kd1/association.json'
    num_track_ids = 50
    save_path = 'cosine_similarities_plot.png'  # Specify the path to save the plot
    main(file_path, num_track_ids, save_path)
