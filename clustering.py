import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from typing import Dict, List

def perform_clustering(embeddings: np.ndarray, distance_threshold: float = 1.5) -> Dict[int, List[int]]:
    """
    Groups vectors using Agglomerative Clustering.
    """
    print(f"Clustering {len(embeddings)} messages (Threshold: {distance_threshold})...")
    
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        metric='euclidean', 
        linkage='ward' 
    )
    
    cluster_labels = clustering_model.fit_predict(embeddings)
    
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if int(label) not in clusters:
            clusters[int(label)] = []
        clusters[int(label)].append(idx)
        
    print(f"Found {len(clusters)} distinct clusters.")
    return clusters

def get_representative_samples(clusters: Dict[int, List[int]], 
                               embeddings: np.ndarray, 
                               df: pd.DataFrame, 
                               samples_per_cluster: int = 5) -> List[Dict]:
    """
    Finds the central messages of each cluster to send to the LLM.
    """
    cluster_summaries = []
    
    for label, indices in clusters.items():
        # Get vectors for this cluster
        cluster_vectors = embeddings[indices]
        
        # Calculate Centroid
        centroid = np.mean(cluster_vectors, axis=0).reshape(1, -1)
        
        # Calculate distances to centroid
        distances = cosine_distances(cluster_vectors, centroid).flatten()
        
        # Get indices of closest N messages
        closest_indices_local = distances.argsort()[:samples_per_cluster]
        closest_indices_global = [indices[i] for i in closest_indices_local]
        
        # Get text
        samples = df.iloc[closest_indices_global]['full_text'].tolist()
        
        cluster_summaries.append({
            'cluster_id': int(label),
            'size': len(indices),
            'percentage': round((len(indices) / len(df)) * 100, 2),
            'representative_samples': samples
        })
        
    # Sort by size (biggest first)
    cluster_summaries.sort(key=lambda x: x['size'], reverse=True)
    return cluster_summaries