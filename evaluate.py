import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import pandas as pd

INPUT_FILE = 'cluster_output.json'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def calculate_metrics():
    print("--- 1. Loading Data & Model ---")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model = SentenceTransformer(MODEL_NAME)
    
    # Re-construct the dataset from the JSON structure
    texts = []
    labels = []
    
    clusters = data['clusters']
    total_messages = data['total_messages']
    clustered_count = 0
    
    print(f"Analyzing {len(clusters)} clusters...")
    
    for c in clusters:
        # We use the samples to represent the cluster
        # (In a real DB scenario, you'd load all messages, but this approximates well)
        for sample in c['representative_samples']:
            texts.append(sample)
            labels.append(c['cluster_id'])
            clustered_count += 1
            
    # Generate embeddings for the validation set
    print("Generating validation embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # --- METRIC 1: SILHOUETTE SCORE ---
    # Measures how similar an object is to its own cluster (cohesion) 
    # compared to other clusters (separation).
    # > 0.1 is okay for text. > 0.2 is good. > 0.5 is amazing.
    sil_score = silhouette_score(embeddings, labels, metric='cosine')
    
    # --- METRIC 2: COVERAGE / NOISE ---
    # How much data did we actually group meaningfully?
    # (Assuming the JSON contains 'representative' but the 'size' attribute tells the real truth)
    total_clustered_volume = sum([c['size'] for c in clusters])
    coverage_ratio = (total_clustered_volume / total_messages) * 100
    
    # --- REPORTING ---
    print("\n" + "="*40)
    print("   QUANTITATIVE PERFORMANCE METRICS")
    print("="*40)
    
    print(f"\n1. Global Silhouette Score: {sil_score:.4f}")
    if sil_score > 0.2:
        print("   -> VERDICT: STRONG. Clusters are well-separated.")
    elif sil_score > 0.1:
        print("   -> VERDICT: ACCEPTABLE. Some overlap, typical for short text.")
    else:
        print("   -> VERDICT: WEAK. Clusters are very mixed.")
        
    print(f"\n2. Data Coverage: {coverage_ratio:.1f}%")
    print(f"   ({total_clustered_volume} out of {total_messages} messages were grouped)")
    
    print("\n3. Cluster Density (Top 3):")
    # Just showing which clusters are the 'tightest'
    for c in clusters[:3]:
        print(f"   - ID {c['cluster_id']} (Size {c['size']})")

if __name__ == "__main__":
    calculate_metrics()