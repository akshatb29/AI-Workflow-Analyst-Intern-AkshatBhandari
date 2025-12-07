import json
import numpy as np
import pandas as pd

# Import our custom modules
from data_loader import load_and_preprocess_data, generate_embeddings, flatten_intent_map
from clustering import perform_clustering, get_representative_samples

INPUT_FILE = 'inputs_for_assignment.json'
OUTPUT_FILE = 'cluster_output.json'

def main():
    # 1. Load & Embed (Module 1 & 2)
    print("--- STEP 1: Ingestion ---")
    df, full_json = load_and_preprocess_data(INPUT_FILE)
    
    if df.empty:
        return

    # Flatten intent map for later use
    intent_map_str = flatten_intent_map(full_json)
    
    print("\n--- STEP 2: Vectorization ---")
    # We pass the combined text column
    vectors = generate_embeddings(df['full_text'].tolist())

    # 2. Cluster (Module 3)
    print("\n--- STEP 3: Clustering ---")
    # Adjust threshold if needed (1.5 is standard for this model)
    raw_clusters = perform_clustering(vectors, distance_threshold=1.5)
    
    refined_clusters = get_representative_samples(raw_clusters, vectors, df)

    # 3. Save Results
    print(f"\n--- STEP 4: Saving {len(refined_clusters)} clusters to JSON ---")
    
    output_data = {
        "total_messages": len(df),
        "intent_map_reference": intent_map_str, # Saving this for the LLM step later
        "clusters": refined_clusters
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"Success! Open '{OUTPUT_FILE}' to see your groups.")

if __name__ == "__main__":
    main()