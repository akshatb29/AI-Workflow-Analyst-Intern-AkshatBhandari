import pandas as pd
import numpy as np
from data_loader import load_and_preprocess_data, generate_embeddings
from clustering import perform_clustering

# --- CONFIGURATION ---
ORIGINAL_FILE = 'inputs_for_assignment.json'
# We will inject a distinct intent that DEFINITELY doesn't exist yet.
HIDDEN_INTENT_NAME = "Partnership / B2B Inquiry"
HIDDEN_MESSAGES = [
    "I want to partner with your brand for a sponsorship",
    "Do you offer franchise options?",
    "I am an influencer looking for a collaboration",
    "Who can I contact for business partnership?",
    "We are a logistics company wanting to work with you",
    "B2B collaboration request",
    "I want to sell your products in my shop",
    "Marketing tie-up proposal",
    "Corporate bulk orders for employees",
    "Affiliate marketing program details please",
    "Do you have a reseller program?",
    "I want to be a distributor",
    "Looking for vendor registration",
    "Partnership opportunity",
    "Can we collaborate on Instagram?",
    "Business query regarding distribution",
    "Sales team contact for bulk purchase",
    "I want to list my products on your site",
    "Sponsorship for college fest",
    "Collab request"
]

def run_injection_test():
    print(f"--- STARTING 'NEEDLE IN A HAYSTACK' VALIDATION ---")
    
    # 1. Load Original Data
    df_orig, _ = load_and_preprocess_data(ORIGINAL_FILE)
    original_count = len(df_orig)
    
    # 2. Create Injected Data
    print(f"\n1. Injecting {len(HIDDEN_MESSAGES)} messages about '{HIDDEN_INTENT_NAME}'...")
    injected_data = []
    for msg in HIDDEN_MESSAGES:
        injected_data.append({
            'history': '', # No history needed for this test
            'current_message': msg,
            'full_text': f"Context: \n New Message: {msg}",
            'is_injected': True # Marker to track them later
        })
    
    df_injected = pd.DataFrame(injected_data)
    
    # 3. Merge
    df_final = pd.concat([df_orig, df_injected], ignore_index=True)
    df_final['is_injected'] = df_final.get('is_injected', False).fillna(False)
    
    print(f"   Total Messages: {len(df_final)}")

    # 4. Run Vectorization & Clustering (Same logic as main pipeline)
    print("\n2. Running Pipeline (Embedding + Clustering)...")
    vectors = generate_embeddings(df_final['full_text'].tolist())
    
    # Use the same threshold as your main pipeline
    clusters = perform_clustering(vectors, distance_threshold=1.5)
    
    # 5. Analyze Results
    print("\n3. Analyzing Clusters for the Hidden Intent...")
    
    best_cluster_id = -1
    best_recall = 0.0
    best_precision = 0.0
    
    for c_id, indices in clusters.items():
        # Get rows for this cluster
        cluster_rows = df_final.iloc[indices]
        
        # Count how many are "injected"
        injected_count = cluster_rows['is_injected'].sum()
        total_in_cluster = len(cluster_rows)
        
        # Calculate Metrics
        # Recall: How many of the 20 did we find?
        recall = injected_count / len(HIDDEN_MESSAGES)
        
        # Precision: How pure is this cluster? (Did we accidentally include "Order Status"?)
        precision = injected_count / total_in_cluster
        
        if recall > 0.5: # If we found more than half the hidden messages
            print(f"   -> FOUND CANDIDATE: Cluster {c_id}")
            print(f"      Size: {total_in_cluster}")
            print(f"      Injected Messages Found: {injected_count}/{len(HIDDEN_MESSAGES)}")
            print(f"      Recall: {recall:.2%}")
            print(f"      Precision: {precision:.2%}")
            best_cluster_id = c_id
            best_recall = recall
            best_precision = precision

    # 6. Final Verdict
    print("\n" + "="*40)
    print("   VALIDATION REPORT")
    print("="*40)
    
    if best_cluster_id != -1:
        print(f"SUCCESS: The pipeline successfully isolated the hidden intent.")
        print(f"Intent: {HIDDEN_INTENT_NAME}")
        print(f"Recall (Recovery Rate): {best_recall:.1%}")
        print(f"Precision (Purity):     {best_precision:.1%}")
        
        if best_recall == 1.0 and best_precision == 1.0:
            print("VERDICT: PERFECT. The new intent was isolated with 100% accuracy.")
        elif best_recall > 0.8:
            print("VERDICT: STRONG. The system reliably detects new patterns.")
        else:
            print("VERDICT: PASSABLE. It found the intent but missed some variations.")
    else:
        print("FAILURE: The pipeline scattered the hidden messages into existing clusters.")
        print("Try lowering the clustering threshold.")

if __name__ == "__main__":
    run_injection_test()