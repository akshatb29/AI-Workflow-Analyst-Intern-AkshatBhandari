import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Reads the JSON file and prepares the text for analysis.
    Combines 'History' + 'Current Message' to ensure context is preserved.
    Returns the DataFrame and the full raw JSON object.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame(), {}

    raw_messages = data.get('customer_messages', [])
    processed_rows = []

    for msg in raw_messages:
        history = msg.get('history', '')
        current_msg = msg.get('current_human_message', '')
        
        # KEY LOGIC: Merging History + Current Message
        # The model needs to see "Context: ... New Message: ..." to understand references
        full_context_text = f"Context: {history} \n New Message: {current_msg}"
        
        processed_rows.append({
            'history': history,
            'current_message': current_msg,
            'full_text': full_context_text
        })
    
    print(f"Loaded {len(processed_rows)} messages.")
    return pd.DataFrame(processed_rows), data

def flatten_intent_map(full_json_data: Dict) -> str:
    """
    Converts the nested JSON Intent Map into a flat text reference.
    This string will eventually be fed to the LLM for comparison.
    """
    intent_map = full_json_data.get('intent_mapper', [])
    text_representation = "CURRENT INTENT MAP:\n"
    
    for primary in intent_map:
        p_name = primary.get('primary_intent_name', 'Unknown')
        
        for secondary in primary.get('secondary_intents', []):
            s_name = secondary.get('name', 'Unknown')
            desc = secondary.get('description', '')
            
            # Format: [Primary > Secondary]: Description
            text_representation += f"- [{p_name} > {s_name}]: {desc}\n"
            
    return text_representation

# In data_loader.py

def generate_embeddings(text_list: List[str], model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> np.ndarray:
    """
    Converts text list to a matrix of numbers (vectors).
    Using a multilingual model to handle Hindi/English mix.
    NOTE: Embeddings are now NORMALIZED for accurate clustering.
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Generating NORMALIZED embeddings for {len(text_list)} messages...")
    
    # normalize_embeddings=True ensures all vectors have length 1.
    # This makes Cosine Similarity and Euclidean Distance work reliably for clustering.
    embeddings = model.encode(text_list, show_progress_bar=True, normalize_embeddings=True)
    
    return embeddings