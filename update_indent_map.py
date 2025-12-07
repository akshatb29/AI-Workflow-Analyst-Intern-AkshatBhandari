import json
import csv
import re

# --- CONFIGURATION ---
ORIGINAL_DATA_FILE = 'inputs_for_assignment.json'
AUDIT_LOG_FILE = 'intent_audit_log.csv'
OUTPUT_FILE = 'updated_intent_map.json'

def slugify(text):
    """Converts 'Product Safety' -> 'product_safety' for IDs."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.replace(' ', '_')
    return text

def load_data():
    """Loads original JSON and the CSV audit log."""
    with open(ORIGINAL_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    changes = []
    try:
        with open(AUDIT_LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We only care about NEW or SPLIT proposals
                if row['Proposal'] in ['NEW', 'SPLIT']:
                    changes.append(row)
    except FileNotFoundError:
        print(f"Error: {AUDIT_LOG_FILE} not found. Run the pipeline first.")
        return None, None
        
    return data, changes

def update_map(data, changes):
    """Injects new intents into the JSON structure."""
    intent_map = data['intent_mapper']
    
    print(f"--- Processing {len(changes)} changes ---")
    
    for change in changes:
        full_name = change['Intent_Name'] # e.g., "About Product > Product Safety"
        proposal_type = change['Proposal']
        reasoning = change['Reasoning']
        
        # 1. Parse Primary > Secondary
        if " > " in full_name:
            parts = full_name.split(" > ")
            primary_name = parts[0].strip()
            secondary_name = parts[1].strip()
        else:
            # Fallback if LLM didn't use ">" format (e.g. "New Intent Name")
            # We assign to a default 'General' or try to guess. 
            # For this script, let's assume valid output or skip.
            print(f"Skipping malformed intent name: {full_name}")
            continue

        # 2. Find the Primary Intent Group
        target_primary = None
        for p in intent_map:
            if p['primary_intent_name'].lower() == primary_name.lower():
                target_primary = p
                break
        
        # If Primary doesn't exist (Rare, but possible for NEW primary intents), create it
        if not target_primary:
            print(f"Creating NEW Primary Intent: {primary_name}")
            new_p_id = slugify(primary_name)
            target_primary = {
                "primary_intent_id": new_p_id,
                "primary_intent_name": primary_name,
                "secondary_intents": []
            }
            intent_map.append(target_primary)

        # 3. Check if Secondary already exists
        exists = False
        for s in target_primary['secondary_intents']:
            if s['name'].lower() == secondary_name.lower():
                exists = True
                break
        
        # 4. Add the New Secondary Intent
        if not exists:
            new_id = slugify(secondary_name)
            print(f"  [+] Adding '{secondary_name}' to '{primary_name}' ({proposal_type})")
            
            new_entry = {
                "id": new_id,
                "name": secondary_name,
                "description": f"({proposal_type}) {reasoning}", # We add the reasoning as description
                "added_by": "auto_discovery_pipeline"
            }
            target_primary['secondary_intents'].append(new_entry)
        else:
            print(f"  [!] Skipped '{secondary_name}' (Already exists)")

    return data

def main():
    print("Loading data...")
    data, changes = load_data()
    
    if data and changes:
        updated_data = update_map(data, changes)
        
        # Save only the intent_mapper part (or full file if you prefer)
        final_output = {"intent_mapper": updated_data['intent_mapper']}
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)
            
        print(f"\nSuccess! Updated map saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()