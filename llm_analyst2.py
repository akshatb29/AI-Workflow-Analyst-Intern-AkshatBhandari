import json
import os
import csv
import time
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIGURATION ---
INPUT_FILE = 'cluster_output.json'
REPORT_FILE = 'final_intent_report.md'
CSV_FILE = 'intent_audit_log.csv'
MIN_CLUSTER_SIZE = 5
MODEL_NAME = "gemini-2.5-flash"

def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)

def get_analysis_prompt(cluster, intent_map):
    """
    Forces the LLM to return strictly formatted JSON.
    """
    samples = cluster['representative_samples']
    
    return f"""
    You are a Data Taxonomist. Analyze this cluster of user messages against the Current Intent Map.
    
    --- CURRENT INTENT MAP ---
    {intent_map}

    --- CLUSTER SAMPLES (Size: {cluster['size']}) ---
    {json.dumps(samples, indent=2)}

    --- INSTRUCTIONS ---
    Determine if this cluster represents:
    1. NEW: A completely missing intent.
    2. SPLIT: A specific sub-topic that deserves to be separated from a broad parent.
    3. EXISTING: Fits perfectly into an existing intent.

    --- OUTPUT FORMAT ---
    You must return a SINGLE Valid JSON object. Do not add markdown formatting like ```json.
    {{
        "proposal": "NEW" or "SPLIT" or "EXISTING",
        "intent_name": "Proposed Name (or Existing Name)",
        "confidence": "High" or "Medium" or "Low",
        "reasoning": "A short, sharp 2-sentence explanation."
    }}
    """

def clean_llm_response(text):
    """
    Cleans potential formatting issues from LLM (like ```json ... ```)
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def generate_reports():
    model = setup_gemini()
    if not model: return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clusters = [c for c in data['clusters'] if c['size'] >= MIN_CLUSTER_SIZE]
    intent_map = data['intent_map_reference']
    
    print(f"Analyzing {len(clusters)} clusters...")
    
    results = []

    # --- 1. ANALYSIS LOOP ---
    for i, cluster in enumerate(clusters):
        print(f"[{i+1}/{len(clusters)}] Processing Cluster {cluster['cluster_id']}...")
        
        prompt = get_analysis_prompt(cluster, intent_map)
        
        try:
            response = model.generate_content(prompt)
            cleaned_json = clean_llm_response(response.text)
            analysis = json.loads(cleaned_json)
            
            # Add metadata for reporting
            analysis['cluster_id'] = cluster['cluster_id']
            analysis['size'] = cluster['size']
            analysis['samples'] = cluster['representative_samples']
            results.append(analysis)
            
            # Rate limit safety
            time.sleep(1)
            
        except Exception as e:
            print(f"  -> Error on Cluster {cluster['cluster_id']}: {e}")

    # --- 2. GENERATE CSV ---
    print(f"Generating CSV: {CSV_FILE}...")
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Cluster_ID', 'Size', 'Proposal', 'Intent_Name', 'Confidence', 'Reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({
                'Cluster_ID': res['cluster_id'],
                'Size': res['size'],
                'Proposal': res['proposal'],
                'Intent_Name': res['intent_name'],
                'Confidence': res['confidence'],
                'Reasoning': res['reasoning']
            })

    # --- 3. GENERATE MARKDOWN REPORT ---
    print(f"Generating Report: {REPORT_FILE}...")
    
    # Sort: NEW first, then SPLIT, then EXISTING
    priority_order = {"NEW": 0, "SPLIT": 1, "EXISTING": 2}
    results.sort(key=lambda x: priority_order.get(x['proposal'], 3))

    md_lines = []
    md_lines.append("# Intent Discovery Report")
    md_lines.append(f"**Total Clusters Analyzed:** {len(clusters)}\n")

    # A. EXECUTIVE SUMMARY TABLE
    md_lines.append("## 1. Executive Summary")
    md_lines.append("| ID | Size | Proposal | Proposed Intent Name | Confidence |")
    md_lines.append("|:---|:---|:---|:---|:---|")
    
    for res in results:
        # Add an emoji for visual scanning
        icon = "🔴" if res['proposal'] == "NEW" else "Mf" if res['proposal'] == "SPLIT" else "✅"
        md_lines.append(f"| {res['cluster_id']} | {res['size']} | {icon} {res['proposal']} | **{res['intent_name']}** | {res['confidence']} |")

    # B. DETAILED SECTIONS
    md_lines.append("\n## 2. Detailed Findings")
    
    for res in results:
        icon = "🔴" if res['proposal'] == "NEW" else "Mf" if res['proposal'] == "SPLIT" else "✅"
        
        md_lines.append(f"### {icon} Cluster {res['cluster_id']}: {res['proposal']} -> {res['intent_name']}")
        md_lines.append(f"**Reasoning:** {res['reasoning']}\n")
        
        # Collapsible Samples
        md_lines.append("<details>")
        md_lines.append(f"<summary>▶ View {len(res['samples'])} Representative User Messages</summary>\n")
        for sample in res['samples'][:5]: # Limit to top 5 in UI
            # Clean up newlines for display
            clean_sample = sample.replace('\n', ' ').replace('Context:', '**Ctx:**').replace('New Message:', '**Msg:**')
            md_lines.append(f"* {clean_sample}")
        md_lines.append("</details>\n")
        md_lines.append("---")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print("Done! Check your folder for the .md and .csv files.")

if __name__ == "__main__":
    generate_reports()