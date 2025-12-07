```markdown
# 🚀 Intent Discovery Pipeline

An automated **intent expansion & discovery system** designed for conversational AI platforms.  
It identifies **missing intents**, **split-worthy intents**, and **taxonomy improvements** using:

- Unsupervised clustering (semantic embeddings + ML)  
- LLM-based qualitative reasoning  
- Validation & sensitivity testing  
- Guardrails & fallback logic  

This pipeline scales to thousands of customer messages and generates clean, auditable intent recommendations.

---

## 📂 Project Structure

```

.
├── pipeline_orchestrator.py        # Main entry point for the full workflow
├── data_loader.py                  # Loads datasets, cleans text, merges context
├── clustering_logic.py             # Semantic vectorization + clustering
├── llm_analyst_v2.py               # Gemini-based cluster analysis & intent proposals
├── evaluation.py                   # Quantitative metrics (silhouette, coverage, confidence)
├── validation_experiment.py        # Novel intent injection test
├── updated_intent_map.json         # Final updated taxonomy
├── final_intent_report.md          # Deep-dive report explaining each discovered intent
└── intent_audit_log.csv            # PM-friendly audit log for all proposed changes

````

---

## 🔧 Technologies Used

| Component | Technology |
|----------|------------|
| Embeddings | SentenceTransformers |
| Clustering | Agglomerative / HDBSCAN |
| LLM Reasoning | Google Gemini API |
| Metrics | Silhouette Score, Davies-Bouldin, Coverage |
| Testing | Synthetic Novel Intent Injection |
| Output Formats | JSON, CSV, Markdown |

---

## 📦 Installation

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn sentence-transformers \
google-generativeai python-dotenv rich
````

### 2️⃣ Add API Key

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Run the Full Pipeline

This executes:

* Data loading
* Preprocessing
* Embedding + clustering
* LLM analysis
* Intent proposals
* Report generation

```bash
python pipeline_orchestrator.py
```

---

## 📊 Outputs

### **1. final_intent_report.md**

A detailed report including:

* Proposed new intents
* Split-worthy intents
* Representative messages
* LLM reasoning
* Business justification

---

### **2. updated_intent_map.json**

A clean, production-ready taxonomy that merges:

* Original intents
* Newly discovered intents
* Guardrails & fallback rules

---

### **3. intent_audit_log.csv**

A structured PM-facing audit log containing:

* Intent name
* Cluster support
* Distinctiveness score
* Human-readable reasoning

---

### **4. cluster_output.json**

Raw cluster diagnostics including:

* Cluster ID
* Representative messages
* Cohesion & density flags

---

## 🧪 Validation & Sensitivity Test

To verify that the system can detect a **completely new intent**, run:

```bash
python validation_experiment.py
```

This injects synthetic examples like:

```
"I want to discuss a partnership"
"We want to stock your products in our retail chain"
```

And checks whether the pipeline discovers a new **“Partnership / B2B”** intent.

---






If you want, I can **auto-generate the PDF or PPT from this README** — just tell me **"Create PDF"** or **"Create PPT"**.
```
