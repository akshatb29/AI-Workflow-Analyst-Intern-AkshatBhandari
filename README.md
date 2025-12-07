# 🚀 Intent Discovery Pipeline

An **intent expansion & discovery system** designed for conversational AI platforms. It identifies **missing intents**, **split-worthy intents**, and **taxonomy improvements** using:

- Unsupervised clustering
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
```

---

## 🔧 Technologies Used

| Component | Technology |
|-----------|------------|
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Clustering** | Agglomerative Clustering |
| **LLM Reasoning** | Google Gemini API |
| **Metrics** | Silhouette Score, Davies-Bouldin, Coverage |
| **Testing** | Synthetic Novel Intent Injection |
| **Output Formats** | JSON |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/intent-discovery-pipeline.git
cd intent-discovery-pipeline
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
# For Linux/Mac
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up API Key

Create a `.env` file in the project root:

Add your Google Gemini API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

## ▶️ Usage

### Run the Full Pipeline

This executes the complete workflow:
- Data loading & preprocessing
- Semantic embedding generation
- Clustering analysis
- LLM-based intent discovery
- Report generation

```bash
python pipeline_orchestrator.py
```

### Run Validation Test

To verify the system can detect completely new intents:

```bash
python validation_experiment.py
```

This injects synthetic examples like:
- "I want to discuss a partnership opportunity"
- "We want to stock your products in our retail chain"

And verifies whether the pipeline discovers a new intents.

---

## 📊 Outputs

After running the pipeline, you'll find these files:

 1)**final_intent_report.md**: A detailed report explaining the new intents, reasoning, and representative user samples.

 2)**intent_audit_log.csv**: A structured CSV list of all proposed changes for Product Manager review.

 3)**updated_intent_map.json**: A production-ready JSON file with the new intents injected into the original taxonomy.

 4)**cluster_output.json**: Intermediate data showing the raw clusters and samples.
 
---

## 🧪 Validation & Testing

```bash
python validation_experiment.py
```

**What it tests:**
- Detection of completely new intent categories
- Separation from existing intents
- Clustering quality with mixed data
- LLM reasoning accuracy

**Success criteria:**
- New synthetic messages form distinct clusters
- LLM proposes appropriate new intent names
- Silhouette score > 0.4

---


