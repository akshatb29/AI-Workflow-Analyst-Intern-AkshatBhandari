# 🚀 Intent Discovery Pipeline

An automated **intent expansion & discovery system** designed for conversational AI platforms. It identifies **missing intents**, **split-worthy intents**, and **taxonomy improvements** using:

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
```

---

## 🔧 Technologies Used

| Component | Technology |
|-----------|------------|
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Clustering** | Agglomerative / HDBSCAN |
| **LLM Reasoning** | Google Gemini API |
| **Metrics** | Silhouette Score, Davies-Bouldin, Coverage |
| **Testing** | Synthetic Novel Intent Injection |
| **Output Formats** | JSON, CSV, Markdown |

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

**If `requirements.txt` doesn't exist, install manually:**

```bash
pip install pandas numpy scikit-learn sentence-transformers google-generativeai python-dotenv rich hdbscan
```

### 4️⃣ Set Up API Key

Create a `.env` file in the project root:

```bash
touch .env
```

Add your Google Gemini API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

**To get a Gemini API key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Get API Key"
4. Copy and paste into `.env`

---

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

**Expected output:**
```
✓ Data loaded successfully
✓ Embeddings generated
✓ Clustering completed
✓ LLM analysis finished
✓ Reports generated
```

### Run Validation Test

To verify the system can detect completely new intents:

```bash
python validation_experiment.py
```

This injects synthetic examples like:
- "I want to discuss a partnership opportunity"
- "We want to stock your products in our retail chain"

And verifies whether the pipeline discovers a new **"Partnership / B2B"** intent.

---

## 📊 Outputs

After running the pipeline, you'll find these files:

### 1. `final_intent_report.md`
A detailed analysis report containing:
- Proposed new intents with justification
- Split-worthy intents (when existing intents are too broad)
- Representative messages from clusters
- LLM reasoning for each recommendation
- Business impact assessment

### 2. `updated_intent_map.json`
Production-ready intent taxonomy including:
```json
{
  "intents": [
    {
      "intent_name": "discovered_intent_name",
      "description": "What this intent covers",
      "examples": ["message 1", "message 2"],
      "confidence": 0.85,
      "cluster_support": 45
    }
  ]
}
```

### 3. `intent_audit_log.csv`
PM-friendly audit log with columns:
- `intent_name`
- `action` (NEW / SPLIT / MERGE)
- `cluster_support`
- `distinctiveness_score`
- `reasoning`
- `timestamp`

### 4. `cluster_output.json`
Raw cluster diagnostics:
```json
{
  "cluster_0": {
    "size": 23,
    "cohesion_score": 0.78,
    "representative_messages": ["msg1", "msg2", "msg3"],
    "proposed_intent": "intent_name"
  }
}
```

---

## 🧪 Validation & Testing

### Sensitivity Test
Ensures the pipeline can detect novel intents by injecting synthetic data:

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
- Silhouette score > 0.3
- No false merging with existing intents

---

## 🎛️ Configuration

### Clustering Parameters

Edit in `clustering_logic.py`:

```python
# Agglomerative Clustering
n_clusters = 15  # Adjust based on expected intent count
linkage = 'ward'
distance_threshold = None

# HDBSCAN (alternative)
min_cluster_size = 5
min_samples = 3
```

### LLM Parameters

Edit in `llm_analyst_v2.py`:

```python
model = genai.GenerativeModel('gemini-1.5-flash')
temperature = 0.7  # Creativity level (0.0 - 1.0)
max_output_tokens = 2048
```

### Embedding Model

Edit in `clustering_logic.py`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, balanced
# model = SentenceTransformer('all-mpnet-base-v2')  # Slower, more accurate
```

---

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Core Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
google-generativeai>=0.3.0
python-dotenv>=0.19.0
rich>=10.0.0
hdbscan>=0.8.0
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'sentence_transformers'"
**Solution:**
```bash
pip install sentence-transformers
```

### Issue: "API key not valid"
**Solution:**
1. Check `.env` file exists in project root
2. Verify API key is correct (no extra spaces)
3. Ensure key has access to Gemini API
4. Try regenerating key at [Google AI Studio](https://makersuite.google.com/app/apikey)

### Issue: "Clustering produces too many/few clusters"
**Solution:**
Adjust `n_clusters` in `clustering_logic.py`:
- Too many clusters → Decrease `n_clusters`
- Too few clusters → Increase `n_clusters`
- Use HDBSCAN for automatic cluster detection

### Issue: "Low silhouette scores"
**Solution:**
1. Increase data quality (remove noise)
2. Try different embedding models
3. Adjust clustering parameters
4. Increase `min_cluster_size` for HDBSCAN

### Issue: "LLM responses are inconsistent"
**Solution:**
1. Lower `temperature` parameter (more deterministic)
2. Add more examples to prompt
3. Increase `max_output_tokens`
4. Switch to `gemini-pro` for better reasoning

---

## 🔒 Security & Privacy

- **API Keys:** Never commit `.env` files to version control
- **Data:** Ensure customer messages are anonymized before processing
- **Outputs:** Review generated reports before sharing externally
- **Logs:** Audit logs may contain sensitive information

Add to `.gitignore`:
```
.env
*.log
__pycache__/
venv/
.DS_Store
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or support:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Link to full docs]

---

## 🙏 Acknowledgments

- **SentenceTransformers** for semantic embeddings
- **Google Gemini** for LLM reasoning
- **scikit-learn** for clustering algorithms
- The open-source ML community

---

## 📈 Roadmap

- [ ] Support for multilingual intent discovery
- [ ] Real-time intent detection API
- [ ] Integration with popular chatbot platforms
- [ ] Active learning with human feedback
- [ ] Intent drift detection over time
- [ ] Automated A/B testing framework

---

**Version:** 1.0.0  
**Last Updated:** December 2025  
**Status:** Production Ready ✅
