# ESG & Safety Incident Intelligence Dashboard

An end-to-end NLP and semantic retrieval intelligence platform designed for renewable energy operations (solar blocks, wind turbines, and substations) to classify incidents, score severity, retrieve historical precedent, extract recurring risk trends, and recommend corrective HSE actions.

## Live Deployment
- **Railway URL:** `<PASTE_YOUR_RAILWAY_URL_HERE_AFTER_DEPLOYING>`

## Architecture & AI/ML Method Selection
1. **Preprocessing & Feature Engineering:** Lowercasing, punctuation/whitespace normalization, and TF-IDF vectorization with `(1,2)` n-grams and English stop-word removal inside a leak-free `scikit-learn` `Pipeline`.
2. **Supervised Classification:** Dual `LogisticRegression` classifiers (`class_weight="balanced"`) predicting **ESG/Safety Category** (`safety`, `environment`, `esg_compliance`) and **Severity** (`low`, `medium`, `high`) with calibrated confidence scores.
3. **Threshold-Filtered Retrieval:** Cosine similarity over TF-IDF vectors retrieves relevant historical incidents above a `0.10` similarity threshold (displaying a clear fallback message when no close precedent exists).
4. **Summary & Preventive Guidance:** Rule-grounded executive summary generator mapping predicted domain and severity tiers to renewable plant escalation protocols and corrective SOPs.
5. **Category-Wise Trend Extraction:** Domain-segmented TF-IDF weighting and document frequency analysis surface recurring operational hazards separately across `safety`, `environment`, and `esg_compliance`.

## Local Setup & Run Commands
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --reload --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

## Evaluation & Validation Approach
- **Leak-Free Pipeline Validation:** `TfidfVectorizer` is fitted strictly inside each `StratifiedKFold` split via a `scikit-learn` `Pipeline` to prevent data leakage.
- **Metrics Reported:** Overall Accuracy, Macro F1-Score, Confusion Matrices, and Per-Class Precision, Recall, and F1-Score (with safe `N/A` fallback if class counts are insufficient).

## Assumptions, Limitations & Failure Cases
- **Assumptions:** Labeled solar/wind operational incidents represent recurring site ESG and safety hazards; lexical n-grams capture primary domain signals (PPE, oil seepage, LOTO, waste segregation).
- **Limitations:** The 24-record corpus is suitable for prototype validation but requires larger plant-wide logs for production scale; TF-IDF lexical similarity may miss deep semantic synonyms compared to dense neural embeddings.
- **Failure Cases:** Multi-hazard narratives (e.g., an oil spill causing a slip injury) receive only a single primary category; negated phrases (e.g., *"no oil leak detected"*) can still trigger hazard keywords.
- **Production Next Steps:** Upgrade retrieval to dense sentence embeddings (`all-MiniLM-L6-v2`) + optional LLM synthesis via environment variables (`OPENAI_API_KEY`), and capture HSE officer feedback for continuous model retraining.