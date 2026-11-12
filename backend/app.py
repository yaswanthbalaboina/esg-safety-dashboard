import os
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

app = FastAPI(title="ESG / Safety Incident Intelligence Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "safety_esg_incidents.csv"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

MAX_INPUT_LENGTH = 1000
MIN_SIMILARITY_THRESHOLD = 0.10


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_classifier_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1200)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
    ])


class IncidentIntelligenceEngine:
    def __init__(self, csv_path: Path):
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {"incident_id", "narrative", "target_category", "target_severity"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

        df = df.dropna(subset=["narrative", "target_category", "target_severity"]).drop_duplicates(subset=["narrative"])
        df["clean_narrative"] = df["narrative"].apply(clean_text)
        self.df = df.reset_index(drop=True)

        # Leak-free ML pipelines (vectorizer fitted inside pipeline)
        self.cat_pipeline = build_classifier_pipeline()
        self.sev_pipeline = build_classifier_pipeline()

        self.cat_pipeline.fit(self.df["clean_narrative"], self.df["target_category"])
        self.sev_pipeline.fit(self.df["clean_narrative"], self.df["target_severity"])

        # Dedicated retrieval vectorizer fitted on corpus for similarity lookup
        self.retrieval_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1200)
        self.corpus_matrix = self.retrieval_vectorizer.fit_transform(self.df["clean_narrative"])

        self.metrics = self._evaluate_models()

    def _evaluate_task(self, y_true: pd.Series, pipeline_builder) -> Dict[str, Any]:
        class_counts = y_true.value_counts()
        min_count = int(class_counts.min()) if not class_counts.empty else 0

        if min_count < 2:
            return {
                "available": False,
                "validation_strategy": "N/A (insufficient samples per class for stratified CV)",
                "accuracy": None,
                "macro_f1": None,
                "labels": sorted(y_true.unique().tolist()),
                "confusion_matrix": [],
                "per_class_report": {},
            }

        n_splits = min(4, min_count)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipeline_builder(), self.df["clean_narrative"], y_true, cv=skf)

        labels = sorted(y_true.unique().tolist())
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        per_class = {
            lbl: {
                "precision": round(float(report[lbl]["precision"]), 3),
                "recall": round(float(report[lbl]["recall"]), 3),
                "f1_score": round(float(report[lbl]["f1-score"]), 3),
                "support": int(report[lbl]["support"]),
            }
            for lbl in labels
        }

        return {
            "available": True,
            "validation_strategy": f"{n_splits}-fold leak-free Stratified CV (Pipeline)",
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 3),
            "labels": labels,
            "confusion_matrix": cm,
            "per_class_report": per_class,
        }

    def _evaluate_models(self) -> Dict[str, Any]:
        cat_eval = self._evaluate_task(self.df["target_category"], build_classifier_pipeline)
        sev_eval = self._evaluate_task(self.df["target_severity"], build_classifier_pipeline)
        return {
            "dataset_size": int(len(self.df)),
            "category": cat_eval,
            "severity": sev_eval,
        }

    def retrieve_similar(
        self,
        clean_query: str,
        top_k: int = 3,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        q_vec = self.retrieval_vectorizer.transform([clean_query])
        sims = cosine_similarity(q_vec, self.corpus_matrix).flatten()
        ranked_indices = np.argsort(sims)[::-1]

        matches = []
        for idx in ranked_indices:
            score = float(sims[idx])
            if score < min_similarity:
                continue
            row = self.df.iloc[idx]
            matches.append({
                "incident_id": str(row["incident_id"]),
                "narrative": str(row["narrative"]),
                "category": str(row["target_category"]),
                "severity": str(row["target_severity"]),
                "similarity_score": round(score, 3),
            })
            if len(matches) >= top_k:
                break

        if not matches:
            return {
                "found": False,
                "threshold": min_similarity,
                "message": "No sufficiently similar historical incidents found.",
                "matches": [],
            }

        return {
            "found": True,
            "threshold": min_similarity,
            "message": f"Retrieved {len(matches)} relevant precedent(s) above similarity threshold ({min_similarity}).",
            "matches": matches,
        }

    def _extract_location_and_impact(self, text: str) -> Dict[str, str]:
        lower = text.lower()
        location_keywords = [
            "solar block", "wind turbine", "gearbox", "substation", "inverter yard",
            "control room", "switchgear room", "warehouse", "rooftop", "cable trench",
            "access road", "storage area", "bess", "battery", "walkway", "tower"
        ]
        found_locs = [loc for loc in location_keywords if loc in lower]
        location = ", ".join(found_locs).title() if found_locs else "Plant Operations Zone (unspecified block)"

        impact_keywords = [
            "sprain", "injury", "oil", "leak", "seepage", "overflow", "arc flash",
            "runoff", "waste", "non-compliance", "harness", "lockout", "near miss", "spike"
        ]
        found_impacts = [imp for imp in impact_keywords if imp in lower]
        impact = ", ".join(found_impacts) if found_impacts else "operational & compliance exposure"

        return {"location": location, "impact": impact}

    def generate_summary_and_actions(
        self,
        text: str,
        category: str,
        severity: str,
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = self._extract_location_and_impact(text)

        action_map = {
            "safety": {
                "high": [
                    "Halt work immediately at the affected zone and verify Lockout/Tagout (LOTO) and fall-arrest barriers.",
                    "Escalate near-miss or high-voltage/lifting hazard to Site HSE Lead within 1 hour.",
                    "Conduct mandatory stand-down Toolbox Talk (TBT) before work resumption.",
                ],
                "medium": [
                    "Inspect walkway housekeeping, PPE usage, and barricading across the active shift.",
                    "Assign corrective action owner with 24-hour verification sign-off.",
                ],
                "low": [
                    "Correct minor PPE/housekeeping observation on the spot and log in weekly safety walkdown.",
                ],
            },
            "environment": {
                "high": [
                    "Deploy secondary containment / absorbent spill kits immediately and isolate leaking valve/equipment.",
                    "Initiate soil/drainage contamination check and notify Environmental Officer within 1 hour.",
                    "Inspect sibling wind/solar assets for identical seal or transformer wear.",
                ],
                "medium": [
                    "Repair valve/runoff control within 24 hours and log water/resource deviation.",
                    "Verify storm-drain and rainwater harvesting filtration barriers.",
                ],
                "low": [
                    "Clean minor residue/coolant trace and schedule preventive maintenance check.",
                ],
            },
            "esg_compliance": {
                "high": [
                    "Quarantine unlabeled hazardous waste immediately inside authorized containment shed.",
                    "Issue formal Non-Conformance Report (NCR) to contractor lead within 4 hours.",
                ],
                "medium": [
                    "Audit fire safety, sensor calibration, and subcontractor induction records within 48 hours.",
                    "Update compliance tracker ahead of internal ESG governance review.",
                ],
                "low": [
                    "Re-brief contractor team on waste segregation and monthly ESG reporting deadlines.",
                ],
            },
        }

        urgency = {
            "high": "IMMEDIATE ESCALATION (Within 1 Hour — Site HSE Lead & Plant Head)",
            "medium": "PRIORITY ACTION (Within 24 Hours — Shift Supervisor & Area Engineer)",
            "low": "ROUTINE FOLLOW-UP (Weekly HSE & ESG Governance Review)",
        }.get(severity, "Standard HSE Review")

        precedent_note = (
            f"Matches historical case {retrieval['matches'][0]['incident_id']} "
            f"(similarity {retrieval['matches'][0]['similarity_score']})."
            if retrieval["found"]
            else "No close historical match found; flagged for fresh HSE review."
        )

        summary = (
            f"[{severity.upper()} | {category.upper()}] Location: {context['location']}. "
            f"Primary hazard/impact: {context['impact']}. {precedent_note}"
        )

        actions = action_map.get(category, {}).get(severity, ["Log incident and assign HSE reviewer."])

        return {
            "synthesis_method": "Deterministic domain-rule & lexical retrieval synthesis (no external LLM dependency)",
            "detected_location": context["location"],
            "detected_impact": context["impact"],
            "executive_summary": summary,
            "escalation_protocol": urgency,
            "recommended_actions": actions,
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        cleaned = clean_text(text)
        if not cleaned:
            raise HTTPException(status_code=400, detail="Narrative must contain alphanumeric characters.")

        pred_cat = str(self.cat_pipeline.predict([cleaned])[0])
        cat_probs = dict(zip(self.cat_pipeline.classes_, np.round(self.cat_pipeline.predict_proba([cleaned])[0], 3)))

        pred_sev = str(self.sev_pipeline.predict([cleaned])[0])
        sev_probs = dict(zip(self.sev_pipeline.classes_, np.round(self.sev_pipeline.predict_proba([cleaned])[0], 3)))

        retrieval = self.retrieve_similar(cleaned, top_k=3, min_similarity=MIN_SIMILARITY_THRESHOLD)
        guidance = self.generate_summary_and_actions(text, pred_cat, pred_sev, retrieval)

        return {
            "input_narrative": text,
            "preprocessed_text": cleaned,
            "predictions": {
                "category": pred_cat,
                "category_confidence": cat_probs,
                "severity": pred_sev,
                "severity_confidence": sev_probs,
            },
            "summary_and_actions": guidance,
            "retrieval": retrieval,
            "reasoning_steps": [
                "1. Normalized narrative text and validated character bounds.",
                f"2. Classified domain as '{pred_cat}' and severity as '{pred_sev}' via leak-free TF-IDF + Logistic Regression pipelines.",
                f"3. Filtered historical corpus with cosine similarity >= {MIN_SIMILARITY_THRESHOLD}.",
                "4. Extracted plant location/impact entities and mapped category-severity escalation SOPs.",
            ],
            "assumptions_and_limitations": {
                "assumptions": [
                    "Labeled solar/wind operational incidents represent recurring site ESG and safety hazards.",
                    "Lexical n-grams capture primary domain signals (PPE, oil seepage, LOTO, waste segregation).",
                ],
                "limitations": [
                    "24-record corpus is suitable for prototype validation but requires larger plant-wide logs for production scale.",
                    "TF-IDF lexical similarity may miss deep semantic synonyms compared to dense sentence transformers.",
                ],
                "failure_cases": [
                    "Multi-hazard narratives (e.g., chemical spill causing worker injury) receive only a single primary category.",
                    "Negated statements (e.g., 'no oil leak detected') can still trigger hazard keywords.",
                ],
                "production_next_steps": [
                    "Upgrade retrieval to dense embeddings (all-MiniLM-L6-v2) + optional LLM synthesis via environment variables.",
                    "Capture HSE officer feedback on predictions to continuously retrain models.",
                ],
            },
        }

    def _extract_category_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        trends_by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for cat, group in self.df.groupby("target_category"):
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50)
            mat = vec.fit_transform(group["clean_narrative"])
            scores = np.asarray(mat.sum(axis=0)).flatten()
            # Also compute document frequency per term within this category
            doc_freq = np.asarray((mat > 0).sum(axis=0)).flatten()
            vocab = vec.get_feature_names_out()
            top_idx = np.argsort(scores)[::-1][:5]

            trends_by_cat[str(cat)] = [
                {
                    "trend_term": str(vocab[i]),
                    "frequency": int(doc_freq[i]),
                    "tfidf_weight": round(float(scores[i]), 3),
                }
                for i in top_idx
            ]
        return trends_by_cat

    def get_dashboard_overview(self) -> Dict[str, Any]:
        cat_counts = self.df["target_category"].value_counts().to_dict()
        sev_counts = self.df["target_severity"].value_counts().to_dict()

        high_df = self.df[self.df["target_severity"] == "high"]
        high_risk = [
            {
                "incident_id": str(r["incident_id"]),
                "narrative": str(r["narrative"]),
                "target_category": str(r["target_category"]),
                "target_severity": str(r["target_severity"]),
                "escalation_priority": "P1 - Immediate (< 1 hr)",
            }
            for _, r in high_df.iterrows()
        ]

        return {
            "total_incidents": int(len(self.df)),
            "category_distribution": cat_counts,
            "severity_distribution": sev_counts,
            "category_wise_trends": self._extract_category_trends(),
            "high_severity_incidents": high_risk,
            "evaluation_metrics": self.metrics,
            "sample_incidents": self.df[["incident_id", "narrative", "target_category", "target_severity"]].to_dict(orient="records"),
        }


engine = IncidentIntelligenceEngine(DATA_PATH)


class Query(BaseModel):
    text: str = ""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if FRONTEND_PATH.exists():
        return FRONTEND_PATH.read_text(encoding="utf-8")
    return "<h1>Frontend not found. Ensure frontend/index.html exists.</h1>"


@app.get("/api/demo")
def demo():
    return {
        "case_study": "ESG / Safety Incident Intelligence Dashboard",
        "status": "operational",
        "llm_provider_configured": bool(os.getenv("OPENAI_API_KEY")),
        "metrics": engine.metrics,
        "expected_outcome": "Leak-free category/severity classification, threshold-filtered retrieval, category-wise trends, and actionable summaries.",
    }


@app.get("/api/dashboard")
def dashboard_stats():
    return engine.get_dashboard_overview()


@app.post("/api/run")
def run_case(query: Query):
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=400, detail="Incident narrative text cannot be empty.")
    if len(query.text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Incident narrative exceeds maximum length of {MAX_INPUT_LENGTH} characters.",
        )
    return engine.analyze(query.text)