# %%
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
from dotenv import load_dotenv
import os
import numpy as np
from jd_requirements import processed_jd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pandas as pd

key = os.getenv("QDRANT_API_KEY")

# %%
client = QdrantClient(
    url="https://056f99e7-5645-4d17-8066-1be39b8a205e.eu-west-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODkwNjg4OWMtNDE4Yy00Nzk2LTg0MzgtMzM3Nzg4MTkzY2ZiIn0.XL6D03Q0OGfYY9zyuAG4O7JLbNrd8D-e8Cb_tcP7Ivc",
    cloud_inference=True
)
#print(client.get_collections())

# %%
model = SentenceTransformer("thenlper/gte-base")

# %%
if len(sys.argv) != 2:
    print("Usage: python redrob_backend.py <jd_file_path>")
    sys.exit(1)

jd_file_path = sys.argv[1]

jd_data = processed_jd(jd_file_path)
# print(jd_data)
# jd_data = processed_jd()
# print(jd_data)

print("JD Ingetion")
career_embedding = model.encode(jd_data["career"], normalize_embeddings=True)
skills_embedding = model.encode(jd_data["skills"], normalize_embeddings=True)
summary_embedding = model.encode(jd_data["summary"], normalize_embeddings=True)

# %%
print("sematic search")
career_results = client.query_points(
    collection_name="Candidates",
    query=career_embedding,
    using="career",
    limit=200
)
summary_results = client.query_points(
    collection_name="Candidates",
    query=summary_embedding,
    using="summary",
    limit=200
)
skills_results = client.query_points(
    collection_name="Candidates",
    query=skills_embedding,
    using="skills",
    limit=200
)

# %%
from collections import defaultdict

candidate_scores = defaultdict(
    lambda: {
        "career": 0,
        "summary": 0,
        "skills": 0,
        "payload": None
    }
)

# %%


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("scoring")
for candidate_id, data in candidate_scores.items():

    # Skip if all scores already exist
    if all(data[k] != 0 for k in ("career", "skills", "summary")):
        continue

    # Fetch the point with all named vectors
    point = client.retrieve(
        collection_name="Candidates",
        ids=[candidate_id],
        with_vectors=True
    )[0]

    vectors = point.vector

    if data["career"] == 0:
        data["career"] = cosine_similarity(
            query_embedding,
            vectors["career"]
        )

    if data["skills"] == 0:
        data["skills"] = cosine_similarity(
            query_embedding,
            vectors["skills"]
        )

    if data["summary"] == 0:
        data["summary"] = cosine_similarity(
            query_embedding,
            vectors["summary"]
        )

# %%
for point in career_results.points:
    candidate_scores[point.id]["career"] = point.score
    candidate_scores[point.id]["payload"] = point.payload

for point in summary_results.points:
    candidate_scores[point.id]["summary"] = point.score
    candidate_scores[point.id]["payload"] = point.payload

for point in skills_results.points:
    candidate_scores[point.id]["skills"] = point.score
    candidate_scores[point.id]["payload"] = point.payload


ranked_candidates = []

for point_id, data in candidate_scores.items():

    final_score = (
        0.2 * data["skills"] +
        0.3 * data["summary"] +
        0.4 * data["career"] 
    )

    ranked_candidates.append({
        "candidate_id": data["payload"]["candidate_id"],
        "score": final_score
    })

# %%
print("ranked candidates")
print(len(ranked_candidates))

# %%
import json
candidates = []

print("readinf candidate data")
with open("/home/storm/Projects/Resume_Filter/dataset/info/candidates.jsonl", "r") as f:
    for line in f:
        candidates.append(json.loads(line)) 

# %%
selected_cands = []
print("selecting candidates")
for c in ranked_candidates:
    for cand in candidates:
        if cand["candidate_id"] == c["candidate_id"]:
            cand["score"] = c["score"]
            selected_cands.append(cand)
            break;

# %%
print("sorting")
sorted_selected_cands = sorted(selected_cands, key=lambda x: x['score'], reverse=True)
print(sorted_selected_cands[0]["score"])
print(len(sorted_selected_cands))

# %%
import json

class CandidateVerificator:
    def __init__(self, buffer_months: float = 6.0):
        # Allow a grace period (e.g., 6 months) for beta testing or rounding errors
        self.buffer_years = buffer_months / 12.0

        # Max allowed YEARS of experience as of 2026
        self.max_tech_age_years = {
            # GenAI & LLM Era (~2022-2023 onwards)
            "langchain": 4, "llamaindex": 4, "prompt engineering": 4,
            "fine-tuning llms": 4, "rag": 6, "peft": 5, "lora": 5,  # Updated RAG to 6 (2020)
            "qlora": 3, "vector search": 5, "vector representations": 5,
            "embeddings": 6, "diffusion models": 6, "tts": 6,
        
            # Vector Databases & Modern ML Infra
            "pinecone": 5, "qdrant": 5, "pgvector": 5, "weaviate": 6,
            "milvus": 7, "bentoml": 7, "weights & biases": 9, "kubeflow": 9,  # Updated both to 9 (2017)
        
            # Modern Web & Backend Frameworks
            "fastapi": 8, "tailwind": 9, "next.js": 10, "grpc": 11,
        
            # Big Data, Search & Transformers
            "apache beam": 10, "apache flink": 11, "opensearch": 5,
            "hugging face transformers": 8, "sentence transformers": 7,
            "airflow": 11, "dbt": 10, "databricks": 13,
        
            # Agentic & Post-2023 GenAI Era
            "ai agents": 3, "agentic workflows": 3, "langgraph": 2,
            "crewai": 3, "autogen": 3, "mcp": 2,  # Updated MCP to 2 (Late 2024)
            "function calling": 3, "structured outputs": 2,
            "reasoning models": 2, "graphrag": 2, "agentic rag": 2,  # Updated reasoning models to 2 (2024)
            "evals": 3, "ragas": 3, "llm-as-judge": 3
        }
    def verify_candidate(self, candidate: dict) -> tuple:
        """
        Scans the candidate's structured skills array. 
        Returns (True, skill_name, claimed_months) if an impossible timeline is found,
        otherwise returns (False, None, 0).
        """
        skills = candidate.get("skills", [])
        
        if isinstance(skills, list):
            for item in skills:
                if not isinstance(item, dict):
                    continue
                
                name = item.get("name", "").lower().strip()
                months = float(item.get("duration_months", 0))
                years_claimed = months / 12.0
                
                if name in self.max_tech_age_years:
                    max_allowed = self.max_tech_age_years[name] + self.buffer_years
                    
                    if years_claimed > max_allowed:
                        # Return the verdict, the exact text string, and the raw month count
                        return True, item.get("name"), item.get("duration_months")
                        
        return False, None, 0

# --- Main Pipeline Execution ---
print("candidates validation")
verifier = CandidateVerificator(buffer_months=18.0)
safe_candidates_pool = []
liar_candidates_pool = []  # Stores structured details of caught candidates

print("Scanning candidates.jsonl...")


        
for cand in sorted_selected_cands:        # Check candidate validity
    is_liar, caught_skill, claimed_months = verifier.verify_candidate(cand)
        
    if is_liar:
        liar_candidates_pool.append({
            "candidate_id": cand["candidate_id"],
            "caught_by_skill": caught_skill,
            "claimed_duration_months": claimed_months,
            "record": cand
        })
    else:
        safe_candidates_pool.append(cand)

print("--- Scanning Complete ---")
print(f"Total liar/honeypot candidates caught: {len(liar_candidates_pool)}")
print(f"Total safe candidates isolated: {len(safe_candidates_pool)}")

# Quick preview of the caught entries with duration reporting
if liar_candidates_pool:
    print("\n--- Sample of Caught Honeypots ---")
    for item in liar_candidates_pool[:5]:
        print(f"ID: {item['candidate_id']} | Skill: {item['caught_by_skill']} | Claimed: {item['claimed_duration_months']} months")

    # for c in liar_candidates_pool:
    #     for cand in sorted_selected_cands:
    #         if c["candidate_id"] == cand["candidate_id"]:
    #           sorted_selected_cands.remove(cand)
    #         break;
    # Rebuild the list with only the candidates that passed verification
    sorted_selected_cands = [cand for cand in sorted_selected_cands if cand in safe_candidates_pool]

# %%
sorted_selected_cands = sorted(sorted_selected_cands, key=lambda x: x['score'], reverse=True)
print(sorted_selected_cands[0]["score"])
print(len(sorted_selected_cands))
print(sorted_selected_cands[0]["career_history"])

# %%


def find_employment_overlaps(history_list):
    # Use the current date for active jobs (None)
    current_date = datetime(2026, 6, 20) 
    parsed_history = []
    
    # 1. Parse dates and convert them to an absolute month index
    for job in history_list:
        start = datetime.strptime(job['start_date'], "%Y-%m-%d")
        
        if job['end_date'] is None:
            end = current_date
        else:
            end = datetime.strptime(job['end_date'], "%Y-%m-%d")
            
        # Absolute month integer (e.g., Jan 2024 = 2024 * 12 + 1)
        start_idx = (start.year * 12) + start.month
        end_idx = (end.year * 12) + end.month
        
        parsed_history.append({
            'start_date': job['start_date'],
            'end_date': job['end_date'] if job['end_date'] else "Present",
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    # 2. Ensure data is sorted chronologically (earliest start date first)
    parsed_history.sort(key=lambda x: x['start_idx'])
    
    overlaps = []
    
    # 3. Slide through history and compare consecutive jobs
    for i in range(len(parsed_history) - 1):
        current_job = parsed_history[i]
        next_job = parsed_history[i + 1]
        
        # If the next job starts before or during the current job's end month
        if next_job['start_idx'] <= current_job['end_idx']:
            # Calculate how many months they overlap
            overlapping_months = (current_job['end_idx'] - next_job['start_idx']) + 1
            
            overlaps.append({
                "job_a": f"{current_job['start_date']} to {current_job['end_date']}",
                "job_b": f"{next_job['start_date']} to {next_job['end_date']}",
                "overlap_detected": True,
                "shared_months": overlapping_months
            })
            
    return overlaps


# %%
for cand in sorted_selected_cands:

    dates = []

    for j in cand["career_history"]:
        dates.append({
            "start_date": j["start_date"],
            "duration_months": j["duration_months"],
            "end_date": j["end_date"]
        })

    detected_overlaps = find_employment_overlaps(dates)

    significant_overlaps = [
        overlap
        for overlap in detected_overlaps
        if overlap["shared_months"] > 1
    ]

    if significant_overlaps:
        print(f"\nCandidate: {cand['candidate_id']}")
        print(json.dumps(significant_overlaps, indent=4))

# %%
def adjust_score_by_skill_validation(candidate: dict, score: float) -> float:
    """
    Adjusts candidate score based on consistency between:
        - claimed proficiency
        - endorsements
        - skill assessment score

    Rules:
    - If both assessment and endorsements are missing -> do nothing.
    - Experts are expected to have high endorsements and high assessment.
    - Beginners are not penalized for low values.
    - Maximum adjustment: +/-5 points.
    """

    skills = candidate.get("skills", [])
    assessments = (
        candidate.get("redrob_signals", {})
        .get("skill_assessment_scores", {})
    )

    adjustment = 0.0

    # Expected endorsement count
    endorsement_expectation = {
        "advanced": 20,
        "expert": 35,
        "intermediate": 8,
        "beginner": 0
    }

    for skill in skills:

        name = skill.get("name")
        proficiency = skill.get("proficiency", "").lower()
        endorsements = skill.get("endorsements")

        assessment = assessments.get(name)

        # No evidence -> ignore
        if endorsements is None and assessment is None:
            continue

        expected = endorsement_expectation.get(proficiency, 0)

        # -------------------------
        # Expert / Advanced Skills
        # -------------------------
        if proficiency in ("advanced", "expert"):

            # Assessment
            # Advanced / Expert
            # Advanced / Expert
            if assessment is not None:
                if assessment >= 65:
                    adjustment += 0.005
                elif assessment >= 50:
                    adjustment += 0.002
                elif assessment < 35:
                    adjustment -= 0.005
            
            if endorsements is not None:
                if endorsements >= expected:
                    adjustment += 0.003
                elif endorsements < expected * 0.4:
                    adjustment -= 0.003

        # -------------------------
        # Intermediate Skills
        # -------------------------
        elif proficiency == "intermediate":

            if assessment is not None:
                if assessment >= 60:
                    adjustment += 0.003
                elif assessment < 30:
                    adjustment -= 0.003
            
            if endorsements is not None:
                if endorsements >= expected:
                    adjustment += 0.002
                elif endorsements < expected * 0.4:
                    adjustment -= 0.002

        # -------------------------
        # Beginners
        # -------------------------
        elif proficiency == "beginner":

            # Reward if they're already performing well.
            if assessment is not None and assessment >= 60:
                adjustment += 0.08

            if endorsements is not None and endorsements >= 15:
                adjustment += 0.08

    # Cap adjustment
    adjustment = max(-0.02, min(0.02, adjustment))
    return max(0.0, min(1.0, score + adjustment))

# %%
for cand in sorted_selected_cands:
    cand["score"] = adjust_score_by_skill_validation(cand, cand["score"])

# %%
sorted_selected_cands = sorted(sorted_selected_cands, key=lambda x: x['score'], reverse=True)
print(sorted_selected_cands[0]["score"])
print(len(sorted_selected_cands))
# print(sorted_selected_cands[0]["education"])
print(sorted_selected_cands[0]["redrob_signals"])

# %%
#open to work - NO
before = len(sorted_selected_cands)

sorted_selected_cands = [
    cand for cand in sorted_selected_cands
    if cand.get("redrob_signals", {}).get("open_to_work_flag", False)
]

after = len(sorted_selected_cands)

print(f"Removed {before - after} candidates who are not open to work.")
print(f"Remaining candidates: {after}")

# %%
print(sorted_selected_cands[210]["score"])
print(len(sorted_selected_cands))
# print(sorted_selected_cands[0]["education"])
print(sorted_selected_cands[210]["profile"])

# %%
#github activity score
def adjust_score_by_github_activity(candidate: dict, score: float) -> float:
    """
    Adjust candidate score using GitHub activity.

    Uses dataset percentiles:
        Q1  = 17.3
        Q2  = 47.2
        Q3  = 70.16
        P90 = 80.54

    Maximum adjustment: ±0.01
    """

    github_score = (
        candidate.get("redrob_signals", {})
                 .get("github_activity_score")
    )

    # Missing value
    if github_score is None or github_score < 0:
        return score

    adjustment = 0.0

    if github_score >= 80.54:
        # Top 10%
        adjustment = 0.0999

    elif github_score >= 70.16:
        # Top 25%
        adjustment = 0.0888

    elif github_score >= 47.20:
        # Above median
        adjustment = 0.0399

    elif github_score >= 17.30:
        # Average
        adjustment = 0.0100

    elif github_score >= 5:
        # Low activity
        adjustment = -0.002

    else:
        # Almost no activity
        adjustment = -0.005

    return max(0.0, min(1.0, score + adjustment))

# %%
#github function calling
for cand in sorted_selected_cands:
    cand["score"] = adjust_score_by_github_activity(
        cand,
        cand["score"]
    )

# %%
sorted_selected_cands = sorted(sorted_selected_cands, key=lambda x: x['score'], reverse=True)
print(sorted_selected_cands[100]["score"])
print(len(sorted_selected_cands))
# print(sorted_selected_cands[0]["education"])
print(sorted_selected_cands[16]["profile"])

# %%
#off JD candidates
OFF_JD_KEYWORDS = {
    "marketing",
    "marketing manager",
    "digital marketing",
    "sales",
    "sales manager",
    "business development",
    "business analyst",
    "hr",
    "human resources",
    "recruiter",
    "talent acquisition",
    "finance",
    "accountant",
    "operations",
    "operations manager",
    "customer success",
    "customer support",
    "content writer",
    "copywriter",
    "graphic designer",
    "ui designer",
    "ux designer",
    "lawyer",
    "legal",
    "procurement",
    "supply chain",
    "purchase",
    "manufacturing",
    "production manager",
    "quality assurance manager",
    "qa manager",
    "mechanical engineer",
    "civil engineer",
    "electrical engineer",
    "electronics engineer"
}

def has_off_jd_headline(candidate: dict) -> bool:
    headline = (
        candidate.get("profile", {})
                 .get("headline", "")
                 .lower()
    )

    # Allow technical project/program managers
    if (
        "technical project manager" in headline or
        "technical program manager" in headline or
        "engineering manager" in headline
    ):
        return False

    return any(keyword in headline for keyword in OFF_JD_KEYWORDS)

before = len(sorted_selected_cands)

sorted_selected_cands = [
    cand for cand in sorted_selected_cands
    if not has_off_jd_headline(cand)
]

print(f"Removed {before - len(sorted_selected_cands)} off-JD candidates.")

# %%
print(len(sorted_selected_cands))
sorted_selected_cands = sorted(sorted_selected_cands, key=lambda x: x['score'], reverse=True)

print(sorted_selected_cands[19])

# %%
# for cand in sorted_selected_cands:
#     print(cand["score"])

print(sorted_selected_cands[100])

# %%

# %%



rows = []

for cand in sorted_selected_cands:

    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})

    rows.append({
        "candidate_id": cand.get("candidate_id"),
        "score": cand.get("score"),

        "name": profile.get("anonymized_name"),
        "headline": profile.get("headline"),
        "current_title": profile.get("current_title"),
    })

df = pd.DataFrame(rows)

df.to_csv("ranked_candidates.csv", index=False)

print(f"\nSaved {len(df)} candidates to ranked_candidates.csv")