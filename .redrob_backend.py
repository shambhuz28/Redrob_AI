import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from jd_requirements import processed_jd


load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "Candidates")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "thenlper/gte-base")
QUERY_LIMIT = int(os.getenv("CANDIDATE_QUERY_LIMIT", "500"))
STRICT_HONEYPOT_FILTER = os.getenv("STRICT_HONEYPOT_FILTER", "false").lower() == "true"

SCORE_WEIGHTS = {
    "career": 0.40,
    "skills": 0.35,
    "summary": 0.25,
}

OFF_JD_KEYWORDS = {
    "accountant",
    "business analyst",
    "business development",
    "civil engineer",
    "content writer",
    "copywriter",
    "customer success",
    "customer support",
    "digital marketing",
    "electrical engineer",
    "electronics engineer",
    "finance",
    "graphic designer",
    "hr",
    "human resources",
    "lawyer",
    "legal",
    "manufacturing",
    "marketing",
    "marketing manager",
    "mechanical engineer",
    "operations",
    "operations manager",
    "procurement",
    "production manager",
    "purchase",
    "qa manager",
    "quality assurance manager",
    "recruiter",
    "sales",
    "sales manager",
    "supply chain",
    "talent acquisition",
    "ui designer",
    "ux designer",
}

TECHNICAL_KEYWORDS = {
    "airflow",
    "api",
    "aws",
    "backend",
    "cloud",
    "cuda",
    "data engineer",
    "databricks",
    "dbt",
    "docker",
    "embedding",
    "fastapi",
    "feature engineering",
    "fine-tuning",
    "gcp",
    "github",
    "kafka",
    "kubernetes",
    "langchain",
    "llm",
    "machine learning",
    "ml",
    "nlp",
    "python",
    "pytorch",
    "qdrant",
    "rag",
    "react",
    "spark",
    "sql",
    "tensorflow",
    "transformer",
    "typescript",
    "vector",
}


class CandidateVerificator:
    def __init__(self, buffer_months: float = 18.0):
        self.buffer_years = buffer_months / 12.0
        self.max_tech_age_years = {
            "agentic rag": 2,
            "agentic workflows": 3,
            "ai agents": 3,
            "airflow": 11,
            "apache beam": 10,
            "apache flink": 11,
            "autogen": 3,
            "bentoml": 7,
            "crewai": 3,
            "databricks": 13,
            "dbt": 10,
            "diffusion models": 6,
            "embeddings": 6,
            "evals": 3,
            "fastapi": 8,
            "fine-tuning llms": 4,
            "function calling": 3,
            "graphrag": 2,
            "grpc": 11,
            "hugging face transformers": 8,
            "kubeflow": 9,
            "langchain": 4,
            "langgraph": 2,
            "llamaindex": 4,
            "llm-as-judge": 3,
            "lora": 5,
            "mcp": 2,
            "milvus": 7,
            "next.js": 10,
            "opensearch": 5,
            "peft": 5,
            "pgvector": 5,
            "pinecone": 5,
            "prompt engineering": 4,
            "qdrant": 5,
            "qlora": 3,
            "rag": 6,
            "ragas": 3,
            "reasoning models": 2,
            "sentence transformers": 7,
            "structured outputs": 2,
            "tailwind": 9,
            "tts": 6,
            "vector representations": 5,
            "vector search": 5,
            "weaviate": 6,
            "weights & biases": 9,
        }

    def verify_candidate(self, candidate: dict) -> tuple[bool, str | None, float]:
        for skill in candidate.get("skills", []):
            if not isinstance(skill, dict):
                continue

            name = skill.get("name", "").lower().strip()
            months = safe_float(skill.get("duration_months"))
            years_claimed = months / 12.0

            if name not in self.max_tech_age_years:
                continue

            max_allowed = self.max_tech_age_years[name] + self.buffer_years
            if years_claimed > max_allowed:
                return True, skill.get("name"), months

        return False, None, 0.0


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python redrob_backend.py <jd_file_path>")
        sys.exit(1)

    jd_file_path = sys.argv[1]
    client = build_qdrant_client()
    model = SentenceTransformer(MODEL_NAME)

    print("JD ingestion")
    jd_data = processed_jd(jd_file_path)
    query_embeddings = {
        "career": model.encode(jd_data["career"], normalize_embeddings=True),
        "skills": model.encode(jd_data["skills"], normalize_embeddings=True),
        "summary": model.encode(jd_data["summary"], normalize_embeddings=True),
    }

    print("Semantic search")
    candidate_scores = collect_candidate_scores(client, query_embeddings)
    hydrate_missing_scores(client, candidate_scores, query_embeddings)

    ranked_candidates = build_ranked_candidates(candidate_scores)
    print(f"Ranked candidates from vector search: {len(ranked_candidates)}")

    candidates_by_id = read_candidates_by_id(resolve_candidates_path())
    selected_candidates = attach_records(ranked_candidates, candidates_by_id)
    print(f"Candidate records matched locally: {len(selected_candidates)}")

    scored_candidates = score_candidate_records(selected_candidates)
    output_path = write_csv(scored_candidates)
    print(output_path)


def build_qdrant_client() -> QdrantClient:
    url = os.getenv(
        "QDRANT_URL",
        "https://056f99e7-5645-4d17-8066-1be39b8a205e.eu-west-1-0.aws.cloud.qdrant.io",
    )
    api_key = os.getenv("QDRANT_API_KEY")

    if not api_key:
        raise ValueError("QDRANT_API_KEY not found in environment or .env")

    return QdrantClient(url=url, api_key=api_key, cloud_inference=True)


def collect_candidate_scores(client: QdrantClient, query_embeddings: dict) -> defaultdict:
    candidate_scores = defaultdict(new_candidate_score)

    for vector_name, embedding in query_embeddings.items():
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            using=vector_name,
            limit=QUERY_LIMIT,
        )

        for point in results.points:
            candidate_scores[point.id][vector_name] = point.score
            if point.payload:
                candidate_scores[point.id]["payload"] = point.payload

    return candidate_scores


def new_candidate_score() -> dict:
    return {
        "career": None,
        "summary": None,
        "skills": None,
        "payload": None,
    }


def hydrate_missing_scores(
    client: QdrantClient,
    candidate_scores: defaultdict,
    query_embeddings: dict,
) -> None:
    candidate_ids = [
        candidate_id
        for candidate_id, data in candidate_scores.items()
        if any(data[name] is None for name in SCORE_WEIGHTS)
    ]

    if not candidate_ids:
        return

    print(f"Hydrating missing vector scores for {len(candidate_ids)} candidates")

    for id_batch in batched(candidate_ids, 64):
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=id_batch,
            with_payload=True,
            with_vectors=True,
        )

        for point in points:
            data = candidate_scores[point.id]
            if point.payload and data["payload"] is None:
                data["payload"] = point.payload

            vectors = point.vector or {}
            if not isinstance(vectors, dict):
                continue

            for vector_name, query_embedding in query_embeddings.items():
                if data[vector_name] is None and vector_name in vectors:
                    data[vector_name] = cosine_similarity(
                        query_embedding,
                        vectors[vector_name],
                    )


def build_ranked_candidates(candidate_scores: defaultdict) -> list[dict]:
    ranked_candidates = []

    for data in candidate_scores.values():
        payload = data.get("payload") or {}
        candidate_id = payload.get("candidate_id")

        if not candidate_id:
            continue

        final_score = sum(
            SCORE_WEIGHTS[name] * safe_score(data.get(name)) for name in SCORE_WEIGHTS
        )

        matched_channels = sum(data.get(name) is not None for name in SCORE_WEIGHTS)
        if matched_channels == 3:
            final_score += 0.015
        elif matched_channels == 1:
            final_score -= 0.015

        ranked_candidates.append(
            {
                "candidate_id": candidate_id,
                "score": clamp_score(final_score),
                "matched_channels": matched_channels,
            }
        )

    return sorted(ranked_candidates, key=lambda item: item["score"], reverse=True)


def read_candidates_by_id(path: Path) -> dict:
    candidates_by_id = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            candidate = json.loads(line)
            candidates_by_id[candidate["candidate_id"]] = candidate

    return candidates_by_id


def resolve_candidates_path() -> Path:
    configured_path = os.getenv("CANDIDATES_JSONL")
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    script_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "dataset" / "info" / "candidates.jsonl",
        script_path.parent / "dataset" / "info" / "candidates.jsonl",
    ]
    candidates.extend(
        parent / "dataset" / "info" / "candidates.jsonl"
        for parent in script_path.parents
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find dataset/info/candidates.jsonl. Set CANDIDATES_JSONL."
    )


def attach_records(ranked_candidates: list[dict], candidates_by_id: dict) -> list[dict]:
    selected_candidates = []

    for ranked in ranked_candidates:
        candidate = candidates_by_id.get(ranked["candidate_id"])
        if not candidate:
            continue

        enriched = dict(candidate)
        enriched["score"] = ranked["score"]
        enriched["matched_channels"] = ranked["matched_channels"]
        selected_candidates.append(enriched)

    return selected_candidates


def score_candidate_records(candidates: list[dict]) -> list[dict]:
    verifier = CandidateVerificator()
    scored_candidates = []
    rejected_count = 0

    for candidate in candidates:
        candidate = dict(candidate)
        notes = []
        score = safe_score(candidate.get("score"))

        is_suspicious, caught_skill, claimed_months = verifier.verify_candidate(candidate)
        if is_suspicious:
            candidate["validation_status"] = "suspicious_skill_duration"
            candidate["validation_detail"] = (
                f"{caught_skill} claimed for {claimed_months:g} months"
            )

            if STRICT_HONEYPOT_FILTER:
                rejected_count += 1
                continue

            score -= 0.08
            notes.append("suspicious_skill_duration")
        else:
            candidate["validation_status"] = "ok"
            candidate["validation_detail"] = ""

        score = adjust_score_by_skill_validation(candidate, score)
        score = adjust_score_by_github_activity(candidate, score)
        score = adjust_score_by_availability(candidate, score, notes)
        score = adjust_score_by_headline(candidate, score, notes)
        score = adjust_score_by_employment_overlap(candidate, score, notes)

        candidate["score"] = clamp_score(score)
        candidate["ranking_notes"] = "; ".join(notes)
        scored_candidates.append(candidate)

    if rejected_count:
        print(f"Strict honeypot filter removed {rejected_count} candidates.")

    return sorted(scored_candidates, key=lambda item: item["score"], reverse=True)


def adjust_score_by_skill_validation(candidate: dict, score: float) -> float:
    skills = candidate.get("skills", [])
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    adjustment = 0.0
    endorsement_expectation = {
        "advanced": 20,
        "beginner": 0,
        "expert": 35,
        "intermediate": 8,
    }

    for skill in skills:
        if not isinstance(skill, dict):
            continue

        name = skill.get("name")
        proficiency = skill.get("proficiency", "").lower()
        endorsements = skill.get("endorsements")
        assessment = assessments.get(name)

        if endorsements is None and assessment is None:
            continue

        expected = endorsement_expectation.get(proficiency, 0)

        if proficiency in ("advanced", "expert"):
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

        elif proficiency == "beginner":
            if assessment is not None and assessment >= 60:
                adjustment += 0.004
            if endorsements is not None and endorsements >= 15:
                adjustment += 0.004

    return clamp_score(score + max(-0.02, min(0.02, adjustment)))


def adjust_score_by_github_activity(candidate: dict, score: float) -> float:
    github_score = candidate.get("redrob_signals", {}).get("github_activity_score")

    if github_score is None or github_score < 0:
        return score

    if github_score >= 80.54:
        adjustment = 0.010
    elif github_score >= 70.16:
        adjustment = 0.008
    elif github_score >= 47.20:
        adjustment = 0.004
    elif github_score >= 17.30:
        adjustment = 0.001
    elif github_score >= 5:
        adjustment = -0.002
    else:
        adjustment = -0.005

    return clamp_score(score + adjustment)


def adjust_score_by_availability(candidate: dict, score: float, notes: list[str]) -> float:
    signals = candidate.get("redrob_signals", {})

    if signals.get("open_to_work_flag") is False:
        notes.append("not_open_to_work")
        return score - 0.03

    return score


def adjust_score_by_headline(candidate: dict, score: float, notes: list[str]) -> float:
    if not has_off_jd_headline(candidate):
        return score

    notes.append("off_jd_headline")
    if has_technical_evidence(candidate):
        return score - 0.04

    return score - 0.12


def adjust_score_by_employment_overlap(
    candidate: dict,
    score: float,
    notes: list[str],
) -> float:
    overlaps = find_employment_overlaps(candidate.get("career_history", []))
    significant_overlaps = [
        overlap for overlap in overlaps if overlap["shared_months"] > 1
    ]

    if not significant_overlaps:
        return score

    candidate["employment_overlap_count"] = len(significant_overlaps)
    notes.append("employment_overlap")
    return score - min(0.04, 0.01 * len(significant_overlaps))


def has_off_jd_headline(candidate: dict) -> bool:
    headline = candidate.get("profile", {}).get("headline", "").lower()

    if any(
        allowed in headline
        for allowed in (
            "engineering manager",
            "technical program manager",
            "technical project manager",
        )
    ):
        return False

    return any(keyword in headline for keyword in OFF_JD_KEYWORDS)


def has_technical_evidence(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    text_parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    text_parts.extend(
        job.get("title", "") + " " + job.get("description", "")
        for job in candidate.get("career_history", [])
        if isinstance(job, dict)
    )
    text_parts.extend(
        skill.get("name", "")
        for skill in candidate.get("skills", [])
        if isinstance(skill, dict)
    )
    evidence_text = " ".join(text_parts).lower()

    return any(keyword in evidence_text for keyword in TECHNICAL_KEYWORDS)


def find_employment_overlaps(history_list: list[dict]) -> list[dict]:
    current_date = datetime.now()
    parsed_history = []

    for job in history_list:
        start_date = job.get("start_date")
        if not start_date:
            continue

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = job.get("end_date")
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else current_date
        start_idx = (start.year * 12) + start.month
        end_idx = (end.year * 12) + end.month

        parsed_history.append(
            {
                "start_date": start_date,
                "end_date": end_date or "Present",
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )

    parsed_history.sort(key=lambda item: item["start_idx"])
    overlaps = []

    for index in range(len(parsed_history) - 1):
        current_job = parsed_history[index]
        next_job = parsed_history[index + 1]

        if next_job["start_idx"] <= current_job["end_idx"]:
            overlaps.append(
                {
                    "job_a": f"{current_job['start_date']} to {current_job['end_date']}",
                    "job_b": f"{next_job['start_date']} to {next_job['end_date']}",
                    "overlap_detected": True,
                    "shared_months": current_job["end_idx"] - next_job["start_idx"] + 1,
                }
            )

    return overlaps


def write_csv(candidates: list[dict]) -> Path:
    rows = []

    for candidate in candidates:
        profile = candidate.get("profile", {})
        rows.append(
            {
                "Candidate_id": candidate.get("candidate_id"),
                "Score": round(candidate.get("score", 0.0), 6),
                "Name": profile.get("anonymized_name"),
                "Reason": profile.get("headline"),
                "current_title": profile.get("current_title"),
                "matched_channels": candidate.get("matched_channels")
            }
        )

    output_path = resolve_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} candidates to {output_path}")
    return output_path


def resolve_output_path() -> Path:
    configured_path = os.getenv("CSV_OUTPUT_PATH", "ranked_candidates.csv")
    return Path(configured_path).expanduser().resolve()


def cosine_similarity(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def safe_score(value) -> float:
    if value is None:
        return 0.0
    return safe_float(value)


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def batched(items: list, size: int) -> Iterable[list]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


if __name__ == "__main__":
    main()

