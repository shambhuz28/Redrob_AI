import os, sys, time
print("JD ingestion", flush=True)
time.sleep(0.2)
print("Semantic search", flush=True)
time.sleep(0.2)
path=os.environ.get("CSV_OUTPUT_PATH", "ranked_candidates.csv")
with open(path, "w", encoding="utf-8") as f:
    f.write("Candidate_id,Score,Name
abc,0.99,Test Candidate
")
print(f"Saved 1 candidates to {path}", flush=True)
print(path, flush=True)
