# RedrobAI Resume Filter Website

MERN-style app with separate frontend and backend folders.

## Setup

```bash
npm run install:all
cp backend/.env.example backend/.env
```

Place your existing `redrob_backend.py` file inside `backend`.

```env
PYTHON_SCRIPT=redrob_backend.py
```

The backend starts the script in a background job like this by default:

```bash
python3 redrob_backend.py /absolute/path/to/uploaded/file.pdf
```

The upload request does not wait for Python to finish. `POST /api/match` returns a `jobId`, and clients poll `GET /api/job/:jobId` until the job is completed or failed.

## CSV Output

The uploaded PDF/DOCX path is passed to `redrob_backend.py`. For each background job, the backend sets `CSV_OUTPUT_PATH` to a unique file under `backend/job-results/`, so concurrent jobs do not overwrite each other. The backend looks for the CSV in this order:

1. The per-job `CSV_OUTPUT_PATH` set for the Python child process
2. A `.csv` path printed by the Python script to stdout
3. `backend/ranked_candidates.csv`
4. CSV content printed directly to stdout

## Run

```bash
npm run dev
```

Frontend: `http://localhost:5173`

Backend: `http://localhost:5000`

## Deployment

Frontend builds read `VITE_API_URL`, for example:

```env
VITE_API_URL=https://redrob-ai-backend.onrender.com
```

Backend deploys need both Node and Python dependencies. On Render, use a backend build command like:

```bash
npm install && pip install -r requirements.txt
```

Set these backend environment variables in Render:

```env
FRONTEND_ORIGIN=https://redrob-ai-frontend-tau.vercel.app
PYTHON_COMMAND=python3
PYTHON_SCRIPT=redrob_backend.py
PYTHON_TIMEOUT_MS=300000
JOB_TTL_MS=1800000
JOB_CLEANUP_INTERVAL_MS=300000
GEMINI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION=Candidates
CANDIDATE_QUERY_LIMIT=150
HYDRATE_MISSING_SCORES=false
TOP_CANDIDATE_LIMIT=100
CANDIDATES_JSONL=/path/to/dataset/info/candidates.jsonl
```

`CANDIDATES_JSONL` must point to a file that exists in the deployed backend environment.
## API Flow

Start matching:

```http
POST /api/match
```

Response:

```json
{ "jobId": "<uuid>" }
```

Poll status:

```http
GET /api/job/<uuid>
```

Queued/running response:

```json
{ "jobId": "<uuid>", "status": "running", "progress": 45 }
```

Completed response includes the same CSV payload shape used by the old synchronous endpoint:

```json
{
  "jobId": "<uuid>",
  "status": "completed",
  "progress": 100,
  "csvPath": "...",
  "rows": { "headers": [], "records": [] }
}
```
