# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/shambhuz28/Redrob_AI.git
cd RedrobAI
```

---

## 2. Install Dependencies

### Backend

```bash
cd backend
npm install
pip install -r requirements.txt
```

### Frontend

```bash
cd ../frontend
npm install
```

---

## 3. Configure Environment Variables

Create a `.env` file inside the `backend` directory with the following variables:

```env
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODkwNjg4OWMtNDE4Yy00Nzk2LTg0MzgtMzM3Nzg4MTkzY2ZiIn0.XL6D03Q0OGfYY9zyuAG4O7JLbNrd8D-e8Cb_tcP7Ivc
QDRANT_URL=https://056f99e7-5645-4d17-8066-1be39b8a205e.eu-west-1-0.aws.cloud.qdrant.io
QDRANT_COLLECTION=Candidates
EMBEDDING_MODEL=thenlper/gte-base

GEMINI_API_KEY=AQ.Ab8RN6JBPKh48iv7xqwKudgAWJiiHLnexSyyJ_5H_5PwAcH6UQ
```



---

## 4. Start the Backend

```bash
cd backend
npm run dev
```

---

## 5. Start the Frontend

Open a new terminal.

```bash
cd frontend
npm run dev
```

---

## 6. Access the Application

Frontend:

```
http://localhost:5173
```

Backend:

```
http://localhost:5000
```

#Keys are provided here beacuse embeddings are stored on cloud and fetching them requires API key