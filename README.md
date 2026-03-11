# GoG CS Bot Demo

Deployable demo for comparing two Guns of Glory customer service approaches:

- `RAG`: vector retrieval from Supabase + grounded answer generation
- `Long Context`: full FAQ corpus stuffed into the model prompt

The repo keeps the original root-level experiment scripts and adds a deployable app split:

- [backend](/c:/Users/selma/Desktop/codex_projects/cs_bot/backend)
- [frontend](/c:/Users/selma/Desktop/codex_projects/cs_bot/frontend)

## Local Run

Backend:

```powershell
cd backend
python -m pip install -r requirements.txt
$env:SUPABASE_URL="https://zsgpdhmplikmmmwwnxbd.supabase.co"
$env:SUPABASE_SERVICE_KEY="your_service_key"
$env:DASHSCOPE_API_KEY="your_dashscope_key"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
$env:VITE_API_URL="http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

## API

Backend endpoints:

- `GET /health`
- `POST /api/rag`
- `POST /api/long-context`

Request body:

```json
{
  "question": "How do I unblock someone in chat?"
}
```

Response shape:

```json
{
  "answer": "string",
  "sources": [],
  "model_used": "qwen3.5-plus",
  "latency_seconds": 19.66,
  "tokens_used": 1103,
  "method": "rag",
  "status": "success"
}
```

## Deployment

### Railway Backend

1. Create a Railway project from this repo.
2. Set the root directory to `backend`.
3. Add env vars:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `DASHSCOPE_API_KEY`
4. Deploy. Railway will use [Procfile](/c:/Users/selma/Desktop/codex_projects/cs_bot/backend/Procfile).

Expected start command:

```text
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Vercel Frontend

1. Create a Vercel project from this repo.
2. Set the root directory to `frontend`.
3. Add env var:
   - `VITE_API_URL=https://your-backend-url.up.railway.app`
4. Deploy.

Build settings:

- Install command: `npm install`
- Build command: `npm run build`
- Output directory: `dist`

## Notes

- RAG is much cheaper and faster than Long Context in this demo.
- Long Context is included only as a comparison path.
- Both paths intentionally return `no_retrieval` for future-update questions such as "When is the next update coming?".
- Rotate secrets before publishing this repo or deploying from shared history.
