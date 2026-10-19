# Model Arena API

The Python service behind **Model Arena**, a tool for comparing LLMs on the
same task by accuracy, latency (TTFT, total) and cost per task.

## Architecture in one picture

```
Lovable app (React + Supabase DB)
   │  user clicks "Run"
   ▼
Supabase edge function "run-model"   ← holds ARENA_SERVICE_TOKEN
   │  POST /run  (adds x-service-token header)
   ▼
This service (FastAPI + LangChain, Python)   ← holds the model API keys
   │  ChatAnthropic / ChatOpenAI
   ▼
Claude / OpenAI
   │
   ▼
Result (label, correct?, tokens, TTFT, total ms, cost) → saved in Supabase
```

**Why this split:** the React app owns the UI and data; Python owns the AI
logic. Model API keys live only here, and only your edge function can call
this service.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Wake-up ping (free hosts sleep when idle) |
| POST | `/run` | Run one test case through one model |
| POST | `/score` | Grade all runs for one model: accuracy, recall, precision, confusion, p50/p95 latency, cost per task (same `summarize()` as `run_eval.py`) |

`POST /run` body:

```json
{
  "provider": "anthropic",
  "model_id": "claude-haiku-5-5",
  "system_prompt": "...",
  "input_text": "Paisa kat gaya but plan not active",
  "expected_label": "payment_failed",
  "input_price_per_million": 1.0,
  "output_price_per_million": 5.0,
  "stream": true,
  "max_tokens": 20
}
```

`provider: "fake"` echoes `model_id` back as the answer: free for testing.

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  ARENA_SERVICE_TOKEN=pick-a-long-random-string
uvicorn main:app --reload
# open http://localhost:8000/docs for an interactive test page
```

## Tests

```bash
pip install pytest httpx
pytest -q
```

## Deploy (Render, free tier)

- New → Web Service → connect this GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ARENA_JWT_SECRET`

## Auth (JWT, expires in 30 days)

```bash
python make_token.py --days 30
```

- `ARENA_JWT_SECRET` → Render only. Signs and verifies tokens.
- `ARENA_SERVICE_TOKEN` (the JWT) → Lovable secrets. Sent as the `x-service-token` header.
- Rotate before expiry: `python make_token.py --days 30 --secret <existing secret>`, then update Lovable.
- Fallback: if `ARENA_JWT_SECRET` is not set, the service accepts a plain `ARENA_SERVICE_TOKEN` string instead.

## Evals

`evals.py` scores a model on the 40-ticket test set; `run_eval.py` runs it.

```bash
python run_eval.py --provider fake --model payment_failed          # free sanity check: 17.5%
python run_eval.py --provider anthropic --model claude-haiku-5-5 --in-price <p> --out-price <p>
```

Metrics: accuracy, valid-label rate, error rate, **payment_failed recall**
(the business-critical miss rate), per-label recall/precision, and a
confusion list of the most common mistakes.
