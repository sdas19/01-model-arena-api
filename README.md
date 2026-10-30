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

## Providers

| `provider` | Key needed | `model_id` example |
| --- | --- | --- |
| `openrouter` | `OPENROUTER_API_KEY` | `vendor/model` or `vendor/model:free` (copy from openrouter.ai) |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| `openai` | `OPENAI_API_KEY` | an OpenAI model name |
| `fake` | none | the answer to echo, e.g. `payment_failed` |

OpenRouter calls retry up to 4 times with backoff, because free models are rate-limited.

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
- Environment variables: `OPENROUTER_API_KEY`, `ARENA_JWT_SECRET` (optional: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` for direct access)

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
python run_eval.py --provider openrouter --model <vendor/model:free>
python run_eval.py --provider anthropic --model claude-haiku-5-5 --in-price <p> --out-price <p>
```

Metrics: accuracy, valid-label rate, error rate, **payment_failed recall**
(the business-critical miss rate), per-label recall/precision, and a
confusion list of the most common mistakes.

## Findings (v1 run, Oct 2026)

Task: label 40 SpottoTV support tickets (English, Hindi, Hinglish) into 9 categories.
Models: three free reasoning models via OpenRouter, 1 repetition, max 1,000 output tokens.

| Model | Accuracy | payment_failed recall | Error rate | TTFT p50 |
| --- | --- | --- | --- | --- |
| Nemotron 3.5 Super VL | 100% | 100% | 0% | ~1.6 s |
| Xing 4.0 29B-A4B | 66.7% | 85.7% | 33.3% | ~13.8 s |
| Flint v1.0 | 45% | 0% | 55% | ~1.5 s |

**What it means**

- Every model labelled every ticket it answered correctly. All lost accuracy came from failed calls:
  rate limits (Xing) and "no upstream available" (Flint) on shared free capacity.
- So the differences were reliability and latency, not intelligence. Next version reports
  "accuracy when answered" and "availability" separately.
- The test set is saturated: it can't separate models on quality. v2 needs harder cases
  (multi-intent, sarcasm, code-mixing, billing vs payment near-misses).
- 40/40 correct still means a plausible true accuracy of roughly 91–100%.
- For background triage, Nemotron's ~6 s p95 is acceptable; for a user-facing router it would not be.
- Production choice: Nemotron-class quality on a paid endpoint with a fallback model;
  free shared pools are not reliable enough to ship on.

**Engineering lessons from building it**: HTTP 200 does not mean success (errors travel in the body);
store readable errors, never `[object Object]`; exclude setup from latency; cap every run with a time budget;
reasoning tokens count against `max_tokens`.
