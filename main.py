"""
Model Arena API: the Python "brain" of Build 1.

The Lovable app (React) owns the UI and the database.
This service does one job: run ONE test case through ONE model and
report what happened: the answer, whether it was right, how long it
took, how many tokens it used and what it cost.

Flow:  Lovable page -> Supabase edge function (adds secret token)
       -> POST /run on this service -> LangChain -> Claude / OpenAI
"""

import os
import time
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Two ways to protect this service (set ONE on your host, e.g. Render):
# 1. ARENA_JWT_SECRET: callers must send a signed JWT that has not
#    expired (made with make_token.py). Preferred: tokens expire.
# 2. ARENA_SERVICE_TOKEN: callers must send this exact string.
JWT_SECRET = os.environ.get("ARENA_JWT_SECRET", "")
SERVICE_TOKEN = os.environ.get("ARENA_SERVICE_TOKEN", "")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

app = FastAPI(title="Model Arena API", version="1.0")

# Allow browser calls during local testing. The edge function is the
# real caller in production, so CORS is not your security boundary;
# the service token is.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response shapes (this is the "contract" with the Lovable app)
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    provider: Literal["anthropic", "openai", "openrouter", "fake"]
    model_id: str
    system_prompt: str
    input_text: str
    expected_label: Optional[str] = None
    input_price_per_million: float = Field(0, ge=0)
    output_price_per_million: float = Field(0, ge=0)
    stream: bool = True
    max_tokens: int = Field(20, ge=1, le=4000)


class ScoreItem(BaseModel):
    """One finished run, as stored in the runs table."""
    expected: str
    predicted: Optional[str] = None  # None = the call errored
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    client_total_ms: Optional[float] = None
    cost_usd: Optional[float] = None


class ScoreRequest(BaseModel):
    """All runs for ONE model in one experiment."""
    items: list[ScoreItem] = Field(..., min_length=1, max_length=20000)


class RunResult(BaseModel):
    output_text: str
    parsed_label: Optional[str]
    is_correct: Optional[bool]
    input_tokens: int
    output_tokens: int
    ttft_ms_server: Optional[float]
    total_ms_server: float
    cost_usd: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Model factory: LangChain gives one interface for every provider
# ---------------------------------------------------------------------------


def build_model(req: RunRequest):
    if req.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=req.model_id, temperature=0, max_tokens=req.max_tokens
        )
    if req.provider == "openai":
        from langchain_openai import ChatOpenAI

        # stream_usage=True makes OpenAI report token counts while streaming
        return ChatOpenAI(
            model=req.model_id,
            temperature=0,
            max_tokens=req.max_tokens,
            stream_usage=True,
        )
    if req.provider == "openrouter":
        from langchain_openai import ChatOpenAI

        # OpenRouter speaks the OpenAI API format, so the same LangChain
        # class works: only the URL and the key change. One key reaches
        # hundreds of models, e.g. "provider/model" or "provider/model:free".
        return ChatOpenAI(
            model=req.model_id,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            temperature=0,
            max_tokens=req.max_tokens,
            stream_usage=True,
            # Free models are rate-limited (HTTP 429). Retry with
            # exponential backoff instead of failing the ticket at once.
            max_retries=4,
            timeout=60,
        )
    # "fake" provider: a free stand-in model for testing the pipeline
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )

    return GenericFakeChatModel(messages=iter([req.model_id]))


def parse_label(text: str) -> str:
    """Models sometimes add punctuation or quotes. Normalise to a label."""
    return text.strip().strip("\"'`.").strip().lower()


def count_tokens(usage: Optional[dict], req: RunRequest, output_text: str):
    """Prefer the provider's real counts; estimate only for the fake model."""
    if usage:
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    est = lambda s: max(1, len(s) // 4)  # ~4 characters per token
    return est(req.system_prompt + req.input_text), est(output_text)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def check_caller(token: str) -> None:
    """Reject the request unless it carries a valid token."""
    if JWT_SECRET:
        import jwt

        try:
            # Verifies the signature AND that "exp" is in the future.
            jwt.decode(
                token, JWT_SECRET, algorithms=["HS256"], options={"require": ["exp"]}
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    elif SERVICE_TOKEN and token != SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Bad service token")


@app.get("/health")
def health():
    """Ping this to wake the service before a run (free hosts sleep)."""
    return {"ok": True}


@app.post("/run", response_model=RunResult)
def run(req: RunRequest, x_service_token: str = Header(default="")):
    """HTTP layer: check who is calling, then hand over to the logic."""
    check_caller(x_service_token)
    return run_one(req)


@app.post("/score")
def score_runs(req: ScoreRequest, x_service_token: str = Header(default="")):
    """Grade a whole batch for one model: the numbers the Results tab shows.
    Uses the same summarize() as run_eval.py, so the dashboard and the
    terminal can never disagree."""
    check_caller(x_service_token)
    from evals import summarize

    return summarize([item.model_dump() for item in req.items])


def run_one(req: RunRequest) -> RunResult:
    """Business logic: run one test case through one model.
    No HTTP, no auth here, so scripts like run_eval.py can call it directly."""
    messages = [
        SystemMessage(content=req.system_prompt),
        HumanMessage(content=req.input_text),
    ]

    start = time.perf_counter()
    ttft_ms = None
    try:
        model = build_model(req)

        if req.stream:
            # Streaming: chunks arrive token by token (the DECODE phase).
            # The first non-empty chunk marks time to first token.
            full = None
            for chunk in model.stream(messages):
                if ttft_ms is None and chunk.content:
                    ttft_ms = (time.perf_counter() - start) * 1000
                full = chunk if full is None else full + chunk
            output_text = full.content if full else ""
            usage = getattr(full, "usage_metadata", None)
        else:
            # Non-streaming: nothing arrives until the whole answer is done,
            # so the user's "first token" moment equals total time.
            msg = model.invoke(messages)
            output_text = msg.content
            usage = getattr(msg, "usage_metadata", None)

        total_ms = (time.perf_counter() - start) * 1000
        if ttft_ms is None:
            ttft_ms = total_ms

        if isinstance(output_text, list):  # some providers return blocks
            output_text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in output_text
            )

        in_tok, out_tok = count_tokens(usage, req, output_text)
        cost = (
            in_tok * req.input_price_per_million
            + out_tok * req.output_price_per_million
        ) / 1_000_000

        label = parse_label(output_text)
        correct = (
            None
            if req.expected_label is None
            else label == req.expected_label.strip().lower()
        )

        return RunResult(
            output_text=output_text,
            parsed_label=label,
            is_correct=correct,
            input_tokens=in_tok,
            output_tokens=out_tok,
            ttft_ms_server=round(ttft_ms, 1),
            total_ms_server=round(total_ms, 1),
            cost_usd=round(cost, 8),
        )

    except Exception as e:  # never crash the batch: report the error
        total_ms = (time.perf_counter() - start) * 1000
        return RunResult(
            output_text="",
            parsed_label=None,
            is_correct=None,
            input_tokens=0,
            output_tokens=0,
            ttft_ms_server=None,
            total_ms_server=round(total_ms, 1),
            cost_usd=0,
            error=f"{type(e).__name__}: {e}"[:500],
        )
