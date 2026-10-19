"""Tests for the Model Arena API. Run with:  pytest -q"""

import csv
import os

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

SYSTEM = "Reply with one label."
CSV_PATH = os.path.join(os.path.dirname(__file__), "spottotv_tickets_test_set.csv")


def payload(**overrides):
    base = dict(
        provider="fake",
        model_id="payment_failed",  # the fake model just echoes this
        system_prompt=SYSTEM,
        input_text="Money debited, plan not active",
        expected_label="payment_failed",
        input_price_per_million=1.0,
        output_price_per_million=5.0,
        stream=True,
        max_tokens=20,
    )
    base.update(overrides)
    return base


def test_health():
    assert client.get("/health").json() == {"ok": True}


def test_streaming_run_scores_and_times():
    r = client.post("/run", json=payload()).json()
    assert r["error"] is None
    assert r["parsed_label"] == "payment_failed"
    assert r["is_correct"] is True
    assert 0 < r["ttft_ms_server"] <= r["total_ms_server"]
    assert r["input_tokens"] > 0 and r["output_tokens"] > 0
    assert r["cost_usd"] > 0


def test_non_streaming_ttft_equals_total():
    r = client.post("/run", json=payload(stream=False)).json()
    assert r["ttft_ms_server"] == r["total_ms_server"]


def test_wrong_label_and_messy_output():
    r = client.post(
        "/run", json=payload(model_id=' "Billing_Query." ')
    ).json()
    assert r["parsed_label"] == "billing_query"
    assert r["is_correct"] is False


def test_no_expected_label_means_unscored():
    r = client.post("/run", json=payload(expected_label=None)).json()
    assert r["is_correct"] is None


def test_token_required_when_configured(monkeypatch):
    monkeypatch.setattr(main, "SERVICE_TOKEN", "secret123")
    assert client.post("/run", json=payload()).status_code == 401
    ok = client.post(
        "/run", json=payload(), headers={"x-service-token": "secret123"}
    )
    assert ok.status_code == 200


def test_provider_error_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post(
        "/run", json=payload(provider="anthropic", model_id="claude-haiku-5-5")
    ).json()
    assert r["error"] is not None
    assert r["parsed_label"] is None


def test_full_test_set_through_fake_model():
    """Simulates a 'perfect' model on all 40 tickets: expect 100%."""
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40
    correct = 0
    for row in rows:
        r = client.post(
            "/run",
            json=payload(
                model_id=row["expected_label"],
                input_text=row["input_text"],
                expected_label=row["expected_label"],
            ),
        ).json()
        correct += r["is_correct"]
    assert correct == 40


def test_eval_metrics():
    from evals import score

    results = [
        {"expected": "payment_failed", "predicted": "payment_failed"},
        {"expected": "payment_failed", "predicted": "general_feedback"},  # costly miss
        {"expected": "billing_query", "predicted": "payment_failed"},
        {"expected": "unclear", "predicted": "not a label"},
    ]
    m = score(results)
    assert m["accuracy"] == 0.25
    assert m["valid_label_rate"] == 0.75
    assert m["critical_recall"] == 0.5          # caught 1 of 2 payment issues
    assert m["per_label"]["payment_failed"]["precision"] == 0.5
    assert m["confusion"]["payment_failed"] == {"general_feedback": 1}
