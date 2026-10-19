"""
Evals for the ticket-triage task.

An eval = a test set + a scoring rule + a report.
- Test set:     spottotv_tickets_test_set.csv (input + expected label)
- Scoring rule: exact match on the label (a "code-based" eval:
                cheap, deterministic, perfect for classification)
- Report:       the metrics below

Each metric answers a different product question.
"""

import math
from collections import Counter, defaultdict

ALLOWED_LABELS = [
    "payment_failed",
    "billing_query",
    "refund_request",
    "subscription_cancel",
    "login_issue",
    "streaming_quality",
    "content_request",
    "general_feedback",
    "unclear",
]

# The category where a miss costs the most (a churned subscriber).
CRITICAL_LABEL = "payment_failed"


def score(results):
    """
    results: list of dicts with keys 'expected' and 'predicted'
             (predicted may be None if the call errored)
    returns: dict of metrics
    """
    n = len(results)
    if n == 0:
        return {}

    correct = sum(r["predicted"] == r["expected"] for r in results)
    valid = sum(r["predicted"] in ALLOWED_LABELS for r in results)
    errors = sum(r["predicted"] is None for r in results)

    # Per-label recall: of tickets that truly ARE label X,
    # how many did the model catch?  ("did we miss any payment issues?")
    # Per-label precision: of tickets the model CALLED label X,
    # how many really were X?  ("are we flooding the payments queue?")
    truth = Counter(r["expected"] for r in results)
    called = Counter(r["predicted"] for r in results)
    hits = Counter(r["expected"] for r in results if r["predicted"] == r["expected"])

    per_label = {}
    for label in ALLOWED_LABELS:
        per_label[label] = {
            "support": truth[label],
            "recall": hits[label] / truth[label] if truth[label] else None,
            "precision": hits[label] / called[label] if called[label] else None,
        }

    # Confusion: which true labels get mistaken for which?
    confusion = defaultdict(Counter)
    for r in results:
        if r["predicted"] != r["expected"]:
            # A failed call has no prediction; name it so it reads clearly
            # (and so the result can be sent as JSON).
            predicted = r["predicted"] if r["predicted"] is not None else "(error)"
            confusion[r["expected"]][predicted] += 1

    return {
        "n": n,
        "accuracy": correct / n,
        # Did the model stay inside the allowed list? If not, the
        # downstream software can't route the ticket at all.
        "valid_label_rate": valid / n,
        "error_rate": errors / n,
        # The business-critical number: missed payment failures.
        "critical_recall": per_label[CRITICAL_LABEL]["recall"],
        "per_label": per_label,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def percentile(values, q):
    """Nearest-rank percentile. q=0.95 -> the value 95% of runs were at or under."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = max(0, math.ceil(q * len(values)) - 1)
    return values[k]


def summarize(items):
    """
    Everything the dashboard shows for ONE model, computed in one place.

    items: list of dicts with keys
      expected, predicted          -> quality metrics (via score)
      ttft_ms, total_ms            -> server latency (Python's stopwatch)
      client_total_ms (optional)   -> what the user waited end to end
      cost_usd                     -> money
    """
    m = score([{"expected": i["expected"], "predicted": i["predicted"]} for i in items])
    if not m:
        return m

    # Latency only from successful calls: an error's timing isn't a real answer.
    ok = [i for i in items if i["predicted"] is not None]
    pick = lambda key: [i.get(key) for i in ok]
    m["latency_ms"] = {
        "ttft_p50": percentile(pick("ttft_ms"), 0.50),
        "ttft_p95": percentile(pick("ttft_ms"), 0.95),
        "total_p50": percentile(pick("total_ms"), 0.50),
        "total_p95": percentile(pick("total_ms"), 0.95),
        "client_total_p50": percentile(pick("client_total_ms"), 0.50),
        "client_total_p95": percentile(pick("client_total_ms"), 0.95),
    }

    costs = [i["cost_usd"] for i in items if i.get("cost_usd") is not None]
    avg = sum(costs) / len(costs) if costs else None
    m["cost"] = {
        "avg_per_task_usd": avg,
        "per_1m_tasks_usd": avg * 1_000_000 if avg is not None else None,
    }
    return m


def format_report(model_name, m):
    pct = lambda x: "  n/a" if x is None else f"{x*100:5.1f}%"
    lines = [
        f"=== Eval report: {model_name} ({m['n']} cases) ===",
        f"Accuracy                   {pct(m['accuracy'])}",
        f"Valid-label rate           {pct(m['valid_label_rate'])}",
        f"Error rate                 {pct(m['error_rate'])}",
        f"payment_failed recall      {pct(m['critical_recall'])}   <- business-critical",
        "",
        f"{'label':<20}{'support':>8}{'recall':>9}{'precision':>11}",
    ]
    for label, s in m["per_label"].items():
        lines.append(
            f"{label:<20}{s['support']:>8}{pct(s['recall']):>9}{pct(s['precision']):>11}"
        )
    if m["confusion"]:
        lines += ["", "Most common mistakes (true -> predicted):"]
        for true_label, preds in m["confusion"].items():
            for pred, count in preds.items():
                lines.append(f"  {true_label} -> {pred}: {count}")

    ms = lambda x: "n/a" if x is None else f"{x:.0f} ms"
    lat = m.get("latency_ms")
    if lat:
        lines += [
            "",
            f"TTFT  p50 {ms(lat['ttft_p50'])} | p95 {ms(lat['ttft_p95'])}",
            f"Total p50 {ms(lat['total_p50'])} | p95 {ms(lat['total_p95'])}",
        ]
    cost = m.get("cost")
    if cost and cost["avg_per_task_usd"] is not None:
        lines.append(
            f"Cost  ${cost['avg_per_task_usd']:.6f} per task | "
            f"${cost['per_1m_tasks_usd']:,.0f} per 1M tasks"
        )
    return "\n".join(lines)
