"""
Evals for the ticket-triage task.

An eval = a test set + a scoring rule + a report.
- Test set:     spottotv_tickets_test_set.csv (input + expected label)
- Scoring rule: exact match on the label (a "code-based" eval:
                cheap, deterministic, perfect for classification)
- Report:       the metrics below

Each metric answers a different product question.
"""

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
            confusion[r["expected"]][r["predicted"]] += 1

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
    return "\n".join(lines)
