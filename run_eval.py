"""
Run the ticket-triage eval from the command line.

Examples:
  python run_eval.py --provider fake --model payment_failed
  python run_eval.py --provider anthropic --model claude-haiku-5-5 \
      --in-price 1.0 --out-price 5.0

Needs ANTHROPIC_API_KEY / OPENAI_API_KEY for real providers.
"""

import argparse
import csv
import json
import statistics

from evals import format_report, score
from main import RunRequest, run_one

SYSTEM_PROMPT = open("system_prompt.txt", encoding="utf-8").read()


def p95(values):
    values = sorted(values)
    return values[max(0, int(round(0.95 * len(values))) - 1)] if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "fake"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", default="spottotv_tickets_test_set.csv")
    ap.add_argument("--in-price", type=float, default=0)
    ap.add_argument("--out-price", type=float, default=0)
    ap.add_argument("--json", help="also save raw results to this file")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results, ttfts, costs = [], [], []
    for row in rows:
        r = run_one(
            RunRequest(
                provider=args.provider,
                model_id=args.model,
                system_prompt=SYSTEM_PROMPT,
                input_text=row["input_text"],
                expected_label=row["expected_label"],
                input_price_per_million=args.in_price,
                output_price_per_million=args.out_price,
            )
        )
        results.append(
            {
                "input": row["input_text"],
                "expected": row["expected_label"],
                "predicted": r.parsed_label if not r.error else None,
                "error": r.error,
            }
        )
        if r.ttft_ms_server is not None and not r.error:
            ttfts.append(r.ttft_ms_server)
        costs.append(r.cost_usd)

    metrics = score(results)
    print(format_report(f"{args.provider}/{args.model}", metrics))
    if ttfts:
        print(f"\nTTFT p50 {statistics.median(ttfts):.0f} ms | p95 {p95(ttfts):.0f} ms")
    print(f"Avg cost per task ${sum(costs)/len(costs):.6f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
