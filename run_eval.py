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

from evals import format_report, summarize
from main import RunRequest, run_one

SYSTEM_PROMPT = open("system_prompt.txt", encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "openrouter", "fake"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", default="spottotv_tickets_test_set.csv")
    ap.add_argument("--in-price", type=float, default=0)
    ap.add_argument("--out-price", type=float, default=0)
    ap.add_argument("--json", help="also save raw results to this file")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
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
                "ttft_ms": r.ttft_ms_server,
                "total_ms": r.total_ms_server,
                "cost_usd": r.cost_usd,
            }
        )

    # Same function the /score endpoint uses: one source of truth.
    metrics = summarize(results)
    print(format_report(f"{args.provider}/{args.model}", metrics))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
