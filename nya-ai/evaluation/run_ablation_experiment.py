"""evaluation/run_ablation_experiment.py
Runs the Leave-One-Out Signal Contribution Ablation Experiment (Section 14).

Computes:
    Delta NDCG@10(signal) = NDCG@10(all signals) - NDCG@10(without signal)
    Delta NDCG@5(signal)  = NDCG@5(all signals) - NDCG@5(without signal)
    Delta MRR(signal)     = MRR(all signals) - MRR(without signal)

Outputs:
    - evaluation/results/ablation_results.json
    - evaluation/results/ablation_results.csv
"""
import os
import sys
import json
import csv
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
_RESULTS_DIR = _HERE / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_HERE))

from app.database import SessionLocal
from ablation import run_leave_one_out_ablations, LEAVE_ONE_OUT_CONFIGS, LEAVE_ONE_OUT_DESCRIPTIONS
from metrics import compute_all_metrics, aggregate_metrics, compute_leave_one_out_deltas


def run_ablation_experiment(quick: bool = False):
    with open(_HERE / "queries.json", "r", encoding="utf-8") as f:
        all_queries = json.load(f)["queries"]

    with open(_HERE / "ground_truth.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)["ground_truth"]
        ground_truth = {item["query_id"]: item for item in gt_data}

    queries = all_queries[:5] if quick else all_queries

    print(f"\n=======================================================")
    print(f" NYA(AI) LEAVE-ONE-OUT SIGNAL CONTRIBUTION ABLATION")
    print(f"=======================================================")
    print(f" Queries evaluated: {len(queries)}")

    db = SessionLocal()
    try:
        config_metrics = {k: [] for k in LEAVE_ONE_OUT_CONFIGS}

        for i, q in enumerate(queries, 1):
            qid = q["query_id"]
            print(f"  [{i}/{len(queries)}] Ablating query {qid}...")
            gt = ground_truth.get(qid, {})
            relevant = set(gt.get("relevant_cases", []))
            highly = set(gt.get("highly_relevant_cases", []))
            partially = set(gt.get("partially_relevant_cases", []))

            res = run_leave_one_out_ablations(db, qid, q["query_text"])
            for cname, r in res.items():
                m = compute_all_metrics(r.ranked_cases, relevant, highly, partially)
                m["latency_ms"] = r.latency_ms
                config_metrics[cname].append(m)

        aggregated = {}
        for cname in LEAVE_ONE_OUT_CONFIGS:
            agg = aggregate_metrics(config_metrics[cname])
            lat = [m["latency_ms"] for m in config_metrics[cname]]
            agg["avg_latency_ms"] = round(sum(lat) / len(lat), 2) if lat else 0.0
            agg["description"] = LEAVE_ONE_OUT_DESCRIPTIONS[cname]
            aggregated[cname] = agg

        # Compute deltas relative to all_six_signals
        full_baseline = aggregated["all_six_signals"]
        for cname, agg in aggregated.items():
            if cname == "all_six_signals":
                agg["delta_ndcg_at_10"] = 0.0
                agg["delta_ndcg_at_5"] = 0.0
                agg["delta_mrr"] = 0.0
            else:
                deltas = compute_leave_one_out_deltas(full_baseline, agg)
                agg.update(deltas)

        print("\n" + "-" * 90)
        print(f"{'Configuration':<24} | {'MRR':<8} | {'P@5':<8} | {'NDCG@5':<8} | {'NDCG@10':<8} | {'dNDCG@10':<9} | {'dMRR':<6}")
        print("-" * 90)
        for cname, agg in aggregated.items():
            print(
                f"{cname:<24} | "
                f"{agg['mrr']:<8.4f} | "
                f"{agg['precision_at_5']:<8.4f} | "
                f"{agg['ndcg_at_5']:<8.4f} | "
                f"{agg['ndcg_at_10']:<8.4f} | "
                f"{agg['delta_ndcg_at_10']:<9.4f} | "
                f"{agg['delta_mrr']:<6.4f}"
            )
        print("-" * 90 + "\n")

        # Save JSON
        out_payload = {
            "_note": "Leave-One-Out signal contribution ablation (Section 14). Positive delta indicates signal improves performance.",
            "leave_one_out_ablation": aggregated,
        }
        out_file = _RESULTS_DIR / "ablation_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"  Saved ablation results to: {out_file.resolve()}")

        # Save CSV
        csv_file = _RESULTS_DIR / "ablation_results.csv"
        fieldnames = [
            "config", "description", "mrr", "precision_at_5", "recall_at_5",
            "recall_at_10", "ndcg_at_5", "ndcg_at_10", "delta_ndcg_at_10",
            "delta_ndcg_at_5", "delta_mrr", "avg_latency_ms"
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cname, agg in aggregated.items():
                row = {"config": cname}
                row.update({k: agg.get(k, 0.0) for k in fieldnames if k != "config"})
                writer.writerow(row)
        print(f"  Saved ablation CSV to: {csv_file.resolve()}")

        return out_payload
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run leave-one-out signal ablation.")
    parser.add_argument("--quick", action="store_true", help="Quick mode on 5 queries")
    args = parser.parse_args()

    run_ablation_experiment(quick=args.quick)
