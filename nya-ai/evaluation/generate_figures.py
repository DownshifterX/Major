"""
evaluation/generate_figures.py

Standalone figure generator — reads actual result JSON/CSV files
and produces charts. Can be run independently after run_evaluation.py.

Usage:
  python evaluation/generate_figures.py

All chart values come from evaluation/results/ files.
No hardcoded values.
"""
import os
import sys
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  SKIP: {name} not found")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_all():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    # ---- Figure 1: Retrieval Method Comparison ----
    ret = _load_json("retrieval_metrics.json")
    if ret and "baselines" in ret:
        baselines = list(ret["baselines"].keys())
        metrics_cfg = [
            ("mrr", "MRR"),
            ("precision_at_1", "Precision@1"),
            ("precision_at_3", "Precision@3"),
            ("precision_at_5", "Precision@5"),
            ("precision_at_10", "Precision@10"),
            ("recall_at_5", "Recall@5"),
            ("recall_at_10", "Recall@10"),
            ("ndcg_at_5", "NDCG@5"),
            ("ndcg_at_10", "NDCG@10"),
        ]

        # Main comparison: MRR, P@5, NDCG@5 side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (metric, title) in zip(axes, [("mrr","MRR"), ("precision_at_5","P@5"), ("ndcg_at_5","NDCG@5")]):
            vals = []
            for b in baselines:
                v = ret["baselines"][b].get(metric, 0)
                vals.append(float(v) if isinstance(v, (int, float)) else 0.0)
            bars = ax.bar(baselines, vals, color=COLORS[:len(baselines)], edgecolor="white", linewidth=0.8)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylim(0, min(1.0, max(vals) * 1.25 + 0.05) if any(vals) else 1.0)
            ax.set_xticks(range(len(baselines)))
            ax.set_xticklabels([b.replace("_", "\n") for b in baselines], fontsize=8)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.suptitle("Retrieval Method Comparison  [[PROXY] PROXY LABELS]", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "retrieval_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] retrieval_comparison.png")

        # Precision@K curve per baseline
        fig, ax = plt.subplots(figsize=(10, 5))
        k_vals = [1, 3, 5, 10]
        metric_keys = ["precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10"]
        for i, (b, color) in enumerate(zip(baselines, COLORS)):
            y = [float(ret["baselines"][b].get(mk, 0) or 0) for mk in metric_keys]
            ax.plot(k_vals, y, "o-", color=color, label=b, linewidth=2, markersize=7)
        ax.set_xlabel("K")
        ax.set_ylabel("Precision@K")
        ax.set_title("Precision@K Comparison  [[PROXY] PROXY LABELS]", fontsize=13, fontweight="bold")
        ax.set_xticks(k_vals)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "precision_at_k.png", dpi=150)
        plt.close()
        print("  [OK] precision_at_k.png")

        # Recall@K
        fig, ax = plt.subplots(figsize=(8, 5))
        recall_ks = [5, 10]
        recall_keys = ["recall_at_5", "recall_at_10"]
        x = np.arange(len(recall_ks))
        w = 0.2
        for i, (b, color) in enumerate(zip(baselines, COLORS)):
            y = [float(ret["baselines"][b].get(mk, 0) or 0) for mk in recall_keys]
            ax.bar(x + i * w, y, w, label=b, color=color, edgecolor="white")
        ax.set_xticks(x + w * (len(baselines)-1) / 2)
        ax.set_xticklabels(["Recall@5", "Recall@10"])
        ax.set_ylabel("Recall")
        ax.set_title("Recall@K Comparison  [[PROXY] PROXY LABELS]", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "recall_at_k.png", dpi=150)
        plt.close()
        print("  [OK] recall_at_k.png")

        # MRR comparison bar
        fig, ax = plt.subplots(figsize=(8, 4))
        mrr_vals = [float(ret["baselines"][b].get("mrr", 0) or 0) for b in baselines]
        bars = ax.barh(baselines, mrr_vals, color=COLORS[:len(baselines)], edgecolor="white")
        ax.set_xlabel("MRR")
        ax.set_title("Mean Reciprocal Rank  [[PROXY] PROXY LABELS]", fontsize=13, fontweight="bold")
        ax.set_xlim(0, max(mrr_vals) * 1.25 + 0.05 if any(mrr_vals) else 1.0)
        for bar, val in zip(bars, mrr_vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2.,
                    f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "mrr_comparison.png", dpi=150)
        plt.close()
        print("  [OK] mrr_comparison.png")

        # NDCG comparison
        fig, ax = plt.subplots(figsize=(8, 4))
        ndcg5 = [float(ret["baselines"][b].get("ndcg_at_5", 0) or 0) for b in baselines]
        ndcg10 = [float(ret["baselines"][b].get("ndcg_at_10", 0) or 0) for b in baselines]
        x = np.arange(len(baselines))
        ax.bar(x - 0.2, ndcg5, 0.4, label="NDCG@5", color=COLORS[0], edgecolor="white")
        ax.bar(x + 0.2, ndcg10, 0.4, label="NDCG@10", color=COLORS[1], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", "\n") for b in baselines], fontsize=8)
        ax.set_ylabel("NDCG")
        ax.set_title("NDCG@K Comparison  [[PROXY] PROXY LABELS]", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ndcg_comparison.png", dpi=150)
        plt.close()
        print("  [OK] ndcg_comparison.png")

    # ---- Figure 2: Ablation Curve ----
    abl = _load_json("ablation_results.json")
    if abl and "ablation_configs" in abl:
        configs = list(abl["ablation_configs"].keys())
        labels = [c.split("_")[0] for c in configs]  # A0, A1, ...

        fig, ax = plt.subplots(figsize=(11, 5))
        for metric, color, marker in [("mrr", COLORS[0], "o"), ("precision_at_5", COLORS[1], "s"),
                                       ("ndcg_at_5", COLORS[2], "^"), ("hit_rate_at_5", COLORS[3], "D")]:
            vals = [float(abl["ablation_configs"][c].get(metric, 0) or 0) for c in configs]
            ax.plot(labels, vals, f"{marker}-", color=color, label=metric.upper().replace("_AT_","@"),
                    linewidth=2, markersize=8)
            for xi, (lbl, val) in enumerate(zip(labels, vals)):
                ax.annotate(f"{val:.3f}", (xi, val), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color=color)

        ax.set_xlabel("Ablation Configuration")
        ax.set_ylabel("Score")
        ax.set_title("Ablation Study: Signal Contribution  [[PROXY] PROXY LABELS]",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ablation_curve.png", dpi=150)
        plt.close()
        print("  [OK] ablation_curve.png")

    # ---- Figure 3: Weight Sensitivity ----
    ws = _load_json("weight_sensitivity.json")
    if ws and "configs" in ws:
        ws_configs = list(ws["configs"].keys())
        short_labels = [c.split("_")[1].upper() if "_" in c else c for c in ws_configs]

        fig, ax = plt.subplots(figsize=(10, 5))
        for metric, color, marker in [("mrr", COLORS[0], "o"), ("precision_at_5", COLORS[1], "s")]:
            vals = [float(ws["configs"][c].get(metric, 0) or 0) for c in ws_configs]
            ax.plot(short_labels, vals, f"{marker}-", color=color,
                    label=metric.replace("_at_","@").upper(), linewidth=2, markersize=8)
        ax.set_xlabel("Weight Configuration")
        ax.set_ylabel("Score")
        ax.set_title("Weight Sensitivity Analysis  [[PROXY] PROXY LABELS]",
                     fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "weight_sensitivity.png", dpi=150)
        plt.close()
        print("  [OK] weight_sensitivity.png")

    # ---- Figure 4: Performance Latency ----
    perf = _load_json("performance.json")
    if perf:
        stages = [r["stage"] for r in perf]
        means = [r["mean_ms"] for r in perf]
        p95s = [r["p95_ms"] for r in perf]

        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(stages))
        ax.bar(x - 0.2, means, 0.4, label="Mean (ms)", color=COLORS[0], edgecolor="white")
        ax.bar(x + 0.2, p95s, 0.4, label="P95 (ms)", color=COLORS[1], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in stages], fontsize=8)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Pipeline Stage Latency Benchmark", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "latency_comparison.png", dpi=150)
        plt.close()
        print("  [OK] latency_comparison.png")


if __name__ == "__main__":
    print("Generating evaluation figures from results/...")
    generate_all()
    print(f"\nFigures saved to: {FIGURES_DIR}")
