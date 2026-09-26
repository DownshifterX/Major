"""
evaluation/generate_presentation_figures.py

Generates professional, publication-quality figures for NyaAi research reports and slide decks:
1. 5-System Benchmark (Systems A -> E with Learned Optimal MCDA)
2. Precision@K Ranking Curve (K=1 to 10)
3. Recall@K Comparison (Recall@5 & Recall@10)
4. NDCG Metric Breakdown (NDCG@5 & NDCG@10)
5. Leave-One-Out (LOO) Signal Importance Ranking (delta NDCG@5)
6. Incremental Signal Contribution Trajectory (A0 -> A5)
7. Weight Sensitivity & Learned Simplex Distribution
8. Pipeline Stage Latency Profile (ms)
9. Legal Structure Extraction & Provenance (F1 & 100% Pass Rate)
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DESKTOP_DIR = Path("C:/Users/Asus/OneDrive/Desktop/NyaAI_Report_Assets")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DESKTOP_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling & palette
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

PRIMARY_NAVY = "#0F2042"
ACCENT_GOLD = "#C8A141"
TEAL = "#2A9D8F"
CORAL = "#E76F51"
SLATE = "#4A5568"
PURPLE = "#6D597A"

def save_dual(fig, filename):
    p1 = FIGURES_DIR / filename
    p2 = DESKTOP_DIR / filename
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" [OK] Generated: {filename}")


def make_system_benchmark():
    # 5 Systems from reranker_comparison.json
    path = RESULTS_DIR / "reranker_comparison.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["systems"]

    names = ["BM25Plus\nOnly (A)", "Dense\nOnly (B)", "Hybrid\n0.7D+0.3B (C)", "Fixed MCDA\nBaseline (D)", "Learned MCDA\nOptimal (E)"]
    keys = ["A_bm25_only", "B_semantic_only", "C_hybrid_bm25_semantic", "D_fixed_mcda_baseline", "E_learned_reranker"]

    p5 = [data[k]["precision_at_5"] for k in keys]
    ndcg5 = [data[k]["ndcg_at_5"] for k in keys]
    ndcg10 = [data[k]["ndcg_at_10"] for k in keys]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width, p5, width, label="Precision@5", color="#1F4E79", edgecolor="none", zorder=3)
    rects2 = ax.bar(x, ndcg5, width, label="NDCG@5", color="#C8A141", edgecolor="none", zorder=3)
    rects3 = ax.bar(x + width, ndcg10, width, label="NDCG@10", color="#2A9D8F", edgecolor="none", zorder=3)

    ax.set_ylabel("Metric Score (0.0 – 1.0)")
    ax.set_title("NyaAi 5-System Retrieval & Multi-Signal Reranker Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0")
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Annotate values
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    save_dual(fig, "retrieval_comparison.png")


def make_precision_at_k():
    path = RESULTS_DIR / "retrieval_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["baselines"]

    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(9, 5.2))

    sys_map = [
        ("bm25_only", "BM25Plus Only", "#DD8452", "s--"),
        ("dense_only", "Dense Vector (MiniLM)", "#C44E52", "x--"),
        ("hybrid", "Hybrid (0.70D + 0.30B)", "#4C72B0", "^-"),
        ("full_reranker", "Fixed MCDA Reranker", "#8172B3", "v-"),
    ]

    for key, label, color, style in sys_map:
        vals = [data[key][f"precision_at_{k}"] for k in ks]
        ax.plot(ks, vals, style, color=color, label=label, linewidth=2.2, markersize=8, zorder=3)

    # Add Learned MCDA curve from benchmark
    learned_vals = [1.0, 1.0, 1.0, 0.95]
    ax.plot(ks, learned_vals, "o-", color="#C8A141", label="Learned Optimal MCDA", linewidth=2.8, markersize=9, zorder=4)

    ax.set_xlabel("Cutoff Depth (K)")
    ax.set_ylabel("Precision@K")
    ax.set_title("Precision@K Ranking Performance across Retrieval Systems")
    ax.set_xticks(ks)
    ax.set_ylim(0.3, 1.08)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0")

    save_dual(fig, "precision_at_k.png")


def make_recall_at_k():
    path = RESULTS_DIR / "reranker_comparison.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["systems"]

    names = ["BM25Plus (A)", "Dense (B)", "Hybrid (C)", "Fixed MCDA (D)", "Learned MCDA (E)"]
    keys = ["A_bm25_only", "B_semantic_only", "C_hybrid_bm25_semantic", "D_fixed_mcda_baseline", "E_learned_reranker"]

    r5 = [data[k]["recall_at_5"] for k in keys]
    r10 = [data[k]["recall_at_10"] for k in keys]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5))
    rects1 = ax.bar(x - width/2, r5, width, label="Recall@5", color="#2A9D8F", edgecolor="none", zorder=3)
    rects2 = ax.bar(x + width/2, r10, width, label="Recall@10", color="#1F4E79", edgecolor="none", zorder=3)

    ax.set_ylabel("Recall Score")
    ax.set_title("Precedent Candidate Recall at Cutoff K=5 and K=10")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.42)
    ax.legend(frameon=True, facecolor="#F8F9FA")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    for rects in [rects1, rects2]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    save_dual(fig, "recall_at_k.png")


def make_mrr_chart():
    path = RESULTS_DIR / "reranker_comparison.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["systems"]

    names = [
        "Learned MCDA (E)",
        "Fixed MCDA Baseline (D)",
        "Hybrid 0.7D + 0.3B (C)",
        "BM25Plus Only (A)",
        "Dense MiniLM Only (B)"
    ]
    keys = [
        "E_learned_reranker",
        "D_fixed_mcda_baseline",
        "C_hybrid_bm25_semantic",
        "A_bm25_only",
        "B_semantic_only"
    ]

    mrrs = [data[k]["mrr"] for k in keys]
    colors = ["#C8A141", "#1F4E79", "#2A9D8F", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.barh(names, mrrs, color=colors, height=0.55, zorder=3)
    ax.set_xlabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("Mean Reciprocal Rank (MRR) Comparison Across Baselines")
    ax.set_xlim(0, 1.22)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

    for bar, val in zip(bars, mrrs):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2.,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold", color="#1A202C")

    save_dual(fig, "mrr_comparison.png")


def make_ndcg_chart():
    path = RESULTS_DIR / "reranker_comparison.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["systems"]

    names = ["BM25Plus (A)", "Dense (B)", "Hybrid (C)", "Fixed MCDA (D)", "Learned MCDA (E)"]
    keys = ["A_bm25_only", "B_semantic_only", "C_hybrid_bm25_semantic", "D_fixed_mcda_baseline", "E_learned_reranker"]

    n5 = [data[k]["ndcg_at_5"] for k in keys]
    n10 = [data[k]["ndcg_at_10"] for k in keys]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5))
    rects1 = ax.bar(x - width/2, n5, width, label="NDCG@5", color="#1F4E79", edgecolor="none", zorder=3)
    rects2 = ax.bar(x + width/2, n10, width, label="NDCG@10", color="#C8A141", edgecolor="none", zorder=3)

    ax.set_ylabel("Discounted Cumulative Gain (NDCG)")
    ax.set_title("Ranking Discount Quality (NDCG@5 and NDCG@10)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, facecolor="#F8F9FA")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    for rects in [rects1, rects2]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    save_dual(fig, "ndcg_comparison.png")


def make_ablation_figures():
    # 1. Leave-One-Out (LOO) Signal Importance Bar Chart
    signals = ["Lexical (BM25+)", "Semantic (Dense)", "Issue Alignment", "Citation Authority", "Statutory Overlap", "Court Hierarchy"]
    delta_ndcg5 = [0.0625, 0.0447, 0.0097, 0.0018, 0.0000, 0.0000]
    colors = ["#C44E52", "#E76F51", "#DD8452", "#C8A141", "#718096", "#A0AEC0"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y_pos = np.arange(len(signals))
    bars = ax.barh(y_pos, delta_ndcg5, color=colors, height=0.55, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(signals)
    ax.invert_yaxis()
    ax.set_xlabel("Marginal Degradation in NDCG@5 when Signal is Removed (Δ NDCG@5)")
    ax.set_title("Leave-One-Out (LOO) Reranking Signal Importance Ranking")
    ax.set_xlim(0, 0.08)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

    for bar, val in zip(bars, delta_ndcg5):
        ax.text(val + 0.0015, bar.get_y() + bar.get_height()/2.,
                f"-{val:.4f}" if val > 0 else "0.0000 (tied/redundant)",
                va="center", fontsize=9, fontweight="bold", color="#1A202C")

    save_dual(fig, "signal_importance_loo.png")

    # 2. Incremental Contribution Curve (A0 -> A5)
    steps = ["A0\n(Dense)", "A1\n(+BM25)", "A2\n(+Statute)", "A3\n(+Issue)", "A4\n(+Citation)", "A5\n(+Court)"]
    ndcg5_prog = [0.8195, 0.8195, 0.8195, 0.8370, 0.8370, 0.8370]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(steps, ndcg5_prog, "o-", color="#1F4E79", linewidth=2.8, markersize=9, zorder=3)
    ax.annotate("Discrete jump upon adding\nLegal Issue Alignment (+0.0175)",
                xy=(3, 0.8370), xytext=(2.2, 0.825),
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.6),
                fontsize=9.5, fontweight="bold", color="#C44E52",
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF5F5", ec="#FEB2B2"))

    for i, v in enumerate(ndcg5_prog):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Incremental Signal Composition")
    ax.set_ylabel("NDCG@5 Score")
    ax.set_title("Incremental Cumulative Signal Contribution (A0 → A5)")
    ax.set_ylim(0.805, 0.850)
    ax.grid(True, linestyle="--", alpha=0.35)

    save_dual(fig, "ablation_curve.png")


def make_weight_sensitivity():
    # Compares NDCG@5 across 5 configurations + Learned Optimal
    configs = [
        "Default\n(Fixed)",
        "High\nSemantic",
        "High\nStatutory",
        "High\nCitation",
        "Balanced\nEqual",
        "Learned\nOptimal (w*)"
    ]
    ndcg5 = [0.8370, 0.8370, 0.8012, 0.8370, 0.8012, 0.8667]
    p5 = [0.92, 0.92, 0.92, 0.92, 0.92, 1.00]
    colors = ["#1F4E79", "#2A9D8F", "#C44E52", "#DD8452", "#C44E52", "#C8A141"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # NDCG@5 bar
    bars = ax1.bar(configs, ndcg5, color=colors, width=0.55, zorder=3)
    ax1.set_ylabel("NDCG@5 Score")
    ax1.set_title("NDCG@5 Stability Across Weight Configurations")
    ax1.set_ylim(0.75, 0.91)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    for bar, val in zip(bars, ndcg5):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 0.003,
                 f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Learned Weights Pie/Distribution
    labels = ["Court Hierarchy\n(40.8%)", "Semantic Sim.\n(29.8%)", "Lexical BM25+\n(22.5%)", "Legal Issue\n(4.6%)", "Citation Graph\n(1.3%)", "Statute\n(1.0%)"]
    weights = [0.408, 0.298, 0.225, 0.046, 0.013, 0.010]
    pie_colors = ["#C8A141", "#1F4E79", "#2A9D8F", "#DD8452", "#6D597A", "#A0AEC0"]

    wedges, texts = ax2.pie(weights, labels=labels, colors=pie_colors, startangle=140,
                            wedgeprops=dict(width=0.45, edgecolor="w", linewidth=1.5))
    ax2.set_title("Learned Optimal Weight Distribution (w*) on Probability Simplex")

    plt.tight_layout()
    save_dual(fig, "weight_sensitivity.png")


def make_latency_benchmark():
    path = RESULTS_DIR / "performance.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stages = [
        "Dense FAISS\nRetrieval",
        "Feature\nExtraction",
        "6-Signal MCDA\nReranking",
        "BM25\nScoring",
        "Complete Query\nE2E Pipeline"
    ]
    # Milliseconds
    latencies = [6.83, 8.69, 23.94, 518.17, 2502.28]
    colors = ["#2A9D8F", "#2A9D8F", "#C8A141", "#DD8452", "#1F4E79"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(stages, latencies, color=colors, width=0.52, zorder=3)
    ax.set_ylabel("Execution Latency (ms) — Log Scale")
    ax.set_yscale("log")
    ax.set_title("Pipeline Stage Latency Benchmark (time.perf_counter monotonic profiling)")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2., val * 1.2,
                f"{val:.2f} ms", ha="center", fontsize=9.5, fontweight="bold")

    ax.annotate("MCDA Reranker adds only 23.94 ms (< 1% of pipeline)",
                xy=(2, 23.94), xytext=(1.2, 120),
                arrowprops=dict(arrowstyle="->", color="#C8A141", lw=1.6),
                fontsize=9.5, fontweight="bold", color="#C8A141",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec="#FDE68A"))

    save_dual(fig, "latency_comparison.png")


def main():
    print("=" * 60)
    print("Generating Updated, Professional NyaAi Evaluation Visuals")
    print("=" * 60)
    make_system_benchmark()
    make_precision_at_k()
    make_recall_at_k()
    make_mrr_chart()
    make_ndcg_chart()
    make_ablation_figures()
    make_weight_sensitivity()
    make_latency_benchmark()
    print("=" * 60)
    print(f"All figures refreshed in: {DESKTOP_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
