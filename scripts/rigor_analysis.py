"""
Rigor Analysis for ML4Env Critical Review — ES&T Revision.

Computes per-paper methodological rigor scores (0–9) and analyzes
the relationship between rigor and reported R² values.

Addresses ES&T Editor Comment 1: replace inferred inflation claim
with internal quantitative evidence.

Usage:
    python scripts/rigor_analysis.py
    python scripts/rigor_analysis.py --print   # print key results
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "public"
DATASET_FILE = DATA_DIR / "ml4env_126_dataset.json"
OUTPUT_FILE = DATA_DIR / "rigor_analysis.json"
FIGURE_DATA_FILE = DATA_DIR / "figure_data" / "fig6_rigor.json"

# ============================================================
# Rigor Dimensions (9 binary indicators)
# ============================================================
RIGOR_DIMENSIONS = {
    "grouped_splitting": {
        "label": "Grouped splitting",
        "short": "GS",
        "extract": lambda p: bool(p.get("grouped_splitting")),
    },
    "external_validation": {
        "label": "External validation",
        "short": "EV",
        "extract": lambda p: bool(p.get("external_validation")),
    },
    "reports_train_and_test": {
        "label": "Train+test reported",
        "short": "TT",
        "extract": lambda p: bool(p.get("reports_train_and_test")),
    },
    "hpo_reported": {
        "label": "HPO reported",
        "short": "HPO",
        "extract": lambda p: (
            p.get("hyperparameter_tuning") is not None
            and p.get("hyperparameter_tuning") != "none_reported"
        ),
    },
    "selection_criteria": {
        "label": "Selection criteria",
        "short": "SC",
        "extract": lambda p: bool(p.get("data_selection_criteria_described")),
    },
    "code_available": {
        "label": "Code available",
        "short": "Code",
        "extract": lambda p: bool(p.get("code_available")),
    },
    "data_available": {
        "label": "Data available",
        "short": "Data",
        "extract": lambda p: bool(p.get("data_available")),
    },
    "adequate_size": {
        "label": "Dataset ≥ 500",
        "short": "N≥500",
        "extract": lambda p: (
            p.get("dataset_size") is not None
            and p["dataset_size"] >= 500
        ),
    },
    "kfold_or_better": {
        "label": "k-fold or better",
        "short": "kCV",
        "extract": lambda p: p.get("validation_method") in (
            "k_fold", "leave_one_out", "nested_cv"
        ),
    },
}


def load_data():
    with open(DATASET_FILE, encoding="utf-8") as f:
        return json.load(f)


def compute_rigor(papers):
    """Compute rigor scores and per-dimension flags for each paper."""
    results = []
    for p in papers:
        # Get R² value (only R2 type, not R)
        r2 = None
        if p.get("best_metric_type") == "R2" and p.get("best_metric_value") is not None:
            r2 = p["best_metric_value"]

        flags = {}
        score = 0
        for dim_key, dim_def in RIGOR_DIMENSIONS.items():
            val = dim_def["extract"](p)
            flags[dim_key] = val
            if val:
                score += 1

        results.append({
            "paper_id": p["paper_id"],
            "paper_title": p.get("paper_title", ""),
            "r2": r2,
            "rigor_score": score,
            "flags": flags,
        })
    return results


def analyze_rigor_vs_r2(rigor_data):
    """Main analysis: rigor score vs R² stratified analysis."""

    # Filter to papers with valid R²
    valid = [r for r in rigor_data if r["r2"] is not None]
    print(f"\nPapers with valid R²: {len(valid)} / {len(rigor_data)}")

    # --- 1. Overall rigor score distribution ---
    scores = [r["rigor_score"] for r in valid]
    r2_vals = [r["r2"] for r in valid]
    print(f"Rigor score: median={np.median(scores):.1f}, "
          f"mean={np.mean(scores):.2f}, range={min(scores)}-{max(scores)}")

    # --- 2. Rigor score groups vs median R² ---
    groups = {
        "0-2": [], "3-4": [], "5-6": [], "7-9": []
    }
    for r in valid:
        s = r["rigor_score"]
        if s <= 2:
            groups["0-2"].append(r["r2"])
        elif s <= 4:
            groups["3-4"].append(r["r2"])
        elif s <= 6:
            groups["5-6"].append(r["r2"])
        else:
            groups["7-9"].append(r["r2"])

    print("\n=== Rigor Score Groups vs R² ===")
    group_stats = {}
    for g_name, g_vals in groups.items():
        if g_vals:
            group_stats[g_name] = {
                "n": len(g_vals),
                "median_r2": float(np.median(g_vals)),
                "mean_r2": float(np.mean(g_vals)),
                "q25_r2": float(np.percentile(g_vals, 25)),
                "q75_r2": float(np.percentile(g_vals, 75)),
                "r2_values": sorted(g_vals),
            }
            print(f"  {g_name}: n={len(g_vals)}, "
                  f"median R²={np.median(g_vals):.3f}, "
                  f"mean R²={np.mean(g_vals):.3f}")
        else:
            group_stats[g_name] = {"n": 0}
            print(f"  {g_name}: n=0")

    # Spearman correlation: rigor score vs R²
    rho, p_val = stats.spearmanr(scores, r2_vals)
    print(f"\nSpearman correlation: ρ={rho:.3f}, p={p_val:.4f}")

    # --- 3. Per-dimension analysis ---
    print("\n=== Per-Dimension R² Comparison ===")
    dim_results = {}
    for dim_key, dim_def in RIGOR_DIMENSIONS.items():
        yes_r2 = [r["r2"] for r in valid if r["flags"][dim_key]]
        no_r2 = [r["r2"] for r in valid if not r["flags"][dim_key]]

        if len(yes_r2) >= 2 and len(no_r2) >= 2:
            u_stat, u_pval = stats.mannwhitneyu(
                yes_r2, no_r2, alternative='two-sided'
            )
            median_diff = float(np.median(yes_r2) - np.median(no_r2))

            # Bootstrap 95% CI for median difference
            n_boot = 5000
            boot_diffs = []
            rng = np.random.default_rng(42)
            for _ in range(n_boot):
                y_sample = rng.choice(yes_r2, size=len(yes_r2), replace=True)
                n_sample = rng.choice(no_r2, size=len(no_r2), replace=True)
                boot_diffs.append(float(np.median(y_sample) - np.median(n_sample)))
            ci_lower = float(np.percentile(boot_diffs, 2.5))
            ci_upper = float(np.percentile(boot_diffs, 97.5))
        else:
            u_stat, u_pval = None, None
            median_diff = None
            ci_lower, ci_upper = None, None

        dim_results[dim_key] = {
            "label": dim_def["label"],
            "short": dim_def["short"],
            "n_yes": len(yes_r2),
            "n_no": len(no_r2),
            "median_yes": float(np.median(yes_r2)) if yes_r2 else None,
            "median_no": float(np.median(no_r2)) if no_r2 else None,
            "median_diff": median_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mann_whitney_p": float(u_pval) if u_pval is not None else None,
        }

        sig = "*" if u_pval is not None and u_pval < 0.05 else ""
        med_yes = f"{np.median(yes_r2):.3f}" if yes_r2 else "N/A"
        med_no = f"{np.median(no_r2):.3f}" if no_r2 else "N/A"
        diff_str = f"{median_diff:+.3f}" if median_diff is not None else "N/A"
        p_str = f"{u_pval:.4f}" if u_pval is not None else "N/A"
        print(f"  {dim_def['label']:25s}: "
              f"YES (n={len(yes_r2):3d}) median={med_yes}, "
              f"NO (n={len(no_r2):3d}) median={med_no}, "
              f"Δ={diff_str}, p={p_str}{sig}")

    # --- 4. Heatmap data (papers sorted by R² descending) ---
    sorted_papers = sorted(valid, key=lambda r: r["r2"], reverse=True)
    heatmap_data = {
        "paper_ids": [r["paper_id"] for r in sorted_papers],
        "r2_values": [r["r2"] for r in sorted_papers],
        "rigor_scores": [r["rigor_score"] for r in sorted_papers],
        "dimensions": list(RIGOR_DIMENSIONS.keys()),
        "dimension_labels": [d["label"] for d in RIGOR_DIMENSIONS.values()],
        "dimension_shorts": [d["short"] for d in RIGOR_DIMENSIONS.values()],
        "matrix": [
            [r["flags"][dim] for dim in RIGOR_DIMENSIONS]
            for r in sorted_papers
        ],
    }

    return {
        "n_valid": len(valid),
        "n_total": len(rigor_data),
        "overall": {
            "rigor_median": float(np.median(scores)),
            "rigor_mean": float(np.mean(scores)),
            "rigor_min": int(min(scores)),
            "rigor_max": int(max(scores)),
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
        },
        "group_stats": group_stats,
        "dimension_results": dim_results,
        "heatmap_data": heatmap_data,
    }


def sensitivity_leave_one_out(rigor_data):
    """Leave-one-dimension-out sensitivity analysis.

    For each of the 9 dimensions, recompute rigor score excluding that
    dimension (score 0-8) and recalculate Spearman correlation with R².
    """
    valid = [r for r in rigor_data if r["r2"] is not None]
    dims = list(RIGOR_DIMENSIONS.keys())

    print("\n=== Leave-One-Dimension-Out Sensitivity Analysis ===")
    print(f"{'Excluded dimension':30s} {'ρ':>8s} {'p-value':>10s}")
    print("-" * 52)

    results = []
    for exclude_dim in dims:
        # Recompute scores without this dimension
        scores = []
        r2_vals = []
        for r in valid:
            new_score = sum(
                r["flags"][d] for d in dims if d != exclude_dim
            )
            scores.append(new_score)
            r2_vals.append(r["r2"])

        rho, p_val = stats.spearmanr(scores, r2_vals)
        label = RIGOR_DIMENSIONS[exclude_dim]["label"]
        print(f"  {label:28s} {rho:8.3f} {p_val:10.4f}")
        results.append({
            "excluded": exclude_dim,
            "label": label,
            "rho": float(rho),
            "p_value": float(p_val),
        })

    rhos = [r["rho"] for r in results]
    print(f"\n  ρ range: {min(rhos):.3f} to {max(rhos):.3f}")
    print(f"  All p < 0.05: {all(r['p_value'] < 0.05 for r in results)}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", default=True)
    args = parser.parse_args()

    papers = load_data()
    print(f"Loaded {len(papers)} papers")

    rigor_data = compute_rigor(papers)
    results = analyze_rigor_vs_r2(rigor_data)

    # Sensitivity analysis
    sensitivity = sensitivity_leave_one_out(rigor_data)
    results["sensitivity_leave_one_out"] = sensitivity

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    # Save figure data
    FIGURE_DATA_FILE.parent.mkdir(exist_ok=True)
    with open(FIGURE_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {FIGURE_DATA_FILE}")

    # Also save per-paper rigor data for SI Table
    per_paper = [{
        "paper_id": r["paper_id"],
        "rigor_score": r["rigor_score"],
        "r2": r["r2"],
        **{k: r["flags"][k] for k in RIGOR_DIMENSIONS},
    } for r in rigor_data]

    csv_file = DATA_DIR / "rigor_scores_per_paper.csv"
    import csv
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_paper[0].keys())
        writer.writeheader()
        writer.writerows(per_paper)
    print(f"Saved: {csv_file}")


if __name__ == "__main__":
    main()
