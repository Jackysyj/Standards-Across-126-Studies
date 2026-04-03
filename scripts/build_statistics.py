"""
Build aggregated statistics for ML4Env Critical Review.

Single source of truth: computes ALL statistics from ml4env_155_dataset.json,
outputs statistics_summary.json + figure_data/*.json.

Usage:
  python scripts/build_statistics.py
  python scripts/build_statistics.py --print  # print key numbers to console
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "public"
DATASET_FILE = DATA_DIR / "ml4env_126_dataset.json"
OUTPUT_FILE = DATA_DIR / "statistics_summary.json"
FIGURE_DIR = DATA_DIR / "figure_data"

FIGURE_DIR.mkdir(exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_data() -> list[dict]:
    with open(DATASET_FILE, encoding="utf-8") as f:
        return json.load(f)


def safe_values(papers, field):
    """Extract non-null values for a field."""
    return [p[field] for p in papers if p.get(field) is not None]


def pct(count, total):
    return round(count / total, 4) if total > 0 else 0


def counter_to_sorted(counter, top_n=None):
    """Counter → sorted list of [name, count] pairs."""
    items = counter.most_common(top_n)
    return [[k, v] for k, v in items]


def flatten_list_field(papers, field):
    """Flatten a list-type field across all papers."""
    items = []
    for p in papers:
        val = p.get(field)
        if isinstance(val, list):
            items.extend(val)
        elif isinstance(val, str) and val:
            items.extend([x.strip() for x in val.split(";")])
    return items


def quantiles(values):
    """Compute basic quantile stats using numpy for standard median/quartiles."""
    if not values:
        return {}
    import numpy as np
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "min": s[0],
        "q25": float(np.percentile(s, 25)),
        "median": float(np.median(s)),
        "q75": float(np.percentile(s, 75)),
        "max": s[-1],
        "mean": round(sum(s) / n, 2),
    }


# ============================================================
# Statistics Computation
# ============================================================

def compute_introduction(papers):
    n = len(papers)
    years = Counter(p["year"] for p in papers if p.get("year"))
    journals = Counter(p["journal"] for p in papers if p.get("journal"))
    research_types = Counter(p["research_type"] for p in papers if p.get("research_type"))
    oa = sum(1 for p in papers if p.get("is_oa"))
    citations = safe_values(papers, "cited_by_count")

    return {
        "n_papers": n,
        "year_distribution": dict(sorted(years.items())),
        "total_journals": len(journals),
        "top_10_journals": counter_to_sorted(journals, 10),
        "research_type_distribution": dict(research_types),
        "adsorption_count": research_types.get("adsorption", 0),
        "degradation_count": research_types.get("degradation", 0),
        "both_count": research_types.get("both", 0),
        "oa_count": oa,
        "oa_rate": pct(oa, n),
        "citation_stats": quantiles(citations),
    }


def compute_pillar1(papers):
    n = len(papers)
    data_sources = Counter(p["data_source"] for p in papers if p.get("data_source"))
    dataset_sizes = safe_values(papers, "dataset_size")

    # Dataset size by data source
    size_by_source = {}
    for src in ["experimental", "literature", "database", "mixed"]:
        vals = [p["dataset_size"] for p in papers
                if p.get("data_source") == src and p.get("dataset_size") is not None]
        if vals:
            size_by_source[src] = quantiles(vals)

    # Pollutants and materials
    pollutants = flatten_list_field(papers, "pollutants")
    materials = flatten_list_field(papers, "materials")
    poll_counter = Counter(pollutants)
    mat_counter = Counter(materials)

    # Selection criteria
    sel_described = sum(1 for p in papers if p.get("data_selection_criteria_described"))

    # Preprocessing
    preproc = flatten_list_field(papers, "data_preprocessing")
    preproc_counter = Counter(preproc)

    # Target variable
    tv_names = Counter(p.get("target_variable_name") for p in papers if p.get("target_variable_name"))

    return {
        "data_source_distribution": dict(data_sources),
        "dataset_size_stats": quantiles(dataset_sizes),
        "dataset_size_n_null": n - len(dataset_sizes),
        "dataset_size_pct_under_100": pct(sum(1 for v in dataset_sizes if v < 100), len(dataset_sizes)),
        "dataset_size_pct_under_500": pct(sum(1 for v in dataset_sizes if v < 500), len(dataset_sizes)),
        "dataset_size_by_source": size_by_source,
        "top_15_pollutants": counter_to_sorted(poll_counter, 15),
        "top_15_materials": counter_to_sorted(mat_counter, 15),
        "total_unique_pollutants": len(poll_counter),
        "total_unique_materials": len(mat_counter),
        "data_selection_criteria_described_count": sel_described,
        "data_selection_criteria_described_rate": pct(sel_described, n),
        "data_preprocessing_distribution": dict(preproc_counter.most_common()),
        "target_variable_distribution": dict(tv_names.most_common()),
    }


def compute_pillar2(papers):
    n = len(papers)

    # Validation method
    val_methods = Counter(p["validation_method"] for p in papers if p.get("validation_method"))
    grouped = sum(1 for p in papers if p.get("grouped_splitting"))
    external = sum(1 for p in papers if p.get("external_validation"))
    train_test = sum(1 for p in papers if p.get("reports_train_and_test"))

    # R² / R distribution
    r2_papers = [p for p in papers if p.get("best_metric_type") == "R2" and p.get("best_metric_value") is not None]
    r_papers = [p for p in papers if p.get("best_metric_type") == "R" and p.get("best_metric_value") is not None]
    adj_r2 = [p for p in papers if p.get("best_metric_type") == "adjusted_R2" and p.get("best_metric_value") is not None]

    r2_values = [p["best_metric_value"] for p in r2_papers]
    r_values = [p["best_metric_value"] for p in r_papers]

    # Hyperparameter tuning
    hp = Counter(p["hyperparameter_tuning"] for p in papers if p.get("hyperparameter_tuning"))

    # Features
    n_features = safe_values(papers, "n_features")

    # Dataset size vs R² scatter data
    scatter = []
    for p in papers:
        if p.get("dataset_size") and p.get("best_metric_value") and p.get("best_metric_type") == "R2":
            scatter.append([p["dataset_size"], p["best_metric_value"]])

    # Train-test ratio
    ratios = Counter(p["train_test_ratio"] for p in papers if p.get("train_test_ratio"))

    # k-fold k values
    k_values = safe_values(papers, "k_fold_k")

    return {
        "validation_method_distribution": dict(val_methods.most_common()),
        "grouped_splitting_count": grouped,
        "grouped_splitting_rate": pct(grouped, n),
        "external_validation_count": external,
        "external_validation_rate": pct(external, n),
        "reports_train_and_test_count": train_test,
        "reports_train_and_test_rate": pct(train_test, n),
        "best_metric_type_counts": {
            "R2": len(r2_papers),
            "R": len(r_papers),
            "adjusted_R2": len(adj_r2),
            "null": n - len(r2_papers) - len(r_papers) - len(adj_r2),
        },
        "r2_stats": quantiles(r2_values),
        "r2_above_090": pct(sum(1 for v in r2_values if v >= 0.90), len(r2_values)) if r2_values else 0,
        "r2_above_095": pct(sum(1 for v in r2_values if v >= 0.95), len(r2_values)) if r2_values else 0,
        "r2_above_099": pct(sum(1 for v in r2_values if v >= 0.99), len(r2_values)) if r2_values else 0,
        "r_stats": quantiles(r_values),
        "hyperparameter_tuning_distribution": dict(hp.most_common()),
        "n_features_stats": quantiles(n_features),
        "dataset_size_vs_r2_scatter": scatter,
        "train_test_ratio_distribution": dict(ratios.most_common()),
        "k_fold_k_distribution": dict(Counter(k_values).most_common()),
    }


def compute_pillar3(papers):
    n = len(papers)

    # Algorithm frequency
    algos = flatten_list_field(papers, "ml_algorithms")
    algo_counter = Counter(algos)

    # Best algorithm
    best_algos = Counter(p["best_algorithm"] for p in papers if p.get("best_algorithm"))

    # N algorithms compared
    n_compared = safe_values(papers, "n_algorithms_compared")

    # Evaluation metrics
    metrics = flatten_list_field(papers, "evaluation_metrics")
    metric_counter = Counter(metrics)

    # Code / data availability
    code = sum(1 for p in papers if p.get("code_available"))
    data = sum(1 for p in papers if p.get("data_available"))
    prior = sum(1 for p in papers if p.get("compared_with_prior_work"))

    # Software tools
    tools = flatten_list_field(papers, "software_tools")
    tool_counter = Counter(tools)

    # Feature selection
    fs = Counter(p["feature_selection_method"] for p in papers if p.get("feature_selection_method"))

    return {
        "algorithm_frequency": counter_to_sorted(algo_counter),
        "top_10_algorithms": counter_to_sorted(algo_counter, 10),
        "total_unique_algorithms": len(algo_counter),
        "best_algorithm_frequency": counter_to_sorted(best_algos, 10),
        "n_algorithms_compared_stats": quantiles(n_compared),
        "n_algorithms_compared_distribution": dict(Counter(n_compared).most_common()),
        "evaluation_metrics_frequency": counter_to_sorted(metric_counter),
        "code_available_count": code,
        "code_available_rate": pct(code, n),
        "data_available_count": data,
        "data_available_rate": pct(data, n),
        "compared_with_prior_work_count": prior,
        "compared_with_prior_work_rate": pct(prior, n),
        "software_tools_distribution": dict(tool_counter.most_common()),
        "feature_selection_distribution": dict(fs.most_common()),
    }


def compute_discussion(papers):
    n = len(papers)

    # Interpretability
    interp = flatten_list_field(papers, "interpretability_methods")
    interp_counter = Counter(interp)

    # Papers with any interpretability (excluding "none")
    has_interp = sum(1 for p in papers
                     if p.get("interpretability_methods")
                     and not (isinstance(p["interpretability_methods"], list)
                              and p["interpretability_methods"] == ["none"]))

    mech = sum(1 for p in papers if p.get("mechanistic_discussion"))

    # Water type
    water = Counter(p["water_type"] for p in papers if p.get("water_type"))

    # Deployment
    scalability = sum(1 for p in papers if p.get("discusses_scalability"))
    engineering = sum(1 for p in papers if p.get("engineering_validation"))
    cost = sum(1 for p in papers if p.get("cost_analysis"))

    return {
        "interpretability_methods_frequency": counter_to_sorted(interp_counter),
        "has_interpretability_count": has_interp,
        "has_interpretability_rate": pct(has_interp, n),
        "mechanistic_discussion_count": mech,
        "mechanistic_discussion_rate": pct(mech, n),
        "water_type_distribution": dict(water.most_common()),
        "discusses_scalability_count": scalability,
        "discusses_scalability_rate": pct(scalability, n),
        "engineering_validation_count": engineering,
        "engineering_validation_rate": pct(engineering, n),
        "cost_analysis_count": cost,
        "cost_analysis_rate": pct(cost, n),
    }


def compute_cross_tabulations(papers):
    """Cross-tabulations for deeper analysis."""

    # R² by validation method
    r2_by_val = {}
    for p in papers:
        vm = p.get("validation_method")
        if vm and p.get("best_metric_type") == "R2" and p.get("best_metric_value") is not None:
            r2_by_val.setdefault(vm, []).append(p["best_metric_value"])
    r2_by_val_stats = {k: quantiles(v) for k, v in r2_by_val.items()}

    # R² by data source
    r2_by_src = {}
    for p in papers:
        src = p.get("data_source")
        if src and p.get("best_metric_type") == "R2" and p.get("best_metric_value") is not None:
            r2_by_src.setdefault(src, []).append(p["best_metric_value"])
    r2_by_src_stats = {k: quantiles(v) for k, v in r2_by_src.items()}

    # Algorithm usage by year
    algo_by_year = {}
    for p in papers:
        yr = p.get("year")
        algos = p.get("ml_algorithms")
        if yr and isinstance(algos, list):
            algo_by_year.setdefault(yr, Counter()).update(algos)
    algo_by_year_dict = {str(k): dict(v.most_common(5)) for k, v in sorted(algo_by_year.items())}

    # Interpretability by year
    interp_by_year = {}
    for p in papers:
        yr = p.get("year")
        methods = p.get("interpretability_methods")
        if yr and isinstance(methods, list):
            has = not (methods == ["none"] or methods == [])
            interp_by_year.setdefault(yr, {"has": 0, "total": 0})
            interp_by_year[yr]["total"] += 1
            if has:
                interp_by_year[yr]["has"] += 1
    for yr in interp_by_year:
        d = interp_by_year[yr]
        d["rate"] = pct(d["has"], d["total"])
    interp_by_year_dict = {str(k): v for k, v in sorted(interp_by_year.items())}

    # Code/data availability by year
    open_by_year = {}
    for p in papers:
        yr = p.get("year")
        if yr:
            open_by_year.setdefault(yr, {"code": 0, "data": 0, "total": 0})
            open_by_year[yr]["total"] += 1
            if p.get("code_available"):
                open_by_year[yr]["code"] += 1
            if p.get("data_available"):
                open_by_year[yr]["data"] += 1
    for yr in open_by_year:
        d = open_by_year[yr]
        d["code_rate"] = pct(d["code"], d["total"])
        d["data_rate"] = pct(d["data"], d["total"])
    open_by_year_dict = {str(k): v for k, v in sorted(open_by_year.items())}

    return {
        "r2_by_validation_method": r2_by_val_stats,
        "r2_by_data_source": r2_by_src_stats,
        "algorithm_by_year": algo_by_year_dict,
        "interpretability_by_year": interp_by_year_dict,
        "open_science_by_year": open_by_year_dict,
    }


# ============================================================
# Figure Data Export
# ============================================================

def export_figure_data(papers, stats):
    """Export pre-computed data for each figure."""

    # Fig 1: Bibliometric
    fig1 = {
        "year_distribution": stats["introduction"]["year_distribution"],
        "top_10_journals": stats["introduction"]["top_10_journals"],
        "research_type_distribution": stats["introduction"]["research_type_distribution"],
        "citation_stats": stats["introduction"]["citation_stats"],
        "citations": safe_values(papers, "cited_by_count"),
    }

    # Fig 2: Data Bias
    fig2 = {
        "data_source_distribution": stats["pillar1_data_bias"]["data_source_distribution"],
        "dataset_sizes": safe_values(papers, "dataset_size"),
        "dataset_size_stats": stats["pillar1_data_bias"]["dataset_size_stats"],
        "top_15_pollutants": stats["pillar1_data_bias"]["top_15_pollutants"],
        "top_15_materials": stats["pillar1_data_bias"]["top_15_materials"],
        "target_variable_distribution": stats["pillar1_data_bias"]["target_variable_distribution"],
    }

    # Fig 3: Validation
    fig3 = {
        "validation_method_distribution": stats["pillar2_validation"]["validation_method_distribution"],
        "r2_values": [p["best_metric_value"] for p in papers
                      if p.get("best_metric_type") == "R2" and p.get("best_metric_value") is not None],
        "r2_stats": stats["pillar2_validation"]["r2_stats"],
        "dataset_size_vs_r2_scatter": stats["pillar2_validation"]["dataset_size_vs_r2_scatter"],
        "grouped_splitting_rate": stats["pillar2_validation"]["grouped_splitting_rate"],
        "external_validation_rate": stats["pillar2_validation"]["external_validation_rate"],
        "reports_train_and_test_rate": stats["pillar2_validation"]["reports_train_and_test_rate"],
        "hyperparameter_tuning_distribution": stats["pillar2_validation"]["hyperparameter_tuning_distribution"],
    }

    # Fig 4: Comparability
    fig4 = {
        "top_10_algorithms": stats["pillar3_comparability"]["top_10_algorithms"],
        "algorithm_frequency": stats["pillar3_comparability"]["algorithm_frequency"],
        "evaluation_metrics_frequency": stats["pillar3_comparability"]["evaluation_metrics_frequency"],
        "code_available_rate": stats["pillar3_comparability"]["code_available_rate"],
        "data_available_rate": stats["pillar3_comparability"]["data_available_rate"],
        "n_algorithms_compared_distribution": stats["pillar3_comparability"]["n_algorithms_compared_distribution"],
        "software_tools_distribution": stats["pillar3_comparability"]["software_tools_distribution"],
    }

    # Fig 5: Discussion
    fig5 = {
        "interpretability_methods_frequency": stats["discussion"]["interpretability_methods_frequency"],
        "mechanistic_discussion_rate": stats["discussion"]["mechanistic_discussion_rate"],
        "water_type_distribution": stats["discussion"]["water_type_distribution"],
        "discusses_scalability_rate": stats["discussion"]["discusses_scalability_rate"],
        "engineering_validation_rate": stats["discussion"]["engineering_validation_rate"],
        "cost_analysis_rate": stats["discussion"]["cost_analysis_rate"],
        "interpretability_by_year": stats["cross_tabulations"]["interpretability_by_year"],
    }

    for name, data in [("fig1_bibliometric", fig1), ("fig2_data_bias", fig2),
                        ("fig3_validation", fig3), ("fig4_comparability", fig4),
                        ("fig5_discussion", fig5)]:
        out = FIGURE_DIR / f"{name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out.name}")


# ============================================================
# Print Key Numbers
# ============================================================

def print_key_numbers(stats):
    s = stats
    intro = s["introduction"]
    p1 = s["pillar1_data_bias"]
    p2 = s["pillar2_validation"]
    p3 = s["pillar3_comparability"]
    disc = s["discussion"]

    print("\n" + "=" * 60)
    print("KEY NUMBERS FOR MANUSCRIPT (n={})".format(intro["n_papers"]))
    print("=" * 60)

    print("\n--- Introduction ---")
    print(f"  Total papers: {intro['n_papers']}")
    print(f"  Year range: {min(intro['year_distribution'])}–{max(intro['year_distribution'])}")
    print(f"  Journals: {intro['total_journals']}")
    print(f"  Adsorption/Degradation/Both: {intro['adsorption_count']}/{intro['degradation_count']}/{intro['both_count']}")
    print(f"  Open access: {intro['oa_count']} ({intro['oa_rate']:.1%})")

    print("\n--- Pillar 1: Data Bias ---")
    ds = p1["dataset_size_stats"]
    print(f"  Dataset size: median={ds.get('median','N/A')}, mean={ds.get('mean','N/A')}, n={ds.get('n','N/A')}")
    print(f"  Dataset size <100: {p1['dataset_size_pct_under_100']:.1%}")
    print(f"  Dataset size <500: {p1['dataset_size_pct_under_500']:.1%}")
    print(f"  Data sources: {p1['data_source_distribution']}")
    print(f"  Selection criteria described: {p1['data_selection_criteria_described_count']} ({p1['data_selection_criteria_described_rate']:.1%})")
    print(f"  Top 5 pollutants: {p1['top_15_pollutants'][:5]}")
    print(f"  Top 5 materials: {p1['top_15_materials'][:5]}")

    print("\n--- Pillar 2: Validation ---")
    print(f"  Validation methods: {p2['validation_method_distribution']}")
    print(f"  Grouped splitting: {p2['grouped_splitting_count']} ({p2['grouped_splitting_rate']:.1%})")
    print(f"  External validation: {p2['external_validation_count']} ({p2['external_validation_rate']:.1%})")
    r2 = p2["r2_stats"]
    print(f"  R² (n={r2.get('n','N/A')}): median={r2.get('median','N/A')}, mean={r2.get('mean','N/A')}")
    print(f"  R² > 0.90: {p2['r2_above_090']:.1%}")
    print(f"  R² > 0.95: {p2['r2_above_095']:.1%}")
    print(f"  R² > 0.99: {p2['r2_above_099']:.1%}")
    print(f"  Metric types: {p2['best_metric_type_counts']}")

    print("\n--- Pillar 3: Comparability ---")
    print(f"  Top 5 algorithms: {p3['top_10_algorithms'][:5]}")
    print(f"  Unique algorithms: {p3['total_unique_algorithms']}")
    nc = p3["n_algorithms_compared_stats"]
    print(f"  Algorithms compared: median={nc.get('median','N/A')}")
    print(f"  Code available: {p3['code_available_count']} ({p3['code_available_rate']:.1%})")
    print(f"  Data available: {p3['data_available_count']} ({p3['data_available_rate']:.1%})")
    print(f"  Compared with prior: {p3['compared_with_prior_work_count']} ({p3['compared_with_prior_work_rate']:.1%})")

    print("\n--- Discussion ---")
    print(f"  Has interpretability: {disc['has_interpretability_count']} ({disc['has_interpretability_rate']:.1%})")
    print(f"  Mechanistic discussion: {disc['mechanistic_discussion_count']} ({disc['mechanistic_discussion_rate']:.1%})")
    print(f"  Water type: {disc['water_type_distribution']}")
    print(f"  Scalability: {disc['discusses_scalability_count']} ({disc['discusses_scalability_rate']:.1%})")
    print(f"  Engineering validation: {disc['engineering_validation_count']} ({disc['engineering_validation_rate']:.1%})")
    print(f"  Cost analysis: {disc['cost_analysis_count']} ({disc['cost_analysis_rate']:.1%})")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build aggregated statistics")
    parser.add_argument("--print", action="store_true", dest="print_numbers",
                        help="Print key numbers to console")
    args = parser.parse_args()

    papers = load_data()
    print(f"Loaded {len(papers)} papers from {DATASET_FILE.name}")

    # Compute all statistics
    stats = {
        "meta": {
            "n_papers": len(papers),
            "excluded_ids": [66, 81, 119],
            "generated_at": datetime.now().isoformat(),
            "data_source": DATASET_FILE.name,
        },
        "introduction": compute_introduction(papers),
        "pillar1_data_bias": compute_pillar1(papers),
        "pillar2_validation": compute_pillar2(papers),
        "pillar3_comparability": compute_pillar3(papers),
        "discussion": compute_discussion(papers),
        "cross_tabulations": compute_cross_tabulations(papers),
    }

    # Save statistics
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_FILE}")

    # Export figure data
    print("Exporting figure data:")
    export_figure_data(papers, stats)

    # Print key numbers
    if args.print_numbers:
        print_key_numbers(stats)
    else:
        print_key_numbers(stats)  # always print for verification


if __name__ == "__main__":
    main()
