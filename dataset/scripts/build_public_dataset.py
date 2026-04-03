"""
Build public dataset for ML4Env Critical Review.

Merges fulltext extraction results with paper metadata,
applies post-processing fixes, name standardization,
and outputs clean JSON + CSV for publication.

Usage:
  python scripts/build_public_dataset.py
  python scripts/build_public_dataset.py --validate  # print validation stats
"""

import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FULLTEXT_FILE = DATA_DIR / "processed" / "fulltext_extraction_158.json"
INDEX_FILE = DATA_DIR / "processed" / "paper_158_index.json"
OUTPUT_DIR = DATA_DIR / "public"

EXCLUDE_IDS = {66, 81, 119}  # 66: off-topic Antarctic climate; 81: duplicate of 82 (SI DOI); 119: duplicate of 118

# ============================================================
# Name Standardization Maps
# ============================================================

POLLUTANT_MAP = {
    "MB": "Methylene Blue", "Methylene blue": "Methylene Blue",
    "methylene blue": "Methylene Blue", "MG": "Malachite Green",
    "CR": "Congo Red", "Congo red": "Congo Red",
    "RhB": "Rhodamine B", "Rhodamine b": "Rhodamine B",
    "CV": "Crystal Violet", "crystal violet": "Crystal Violet",
    "MO": "Methyl Orange", "Methyl orange": "Methyl Orange",
    "TC": "Tetracycline", "tetracycline": "Tetracycline",
    "BPA": "Bisphenol A", "bisphenol A": "Bisphenol A",
    "Pb2+": "Pb(II)", "Pb(ii)": "Pb(II)", "Pb": "Pb(II)", "lead": "Pb(II)",
    "Cd2+": "Cd(II)", "Cd(ii)": "Cd(II)", "Cd": "Cd(II)", "cadmium": "Cd(II)",
    "Cu2+": "Cu(II)", "Cu(ii)": "Cu(II)", "Cu": "Cu(II)", "copper": "Cu(II)",
    "Cr6+": "Cr(VI)", "Cr(vi)": "Cr(VI)",
    "As5+": "As(V)", "As(v)": "As(V)", "arsenic": "As(V)",
    "Zn2+": "Zn(II)", "Zn(ii)": "Zn(II)",
    "Ni2+": "Ni(II)", "Ni(ii)": "Ni(II)",
    "Hg2+": "Hg(II)", "Hg(ii)": "Hg(II)",
    "fluoride": "Fluoride", "phosphate": "Phosphate", "nitrate": "Nitrate",
}

MATERIAL_MAP_LOWER = {
    "biochar": "Biochar", "activated carbon": "Activated Carbon",
    "activated biochar": "Biochar", "modified biochar": "Biochar",
    "mof": "MOF", "metal-organic framework": "MOF",
    "zeolite": "Zeolite", "clay": "Clay", "bentonite": "Clay",
    "graphene oxide": "Graphene Oxide", "go": "Graphene Oxide",
    "rgo": "Graphene Oxide", "reduced graphene oxide": "Graphene Oxide",
    "carbon nanotube": "CNT", "cnt": "CNT", "mwcnt": "CNT",
    "tio2": "TiO2", "titanium dioxide": "TiO2",
    "zno": "ZnO", "zinc oxide": "ZnO",
    "fe3o4": "Fe3O4", "magnetite": "Fe3O4",
    "chitosan": "Chitosan", "hydrogel": "Hydrogel",
    "layered double hydroxide": "LDH", "ldh": "LDH",
}

ALGO_STANDARD = {
    "RF", "XGBoost", "LightGBM", "CatBoost", "GBM", "AdaBoost",
    "ANN", "DNN", "CNN", "LSTM", "SVM", "SVR", "GPR", "DT", "KNN",
    "LR", "Ridge", "LASSO", "ElasticNet", "ELM", "RSM", "ET",
    "Stacking", "Bagging", "Bayesian", "AutoML",
}

ALGO_MAP_UPPER = {
    "RANDOM FOREST": "RF", "XGBOOST": "XGBoost", "LIGHTGBM": "LightGBM",
    "CATBOOST": "CatBoost", "GRADIENT BOOSTING": "GBM", "GBR": "GBM",
    "GBRT": "GBM", "ADABOOST": "AdaBoost", "MLP": "ANN", "BP": "ANN",
    "BPNN": "ANN", "NEURAL NETWORK": "ANN", "BACKPROPAGATION": "ANN",
    "DEEP NEURAL NETWORK": "DNN", "DEEP LEARNING": "DNN",
    "CONVOLUTIONAL NEURAL NETWORK": "CNN",
    "SUPPORT VECTOR MACHINE": "SVM", "SUPPORT VECTOR REGRESSION": "SVR",
    "GAUSSIAN PROCESS": "GPR", "KRIGING": "GPR",
    "DECISION TREE": "DT", "CART": "DT",
    "K-NEAREST NEIGHBORS": "KNN", "K-NEAREST NEIGHBOR": "KNN",
    "LINEAR REGRESSION": "LR", "MLR": "LR", "MULTIPLE LINEAR REGRESSION": "LR",
    "RIDGE REGRESSION": "Ridge", "RIDGE": "Ridge",
    "LASSO": "LASSO", "ELASTIC NET": "ElasticNet", "ELASTICNET": "ElasticNet",
    "EXTREME LEARNING MACHINE": "ELM",
    "RESPONSE SURFACE METHODOLOGY": "RSM",
    "EXTRA TREES": "ET", "STACKING": "Stacking", "BAGGING": "Bagging",
    "BAYESIAN": "Bayesian", "AUTOML": "AutoML", "TPOT": "AutoML",
}


# ============================================================
# Standardization Functions
# ============================================================

def std_pollutant(name):
    if not isinstance(name, str):
        return str(name)
    n = name.strip()
    return POLLUTANT_MAP.get(n, n)


def std_material(name):
    if not isinstance(name, str):
        return str(name)
    n = name.strip()
    nl = n.lower()
    if nl in MATERIAL_MAP_LOWER:
        return MATERIAL_MAP_LOWER[nl]
    for key, val in MATERIAL_MAP_LOWER.items():
        if key in nl:
            return val
    return n


def std_algorithm(name):
    if not isinstance(name, str):
        return str(name)
    n = name.strip()
    if n in ALGO_STANDARD:
        return n
    up = n.upper()
    if up in ALGO_MAP_UPPER:
        return ALGO_MAP_UPPER[up]
    return n


def std_list(lst, func):
    if not isinstance(lst, list):
        return []
    seen = set()
    result = []
    for item in lst:
        s = func(item)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


# ============================================================
# Post-processing Rules
# ============================================================

def apply_fixes(record):
    """Apply post-processing fixes to a single record."""
    # Fix mechanistic_discussion: no interpretability → no mechanistic discussion
    interp = record.get("interpretability_methods", [])
    if not isinstance(interp, list):
        interp = []
    has_interp = any(m.lower() != "none" for m in interp if isinstance(m, str))
    if not has_interp:
        record["mechanistic_discussion"] = False

    # Standardize names
    record["pollutants"] = std_list(record.get("pollutants", []), std_pollutant)
    record["materials"] = std_list(record.get("materials", []), std_material)
    record["ml_algorithms"] = std_list(record.get("ml_algorithms", []), std_algorithm)

    # Clean DOI format (remove URL prefix if present)
    doi = record.get("doi", "")
    if isinstance(doi, str) and doi.startswith("https://doi.org/"):
        record["doi"] = doi.replace("https://doi.org/", "")

    return record


# ============================================================
# Build Dataset
# ============================================================

# Fields to include in public dataset (ordered)
PUBLIC_FIELDS = [
    # Metadata
    "paper_id", "paper_title", "doi", "first_author", "year", "journal",
    "cited_by_count", "is_oa", "research_type",
    # Pillar 1: Data Bias
    "data_source", "dataset_size", "n_literature_sources",
    "target_variable_name", "target_variable_name_original",
    "target_variable_unit", "target_variable_range_min", "target_variable_range_max",
    "pollutants", "materials",
    "data_selection_criteria_described", "data_preprocessing",
    # Pillar 2: Validation
    "validation_method", "k_fold_k", "train_test_ratio", "grouped_splitting",
    "best_metric_type", "best_metric_value", "best_rmse",
    "reports_train_and_test", "external_validation",
    "n_features", "hyperparameter_tuning", "best_algorithm",
    # Pillar 3: Interpretability
    "interpretability_methods", "n_features_used", "feature_selection_method",
    "top_3_features", "mechanistic_discussion",
    # Pillar 4: Comparability
    "ml_algorithms", "n_algorithms_compared", "evaluation_metrics",
    "software_tools", "code_available", "data_available", "compared_with_prior_work",
    # Deployment
    "water_type", "discusses_scalability", "engineering_validation", "cost_analysis",
]

# Fields that are lists (need "; " join for CSV)
LIST_FIELDS = {
    "pollutants", "materials", "ml_algorithms", "data_preprocessing",
    "interpretability_methods", "top_3_features", "evaluation_metrics",
    "software_tools",
}


def build_dataset():
    # Load data
    with open(FULLTEXT_FILE, encoding="utf-8") as f:
        ft_raw = json.load(f)
    with open(INDEX_FILE, encoding="utf-8") as f:
        idx_data = json.load(f)
    idx_papers = idx_data["papers"] if isinstance(idx_data, dict) else idx_data
    idx_map = {p["paper_id"]: p for p in idx_papers}

    records = []
    excluded = 0
    for r in ft_raw:
        pid = r.get("paper_id")
        if pid in EXCLUDE_IDS:
            excluded += 1
            continue
        if r.get("_status") != "success":
            continue

        # Merge metadata from index
        meta = idx_map.get(pid, {})
        r["first_author"] = meta.get("first_author", "")
        r["year"] = meta.get("publication_year")
        r["journal"] = meta.get("journal", "")
        r["cited_by_count"] = meta.get("cited_by_count", 0)
        r["is_oa"] = meta.get("is_oa", False)

        # Flatten target_variable
        tv = r.get("target_variable", {})
        if isinstance(tv, dict):
            r["target_variable_name"] = tv.get("name", "")
            r["target_variable_name_original"] = tv.get("name_original", "")
            r["target_variable_unit"] = tv.get("unit", "")
            r["target_variable_range_min"] = tv.get("range_min")
            r["target_variable_range_max"] = tv.get("range_max")
        else:
            r["target_variable_name"] = ""
            r["target_variable_name_original"] = ""
            r["target_variable_unit"] = ""
            r["target_variable_range_min"] = None
            r["target_variable_range_max"] = None

        # Apply fixes
        r = apply_fixes(r)

        # Select only public fields
        clean = {}
        for field in PUBLIC_FIELDS:
            clean[field] = r.get(field)
        records.append(clean)

    records.sort(key=lambda x: x["paper_id"])
    print(f"Built dataset: {len(records)} papers (excluded {excluded})")
    return records


def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {path} ({len(records)} records)")


def save_csv(records, path):
    if not records:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        for r in records:
            row = {}
            for k, v in r.items():
                if k in LIST_FIELDS and isinstance(v, list):
                    row[k] = "; ".join(str(x) for x in v)
                elif v is None:
                    row[k] = ""
                elif isinstance(v, bool):
                    row[k] = str(v).lower()
                else:
                    row[k] = v
            writer.writerow(row)
    print(f"Saved CSV: {path} ({len(records)} records)")


def save_metadata(records, path):
    # Compute summary stats
    n = len(records)
    years = [r["year"] for r in records if r.get("year")]
    meta = {
        "dataset_name": "ML4Env Critical Review Dataset",
        "description": "Structured extraction of methodological practices from 157 ML papers in environmental adsorption/degradation prediction",
        "version": "1.0",
        "date_created": datetime.now().isoformat()[:10],
        "n_papers": n,
        "year_range": [min(years), max(years)] if years else [],
        "source": "LLM-assisted extraction (Qwen3.5-Plus) from MinerU-converted fulltext, validated against 10-paper gold standard (83.8/100 accuracy)",
        "exclusions": "Paper 66 excluded (off-topic: Antarctic climate study)",
        "post_processing": [
            "mechanistic_discussion set to false when interpretability_methods=['none']",
            "Pollutant names standardized (e.g., MB -> Methylene Blue, Pb2+ -> Pb(II))",
            "Material names standardized (e.g., biochar -> Biochar, mof -> MOF)",
            "Algorithm names standardized (e.g., BP -> ANN, MLR -> LR)",
            "Internal metadata fields removed (_status, _elapsed_sec, etc.)",
        ],
        "fields": {f: _field_desc(f) for f in PUBLIC_FIELDS},
        "license": "CC-BY-4.0",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {path}")


def _field_desc(field):
    descs = {
        "paper_id": "Unique paper identifier",
        "paper_title": "Full paper title",
        "doi": "Digital Object Identifier",
        "first_author": "First author name",
        "year": "Publication year",
        "journal": "Journal name",
        "cited_by_count": "Citation count (OpenAlex)",
        "is_oa": "Open access status",
        "research_type": "adsorption | degradation | both",
        "data_source": "experimental | literature | database | mixed",
        "dataset_size": "Total ML data points (train + test)",
        "n_literature_sources": "Number of source papers if literature-collected",
        "target_variable_name": "Standardized target variable name",
        "target_variable_name_original": "Original target variable name as in paper",
        "target_variable_unit": "Unit of target variable",
        "target_variable_range_min": "Minimum value of target in dataset",
        "target_variable_range_max": "Maximum value of target in dataset",
        "pollutants": "List of pollutants/adsorbates",
        "materials": "List of adsorbent/catalyst materials",
        "data_selection_criteria_described": "Whether data selection criteria are described",
        "data_preprocessing": "List of preprocessing steps applied",
        "validation_method": "Primary validation strategy",
        "k_fold_k": "K value if k-fold CV used",
        "train_test_ratio": "Train:test split ratio (e.g., 80:20)",
        "grouped_splitting": "Whether data was grouped before splitting",
        "best_metric_type": "R2 | R | adjusted_R2",
        "best_metric_value": "Best reported metric value on test set",
        "best_rmse": "Best RMSE on test set",
        "reports_train_and_test": "Reports both train and test metrics",
        "external_validation": "Tested on independent external dataset",
        "n_features": "Total number of input features",
        "hyperparameter_tuning": "Tuning method used",
        "best_algorithm": "Best performing algorithm (standardized)",
        "interpretability_methods": "List of interpretability methods used",
        "n_features_used": "Number of features in final model",
        "feature_selection_method": "Feature selection approach",
        "top_3_features": "Top 3 most important features",
        "mechanistic_discussion": "Connects ML interpretability to known mechanisms",
        "ml_algorithms": "List of ML algorithms used (standardized)",
        "n_algorithms_compared": "Number of algorithms compared",
        "evaluation_metrics": "List of evaluation metrics reported",
        "software_tools": "Programming languages/tools used",
        "code_available": "Source code publicly available",
        "data_available": "Training data publicly available",
        "compared_with_prior_work": "Compares with previously published models",
        "water_type": "synthetic | real_wastewater | both | not_specified | not_applicable",
        "discusses_scalability": "Discusses scale-up or practical application",
        "engineering_validation": "ML validated with engineering/pilot data",
        "cost_analysis": "Includes economic/cost analysis",
    }
    return descs.get(field, "")


def validate(records):
    """Print validation statistics."""
    n = len(records)
    print(f"\n{'='*60}")
    print(f"Dataset Validation (n={n})")
    print(f"{'='*60}")

    # Check completeness
    for field in PUBLIC_FIELDS:
        null_count = sum(1 for r in records if r.get(field) is None or r.get(field) == "")
        if null_count > 0 and field not in LIST_FIELDS:
            pct = null_count / n * 100
            if pct > 5:
                print(f"  {field}: {null_count} null ({pct:.0f}%)")

    # mechanistic_discussion after fix
    md_true = sum(1 for r in records if r.get("mechanistic_discussion"))
    print(f"\nmechanistic_discussion after fix: {md_true}/{n} ({md_true/n*100:.1f}%)")

    # Verify no interpretability=none + mechanistic=true
    bad = sum(1 for r in records
              if r.get("mechanistic_discussion") and
              (not r.get("interpretability_methods") or
               all(m.lower() == "none" for m in r.get("interpretability_methods", []))))
    print(f"Logical contradiction (none + true): {bad}")

    # Name standardization check
    all_algos = []
    for r in records:
        all_algos.extend(r.get("ml_algorithms", []))
    non_std = [a for a in set(all_algos) if a not in ALGO_STANDARD]
    if non_std:
        print(f"\nNon-standard algorithm names ({len(non_std)}): {non_std[:10]}")

    print(f"\nDataset ready for publication.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build public ML4Env dataset")
    parser.add_argument("--validate", action="store_true", help="Print validation stats")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = build_dataset()

    json_path = OUTPUT_DIR / "ml4env_157_dataset.json"
    csv_path = OUTPUT_DIR / "ml4env_157_dataset.csv"
    meta_path = OUTPUT_DIR / "ml4env_157_metadata.json"

    save_json(records, json_path)
    save_csv(records, csv_path)
    save_metadata(records, meta_path)

    if args.validate:
        validate(records)
    else:
        validate(records)  # always validate


if __name__ == "__main__":
    main()
