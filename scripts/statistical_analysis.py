"""
Step 6: Statistical Analysis & Visualization for ML4Env Critical Review.

Generates Fig 2-5 from the clean public dataset (155 papers).
Uses plot_config.py for consistent styling.

Usage:
  python scripts/statistical_analysis.py           # generate all figures
  python scripts/statistical_analysis.py --fig 2   # generate specific figure
  python scripts/statistical_analysis.py --stats    # print statistics only
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from collections import Counter
from pathlib import Path
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import (
    PILLAR_COLORS, FIELD_COLORS, CATEGORICAL_PALETTE, ACCENT_COLORS,
    HEATMAP_CMAP, FONT_SIZES, LINE_WIDTHS, FIGURE_SIZE_MULTI,
    DPI, FIGURE_DIR, apply_plot_style, add_panel_label, style_axis, save_figure,
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "public"
DATASET_FILE = DATA_DIR / "ml4env_126_dataset.json"

# ============================================================
# Data Loading (from pre-cleaned public dataset)
# ============================================================

def load_data():
    """Load the clean 155-paper public dataset."""
    with open(DATASET_FILE, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} papers from {DATASET_FILE.name}")
    return df


def standardize_pollutants(name):
    """Map common pollutant name variants to standard form."""
    if not isinstance(name, str):
        return str(name)
    n = name.strip()
    mapping = {
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
        "Cr6+": "Cr(VI)", "Cr(vi)": "Cr(VI)", "Cr(III)": "Cr(III)",
        "As5+": "As(V)", "As(v)": "As(V)", "As(III)": "As(III)",
        "Zn2+": "Zn(II)", "Zn(ii)": "Zn(II)", "Ni2+": "Ni(II)", "Ni(ii)": "Ni(II)",
        "Hg2+": "Hg(II)", "Hg(ii)": "Hg(II)",
        "fluoride": "Fluoride", "phosphate": "Phosphate", "nitrate": "Nitrate",
        "arsenic": "As(V)",
        "CO2": r"$\mathrm{CO_2}$", "CH4": r"$\mathrm{CH_4}$",
        "N2": r"$\mathrm{N_2}$", "H2": r"$\mathrm{H_2}$", "O2": r"$\mathrm{O_2}$",
        "H2S": r"$\mathrm{H_2S}$", "H2O": r"$\mathrm{H_2O}$",
        "C2H6": r"$\mathrm{C_2H_6}$", "C3H8": r"$\mathrm{C_3H_8}$",
        "C2H2": r"$\mathrm{C_2H_2}$", "C2H4": r"$\mathrm{C_2H_4}$",
        "CF4": r"$\mathrm{CF_4}$", "NH4+": r"$\mathrm{NH_4^+}$",
        "O3": r"$\mathrm{O_3}$", "SO2": r"$\mathrm{SO_2}$", "NO2": r"$\mathrm{NO_2}$",
    }
    return mapping.get(n, n)


def standardize_materials(name):
    """Map common material name variants to standard form."""
    if not isinstance(name, str):
        return str(name)
    n = name.strip().lower()
    mapping = {
        "biochar": "Biochar", "activated carbon": "Activated Carbon",
        "activated biochar": "Biochar", "modified biochar": "Biochar",
        "mof": "MOF", "metal-organic framework": "MOF",
        "zeolite": "Zeolite", "clay": "Clay", "bentonite": "Clay",
        "graphene oxide": "Graphene Oxide", "go": "Graphene Oxide",
        "rgo": "Graphene Oxide", "reduced graphene oxide": "Graphene Oxide",
        "carbon nanotube": "CNT", "cnt": "CNT", "mwcnt": "CNT",
        "tio2": r"$\mathrm{TiO_2}$", "titanium dioxide": r"$\mathrm{TiO_2}$",
        "zno": "ZnO", "zinc oxide": "ZnO",
        "fe3o4": r"$\mathrm{Fe_3O_4}$", "magnetite": r"$\mathrm{Fe_3O_4}$",
        "chitosan": "Chitosan", "hydrogel": "Hydrogel",
        "nanocomposite": "Nanocomposite", "nano-composite": "Nanocomposite",
        "layered double hydroxide": "LDH", "ldh": "LDH",
    }
    # Try exact match first
    if n in mapping:
        return mapping[n]
    # Partial match
    for key, val in mapping.items():
        if key in n:
            return val
    return name.strip()


def standardize_algorithms(name):
    """Normalize ML algorithm names."""
    if not isinstance(name, str):
        return str(name)
    n = name.strip().upper()
    mapping = {
        "RANDOM FOREST": "RF", "XGBOOST": "XGBoost", "LIGHTGBM": "LightGBM",
        "CATBOOST": "CatBoost", "GRADIENT BOOSTING": "GBM", "GBR": "GBM",
        "GBRT": "GBM", "ADABOOST": "AdaBoost", "MLP": "ANN", "BP": "ANN",
        "BPNN": "ANN", "NEURAL NETWORK": "ANN", "BACKPROPAGATION": "ANN",
        "DEEP NEURAL NETWORK": "DNN", "DEEP LEARNING": "DNN",
        "CONVOLUTIONAL NEURAL NETWORK": "CNN",
        "SUPPORT VECTOR MACHINE": "SVM", "SUPPORT VECTOR REGRESSION": "SVR",
        "GAUSSIAN PROCESS": "GPR", "KRIGING": "GPR",
        "DECISION TREE": "DT", "CART": "DT",
        "K-NEAREST NEIGHBOR": "KNN", "K-NEAREST NEIGHBORS": "KNN",
        "LINEAR REGRESSION": "LR", "MLR": "LR",
        "MULTIPLE LINEAR REGRESSION": "LR",
        "EXTREME LEARNING MACHINE": "ELM",
        "RESPONSE SURFACE METHODOLOGY": "RSM",
        "EXTRA TREES": "ET", "STACKING": "Stacking", "BAGGING": "Bagging",
        "BAYESIAN": "Bayesian", "RIDGE": "Ridge", "LASSO": "LASSO",
        "ELASTIC NET": "ElasticNet", "ELASTICNET": "ElasticNet",
    }
    # Already standard abbreviation
    standard = {"RF", "XGBoost", "LightGBM", "CatBoost", "GBM", "AdaBoost",
                "ANN", "DNN", "CNN", "LSTM", "SVM", "SVR", "GPR", "DT", "KNN",
                "LR", "Ridge", "LASSO", "ElasticNet", "ELM", "AutoML", "RSM",
                "ET", "Stacking", "Bagging", "Bayesian"}
    orig = name.strip()
    if orig in standard:
        return orig
    return mapping.get(n, orig)


def _sig_label(p):
    """Return significance label from p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _add_sig_bracket(ax, x1, x2, y, h, label, fs=None):
    """Draw a significance bracket between two x positions."""
    if fs is None:
        fs = FONT_SIZES
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color='#333333', linewidth=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha='center', va='bottom',
            fontsize=fs['annotation'], fontweight='bold', color='#333333')


# ============================================================
# Statistics Summary
# ============================================================

def print_stats(df):
    """Print key statistics for the paper."""
    print("\n" + "="*60)
    print("ML4Env Critical Review - Key Statistics (n={})".format(len(df)))
    print("="*60)

    # --- Pillar 1: Data Bias ---
    print("\n--- PILLAR 1: DATA BIAS ---")

    # Data source
    src = df["data_source"].value_counts()
    print(f"\nData source distribution:")
    for k, v in src.items():
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    # Dataset size
    ds = df["dataset_size"].dropna()
    print(f"\nDataset size (n={len(ds)}, {len(df)-len(ds)} null):")
    print(f"  Median: {ds.median():.0f}, Mean: {ds.mean():.0f}")
    print(f"  Min: {ds.min():.0f}, Max: {ds.max():.0f}")
    print(f"  Q1: {ds.quantile(0.25):.0f}, Q3: {ds.quantile(0.75):.0f}")

    # Separate experimental vs database
    exp_mask = df["data_source"].isin(["experimental", "literature", "mixed"])
    db_mask = df["data_source"] == "database"
    ds_exp = df.loc[exp_mask, "dataset_size"].dropna()
    ds_db = df.loc[db_mask, "dataset_size"].dropna()
    if len(ds_exp) > 0:
        print(f"  Experimental/Literature (n={len(ds_exp)}): median={ds_exp.median():.0f}")
    if len(ds_db) > 0:
        print(f"  Database (n={len(ds_db)}): median={ds_db.median():.0f}")

    # Research type
    rt = df["research_type"].value_counts()
    print(f"\nResearch type:")
    for k, v in rt.items():
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    # --- Pillar 2: Validation ---
    print("\n--- PILLAR 2: VALIDATION ---")

    vm = df["validation_method"].value_counts()
    print(f"\nValidation method:")
    for k, v in vm.items():
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    # R2 vs R
    mt = df["best_metric_type"].value_counts()
    print(f"\nMetric type:")
    for k, v in mt.items():
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    # Best metric value (R2 only)
    r2_vals = df.loc[df["best_metric_type"] == "R2", "best_metric_value"].dropna()
    print(f"\nR² values (n={len(r2_vals)}):")
    if len(r2_vals) > 0:
        print(f"  Median: {r2_vals.median():.3f}, Mean: {r2_vals.mean():.3f}")
        print(f"  >0.95: {(r2_vals > 0.95).sum()} ({(r2_vals > 0.95).sum()/len(r2_vals)*100:.1f}%)")
        print(f"  >0.90: {(r2_vals > 0.90).sum()} ({(r2_vals > 0.90).sum()/len(r2_vals)*100:.1f}%)")

    # Grouped splitting
    gs = df["grouped_splitting"].sum()
    print(f"\nGrouped splitting: {gs} ({gs/len(df)*100:.1f}%)")

    # External validation
    ev = df["external_validation"].sum()
    print(f"External validation: {ev} ({ev/len(df)*100:.1f}%)")

    # Reports train and test
    rtt = df["reports_train_and_test"].sum()
    print(f"Reports train & test: {rtt} ({rtt/len(df)*100:.1f}%)")

    # Hyperparameter tuning
    ht = df["hyperparameter_tuning"].value_counts()
    print(f"\nHyperparameter tuning:")
    for k, v in ht.items():
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    # --- Pillar 3: Comparability ---
    print("\n--- PILLAR 3: COMPARABILITY ---")

    # Algorithm frequency
    all_algos = []
    for algos in df["ml_algorithms"]:
        if isinstance(algos, list):
            all_algos.extend([standardize_algorithms(a) for a in algos])
    algo_counts = Counter(all_algos)
    print(f"\nTop 15 algorithms (total mentions: {len(all_algos)}):")
    for algo, cnt in algo_counts.most_common(15):
        print(f"  {algo}: {cnt}")

    # N algorithms compared
    nac = df["n_algorithms_compared"].dropna()
    print(f"\nAlgorithms compared per paper: median={nac.median():.0f}, mean={nac.mean():.1f}")

    # Code/data availability
    ca = df["code_available"].sum()
    da = df["data_available"].sum()
    print(f"\nCode available: {ca} ({ca/len(df)*100:.1f}%)")
    print(f"Data available: {da} ({da/len(df)*100:.1f}%)")

    # Compared with prior work
    cpw = df["compared_with_prior_work"].sum()
    print(f"Compared with prior work: {cpw} ({cpw/len(df)*100:.1f}%)")

    # --- Discussion: Interpretability + Deployment ---
    print("\n--- DISCUSSION: INTERPRETABILITY + DEPLOYMENT ---")

    # Interpretability
    all_interp = []
    for methods in df["interpretability_methods"]:
        if isinstance(methods, list):
            all_interp.extend(methods)
    interp_counts = Counter(all_interp)
    print(f"\nInterpretability methods:")
    for m, cnt in interp_counts.most_common():
        print(f"  {m}: {cnt}")

    md = df["mechanistic_discussion"].sum()
    print(f"Mechanistic discussion: {md} ({md/len(df)*100:.1f}%)")

    # Deployment
    wt = df["water_type"].value_counts()
    print(f"\nWater type:")
    for k, v in wt.items():
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    ds_flag = df["discusses_scalability"].sum()
    ev_flag = df["engineering_validation"].sum()
    cost = df["cost_analysis"].sum()
    print(f"Discusses scalability: {ds_flag} ({ds_flag/len(df)*100:.1f}%)")
    print(f"Engineering validation: {ev_flag} ({ev_flag/len(df)*100:.1f}%)")
    print(f"Cost analysis: {cost} ({cost/len(df)*100:.1f}%)")


# ============================================================
# Fig 2: Pillar 1 - Data Bias & Quality
# ============================================================

def fig2_data_bias(df):
    """
    Fig 2: The Fragile Foundation - Data Bias & Quality
    (a) Dataset size distribution (log scale histogram)
    (b) Data source breakdown (horizontal bar)
    (c) Top 15 pollutants (horizontal bar)
    (d) Top 15 materials (horizontal bar)
    """
    apply_plot_style()
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # --- (a) Dataset size distribution ---
    ax_a = fig.add_subplot(gs[0, 0])
    ds = df["dataset_size"].dropna()
    # Use log10 bins
    ds_pos = ds[ds > 0]
    log_vals = np.log10(ds_pos)
    bins = np.linspace(log_vals.min(), log_vals.max(), 20)
    ax_a.hist(log_vals, bins=bins, color=PILLAR_COLORS["data"],
              edgecolor="white", linewidth=0.5, alpha=0.85)
    # Apply gradient coloring to histogram bars
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'ds_grad', [CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[2], CATEGORICAL_PALETTE[4]])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm = plt.Normalize(bin_centers.min(), bin_centers.max())
    for patch, center in zip(ax_a.patches, bin_centers):
        patch.set_facecolor(cmap(norm(center)))
        patch.set_alpha(0.85)
    # Mark median
    med = np.log10(ds_pos.median())
    ax_a.axvline(med, color=ACCENT_COLORS["warning"], linestyle="--", linewidth=1.5)
    ax_a.text(med + 0.08, ax_a.get_ylim()[1] * 0.9,
              f"Median = {ds_pos.median():.0f}",
              color=ACCENT_COLORS["warning"], fontsize=FONT_SIZES["annotation"])
    ax_a.set_xlabel(r"Dataset Size ($\log_{10}$ scale)")
    ax_a.set_ylabel("Number of Papers")
    # Custom x-ticks
    tick_vals = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    tick_labels = ["10", "30", "100", "300", "1K", "3K", "10K", "30K", "100K"]
    valid = [(v, l) for v, l in zip(tick_vals, tick_labels) if log_vals.min() <= v <= log_vals.max() + 0.2]
    if valid:
        ax_a.set_xticks([v for v, _ in valid])
        ax_a.set_xticklabels([l for _, l in valid])
    style_axis(ax_a)
    add_panel_label(ax_a, "(a)")

    # --- (b) Data source breakdown ---
    ax_b = fig.add_subplot(gs[0, 1])
    src_order = ["experimental", "literature", "database", "mixed"]
    src_labels = ["Experimental", "Literature", "Database", "Mixed"]
    src_counts = [len(df[df["data_source"] == s]) for s in src_order]
    src_colors = [CATEGORICAL_PALETTE[i] for i in range(len(src_order))]
    bars = ax_b.barh(src_labels, src_counts, color=src_colors,
                     edgecolor="white", linewidth=0.5, height=0.6)
    for bar, cnt in zip(bars, src_counts):
        pct = cnt / len(df) * 100
        ax_b.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                  f"{cnt} ({pct:.0f}%)", va="center",
                  fontsize=FONT_SIZES["annotation"])
    ax_b.set_xlabel("Number of Papers")
    ax_b.invert_yaxis()
    style_axis(ax_b, grid_y=False)
    add_panel_label(ax_b, "(b)")

    # --- (c) Top 15 pollutants ---
    ax_c = fig.add_subplot(gs[1, 0])
    all_poll = []
    for polls in df["pollutants"]:
        if isinstance(polls, list):
            all_poll.extend([standardize_pollutants(p) for p in polls])
    poll_counts = Counter(all_poll).most_common(15)
    poll_names = [p for p, _ in reversed(poll_counts)]
    poll_vals = [c for _, c in reversed(poll_counts)]
    ax_c.barh(poll_names, poll_vals, color=PILLAR_COLORS["data"],
              edgecolor="white", linewidth=0.5, height=0.7)
    for i, (name, val) in enumerate(zip(poll_names, poll_vals)):
        ax_c.text(val + 0.3, i, str(val), va="center",
                  fontsize=FONT_SIZES["annotation"] - 1)
    ax_c.set_xlabel("Frequency")
    ax_c.set_title("Top 15 Pollutants / Adsorbates", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_c, grid_y=False)
    add_panel_label(ax_c, "(c)")

    # --- (d) Top 15 materials ---
    ax_d = fig.add_subplot(gs[1, 1])
    all_mat = []
    for mats in df["materials"]:
        if isinstance(mats, list):
            all_mat.extend([standardize_materials(m) for m in mats])
    mat_counts = Counter(all_mat).most_common(15)
    mat_names = [m for m, _ in reversed(mat_counts)]
    mat_vals = [c for _, c in reversed(mat_counts)]
    ax_d.barh(mat_names, mat_vals, color=PILLAR_COLORS["comparability"],
              edgecolor="white", linewidth=0.5, height=0.7)
    for i, (name, val) in enumerate(zip(mat_names, mat_vals)):
        ax_d.text(val + 0.3, i, str(val), va="center",
                  fontsize=FONT_SIZES["annotation"] - 1)
    ax_d.set_xlabel("Frequency")
    ax_d.set_title("Top 15 Adsorbent / Catalyst Materials", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_d, grid_y=False)
    add_panel_label(ax_d, "(d)")

    save_figure(fig, "Fig2_data_bias.png")
    print("Fig 2 saved.")


# ============================================================
# Fig 3: Pillar 2 - Validation Failures
# ============================================================

def fig3_validation(df):
    """
    Fig 3: The Leaky Pipeline - Validation Failures
    (a) R² / R value distribution (histogram with kernel density)
    (b) Validation method breakdown (waffle chart)
    (c) Dataset size vs best R² (scatter, colored by data_source)
    (d) R² by validation method (strip plot with medians)
    """
    apply_plot_style()
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # --- (a) R² distribution ---
    ax_a = fig.add_subplot(gs[0, 0])
    r2_vals = df.loc[df["best_metric_type"] == "R2", "best_metric_value"].dropna()
    r_vals = df.loc[df["best_metric_type"] == "R", "best_metric_value"].dropna()

    bins = np.arange(0.60, 1.005, 0.02)
    if len(r2_vals) > 0:
        ax_a.hist(r2_vals, bins=bins, color=CATEGORICAL_PALETTE[0],
                  edgecolor="white", linewidth=0.5, alpha=0.8, label=f"R² (n={len(r2_vals)})")
    if len(r_vals) > 0:
        ax_a.hist(r_vals, bins=bins, color='#E07B6C',
                  edgecolor="white", linewidth=0.5, alpha=0.6, label=f"R (n={len(r_vals)})")

    # Mark 0.90 threshold
    ax_a.axvline(0.90, color="#666666", linestyle=":", linewidth=2.0)
    ax_a.text(0.895, ax_a.get_ylim()[1] * 0.95, "R² = 0.90",
              ha="right", fontsize=FONT_SIZES["annotation"], color="#666666")

    # Annotation: % above 0.90
    if len(r2_vals) > 0:
        pct_above = (r2_vals > 0.90).sum() / len(r2_vals) * 100
        ax_a.text(0.78, ax_a.get_ylim()[1] * 0.55,
                  f"{pct_above:.0f}% > 0.90",
                  fontsize=FONT_SIZES["annotation"] + 1, fontweight="bold",
                  color="#000000")

    ax_a.set_xlabel("Best Reported Metric Value")
    ax_a.set_ylabel("Number of Papers")
    ax_a.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    style_axis(ax_a)
    add_panel_label(ax_a, "(a)")

    # --- (b) Validation method waffle chart (10×10) ---
    ax_b = fig.add_subplot(gs[0, 1])
    vm_order = ["random_split", "k_fold", "none_reported", "external"]
    vm_labels = ["Random Split", "K-Fold CV", "None Reported", "External"]
    vm_counts = []
    vm_labels_final = []
    vm_colors = []
    for i, (method, label) in enumerate(zip(vm_order, vm_labels)):
        cnt = len(df[df["validation_method"] == method])
        if cnt > 0:
            vm_counts.append(cnt)
            vm_labels_final.append(label)
            vm_colors.append(CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])

    total = sum(vm_counts)
    # Convert to 100-cell waffle proportions
    vm_pcts = [c / total * 100 for c in vm_counts]
    vm_cells = [round(p) for p in vm_pcts]
    diff = 100 - sum(vm_cells)
    if diff != 0:
        idx = vm_pcts.index(max(vm_pcts))
        vm_cells[idx] += diff

    cell_colors_b = []
    for c, color in zip(vm_cells, vm_colors):
        cell_colors_b.extend([color] * c)

    gap = 0.08
    for i, color in enumerate(cell_colors_b):
        row, col = divmod(i, 10)
        rect = plt.Rectangle((col + gap / 2, row + gap / 2),
                              1 - gap, 1 - gap,
                              facecolor=color, edgecolor='white', linewidth=0.5)
        ax_b.add_patch(rect)

    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.set_aspect('equal')
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    legend_labels_b = [f'{lab}: {cnt} ({cnt/total*100:.0f}%)'
                       for lab, cnt in zip(vm_labels_final, vm_counts)]
    legend_patches_b = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in vm_colors]
    ax_b.legend(legend_patches_b, legend_labels_b, loc='upper center',
                bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
                fontsize=FONT_SIZES['legend'] - 1)
    ax_b.set_title("Validation Strategy", fontsize=FONT_SIZES["axis_label"], pad=15)
    add_panel_label(ax_b, "(b)")

    # --- (c) R² by data source (strip plot with medians + significance) ---
    ax_c = fig.add_subplot(gs[1, 0])
    mask_r2 = (df["best_metric_type"] == "R2") & df["best_metric_value"].notna()
    plot_df = df[mask_r2].copy()

    src_order_c = ["Experimental", "Literature", "Database"]
    src_keys_c = ["experimental", "literature", "database"]
    src_colors_c = [CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[1], CATEGORICAL_PALETTE[2]]

    # Collect data per group
    group_data_c = []
    for src in src_keys_c:
        vals = plot_df.loc[plot_df["data_source"] == src, "best_metric_value"].dropna()
        group_data_c.append(vals.values)

    # Strip plot with median lines (same style as panel d)
    for i, (vals, color) in enumerate(zip(group_data_c, src_colors_c)):
        if len(vals) > 0:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
            ax_c.scatter(np.full(len(vals), i) + jitter, vals,
                        c=color, s=50, alpha=0.6, edgecolors="white",
                        linewidth=0.3, zorder=3)
            med = np.median(vals)
            ax_c.hlines(med, i - 0.3, i + 0.3, colors="#333333", linewidth=2.0, zorder=4)
            ax_c.text(i + 0.35, med, f"{med:.3f}",
                      fontsize=FONT_SIZES["annotation"] - 1, fontweight="bold",
                      color="#333333", va="center")

    # Mann-Whitney U tests: pairwise significance brackets
    comparisons = [(0, 1), (0, 2), (1, 2)]
    y_max = 1.02
    bracket_h = 0.012
    bracket_gap = 0.035
    for idx, (i, j) in enumerate(comparisons):
        if len(group_data_c[i]) > 0 and len(group_data_c[j]) > 0:
            _, p = mannwhitneyu(group_data_c[i], group_data_c[j], alternative='two-sided')
            label = _sig_label(p)
            if label != 'ns':
                y = y_max + idx * bracket_gap
                _add_sig_bracket(ax_c, i, j, y, bracket_h, label)

    ax_c.set_xticks(range(len(src_order_c)))
    ax_c.set_xticklabels(src_order_c, fontsize=FONT_SIZES["tick_label"])
    ax_c.set_ylabel(r"Best $\mathrm{R^2}$")
    ax_c.set_ylim(0.50, 1.15)
    ax_c.set_xlim(-0.5, len(src_order_c) - 0.5)
    ax_c.axhline(0.90, color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_c.set_title(r"$\mathrm{R^2}$ by Data Source", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_c)
    add_panel_label(ax_c, "(c)")

    # --- (d) R² by validation method (strip plot with medians + significance) ---
    ax_d = fig.add_subplot(gs[1, 1])
    vm_order_d = ["random_split", "k_fold", "none_reported"]
    vm_labels_d = ["Random\nSplit", "K-Fold\nCV", "None\nReported"]
    vm_colors_d = [CATEGORICAL_PALETTE[i] for i in range(len(vm_order_d))]

    mask_r2_d = df["best_metric_type"] == "R2"
    group_data_d = []
    for i, (method, label, color) in enumerate(zip(vm_order_d, vm_labels_d, vm_colors_d)):
        vals_d = df.loc[mask_r2_d & (df["validation_method"] == method), "best_metric_value"].dropna()
        group_data_d.append(vals_d.values)
        if len(vals_d) > 0:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals_d))
            ax_d.scatter(np.full(len(vals_d), i) + jitter, vals_d,
                        c=color, s=50, alpha=0.6, edgecolors="white",
                        linewidth=0.3, zorder=3)
            med = vals_d.median()
            ax_d.hlines(med, i - 0.3, i + 0.3, colors="#333333", linewidth=2.0, zorder=4)
            ax_d.text(i + 0.35, med, f"{med:.3f}",
                      fontsize=FONT_SIZES["annotation"] - 1, fontweight="bold",
                      color="#333333", va="center")

    # Pairwise significance brackets
    comparisons_d = [(0, 1), (0, 2)]
    y_max_d = 1.02
    bracket_h_d = 0.012
    bracket_gap_d = 0.035
    for idx, (i, j) in enumerate(comparisons_d):
        if len(group_data_d[i]) > 0 and len(group_data_d[j]) > 0:
            _, p = mannwhitneyu(group_data_d[i], group_data_d[j], alternative='two-sided')
            label = _sig_label(p)
            if label != 'ns':
                y = y_max_d + idx * bracket_gap_d
                _add_sig_bracket(ax_d, i, j, y, bracket_h_d, label)

    ax_d.set_xticks(range(len(vm_order_d)))
    ax_d.set_xticklabels(vm_labels_d, fontsize=FONT_SIZES["tick_label"] - 1)
    ax_d.set_ylabel(r"Best $\mathrm{R^2}$")
    ax_d.set_ylim(0.50, 1.15)
    ax_d.set_xlim(-0.5, len(vm_order_d) - 0.5)
    ax_d.axhline(0.90, color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_d.set_title(r"$\mathrm{R^2}$ by Validation Method", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_d)
    add_panel_label(ax_d, "(d)")

    save_figure(fig, "Fig3_validation.png")
    print("Fig 3 saved.")


# ============================================================
# Fig 4: Pillar 3 - Comparability Void
# ============================================================

def fig4_comparability(df):
    """
    Fig 4: The Benchmarking Void - Limited Standards, Uncertain Progress
    (a) Algorithm usage frequency (horizontal bar, top 15)
    (b) Evaluation metrics usage (horizontal bar)
    (c) Code & data availability (grouped bar)
    (d) Number of algorithms compared per paper (histogram)
    """
    apply_plot_style()
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # --- (a) Algorithm frequency ---
    ax_a = fig.add_subplot(gs[0, 0])
    all_algos = []
    for algos in df["ml_algorithms"]:
        if isinstance(algos, list):
            all_algos.extend([standardize_algorithms(a) for a in algos])
    algo_counts = Counter(all_algos).most_common(15)
    algo_names = [a for a, _ in reversed(algo_counts)]
    algo_vals = [c for _, c in reversed(algo_counts)]

    cmap_algo = mcolors.LinearSegmentedColormap.from_list(
        'algo_grad', [CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[2], CATEGORICAL_PALETTE[4]])
    norm_algo = plt.Normalize(0, len(algo_names) - 1)
    colors_algo = [cmap_algo(norm_algo(i)) for i in range(len(algo_names))]
    bars = ax_a.barh(algo_names, algo_vals, color=colors_algo,
                     edgecolor="white", linewidth=0.5, height=0.7)
    for i, (name, val) in enumerate(zip(algo_names, algo_vals)):
        ax_a.text(val + 0.5, i, str(val), va="center",
                  fontsize=FONT_SIZES["annotation"] - 1)
    ax_a.set_xlabel("Number of Papers Using Algorithm")
    ax_a.set_title("Top 15 ML Algorithms", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_a, grid_y=False)
    add_panel_label(ax_a, "(a)")

    # --- (b) Evaluation metrics ---
    ax_b = fig.add_subplot(gs[0, 1])
    all_metrics = []
    for metrics in df["evaluation_metrics"]:
        if isinstance(metrics, list):
            all_metrics.extend(metrics)
    metric_counts = Counter(all_metrics)
    # Keep top metrics, merge rest into "Other"
    main_metrics = ["R2", "RMSE", "MAE", "MSE", "MAPE", "R"]
    metric_names = []
    metric_vals = []
    other_count = 0
    for m in main_metrics:
        if m in metric_counts:
            display = r"$\mathrm{R^2}$" if m == "R2" else m
            metric_names.append(display)
            metric_vals.append(metric_counts[m])
    for m, c in metric_counts.most_common():
        if m not in main_metrics:
            other_count += c
    if other_count > 0:
        metric_names.append("Other")
        metric_vals.append(other_count)

    metric_names_r = list(reversed(metric_names))
    metric_vals_r = list(reversed(metric_vals))
    cmap_metric = mcolors.LinearSegmentedColormap.from_list(
        'metric_grad', [CATEGORICAL_PALETTE[4], CATEGORICAL_PALETTE[2], CATEGORICAL_PALETTE[0]])
    norm_metric = plt.Normalize(0, len(metric_names_r) - 1)
    colors_metric = [cmap_metric(norm_metric(i)) for i in range(len(metric_names_r))]
    bars = ax_b.barh(metric_names_r, metric_vals_r,
                     color=colors_metric,
                     edgecolor="white", linewidth=0.5, height=0.6)
    for i, val in enumerate(metric_vals_r):
        pct = val / len(df) * 100
        ax_b.text(val + 0.5, i, f"{val} ({pct:.0f}%)", va="center",
                  fontsize=FONT_SIZES["annotation"] - 1)
    ax_b.set_xlabel("Number of Papers")
    ax_b.set_title("Evaluation Metrics Used", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_b, grid_y=False)
    add_panel_label(ax_b, "(b)")

    # --- (c) Code & data availability + compared with prior work ---
    ax_c = fig.add_subplot(gs[1, 0])
    avail_items = {
        "Code\nAvailable": df["code_available"].sum(),
        "Data\nAvailable": df["data_available"].sum(),
        "Compared w/\nPrior Work": df["compared_with_prior_work"].sum(),
    }
    x_labels = list(avail_items.keys())
    yes_vals = list(avail_items.values())
    no_vals = [len(df) - v for v in yes_vals]
    x_pos = np.arange(len(x_labels))
    w = 0.35

    bars_yes = ax_c.bar(x_pos - w/2, yes_vals, w, color=CATEGORICAL_PALETTE[0],
                        edgecolor="white", linewidth=0.5, alpha=0.75, label="Yes")
    bars_no = ax_c.bar(x_pos + w/2, no_vals, w, color=CATEGORICAL_PALETTE[4],
                       edgecolor="white", linewidth=0.5, alpha=0.75, label="No")
    for bar, val in zip(bars_yes, yes_vals):
        pct = val / len(df) * 100
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f"{pct:.0f}%", ha="center", fontsize=FONT_SIZES["annotation"])
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(x_labels)
    ax_c.set_ylabel("Number of Papers")
    ax_c.legend(fontsize=FONT_SIZES["legend"], frameon=False)
    ax_c.set_title("Open Science Practices", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_c)
    add_panel_label(ax_c, "(c)")

    # --- (d) N algorithms compared ---
    ax_d = fig.add_subplot(gs[1, 1])
    nac = df["n_algorithms_compared"].dropna()
    bins_nac = np.arange(0.5, nac.max() + 1.5, 1)
    ax_d.hist(nac, bins=bins_nac, color=CATEGORICAL_PALETTE[0],
              edgecolor="white", linewidth=0.5, alpha=0.85)
    # Apply blue gradient to histogram bars
    cmap_nac = mcolors.LinearSegmentedColormap.from_list(
        'nac_grad', ['#A8C5D6', '#7EAAC4', '#5A8BA8'])
    bin_centers_nac = 0.5 * (bins_nac[:-1] + bins_nac[1:])
    norm_nac = plt.Normalize(bin_centers_nac.min(), bin_centers_nac.max())
    for patch, center in zip(ax_d.patches, bin_centers_nac):
        patch.set_facecolor(cmap_nac(norm_nac(center)))
        patch.set_alpha(0.85)
    med_nac = nac.median()
    ax_d.axvline(med_nac, color="#555555", linestyle="--", linewidth=1.5)
    ax_d.text(med_nac + 0.3, ax_d.get_ylim()[1] * 0.9,
              f"Median = {med_nac:.0f}",
              color="#555555", fontsize=FONT_SIZES["annotation"])
    ax_d.set_xlabel("Number of Algorithms Compared")
    ax_d.set_ylabel("Number of Papers")
    ax_d.set_title("Algorithm Comparison Breadth", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_d)
    add_panel_label(ax_d, "(d)")

    save_figure(fig, "Fig4_benchmarking.png")
    print("Fig 4 saved.")


# ============================================================
# Fig 5: Discussion - Interpretability + Deployment
# ============================================================

def fig5_discussion(df):
    """
    Fig 5: The Missing Link - From Metrics to Impact
    (a) Interpretability methods adoption (horizontal bar)
    (b) Mechanistic discussion rate (pie/donut)
    (c) Water type distribution (donut)
    (d) Deployment readiness radar (bar chart of 4 boolean rates)
    """
    apply_plot_style()
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # --- (a) Interpretability methods ---
    ax_a = fig.add_subplot(gs[0, 0])
    all_interp = []
    for methods in df["interpretability_methods"]:
        if isinstance(methods, list):
            for m in methods:
                if m and m.lower() != "none":
                    all_interp.append(m)
    interp_counts = Counter(all_interp)

    # Also count papers with NO interpretability
    n_none = sum(1 for methods in df["interpretability_methods"]
                 if isinstance(methods, list) and
                 (len(methods) == 0 or all(m.lower() == "none" for m in methods)))
    n_any = len(df) - n_none

    interp_order = ["feature_importance", "SHAP", "PDP", "sensitivity_analysis", "LIME"]
    interp_labels = []
    interp_vals = []
    for m in interp_order:
        if m in interp_counts:
            interp_labels.append(m.replace("_", " ").title())
            interp_vals.append(interp_counts[m])
    # Add remaining
    for m, c in interp_counts.most_common():
        if m not in interp_order:
            interp_labels.append(m.replace("_", " ").title())
            interp_vals.append(c)

    interp_labels_r = list(reversed(interp_labels))
    interp_vals_r = list(reversed(interp_vals))
    cmap_interp = mcolors.LinearSegmentedColormap.from_list(
        'interp_grad', [CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[2], CATEGORICAL_PALETTE[4]])
    norm_interp = plt.Normalize(0, len(interp_labels_r) - 1)
    colors_interp = [cmap_interp(norm_interp(i)) for i in range(len(interp_labels_r))]
    bars = ax_a.barh(interp_labels_r, interp_vals_r, color=colors_interp,
                     edgecolor="white", linewidth=0.5, height=0.6)
    for i, val in enumerate(interp_vals_r):
        pct = val / len(df) * 100
        ax_a.text(val + 0.5, i, f"{val} ({pct:.0f}%)", va="center",
                  fontsize=FONT_SIZES["annotation"])
    ax_a.set_xlabel("Number of Papers")
    ax_a.set_title(f"Interpretability Methods (any: {n_any}/{len(df)}, {n_any/len(df)*100:.0f}%)",
                   fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_a, grid_y=False)
    add_panel_label(ax_a, "(a)")

    # --- (b) Mechanistic discussion waffle chart (10×10) ---
    ax_b = fig.add_subplot(gs[0, 1])
    md_yes = int(df["mechanistic_discussion"].sum())
    md_no = len(df) - md_yes
    total_md = len(df)
    labels_md = ["Yes", "No"]
    counts_md = [md_yes, md_no]
    colors_md = [CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[4]]

    pcts_md = [c / total_md * 100 for c in counts_md]
    cells_md = [round(p) for p in pcts_md]
    diff_md = 100 - sum(cells_md)
    if diff_md != 0:
        idx_md = pcts_md.index(max(pcts_md))
        cells_md[idx_md] += diff_md

    cell_colors_md = []
    for c, color in zip(cells_md, colors_md):
        cell_colors_md.extend([color] * c)

    gap = 0.08
    for i, color in enumerate(cell_colors_md):
        row, col = divmod(i, 10)
        rect = plt.Rectangle((col + gap / 2, row + gap / 2),
                              1 - gap, 1 - gap,
                              facecolor=color, edgecolor='white', linewidth=0.5)
        ax_b.add_patch(rect)

    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.set_aspect('equal')
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    legend_labels_md = [f'{lab}: {cnt} ({cnt/total_md*100:.0f}%)'
                        for lab, cnt in zip(labels_md, counts_md)]
    legend_patches_md = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors_md]
    ax_b.legend(legend_patches_md, legend_labels_md, loc='upper center',
                bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
                fontsize=FONT_SIZES['legend'])
    ax_b.set_title("Connects ML Results to\nKnown Mechanisms?",
                   fontsize=FONT_SIZES["axis_label"], pad=15)
    add_panel_label(ax_b, "(b)")

    # --- (c) Water type waffle chart (10×10) ---
    ax_c = fig.add_subplot(gs[1, 0])
    wt_order = ["synthetic", "real_wastewater", "both", "not_specified", "not_applicable"]
    wt_labels = ["Synthetic", "Real WW", "Both", "Not Spec.", "N/A"]
    wt_counts = []
    wt_labels_final = []
    wt_colors = []
    for i, (wt, label) in enumerate(zip(wt_order, wt_labels)):
        cnt = len(df[df["water_type"] == wt])
        if cnt > 0:
            wt_counts.append(cnt)
            wt_labels_final.append(label)
            wt_colors.append(CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])

    total_wt = sum(wt_counts)
    pcts_wt = [c / total_wt * 100 for c in wt_counts]
    cells_wt = [round(p) for p in pcts_wt]
    diff_wt = 100 - sum(cells_wt)
    if diff_wt != 0:
        idx_wt = pcts_wt.index(max(pcts_wt))
        cells_wt[idx_wt] += diff_wt

    cell_colors_wt = []
    for c, color in zip(cells_wt, wt_colors):
        cell_colors_wt.extend([color] * c)

    gap = 0.08
    for i, color in enumerate(cell_colors_wt):
        row, col = divmod(i, 10)
        rect = plt.Rectangle((col + gap / 2, row + gap / 2),
                              1 - gap, 1 - gap,
                              facecolor=color, edgecolor='white', linewidth=0.5)
        ax_c.add_patch(rect)

    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.set_aspect('equal')
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_visible(False)

    legend_labels_wt = [f'{lab}: {cnt} ({cnt/total_wt*100:.0f}%)'
                        for lab, cnt in zip(wt_labels_final, wt_counts)]
    legend_patches_wt = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in wt_colors]
    ax_c.legend(legend_patches_wt, legend_labels_wt, loc='upper center',
                bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
                fontsize=FONT_SIZES['legend'])
    ax_c.set_title("Water Type in Experiments", fontsize=FONT_SIZES["axis_label"], pad=15)
    add_panel_label(ax_c, "(c)")

    # --- (d) Deployment readiness ---
    ax_d = fig.add_subplot(gs[1, 1])
    deploy_items = {
        "Discusses\nScalability": df["discusses_scalability"].sum(),
        "Engineering\nValidation": df["engineering_validation"].sum(),
        "Cost\nAnalysis": df["cost_analysis"].sum(),
        "Real\nWastewater": len(df[df["water_type"].isin(["real_wastewater", "both"])]),
    }
    d_names = list(deploy_items.keys())
    d_vals = list(deploy_items.values())
    d_pcts = [v / len(df) * 100 for v in d_vals]

    cmap_deploy = mcolors.LinearSegmentedColormap.from_list(
        'deploy_grad', ['#F0B27A', '#D4897C', '#C0605A'])
    norm_deploy = plt.Normalize(0, len(d_names) - 1)
    colors_deploy = [cmap_deploy(norm_deploy(i)) for i in range(len(d_names))]
    bars = ax_d.bar(d_names, d_pcts, color=colors_deploy,
                    edgecolor="white", linewidth=0.5, width=0.6, alpha=0.8)
    for bar, pct, val in zip(bars, d_pcts, d_vals):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                  f"{pct:.0f}%\n(n={val})", ha="center",
                  fontsize=FONT_SIZES["annotation"], va="bottom")
    ax_d.set_ylabel("Percentage of Papers (%)")
    ax_d.set_ylim(0, max(d_pcts) * 1.3 + 10)
    ax_d.set_title("Deployment Readiness Indicators", fontsize=FONT_SIZES["axis_label"])
    style_axis(ax_d)
    add_panel_label(ax_d, "(d)")

    save_figure(fig, "Fig6_deployment.png")
    print("Fig 6 saved.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ML4Env Statistical Analysis & Visualization")
    parser.add_argument("--fig", type=int, help="Generate specific figure (2-5)")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")
    args = parser.parse_args()

    df = load_data()

    if args.stats:
        print_stats(df)
        return

    if args.fig:
        print_stats(df)
        fig_map = {2: fig2_data_bias, 3: fig3_validation,
                   4: fig4_comparability, 5: fig5_discussion}
        if args.fig in fig_map:
            fig_map[args.fig](df)
        else:
            print(f"Unknown figure: {args.fig}. Available: 2-5")
        return

    # Generate all
    print_stats(df)
    fig2_data_bias(df)
    fig3_validation(df)
    fig4_comparability(df)
    fig5_discussion(df)
    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
