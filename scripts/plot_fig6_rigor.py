"""
Plot Fig 5: Methodological Rigor vs. Reported Performance.

Four-panel figure (2×2) for ES&T revision:
  (a) Radar chart — adoption rate for each of the 9 rigor dimensions
  (b) Rigor heatmap — papers sorted by R², columns = rigor dimensions
  (c) Rigor score vs R² boxplot with significance brackets
  (d) Per-dimension forest plot (median R² difference with 95% CI)

Usage:
    python scripts/plot_fig6_rigor.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.stats import mannwhitneyu

# Import shared plot config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import (
    apply_plot_style, add_panel_label, style_axis, save_figure,
    PILLAR_COLORS, ACCENT_COLORS, CATEGORICAL_PALETTE,
    FONT_SIZES, LINE_WIDTHS,
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "public" / "figure_data" / "fig6_rigor.json"


def load_data():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def _sig_label(pv):
    """Return significance label from p-value."""
    if pv < 0.001:
        return '***'
    elif pv < 0.01:
        return '**'
    elif pv < 0.05:
        return '*'
    return 'ns'


def plot_fig5(data, font_sizes=None):
    """Generate the 4-panel (2×2) rigor figure."""
    fs = font_sizes or FONT_SIZES

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

    # -------------------------------------------------------
    # Panel (a): Radar chart — dimension adoption rates
    # -------------------------------------------------------
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)

    dim_results = data["dimension_results"]
    dim_keys = list(dim_results.keys())
    dim_labels = [dim_results[k]["label"] for k in dim_keys]
    n_total = dim_results[dim_keys[0]]["n_yes"] + dim_results[dim_keys[0]]["n_no"]
    adoption_pcts = [dim_results[k]["n_yes"] / n_total * 100 for k in dim_keys]

    n_dims = len(dim_keys)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    values = adoption_pcts + adoption_pcts[:1]

    ax_radar.plot(angles, values, 'o-', color=CATEGORICAL_PALETTE[0],
                  linewidth=2, markersize=5)
    ax_radar.fill(angles, values, color=CATEGORICAL_PALETTE[0], alpha=0.25)

    # Add percentage labels at each vertex
    for angle, val, label in zip(angles[:-1], adoption_pcts, dim_labels):
        ha = 'left' if 0 < angle < np.pi else 'right'
        if abs(angle) < 0.1 or abs(angle - np.pi) < 0.1:
            ha = 'center'
        ax_radar.text(angle, val + 8, f'{val:.1f}%',
                      ha='center', va='center',
                      fontsize=fs['annotation'] - 1, fontweight='bold',
                      color='#333')

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(dim_labels, fontsize=fs['tick_label'] - 1)
    ax_radar.set_ylim(0, 70)
    ax_radar.set_yticks([20, 40, 60])
    ax_radar.set_yticklabels(['20%', '40%', '60%'], fontsize=fs['tick_label'] - 2,
                             color='#888')
    ax_radar.spines['polar'].set_color('#CCCCCC')
    ax_radar.grid(color='#DDDDDD', linewidth=0.5)

    # Panel label for polar axes
    ax_radar.text(-0.05, 1.10, '(a)', transform=ax_radar.transAxes,
                  fontsize=fs['panel_label'], fontweight='bold',
                  va='top', ha='left')

    # -------------------------------------------------------
    # Panel (b): Aggregated heatmap — R² tiers × dimensions
    # -------------------------------------------------------
    ax_heat = fig.add_subplot(gs[0, 1])
    hm = data["heatmap_data"]
    matrix = np.array(hm["matrix"], dtype=float)
    r2_vals = np.array(hm["r2_values"])

    # Define R² tiers (matching panel c grouping logic)
    tier_labels = ['>0.99', '0.95–0.99', '0.90–0.95', '<0.90']
    tier_bounds = [(0.99, 1.01), (0.95, 0.99), (0.90, 0.95), (0.0, 0.90)]

    n_dims = matrix.shape[1]
    adoption_matrix = np.zeros((len(tier_labels), n_dims))
    tier_ns = []

    for t_idx, (lo, hi) in enumerate(tier_bounds):
        mask = (r2_vals > lo) if lo == 0.0 else (r2_vals > lo) & (r2_vals <= hi)
        # Special case: >0.99 means R² > 0.99
        if t_idx == 0:
            mask = r2_vals > 0.99
        elif t_idx == 3:
            mask = r2_vals <= 0.90
        n_tier = mask.sum()
        tier_ns.append(n_tier)
        if n_tier > 0:
            adoption_matrix[t_idx] = matrix[mask].mean(axis=0) * 100

    # Custom colormap: white → CATEGORICAL_PALETTE[0]
    cmap_adopt = LinearSegmentedColormap.from_list(
        'adoption', ['#FFFFFF', CATEGORICAL_PALETTE[0]])

    im = ax_heat.imshow(adoption_matrix, aspect='auto', cmap=cmap_adopt,
                        vmin=0, vmax=100, interpolation='nearest')

    # Annotate each cell with percentage
    for i in range(len(tier_labels)):
        for j in range(n_dims):
            val = adoption_matrix[i, j]
            text_color = 'white' if val > 55 else '#333'
            ax_heat.text(j, i, f'{val:.0f}%', ha='center', va='center',
                         fontsize=fs['annotation'] - 1, fontweight='bold',
                         color=text_color)

    ax_heat.set_xticks(range(n_dims))
    ax_heat.set_xticklabels(hm["dimension_shorts"],
                            fontsize=fs['tick_label'] - 1, rotation=45, ha='right')
    y_labels = [f'{lab}\n(n={n})' for lab, n in zip(tier_labels, tier_ns)]
    ax_heat.set_yticks(range(len(tier_labels)))
    ax_heat.set_yticklabels(y_labels, fontsize=fs['tick_label'] - 1)
    ax_heat.set_ylabel('R² tier', fontsize=fs['axis_label'])

    # Color bar
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.set_label('Adoption rate (%)', fontsize=fs['axis_label'] - 1)
    cbar.ax.tick_params(labelsize=fs['tick_label'] - 2)

    add_panel_label(ax_heat, '(b)', fs)

    # -------------------------------------------------------
    # Panel (c): Boxplot — rigor score groups vs R²
    # -------------------------------------------------------
    ax_box = fig.add_subplot(gs[1, 0])
    group_stats = data["group_stats"]
    group_names = ["0-2", "3-4", "5-6", "7-9"]
    group_data = []
    group_ns = []
    for g in group_names:
        vals = group_stats[g].get("r2_values", [])
        group_data.append(vals)
        group_ns.append(len(vals))

    box_colors = [CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[1],
                  CATEGORICAL_PALETTE[2], CATEGORICAL_PALETTE[3]]

    bp = ax_box.boxplot(group_data, positions=range(len(group_names)),
                        widths=0.6, patch_artist=True,
                        medianprops=dict(color='#333', linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#666')
        patch.set_linewidth(0.8)
        patch.set_alpha(0.85)

    # Overlay jitter points on boxplot
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(group_data, box_colors)):
        if len(vals) > 0:
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax_box.scatter(np.full(len(vals), i) + jitter, vals,
                           color=color, edgecolors='#555', linewidth=0.5,
                           s=20, alpha=0.7, zorder=5)

    for i, (g, n) in enumerate(zip(group_names, group_ns)):
        ax_box.text(i, 0.48, f'n={n}', ha='center',
                    fontsize=fs['annotation'] - 1, color='#666')

    ax_box.set_xticks(range(len(group_names)))
    ax_box.set_xticklabels(group_names, fontsize=fs['tick_label'])
    ax_box.set_xlabel('Rigor score', fontsize=fs['axis_label'])
    ax_box.set_ylabel('Reported R²', fontsize=fs['axis_label'])

    # Spearman annotation
    rho = data["overall"]["spearman_rho"]
    p = data["overall"]["spearman_p"]
    p_str = "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"
    ax_box.text(0.95, 0.15, f'ρ = {rho:.3f}\n{p_str}',
                transform=ax_box.transAxes, ha='right', va='bottom',
                fontsize=fs['annotation'], style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.9))

    # Pairwise significance brackets (0-2 vs higher groups)
    bracket_comparisons = [(0, 1), (0, 2), (0, 3)]
    y_base = 1.03
    bracket_h = 0.012
    bracket_gap = 0.035
    for idx, (i, j) in enumerate(bracket_comparisons):
        if len(group_data[i]) > 1 and len(group_data[j]) > 1:
            _, pv = mannwhitneyu(group_data[i], group_data[j], alternative='two-sided')
            label = _sig_label(pv)
            if label != 'ns':
                y = y_base + idx * bracket_gap
                ax_box.plot([i, i, j, j], [y, y + bracket_h, y + bracket_h, y],
                            color='#333333', linewidth=1.0, clip_on=False)
                ax_box.text((i + j) / 2, y + bracket_h, label, ha='center', va='bottom',
                            fontsize=fs['annotation'], fontweight='bold', color='#333333')

    ax_box.set_ylim(0.45, 1.15)
    style_axis(ax_box)
    add_panel_label(ax_box, '(c)', fs)

    # -------------------------------------------------------
    # Panel (d): Forest plot — per-dimension ΔMedian R²
    # -------------------------------------------------------
    ax_forest = fig.add_subplot(gs[1, 1])

    dim_items = [(k, v) for k, v in dim_results.items()
                 if v["median_diff"] is not None]
    dim_items.sort(key=lambda x: x[1]["median_diff"])

    y_pos = range(len(dim_items))
    labels = []
    diffs = []
    ci_lows = []
    ci_highs = []
    p_vals = []

    for k, v in dim_items:
        labels.append(v["label"])
        diffs.append(v["median_diff"])
        ci_lows.append(v["ci_lower"])
        ci_highs.append(v["ci_upper"])
        p_vals.append(v["mann_whitney_p"])

    diffs = np.array(diffs)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    for i, (d, cl, ch, pv) in enumerate(zip(diffs, ci_lows, ci_highs, p_vals)):
        color = CATEGORICAL_PALETTE[0] if pv < 0.05 else '#999999'
        ax_forest.plot([cl, ch], [i, i], color=color, linewidth=2, zorder=2)
        ax_forest.plot(d, i, 'o', color=color, markersize=7, zorder=3)
        if pv < 0.05:
            ax_forest.text(ch + 0.002, i, '*', fontsize=fs['annotation'],
                           fontweight='bold', color=color, va='center')

    ax_forest.axvline(x=0, color='#999', linestyle='--', linewidth=0.8, zorder=1)

    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels(labels, fontsize=fs['tick_label'] - 1)
    ax_forest.set_xlabel('ΔMedian R² (with − without)', fontsize=fs['axis_label'])
    ax_forest.invert_yaxis()

    sig_patch = mpatches.Patch(facecolor=CATEGORICAL_PALETTE[0], label='p < 0.05')
    ns_patch = mpatches.Patch(facecolor='#999999', label='p ≥ 0.05')
    ax_forest.legend(handles=[sig_patch, ns_patch], loc='lower left',
                     fontsize=fs['legend'], framealpha=0.9)

    style_axis(ax_forest, grid_y=False)
    ax_forest.xaxis.grid(True, linestyle='--', alpha=0.3,
                         color='#CCCCCC', linewidth=LINE_WIDTHS['grid'])
    add_panel_label(ax_forest, '(d)', fs)

    fig.suptitle('')  # No title for journal submission
    return fig


def main():
    apply_plot_style()
    data = load_data()

    fig = plot_fig5(data, font_sizes=FONT_SIZES)
    save_figure(fig, 'Fig5_rigor_analysis.png')

    # Export data summary CSV
    import csv
    csv_file = BASE_DIR / "data" / "public" / "figure_data" / "fig6_data_summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "label", "n_yes", "n_no",
                         "median_yes", "median_no", "median_diff",
                         "ci_lower", "ci_upper", "mann_whitney_p"])
        for k, v in data["dimension_results"].items():
            writer.writerow([k, v["label"], v["n_yes"], v["n_no"],
                             v["median_yes"], v["median_no"], v["median_diff"],
                             v["ci_lower"], v["ci_upper"], v["mann_whitney_p"]])
    print(f"Saved: {csv_file}")


if __name__ == "__main__":
    main()
