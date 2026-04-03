"""
ML4Env Critical Review - Bibliometric Analysis (Fig 1, formerly Fig 2)
Input:  data/public/ml4env_155_dataset.json
Output: figures/Fig1_bibliometric_overview.png (multi-panel)

Panels:
  (a) Publication trend by year, stacked by field
  (b) Journal distribution (treemap)
  (c) Field distribution (donut chart)
  (d) Citation distribution (gradient histogram)
  (e) Open access ratio by year (stacked bar)
  (f) Cumulative growth curve
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import squarify
from collections import Counter
from pathlib import Path

from plot_config import (
    FIELD_COLORS, CATEGORICAL_PALETTE, FONT_SIZES,
    LINE_WIDTHS, FIGURE_DIR, DPI,
    apply_plot_style, add_panel_label, style_axis, save_figure,
)

# ============================================================
# Load Data (from pre-cleaned 155-paper public dataset)
# ============================================================

DATA_FILE = Path(__file__).parent.parent / 'data' / 'public' / 'ml4env_126_dataset.json'

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    papers = json.load(f)

print(f"Loaded {len(papers)} papers from {DATA_FILE.name}")

# ============================================================
# Preprocess
# ============================================================

years = [p['year'] for p in papers]
fields = [p.get('research_type', 'unknown') for p in papers]
journals = [p.get('journal', 'Unknown') for p in papers]
citations = [p.get('cited_by_count', 0) for p in papers]
is_oa = [p.get('is_oa', False) for p in papers]

YEAR_RANGE = range(2018, 2026)

# Year x field counts
year_field = {}
for yr in YEAR_RANGE:
    year_field[yr] = {'adsorption': 0, 'degradation': 0, 'both': 0}
for p in papers:
    yr = p['year']
    fld = p.get('research_type', 'unknown')
    if yr in year_field and fld in year_field[yr]:
        year_field[yr][fld] += 1

# OA by year
year_oa = {}
for yr in YEAR_RANGE:
    year_oa[yr] = {'oa': 0, 'closed': 0}
for p in papers:
    yr = p['year']
    if yr in year_oa:
        if p.get('is_oa', False):
            year_oa[yr]['oa'] += 1
        else:
            year_oa[yr]['closed'] += 1

# ============================================================
# Plot
# ============================================================

apply_plot_style()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.subplots_adjust(hspace=0.35, wspace=0.30)

# ------ (a) Publication trend by year, stacked by field ------
ax = axes[0, 0]
x = np.array(list(YEAR_RANGE))
ads = [year_field[yr]['adsorption'] for yr in YEAR_RANGE]
deg = [year_field[yr]['degradation'] for yr in YEAR_RANGE]
both = [year_field[yr]['both'] for yr in YEAR_RANGE]

ax.bar(x, ads, width=0.7, label='Adsorption',
       color=FIELD_COLORS['adsorption'], edgecolor='white', linewidth=0.5)
ax.bar(x, deg, width=0.7, bottom=ads, label='Degradation',
       color=FIELD_COLORS['degradation'], edgecolor='white', linewidth=0.5)
ax.bar(x, both, width=0.7, bottom=np.array(ads) + np.array(deg), label='Both',
       color=FIELD_COLORS['both'], edgecolor='white', linewidth=0.5)

totals = [a + d + b for a, d, b in zip(ads, deg, both)]
for xi, t in zip(x, totals):
    if t > 0:
        ax.text(xi, t + 0.5, str(t), ha='center', va='bottom',
                fontsize=FONT_SIZES['annotation'] - 2, fontweight='bold')

ax.set_xlabel('Year')
ax.set_ylabel('Number of Publications')
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEAR_RANGE], rotation=45, ha='right')
ax.legend(fontsize=FONT_SIZES['legend'] - 1, frameon=False)
style_axis(ax)
add_panel_label(ax, '(a)')

# ------ (b) Journal distribution (treemap) ------
ax = axes[0, 1]
journal_counts = Counter(journals)
top_n = 15
top_journals = journal_counts.most_common(top_n)

# Journal abbreviations (initials only, see figure_captions.md for key)
JOURNAL_ABBREV = {
    'Scientific Reports': 'SR',
    'Water': 'Water',
    'The Journal of Physical Chemistry C': 'JPCC',
    'Sustainability': 'Sustain.',
    'Molecules': 'Mol.',
    'Environmental Engineering Research': 'EER',
    'Nanomaterials': 'NM',
    'Carbon Research': 'CR',
    'Energies': 'Energ.',
    'Catalysts': 'Catal.',
    'Journal of Hazardous Materials': 'JHM',
    'Chemosphere': 'CS',
    'Journal of Environmental Chemical Engineering': 'JECE',
    'Chemical Engineering Journal': 'CEJ',
    'Environmental Science and Pollution Research': 'ESPR',
    'Journal of Cleaner Production': 'JCP',
    'Science of The Total Environment': 'STOTEN',
    'ACS Omega': 'ACSO',
    'RSC Advances': 'RSCA',
    'Applied Sciences': 'AS',
    'Journal of Water Process Engineering': 'JWPE',
    'Environmental Research': 'ER',
    'Desalination and Water Treatment': 'DWT',
    'Arabian Journal of Chemistry': 'AJC',
    'Bioresource Technology': 'BT',
    'Journal of Molecular Liquids': 'JML',
    'Colloids and Surfaces A: Physicochemical and Engineering Aspects': 'CSA',
    'International Journal of Molecular Sciences': 'IJMS',
    'Materials': 'Mat.',
    'Results in Engineering': 'RE',
    'Journal of Environmental Management': 'JEM',
    'Separation and Purification Technology': 'SPT',
    'Applied Water Science': 'AWS',
    'Journal of Chemical Theory and Computation': 'JCTC',
    'Biomass Conversion and Biorefinery': 'BCB',
}

labels = []
sizes = []
colors = []
for i, (name, count) in enumerate(top_journals):
    abbrev = JOURNAL_ABBREV.get(name, name[:20])
    labels.append(f"{abbrev}\n{count}")
    sizes.append(count)
    colors.append(CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])

# Remove axis spines for treemap
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85,
              ax=ax, text_kwargs={'fontsize': 11, 'fontweight': 'bold'},
              edgecolor='white', linewidth=2)
add_panel_label(ax, '(b)')

# ------ (c) Field distribution (waffle chart, 10×10) ------
ax = axes[0, 2]
field_counts = Counter(fields)
labels_c = ['Adsorption', 'Degradation', 'Both']
sizes_c = [field_counts.get('adsorption', 0),
           field_counts.get('degradation', 0),
           field_counts.get('both', 0)]
colors_c = [FIELD_COLORS['adsorption'], FIELD_COLORS['degradation'], FIELD_COLORS['both']]
total = sum(sizes_c)

# Convert to 100-cell waffle proportions
pcts = [s / total * 100 for s in sizes_c]
cells = [round(p) for p in pcts]
# Adjust rounding to ensure exactly 100 cells
diff = 100 - sum(cells)
if diff != 0:
    idx = pcts.index(max(pcts))
    cells[idx] += diff

# Build color grid (fill left-to-right, bottom-to-top)
cell_colors = []
for c, color in zip(cells, colors_c):
    cell_colors.extend([color] * c)

gap = 0.08
for i, color in enumerate(cell_colors):
    row, col = divmod(i, 10)
    rect = plt.Rectangle((col + gap / 2, row + gap / 2),
                          1 - gap, 1 - gap,
                          facecolor=color, edgecolor='white', linewidth=0.5)
    ax.add_patch(rect)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Legend below chart
legend_labels = [f'{lab}: {sz} ({sz/total*100:.1f}%)'
                 for lab, sz in zip(labels_c, sizes_c)]
legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors_c]
ax.legend(legend_patches, legend_labels, loc='upper center',
          bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
          fontsize=FONT_SIZES['legend'])
add_panel_label(ax, '(c)')

# ------ (d) Citation distribution (gradient histogram) ------
ax = axes[1, 0]
median_c = np.median(citations)
mean_c = np.mean(citations)

# Create gradient-colored histogram
bin_edges = np.arange(0, max(citations) + 20, 15)
n_vals, bins, patches = ax.hist(citations, bins=bin_edges,
                                 edgecolor='white', linewidth=0.5)

# Color gradient: dusty blue -> warm amber -> terracotta
cmap = mcolors.LinearSegmentedColormap.from_list(
    'cite_grad', [CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[2], CATEGORICAL_PALETTE[4]]
)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
norm = plt.Normalize(bin_centers.min(), bin_centers.max())
for patch, center in zip(patches, bin_centers):
    patch.set_facecolor(cmap(norm(center)))
    patch.set_alpha(0.85)

ax.set_xlabel('Cited-by Count')
ax.set_ylabel('Number of Papers')
style_axis(ax)

ax.axvline(median_c, color='#555555', linestyle='--',
           linewidth=1.2, label=f'Median = {median_c:.0f}')
ax.axvline(mean_c, color='#555555', linestyle='-.',
           linewidth=1.2, label=f'Mean = {mean_c:.1f}')
ax.legend(fontsize=FONT_SIZES['legend'] - 1, frameon=False)
add_panel_label(ax, '(d)')

# ------ (e) Open access ratio by year ------
ax = axes[1, 1]
oa_vals = [year_oa[yr]['oa'] for yr in YEAR_RANGE]
closed_vals = [year_oa[yr]['closed'] for yr in YEAR_RANGE]

ax.bar(x, oa_vals, width=0.7, label='Open Access',
       color=CATEGORICAL_PALETTE[1], edgecolor='white', linewidth=0.5)
ax.bar(x, closed_vals, width=0.7, bottom=oa_vals, label='Closed',
       color=CATEGORICAL_PALETTE[8], edgecolor='white', linewidth=0.5)

for xi, oa, cl in zip(x, oa_vals, closed_vals):
    total = oa + cl
    if total > 0:
        pct = oa / total * 100
        ax.text(xi, total + 0.3, f'{pct:.0f}%', ha='center', va='bottom',
                fontsize=FONT_SIZES['annotation'] - 3, color='#000000')

ax.set_xlabel('Year')
ax.set_ylabel('Number of Publications')
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEAR_RANGE], rotation=45, ha='right')
ax.legend(fontsize=FONT_SIZES['legend'] - 1, frameon=False)
style_axis(ax)
add_panel_label(ax, '(e)')

# ------ (f) Cumulative growth curve ------
ax = axes[1, 2]
yearly_totals = [sum([year_field[yr][f] for f in ['adsorption', 'degradation', 'both']])
                 for yr in YEAR_RANGE]
cumulative = np.cumsum(yearly_totals)

ax.fill_between(x, cumulative, alpha=0.3, color=CATEGORICAL_PALETTE[0])
ax.plot(x, cumulative, 'o-', color=CATEGORICAL_PALETTE[0],
        linewidth=LINE_WIDTHS['line'], markersize=6)

for xi, c in zip(x, cumulative):
    ax.text(xi, c + 2, str(c), ha='center', va='bottom',
            fontsize=FONT_SIZES['annotation'] - 2, fontweight='bold')

ax.set_xlabel('Year')
ax.set_ylabel('Cumulative Publications')
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEAR_RANGE], rotation=45, ha='right')
ax.set_ylim(0, max(cumulative) * 1.12)
style_axis(ax)
ax.text(-0.18, 1.05, '(f)', transform=ax.transAxes,
        fontsize=FONT_SIZES['panel_label'], fontweight='bold',
        va='top', ha='left')

# ============================================================
# Save
# ============================================================

save_figure(fig, 'Fig1_bibliometric_overview.png')

# ============================================================
# Print Summary Statistics
# ============================================================

print("\n" + "=" * 50)
print(f"BIBLIOMETRIC SUMMARY (n={len(papers)})")
print("=" * 50)
print(f"\nYear distribution:")
for yr in YEAR_RANGE:
    t = sum(year_field[yr].values())
    print(f"  {yr}: {t} papers")

print(f"\nField distribution:")
for fld, cnt in field_counts.most_common():
    print(f"  {fld}: {cnt} ({cnt/len(papers)*100:.1f}%)")

print(f"\nTop 10 journals:")
for j, cnt in journal_counts.most_common(10):
    print(f"  {j}: {cnt}")

print(f"\nCitation statistics:")
print(f"  Mean: {mean_c:.1f}")
print(f"  Median: {median_c:.0f}")
print(f"  Max: {max(citations)}")
print(f"  Min: {min(citations)}")

oa_total = sum(1 for p in papers if p.get('is_oa', False))
print(f"\nOpen Access: {oa_total}/{len(papers)} ({oa_total/len(papers)*100:.1f}%)")
