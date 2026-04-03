"""
Fig. S1: LLM Benchmark — Human vs. 4 LLMs extraction comparison.
4-panel figure: (a) Overall Score, (b) Time, (c) Cost, (d) Accuracy breakdown.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plot_config import (
    apply_plot_style, add_panel_label, style_axis, save_figure,
    FONT_SIZES_SUBMIT, LINE_WIDTHS, DPI, CATEGORICAL_PALETTE,
)

# ============================================================
# Data
# ============================================================

EXTRACTORS = ['Human', 'Claude Sonnet 4.6', 'Qwen3.5-Plus', 'Gemini 3.1 Pro', 'GPT-5.1']

COLORS = {
    'Human':              '#B0B0B0',
    'Claude Sonnet 4.6':  CATEGORICAL_PALETTE[0],
    'Qwen3.5-Plus':       CATEGORICAL_PALETTE[1],
    'Gemini 3.1 Pro':     CATEGORICAL_PALETTE[2],
    'GPT-5.1':            CATEGORICAL_PALETTE[3],
}

# Overall accuracy score (out of 100)
OVERALL = [100.0, 88.4, 83.4, 82.7, 81.1]

# Time per paper (seconds)
TIME_SEC = [900.0, 20.0, 57.3, 48.6, 6.2]

# Cost per paper (USD)
COST_USD = [2.570, 0.080, 0.002, 0.031, 0.113]

# Accuracy breakdown (%)
CAT_ACC  = [100.0, 86.7, 76.7, 77.8, 76.7]
BOOL_ACC = [100.0, 92.7, 88.2, 90.0, 88.2]
NUM_ACC  = [100.0, 98.5, 96.6, 91.4, 94.7]  # 100*(1 - MRE/100)
LIST_F1  = [100.0, 75.9, 72.2, 71.5, 65.0]  # F1 * 100

# Standard deviations from 3 repeated extractions (Human = 0, gold standard)
OVERALL_STD = [0.0, 1.8, 2.3, 2.1, 2.5]
TIME_STD    = [0.0, 2.5, 7.2, 5.8, 0.8]
COST_STD    = [0.0, 0.008, 0.0003, 0.004, 0.012]
CAT_STD     = [0.0, 2.1, 3.2, 2.8, 3.5]
BOOL_STD    = [0.0, 1.5, 2.0, 1.8, 2.2]
NUM_STD     = [0.0, 0.8, 1.2, 2.0, 1.5]
LIST_STD    = [0.0, 2.5, 3.0, 3.2, 3.8]

ERR_KW = dict(capsize=2, error_kw=dict(ecolor='#555555', elinewidth=0.8, capthick=0.8))

# ============================================================
# Plot
# ============================================================

apply_plot_style(FONT_SIZES_SUBMIT)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
fs = FONT_SIZES_SUBMIT

y_pos = np.arange(len(EXTRACTORS))
bar_colors = [COLORS[e] for e in EXTRACTORS]

# --- Panel (a): Overall Score ---
ax = axes[0, 0]
bars = ax.barh(y_pos, OVERALL, xerr=OVERALL_STD, color=bar_colors, edgecolor='white',
               linewidth=0.5, height=0.65, **ERR_KW)
# Hatching for Human bar
bars[0].set_hatch('///')
bars[0].set_edgecolor('#888888')
ax.set_yticks(y_pos)
ax.set_yticklabels(EXTRACTORS, fontsize=fs['tick_label'])
ax.set_xlabel('Overall Score', fontsize=fs['axis_label'])
ax.set_xlim(0, 110)
ax.invert_yaxis()
for i, v in enumerate(OVERALL):
    ax.text(v + 1, i, f'{v:.1f}', va='center', fontsize=fs['annotation'])
style_axis(ax, grid_y=False)
ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC', linewidth=LINE_WIDTHS['grid'])
add_panel_label(ax, '(a)', fs)

# --- Panel (b): Time per Paper (log scale) ---
ax = axes[0, 1]
bars = ax.barh(y_pos, TIME_SEC, xerr=TIME_STD, color=bar_colors, edgecolor='white',
               linewidth=0.5, height=0.65, **ERR_KW)
bars[0].set_hatch('///')
bars[0].set_edgecolor('#888888')
ax.set_xscale('log')
ax.set_yticks(y_pos)
ax.set_yticklabels(EXTRACTORS, fontsize=fs['tick_label'])
ax.set_xlabel('Time per Paper (seconds)', fontsize=fs['axis_label'])
ax.invert_yaxis()
# Annotate speedup vs human
for i in range(1, len(EXTRACTORS)):
    speedup = TIME_SEC[0] / TIME_SEC[i]
    ax.text(TIME_SEC[i] * 1.3, i, f'×{speedup:.0f}', va='center', fontsize=fs['annotation'], color='#555555')
style_axis(ax, grid_y=False)
ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC', linewidth=LINE_WIDTHS['grid'])
add_panel_label(ax, '(b)', fs)

# --- Panel (c): Cost per Paper (log scale) ---
ax = axes[1, 0]
bars = ax.barh(y_pos, COST_USD, xerr=COST_STD, color=bar_colors, edgecolor='white',
               linewidth=0.5, height=0.65, **ERR_KW)
bars[0].set_hatch('///')
bars[0].set_edgecolor('#888888')
ax.set_xscale('log')
ax.set_yticks(y_pos)
ax.set_yticklabels(EXTRACTORS, fontsize=fs['tick_label'])
ax.set_xlabel('Cost per Paper (USD)', fontsize=fs['axis_label'])
ax.invert_yaxis()
# Annotate cost reduction vs human
for i in range(1, len(EXTRACTORS)):
    reduction = COST_USD[0] / COST_USD[i]
    ax.text(COST_USD[i] * 1.5, i, f'×{reduction:.0f}', va='center', fontsize=fs['annotation'], color='#555555')
style_axis(ax, grid_y=False)
ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC', linewidth=LINE_WIDTHS['grid'])
add_panel_label(ax, '(c)', fs)

# --- Panel (d): Accuracy Breakdown (grouped bar) ---
ax = axes[1, 1]
categories = ['Categorical', 'Boolean', 'Numeric', 'List F1']
x = np.arange(len(categories))
width = 0.15
llm_names = EXTRACTORS[1:]  # Exclude Human
llm_data = {
    'Claude Sonnet 4.6': [CAT_ACC[1], BOOL_ACC[1], NUM_ACC[1], LIST_F1[1]],
    'Qwen3.5-Plus':      [CAT_ACC[2], BOOL_ACC[2], NUM_ACC[2], LIST_F1[2]],
    'Gemini 3.1 Pro':    [CAT_ACC[3], BOOL_ACC[3], NUM_ACC[3], LIST_F1[3]],
    'GPT-5.1':           [CAT_ACC[4], BOOL_ACC[4], NUM_ACC[4], LIST_F1[4]],
}
llm_errs = {
    'Claude Sonnet 4.6': [CAT_STD[1], BOOL_STD[1], NUM_STD[1], LIST_STD[1]],
    'Qwen3.5-Plus':      [CAT_STD[2], BOOL_STD[2], NUM_STD[2], LIST_STD[2]],
    'Gemini 3.1 Pro':    [CAT_STD[3], BOOL_STD[3], NUM_STD[3], LIST_STD[3]],
    'GPT-5.1':           [CAT_STD[4], BOOL_STD[4], NUM_STD[4], LIST_STD[4]],
}

for j, name in enumerate(llm_names):
    offset = (j - 1.5) * width
    vals = llm_data[name]
    errs = llm_errs[name]
    ax.bar(x + offset, vals, width, yerr=errs, color=COLORS[name], edgecolor='white',
           linewidth=0.5, label=name, **ERR_KW)

# Human reference line at 100%
ax.axhline(y=100, color='#B0B0B0', linestyle='--', linewidth=1.0, alpha=0.7, label='Human (100%)')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=fs['tick_label'])
ax.set_ylabel('Accuracy (%)', fontsize=fs['axis_label'])
ax.set_ylim(50, 105)
ax.legend(fontsize=fs['legend'] - 0.5, loc='lower left', framealpha=0.9, ncol=2)
style_axis(ax, grid_y=True)
add_panel_label(ax, '(d)', fs)

plt.tight_layout(w_pad=2.5, h_pad=2.0)
save_figure(fig, 'FigS1_LLM_benchmark.png')
save_figure(fig, 'Fig1_LLM_benchmark.pdf')
print('Done.')
