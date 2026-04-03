"""
ML4Env Critical Review - Plot Configuration
Unified color palette, font sizes, and style settings.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ============================================================
# Color Palettes
# ============================================================

# Five-pillar colors (core mapping)
PILLAR_COLORS = {
    'data':              '#7EAAC4',  # Dusty blue
    'validation':        '#A3C9A8',  # Sage green
    'interpretability':  '#E8B87C',  # Warm amber
    'comparability':     '#C49BBB',  # Muted mauve
    'deployment':        '#D4897C',  # Terracotta rose
}

# Research field colors
FIELD_COLORS = {
    'adsorption':  '#7EAAC4',  # Dusty blue
    'degradation': '#D4897C',  # Terracotta rose
    'both':        '#D4B96A',  # Soft gold
}

# 10-color categorical palette (macaron style)
CATEGORICAL_PALETTE = [
    '#7EAAC4',  # Dusty blue
    '#A3C9A8',  # Sage green
    '#E8B87C',  # Warm amber
    '#C49BBB',  # Muted mauve
    '#D4897C',  # Terracotta rose
    '#8DBEB5',  # Seafoam
    '#D4B96A',  # Soft gold
    '#9BAFD4',  # Periwinkle
    '#C7A07C',  # Camel
    '#A8C5D6',  # Powder blue
]

# Accent colors
ACCENT_COLORS = {
    'warning':  '#E07B6C',  # Deep terracotta (problem areas)
    'positive': '#7CB88A',  # Deep sage (good practices)
}

# Heatmap colormap: terracotta (bad) -> white -> sage (good)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'ml4env', ['#D4897C', '#FFFFFF', '#A3C9A8']
)

# ============================================================
# Font Sizes - Preview (for writing stage)
# ============================================================

FONT_SIZES = {
    'panel_label': 18,
    'title':       16,
    'axis_label':  14,
    'tick_label':  12,
    'legend':      11,
    'annotation':  11,
}

# Font sizes for journal submission
FONT_SIZES_SUBMIT = {
    'panel_label': 10,
    'title':       9,
    'axis_label':  9,
    'tick_label':  8,
    'legend':      7,
    'annotation':  7,
}

# ============================================================
# Line Widths
# ============================================================

LINE_WIDTHS = {
    'spine':  0.8,
    'grid':   0.5,
    'bar_edge': 0.5,
    'line':   1.5,
}

# ============================================================
# Figure Sizes (inches)
# ============================================================

# Preview sizes
FIGURE_SIZE_SINGLE = (10, 7)
FIGURE_SIZE_MULTI  = (14, 10)

# Submission sizes
FIGURE_SIZE_1COL = (3.5, 2.8)   # ~89mm single column
FIGURE_SIZE_1_5COL = (5.5, 4.0) # ~140mm
FIGURE_SIZE_2COL = (7.0, 5.0)   # ~178mm full width
FIGURE_SIZE_2COL_TALL = (7.0, 9.0)

DPI = 300

# ============================================================
# Output Directories
# ============================================================

BASE_DIR = Path(__file__).parent.parent
FIGURE_DIR = BASE_DIR / 'figures'
FIGURE_DIR.mkdir(exist_ok=True)

# ============================================================
# Style Functions
# ============================================================

def apply_plot_style(font_sizes=None):
    """Apply consistent plot style for all figures."""
    fs = font_sizes or FONT_SIZES
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': fs['tick_label'],
        'axes.titlesize': fs['title'],
        'axes.labelsize': fs['axis_label'],
        'xtick.labelsize': fs['tick_label'],
        'ytick.labelsize': fs['tick_label'],
        'legend.fontsize': fs['legend'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': LINE_WIDTHS['spine'],
        'axes.grid': False,
        'figure.dpi': DPI,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': DPI,
    })


def add_panel_label(ax, label, font_sizes=None):
    """Add panel label like (a), (b), (c) to top-left corner."""
    fs = font_sizes or FONT_SIZES
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=fs['panel_label'], fontweight='bold',
            va='top', ha='left')


def style_axis(ax, grid_y=True):
    """Apply standard axis styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTHS['spine'])
    ax.spines['bottom'].set_linewidth(LINE_WIDTHS['spine'])
    if grid_y:
        ax.yaxis.grid(True, linestyle='--', alpha=0.3,
                       color='#CCCCCC', linewidth=LINE_WIDTHS['grid'])
        ax.set_axisbelow(True)


def save_figure(fig, filename, dpi=None):
    """Save figure to the figures directory."""
    out = FIGURE_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi or DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.close(fig)
