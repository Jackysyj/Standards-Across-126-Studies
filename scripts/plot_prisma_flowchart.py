#!/usr/bin/env python3
"""
Generate PRISMA 2020 flow diagram for literature screening process.

Based on: Page et al. (2021) The PRISMA 2020 statement. BMJ 372:n71.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Configuration
DPI = 300
WIDTH_CM = 8.0
HEIGHT_CM = 12.0
WIDTH_INCH = WIDTH_CM / 2.54
HEIGHT_INCH = HEIGHT_CM / 2.54

FONT_FAMILY = 'Arial'
FONT_SIZE_TITLE = 9
FONT_SIZE_BODY = 8
FONT_SIZE_SMALL = 7

BOX_COLOR = '#E8F4F8'
EXCLUDE_COLOR = '#FFE6E6'
EDGE_COLOR = '#2C3E50'
ARROW_COLOR = '#34495E'

# Create figure
fig, ax = plt.subplots(figsize=(WIDTH_INCH, HEIGHT_INCH), dpi=DPI)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

def add_box(ax, x, y, width, height, text, bgcolor=BOX_COLOR, fontsize=FONT_SIZE_BODY, bold=False):
    """Add a rounded rectangle box with text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05",
        edgecolor=EDGE_COLOR, facecolor=bgcolor,
        linewidth=1.0, zorder=2
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontfamily=FONT_FAMILY, fontweight=weight,
            wrap=True, zorder=3)
    return box

def add_arrow(ax, x1, y1, x2, y2):
    """Add a downward arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=15,
        color=ARROW_COLOR, linewidth=1.5, zorder=1
    )
    ax.add_patch(arrow)

# PRISMA flow diagram structure
# Box 1: Identification
add_box(ax, 5, 13, 5, 1.0,
        'Records identified from\nOpenAlex (n = 158)',
        fontsize=FONT_SIZE_BODY, bold=True)
add_arrow(ax, 5, 12.5, 5, 11.8)

# Box 2: Screening
add_box(ax, 5, 11.3, 5, 1.0,
        'Records screened\n(n = 158)',
        fontsize=FONT_SIZE_BODY)
add_arrow(ax, 5, 10.8, 5, 9.3)

# Box 3: Excluded (now in center, below screening)
exclude_text = ('Records excluded (n = 32):\n'
                '• Review articles (n = 9)\n'
                '• Conference proceedings (n = 2)\n'
                '• Preprints/incomplete metadata (n = 4)\n'
                '• Outside scope (n = 2)\n'
                '• No quantitative ML metrics (n = 12)\n'
                '• Off-topic/duplicates (n = 3)')
add_box(ax, 5, 7.8, 5, 3.0,
        exclude_text,
        bgcolor=EXCLUDE_COLOR, fontsize=FONT_SIZE_SMALL)
add_arrow(ax, 5, 6.3, 5, 5.8)

# Box 4: Included
add_box(ax, 5, 5.3, 5, 1.0,
        'Studies included in review\n(n = 126)',
        fontsize=FONT_SIZE_BODY, bold=True)

# Add title
ax.text(5, 13.8, 'PRISMA 2020 Flow Diagram',
        ha='center', va='center',
        fontsize=FONT_SIZE_TITLE, fontfamily=FONT_FAMILY, fontweight='bold')

# Adjust layout
plt.tight_layout(pad=0.2)

# Save figure
output_path = '../figures/fig_s2_prisma.png'
plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f"PRISMA flow diagram saved to: {output_path}")
print(f"Size: {WIDTH_CM:.1f} cm × {HEIGHT_CM:.1f} cm @ {DPI} DPI")

plt.close()
