"""
Generate a scientific paper-style visualization of the WaterCNN architecture.
Uses stacked-feature-map (3D cuboid) style as seen in academic CNN papers.
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for HPC (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np

# ============================================================================
# Drawing helpers: 3D cuboid "feature map" blocks
# ============================================================================

def draw_cuboid(ax, x, y, w, h, d, face_color, edge_color='#333333', alpha=0.95, n_slices=5):
    """
    Draw a 3D-looking cuboid representing a feature map stack.
    (x, y) = bottom-left of the front face.
    w, h   = width and height of the front face.
    d      = isometric depth offset (applied top-right).
    n_slices = number of thin stacked planes drawn on top to show depth.
    """
    import colorsys

    def darken(hex_color, factor=0.65):
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v * factor)
        return (r2, g2, b2)

    def lighten(hex_color, factor=1.25):
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s * 0.7, min(1.0, v * factor))
        return (r2, g2, b2)

    dark   = darken(face_color)
    light  = lighten(face_color)

    # Front face
    front = plt.Polygon([
        (x,     y),
        (x + w, y),
        (x + w, y + h),
        (x,     y + h),
    ], closed=True, facecolor=face_color, edgecolor=edge_color, linewidth=0.8, alpha=alpha, zorder=3)

    # Top face
    top = plt.Polygon([
        (x,         y + h),
        (x + w,     y + h),
        (x + w + d, y + h + d * 0.5),
        (x     + d, y + h + d * 0.5),
    ], closed=True, facecolor=light, edgecolor=edge_color, linewidth=0.8, alpha=alpha, zorder=3)

    # Right face
    right = plt.Polygon([
        (x + w,     y),
        (x + w + d, y + d * 0.5),
        (x + w + d, y + h + d * 0.5),
        (x + w,     y + h),
    ], closed=True, facecolor=dark, edgecolor=edge_color, linewidth=0.8, alpha=alpha, zorder=3)

    for patch in (front, top, right):
        ax.add_patch(patch)

    # Thin slice lines on the front face to suggest multiple channels
    if n_slices > 1:
        slice_w = d / n_slices
        for i in range(1, n_slices):
            sx = x + i * slice_w
            sy = y + i * slice_w * 0.5
            line_alpha = 0.4
            ax.plot([sx, sx + w], [sy + h, sy + h], color=edge_color,
                    linewidth=0.5, alpha=line_alpha, zorder=4)
            ax.plot([sx + w, sx + w], [sy, sy + h], color=edge_color,
                    linewidth=0.5, alpha=line_alpha, zorder=4)
            ax.plot([x + w, sx + w], [y + h, sy + h], color=edge_color,
                    linewidth=0.5, alpha=line_alpha, zorder=4)

    # 3x3 grid subdivision lines — front, right, and top faces
    grid_alpha = 0.35
    grid_color = edge_color
    grid_lw    = 0.7

    # --- Front face grid ---
    for col in range(1, 3):
        gx = x + col * w / 3
        ax.plot([gx, gx], [y, y + h], color=grid_color,
                linestyle=':', linewidth=grid_lw, alpha=grid_alpha, zorder=4)
    for row in range(1, 3):
        gy = y + row * h / 3
        ax.plot([x, x + w], [gy, gy], color=grid_color,
                linestyle=':', linewidth=grid_lw, alpha=grid_alpha, zorder=4)

    # --- Right face grid (parallelogram: x+w..x+w+d, y..y+h sheared by d*0.5) ---
    # Horizontal divisions on right face
    for row in range(1, 3):
        t = row / 3
        y0 = y + t * h;          y1 = y0 + d * 0.5
        ax.plot([x + w, x + w + d], [y0, y1], color=grid_color,
                linestyle=':', linewidth=grid_lw, alpha=grid_alpha, zorder=4)
    # Depth (slice) divisions on right face — vertical lines along depth direction
    if n_slices > 1:
        for col in range(1, 3):
            t = col / 3
            sx = x + w + t * d;  sy = t * d * 0.5
            ax.plot([sx, sx], [y + sy, y + sy + h], color=grid_color,
                    linestyle=':', linewidth=grid_lw, alpha=grid_alpha, zorder=4)

    # --- Top face grid ---
    # Lines parallel to the depth direction (left-right across top)
    for row in range(1, 3):
        t = row / 3
        gx0 = x + t * w;        gx1 = gx0 + d
        gy0 = y + h;             gy1 = y + h + d * 0.5
        ax.plot([gx0, gx1], [gy0, gy1], color=grid_color,
                linestyle=':', linewidth=grid_lw, alpha=grid_alpha, zorder=4)
    # Lines parallel to the width direction
    for col in range(1, 3):
        t = col / 3
        sx = t * d;  sy = t * d * 0.5
        ax.plot([x + sx, x + w + sx], [y + h + sy, y + h + sy], color=grid_color,
                linestyle=':', linewidth=grid_lw, alpha=grid_alpha, zorder=4)

    return {
        'front_cx': x + w / 2,
        'front_cy': y + h / 2,
        'right_x':  x + w + d,
        'right_cy': y + d * 0.5 + h / 2,
        'top_cy':   y + h + d * 0.5,
        'left_x':   x,
        'left_cy':  y + h / 2,
        'bottom_y': y,
        'top_y':    y + h + d * 0.5,
    }


def draw_neuron_column(ax, cx, ys, color, radius=0.18):
    """Draw a vertical column of neurons (circles)."""
    for y in ys:
        c = plt.Circle((cx, y), radius, color=color, ec='#333333', linewidth=1.2, zorder=5)
        ax.add_patch(c)


def connect(ax, x1, y1, x2, y2, color='#555555', lw=1.2, style='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'),
                zorder=6)


def label_below(ax, x, y, text, fontsize=8, color='#222222'):
    ax.text(x, y, text, ha='center', va='top', fontsize=fontsize,
            color=color, zorder=7)


def label_above(ax, x, y, text, fontsize=8, color='#222222'):
    ax.text(x, y, text, ha='center', va='bottom', fontsize=fontsize,
            color=color, zorder=7)


# ============================================================================
# Canvas
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-1.5, 8.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ============================================================================
# Color palette
# ============================================================================
C_INPUT    = '#B0BEC5'   # gray
C_WATER    = '#1565C0'   # deep blue   – water branch
C_TERRAIN  = '#2E7D32'   # deep green  – terrain branch
C_COMBINED = '#6A1B9A'   # purple      – combined branch
C_FLAT     = '#E65100'   # orange      – flattened
C_FC       = '#F9A825'   # amber       – FC neurons
C_OUT      = '#C62828'   # red         – output

# ============================================================================
# Cuboid parameters: (x_left, y_bottom, width, height, depth, color, n_slices)
# Three rows:  water (top), combined (middle), terrain (bottom)
# ============================================================================

BOX_H = 1.0   # front-face height
BOX_W = 1.0  # front-face width
D_IN  = 0.1  # input depth offset  (1 channel)
D_8   = 0.8   # depth for 8 channels
D_16  = 1.6   # depth for 16 channels
D_2   = 0.2   # 2-channel concat for combined branch

# Row y-baselines
Y_W = 5   # water row
Y_C = 2.5   # combined row
Y_T = 0   # terrain row

GAP = 0.5   # horizontal gap between blocks

# X positions (left edge of each block, going left → right)
# Inputs sit at x=0
X_IN    = 0.0

# Branch water/terrain conv+relu output
X_CONV1 = X_IN + BOX_W + D_IN + GAP + 0.5
# Branch water/terrain after conv
X_OUT1  = X_CONV1 + BOX_W + D_8 + GAP

# Combined branch splits off from inputs: concat at same x as inputs
X_CONC  = X_IN + BOX_W + D_IN + GAP + 0.5   # same x as water/terrain conv
X_1x1   = X_CONC + BOX_W + D_2 + GAP
X_1x1R  = X_1x1 + BOX_W + D_8 + GAP
X_3x3   = X_1x1R + GAP
X_3x3R  = X_3x3 + BOX_W + D_16 + GAP      # output of combined branch

# Merge / FC section
X_MERGE = max(X_OUT1, X_3x3R) + GAP + 0.3
X_FC1   = X_MERGE + 1.3
X_FC2   = X_FC1   + 0.7
X_FC_OUT= X_FC2   + 1.2

# ============================================================================
# WATER BRANCH  (top row)
# ============================================================================
b_w_in = draw_cuboid(ax, X_IN, Y_W, BOX_W, BOX_H, D_IN, C_INPUT,     n_slices=1)
label_below(ax, X_IN + BOX_W/2 + D_IN/2, Y_W - 0.1,  'Water\n3×3×1', fontsize=8)

connect(ax, b_w_in['right_x'], b_w_in['right_cy'],
           X_3x3, Y_W + BOX_H/2 + D_8*0.0)

b_w_c1 = draw_cuboid(ax, X_3x3, Y_W, BOX_W, BOX_H, D_8, C_WATER,   n_slices=8)
label_below(ax, X_3x3 + BOX_W/2, Y_W - 0.1, 'Conv 3×3\n+ReLU\n1→8', fontsize=8)
label_above(ax, X_3x3 + BOX_W/2 + D_8, Y_W + BOX_H + D_8*0.5 + 0.1, '(8,1,1)', fontsize=7.5, color='#1565C0')

# ============================================================================
# TERRAIN BRANCH  (bottom row)
# ============================================================================
b_t_in = draw_cuboid(ax, X_IN, Y_T, BOX_W, BOX_H, D_IN, C_INPUT,     n_slices=1)
label_below(ax, X_IN + BOX_W/2 + D_IN/2, Y_T - 0.1,  'Terrain\n3×3×1', fontsize=8)

connect(ax, b_t_in['right_x'], b_t_in['right_cy'],
           X_3x3, Y_T + BOX_H/2 + D_8*0.0)

b_t_c1 = draw_cuboid(ax, X_3x3, Y_T, BOX_W, BOX_H, D_8, C_TERRAIN, n_slices=8)
label_below(ax, X_3x3 + BOX_W/2, Y_T - 0.1, 'Conv 3×3\n+ReLU\n1→8', fontsize=8)
label_above(ax, X_3x3 + BOX_W/2 + D_8, Y_T + BOX_H + D_8*0.5 + 0.1, '(8,1,1)', fontsize=7.5, color='#2E7D32')

# ============================================================================
# COMBINED BRANCH  (middle row)
# ============================================================================
# Concat input: 2-channel (water + terrain)
b_c_in = draw_cuboid(ax, X_CONC, Y_C, BOX_W, BOX_H, D_2, C_COMBINED, n_slices=2)
label_below(ax, X_CONC + BOX_W/2 + D_2/2, Y_C - 0.1, 'Concat\n2×3×3', fontsize=8)

# Arrows from water and terrain inputs to combined concat
ax.annotate('', xy=(X_CONC, Y_C + BOX_H * 0.8), xytext=(b_w_in['right_x'] - D_IN*0.2, b_w_in['right_cy']),
            arrowprops=dict(arrowstyle='->', color='#777777', lw=1.0,
                            connectionstyle='arc3,rad=0.2'), zorder=6)
ax.annotate('', xy=(X_CONC, Y_C + BOX_H * 0.2), xytext=(b_t_in['right_x'] - D_IN*0.2, b_t_in['right_cy']),
            arrowprops=dict(arrowstyle='->', color='#777777', lw=1.0,
                            connectionstyle='arc3,rad=-0.2'), zorder=6)

connect(ax, X_CONC + BOX_W + D_2, Y_C + BOX_H/2 + D_2*0.0,
           X_1x1, Y_C + BOX_H/2 + D_8*0.0)

b_c_1x1 = draw_cuboid(ax, X_1x1, Y_C, BOX_W, BOX_H, D_8, C_COMBINED, n_slices=8)
label_below(ax, X_1x1 + BOX_W/2, Y_C - 0.1,  'Conv 1×1\n+ReLU\n2→8', fontsize=8)

connect(ax, X_1x1 + BOX_W + D_8, Y_C + BOX_H/2 + D_8*0.0,
           X_3x3, Y_C + BOX_H/2 + D_16*0.0)

b_c_3x3 = draw_cuboid(ax, X_3x3, Y_C, BOX_W, BOX_H, D_16, C_COMBINED, n_slices=12)
label_below(ax, X_3x3 + BOX_W/2, Y_C - 0.1, 'Conv 3×3\n+ReLU\n8→16', fontsize=8)
label_above(ax, X_3x3 + BOX_W/2 + D_16, Y_C + BOX_H + D_16*0.5 + 0.1, '(16,1,1)', fontsize=7.5, color='#6A1B9A')

# ============================================================================
# FLATTEN / MERGE ARROWS → FC
# ============================================================================
x_merge_left = X_MERGE - 0.15

# Arrows from water branch output → merge
ax.annotate('', xy=(x_merge_left, b_w_c1['right_cy']),
            xytext=(b_w_c1['right_x'], b_w_c1['right_cy']),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2,
                            connectionstyle='arc3,rad=0.0'), zorder=6)

# Arrows from terrain branch output → merge
ax.annotate('', xy=(x_merge_left, b_t_c1['right_cy']),
            xytext=(b_t_c1['right_x'], b_t_c1['right_cy']),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2,
                            connectionstyle='arc3,rad=0.0'), zorder=6)

# Arrow from combined branch output → merge
ax.annotate('', xy=(x_merge_left, b_c_3x3['right_cy']),
            xytext=(b_c_3x3['right_x'], b_c_3x3['right_cy']),
            arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.2,
                            connectionstyle='arc3,rad=0.0'), zorder=6)

# Flatten box — 32 colour-coded slices
# Sections: 8 blue (water), 16 purple (combined), 8 green (terrain)
# Order top→bottom: water(8), combined(16), terrain(8) — matches branch vertical order
flat_x    = X_MERGE - 0.1
flat_y_bot = 0
flat_h    = 7.5
flat_w    = 0.5
n_total   = 32
slice_h   = flat_h / n_total

section_colors = (
    [(0, 8, '#43A047')],   # green – terrain (8)
    [(8,  24, '#8E24AA')],   # purple – combined (16)
    [(24, 32, '#1E88E5')],   # blue  – water (8)
)
for start, end, color in [t for s in section_colors for t in s]:
    for i in range(start, end):
        yb = flat_y_bot + i * slice_h
        rect = mpatches.Rectangle((flat_x, yb), flat_w, slice_h,
                                   facecolor=color, edgecolor='none',
                                   alpha=0.55, zorder=5)
        ax.add_patch(rect)

# Faint horizontal division lines between slices
for i in range(1, n_total):
    yb = flat_y_bot + i * slice_h
    ax.plot([flat_x, flat_x + flat_w], [yb, yb],
            color='white', linewidth=0.4, alpha=0.6, zorder=6)

# Section dividers (thicker)
for boundary in [8, 24]:
    yb = flat_y_bot + boundary * slice_h
    ax.plot([flat_x, flat_x + flat_w], [yb, yb],
            color='white', linewidth=1.5, alpha=0.9, zorder=6)

# Outer border
flat_border = mpatches.FancyBboxPatch((flat_x, flat_y_bot), flat_w, flat_h,
                                       boxstyle='round,pad=0.02',
                                       facecolor='none', edgecolor='#555',
                                       linewidth=1.5, zorder=7)
ax.add_patch(flat_border)

# Label
ax.text(flat_x + flat_w/2, flat_y_bot + flat_h/2, 'Flatten\n(B,32)',
        ha='center', va='center', fontsize=7.5, weight='bold',
        rotation=90, color='white', zorder=8)

# Small legend for the flatten bar
for label, color, xa, ya in [("T (8)", '#43A047', flat_x + 1.5, flat_y_bot  + 0.0),
                          ('C (16)', '#8E24AA', flat_x + 0.75, flat_y_bot + 0.),
                          ('W (8)', '#1E88E5', flat_x + 0.0, flat_y_bot + 0.)]:
    ax.add_patch(mpatches.Rectangle((xa - 0.7 - 0.05, ya - 0.4 - 0.12), 0.12, 0.25,
                                     facecolor=color, edgecolor='none', alpha=0.8, zorder=8))
    ax.text(xa - 0.7 + 0.12, ya - 0.4, label, fontsize=6.5, va='center', zorder=8)

# ============================================================================
# FC NEURONS
# ============================================================================
N_IN  = 32   # visual neurons for 32-dim input (8 dots)
N_H1  = 16   # visual neurons for 16-dim hidden
N_OUT = 1   # output

fc_y_center = flat_y_bot + flat_h / 2

ys_in  = np.linspace(fc_y_center - N_IN/5, fc_y_center + N_IN/5, N_IN)
ys_h1  = np.linspace(fc_y_center - N_H1/6, fc_y_center + N_H1/6, N_H1)

#draw_neuron_column(ax, X_FC1, ys_in,  color='#90CAF9')
draw_neuron_column(ax, X_FC2, ys_h1,  color=C_FC)

out_circle = plt.Circle((X_FC_OUT, fc_y_center), 0.28,
                         color=C_OUT, ec='#333333', linewidth=2, zorder=5)
ax.add_patch(out_circle)
ax.text(X_FC_OUT + 0.65, fc_y_center, 'Pred.\nWater\nValue', ha='center', va='center',
        fontsize=10, weight='bold', color='#333')

# Connection lines (FC weight visualization)
#for y1 in ys_in:
#    ax.plot([flat_x + flat_w, X_FC1 - 0.18], [fc_y_center + (y1 - fc_y_center) * 1, y1],
#            color='#90CAF9', lw=0.5, alpha=0.5, zorder=4)
for y1 in ys_in:
    for y2 in ys_h1:
        ax.plot([flat_x + flat_w + 0.05, X_FC2 - 0.18], [fc_y_center + (y1 - fc_y_center) * 0.57, y2],
                color='#aaa', lw=0.4, alpha=0.4, zorder=4)
for y1 in ys_h1:
    ax.plot([X_FC2 + 0.18, X_FC_OUT - 0.28], [y1, fc_y_center],
            color=C_OUT, lw=0.6, alpha=0.5, zorder=4)

label_below(ax, X_FC1 - 0.05, fc_y_center + (ys_in[0] - fc_y_center) * 0.57 + 0.3,  'Linear\n32→16\n+ReLU', fontsize=8)
label_below(ax, X_FC2+0.75, ys_h1[0] + 0.9,  'Linear\n16→1', fontsize=8)
label_below(ax, X_FC_OUT, fc_y_center - 0.4, 'Output\n(scalar)', fontsize=8)

# ============================================================================
# BRANCH LABELS
# ============================================================================
ax.text(-0.4, Y_W + BOX_H/2 + D_IN*0.5,  'Water\nbranch',
        ha='right', va='center', fontsize=9, color=C_WATER, weight='bold')
ax.text(-0.4, Y_T + BOX_H/2 + D_IN*0.5,  'Terrain\nbranch',
        ha='right', va='center', fontsize=9, color=C_TERRAIN, weight='bold')
ax.text(-0.4, Y_C + BOX_H/2 + D_2*0.5,   'Combined\nbranch',
        ha='right', va='center', fontsize=9, color=C_COMBINED, weight='bold')

# ============================================================================
# TITLE
# ============================================================================
ax.text(7, 8.5, 'WaterCNN Architecture',
        ha='center', va='top', fontsize=16, weight='bold', color='#111')
ax.text(7, 8.0,
        'Multi-branch CNN predicting water cell value at t+1 from 3×3 patches',
        ha='center', va='top', fontsize=10, color='#555', style='italic')

plt.tight_layout()
out_path = 'watercnn_architecture.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.show()
