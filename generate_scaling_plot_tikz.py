#!/usr/bin/env python3
"""
Generate a standalone TikZ scaling plot:
  - Sized to fit exactly within an A4 document textwidth (146-148 mm <= 150 mm).
  - X-axis: Solver Nodes (1, 2, 4, 8, 16)
  - Y-axis: Median ML-step time (ms)
  - 4 Subgraphs (1x4 horizontal):
      * Subgraph 1: AIx vs. SmartSim (Default Layout) [Linear scale 1: 0 - 22,000 ms]
      * Common overarching headline for Subgraphs 2-4: "AIx Layout Variations"
      * Subgraphs 2, 3, 4: Shared Linear scale 2: 0 - 250 ms
          - Subgraph 2: Default
          - Subgraph 3: Affine
          - Subgraph 4: NUMA
  - Legend below the figure:
      * SmartSim (default) instead of SmartSim c0
      * Grouped cleanly in 2 rows x 3 columns
"""

from pathlib import Path
import numpy as np
import pandas as pd

MINI_APP_DIR = Path("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app")
CSV_HET3 = MINI_APP_DIR / "results/hethires3/ml_step_makespan.csv"
CSV_HET2 = MINI_APP_DIR / "results/hethires2/ml_step_makespan.csv"

def compute_medians():
    df3 = pd.read_csv(CSV_HET3)
    df2 = pd.read_csv(CSV_HET2)

    # Steady steps (solver_step > 2)
    s3 = df3[df3["solver_step"] > 2]
    s2 = df2[df2["solver_step"] > 2]

    scens = [7, 8, 9, 10, 11]
    nodes = [1, 2, 4, 8, 16]

    curves = {
        # Graph 1
        "aix_coll_default": [f"s{s}_aix_coll_default" for s in scens],
        "smartsim_c0": [f"s{s}_smartsim_c0_default" if s > 7 else "s7_smartsim_c0" for s in scens],
        "smartsim_c3": [f"s{s}_smartsim_c3_default" if s > 7 else "s7_smartsim_c3" for s in scens],
        "smartsim_per_gpu_db": [f"s{s}_smartsim_per_gpu_db_default" for s in scens],

        # Graph 2 (default)
        "aix_p2p_default": [f"s{s}_aix_p2p_default" for s in scens],
        "aix_p2p_fc_default": [f"s{s}_aix_p2p_fullcredits_default" for s in scens],

        # Graph 3 (affine)
        "aix_coll_affine": [f"s{s}_aix_coll_affine" for s in scens],
        "aix_p2p_affine": [f"s{s}_aix_p2p_affine" for s in scens],
        "aix_p2p_fc_affine": [f"s{s}_aix_p2p_fullcredits_affine" for s in scens],

        # Graph 4 (numa)
        "aix_coll_numa": [f"s{s}_aix_coll_numa" for s in scens],
        "aix_p2p_numa": [f"s{s}_aix_p2p_numa" for s in scens],
        "aix_p2p_fc_numa": [f"s{s}_aix_p2p_fullcredits_numa" for s in scens],
    }

    data = {}
    for name, cols in curves.items():
        vals = []
        for s, col in zip(scens, cols):
            if col in s3:
                series = s3[col].dropna()
            elif col in s2:
                series = s2[col].dropna()
            else:
                series = []
            vals.append(float(np.median(series)) if len(series) else None)
        data[name] = vals

    return nodes, data

def generate_tikz(nodes, data, output_tex: Path):
    # Geometry constrained to A4 \textwidth (150 mm = 15.0 cm)
    W = 2.85   # cm width per plot
    H = 2.80   # cm height per plot
    gap12 = 1.35  # cm gap between graph 1 and graph 2 (room for shared Y-axis label & ticks)
    gap_shared = 0.42  # cm gap between shared Y subgraphs (2, 3, 4)

    x_offsets = [
        0.0,
        W + gap12,
        W + gap12 + W + gap_shared,
        W + gap12 + 2 * (W + gap_shared)
    ]

    y_max1 = 22000.0
    y_ticks1 = [0, 5000, 10000, 15000, 20000]

    y_max2 = 250.0
    y_ticks2 = [0, 50, 100, 150, 200, 250]

    def map_x(idx):
        return (idx / 4.0) * W

    def map_y1(val):
        return (val / y_max1) * H

    def map_y2(val):
        return (val / y_max2) * H

    def format_pts(ys, mapper):
        return " -- ".join(f"({map_x(i):.3f},{mapper(y):.3f})" for i, y in enumerate(ys))

    tex_lines = [
        r"\documentclass[tikz,border=4pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{siunitx}",
        r"\sisetup{locale = US, group-separator = {,}}",
        r"\usetikzlibrary{calc,positioning}",
        r"",
        r"% Color palette matching thesis standards",
        r"\definecolor{colorAixColl}{HTML}{1F77B4}",    # Tab:Blue
        r"\definecolor{colorAixP2P}{HTML}{FF7F0E}",     # Tab:Orange
        r"\definecolor{colorAixP2PFC}{HTML}{2CA02C}",   # Tab:Green
        r"\definecolor{colorSSZero}{HTML}{D62728}",     # Tab:Red
        r"\definecolor{colorSSThree}{HTML}{9467BD}",    # Tab:Purple
        r"\definecolor{colorSSPgdb}{HTML}{8C564B}",     # Tab:Brown
        r"",
        r"\begin{document}",
        r"\begin{tikzpicture}[",
        r"  font=\fontsize{8}{10}\selectfont,",
        r"  axis line/.style={draw=black!85, line width=0.5pt},",
        r"  grid line/.style={draw=black!14, line width=0.35pt, dotted},",
        r"  tick text/.style={font=\fontsize{6.5}{8}\selectfont},",
        r"  main title/.style={font=\fontsize{7.5}{9}\selectfont\bfseries, align=center},",
        r"  sub title/.style={font=\fontsize{7.5}{9}\selectfont\bfseries, align=center},",
        r"  axis label/.style={font=\fontsize{7}{8.5}\selectfont},",
        r"  legend label/.style={font=\fontsize{7}{8.5}\selectfont},",
        r"]",
        r"",
    ]

    # Overall headline over subgraphs 2, 3, 4
    x_aix_center = (x_offsets[1] + x_offsets[3] + W) / 2.0
    tex_lines.append("% --- Common headline over the 3 AIx subgraphs ---")
    tex_lines.append(f"\\node[main title] at ({x_aix_center:.3f}, {H + 0.68:.3f}) {{AIx Layouts}};")
    tex_lines.append(f"\\draw[draw=black!35, line width=0.4pt] ({x_offsets[1] + 0.1:.3f}, {H + 0.46:.3f}) -- ({x_offsets[3] + W - 0.1:.3f}, {H + 0.46:.3f});")
    tex_lines.append("")

    for g_idx in range(4):
        x0 = x_offsets[g_idx]
        mapper = map_y1 if g_idx == 0 else map_y2
        y_ticks = y_ticks1 if g_idx == 0 else y_ticks2

        tex_lines.append(f"% ==================== SUBGRAPH {g_idx+1} ====================")
        tex_lines.append(f"\\begin{{scope}}[shift={{({x0:.3f},0)}}]")

        # Titles
        if g_idx == 0:
            tex_lines.append(f"  \\node[main title] at ({W/2:.3f}, {H + 0.68:.3f}) {{AIx vs.\\ SmartSim}};")
            tex_lines.append(f"  \\node[tick text, text=black!70] at ({W/2:.3f}, {H + 0.32:.3f}) {{(Default)}};")
        elif g_idx == 1:
            tex_lines.append(f"  \\node[sub title] at ({W/2:.3f}, {H + 0.26:.3f}) {{Default}};")
        elif g_idx == 2:
            tex_lines.append(f"  \\node[sub title] at ({W/2:.3f}, {H + 0.26:.3f}) {{Affine}};")
        elif g_idx == 3:
            tex_lines.append(f"  \\node[sub title] at ({W/2:.3f}, {H + 0.26:.3f}) {{NUMA}};")

        # Grid lines (horizontal)
        for yt in y_ticks:
            y_c = mapper(yt)
            tex_lines.append(f"  \\draw[grid line] (0,{y_c:.3f}) -- ({W:.3f},{y_c:.3f});")

        # Grid lines (vertical)
        for i in range(5):
            x_c = map_x(i)
            tex_lines.append(f"  \\draw[grid line] ({x_c:.3f},0) -- ({x_c:.3f},{H:.3f});")

        # Axes box
        tex_lines.append(f"  \\draw[axis line] (0,0) rectangle ({W:.3f},{H:.3f});")

        # X ticks & labels: "Solver Nodes"
        for i, nd in enumerate(nodes):
            x_c = map_x(i)
            tex_lines.append(f"  \\draw[axis line] ({x_c:.3f},0) -- ({x_c:.3f},-0.08);")
            tex_lines.append(f"  \\node[tick text, below=1pt] at ({x_c:.3f},-0.08) {{{nd}}};")
        tex_lines.append(f"  \\node[axis label, below=10.5pt] at ({W/2:.3f},0) {{Solver Nodes}};")

        # Y ticks & labels
        if g_idx == 0:
            for yt in y_ticks:
                y_c = mapper(yt)
                tex_lines.append(f"  \\draw[axis line] (0,{y_c:.3f}) -- (-0.08,{y_c:.3f});")
                lbl = "0" if yt == 0 else f"{int(yt):,}"
                tex_lines.append(f"  \\node[tick text, left=1pt] at (-0.08,{y_c:.3f}) {{{lbl}}};")
            tex_lines.append(f"  \\node[axis label, rotate=90, anchor=south] at (-1.08, {H/2:.3f}) {{Median ML-Step Time (ms)}};")

        elif g_idx == 1:
            for yt in y_ticks:
                y_c = mapper(yt)
                tex_lines.append(f"  \\draw[axis line] (0,{y_c:.3f}) -- (-0.08,{y_c:.3f});")
                tex_lines.append(f"  \\node[tick text, left=1pt] at (-0.08,{y_c:.3f}) {{{yt}}};")
            tex_lines.append(f"  \\node[axis label, rotate=90, anchor=south] at (-0.68, {H/2:.3f}) {{Median ML-Step Time (ms)}};")
        else:
            # Subgraphs 3 and 4: tick marks without labels
            for yt in y_ticks:
                y_c = mapper(yt)
                tex_lines.append(f"  \\draw[axis line] (0,{y_c:.3f}) -- (-0.05,{y_c:.3f});")

        # Data curves
        if g_idx == 0:
            # SmartSim c0 (default)
            p_ss0 = format_pts(data["smartsim_c0"], mapper)
            tex_lines.append(f"  \\draw[colorSSZero, line width=1.0pt, dashed] {p_ss0};")
            for i, y in enumerate(data["smartsim_c0"]):
                tex_lines.append(f"  \\fill[colorSSZero] ({map_x(i):.3f},{mapper(y):.3f}) circle (1.6pt);")

            # SmartSim per-gpu-db
            p_pgdb = format_pts(data["smartsim_per_gpu_db"], mapper)
            tex_lines.append(f"  \\draw[colorSSPgdb, line width=1.0pt, densely dotted] {p_pgdb};")
            for i, y in enumerate(data["smartsim_per_gpu_db"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\draw[colorSSPgdb, line width=0.8pt, fill=white] ({cx-0.045:.3f},{cy-0.045:.3f}) rectangle ({cx+0.045:.3f},{cy+0.045:.3f});")

            # SmartSim c3
            p_ss3 = format_pts(data["smartsim_c3"], mapper)
            tex_lines.append(f"  \\draw[colorSSThree, line width=1.0pt, dashdotted] {p_ss3};")
            for i, y in enumerate(data["smartsim_c3"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\fill[colorSSThree] ({cx:.3f},{cy:.3f}) +(-1.6pt,0) -- +(0,1.6pt) -- +(1.6pt,0) -- +(0,-1.6pt) -- cycle;")

            # AIx coll default
            p_coll = format_pts(data["aix_coll_default"], mapper)
            tex_lines.append(f"  \\draw[colorAixColl, line width=1.1pt] {p_coll};")
            for i, y in enumerate(data["aix_coll_default"]):
                tex_lines.append(f"  \\fill[colorAixColl] ({map_x(i):.3f},{mapper(y):.3f}) circle (1.7pt);")

        elif g_idx == 1:
            p_coll = format_pts(data["aix_coll_default"], mapper)
            tex_lines.append(f"  \\draw[colorAixColl, line width=1.1pt] {p_coll};")
            for i, y in enumerate(data["aix_coll_default"]):
                tex_lines.append(f"  \\fill[colorAixColl] ({map_x(i):.3f},{mapper(y):.3f}) circle (1.7pt);")

            p_p2p = format_pts(data["aix_p2p_default"], mapper)
            tex_lines.append(f"  \\draw[colorAixP2P, line width=1.1pt] {p_p2p};")
            for i, y in enumerate(data["aix_p2p_default"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\fill[colorAixP2P] ({cx - 0.04:.3f},{cy - 0.04:.3f}) rectangle ({cx + 0.04:.3f},{cy + 0.04:.3f});")

            p_fc = format_pts(data["aix_p2p_fc_default"], mapper)
            tex_lines.append(f"  \\draw[colorAixP2PFC, line width=1.1pt] {p_fc};")
            for i, y in enumerate(data["aix_p2p_fc_default"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\fill[colorAixP2PFC] ({cx:.3f},{cy + 0.055:.3f}) -- ({cx - 0.045:.3f},{cy - 0.035:.3f}) -- ({cx + 0.045:.3f},{cy - 0.035:.3f}) -- cycle;")

        elif g_idx == 2:
            p_coll = format_pts(data["aix_coll_affine"], mapper)
            tex_lines.append(f"  \\draw[colorAixColl, line width=1.1pt] {p_coll};")
            for i, y in enumerate(data["aix_coll_affine"]):
                tex_lines.append(f"  \\fill[colorAixColl] ({map_x(i):.3f},{mapper(y):.3f}) circle (1.7pt);")

            p_p2p = format_pts(data["aix_p2p_affine"], mapper)
            tex_lines.append(f"  \\draw[colorAixP2P, line width=1.1pt] {p_p2p};")
            for i, y in enumerate(data["aix_p2p_affine"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\fill[colorAixP2P] ({cx - 0.04:.3f},{cy - 0.04:.3f}) rectangle ({cx + 0.04:.3f},{cy + 0.04:.3f});")

            p_fc = format_pts(data["aix_p2p_fc_affine"], mapper)
            tex_lines.append(f"  \\draw[colorAixP2PFC, line width=1.1pt] {p_fc};")
            for i, y in enumerate(data["aix_p2p_fc_affine"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\fill[colorAixP2PFC] ({cx:.3f},{cy + 0.055:.3f}) -- ({cx - 0.045:.3f},{cy - 0.035:.3f}) -- ({cx + 0.045:.3f},{cy - 0.035:.3f}) -- cycle;")

        elif g_idx == 3:
            p_coll = format_pts(data["aix_coll_numa"], mapper)
            tex_lines.append(f"  \\draw[colorAixColl, line width=1.1pt] {p_coll};")
            for i, y in enumerate(data["aix_coll_numa"]):
                tex_lines.append(f"  \\fill[colorAixColl] ({map_x(i):.3f},{mapper(y):.3f}) circle (1.7pt);")

            p_p2p = format_pts(data["aix_p2p_numa"], mapper)
            tex_lines.append(f"  \\draw[colorAixP2P, line width=1.1pt] {p_p2p};")
            for i, y in enumerate(data["aix_p2p_numa"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\fill[colorAixP2P] ({cx - 0.04:.3f},{cy - 0.04:.3f}) rectangle ({cx + 0.04:.3f},{cy + 0.04:.3f});")

            p_fc = format_pts(data["aix_p2p_fc_numa"], mapper)
            tex_lines.append(f"  \\draw[colorAixP2PFC, line width=1.1pt] {p_fc};")
            for i, y in enumerate(data["aix_p2p_fc_numa"]):
                cx, cy = map_x(i), mapper(y)
                tex_lines.append(f"  \\fill[colorAixP2PFC] ({cx:.3f},{cy + 0.055:.3f}) -- ({cx - 0.045:.3f},{cy - 0.035:.3f}) -- ({cx + 0.045:.3f},{cy - 0.035:.3f}) -- cycle;")

        tex_lines.append(r"\end{scope}")
        tex_lines.append("")

    # ==================== SHARED LEGEND BELOW ====================
    total_plot_width = x_offsets[3] + W
    box_left = -0.90
    box_w = total_plot_width - box_left
    leg_y = -1.05  # cm below x-axis
    box_h = 0.95

    tex_lines.append("% ==================== SHARED LEGEND ====================")
    tex_lines.append(f"\\begin{{scope}}[shift={{(0,{leg_y:.3f})}}]")
    tex_lines.append(f"  \\draw[draw=black!25, fill=gray!3, rounded corners=2pt, line width=0.45pt]")
    tex_lines.append(f"    ({box_left:.3f}, -{box_h:.3f}) rectangle ({total_plot_width:.3f}, 0);")

    # 3 equal columns inside the legend box
    cw = box_w / 3.0
    c_starts = [box_left + 0.35 + i * cw for i in range(3)]

    # Row 1: AIx modes (y = -0.28)
    r1_y = -0.28
    # Item 1: AIx Collective
    tex_lines.append(f"  \\draw[colorAixColl, line width=1.1pt] ({c_starts[0]:.3f},{r1_y}) -- ({c_starts[0]+0.45:.3f},{r1_y});")
    tex_lines.append(f"  \\fill[colorAixColl] ({c_starts[0]+0.225:.3f},{r1_y}) circle (1.7pt);")
    tex_lines.append(f"  \\node[legend label, anchor=west] at ({c_starts[0]+0.55:.3f},{r1_y}) {{AIx Collective}};")

    # Item 2: AIx P2P (Credit = 1)
    tex_lines.append(f"  \\draw[colorAixP2P, line width=1.1pt] ({c_starts[1]:.3f},{r1_y}) -- ({c_starts[1]+0.45:.3f},{r1_y});")
    tex_lines.append(f"  \\fill[colorAixP2P] ({c_starts[1]+0.225-0.04:.3f},{r1_y-0.04:.3f}) rectangle ({c_starts[1]+0.225+0.04:.3f},{r1_y+0.04:.3f});")
    tex_lines.append(f"  \\node[legend label, anchor=west] at ({c_starts[1]+0.55:.3f},{r1_y}) {{AIx P2P (Credit = 1)}};")

    # Item 3: AIx P2P (Full Credits)
    tex_lines.append(f"  \\draw[colorAixP2PFC, line width=1.1pt] ({c_starts[2]:.3f},{r1_y}) -- ({c_starts[2]+0.45:.3f},{r1_y});")
    tex_lines.append(f"  \\fill[colorAixP2PFC] ({c_starts[2]+0.225:.3f},{r1_y+0.055:.3f}) -- ({c_starts[2]+0.225-0.045:.3f},{r1_y-0.035:.3f}) -- ({c_starts[2]+0.225+0.045:.3f},{r1_y-0.035:.3f}) -- cycle;")
    tex_lines.append(f"  \\node[legend label, anchor=west] at ({c_starts[2]+0.55:.3f},{r1_y}) {{AIx P2P (Full Credits)}};")

    # Row 2: SmartSim modes (y = -0.66)
    r2_y = -0.66
    # Item 4: SmartSim (default) [replaces SmartSim c0]
    tex_lines.append(f"  \\draw[colorSSZero, line width=1.0pt, dashed] ({c_starts[0]:.3f},{r2_y}) -- ({c_starts[0]+0.45:.3f},{r2_y});")
    tex_lines.append(f"  \\fill[colorSSZero] ({c_starts[0]+0.225:.3f},{r2_y}) circle (1.6pt);")
    tex_lines.append(f"  \\node[legend label, anchor=west] at ({c_starts[0]+0.55:.3f},{r2_y}) {{SmartSim (default)}};")

    # Item 5: SmartSim c3 (Sequential chain)
    tex_lines.append(f"  \\draw[colorSSThree, line width=1.0pt, dashdotted] ({c_starts[1]:.3f},{r2_y}) -- ({c_starts[1]+0.45:.3f},{r2_y});")
    tex_lines.append(f"  \\fill[colorSSThree] ({c_starts[1]+0.225:.3f},{r2_y}) +(-1.6pt,0) -- +(0,1.6pt) -- +(1.6pt,0) -- +(0,-1.6pt) -- cycle;")
    tex_lines.append(f"  \\node[legend label, anchor=west] at ({c_starts[1]+0.55:.3f},{r2_y}) {{SmartSim $c=3$ (Sequential)}};")

    # Item 6: SmartSim Per-GPU Database
    tex_lines.append(f"  \\draw[colorSSPgdb, line width=1.0pt, densely dotted] ({c_starts[2]:.3f},{r2_y}) -- ({c_starts[2]+0.45:.3f},{r2_y});")
    tex_lines.append(f"  \\draw[colorSSPgdb, line width=0.8pt, fill=white] ({c_starts[2]+0.225-0.045:.3f},{r2_y-0.045:.3f}) rectangle ({c_starts[2]+0.225+0.045:.3f},{r2_y+0.045:.3f});")
    tex_lines.append(f"  \\node[legend label, anchor=west] at ({c_starts[2]+0.55:.3f},{r2_y}) {{SmartSim Per-GPU DB}};")

    tex_lines.append(r"\end{scope}")
    tex_lines.append(r"\end{tikzpicture}")
    tex_lines.append(r"\end{document}")

    output_tex.write_text("\n".join(tex_lines) + "\n")
    print(f"Generated {output_tex}")

if __name__ == "__main__":
    nodes, data = compute_medians()
    out_file = MINI_APP_DIR / "scaling_4x1_standalone.tex"
    generate_tikz(nodes, data, out_file)
