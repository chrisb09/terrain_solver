#!/usr/bin/env python3
"""Generate standalone previews and independently includable TikZ/PGF fragments."""

from pathlib import Path

import numpy as np
import pandas as pd

MINI_APP_DIR = Path(__file__).resolve().parent
CSV_HET3 = MINI_APP_DIR / "results/hethires3/ml_step_makespan.csv"
CSV_HET2 = MINI_APP_DIR / "results/hethires2/ml_step_makespan.csv"

# label, color, line pattern, marker; reused in panels and legends.
STYLES = {
    "aix_coll": ("AIx Collective", "1F77B4", "solid", "circle"),
    "aix_p2p": ("AIx P2P (Credit = 1)", "FF7F0E", "solid", "square"),
    "aix_p2p_fc": ("AIx P2P (Full Credits)", "2CA02C", "solid", "triangle"),
    "phydll_default": ("PhyDLL (default)", "008080", "dotted", "diamond"),
    "phydll_readiness": ("PhyDLL (readiness)", "17BECF", "dashdotted", "circle"),
    "smartsim_c0": ("SmartSim (default)", "D62728", "dashed", "circle"),
    "smartsim_c3": ("SmartSim c=3 (Sequential)", "9467BD", "dashdotted", "diamond"),
    "smartsim_per_gpu_db": ("SmartSim Per-GPU DB", "8C564B", "densely dotted", "open square"),
}


def compute_medians():
    df3 = pd.read_csv(CSV_HET3)
    df2 = pd.read_csv(CSV_HET2)
    s3 = df3[df3["solver_step"] > 2]
    s2 = df2[df2["solver_step"] > 2]
    scenarios = [7, 8, 9, 10, 11]
    curves = {
        "smartsim_c0": [f"s{s}_smartsim_c0_default" if s > 7 else "s7_smartsim_c0" for s in scenarios],
        "smartsim_c3": [f"s{s}_smartsim_c3_default" if s > 7 else "s7_smartsim_c3" for s in scenarios],
        "smartsim_per_gpu_db": [f"s{s}_smartsim_per_gpu_db_default" for s in scenarios],
        "phydll_default": [f"s{s}_phydll_default" for s in scenarios],
        "phydll_readiness": [f"s{s}_phydll_readiness" for s in scenarios],
    }
    for layout in ("default", "affine", "numa"):
        for mode, column in (("coll", "coll"), ("p2p", "p2p"), ("p2p_fc", "p2p_fullcredits")):
            curves[f"aix_{mode}_{layout}"] = [f"s{s}_aix_{column}_{layout}" for s in scenarios]

    data = {}
    for name, columns in curves.items():
        values = []
        for column in columns:
            series = s3[column].dropna() if column in s3 else s2[column].dropna() if column in s2 else []
            values.append(float(np.median(series)) if len(series) else None)
        data[name] = values
    return [1, 2, 4, 8, 16], data


def figure_specs():
    reference = [("aix_coll_default", "aix_coll"), ("phydll_default", "phydll_default"),
                 ("phydll_readiness", "phydll_readiness")]
    smartsim = [(key, key) for key in ("smartsim_c0", "smartsim_c3", "smartsim_per_gpu_db")]
    return [
        ("scaling_framework_comparison", "Framework Comparison (Default Layout)", [
            ("(a) All Frameworks", 22000, [0, 5000, 10000, 15000, 20000], reference + smartsim),
            ("(b) Without SmartSim", 1000, list(range(0, 1001, 200)), reference),
        ], [key for _, key in reference + smartsim]),
        ("scaling_aix_layouts", "AIx Layout Comparison", [
            (f"({letter}) {layout.title() if layout != 'numa' else 'NUMA'}", 250,
             list(range(0, 251, 50)), [(f"{key}_{layout}", key) for key in ("aix_coll", "aix_p2p", "aix_p2p_fc")])
            for letter, layout in zip("abc", ("default", "affine", "numa"))
        ], ["aix_coll", "aix_p2p", "aix_p2p_fc"]),
    ]


def validate_panels(nodes, data, panels):
    for title, maximum, _, curves in panels:
        for name, _ in curves:
            values = data[name]
            if len(values) != len(nodes):
                raise ValueError(f"{title}: {name} has {len(values)} points for {len(nodes)} nodes")
            for node, value in zip(nodes, values):
                if value is None or not np.isfinite(value) or not 0 <= value <= maximum:
                    raise ValueError(f"{title}: {name} at {node} nodes: {value} outside [0, {maximum}] ms")


def draw_marker(key, x, y):
    marker = STYLES[key][3]
    color = f"c{key}"
    point = f"({x:.4f},{y:.4f})"
    if marker == "circle":
        return f"\\fill[{color}] {point} circle (1.6pt);"
    if marker in ("square", "open square"):
        fill = "white" if marker == "open square" else color
        return f"\\draw[{color},fill={fill},line width=0.8pt] ({x-0.045:.4f},{y-0.045:.4f}) rectangle ({x+0.045:.4f},{y+0.045:.4f});"
    if marker == "diamond":
        return f"\\fill[{color}] {point} +(-1.8pt,0) -- +(0,1.8pt) -- +(1.8pt,0) -- +(0,-1.8pt) -- cycle;"
    return f"\\fill[{color}] ({x:.4f},{y+0.055:.4f}) -- ({x-0.05:.4f},{y-0.035:.4f}) -- ({x+0.05:.4f},{y-0.035:.4f}) -- cycle;"


def generate_tikz(nodes, data, output_tex, title, panels, legend):
    validate_panels(nodes, data, panels)
    count = len(panels)
    width, gap = (5.65, 1.65) if count == 2 else (3.95, 0.55)
    height = 2.1
    total_width = count * width + (count - 1) * gap
    lines = [
        r"\documentclass[tikz,border=4pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        r"\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont,",
        r"axis/.style={black!85,line width=0.5pt},",
        r"grid/.style={black!14,line width=0.35pt,dotted},",
        r"tick/.style={font=\fontsize{7}{8.5}\selectfont},",
        r"title/.style={font=\fontsize{8}{10}\selectfont\bfseries}]",
    ]
    for key, (_, color, _, _) in STYLES.items():
        lines.append(f"\\definecolor{{c{key}}}{{HTML}}{{{color}}}")
    picture_header = lines[4:].copy()
    panel_names = ("all", "without_smartsim") if count == 2 else ("default", "affine", "numa")
    fragments = {}
    lines.append(f"\\node[title] at ({total_width/2:.3f},{height+0.90:.3f}) {{{title}}};")

    for index, (panel_title, maximum, ticks, curves) in enumerate(panels):
        lines.append(f"\\begin{{scope}}[shift={{({index*(width+gap):.3f},0)}}]")
        panel_start = len(lines)
        lines.append(f"\\node[title] at ({width/2:.3f},{height+0.35:.3f}) {{{panel_title}}};")
        for tick in ticks:
            y = tick / maximum * height
            lines.append(f"\\draw[grid] (0,{y:.4f}) -- ({width:.3f},{y:.4f});")
            lines.append(f"\\draw[axis] (0,{y:.4f}) -- (-0.07,{y:.4f});")
            if count == 2 or index == 0:
                lines.append(f"\\node[tick,anchor=east] at (-0.12,{y:.4f}) {{{tick:,}}};")
        for i, node in enumerate(nodes):
            x = 0.08 + i * (width - 0.16) / (len(nodes) - 1)
            lines.append(f"\\draw[grid] ({x:.4f},0) -- ({x:.4f},{height});")
            lines.append(f"\\draw[axis] ({x:.4f},0) -- ({x:.4f},-0.07);")
            lines.append(f"\\node[tick,anchor=north] at ({x:.4f},-0.12) {{{node}}};")
        lines.append(f"\\node at ({width/2:.3f},-0.65) {{Solver Nodes}};")
        if count == 2 or index == 0:
            label_x = (-1.25 if index == 0 else -1.05) if count == 2 else -1.2
            lines.append(f"\\node[rotate=90,align=center] at ({label_x},{height/2:.3f}) {{Median ML-Step\\\\Time (ms)}};")

        # Validate before clipping: clipping must not conceal out-of-range data.
        lines.extend([r"\begin{scope}", f"\\clip (0,0) rectangle ({width},{height});"])
        for name, key in curves:
            points = [(0.08 + i * (width - 0.16) / (len(nodes)-1), value / maximum * height)
                      for i, value in enumerate(data[name])]
            path = " -- ".join(f"({x:.4f},{y:.4f})" for x, y in points)
            lines.append(f"% {name}: " + ", ".join(f"{v:.5f}" for v in data[name]) + " ms")
            lines.append(f"\\draw[c{key},line width=1pt,{STYLES[key][2]}] {path};")
            lines.extend(draw_marker(key, x, y) for x, y in points)
        lines.extend([r"\end{scope}", f"\\draw[axis] (0,0) rectangle ({width},{height});", r"\end{scope}"])
        fragments[panel_names[index]] = lines[panel_start:-1]

    # One row for the shared reference curves, another for SmartSim (left only).
    rows = (len(legend) + 2) // 3
    left, right = -0.75, total_width + 0.1
    top, bottom = -1.05, -1.05 - rows * 0.5 - 0.15
    legend_start = len(lines)
    lines.append(f"\\draw[black!25,fill=gray!3,rounded corners=2pt] ({left},{bottom}) rectangle ({right:.3f},{top});")
    column_width = (right - left) / 3
    for index, key in enumerate(legend):
        x = left + 0.18 + (index % 3) * column_width
        y = top - 0.33 - (index // 3) * 0.5
        label, _, pattern, _ = STYLES[key]
        lines.append(f"\\draw[c{key},line width=1pt,{pattern}] ({x:.3f},{y:.3f}) -- ({x+0.42:.3f},{y:.3f});")
        lines.append(draw_marker(key, x+0.21, y))
        lines.append(f"\\node[anchor=west,font=\\fontsize{{7}}{{8.5}}\\selectfont] at ({x+0.5:.3f},{y:.3f}) {{{label}}};")
    fragments["legend"] = lines[legend_start:]
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    output_tex.write_text("\n".join(lines) + "\n")
    print(f"Generated {output_tex}")
    for suffix, content in fragments.items():
        output_pgf = output_tex.with_name(f"{output_tex.stem}_{suffix}.pgf")
        output_pgf.write_text("\n".join([
            "% Generated TikZ/PGF fragment; requires \\usepackage{tikz}.",
            *picture_header, *content, r"\end{tikzpicture}",
        ]) + "\n")
        print(f"Generated {output_pgf}")


if __name__ == "__main__":
    nodes, data = compute_medians()
    specs = figure_specs()
    # Check both figures before replacing either generated source.
    for _, _, panels, _ in specs:
        validate_panels(nodes, data, panels)
    for name, title, panels, legend in specs:
        generate_tikz(nodes, data, MINI_APP_DIR / f"{name}.tex", title, panels, legend)
