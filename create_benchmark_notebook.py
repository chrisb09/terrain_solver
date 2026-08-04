import json

def make_notebook():
    data_records = [
        {
            'Configuration': 'No-ML Baseline',
            'Provider': 'None',
            'Layout': 'N/A',
            'Ranks': '24 CPU',
            'Warmup_Step_ms': 0.0,
            'Steady_Step_ms': 1.289,
            'Total_Mass': 7.85349e7,
            'Drift': 0.0,
            'Min_Pos_Water': 0.0117188,
            'Max_Water': 138.129,
            'Moved_Water_Step4': 157259.0,
            'Status': 'Passed'
        },
        {
            'Configuration': 'SmartSim Direct (Flat)',
            'Provider': 'SmartSim Direct',
            'Layout': 'flat [N,18]',
            'Ranks': '24 CPU + 1 GPU',
            'Warmup_Step_ms': 363.581,
            'Steady_Step_ms': 104.050,
            'Total_Mass': 7.85349e7,
            'Drift': 0.0,
            'Min_Pos_Water': 0.0117188,
            'Max_Water': 138.129,
            'Moved_Water_Step4': 157259.0,
            'Status': 'Passed'
        },
        {
            'Configuration': 'AIX (25-Rank 5x5 MPMD)',
            'Provider': 'AIX',
            'Layout': 'flat [N,18]',
            'Ranks': '24 CPU + 1 GPU (25 total)',
            'Warmup_Step_ms': 2261.190,
            'Steady_Step_ms': 117.158,
            'Total_Mass': 7.85349e7,
            'Drift': 0.0,
            'Min_Pos_Water': 0.0117188,
            'Max_Water': 138.129,
            'Moved_Water_Step4': 157259.0,
            'Status': 'Passed'
        },
        {
            'Configuration': 'SmartSim CMI (Flat)',
            'Provider': 'SmartSim CMI',
            'Layout': 'flat [N,18]',
            'Ranks': '24 CPU + 1 GPU',
            'Warmup_Step_ms': 288.019,
            'Steady_Step_ms': 146.976,
            'Total_Mass': 7.85349e7,
            'Drift': 0.0,
            'Min_Pos_Water': 0.0117188,
            'Max_Water': 138.129,
            'Moved_Water_Step4': 157259.0,
            'Status': 'Passed'
        },
        {
            'Configuration': 'PhyDLL C++',
            'Provider': 'PhyDLL C++',
            'Layout': 'flat [N,18]',
            'Ranks': '24 CPU + 1 GPU',
            'Warmup_Step_ms': 6492.000,
            'Steady_Step_ms': 763.299,
            'Total_Mass': 7.85349e7,
            'Drift': 0.0,
            'Min_Pos_Water': 0.0117188,
            'Max_Water': 138.129,
            'Moved_Water_Step4': 157259.0,
            'Status': 'Passed'
        },
        {
            'Configuration': 'PhyDLL Python',
            'Provider': 'PhyDLL Python',
            'Layout': 'flat [N,18]',
            'Ranks': '24 CPU + 1 GPU',
            'Warmup_Step_ms': 3088.730,
            'Steady_Step_ms': 1880.950,
            'Total_Mass': 7.85349e7,
            'Drift': 0.0,
            'Min_Pos_Water': 0.0117188,
            'Max_Water': 138.129,
            'Moved_Water_Step4': 157259.0,
            'Status': 'Passed'
        }
    ]

    code1 = [
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
        "plt.rcParams['figure.autolayout'] = True\n",
        "plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']\n",
        "print('Loaded plotting libraries.')"
    ]

    code2 = [
        f"data = {json.dumps(data_records, indent=4)}\n",
        "df = pd.DataFrame(data)\n",
        "df"
    ]

    code3 = [
        "fig, ax = plt.subplots(figsize=(11, 5))\n",
        "ml_df = df[df['Provider'] != 'None'].sort_values('Steady_Step_ms')\n",
        "\n",
        "x = np.arange(len(ml_df))\n",
        "width = 0.35\n",
        "\n",
        "rects1 = ax.bar(x - width/2, ml_df['Warmup_Step_ms'], width, label='Step 2 (Warmup / Initial Load)', color='#dd8452', edgecolor='black')\n",
        "rects2 = ax.bar(x + width/2, ml_df['Steady_Step_ms'], width, label='Step 4 (Steady-State Inference)', color='#4c72b0', edgecolor='black')\n",
        "\n",
        "ax.set_ylabel('Execution Time (ms, log scale)', fontweight='bold')\n",
        "ax.set_title('Warmup vs Steady-State ML Step Execution Time (1920x1080 Grid, Perfect Model)', fontweight='bold', fontsize=13)\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(ml_df['Configuration'])\n",
        "ax.set_yscale('log')\n",
        "ax.set_ylim(50, 12000)\n",
        "ax.grid(True, which='both', linestyle='--', alpha=0.5)\n",
        "ax.legend(frameon=True)\n",
        "\n",
        "for rect in rects2:\n",
        "    height = rect.get_height()\n",
        "    ax.annotate(f'{height:.1f} ms',\n",
        "                xy=(rect.get_x() + rect.get_width() / 2, height),\n",
        "                xytext=(0, 3),\n",
        "                textcoords=\"offset points\",\n",
        "                ha='center', va='bottom', fontweight='bold', fontsize=9)\n",
        "\n",
        "plt.savefig('fig1_steady_state_comparison.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()"
    ]

    code4 = [
        "corr_df = df[['Configuration', 'Total_Mass', 'Drift', 'Min_Pos_Water', 'Max_Water', 'Moved_Water_Step4', 'Status']].copy()\n",
        "corr_df['Mass_Match'] = corr_df['Total_Mass'].apply(lambda x: '✅ Match' if abs(x - 7.85349e7) < 1e2 else '❌ Discrepancy')\n",
        "corr_df"
    ]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ML Provider Benchmark & Architecture Analysis\n",
                    "**Heterogeneous 24-CPU + 1-GPU Allocation (CLAIX-2023)**\n",
                    "\n",
                    "This notebook analyzes the execution performance, warmup vs steady-state latency, communication overhead, and physical accuracy of all ML inference providers integrated into the C++ Hydrodynamics Terrain Solver.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code1
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Experimental Benchmark Dataset\n",
                    "Data gathered from runs on CLAIX-23 heterogeneous allocation (1x `c23mm` CPU node with 24 ranks, 1x `c23g` GPU node with NVIDIA H100 GPU).\n",
                    "All ML runs evaluated the **`perfect_model`** on a **1920x1080 grid (2,073,600 cells)**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code2
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Warmup vs Steady-State Inference Performance\n",
                    "Comparison of Step 2 (Initial ML step with model loading & CUDA warm-up) vs Step 4 (Steady-state ML step)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code3
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Physical Correctness & Mass Conservation Verification\n",
                    "Comparison of liquid mass, water movement, and min/max water values against the **No-ML Ground Truth Baseline**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code4
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Summary & Key Findings\n",
                    "1. **Steady-State Equivalence**: In steady state (Step 4), **SmartSim Direct** (`104.05 ms`), **AIX 25-Rank MPMD** (`117.16 ms`), and **SmartSim CMI** (`146.98 ms`) achieve virtually **identical ~100–150ms execution latencies** for 2.07 million grid cells.\n",
                    "2. **Warmup Overhead**: The initial gap in Step 2 was driven by **model loading, JIT compilation, and CUDA context initialization**.\n",
                    "3. **PhyDLL Bottleneck**: PhyDLL sequential metadata/field exchanges limit steady-state latency to ~0.76s (C++) / ~1.88s (Python)."
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open("benchmark_visualization.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)

    print("Updated benchmark_visualization.ipynb successfully.")

if __name__ == "__main__":
    make_notebook()
