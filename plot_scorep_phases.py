#!/usr/bin/env python3
"""Render separate application and provider mean-phase figures from extracted CSVs."""
import argparse
import csv
import math
from decimal import Decimal
from pathlib import Path

from extract_scorep_tables import ALIASES, ORDER
from analyze_cmi_scorep_profiles import COLOR_MAP
from hh3_render import PGF, plt, save_figure, close_figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text

MODELS = ["watercnn", "giant"]
TITLES = {"watercnn": "[TS] WaterCNN", "giant": "[TS] Giant MLP"}
PHASES = {
    "application": ["app_prepare_input", "app_provider_inference", "app_finalize_output", "Other / overhead"],
    "provider_dl": ["dl_recv / dl_frame_copy", "dl_allocate", "dl_input_unpack", "dl_h2d",
                    "dl_torch_forward", "dl_d2h", "dl_output_reorder", "dl_send_output"],
    "provider": ["smartsim_chunk_plan", "smartsim_token_wait", "smartsim_put_tensor",
                 "smartsim_run_model", "smartsim_unpack_tensor", "Other / overhead"],
}
LABELS = {"smartsim_unpack_tensor": "smartsim_unpack_tensor (retrieval + unpack)",
          "smartsim_run_model": "smartsim_run_model (model request)",
           "dl_recv / dl_frame_copy": "dl_recv + dl_frame_copy (DL receive/handling)",
           "dl_allocate": "dl_input_allocate + dl_output_allocate",
           "dl_h2d": "dl_h2d (input tensor staging)",
           "dl_d2h": "dl_d2h (output tensor staging)",
            "dl_send_output": "dl_send_output (DL-side output transfer)"}


CONFIG_LABELS = {
    "AIx": "AIx", "PHYDLL_CPP": "PhyDLL C++", "SS_DEFAULT": "SmartSim\ndefault",
    "TPQ1_I96": "SmartSim\nTPQ=1\nINTRA=96",
    "TPQ12_I8": "SmartSim\nTPQ=12\nINTRA=8",
    "TPQ96_I1": "SmartSim\nTPQ=96\nINTRA=1",
}
SIMPLIFIED = {
    "AIx": [("Inference", ["app_provider_inference"])],
    "PHYDLL_CPP": [
        ("Input receive", ["dl_recv / dl_frame_copy"]),
        ("Forward", ["dl_torch_forward"]),
        ("Output send", ["dl_send_output"]),
        ("Other", ["dl_allocate", "dl_input_unpack", "dl_h2d", "dl_d2h",
                   "dl_output_reorder", "Other / overhead"]),
    ],
    "SmartSim": [
        ("Put", ["smartsim_put_tensor"]),
        ("Model request", ["smartsim_run_model"]),
        ("Retrieval/unpack", ["smartsim_unpack_tensor"]),
        ("Other", ["smartsim_chunk_plan", "smartsim_token_wait", "Other / overhead"]),
    ],
}
SIMPLE_COLORS = {
    "Inference": "#8064A2", "Input receive": "#4C78A8", "Forward": "#F58518",
    "Output send": "#54A24B", "Put": "#72B7B2", "Model request": "#E45756",
    "Retrieval/unpack": "#B279A2", "Other": "#BAB0AC",
    "Prepare input": "#4C78A8", "Finalize output": "#54A24B",
}


def format_ms(value):
    """Use compact fixed decimals, preserving nonzero tiny measurements."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    text = format(Decimal(f"{value:.3g}"), 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def column_data(rows, application=False):
    """Aggregate only extracted disjoint self bins, never inclusive callpaths."""
    data = {}
    for r in rows:
        cfg = ALIASES.get(r['config'], r['config'])
        view = 'application' if application or cfg == 'AIx' else (
            'provider_dl' if cfg == 'PHYDLL_CPP' else 'provider')
        if cfg not in ORDER or r['model'] not in MODELS or r['view'] != view:
            continue
        key = (r['model'], cfg, r['phase'])
        value = float(r['mean_ms_per_step'])
        if key in data or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid or duplicate phase: {key}')
        scope = 'dl_steady_count_validated' if view == 'provider_dl' else 'solver_steady'
        if r['scope'] != scope:
            raise ValueError(f'Unexpected scope for {key}: {r["scope"]}')
        data[key] = r
    panels = {}
    for model in MODELS:
        panels[model] = []
        for cfg in ORDER:
            groups = ([(label, [phase]) for label, phase in zip(
                ['Prepare input', 'Inference', 'Finalize output', 'Other'], PHASES['application'])]
                if application else SIMPLIFIED.get(cfg, SIMPLIFIED['SmartSim']))
            required = [phase for _, phases in groups for phase in phases]
            missing = [p for p in required if (model, cfg, p) not in data]
            if missing:
                raise ValueError(f'Missing phases: {model} {cfg}: {missing}')
            entries = [data[(model, cfg, p)] for p in required]
            counts = {int(r['rep_count']) for r in entries}
            if len(counts) != 1 or min(counts) < 1:
                raise ValueError(f'Inconsistent replicate coverage: {model} {cfg}')
            # DL total is the selected instrumented DL sum, not a solver residual.
            values = [(label, sum(float(data[(model, cfg, p)]['mean_ms_per_step'])
                                  for p in phases)) for label, phases in groups]
            panels[model].append((cfg, next(iter(counts)), values))
    return panels


def render_columns(rows, output, stacked, application=False):
    panels = column_data(rows, application)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    legend_phases = []
    label_lanes = []
    for ax, model in zip(axes, MODELS):
        series = panels[model]
        limit = max(sum(v for _, v in phases) if stacked else max(v for _, v in phases)
                    for _, _, phases in series) or 1
        highest = limit
        ticks, ticklabels = [], []
        for ci, (cfg, _, phases) in enumerate(series):
            x = float(ci)
            ticks.append(x)
            ticklabels.append(CONFIG_LABELS[cfg])
            bottom, external_y = 0.0, -.03 * limit
            external_labels = []
            for pi, (phase, value) in enumerate(phases):
                if phase not in legend_phases:
                    legend_phases.append(phase)
                color = SIMPLE_COLORS[phase]
                xpos = x if stacked else x + (pi - (len(phases) - 1) / 2) * .19
                ax.bar(xpos, value, bottom=bottom if stacked else 0,
                       width=.65 if stacked else .18, color=color,
                       edgecolor='white' if stacked else '#444444', linewidth=.4)
                if not stacked:
                    ax.annotate(format_ms(value), (xpos, value), xytext=(0, 4),
                                textcoords='offset points', ha='center', va='bottom',
                                fontsize=6.5, rotation=90)
                elif value >= .075 * limit and len(format_ms(value)) <= 7:
                    rgb = color.lstrip('#')
                    light = sum(int(rgb[j:j+2], 16) for j in (0, 2, 4)) / 3
                    ax.text(x, bottom + value / 2, format_ms(value), ha='center',
                            va='center', fontsize=7, color='white' if light < 130 else 'black')
                else:
                    # Distinct external lanes also label zero/tiny segments. Leaders
                    # retain their actual stack positions even when labels are shifted.
                    external_y = max(bottom + value / 2, external_y + .16 * limit)
                    annotation = ax.annotate(format_ms(value), (x + .325, bottom + value / 2),
                                xytext=(x + .39, external_y), textcoords='data',
                                ha='left', va='center', fontsize=6.5, rotation=90,
                                 arrowprops={'arrowstyle': '-', 'lw': .4, 'color': color})
                    external_labels.append(annotation)
                bottom += value
            if stacked:
                total_y = max(bottom, external_y) + .045 * limit
                total_label = ax.text(x, total_y, f'Total {format_ms(bottom)}', ha='center', va='bottom',
                                      fontsize=7, rotation=90 if len(format_ms(bottom)) > 7 else 0)
                label_lanes.append((ax, external_labels, total_label, bottom))
                highest = max(highest, total_y)
        ax.set_title(TITLES[model])
        ax.set_xticks(ticks, ticklabels, fontsize=8)
        ax.set_xlim(ticks[0] - .5, ticks[-1] + .6)
        ax.set_ylim(0, highest + .27 * limit)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_ms(value)))
        ax.set_ylabel('Mean ms / steady step / scope rank')
        ax.grid(False)
        ax.grid(axis='y', alpha=.25, linewidth=.4)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    title = ('Application phases: solver-rank steady scope\nDisjoint steady application phases'
             if application else 'Library inference phases')
    fig.suptitle(title, fontsize=10)
    legend_labels = {} if application else {
        'Inference': 'Inference (AIx library call)',
        'Input receive': 'Input receive (DL-side receive + handling)',
        'Forward': 'Forward (DL-side model execution)',
        'Output send': 'Output send (DL-side output transfer)',
        'Put': 'Put (client tensor submission)',
        'Model request': 'Model request (client elapsed time)',
        'Retrieval/unpack': 'Retrieval/unpack (client retrieval + unpack)',
        'Other': 'Other (staging / bookkeeping)',
    }
    if not application:
        # Legend columns fill top-to-bottom; interleave the two visual rows.
        legend_phases = ['Input receive', 'Put', 'Forward', 'Model request',
                         'Output send', 'Retrieval/unpack', 'Other', 'Inference']
    legend = fig.legend(handles=[Patch(facecolor=SIMPLE_COLORS[p], label=legend_labels.get(p, p))
                    for p in legend_phases],
               loc='lower center', bbox_to_anchor=(.5, .025 if not application else .045),
               ncol=4, fontsize=7, frameon=False,
               borderaxespad=.5 if application else 0,
               labelspacing=.5 if application else 1.5)
    counts = sorted({reps for series in panels.values() for _, reps, _ in series})
    coverage = (f'n={counts[0]} reps per configuration' if len(counts) == 1 else
                'Rep counts: ' + '; '.join(f'{TITLES[m]}: ' + ', '.join(
                    f'{cfg}={reps}' for cfg, reps, _ in panels[m]) for m in MODELS))
    if application:
        fig.text(.5, .015, coverage, ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .14 if not application else .12, 1, .98), w_pad=1.5)
    if stacked:
        # Pack actual text extents, excluding leader lines. Fixed decimal labels
        # can be much longer than scientific notation, especially for tiny bins.
        for _ in range(8):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            upper = {ax: ax.get_ylim()[1] for ax in axes}
            for ax, labels, total, bottom in label_lanes:
                previous_top = ax.bbox.y0
                for label in labels:
                    bounds = Text.get_window_extent(label, renderer)
                    shift = max(0, previous_top + 4 - bounds.y0)
                    if shift:
                        px, py = ax.transData.transform(label.get_position())
                        label.set_position(ax.transData.inverted().transform((px, py + shift)))
                    previous_top = bounds.y1 + shift
                total_bottom = max(previous_top, ax.transData.transform((0, bottom))[1]) + 5
                total.set_y(ax.transData.inverted().transform((0, total_bottom))[1])
                top = Text.get_window_extent(total, renderer).y1 + 5
                upper[ax] = max(upper[ax], ax.transData.inverted().transform((0, top))[1])
            if all(upper[ax] <= ax.get_ylim()[1] for ax in axes):
                break
            for ax in axes:
                ax.set_ylim(0, upper[ax] * 1.03)
    if not application:
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.transforms import Bbox

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Position below the actual multiline ticks, rather than the axes alone.
        tick_bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes)
        gap = renderer.points_to_pixels(7)
        height = legend.get_window_extent(renderer).height
        legend.set_bbox_to_anchor((.5, (tick_bottom - gap - height) / fig.bbox.height))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        handles = legend.get_patches()
        texts = legend.get_texts()
        padding = renderer.points_to_pixels(3)
        for indices in ((0, 2, 4, 6), (1, 3, 5)):
            bounds = Bbox.union([artist.get_window_extent(renderer)
                                 for i in indices for artist in (handles[i], texts[i])])
            bounds = Bbox.from_extents(bounds.x0 - padding, bounds.y0 - padding,
                                       bounds.x1 + padding, bounds.y1 + padding)
            bounds = bounds.transformed(fig.transFigure.inverted())
            fig.add_artist(FancyBboxPatch(
                (bounds.x0, bounds.y0), bounds.width, bounds.height,
                boxstyle='round,pad=0,rounding_size=0.006',
                transform=fig.transFigure, facecolor='none', edgecolor='#777777',
                linewidth=.7, linestyle='--'))
    name = 'application_stacked' if application else f'simplified_provider_{"stacked" if stacked else "grouped"}'
    save_figure(fig, output / f'{name}.png')
    if not PGF:
        fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    close_figure(fig)


def render_combined(rows, output, stacked):
    """Compare five providers without conflating their process/measurement scopes."""
    configs = ORDER[1:]
    phases = list(dict.fromkeys(PHASES['provider_dl'] + PHASES['provider']))
    colors = {p: COLOR_MAP.get(p.split(' / ')[0], '#BDBDBD') for p in phases}
    colors['dl_allocate'] = '#E0E0E0'
    data = {}
    for r in rows:
        if r['config'] not in configs or r['view'] not in ('provider_dl', 'provider'):
            continue
        key = (r['model'], r['config'], r['phase'])
        if key in data or not math.isfinite(float(r['mean_ms_per_step'])) or float(r['mean_ms_per_step']) < 0:
            raise ValueError(f'Invalid or duplicate provider phase: {key}')
        expected_scope = 'dl_steady_count_validated' if r['config'] == 'PHYDLL_CPP' else 'solver_steady'
        if r['scope'] != expected_scope:
            raise ValueError(f'Unexpected scope for {key}: {r["scope"]}')
        data[key] = r
    fig, axes = plt.subplots(1, 2, figsize=(16, 7 if stacked else 13))
    for ax, model in zip(axes, MODELS):
        ax.set_title(TITLES[model])
        ticks, labels, totals = [], [], []
        for ci, cfg in enumerate(configs):
            active = PHASES['provider_dl' if cfg == 'PHYDLL_CPP' else 'provider']
            entries = [data[(model, cfg, p)] for p in active]
            counts = {int(r['rep_count']) for r in entries}
            if len(counts) != 1 or min(counts) < 1:
                raise ValueError(f'Inconsistent replicate coverage: {model} {cfg}')
            scope = 'DL ranks' if cfg == 'PHYDLL_CPP' else 'solver-client ranks'
            label = f'{cfg} [{scope}; n={next(iter(counts))} reps]'
            values = [float(r['mean_ms_per_step']) for r in entries]
            if stacked:
                left = 0
                for p, value in zip(active, values):
                    ax.barh(ci, value, left=left, height=.6, color=colors[p], edgecolor='white', linewidth=.4)
                    left += value
                totals.append(left)
                ticks.append(ci)
                labels.append(label.replace(' [', '\n['))
            else:
                start = ci * 10
                ax.text(0, start, label, transform=ax.get_yaxis_transform(), fontsize=8, weight='bold')
                for pi, (p, value) in enumerate(zip(active, values)):
                    y = start + pi + 1
                    ax.barh(y, value, height=.7, color=colors[p], edgecolor='#444444', linewidth=.3)
                    ax.annotate(format_ms(value), (value, y), xytext=(3, 0), textcoords='offset points',
                                va='center', fontsize=7)
                    ticks.append(y)
                    labels.append(p.replace('smartsim_', '').replace('dl_', ''))
        if stacked:
            limit = max(totals) or 1
            ax.set_xlim(0, limit * 1.22)
            for ci, (cfg, total) in enumerate(zip(configs, totals)):
                left = 0
                for pi, p in enumerate(PHASES['provider_dl' if cfg == 'PHYDLL_CPP' else 'provider']):
                    value = float(data[(model, cfg, p)]['mean_ms_per_step'])
                    if value >= .14 * limit:
                        rgb = colors[p].lstrip('#')
                        light = sum(int(rgb[j:j+2], 16) for j in (0, 2, 4)) / 3
                        ax.text(left + value / 2, ci, format_ms(value), ha='center', va='center', fontsize=8,
                                color='white' if light < 130 else 'black')
                    elif value >= .025 * limit:
                        ax.annotate(format_ms(value), (left + value / 2, ci - .3), xytext=(0, 3 + 9 * (pi % 2)),
                                    textcoords='offset points', ha='center', va='bottom', fontsize=7,
                                    arrowprops={'arrowstyle': '-', 'lw': .4})
                    left += value
                ax.text(total + .01 * limit, ci, format_ms(total), va='center', fontsize=8)
            ax.set_ylim(-.8, len(configs) - .2)
        else:
            ax.margins(x=.22)
        ax.set_yticks(ticks, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Mean ms per steady step per scope rank')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_ms(value)))
        ax.grid(False)
        ax.grid(axis='x', alpha=.25, linewidth=.4)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Provider detail comparison: PhyDLL DL scope vs SmartSim solver-client scope\n'
                 'Separate instrumented scopes, NOT an end-to-end latency decomposition', fontsize=12)
    fig.legend(handles=[Patch(facecolor=colors[p], label=LABELS.get(p, p)) for p in phases],
               loc='lower center', ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, .20 if stacked else .12, 1, .93))
    name = f'provider_combined_{"stacked" if stacked else "grouped"}'
    save_figure(fig, output / f'{name}.png')
    if not PGF:
        fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    close_figure(fig)


def render(rows, output, view, stacked):
    if view == "application":
        panels = [(m, [(m, c, c) for c in ORDER]) for m in MODELS]
    elif view == "provider_dl":
        panels = [(m, [(m, "PHYDLL_CPP", "PhyDLL C++ DL")]) for m in MODELS]
    else:
        panels = [(c, [(m, c, TITLES[m]) for m in MODELS]) for c in ORDER[2:]]
    phases = PHASES[view]
    data = {(r['model'], r['config'], r['phase']): r for r in rows if r['view'] == view}
    colors = {p: COLOR_MAP.get(p.split(" / ")[0], "#BDBDBD") for p in phases}
    colors['dl_allocate'] = '#E0E0E0'
    fig, axes = plt.subplots(1 if len(panels) == 2 else 2, 2,
                             figsize=(12.5, 5.2 if len(panels) == 2 else 8.5), squeeze=False)
    for ax, (title, series) in zip(axes.flat, panels):
        ax.set_title(TITLES.get(title, title))
        if not all((m, c, phases[0]) in data for m, c, _ in series):
            ax.text(.5, .5, "Missing profile data", transform=ax.transAxes, ha="center")
            continue
        reps = sorted({int(data[(m, c, phases[0])]['rep_count']) for m, c, _ in series})
        if stacked:
            totals = [sum(float(data[(m, c, p)]['mean_ms_per_step']) for p in phases) for m, c, _ in series]
            limit = max(totals, default=0) or 1
            left = [0.0] * len(series)
            for phase_index, phase in enumerate(phases):
                values = [float(data[(m, c, phase)]['mean_ms_per_step']) for m, c, _ in series]
                ax.barh(range(len(series)), values, left=left, height=.6, color=colors[phase], edgecolor="white", linewidth=.4)
                for i, (v, l) in enumerate(zip(values, left)):
                    if v >= .15 * limit:
                        rgb = colors[phase].lstrip('#')
                        light = sum(int(rgb[j:j+2], 16) for j in (0, 2, 4)) / 3
                        ax.text(l + v / 2, i, format_ms(v), ha="center", va="center", fontsize=8,
                                color="white" if light < 130 else "black")
                    elif v >= .03 * limit:
                        ax.annotate(format_ms(v), (l + v / 2, i - .3),
                                    xytext=(0, 3 if view == "application" else 3 + 8 * (phase_index % 2)),
                                    textcoords="offset points", ha="center", va="bottom", fontsize=7,
                                    arrowprops={"arrowstyle": "-", "lw": .4})
                left = [l + v for l, v in zip(left, values)]
            for i, v in enumerate(totals):
                ax.text(v + .01 * limit, i, format_ms(v), va="center", fontsize=8)
            ax.set_xlim(0, limit * 1.2)
            ax.set_yticks(range(len(series)), [s[2] for s in series])
            ax.set_ylim(-.8, len(series) - .2)
        else:
            width = .75 / len(series)
            for j, (m, c, label) in enumerate(series):
                values = [float(data[(m, c, p)]['mean_ms_per_step']) for p in phases]
                y = [i + (j - (len(series) - 1) / 2) * width for i in range(len(phases))]
                bars = ax.barh(y, values, height=width, color=[colors[p] for p in phases],
                               edgecolor="#444444", linewidth=.4, hatch="//" if j else None, label=label)
                for bar, v in zip(bars, values):
                    ax.text(v, bar.get_y() + bar.get_height() / 2, ' ' + format_ms(v), va="center", fontsize=7)
            ax.set_yticks(range(len(phases)), [p.replace("smartsim_", "").replace("dl_", "") for p in phases], fontsize=8)
            ax.margins(x=.2)
            if len(series) > 1:
                ax.legend(handles=[Patch(facecolor="white", edgecolor="#444444", hatch="//" if j else None,
                                         label=s[2]) for j, s in enumerate(series)],
                          loc="lower right", fontsize=8, frameon=False)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean ms / steady step / scope rank (n={','.join(map(str, reps))} reps)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_ms(value)))
        ax.grid(False)
        ax.grid(axis="x", alpha=.25, linewidth=.4)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Patch(facecolor=colors[p], label=LABELS.get(p, p)) for p in phases]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    scope = "solver ranks" if view != "provider_dl" else "DL ranks only (not solver waiting)"
    fig.suptitle(f"Steady {'application phases' if view == 'application' else 'provider details'}: {scope}")
    fig.tight_layout(rect=(0, .17 if len(phases) > 6 else .13, 1, .95))
    name = f"{view}_{'stacked' if stacked else 'grouped'}"
    save_figure(fig, output / f"{name}.png")
    if not PGF:
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    close_figure(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-input", type=Path, default=Path(__file__).resolve().parent / "campaign_figures/steady_phases/phase_means.csv")
    parser.add_argument("--output-dir", type=Path, help="Default: input CSV directory")
    parser.add_argument("--pgf", action="store_true", help="Use existing thesis_style PGF/PDF/PNG hook (requires thesis environment)")
    parser.add_argument("--detailed", action="store_true", help="Also render supplemental raw-phase detail figures")
    args = parser.parse_args()
    with args.results_input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        r['config'] = ALIASES.get(r['config'], r['config'])
    output = args.output_dir or args.results_input.parent
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 11, "figure.dpi": 200})
    render_columns(rows, output, True, application=True)
    for stacked in (False, True):
        render_columns(rows, output, stacked)
    if args.detailed:
        for stacked in (False, True):
            render_combined(rows, output, stacked)
        for view in ("provider_dl", "provider"):
            for stacked in (False, True):
                render(rows, output, view, stacked)


if __name__ == "__main__":
    main()
