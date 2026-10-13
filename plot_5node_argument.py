#!/usr/bin/env python3
"""Two thesis figures: traffic validation and steady runtime attribution.

Run from any directory with python3 mini_app/plot_5node_argument.py.
The previous six-figure set is left untouched.
"""
import argparse
import csv
import sys
from pathlib import Path
from scipy.stats import t

from plot_5node_traffic import (
    MINI_APP, ML_STEPS, RANKS, SHARDS, WIDTH, HEIGHT, STEPS,
    compute_analytical_predictions, parse_manifest, parse_net_usage,
    parse_step_timings, predicted_directions, ts, plt, np,
)

sys.path.insert(0, str(MINI_APP.parent / "CPP-ML-Interface" / "scripts"))
with plt.rc_context():
    from analyze_cmi_scorep_profiles import parse_cubex_file

BLUE = "#5799c6"
RED = "#df6565"
GRAY = "#bbbbbb"


def profile_times(path):
    """Normalize inclusive steady regions by step visits, not operation calls."""
    regions = {'put_ms': 'smartsim_put_tensor', 'model_ms': 'smartsim_run_model',
               'unpack_ms': 'smartsim_unpack_tensor'}
    samples = {key: [] for key in ('profile_ms', *regions)}
    counts = []
    for nodes in parse_cubex_file(path):
        roots = [n for n in nodes.values()
                 if n.name == "solver_step_ml_steady" and n.visits]
        if not roots:
            continue
        count = sum(n.visits for n in roots)
        root_ids = {n.id for n in roots}
        seconds = {key: 0.0 for key in regions}
        visits = {key: 0 for key in regions}
        for node in nodes.values():
            key = next((key for key, name in regions.items() if node.name == name), None)
            if key is None:
                continue
            ancestor = node.parent_id
            while ancestor is not None and ancestor not in root_ids:
                ancestor = nodes[ancestor].parent_id
            if ancestor in root_ids:
                seconds[key] += node.incl_time
                visits[key] += node.visits
        for key in regions:
            expected = 2 * count if key == 'put_ms' else count
            if visits[key] != expected:
                raise ValueError(f"{key}/step visit mismatch: {path}")
            samples[key].append(seconds[key] * 1000 / count)
        samples['profile_ms'].append(sum(n.incl_time for n in roots) * 1000 / count)
        counts.append(count)
    if len(counts) != RANKS or set(counts) != {ML_STEPS - 1}:
        raise ValueError(f"Unexpected solver ranks or steady visits: {path}")
    result = {key: np.mean(sample) for key, sample in samples.items()}
    if not 0 < sum(result[key] for key in regions) <= result['profile_ms']:
        raise ValueError(f"Invalid additive timing scopes: {path}")
    return result


def load(manifest):
    rows = []
    for entry in parse_manifest(manifest):
        for condition in ("standalone", "clustered"):
            log = MINI_APP / f"logs/solver_{condition}_{entry['job_id']}.log"
            net = parse_net_usage(log)
            timings = parse_step_timings(log)
            steady = [ms for step, (ms, _, _) in timings.items() if step > 2]
            if net is None or len(steady) != ML_STEPS - 1:
                raise ValueError(f"Incomplete replicate: {log}")
            profile = MINI_APP / f"results/5node_traffic_{entry['job_id']}" / condition / "scorep/profile.cubex"
            times = profile_times(profile)
            rows.append(dict(rep=entry['rep'], condition=condition,
                             rx=net['rx_mb'] / ML_STEPS,
                             tx=net['tx_mb'] / ML_STEPS,
                             wall_ms=np.mean(steady), **times))
    if not rows:
        raise ValueError("No replicates found")
    return rows


def values(rows, condition, key):
    return np.array([r[key] for r in rows if r['condition'] == condition])


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(axis="y", linestyle=":", color="0.83")
    ax.tick_params(axis="x", labelsize=9)


def ci95(sample):
    """Two-sided Student-t confidence interval half-width for the run mean."""
    if len(sample) < 2:
        raise ValueError('At least two independent runs are required for a CI')
    return t.ppf(0.975, len(sample) - 1) * np.std(sample, ddof=1) / np.sqrt(len(sample))


def traffic_figure(pred, rows, outdir):
    directions = predicted_directions(pred)
    display = {}
    for condition in ('standalone', 'clustered'):
        for source in ('predicted', 'measured'):
            parts = np.array([directions[condition][key] if source == 'predicted'
                              else np.mean(values(rows, condition, key))
                              for key in ('tx', 'rx')])
            total = int(np.rint(parts.sum()))
            rounded = np.floor(parts).astype(int)
            # Largest-remainder rounding preserves the displayed total.
            for index in np.argsort(-(parts - rounded))[:total - rounded.sum()]:
                rounded[index] += 1
            display[condition, source] = dict(tx=rounded[0], rx=rounded[1], total=total)

    def draw(ax, keys):
        predicted, measured, errors, labels = [], [], [], []
        annotations = {'predicted': [], 'measured': []}
        for condition, label in (("standalone", "Standalone"), ("clustered", "Clustered")):
            for key in keys:
                sample = (values(rows, condition, 'rx') + values(rows, condition, 'tx')
                          if key == 'total' else values(rows, condition, key))
                predicted.append(sum(directions[condition].values()) if key == 'total'
                                 else directions[condition][key])
                measured.append(np.mean(sample))
                errors.append(ci95(sample))
                labels.append(label if key == 'total' else key.upper())
                for source in annotations:
                    annotations[source].append(str(display[condition, source][key]))
        x = np.arange(len(labels))
        p = ax.bar(x - 0.19, predicted, width=0.35, color=GRAY,
                   edgecolor='black', linewidth=0.6, label='Predicted payload')
        m = ax.bar(x + 0.19, measured, width=0.35, color=BLUE,
                   edgecolor='black', linewidth=0.6, yerr=errors, capsize=2,
                   label='Measured traffic')
        for bars, source, padding in ((p, 'predicted', 3), (m, 'measured', 9)):
            ax.bar_label(bars, labels=annotations[source], padding=padding, fontsize=8)
        ax.set_xticks(x, labels)
        if keys != ('total',):
            for xpos, label in ((0.5, 'Standalone'), (2.5, 'Clustered')):
                ax.text(xpos, -0.29, label, transform=ax.get_xaxis_transform(),
                        ha='center', fontsize=9)
        ax.set_ylim(0, 2400)
        ax.set_yticks([0, 1000, 2000])
        axes_style(ax)
        return [p, m]

    height = 2.05
    top = 1 - 0.32 / height
    fig, axes = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH_IN, height),
                             gridspec_kw={'width_ratios': [1, 1.35]})
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.48 / height, top=top, wspace=0.36)
    for ax, keys in zip(axes, [('total',), ('tx', 'rx')]):
        handles = draw(ax, keys)
    axes[0].set_ylabel('Traffic (MiB/step)', fontsize=9)
    fig.legend(handles, ['Predicted payload', 'Measured traffic'], loc='lower center',
               bbox_to_anchor=(0.55, top + 0.07 / height), ncol=2,
               frameon=False, fontsize=9, borderaxespad=0)
    ts.save(fig, outdir / 'traffic_validation')
    plt.close(fig)

    # Half-textwidth assets keep document-sized fonts when used as subfigures.
    for name, keys in [('traffic_total', ('total',)), ('traffic_directions', ('tx', 'rx'))]:
        height = 2.10
        top = 1 - 0.50 / height
        fig, ax = plt.subplots(figsize=(0.49 * ts.TEXTWIDTH_IN, height))
        fig.subplots_adjust(left=0.24, right=0.985, bottom=0.48 / height, top=top)
        draw(ax, keys)
        ax.set_ylabel('Traffic (MiB/step)', fontsize=9)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1 + 0.07 / (height - 0.98)),
                  fontsize=8, frameon=False, borderaxespad=0)
        ts.save(fig, outdir / name)
        plt.close(fig)


def runtime_figure(rows, outdir):
    height = 1.88
    top = 1 - 0.32 / height
    fig, ax = plt.subplots(figsize=(ts.TEXTWIDTH_IN, height))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.28 / height, top=top)
    keys = ['profile_ms', 'put_ms', 'model_ms', 'unpack_ms']
    x = np.arange(len(keys))
    for condition, color, offset, label in [('standalone', BLUE, -0.19, 'Standalone'),
                                            ('clustered', RED, 0.19, 'Clustered')]:
        samples = [values(rows, condition, key) for key in keys]
        means = [np.mean(s) for s in samples]
        errors = [ci95(s) for s in samples]
        ax.bar(x + offset, means, width=0.35, color=color, edgecolor='black',
               linewidth=0.6, yerr=errors, capsize=3, label=label)
        for xpos, mean, error in zip(x + offset, means, errors):
            ax.annotate(f'{mean:.1f}', (xpos, mean + error), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks(x, ['Total', '$\\mathtt{put\\_tensor}$', '$\\mathtt{run\\_model}$',
                     '$\\mathtt{unpack\\_tensor}$'])
    ax.set_ylabel('Time per step (ms)', fontsize=9)
    ax.set_ylim(0, 390)
    ax.set_yticks([0, 100, 200, 300])
    axes_style(ax)
    fig.legend(loc='lower center', bbox_to_anchor=(0.55, top + 0.07 / height), ncol=2,
               frameon=False, fontsize=9, borderaxespad=0)
    ts.save(fig, outdir / 'runtime_attribution')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=MINI_APP / '5node_traffic_manifest.txt')
    ap.add_argument('--outdir', type=Path, default=MINI_APP / 'campaign_figures/5node_traffic_argument')
    args = ap.parse_args()
    rows = load(args.manifest)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / 'figure_data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pred = compute_analytical_predictions(RANKS, SHARDS, WIDTH, HEIGHT, STEPS)
    traffic_figure(pred, rows, args.outdir)
    runtime_figure(rows, args.outdir)
    print(f"Wrote figures and underlying data to {args.outdir}")


if __name__ == '__main__':
    main()
