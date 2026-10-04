#!/usr/bin/env python3
"""Validate and plot new job_*/model/repN campaign latency and external memory.

Memory is sampled simultaneous process-set PSS/RSS, not cgroup charges or sums
of independent process/role peaks. Both models and all six configurations are
required. Lifecycle-race reads are excluded per metric; live failures are invalid.
"""
import argparse
import csv
import json
import math
import re
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

from extract_scorep_tables import ALIASES, ORDER, write_csv
from terrain_memory_sampler import LIFECYCLE_ERRORS, ROLES
from hh3_render import PGF, plt, save_figure, close_figure

MODELS = ('watercnn', 'giant')
DISPLAY = {'AIx': 'AIx', 'PHYDLL_CPP': 'PHYDLL_CPP', 'SS_DEFAULT': 'SS_DEFAULT',
           'TPQ1_I96': 'TPQ1_I96', 'TPQ12_I8': 'TPQ12_I8', 'TPQ96_I1': 'TPQ96_I1'}
COLORS = ['#0072B2', '#CC79A7', '#D55E00', '#009E73', '#56B4E9', '#117733']
STEP = re.compile(r'STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.eE+-]+)')
ACCOUNTING = ('External same-sweep process-set peaks over configuration lifetime; includes solver, DL, DB, '
              'and support/launchers; excludes sampler and batch shell. Sampled lower bounds, not atomic, '
              'not summed per-PID HWMs or separate role peaks. RSS double-counts shared pages; PSS apportions '
              'pages across all sharers. Neither is cgroup memory/cache/kernel accounting. '
              'Total peak sweeps must have exact expected direct solver/DL/DB counts and complete metric reads. '
              'Lifecycle-race incomplete reads are excluded per role/metric; live read failures are invalid. '
              'Raw lifetime peaks outside the covered census are rejected, never silently lowered. '
              'Sampling overruns are resolution warnings, not invalid measurements. Single-node only.')


def number(value, name, positive=False):
    if value is None or isinstance(value, bool):
        raise ValueError(f'Missing/invalid {name}: {value}')
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f'Invalid {name}: {value}')
    return result


def validate_memory(directory, config, ranks):
    """Audit new-schema peaks, direct census, lifecycle exclusions, and cadence."""
    source = directory / 'memory_summary.json'
    summary = json.loads(source.read_text())
    expected = {'solver': ranks, 'dl': ranks if config == 'PHYDLL_CPP' else 0,
                'db': 0 if config in ('AIx', 'PHYDLL_CPP') else 1}
    if summary['coverage_ok'] is not True or summary['expected_processes'] != expected:
        raise ValueError(f'{source}: invalid coverage or expected role census')
    samples = number(summary['samples'], 'samples', True)
    coverage = number(summary['simultaneous_coverage_samples'], 'coverage samples', True)
    interval = number(summary['interval_s'], 'interval', True)
    sweep = number(summary['max_sweep_s'], 'maximum sweep')
    if samples != int(samples) or coverage != int(coverage) or not 0 < coverage <= samples or not .2 <= interval <= 1:
        raise ValueError(f'{source}: invalid sampling coverage/cadence')
    if summary['live_read_errors'] != 0 or summary['cleanup_sigkill_pids']:
        raise ValueError(f'{source}: live read errors or forced cleanup; investigate measurement')
    lifecycle = number(summary['lifecycle_read_races'], 'lifecycle races')
    errors = summary['read_error_counts']
    if (lifecycle != int(lifecycle) or any(k not in LIFECYCLE_ERRORS for k in errors) or
            any(number(v, 'error count') != int(v) for v in errors.values()) or
            sum(errors.values()) != lifecycle or summary['process_read_errors'] != lifecycle):
        raise ValueError(f'{source}: inconsistent read-error categories/counts')
    if not summary['token'] or not summary['host']:
        raise ValueError(f'{source}: missing process-set attribution')
    by_sample = defaultdict(dict)
    with (directory / 'memory_samples.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            sample, role = int(row['sample']), row['role']
            if role not in (*ROLES, 'total') or role in by_sample[sample]:
                raise ValueError(f'{directory}: duplicate/unknown sample role')
            count = number(row['processes'], 'process count')
            if count != int(count):
                raise ValueError(f'{directory}: invalid process count')
            for metric, read in (('rss_bytes', 'rss_read'), ('pss_bytes', 'pss_read')):
                reads = number(row[read], read)
                if reads != int(reads) or reads > count:
                    raise ValueError(f'{directory}: invalid read count')
                if reads == count:
                    number(row[metric], metric)
                    if count == 0 and float(row[metric]) != 0:
                        raise ValueError(f'{directory}: nonzero memory for empty process set')
                elif row[metric] != '':
                    raise ValueError(f'{directory}: incomplete metric must be blank, not a partial sum')
            if row['pss_bytes'] and row['rss_bytes'] and float(row['pss_bytes']) > float(row['rss_bytes']):
                raise ValueError(f'{directory}: PSS exceeds RSS')
            number(row['sweep_s'], 'sweep')
            for r in expected:
                direct = number(row['direct_' + r], 'direct census')
                if direct != int(direct):
                    raise ValueError(f'{directory}: invalid direct census')
            if row['sample_interval_s']:
                number(row['sample_interval_s'], 'actual interval', True)
            by_sample[sample][role] = row
    if set(by_sample) != set(range(int(samples))):
        raise ValueError(f'{directory}: truncated/nonconsecutive sample stream')
    if max(float(r['sweep_s']) for roles in by_sample.values() for r in roles.values()) != sweep:
        raise ValueError(f'{directory}: JSON/CSV maximum sweep mismatch')
    for roles in by_sample.values():
        if set(roles) != set((*ROLES, 'total')):
            raise ValueError(f'{directory}: missing sample roles')
        for key in ('direct_solver', 'direct_dl', 'direct_db', 'sample_interval_s', 'sweep_s'):
            if len({r[key] for r in roles.values()}) != 1:
                raise ValueError(f'{directory}: inconsistent same-sweep census/cadence metadata')
        for r in expected:
            if int(roles['total']['direct_' + r]) > int(roles[r]['processes']):
                raise ValueError(f'{directory}: direct census exceeds attributed process count')
        for key in ('processes', 'rss_read', 'pss_read'):
            if float(roles['total'][key]) != sum(float(roles[r][key]) for r in ROLES):
                raise ValueError(f'{directory}: same-sweep {key} sum mismatch')
        for metric in ('rss_bytes', 'pss_bytes'):
            if all(roles[r][metric] != '' for r in ROLES):
                if roles['total'][metric] == '' or float(roles['total'][metric]) != sum(float(roles[r][metric]) for r in ROLES):
                    raise ValueError(f'{directory}: same-sweep {metric} sum mismatch')
            elif roles['total'][metric] != '':
                raise ValueError(f'{directory}: incomplete role readings require blank total')
    for read in ('rss_read', 'pss_read'):
        missing = sum(int(s[r]['processes']) - int(s[r][read]) for s in by_sample.values() for r in ROLES)
        if missing != lifecycle:
            raise ValueError(f'{directory}: incomplete reads do not match lifecycle races')
    direct_max = {r: max(int(s['total']['direct_' + r]) for s in by_sample.values()) for r in expected}
    covered = [s for s in by_sample.values() if all(int(s['total']['direct_' + r]) == n for r, n in expected.items())]
    census_coverage = sum(all(int(s['total']['direct_' + r]) >= n for r, n in expected.items()) for s in by_sample.values())
    if summary['max_direct_processes'] != direct_max or census_coverage != coverage:
        raise ValueError(f'{directory}: JSON/CSV direct census coverage mismatch')
    peaks = summary['sampled_simultaneous_peaks']
    for role in (*ROLES, 'total'):
        if peaks[role]['max_processes'] != max(int(s[role]['processes']) for s in by_sample.values()):
            raise ValueError(f'{directory}: JSON/CSV process count mismatch')
        for metric in ('rss_bytes', 'pss_bytes'):
            value = peaks[role][metric]
            readings = [float(s[role][metric]) for s in by_sample.values()
                        if s[role][metric] != '' and int(s[role]['processes']) > 0]
            actual = max(readings) if readings else None
            if value is None and actual is None and role != 'total':
                continue
            if number(value, metric, role == 'total') != actual:
                raise ValueError(f'{directory}: JSON/CSV {role} {metric} peak mismatch')
            if role == 'total' and any(s['total'][metric] != '' and float(s['total'][metric]) == actual and
                                      s not in covered for s in by_sample.values()):
                raise ValueError(f'{directory}: raw lifetime {metric} peak outside exact expected direct census; refusing to lower peak')
    actual_intervals = [float(by_sample[i]['total']['sample_interval_s']) for i in range(1, int(samples))
                        if by_sample[i]['total']['sample_interval_s']]
    if by_sample[0]['total']['sample_interval_s'] != '' or len(actual_intervals) != int(samples) - 1:
        raise ValueError(f'{directory}: missing/invalid actual sampling intervals')
    ordered = sorted(actual_intervals)
    stats = {'count': len(ordered), 'min': min(ordered) if ordered else None,
             'mean': statistics.mean(ordered) if ordered else None, 'median': statistics.median(ordered) if ordered else None,
             'p95': ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None, 'max': max(ordered) if ordered else None}
    for key, value in stats.items():
        recorded = summary['actual_intervals_s'][key]
        if (value is None and recorded is not None) or (value is not None and
                not math.isclose(number(recorded, key), value, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f'{directory}: JSON/CSV actual interval statistics mismatch')
    cycle = number(summary['max_cycle_s'], 'maximum cycle')
    overruns = number(summary['overrun_samples'], 'overrun samples')
    if cycle < sweep or overruns != int(overruns) or overruns > samples:
        raise ValueError(f'{directory}: invalid cycle metadata')
    quality = ''
    if overruns or cycle > interval or (ordered and max(ordered) > 2 * interval):
        quality = f'Sampling resolution warning: {int(overruns)}/{int(samples)} overruns; max cycle={cycle:.6g}s; actual interval p95={stats["p95"]}s'
        warnings.warn(f'{source}: {quality}', UserWarning)
    return {'pss_peak_bytes': peaks['total']['pss_bytes'], 'rss_peak_bytes': peaks['total']['rss_bytes'],
             'samples': int(samples), 'coverage_samples': int(coverage), 'interval_s': interval,
             'exact_direct_coverage_samples': len(covered), 'lifecycle_read_races': int(lifecycle),
             'live_read_errors': summary['live_read_errors'], 'read_error_counts': json.dumps(errors, sort_keys=True),
             'excluded_total_pss_samples': sum(s['total']['pss_bytes'] == '' for s in by_sample.values()),
             'excluded_total_rss_samples': sum(s['total']['rss_bytes'] == '' for s in by_sample.values()),
             'actual_intervals_s': json.dumps(stats, sort_keys=True), 'overrun_samples': int(overruns),
             'max_cycle_s': cycle, 'quality_warning': quality,
            'max_sweep_s': sweep, 'host': summary['host'], 'token': summary['token'],
            'memory_json': str(source.resolve()), 'accounting': ACCOUNTING}


def collect(roots, results_inputs, logs_root, ranks, steady_steps):
    manifests = set(results_inputs or [])
    for root in roots:
        manifests.update(root.glob('*/rep[0-9]*/results.csv'))
        manifests.update(root.glob('job_*/*/rep[0-9]*/results.csv'))
    rows, seen, tokens = [], set(), set()
    for manifest in sorted(manifests):
        with manifest.open(newline='') as stream:
            results = list(csv.DictReader(stream))
        for r in results:
            if r['config'] not in ALIASES:
                raise ValueError(f'{manifest}: unknown config {r["config"]}')
            model, cfg, rep = r['model'], ALIASES[r['config']], r['rep']
            if model not in MODELS or r['status'] != 'SUCCESS':
                raise ValueError(f'{manifest}: failed/unknown run: {r}')
            if manifest.parent.name != f'rep{rep}' or manifest.parent.parent.name != model:
                raise ValueError(f'{manifest}: model/rep metadata disagrees with path')
            key = (model, cfg, rep)
            if key in seen:
                raise ValueError(f'Duplicate run: {key}')
            seen.add(key)
            memory = validate_memory(manifest.parent / r['config'], cfg, ranks)
            if memory['token'] in tokens:
                raise ValueError(f'Reused memory attribution token: {key}')
            tokens.add(memory['token'])
            log_dir = logs_root or manifest.parents[2] / 'logs'
            logs = sorted(log_dir.glob(f'terrain_nobatch_{model}_r{rep}_{r["config"]}_*.log'))
            if len(logs) != 1:
                raise ValueError(f'{key}: expected one latency log in {log_dir}, found {len(logs)}')
            times = {}
            for line in logs[0].read_text().splitlines():
                match = STEP.search(line)
                if not match:
                    continue
                step = int(match[1])
                if step == 2:
                    continue
                if step in times:
                    raise ValueError(f'{logs[0]}: duplicate steady step {step}')
                times[step] = number(match[2], 'step_ms', True)
            if set(times) != set(range(4, 4 + 2 * steady_steps, 2)):
                raise ValueError(f'{logs[0]}: incomplete/unexpected steady steps {sorted(times)}')
            rows.append({'model': model, 'config': cfg, 'rep': rep, 'solver_ranks': ranks,
                         'steady_steps': len(times), 'steady_mean_ms': statistics.mean(times.values()),
                          'steady_min_ms': min(times.values()), 'steady_max_ms': max(times.values()),
                          'steady_samples_ms': json.dumps(times, sort_keys=True),
                         **memory, 'latency_log': str(logs[0].resolve()), 'results_csv': str(manifest.resolve())})
    if not rows:
        raise ValueError('No completed new-layout results found')
    for model in MODELS:
        reps = [{r['rep'] for r in rows if r['model'] == model and r['config'] == cfg} for cfg in ORDER]
        if not reps[0] or any(s != reps[0] for s in reps):
            raise ValueError(f'{model}: require matching replicate coverage for all six configs: {reps}')
    return rows


def render(rows, output, memory):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model in zip(axes, MODELS):
        for ci, cfg in enumerate(ORDER):
            group = [r for r in rows if r['model'] == model and r['config'] == cfg]
            metrics = ('pss_peak_bytes', 'rss_peak_bytes') if memory else ('steady_mean_ms',)
            for mi, metric in enumerate(metrics):
                values = [float(r[metric]) / (2**30 if memory else 1000) for r in group]
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0
                x = ci + (mi - .5) * .36 if memory else ci
                ax.bar(x, mean, width=.34 if memory else .65, yerr=std, capsize=3,
                       color=('#0072B2' if mi == 0 else '#E69F00') if memory else COLORS[ci],
                       hatch='//' if memory and mi else None, edgecolor='#444444', linewidth=.5,
                       label=('Sampled PSS peak' if mi == 0 else 'Sampled RSS peak') if memory and ci == 0 else None)
                ax.annotate(f'{mean:.2f}', (x, mean + std), xytext=(0, 4), textcoords='offset points',
                            ha='center', fontsize=8)
        reps = len([r for r in rows if r['model'] == model and r['config'] == ORDER[0]])
        ax.set_title(f'{"WaterCNN" if model == "watercnn" else "Giant MLP"} (n={reps} reps)')
        ax.set_xticks(range(len(ORDER)), [DISPLAY[c] for c in ORDER], rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Mean external sampled process-set peak (GiB)' if memory else 'Mean steady ML-step latency (s)')
        ax.grid(False)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=.2)
        if memory:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle('New CPU campaign: external sampled PSS/RSS memory' if memory else 'Supplemental: replicate-mean steady latency, warmup excluded')
    caption = ('Same-sweep total peaks; solver + DL + DB + support. Not cgroup charges or summed per-process HWMs.\n'
               'RSS double-counts shared pages; PSS apportions sharing. Sampled lower bounds, non-atomic sweeps.') if memory else \
              'Each replicate contributes its mean of validated steady STEP_TIMING samples; solver end-to-end latency, not provider-only time.'
    fig.text(.5, .015, caption + '\nBars: arithmetic mean across replicates; error bars: replicate standard deviation.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .94))
    name = 'external_memory_peaks' if memory else 'steady_latency'
    save_figure(fig, output / f'{name}.png')
    if not PGF:
        fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    close_figure(fig)


def render_latency_boxplots(rows, output, scale='log'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model in zip(axes, MODELS):
        groups = [[float(ms) / 1000 for r in rows if r['model'] == model and r['config'] == cfg
                   for ms in json.loads(r['steady_samples_ms']).values()] for cfg in ORDER]
        boxes = ax.boxplot(groups, patch_artist=True, widths=.6, whis=1.5, showfliers=True,
                           medianprops={'color': 'black', 'linewidth': 1.3},
                           flierprops={'marker': 'd', 'markersize': 3, 'alpha': .5})
        for patch, color in zip(boxes['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(.65)
        for i, values in enumerate(groups):
            ax.annotate(f'median {statistics.median(values):.3f}\nn={len(values)} steps',
                        (i + 1, max(values)), xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=7)
        ax.set_title('WaterCNN' if model == 'watercnn' else 'Giant MLP')
        ax.set_xticks(range(1, len(ORDER) + 1), [DISPLAY[c] for c in ORDER], rotation=25, ha='right', fontsize=8)
        ax.set_yscale(scale)
        ax.set_ylabel(f'Steady ML-step latency (s, {scale} scale)')
        ax.grid(False)
        ax.grid(axis='y', which='both', alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=.25)
    fig.suptitle('New CPU campaign: all steady-step latency samples, warmup excluded')
    fig.text(.5, .02, 'Every validated per-step STEP_TIMING sample across replicates; not replicate means.\n'
             'Boxes: median and interquartile range; whiskers: 1.5 IQR; diamonds: outliers. Steps within a run are correlated.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .94))
    save_figure(fig, output / 'steady_latency_boxplots.png')
    if not PGF:
        fig.savefig(output / 'steady_latency_boxplots.pdf', bbox_inches='tight')
    close_figure(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, action='append', default=[])
    parser.add_argument('--results-input', type=Path, action='append', default=[])
    parser.add_argument('--logs-root', type=Path, help='Override new campaign log directory')
    parser.add_argument('--solver-ranks', type=int, default=96)
    parser.add_argument('--steady-steps', type=int, default=10)
    parser.add_argument('--latency-scale', choices=('linear', 'log'), default='log', help='Boxplot axis scale')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'campaign_figures/memory_latency')
    parser.add_argument('--pgf', action='store_true', help='Use existing thesis-style hook')
    args = parser.parse_args()
    if args.solver_ranks < 1 or args.steady_steps < 1:
        parser.error('rank and step counts must be positive')
    try:
        rows = collect(args.campaign_root or ([] if args.results_input else [Path(__file__).resolve().parent / 'results/terrain_nobatch']),
                       args.results_input, args.logs_root, args.solver_ranks, args.steady_steps)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'memory_latency_replicates.csv', rows)
    steps = [{'model': r['model'], 'config': r['config'], 'rep': r['rep'], 'step': int(step),
              'steady_step_ms': ms, 'latency_log': r['latency_log']}
             for r in rows for step, ms in json.loads(r['steady_samples_ms']).items()]
    write_csv(args.output_dir / 'steady_latency_steps.csv', steps)
    (args.output_dir / 'memory_latency_accounting.md').write_text('# External Memory / Latency Accounting\n\n' + ACCOUNTING +
        '\n\nValidation rejects failed runs, duplicate runs/tokens, incomplete six-config/two-model coverage, role census '
        'mismatches, live read failures, genuine SIGKILL cleanup, unexplained incomplete reads, JSON/CSV peak mismatches, '
        'and raw total lifetime peaks outside exact expected direct census. Lifecycle-race incomplete readings do not '
        'enter that role/metric peak. Overruns and long cycles are quality warnings, not invalid data. Latency boxplots '
        'use all expected steady steps, not replicate means or the results.csv step-4-only value. Mean plots are supplemental.\n\n'
        '## Sampling Quality\n\n| Model | Config | Rep | Lifecycle races | Excluded total RSS/PSS samples | '
        'Overruns | Max cycle s | Actual intervals s | Warning |\n|---|---|---|---:|---|---:|---:|---|---|\n' +
        ''.join(f"| {r['model']} | {r['config']} | {r['rep']} | {r['lifecycle_read_races']} | "
                f"{r['excluded_total_rss_samples']}/{r['excluded_total_pss_samples']} | {r['overrun_samples']} | "
                f"{r['max_cycle_s']} | `{r['actual_intervals_s']}` | {r['quality_warning'] or 'none'} |\n" for r in rows))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
    render(rows, args.output_dir, True)
    render_latency_boxplots(rows, args.output_dir, args.latency_scale)
    render(rows, args.output_dir, False)
    print(f'Validated {len(rows)} config replicates; wrote figures and accounting to {args.output_dir}')


if __name__ == '__main__':
    main()
