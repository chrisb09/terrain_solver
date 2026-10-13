#!/usr/bin/env python3
"""Explicit production-only terrain CPU regeneration, with no solver execution."""
import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from extract_scorep_tables import ALIASES, ORDER as BASE_ORDER, extract, parse_cubex_file
from terrain_cpu_final_analysis import HERE, read_csv, write_csv, source, resolve_path, validate_policy, THREADS, TOTAL_BS, RUNTIME
from terrain_cpu_diagnostics import timings
from plot_terrain_nobatch import validate_memory, COLORS, ACCOUNTING, STEP
from validate_terrain_phydll_profile import validate as validate_phydll
from hh3_render import plt, save_figure, close_figure
from plot_scorep_phases import SIMPLIFIED, SIMPLE_COLORS

ORDER = ['AIx', 'PHYDLL_FALSE', 'PHYDLL_TRUE', *BASE_ORDER[2:]]
LABELS = ['AIx', 'PhyDLL\nblocking', 'PhyDLL\nreadiness', 'SmartSim\ndefault',
          'SmartSim\nTPQ=1\nINTRA=96', 'SmartSim\nTPQ=12\nINTRA=8', 'SmartSim\nTPQ=96\nINTRA=1']
PALETTE = [COLORS[0], COLORS[1], '#8064A2', *COLORS[2:]]
MODELS = ('watercnn', 'giant')
ROOT = HERE / 'results/terrain_cpu_final'
OUTPUT = HERE / 'campaign_figures/terrain_cpu_complete_20261013'


def coverage(rows):
    keys = [(r['model'], r['config'], str(r['rep'])) for r in rows]
    expected = {(m, c, str(rep)) for m in MODELS for c in ORDER for rep in range(1, 6)}
    if len(keys) != 70 or set(keys) != expected:
        raise ValueError(f'Exact 70-run coverage failed: missing={sorted(expected-set(keys))}; duplicates/unexpected={len(keys)-len(set(keys))}')


def selectors(kind):
    other = set(BASE_ORDER) - {'PHYDLL_CPP'}
    if kind == 'timing':
        selected = [('terrain_missing_20261012_other', other, None, '4952588')]
        prefix = 'terrain_missing_20261012'
    else:
        selected = [('production_4644312', other if kind == 'performance' else {'TPQ1_I96', 'TPQ12_I8'}, None, None)]
        prefix = 'phydll_master_20261012'
    selected += [(prefix + '_readiness_' + flag, {'PHYDLL_CPP'}, flag,
                  '4952363' if kind == 'timing' else None) for flag in ('false', 'true')]
    if kind == 'memory':
        selected += [('terrain_missing_20261012_other', {'AIx', 'SS_DEFAULT', 'TPQ96_I1'}, None, '4952361'),
                     ('terrain_missing_20261012_memory_task0_clean1_other', {'AIx', 'SS_DEFAULT', 'TPQ96_I1'}, None, '4952823')]
    return selected


def collect(kind):
    rows, tokens = [], set()
    for campaign, configs, readiness, array in selectors(kind):
        manifests = sorted((ROOT / campaign / kind).glob('job_*/*/rep[0-9]*/results.csv'))
        if not manifests:
            raise ValueError(f'Missing production selector {campaign}/{kind}')
        for manifest in manifests:
            if (manifest.parent / 'SUPERSEDED.json').exists():
                if kind == 'memory' and campaign == 'terrain_missing_20261012_other' and manifest.parent.parent.name == 'watercnn' and manifest.parent.name == 'rep1':
                    continue
                raise ValueError(f'Unexpected supersession: {manifest}')
            for original in read_csv(manifest):
                base = ALIASES.get(original['config'])
                if base not in configs:
                    continue
                r = dict(original)
                directory = manifest.parent / r['config']
                if r['status'] != 'SUCCESS' or r['mode'] != kind or manifest.parent.name != 'rep' + r['rep'] or manifest.parent.parent.name != r['model']:
                    raise ValueError(f'Failed/mismatched production run {manifest}')
                expected = dict(solver_ranks='96', dl_ranks='96' if readiness else '0', rank_grid='12x8',
                                timing_sync='1', total_bs=str(TOTAL_BS[r['model']]),
                                monitoring=str(int(kind == 'memory')), memory_sampler=str(int(kind == 'memory')),
                                steps='6' if kind == 'memory' else '22', steady_steps='2' if kind == 'memory' else '10',
                                phydll_logging='0', phydll_io_log='0', phydll_io_log_barriers='0', internal_memory='0',
                                internal_mem_verbose='0', internal_mem_collective='0', telemetry='0', db_min_batch='0', db_timeout='0')
                tpq, intra, omit = THREADS[base]
                expected.update(tpq=str(tpq), intra=str(intra), omit_threads=str(omit), omit_set_cpus=str(omit))
                if any(r.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'Runtime metadata policy mismatch {manifest}: {expected}')
                validate_policy(r, base)
                env_path = resolve_path(manifest, r['run_environment'], r['config'])
                env = dict(line.split('=', 1) for line in env_path.read_text().splitlines() if re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', line))
                digest = source(env_path)['sha256']
                attestation = resolve_path(manifest, r['environment_sha256'], r['config']).read_text().split()
                if digest != r['run_environment_sha256'] or attestation[0] != digest or Path(attestation[1]).resolve() != env_path:
                    raise ValueError(f'Environment attestation mismatch {env_path}')
                if array and env.get('SLURM_ARRAY_JOB_ID') != array:
                    raise ValueError(f'Wrong production array {env_path}: {env.get("SLURM_ARRAY_JOB_ID")} != {array}')
                command = shlex.split((directory / 'command.sh').read_text())
                runtime = {k: v for k, v in RUNTIME.items() if not k.startswith('SCOREP_')}
                runtime.update(MODE=kind, MODEL=r['model'], REP=r['rep'], STEPS=r['steps'])
                if any(env.get(k) != v for k, v in runtime.items()):
                    raise ValueError(f'Effective launch exports disagree with policy {env_path}')
                rank_positions = [i for i, word in enumerate(command) if word == '-n']
                if len(rank_positions) != (2 if readiness else 1) or any(command[i+1] != '96' for i in rank_positions):
                    raise ValueError(f'Actual MPI rank launch mismatch {directory}')
                pinned = base in ('AIx', 'PHYDLL_CPP')
                if r['binding'] != ('pinned' if pinned else 'unbound') or any((Path(command[i+2]).name == 'pin_wrapper.sh') != pinned for i in rank_positions):
                    raise ValueError(f'Binding mismatch {directory}')
                if base.startswith(('SS', 'TPQ')):
                    ss = dict(SMARTSIM_THREADS_PER_QUEUE=str(tpq), SMARTSIM_INTRA_OP_THREADS=str(intra), SMARTSIM_INTER_OP_THREADS='1',
                              SMARTSIM_OMIT_THREADS=str(omit), SMARTSIM_OMIT_SET_CPUS=str(omit))
                    if any(env.get(k) != v for k, v in ss.items()) or command[command.index('--ml-db-batch-size')+1] != r['db_batch']:
                        raise ValueError(f'SmartSim effective thread/batch mismatch {directory}')
                for flag, value in {'--device': 'CPU', '--steps': expected['steps'], '--rank-grid-x': '12', '--rank-grid-z': '8'}.items():
                    if command.count(flag) != 1 or command[command.index(flag)+1] != value:
                        raise ValueError(f'Launch policy {flag}: {directory}')
                if readiness:
                    config = Path(command[command.index('--cpp-ml-config')+1])
                    if r.get('solver_readiness_wait') != readiness or config.name != f'config_terrain_phydll_{readiness}.toml' or not re.search(r'solver_readiness_wait\s*=\s*' + readiness, config.read_text()):
                        raise ValueError(f'Readiness collision {directory}')
                provenance_path = resolve_path(manifest, r['provenance'], r['config'])
                provenance = json.loads(provenance_path.read_text())
                config_path = Path(command[command.index('--cpp-ml-config')+1]).resolve()
                if provenance['inputs'].get(str(config_path)) != source(config_path)['sha256']:
                    raise ValueError(f'Runtime configuration hash mismatch {directory}')
                log = resolve_path(manifest, r['log'], r['config'])
                samples = timings(log, int(r['steps']))
                warmup = [STEP.search(line) for line in log.read_text().splitlines()]
                if sum(bool(m and m[1] == '2') for m in warmup) != 1:
                    raise ValueError(f'Missing/duplicate warmup {log}')
                r.update(config='PHYDLL_' + readiness.upper() if readiness else base, source_config=original['config'],
                         results_csv=str(manifest), campaign=campaign, log=str(log),
                         steady_samples_ms=json.dumps({s['step']: s['step_ms'] for s in samples}),
                         steady_mean_ms=statistics.mean(s['step_ms'] for s in samples),
                         provenance_json=json.dumps({k: v for k, v in provenance.items() if k != 'environment'}, sort_keys=True),
                         scheduler_id=env['SLURM_ARRAY_JOB_ID'] + '_' + env['SLURM_ARRAY_TASK_ID'],
                         artifacts=json.dumps([source(p) for p in (manifest, env_path, directory / 'command.sh', provenance_path, log)]))
                if kind == 'timing':
                    if r.get('scorep') != '0' or r['cubex'] or any(directory.glob('scorep*')) or any(directory.glob('memory*')) or any(k.startswith('SCOREP_') for k in env):
                        raise ValueError(f'Not plain timing {directory}')
                    if 'build_gpu_refresh_plain/terrain_solver' not in ' '.join(command):
                        raise ValueError(f'Wrong plain build {directory}')
                else:
                    profile = resolve_path(manifest, r['cubex'], original['config'], True)
                    r.update(cubex=str(profile), cubex_sha256=source(profile)['sha256'])
                    if env.get('SCOREP_ENABLE_PROFILING') != 'true':
                        raise ValueError(f'Missing Score-P treatment {directory}')
                if kind == 'memory':
                    memory = validate_memory(directory, base, 96)
                    if memory['token'] in tokens or memory['token'] != env.get('TERRAIN_MEMORY_TOKEN'):
                        raise ValueError(f'Memory attribution collision {directory}')
                    tokens.add(memory['token'])
                    r.update(memory, memory_json_sha256=source(directory / 'memory_summary.json')['sha256'],
                             memory_csv_sha256=source(directory / 'memory_samples.csv')['sha256'])
                rows.append(r)
    coverage(rows)
    if kind != 'timing' and len({r['cubex'] for r in rows}) != 70:
        raise ValueError('Profile reused across production runs')
    return rows


def maxima(rows):
    result = []
    for m in MODELS:
        for c in ORDER:
            group = [r for r in rows if (r['model'], r['config']) == (m, c)]
            record = dict(model=m, config=c, rep_count=len(group))
            for metric in ('pss_peak_bytes', 'rss_peak_bytes'):
                winner = max(group, key=lambda r: r[metric])
                record.update({metric: winner[metric], metric + '_rep': winner['rep'], metric + '_source': winner['memory_json']})
            result.append(record)
    return result


def render(rows, output, memory=False):
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, m in zip(axes, MODELS):
        groups = [[r for r in rows if (r['model'], r['config']) == (m, c)] for c in ORDER]
        if memory:
            for i, group in enumerate(groups):
                for j, metric in enumerate(('pss_peak_bytes', 'rss_peak_bytes')):
                    value = group[0][metric] / 2**30
                    x = i + (j-.5)*.36
                    ax.bar(x, value, width=.34, color=PALETTE[i], hatch='//' if j else None, edgecolor='#444444')
                    ax.annotate(f'{value:.2f}', (x, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=7)
            ax.legend(handles=[Patch(facecolor='#888888', label='Sampled PSS'), Patch(facecolor='#888888', hatch='//', label='Sampled RSS')], frameon=False)
        else:
            values = [[v/1000 for r in g for v in json.loads(r['steady_samples_ms']).values()] for g in groups]
            boxes = ax.boxplot(values, positions=range(7), patch_artist=True, medianprops={'color': 'black'})
            for box, color in zip(boxes['boxes'], PALETTE):
                box.set_facecolor(color)
                box.set_alpha(.7)
            ax.set_yscale('log')
        ax.set_xticks(range(7), LABELS)
        ax.set_title('[TS] WaterCNN' if m == 'watercnn' else '[TS] Giant MLP')
        ax.set_ylabel('Maximum sampled lifetime peak (GiB)' if memory else 'Plain steady ML-step latency (s)')
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=.2)
    caption = ('Audited same-sweep process-set peaks; maximum across 5 reps per model/config. Lifetime includes warmup.\n'
               'Score-P memory treatment; sampled lower bounds, not atomic sweeps or cgroup accounting.') if memory else (
               'Plain build, no Score-P or sampler. Warmup step 2 excluded; 50 steady samples from 5 independent runs per model/config.\n'
               'Median/IQR; whiskers 1.5 IQR. Steps within a run are correlated; solver end-to-end synchronized timing.')
    fig.text(.5, .02, caption, ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, 1))
    name = 'memory_peaks' if memory else 'latency_boxplots'
    save_figure(fig, output / (name+'.png'))
    fig.savefig(output / (name+'.pdf'), bbox_inches='tight')
    close_figure(fig)


def phases(rows, output):
    raw, records = [], []
    for r in rows:
        trees = parse_cubex_file(Path(r['cubex']))
        base = 'PHYDLL_CPP' if r['config'].startswith('PHYDLL') else r['config']
        if base == 'PHYDLL_CPP':
            validate_phydll(trees, 22, r['cubex'])
        meta = {k: r[k] for k in ('model', 'rep', 'cubex')}
        a, b = extract(trees, dict(meta, config=base), 10)
        expected_views = {'application', 'provider_dl' if base == 'PHYDLL_CPP' else 'provider'}
        if {v['view'] for v in b} != expected_views or any(v['rank_count'] != 96 for v in b):
            raise ValueError(f'Invalid profile scopes {meta}')
        for v in a + b:
            v['config'] = r['config']
        raw.extend(a)
        records.extend(b)
        print(f'Extracted {r["model"]} {r["config"]} rep {r["rep"]}', flush=True)
    grouped = defaultdict(list)
    for r in records:
        grouped[tuple(r[k] for k in ('model', 'config', 'view', 'phase', 'scope'))].append(r)
    means = []
    for key, group in sorted(grouped.items()):
        if len(group) != 5 or {r['rep'] for r in group} != set('12345'):
            raise ValueError(f'Phase replicate coverage {key}')
        means.append(dict(zip(('model', 'config', 'view', 'phase', 'scope'), key), rep_count=5,
                          mean_ms_per_step=statistics.mean(r['ms_per_step'] for r in group),
                          std_ms_per_step=statistics.stdev(r['ms_per_step'] for r in group), rank_count=96, steady_steps=10))
    write_csv(output / 'raw_callpaths.csv', raw)
    write_csv(output / 'phase_replicates.csv', records)
    write_csv(output / 'phase_means.csv', means)
    render_phases(means, output)


def render_phases(means, output):
    for application in (True, False):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        legend = {}
        for ax, model in zip(axes, MODELS):
            totals = []
            for i, cfg in enumerate(ORDER):
                view = 'application' if application or cfg == 'AIx' else 'provider_dl' if cfg.startswith('PHYDLL') else 'provider'
                data = {r['phase']: r['mean_ms_per_step'] for r in means if (r['model'], r['config'], r['view']) == (model, cfg, view)}
                groups = [('Prepare input', ['app_prepare_input']), ('Inference', ['app_provider_inference']),
                          ('Finalize output', ['app_finalize_output']), ('Other', ['Other / overhead'])] if application else SIMPLIFIED.get('PHYDLL_CPP' if cfg.startswith('PHYDLL') else cfg, SIMPLIFIED['SmartSim'])
                bottom = 0
                for label, bins in groups:
                    value = sum(data[p] for p in bins)
                    bar = ax.bar(i, value, bottom=bottom, width=.65, color=SIMPLE_COLORS[label], edgecolor='white', linewidth=.4)
                    legend[label] = bar[0]
                    bottom += value
                ax.annotate(f'{bottom:,.1f}', (i, bottom), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=7)
                totals.append(bottom)
            ax.set_xticks(range(7), LABELS)
            ax.set_title('[TS] WaterCNN' if model == 'watercnn' else '[TS] Giant MLP')
            ax.set_ylabel('Mean ms / steady step / scope rank')
            ax.grid(axis='y', alpha=.25)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(0, max(totals) * 1.2)
        fig.legend(legend.values(), legend.keys(), loc='lower center', bbox_to_anchor=(.5, .055), ncol=4, frameon=False, fontsize=8)
        fig.suptitle('Score-P application phases: 96 solver ranks' if application else 'Score-P provider phases: AIx/SmartSim solver scope; PhyDLL 96 DL ranks')
        fig.text(.5, .015, '5 reps; 10 steady steps; disjoint exclusive bins; warmup excluded. Actual instrumented scope times, NOT scaled to plain latency.\nSmartSim model request is client elapsed time, not isolated server compute. PhyDLL excludes DL root/lifecycle time.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .19, 1, .95))
        name = 'application_stacked' if application else 'provider_stacked'
        save_figure(fig, output / (name+'.png'))
        fig.savefig(output / (name+'.pdf'), bbox_inches='tight')
        close_figure(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Audit all 210 source runs without creating outputs or parsing Cube')
    parser.add_argument('--refresh-local', action='store_true', help='Refresh only this entrypoint\'s fixed output namespace, never legacy figures')
    args = parser.parse_args()
    datasets = {kind: collect(kind) for kind in ('timing', 'memory', 'performance')}
    arrays = sorted({r['scheduler_id'].split('_')[0] for rows in datasets.values() for r in rows})
    accounting_command = ['sacct', '-j', ','.join(arrays), '--format=JobID,State,ExitCode', '--parsable2', '--noheader', '-X']
    accounting = subprocess.run(accounting_command, check=True, capture_output=True, text=True).stdout
    states = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines()}
    for rows in datasets.values():
        for r in rows:
            if states.get(r['scheduler_id']) != ['COMPLETED', '0:0']:
                raise ValueError(f'Selected task not cleanly completed: {r["scheduler_id"]}')
    print('Preflight passed: timing=70, memory=70, Score-P profiles=70', flush=True)
    if args.preflight:
        return
    if args.refresh_local and not (OUTPUT / 'manifest.json').is_file():
        raise ValueError('Refresh requires a previously generated local manifest')
    OUTPUT.mkdir(exist_ok=args.refresh_local)
    (OUTPUT / '.gitignore').write_text('*\n')
    write_csv(OUTPUT / 'scheduler_accounting.csv', [dict(job_id=line.split('|')[0], state=line.split('|')[1], exit_code=line.split('|')[2]) for line in accounting.splitlines()])
    for kind, rows in datasets.items():
        write_csv(OUTPUT / (kind+'_replicates.csv'), rows)
    write_csv(OUTPUT / 'latency_steps.csv', [dict(model=r['model'], config=r['config'], rep=r['rep'], step=step, step_ms=ms, log=r['log']) for r in datasets['timing'] for step, ms in json.loads(r['steady_samples_ms']).items()])
    write_csv(OUTPUT / 'profile_manifest.csv', [{k: r[k] for k in ('model', 'config', 'rep', 'cubex', 'cubex_sha256', 'results_csv')} for r in datasets['performance']])
    peaks = maxima(datasets['memory'])
    write_csv(OUTPUT / 'memory_maxima.csv', peaks)
    render(datasets['timing'], OUTPUT)
    render(peaks, OUTPUT, True)
    phases(datasets['performance'], OUTPUT)
    report = dict(command=[sys.executable, *sys.argv], accounting_command=accounting_command, selectors={k: [(a, sorted(b), c, d) for a, b, c, d in selectors(k)] for k in datasets},
                  coverage={k: len(v) for k, v in datasets.items()}, memory_accounting=ACCOUNTING,
                  scripts=[source(HERE / name) for name in ('terrain_cpu_complete_analysis.py', 'extract_scorep_tables.py', 'plot_terrain_nobatch.py', 'terrain_cpu_diagnostics.py')],
                  caveats='Different builds/treatments: plain timing; Score-P profiles and sampled memory retain instrumentation. Numerical parity not done. Recorded runtime metadata/environment and launch evidence, not remote-process inspection. Historical provenance hashes retained, not asserted equal to mutable current sources.',
                  records=datasets)
    (OUTPUT / 'manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    (OUTPUT / 'README.md').write_text('# Terrain CPU Complete, 2026-10-13\n\n70 plain timing runs, 70 audited memory runs, 70 Score-P profiles. Both models, seven configurations, exactly reps 1..5.\n\nCommand: `' + ' '.join(report['command']) + '`\n\n' + report['caveats'] + '\n\n' + ACCOUNTING + '\n\nTiming uses 700 steady samples (steps 4..22), excludes step 2. Memory uses six-step lifetime observations including warmup; each metric is independently maximized across five reps, with winning source in memory_maxima.csv. Profiles use ten steady calls per rank, solver steady ancestry excludes warmup; DL visits must match ten, with 96 solver and 96 DL ranks. Exclusive time belongs to deepest semantic ancestor, never inclusive double counting. SmartSim request time is client elapsed time. No normalization to plain timing.\n\nExplicit sources: timing other array 4952588, readiness array 4952363; profiles baseline production_4644312 AIx/SmartSim plus patched phydll_master_20261012 false/true; memory baseline TPQ1/12 plus patched PhyDLL plus new AIx/default/TPQ96. Superseded watercnn rep1 excluded, clean 4952823_0 included. Smoke/failed namespaces never selected. Runtime logs, environment attestations, launch/config/readiness, raw memory census and Cube counts audited. All input paths and hashes are preserved in replicate CSVs and manifest.json.\n')
    print(f'Generated {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
