#!/usr/bin/env python3
"""Analyze final CPU terrain performance or separate lifetime memory campaigns."""
import argparse
import csv
import hashlib
import json
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from extract_scorep_tables import ALIASES, ORDER, discover, validate_campaign_coverage
from plot_terrain_nobatch import ACCOUNTING, COLORS, MODELS, STEP, number, validate_memory

HERE = Path(__file__).resolve().parent
PINNED_PYTHON = HERE.parent / 'CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python'
LABELS = ['AIx', 'PhyDLL C++', 'SmartSim\ndefault', 'SmartSim\nTPQ=1\nINTRA=96',
          'SmartSim\nTPQ=12\nINTRA=8', 'SmartSim\nTPQ=96\nINTRA=1']
FIELDS = {'model', 'rep', 'config', 'status', 'cubex', 'log', 'mode', 'db_batch', 'total_bs',
          'phydll_io_log', 'phydll_io_log_barriers', 'internal_mem_verbose', 'internal_mem_collective',
          'monitoring', 'timing_sync', 'solver_ranks', 'dl_ranks', 'rank_grid', 'db_min_batch',
          'db_timeout', 'omit_threads', 'omit_set_cpus', 'tpq', 'intra', 'inter', 'telemetry',
          'steps', 'steady_steps', 'policy', 'thread_settings', 'provenance', 'run_environment',
          'environment_sha256', 'run_environment_sha256', 'memory_sampler', 'phydll_logging', 'internal_memory', 'binding'}
APPROVED_PHYDLL_SHA256 = 'dd551169e436d07e367fc5304389eafba3999f43587a20af0ec0fc8b8c87f553'
TOTAL_BS = {'watercnn': 8294400, 'giant': 46080}
THREADS = {'AIx': (0, 0, 0), 'PHYDLL_CPP': (0, 1, 0), 'SS_DEFAULT': (1, 1, 1),
           'TPQ1_I96': (1, 96, 0), 'TPQ12_I8': (12, 8, 0), 'TPQ96_I1': (96, 1, 0)}
RUNTIME = {'PHYDLL_IO_LOG': '0', 'PHYDLL_IO_LOG_BARRIERS': '0', 'PHYDLL_TELEMETRY': '0',
           'SMARTSIM_FLAG_TELEMETRY': '0', 'MLCOUPLING_MEM_LOG_VERBOSE': '0',
           'MLCOUPLING_MEM_LOG_COLLECTIVE': '0', 'MLCOUPLING_STEP_TIMING_SYNC': '1',
           'SCOREP_ENABLE_PROFILING': 'true', 'SCOREP_ENABLE_TRACING': 'false', 'SCOREP_METRIC_PAPI': '',
           'SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY': 'false'}
MEMORY_NOTE = (ACCOUNTING + ' Memory and performance are separate datasets; memory-run timings are never '
               'pooled with performance. Lifetime peaks include warmup. Six-step and 22-step runs have '
               'different observation durations, not identical workloads; sampled peaks are lower bounds.')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No records for {path}')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}


def validate_runtime(manifest, entry, cfg, mode, steady_steps, ranks, expected_provenance, checked_inputs=None):
    """Audit recorded launch exports and inputs, not the analysis shell's environment."""
    if ranks != 96:
        raise ValueError('Final runner requires 96 solver ranks')
    tpq, intra, omit = THREADS[cfg]
    expected = dict(phydll_io_log=0, phydll_io_log_barriers=0, internal_mem_verbose=0,
                    internal_mem_collective=0, monitoring=int(mode == 'memory'), timing_sync=1,
                    solver_ranks=96, dl_ranks=96 if cfg == 'PHYDLL_CPP' else 0, rank_grid='12x8',
                    db_min_batch=0, db_timeout=0, omit_threads=omit, omit_set_cpus=omit,
                    tpq=tpq, intra=intra, inter=0 if cfg == 'AIx' else 1,
                    thread_settings='unforced' if cfg == 'AIx' else 'omitted_nominal' if omit else 'explicit',
                    memory_sampler=int(mode == 'memory'), phydll_logging=0, internal_memory=0,
                    telemetry=0, total_bs=TOTAL_BS[entry['model']],
                    steps=2 * steady_steps + 2, steady_steps=steady_steps,
                    policy='final_total_per_queue_except_default_and_TPQ96',
                    binding='pinned' if cfg in ('AIx', 'PHYDLL_CPP') else 'unbound')
    for field, value in expected.items():
        if entry[field] != str(value):
            raise ValueError(f'{manifest}: {cfg} {field}={entry[field]!r}, expected {value}')
    job = manifest.parents[2]
    directory = manifest.parent / entry['config']
    provenance_path = job / 'provenance.json'
    for field, path in [('provenance', provenance_path), ('run_environment', directory / 'run.env'),
                        ('environment_sha256', directory / 'environment.sha256')]:
        if resolve_path(manifest, entry[field], entry['config']) != path.resolve():
            raise ValueError(f'{manifest}: {field} path mismatch')
    provenance = json.loads(provenance_path.read_text())
    approved = dict(phydll_default_logging=0, phydll_default_barriers=0,
                    approved_phydll_sha256=APPROVED_PHYDLL_SHA256)
    if expected_provenance is not None:
        if set(expected_provenance) - set(approved):
            raise ValueError('Expected provenance permits only logging/barrier defaults and approved_phydll_sha256')
        approved.update(expected_provenance)
    if (approved['phydll_default_logging'] != 0 or approved['phydll_default_barriers'] != 0 or
            not re.fullmatch('[0-9a-f]{64}', str(approved['approved_phydll_sha256']))):
        raise ValueError('Invalid expected provenance policy')
    if not isinstance(provenance, dict) or any(provenance.get(k) != v for k, v in approved.items()):
        raise ValueError(f'{provenance_path}: expected provenance mismatch')
    semantics = dict(device='CPU', mode=mode, memory_sampler=int(mode == 'memory'),
                     phydll_logging=0, phydll_io_log=0, phydll_io_log_barriers=0, internal_memory=0,
                     internal_mem_verbose=0, internal_mem_collective=0, telemetry=0, timing_sync=1,
                     solver_ranks=96, rank_grid='12x8', db_min_batch=0, db_timeout=0,
                     policy=expected['policy'])
    if any(type(provenance.get(k)) is not type(v) or provenance[k] != v for k, v in semantics.items()):
        raise ValueError(f'{provenance_path}: provenance runtime policy mismatch')
    inputs = provenance['inputs']
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('Missing provenance input hashes')
    for path, digest in inputs.items():
        identity = (path, digest)
        if checked_inputs is not None and identity in checked_inputs:
            continue
        if not re.fullmatch('[0-9a-f]{64}', str(digest)) or source(Path(path))['sha256'] != digest:
            raise ValueError(f'Provenance input SHA mismatch: {path}')
        if checked_inputs is not None:
            checked_inputs.add(identity)
    libraries = []
    for name in ('solver', 'dl'):
        text = (job / f'{name}_libraries.txt').read_text()
        matches = re.findall(r'^\s*libphydll\.so(?:\.\d+)*\s+=>\s+(\S+)', text, re.M)
        if 'not found' in text or len(matches) != 1:
            raise ValueError(f'{name}: unresolved/ambiguous PhyDLL library')
        libraries.append(Path(matches[0]).resolve())
    if libraries[0] != libraries[1] or inputs.get(str(libraries[0])) != approved['approved_phydll_sha256']:
        raise ValueError('Loaded PhyDLL library approval mismatch')
    env_path = directory / 'run.env'
    attestations = (directory / 'environment.sha256').read_text().splitlines()
    if len(attestations) != 1:
        raise ValueError('Expected one run.env SHA attestation')
    digest, recorded_path = attestations[0].split(maxsplit=1)
    if (Path(recorded_path.lstrip('*')).resolve() != env_path.resolve() or
            digest != source(env_path)['sha256'] or entry['run_environment_sha256'] != digest):
        raise ValueError(f'{env_path}: SHA attestation mismatch')
    environment = {}
    for line in env_path.read_text().splitlines():
        # `env` may include multiline exported module functions. Only scalar exports
        # are policy inputs; the complete original snapshot remains SHA-attested.
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', line):
            continue
        key, separator, value = line.partition('=')
        if not separator or key in environment:
            raise ValueError(f'{env_path}: malformed/duplicate export')
        environment[key] = value
    runtime = {**RUNTIME, 'MODE': mode, 'MODEL': entry['model'], 'REP': entry['rep'],
               'STEPS': entry['steps'], 'APPROVED_PHYDLL_SHA256': approved['approved_phydll_sha256']}
    if cfg == 'PHYDLL_CPP':
        runtime.update(MLCOUPLING_INTRA_OP_THREADS='1', MLCOUPLING_INTER_OP_THREADS='1',
                       PHYDLL_DL_COUNT='1', PHYDLL_DL_FIELD_COUNT='1')
    if cfg.startswith('TPQ') or cfg == 'SS_DEFAULT':
        runtime.update(SMARTSIM_OMIT_THREADS=str(omit), SMARTSIM_OMIT_SET_CPUS=str(omit),
                       SMARTSIM_THREADS_PER_QUEUE=str(tpq), SMARTSIM_INTRA_OP_THREADS=str(intra),
                       SMARTSIM_INTER_OP_THREADS='1')
    if any(environment.get(k) != v for k, v in runtime.items()):
        raise ValueError(f'{env_path}: effective runtime exports disagree with policy')
    if Path(environment.get('SCOREP_EXPERIMENT_DIRECTORY', '')).resolve() != (directory / 'scorep').resolve():
        raise ValueError('Score-P output directory mismatch')
    token = environment.get('TERRAIN_MEMORY_TOKEN')
    if (mode == 'memory' and not token) or (mode == 'performance' and token is not None):
        raise ValueError('Runtime monitoring token disagrees with mode')
    if cfg != 'PHYDLL_CPP' and any(k in environment for k in
            ('MLCOUPLING_INTRA_OP_THREADS', 'MLCOUPLING_INTER_OP_THREADS', 'PHYDLL_DL_COUNT', 'PHYDLL_DL_FIELD_COUNT')):
        raise ValueError('Leaked provider thread exports')
    command_path = directory / 'command.sh'
    command = shlex.split(command_path.read_text())
    rank_positions = [i for i, word in enumerate(command) if word == '-n']
    if len(rank_positions) != (2 if cfg == 'PHYDLL_CPP' else 1) or any(
            command[i + 1:i + 2] != ['96'] for i in rank_positions):
        raise ValueError('MPI launch rank count mismatch')
    pin_wrapper = str((HERE / 'pin_wrapper.sh').resolve())
    for pos in rank_positions:
        exec_after_n = str(Path(command[pos + 2]).resolve())
        if cfg in ('AIx', 'PHYDLL_CPP'):
            if exec_after_n != pin_wrapper:
                raise ValueError(f'{command_path}: expected pin_wrapper for {cfg}')
        else:
            if exec_after_n == pin_wrapper:
                raise ValueError(f'{command_path}: expected unbound launch for {cfg}')
    options = {'--device': 'CPU', '--gpus-per-node': '0', '--rank-grid-x': '12', '--rank-grid-z': '8',
               '--steps': entry['steps'], '--ml-batch-size': '10000000', '--model-backend': 'TORCH',
               '--model-io-layout': 'split_3x3' if cfg.startswith('TPQ') or cfg == 'SS_DEFAULT' else 'flat_contiguous'}
    if cfg.startswith('TPQ') or cfg == 'SS_DEFAULT':
        options.update({'--ml-db-batch-size': entry['db_batch'], '--ml-db-min-batch-size': '0',
                        '--ml-db-min-batch-timeout': '0'})
    for flag, value in options.items():
        positions = [i for i, word in enumerate(command) if word == flag]
        if len(positions) != 1 or command[positions[0] + 1:positions[0] + 2] != [value]:
            raise ValueError(f'{command_path}: {flag} policy mismatch')
    for key in RUNTIME:
        if command.count(key) != (2 if cfg == 'PHYDLL_CPP' else 1) or any(
                command[i - 1] != '-x' for i, word in enumerate(command) if word == key):
            raise ValueError(f'{command_path}: missing MPI runtime export {key}')
    if cfg == 'PHYDLL_CPP':
        for key in ('MLCOUPLING_INTRA_OP_THREADS', 'MLCOUPLING_INTER_OP_THREADS', 'PHYDLL_DL_COUNT', 'PHYDLL_DL_FIELD_COUNT'):
            if command.count(key) != 2 or any(command[i - 1] != '-x' for i, word in enumerate(command) if word == key):
                raise ValueError(f'Missing PhyDLL MPI export {key}')
    for flag in ('--model-path', '--input-hdf5', '--cpp-ml-config'):
        if command.count(flag) != 1:
            raise ValueError(f'Missing/duplicate input option {flag}')
        path = Path(command[command.index(flag) + 1]).resolve()
        if str(path) not in inputs:
            raise ValueError(f'Unattested runtime input {path}')
        expected_name = (f'{entry["model"]}{"_split" if cfg.startswith("TPQ") or cfg == "SS_DEFAULT" else ""}_cpu.pt'
                         if flag == '--model-path' else
                         'prep_watercnn_4k.h5' if flag == '--input-hdf5' and entry['model'] == 'watercnn' else
                         'prep_giant_240x192.h5' if flag == '--input-hdf5' else
                         'config_phydll.toml' if cfg == 'PHYDLL_CPP' else 'config_aix.toml' if cfg == 'AIx' else 'config.toml')
        if path.name != expected_name:
            raise ValueError(f'Unexpected runtime input {path}')
    return provenance, {'provenance_source': source(provenance_path), 'runtime_environment': environment,
                        'run_env_source': source(env_path), 'command_source': source(command_path)}


def resolve_path(manifest, value, config, profile=False):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = [manifest.parent / path]
    if profile:
        candidates.append(manifest.parent / config / 'scorep' / path)
    candidates.append(manifest.parents[2] / path)
    matches = {p.resolve() for p in candidates if p.is_file()}
    if len(matches) != 1:
        raise ValueError(f'{manifest}: missing/ambiguous artifact {value}')
    return matches.pop()


def validate_policy(row, config):
    total = number(row['total_bs'], 'total_bs', True)
    batch = number(row['db_batch'], 'db_batch')
    if total != int(total) or batch != int(batch):
        raise ValueError('Batch sizes must be integers')
    expected = total if config == 'TPQ1_I96' else total / 12 if config == 'TPQ12_I8' else 0
    if config == 'TPQ12_I8' and total % 12:
        raise ValueError('total_bs must be divisible by 12')
    if batch != expected:
        raise ValueError(f'{config}: db_batch={batch}, expected {expected}')
    if config in ('SS_DEFAULT', 'TPQ96_I1') and row['db_batch'] != '0':
        raise ValueError(f'{config}: db_batch must be literal 0')


def collect(roots, mode, reps, steady_steps, ranks=96, expected_provenance=None):
    manifests = set()
    for root in roots:
        manifests.update(root.glob('job_*/*/rep[0-9]*/results.csv'))
        manifests.update(root.glob('*/rep[0-9]*/results.csv'))
        manifests.update((root / mode).glob('job_*/*/rep[0-9]*/results.csv'))
    rows, seen, tokens, checked_inputs = [], set(), set(), set()
    for manifest in sorted(manifests):
        entries = read_csv(manifest)
        if not entries or not FIELDS <= entries[0].keys():
            raise ValueError(f'{manifest}: required fields: {sorted(FIELDS)}')
        for entry in entries:
            cfg = ALIASES.get(entry['config'])
            key = (entry['model'], cfg, entry['rep'])
            if cfg not in ORDER or entry['model'] not in MODELS or entry['status'] != 'SUCCESS':
                raise ValueError(f'{manifest}: invalid/failed run {key}')
            if entry['mode'] != mode:
                raise ValueError(f'{manifest}: expected mode={mode}, got {entry["mode"]}')
            if manifest.parent.name != 'rep' + entry['rep'] or manifest.parent.parent.name != entry['model']:
                raise ValueError(f'{manifest}: model/rep path mismatch')
            if key in seen:
                raise ValueError(f'Duplicate run {key}')
            seen.add(key)
            validate_policy(entry, cfg)
            provenance, audit = validate_runtime(manifest, entry, cfg, mode, steady_steps, ranks, expected_provenance, checked_inputs)
            row = {**entry, **audit, 'config': cfg,
                   'results_csv': str(manifest.resolve()), 'dataset': mode,
                   'provenance_json': json.dumps(provenance, sort_keys=True, separators=(',', ':')),
                   'metadata_sha256': source(manifest)['sha256']}
            if mode == 'performance':
                log = resolve_path(manifest, entry['log'], entry['config'])
                times = {}
                for line in log.read_text().splitlines():
                    match = STEP.search(line)
                    if match:
                        step = int(match[1])
                        if step in times:
                            raise ValueError(f'{log}: duplicate step {step}')
                        times[step] = number(match[2], 'step_ms', True)
                if set(times) != {2, *range(4, 4 + 2 * steady_steps, 2)}:
                    raise ValueError(f'{log}: unexpected/incomplete ML steps {sorted(times)}')
                del times[2]
                row.update(log=str(log), log_sha256=source(log)['sha256'],
                           steady_samples_ms=json.dumps(times, sort_keys=True),
                           steady_mean_ms=statistics.mean(times.values()))
                profile = resolve_path(manifest, entry['cubex'], entry['config'], True)
                row.update(cubex=str(profile), cubex_sha256=source(profile)['sha256'])
            else:
                memory = validate_memory(manifest.parent / entry['config'], cfg, ranks)
                if memory['token'] in tokens:
                    raise ValueError(f'{key}: reused sampler token')
                tokens.add(memory['token'])
                row.update(memory)
                if memory['token'] != audit['runtime_environment']['TERRAIN_MEMORY_TOKEN']:
                    raise ValueError('Sampler/runtime attribution token mismatch')
                row['memory_json_sha256'] = source(Path(memory['memory_json']))['sha256']
                row['memory_csv_sha256'] = source(manifest.parent / entry['config'] / 'memory_samples.csv')['sha256']
            rows.append(row)
    configs = set(ORDER) if mode == 'performance' else {r['config'] for r in rows}
    if mode == 'memory' and not {'TPQ1_I96', 'TPQ12_I8'} <= configs <= {'TPQ1_I96', 'TPQ12_I8', 'PHYDLL_CPP'}:
        raise ValueError('Memory requires TPQ1 and TPQ12, optionally PHYDLL_CPP only')
    expected = {(m, c, str(r)) for m in MODELS for c in configs for r in range(1, reps + 1)}
    if seen != expected:
        raise ValueError(f'Coverage: expected {len(expected)} runs; missing={sorted(expected-seen)}, unexpected={sorted(seen-expected)}')
    for model in MODELS:
        if len({r['total_bs'] for r in rows if r['model'] == model}) != 1:
            raise ValueError(f'{model}: inconsistent total_bs')
    return rows


def memory_maxima(rows):
    """Maximize each metric's audited same-sweep total peak across runs."""
    result = []
    for model in MODELS:
        for cfg in ORDER:
            group = [r for r in rows if r['model'] == model and r['config'] == cfg]
            if not group:
                continue
            summary = {'model': model, 'config': cfg, 'rep_count': len(group),
                       'datasets': ';'.join(sorted({r.get('dataset', '') for r in group}))}
            for metric in ('pss_peak_bytes', 'rss_peak_bytes'):
                peak = max(number(r[metric], metric, True) for r in group)
                winners = [r for r in group if float(r[metric]) == peak]
                summary[f'max_{metric}'] = int(peak)
                summary[f'max_{metric}_reps'] = ';'.join(str(r['rep']) for r in winners)
                summary[f'max_{metric}_sources'] = ';'.join(r.get('memory_json', '') for r in winners)
            result.append(summary)
    return result


def render(rows, output, memory=False, scale='log'):
    from hh3_render import plt, save_figure, close_figure
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'figure.dpi': 200})
    configs = [c for c in ORDER if any(r['config'] == c for r in rows)]
    maxima = {(r['model'], r['config']): r for r in memory_maxima(rows)} if memory else {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model in zip(axes, MODELS):
        groups = [[r for r in rows if r['model'] == model and r['config'] == c] for c in configs]
        if memory:
            for i, group in enumerate(groups):
                for j, metric in enumerate(('pss_peak_bytes', 'rss_peak_bytes')):
                    peak = maxima[model, configs[i]][f'max_{metric}'] / 2**30
                    x = i + (j - .5) * .36
                    ax.bar(x, peak, width=.34,
                           color=COLORS[ORDER.index(configs[i])], hatch='//' if j else None,
                           edgecolor='#444444')
                    ax.annotate(f'{peak:.2f}', (x, peak), xytext=(0, 4),
                                textcoords='offset points', ha='center', fontsize=8)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor='#888888', edgecolor='#444444', label='Sampled PSS peak'),
                               Patch(facecolor='#888888', hatch='//', edgecolor='#444444', label='Sampled RSS peak')],
                      frameon=False)
        else:
            values = [[float(v) / 1000 for r in group for v in json.loads(r['steady_samples_ms']).values()] for group in groups]
            boxes = ax.boxplot(values, positions=range(len(configs)), patch_artist=True,
                               medianprops={'color': 'black'}, flierprops={'marker': 'd', 'markersize': 3})
            for box, cfg in zip(boxes['boxes'], configs):
                box.set_facecolor(COLORS[ORDER.index(cfg)])
                box.set_alpha(.7)
            for i, samples in enumerate(values):
                ax.annotate(f'{statistics.median(samples):.3f}', (i, max(samples)), xytext=(0, 5),
                            textcoords='offset points', ha='center', fontsize=8)
            ax.set_yscale(scale)
        ax.set_xticks(range(len(configs)), [LABELS[ORDER.index(c)] for c in configs])
        ax.set_title('[TS] WaterCNN' if model == 'watercnn' else '[TS] Giant MLP')
        ax.set_ylabel('Maximum sampled peak (GiB)' if memory else 'Steady ML-step latency (s)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format(v, ',.6f').rstrip('0').rstrip('.') if v else '0'))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        ax.margins(y=.2)
        ax.grid(False)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    if memory:
        fig.tight_layout()
    else:
        caption = ('All steady STEP_TIMING samples, warmup excluded. Boxes: median/IQR; whiskers: 1.5 IQR.\n'
                   'Labels: median seconds; steps within a run are correlated. Score-P enabled; no external sampler.')
        counts = {len([r for r in rows if r['model'] == m and r['config'] == c]) for m in MODELS for c in configs}
        caption += '\n' + (f'n={counts.pop()} independent runs per model/config.' if len(counts) == 1 else
                           'Independent run counts vary by model/config; see replicate CSV.')
        fig.text(.5, .02, caption, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, 1))
    name = 'memory_peaks' if memory else 'latency_boxplots'
    save_figure(fig, output / f'{name}.png')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    close_figure(fig)


def reference_memory(path, new_rows, reps, ranks):
    """Re-audit legacy sampler artifacts; never import legacy timings or old batched TPQ1/12."""
    records = read_csv(path)
    selected = {'AIx', 'SS_DEFAULT', 'TPQ96_I1'}
    if not any(r['config'] == 'PHYDLL_CPP' for r in new_rows):
        selected.add('PHYDLL_CPP')
    result, seen, tokens = [], set(), {r['token'] for r in new_rows}
    for entry in records:
        cfg = ALIASES.get(entry['config'])
        if cfg not in selected:
            continue
        key = (entry['model'], cfg, entry['rep'])
        if key in seen or int(entry['solver_ranks']) != ranks:
            raise ValueError(f'Invalid reference coverage/ranks {key}')
        seen.add(key)
        manifest = Path(entry['results_csv'])
        matches = [r for r in read_csv(manifest) if ALIASES.get(r['config']) == cfg and
                   r['model'] == entry['model'] and r['rep'] == entry['rep']]
        if len(matches) != 1 or matches[0]['status'] != 'SUCCESS':
            raise ValueError(f'{key}: reference manifest mismatch')
        original = matches[0]
        if cfg in ('SS_DEFAULT', 'TPQ96_I1'):
            # Old manifests lack policy fields. Require the actual launch log's literal-zero declaration.
            job = re.match(r'job_(\d+)_', manifest.parents[2].name)
            log = HERE / 'logs' / f'terrain_nobatch_{job[1]}.log' if job else Path(entry['latency_log'])
            declaration = re.compile(r'>>> ' + re.escape(original['config']) + r':[^\n]*db_batch=0 min_batch=0 timeout=0')
            if not declaration.search(log.read_text()):
                raise ValueError(f'{key}: no reference zero-batch launch evidence')
            if original.get('db_batch', '0') != '0':
                raise ValueError(f'{key}: reference batch policy mismatch')
        memory = validate_memory(Path(entry['memory_json']).parent, cfg, ranks)
        if memory['token'] in tokens:
            raise ValueError(f'{key}: reused reference token')
        tokens.add(memory['token'])
        result.append({'model': entry['model'], 'config': cfg, 'rep': entry['rep'],
                       'dataset': 'legacy_memory', 'results_csv': str(manifest.resolve()),
                       'reference_csv': str(path.resolve()), 'reference_csv_sha256': source(path)['sha256'],
                       'policy_evidence': source(log) if cfg in ('SS_DEFAULT', 'TPQ96_I1') else None,
                       'steady_steps': entry['steady_steps'], 'solver_ranks': ranks,
                       'policy': 'zero batching' if cfg in ('SS_DEFAULT', 'TPQ96_I1') else 'not applicable',
                       'logging_policy': 'legacy; not equivalent to final PhyDLL logging=0' if cfg == 'PHYDLL_CPP' else 'legacy',
                       **memory, 'memory_json_sha256': source(Path(memory['memory_json']))['sha256'],
                       'memory_csv_sha256': source(Path(memory['memory_json']).parent / 'memory_samples.csv')['sha256']})
    expected = {(m, c, str(r)) for m in MODELS for c in selected for r in range(1, reps + 1)}
    if seen != expected:
        raise ValueError('Reference memory requires exact model/config/rep coverage')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('performance', 'memory'), default='performance')
    parser.add_argument('--campaign-root', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--steady-steps', type=int, default=10)
    parser.add_argument('--solver-ranks', type=int, default=96)
    parser.add_argument('--python', type=Path, default=PINNED_PYTHON, help='Pinned extraction/rendering interpreter')
    parser.add_argument('--expected-provenance', type=Path, help='Stable logging defaults and approved library hash JSON')
    parser.add_argument('--reference-memory', type=Path, help='Validated legacy memory_latency_replicates.csv')
    parser.add_argument('--latency-scale', choices=('linear', 'log'), default='log')
    args = parser.parse_args()
    if min(args.reps, args.steady_steps, args.solver_ranks) < 1:
        parser.error('Counts must be positive')
    if args.mode == 'performance' and args.reference_memory:
        parser.error('--reference-memory is only for memory analysis')
    try:
        expected = json.loads(args.expected_provenance.read_text()) if args.expected_provenance else None
        if expected is not None and not isinstance(expected, dict):
            raise ValueError('Expected provenance must be a JSON object')
        rows = collect(args.campaign_root, args.mode, args.reps, args.steady_steps, args.solver_ranks, expected)
        legacy = reference_memory(args.reference_memory, rows, args.reps, args.solver_ranks) if args.reference_memory else []
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        if args.mode == 'performance':
            manifest = output / 'profile_manifest.csv'
            write_csv(manifest, [{k: r[k] for k in ('model', 'config', 'rep', 'cubex')} for r in rows])
            discovered = discover([], manifest)
            if len(discovered) != len(rows):
                raise ValueError('Profile discovery coverage mismatch')
            phases = output / 'phases'
            subprocess.run([str(args.python), str(HERE / 'extract_scorep_tables.py'), '--results-input', str(manifest),
                            '--steady-steps', str(args.steady_steps), '--output-dir', str(phases)], check=True)
            validate_campaign_coverage(rows, read_csv(phases / 'phase_replicates.csv'),
                                       range(1, args.reps + 1), args.steady_steps, args.solver_ranks)
            subprocess.run([str(args.python), str(HERE / 'plot_scorep_phases.py'), '--results-input',
                            str(phases / 'phase_means.csv'), '--output-dir', str(phases)], check=True)
            write_csv(output / 'latency_steps.csv', [{'model': r['model'], 'config': r['config'], 'rep': r['rep'],
                       'step': step, 'step_ms': ms, 'log': r['log']} for r in rows
                       for step, ms in json.loads(r['steady_samples_ms']).items()])
        write_csv(output / f'{args.mode}_replicates.csv', rows)
        metrics = ('pss_peak_bytes', 'rss_peak_bytes') if args.mode == 'memory' else ('steady_mean_ms',)
        means = []
        for model in MODELS:
            for cfg in ORDER:
                group = [r for r in rows if r['model'] == model and r['config'] == cfg]
                if group:
                    means.append({'model': model, 'config': cfg, 'rep_count': len(group),
                                  **{f'mean_{m}': statistics.mean(float(r[m]) for r in group) for m in metrics}})
        write_csv(output / f'{args.mode}_means.csv', means)
        if legacy:
            write_csv(output / 'combined_memory_replicates.csv', rows + legacy)
            combined_means = []
            for model in MODELS:
                for cfg in ORDER:
                    group = [r for r in rows + legacy if r['model'] == model and r['config'] == cfg]
                    combined_means.append({'model': model, 'config': cfg, 'dataset': group[0]['dataset'],
                                           'rep_count': len(group),
                                           **{f'mean_{m}': statistics.mean(float(r[m]) for r in group) for m in metrics}})
            write_csv(output / 'combined_memory_means.csv', combined_means)
        if args.mode == 'memory':
            write_csv(output / 'memory_maxima.csv', memory_maxima(rows + legacy))
        render(rows + legacy, output, args.mode == 'memory', args.latency_scale)
        report = {'mode': args.mode, 'reps': args.reps, 'steady_steps': args.steady_steps,
                  'solver_ranks': args.solver_ranks, 'expected_provenance': expected,
                  'policy': 'TPQ1=total_bs; TPQ12=total_bs/12; TPQ96/default=literal zero',
                  'semantic_metadata': 'Strict runner CSV, hash-attested run.env, MPI exports, CPU launch inputs, '
                                       'provenance input hashes and approved loaded PhyDLL library validated. '
                                       'Recorded launch environment, not direct remote-process inspection.',
                  'accounting': MEMORY_NOTE if args.mode == 'memory' else 'All steady log samples; warmup excluded; no memory artifacts required.',
                  'records': rows, 'reference_records': legacy}
        (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'Validated {len(rows)} {args.mode} runs; outputs: {output}')
    except (ValueError, KeyError, TypeError, IndexError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
