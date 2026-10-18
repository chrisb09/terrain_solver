#!/usr/bin/env python3
"""Read-only terrain baseline Score-P phase gallery, with explicit scope receipts."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from extract_scorep_tables import callpath, extract, parse_cubex_file

MINI = Path(__file__).resolve().parent
OUT = MINI / 'results/thesis_gpu_figures/ch6_phases'
TABLES = MINI / 'results/thesis_gpu_figures/tables/profiles'
MODELS = ('watercnn8k', 'giantmlp288')
HARDWARE = ('24c1g', '24p24c1g', '96c4g', '96p96c4g')
PROVIDERS = ('AIX', 'PHYDLL', 'shared', 'pergpu')
LABELS = ('AIx', 'PhyDLL', 'SmartSim shared', 'SmartSim per-GPU')
# Match plot_cmi_ch6_phases.py's authored canvas AND physical plotting height.
DIMENSIONS_MM = (150, 60)
AXES_BOTTOM = .17
AXES_TOP = .80
AXES_HEIGHT_MM = 60 * (AXES_TOP - AXES_BOTTOM)
# Each semantic ancestor owns descendant exclusive time exactly once.
MAPS = {
    'AIX': {'gatherInputData': 'Gather (host)', 'aix_controller_inference': 'Device (host)',
            'scatterOutputData': 'Scatter (host)'},
    'PHYDLL': {'dl_recv': 'Input receive', 'dl_frame_copy': 'Input receive',
               'dl_torch_forward': 'Forward (host)', 'dl_send_output': 'Output send',
               'dl_send': 'Output send', 'dl_h2d': 'H2D (host)', 'dl_d2h': 'D2H (host)',
               'dl_input_unpack': 'Other DL', 'dl_output_reorder': 'Other DL',
               'dl_input_allocate': 'Other DL', 'dl_output_allocate': 'Other DL'},
    'shared': {'smartsim_put_tensor': 'Put', 'smartsim_run_model': 'Model request',
               'smartsim_unpack_tensor': 'Get/unpack', 'smartsim_chunk_plan': 'Chunk plan'},
}
MAPS['pergpu'] = MAPS['shared']
PHASES = {'AIX': ('Gather (host)', 'Device (host)', 'Scatter (host)'),
          'PHYDLL': ('Input receive', 'H2D (host)', 'Forward (host)', 'D2H (host)', 'Output send', 'Other DL'),
          'shared': ('Put', 'Model request', 'Get/unpack', 'Chunk plan')}
PHASES['pergpu'] = PHASES['shared']
COLORS = {'Gather (host)': '#4C78A8', 'Device (host)': '#C99449', 'Scatter (host)': '#B279A2',
          'Input receive': '#4C78A8', 'H2D (host)': '#72B7B2', 'Forward (host)': '#F58518',
          'D2H (host)': '#54A24B', 'Output send': '#B279A2', 'Other DL': '#BAB0AC',
          'Put': '#72B7B2', 'Model request': '#E45756', 'Get/unpack': '#B279A2', 'Chunk plan': '#BAB0AC'}
DISPLAY_PHASES = {'Scatter (host)': 'Scatter/wait (host)', 'Put': 'Put (host)',
                  'Model request': 'Model request (host)', 'Get/unpack': 'Get/unpack (host)'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)


def environment(directory):
    path = directory / 'runtime.env.json'
    if path.exists():
        return json.loads(path.read_text()), path
    path = directory / 'runtime.env'
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line), path


def inventory():
    """Only receipt-backed, current-workload baselines; never select by duration/mtime."""
    manifests = sorted((MINI / 'results/gpu_refresh').glob('*/*/manifest.csv'))
    manifests += sorted((MINI / 'results/scorep_pilots/pilot22_20261007_retry1').glob('*/manifest.csv'))
    candidates = []
    for manifest in manifests:
        for row in read_csv(manifest):
            if row.get('model') not in MODELS:
                continue
            provider, variant = row['provider'], row['variant']
            if row.get('layout', 'default') != 'default':
                continue
            if (provider in ('AIX', 'PHYDLL') and variant == 'default'):
                slot = provider
            elif provider == 'SMARTSIM' and variant in ('shared', 'shared_c0', 'pergpu'):
                slot = 'shared' if variant.startswith('shared') else 'pergpu'
            else:
                continue
            if row.get('mode', 'profile') not in ('profile', 'trace'):
                continue
            directory = Path(row.get('profiles', row.get('profiles_file', ''))).parent
            if not (directory / 'profiles.txt').is_file():
                continue
            env, env_path = environment(directory)
            hw = row.get('hardware', row.get('config'))
            if hw not in HARDWARE:
                continue
            profiles = sorted({str(Path(p).resolve()) for p in (directory / 'profiles.txt').read_text().splitlines()
                               if p.endswith('.cubex') and Path(p).is_file()})
            tracing = env.get('SCOREP_ENABLE_TRACING_ENV', 'false').lower() in ('true', '1')
            reason = ''
            expected = ('watercnn', '7680', '4320', '60', '20000000') if row['model'] == 'watercnn8k' else (
                'benchmark_giant_mlp', '480', '288', '16', '500000')
            actual = tuple(env.get(k, '') for k in ('MODEL_NAME_ENV', 'TARGET_WIDTH_ENV', 'TARGET_HEIGHT_ENV',
                                                   'CHUNK_SIZE_ENV', 'ML_BATCH_SIZE_ENV'))
            if row.get('rc') != '0' or row.get('validation_error', '') or row.get('cleanup_safe', 'True') != 'True':
                reason = 'unsuccessful receipt'
            elif actual != expected:
                reason = f'workload mismatch: {actual}'
            elif not profiles:
                reason = 'no extant cubex'
            elif slot in ('shared', 'pergpu') and tuple(env.get(k, '') for k in (
                    'SMARTSIM_THREADS_PER_QUEUE', 'SMARTSIM_INTRA_OP_THREADS', 'SMARTSIM_INTER_OP_THREADS')) != ('1', '8', '1'):
                reason = 'SmartSim thread mismatch'
            elif slot == 'AIX' and env.get('AIX_COMMUNICATION_MODE_ENV', 'collective') != 'collective':
                reason = 'not collective'
            config = directory / 'config.toml'
            if slot == 'PHYDLL' and ('solver_readiness_wait = false' not in config.read_text()
                                     or 'transport_layout = "packed"' not in config.read_text()):
                reason = 'not optimized blocking PhyDLL'
            candidates.append(dict(model=row['model'], hardware=hw, provider=slot,
                variant=variant, layout='default', job_id=row.get('job_id', env.get('SLURM_JOB_ID')),
                repeat=row.get('repeat', '1'), steady_steps=int(row.get('steady', int(env['TOTAL_STEPS_ENV']) // 2 - 1)),
                warmup_calls=int(row.get('warm', 1)), manifest=str(manifest.resolve()),
                manifest_sha256=sha256(manifest), directory=str(directory.resolve()),
                environment=str(env_path.resolve()), environment_sha256=sha256(env_path),
                config=str(config.resolve()), config_sha256=sha256(config),
                workload=json.dumps(dict(zip(('model', 'width', 'height', 'chunk', 'batch'), actual))),
                profiles=';'.join(profiles), traced=tracing, rejection=reason, decision='candidate'))
    selected = []
    for model in MODELS:
        for hw in HARDWARE:
            for provider in PROVIDERS:
                group = [r for r in candidates if (r['model'], r['hardware'], r['provider']) == (model, hw, provider)
                         and not r['rejection']]
                # Retry campaigns explicitly replace earlier failed/obsolete captures.
                # Prefer untraced; within that class use the authorized replacement job order.
                group.sort(key=lambda r: (r['traced'], -int(r['job_id'])))
                if group:
                    group[0]['decision'] = 'selected untraced' if not group[0]['traced'] else 'selected trace-backed profile'
                    selected.append(group[0])
                    for other in group[1:]:
                        other['decision'] = 'not selected: untraced preference / explicit later replacement'
    return candidates, selected


def phase_metrics(trees, provider, steps):
    """Exclusive ownership with per-phase active-rank normalization, not GPU sums."""
    mapping = MAPS[provider]
    totals = defaultdict(float)
    anchors = defaultdict(dict)
    rank_rows = []
    for rank, nodes in enumerate(trees):
        owned = defaultdict(float)
        calls = defaultdict(float)
        for node in nodes.values():
            path = callpath(nodes, node)
            selected = ('solver_step_ml_steady' in path and 'solver_step_ml_warmup' not in path)
            if provider == 'PHYDLL':
                selected = path[0] == 'phydll_dl_client'
            if not selected:
                continue
            semantic = next((mapping[n] for n in reversed(path) if n in mapping), None)
            if semantic is None:
                continue
            owned[semantic] += node.excl_time
            if node.name in mapping and not any(n in mapping and mapping[n] == semantic for n in path[:-1]):
                valid_visits = (node.visits % steps == 0 if provider in ('shared', 'pergpu')
                                else node.visits == steps)
                if semantic not in ('Other DL', 'Chunk plan') and node.visits and not valid_visits:
                    raise ValueError(f'{provider} rank {rank} {node.name}: {node.visits} != {steps} steady visits')
                calls[semantic] += node.visits
        for phase, seconds in owned.items():
            visits = calls[phase]
            if not visits and not seconds:
                continue
            # Allocation and chunk phases may have multiple visits within a step.
            if provider not in ('shared', 'pergpu') and phase not in ('Other DL', 'Chunk plan', 'Input receive') and visits != steps:
                raise ValueError(f'{provider} rank {rank} {phase}: {visits} != {steps} steady visits')
            anchors[phase][rank] = visits
            totals[phase] += seconds
            rank_rows.append(dict(rank_tree_index=rank, phase=phase, raw_exclusive_owned_s=seconds,
                                  anchor_visits=visits, steady_steps=steps, seconds_per_step=seconds / steps))
    rows = []
    for phase in PHASES[provider]:
        ranks = len(anchors[phase])
        denominator = ranks * steps
        rows.append(dict(phase=phase, scope='native DL host' if provider == 'PHYDLL' else 'solver host',
            rank_count=ranks, steady_steps=steps, denominator=denominator,
            rank_tree_indices=';'.join(map(str, anchors[phase])), raw_total_s=totals[phase] if ranks else '',
            raw_calls=sum(anchors[phase].values()), mean_s=totals[phase] / denominator if ranks else '',
            status='measured' if ranks else 'missing region', independent_captures=1, ci95_low_s='', ci95_high_s=''))
    return rows, rank_rows


def collect(selected):
    values, raw_rows, rank_rows, sources, coverage = [], [], [], [], []
    audit = []
    for receipt in selected:
        meta = {k: receipt[k] for k in ('model', 'hardware', 'provider', 'job_id', 'repeat', 'traced')}
        trees = []
        for name in receipt['profiles'].split(';'):
            path = Path(name)
            parsed = parse_cubex_file(path)
            if not parsed:
                raise ValueError(f'No trees: {path}')
            auxiliary = [path.parent / n for n in ('scorep.cfg', 'run_metadata.json')]
            auxiliary += [Path(receipt['directory']) / 'driver.log']
            recorded = path.parent / 'run_metadata.json'
            model_metadata = json.loads(recorded.read_text()) if recorded.is_file() else {}
            sources.append(dict(**meta, path=name, sha256=sha256(path), tree_count=len(parsed),
                recorded_model=json.dumps(model_metadata, sort_keys=True),
                receipt_hashes=json.dumps({str(p): sha256(p) for p in auxiliary if p.is_file()}, sort_keys=True)))
            trees.extend(parsed)
        config = {'AIX': 'AIx', 'PHYDLL': 'PHYDLL_CPP', 'shared': 'SS_DEFAULT', 'pergpu': 'SS_DEFAULT'}[receipt['provider']]
        raw, extracted = extract(trees, dict(**meta, cubex=receipt['profiles'], config=config), receipt['steady_steps'])
        expected_solver = {'24c1g': 24, '24p24c1g': 48, '96c4g': 96, '96p96c4g': 192}[receipt['hardware']]
        if {r['rank_count'] for r in extracted if r['scope'] == 'solver_steady'} != {expected_solver}:
            raise ValueError(f'Wrong solver rank coverage: {meta}')
        if receipt['provider'] == 'PHYDLL':
            expected_dl = 4 if '4g' in receipt['hardware'] else 1
            if {r['rank_count'] for r in extracted if r['scope'].startswith('dl')} != {expected_dl}:
                raise ValueError(f'Wrong native DL rank coverage: {meta}')
        raw_rows.extend(raw)
        phases, ranks = phase_metrics(trees, receipt['provider'], receipt['steady_steps'])
        if receipt['provider'] in ('shared', 'pergpu'):
            # Independent inclusive-anchor check, not a second use of the ownership map.
            for region, phase in MAPS[receipt['provider']].items():
                anchors = [n for ns in trees for n in ns.values() if n.name == region
                           and 'solver_step_ml_steady' in callpath(ns, n)
                           and 'solver_step_ml_warmup' not in callpath(ns, n) and n.visits]
                row = next(p for p in phases if p['phase'] == phase)
                inclusive = sum(n.incl_time for n in anchors)
                if not math.isclose(inclusive, row['raw_total_s'], rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(f'SmartSim inclusive/owned mismatch: {meta} {region}')
                barriers = [n for ns in trees for n in ns.values() if n.name == 'MPI_Barrier'
                            and region in callpath(ns, n) and 'solver_step_ml_steady' in callpath(ns, n)]
                audit.append(dict(**meta, phase=phase, region=region,
                    inclusive_anchor_total_s=inclusive, exclusive_owned_total_s=row['raw_total_s'],
                    anchor_rank_count=len(anchors), denominator=row['denominator'],
                    inclusive_mean_s=inclusive / row['denominator'], mean_s=row['mean_s'],
                    min_rank_s=min(n.incl_time for n in anchors) / receipt['steady_steps'],
                    max_rank_s=max(n.incl_time for n in anchors) / receipt['steady_steps'],
                    calls_per_rank_step=sum(n.visits for n in anchors) / row['denominator'],
                    nested_barrier_s=sum(n.excl_time for n in barriers),
                    source_paths=receipt['profiles']))
        values.extend(dict(**meta, **p, source_paths=receipt['profiles']) for p in phases)
        rank_rows.extend(dict(**meta, **r) for r in ranks)
        coverage.append(dict(**meta, status='measured host phases', cuda_status='not recorded / not admitted',
                             steady_steps=receipt['steady_steps'], source_kind=receipt['decision']))
        print(f"{receipt['model']} {receipt['hardware']} {receipt['provider']}: {len(trees)} trees, "
              f"{receipt['steady_steps']} steady calls, traced={receipt['traced']}", flush=True)
    write_csv(TABLES / 'smartsim_scope_audit.csv', audit)
    return values, raw_rows, rank_rows, sources, coverage


def render(values, model, hw, ts, *, include_pergpu=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    providers = PROVIDERS if include_pergpu else PROVIDERS[:3]
    labels = LABELS if include_pergpu else ('AIx', 'PhyDLL', 'SmartSim')
    fig, axes = plt.subplots(1, len(providers), figsize=tuple(mm / 25.4 for mm in DIMENSIONS_MM), sharey=True)
    fig.subplots_adjust(left=.11, right=.98, bottom=AXES_BOTTOM, top=AXES_TOP, wspace=.13)
    subset = [r for r in values if r['model'] == model and r['hardware'] == hw]
    maximum = max((r['mean_s'] for r in subset if r['mean_s'] != ''), default=1)
    for ax, provider, label in zip(axes, providers, labels):
        rows = [r for r in subset if r['provider'] == provider]
        for x, phase in enumerate(PHASES[provider]):
            row = next((r for r in rows if r['phase'] == phase), None)
            if row is None or row['mean_s'] == '':
                ax.text(x, .03, 'N/A', transform=ax.get_xaxis_transform(), ha='center', fontsize=5.5)
                continue
            y = row['mean_s']
            ax.bar(x, y, .78, bottom=0, color=COLORS[phase], edgecolor='.3', linewidth=.35)
            ax.annotate(f'{y:.3g}', (x, y), xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', rotation=90, fontsize=5.5)
        ax.set_xlim(-.6, len(PHASES[provider]) - .4)
        ax.set_ylim(0, maximum * 1.4)
        ax.set_xticks([])
        ax.set_xlabel(label, fontsize=8, labelpad=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False, axis='x')
        ax.grid(axis='y', color='.88', linewidth=.4)
        ax.set_axisbelow(True)
        handles = [Patch(facecolor=COLORS[p], label=DISPLAY_PHASES.get(p, p)) for p in PHASES[provider]]
        ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-.03, 1.01),
                  fontsize=5, frameon=False, handlelength=1, handletextpad=.4,
                  borderaxespad=0, borderpad=0, labelspacing=0)
    axes[0].set_ylabel('Phase time (s)', fontsize=9)
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    for ax in axes:
        for text in [*ax.texts, ax.xaxis.label, ax.yaxis.label, *ax.get_legend().get_texts()]:
            box = text.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                raise ValueError(f'Out-of-bounds figure text: {text.get_text()}')
    suffix = '' if include_pergpu else '_without_pergpu'
    ts.save(fig, OUT / f'{model}_{hw}_phases{suffix}')
    plt.close(fig)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, selected = inventory()
    write_csv(TABLES / 'selection_candidates.csv', candidates)
    write_csv(TABLES / 'selected_profiles.csv', selected)
    if args.inventory_only:
        print(json.dumps([dict((k, r[k]) for k in ('model', 'hardware', 'provider', 'job_id', 'traced')) for r in selected], indent=2))
        return
    previous = TABLES / 'plot_values_before_scope_audit.csv'
    if not previous.exists() and (TABLES / 'plot_values.csv').exists():
        write_csv(previous, read_csv(TABLES / 'plot_values.csv'))
    cube_bins = sorted((MINI.parent / 'CPP-ML-Interface/tmp/opencode').glob('*/bin/cube_dump'))
    if not cube_bins:
        raise RuntimeError('No installed cube_dump in tmp/opencode/*/bin')
    os.environ['PATH'] = str(cube_bins[-1].parent) + os.pathsep + os.environ['PATH']
    values, raw, ranks, sources, coverage = collect(selected)
    if previous.exists():
        old = {(r['model'], r['hardware'], r['provider'], r['phase']): r for r in read_csv(previous)}
        write_csv(TABLES / 'scope_audit_numeric_diff.csv', [dict(
            model=r['model'], hardware=r['hardware'], provider=r['provider'], phase=r['phase'],
            old_mean_s=old[(r['model'], r['hardware'], r['provider'], r['phase'])]['mean_s'],
            new_mean_s=r['mean_s'], delta_s=float(r['mean_s']) - float(
                old[(r['model'], r['hardware'], r['provider'], r['phase'])]['mean_s'])) for r in values])
    for name, rows in (('plot_values', values), ('raw_callpaths', raw), ('rank_metrics', ranks),
                       ('profile_sources', sources), ('coverage', coverage)):
        write_csv(TABLES / f'{name}.csv', rows)
    write_csv(TABLES / 'missing_cuda.csv', [dict(model=model, hardware=hw, provider='AIX',
        phase=phase, mean_s='', status='unavailable',
        reason='Host Score-P regions do not measure CUDA-event stage durations')
        for model in MODELS for hw in HARDWARE for phase in ('H2D (CUDA)', 'Forward (CUDA)', 'D2H (CUDA)')])
    # Inspect new collective trace profiles too, but never pool them as repetitions.
    trace_candidates = [r for r in candidates if r['provider'] == 'AIX' and r['traced'] and not r['rejection']]
    if trace_candidates:
        tv, traw, tranks, tsources, tcov = collect(trace_candidates)
        for name, rows in (('trace_profile_values', tv), ('trace_profile_callpaths', traw),
                           ('trace_profile_ranks', tranks), ('trace_profile_sources', tsources)):
            write_csv(TABLES / f'{name}.csv', rows)
    sys.path.insert(0, str(Path.home() / 'Master-Thesis/gitlab/sourcecode'))
    import thesis_style as ts
    ts.apply()
    gallery = []
    verification = []
    for model in MODELS:
        for hw, include_pergpu in ((hw, variant) for hw in HARDWARE for variant in (True, False)):
            fig = render(values, model, hw, ts, include_pergpu=include_pergpu)
            stem = f'{model}_{hw}_phases' + ('' if include_pergpu else '_without_pergpu')
            info = subprocess.check_output(['pdfinfo', str(OUT / (stem + '.pdf'))], text=True)
            fonts = subprocess.check_output(['pdffonts', str(OUT / (stem + '.pdf'))], text=True)
            size = next(line for line in info.splitlines() if line.startswith('Page size:')).split()
            width, height = float(size[2]) * 25.4 / 72, float(size[4]) * 25.4 / 72
            if not math.isclose(width, 150, abs_tol=.01) or not math.isclose(height, 60, abs_tol=.01):
                raise ValueError(f'Wrong physical PDF dimensions: {stem}')
            if 'Type 3' in fonts or not all(' yes ' in line for line in fonts.splitlines()[2:]):
                raise ValueError(f'Nonembedded or Type 3 font: {stem}')
            from PIL import Image
            with Image.open(OUT / (stem + '.png')) as image:
                png_size = image.size
                if any(abs(actual - mm / 25.4 * 300) > 1 for actual, mm in zip(png_size, DIMENSIONS_MM)):
                    raise ValueError(f'Wrong 300 dpi PNG dimensions: {stem}: {png_size}')
            axes_height = fig.axes[0].get_position().height * fig.get_figheight() * 25.4
            if not math.isclose(axes_height, AXES_HEIGHT_MM, abs_tol=1e-9):
                raise ValueError(f'Wrong physical axes height: {stem}')
            verification.append(dict(figure=stem, pdf_width_mm=width, pdf_height_mm=height,
                                      axes_height_mm=axes_height, provider_groups=len(fig.axes),
                                     png_size_px=str(png_size), fonts=fonts, text_bounds='passed', formats='PGF/PDF/PNG'))
            gallery.append(f'## {model}: {hw}\n\n[PDF]({stem}.pdf) | [PGF]({stem}.pgf)\n\n![{stem}]({stem}.png)\n')
    write_csv(TABLES / 'render_verification.csv', verification)
    (OUT / 'README.md').write_text(README + '\n'.join(gallery))
    (TABLES / 'README.md').write_text(README.split('## Gallery')[0])
    artifacts = [p for root in (OUT, TABLES) for p in sorted(root.iterdir()) if p.is_file() and p.name != 'metadata.json']
    metadata = dict(selected_profiles=len(selected), figure_triplets=16, dimensions_mm=list(DIMENSIONS_MM),
                     axes_height_mm=AXES_HEIGHT_MM,
                     geometry_reference='MMCP scripts/plot_cmi_ch6_phases.py: 150x60 mm, bottom=.17, top=.80',
                     previous_terrain_geometry=dict(dimensions_mm=[150, 64], axes_height_mm=64 * (.69 - .16)),
                    measured_values=sum(r['status'] == 'measured' for r in values),
                    missing_regions=sum(r['status'] != 'measured' for r in values),
                    missing_aix_cuda_stage_values=24, supplemental_collective_trace_profiles=len(trace_candidates),
                     cube_dump=str(cube_bins[-1]), source_hashes={str(p): sha256(p) for p in (
                         Path(__file__), MINI / 'extract_scorep_tables.py',
                         MINI.parent / 'CPP-ML-Interface/scripts/analyze_cmi_scorep_profiles.py',
                         MINI / 'test_render_terrain_gpu_phases.py',
                         MINI / 'solver_cpp/terrain_solver.cpp',
                         MINI.parent / 'CPP-ML-Interface/include/provider/ml_coupling_provider_smartsim.hpp',
                         MINI.parent / 'CPP-ML-Interface/extern/SmartRedis/src/cpp/client.cpp',
                         MINI.parent / 'CPP-ML-Interface/extern/SmartRedis/src/cpp/redis.cpp',
                         MINI.parent / 'CPP-ML-Interface/extern/SmartRedis/src/cpp/rediscluster.cpp',
                         MINI.parent / 'CPP-ML-Interface/extern/AIxeleratorService/src/aixeleratorService/aixeleratorService.cpp')},
                    artifact_sha256={str(p): sha256(p) for p in artifacts})
    (TABLES / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k not in ('source_hashes', 'artifact_sha256')}, indent=2))


README = '''# Terrain GPU Baseline Profile Phases

Sixteen independent 150 x 60 mm caption-free model/hardware figures, PGF/PDF/300 dpi
PDF-derived PNG. Adjacent zero-based phase columns, grouped by provider, not stacks.
Exact values and receipts: [profile tables](../tables/profiles/).

Each model/hardware has two versions: `*_phases` has AIx, PhyDLL, SmartSim shared,
and SmartSim per-GPU; `*_phases_without_pergpu` has AIx, PhyDLL, and SmartSim
(the same shared capture). Omitting per-GPU changes no measured value or y limit.
Both versions retain a common y scale across provider groups. Audit tables retain
all 32 captures and all 136 values, including the omitted per-GPU observations.
Each figure is individually authored at the thesis full width of 150 mm; there
are no paired-model panels. No additional half-width provider panels are needed.

Geometry matches the current MMCP `scripts/plot_cmi_ch6_phases.py`: 150 x 60 mm,
bottom .17 and top .80, hence a physical y-axis height of 37.8 mm. The previous
terrain baseline was 150 x 64 mm with bottom .16/top .69, hence only 33.92 mm of
physical y axis. Matching canvas height alone would not match plotting height.
The MMCP script has no tracked HEAD baseline in the inspected repository.

## LaTeX Inclusion

Load `pgf` and `lmodern` in the preamble, then input one figure at its authored
150 mm width, without resizebox, scalebox, or a paired-model subfigure layout:

```latex
\\begin{figure}[tbp]
  \\centering
  \\input{figures/ch6_phases/watercnn8k_24c1g_phases_without_pergpu.pgf}
  \\caption{WaterCNN, 24c1g: measured host phases by provider.}
  \\label{fig:terrain-watercnn-phases}
\\end{figure}
```

The thesis text width must be 150 mm. Substitute `watercnn8k_24c1g_phases.pgf`
for the version including both SmartSim deployments; do not rescale either file.

Coverage: 32/32 baseline cells, 136 measured host-phase columns: 24 AIx,
48 native PhyDLL, 32 SmartSim shared, 32 SmartSim per-GPU. No missing host
phase regions in the selected captures. 31 untraced captures and one trace-backed
WaterCNN 24c1g SmartSim per-GPU capture (job 5006151). Sixteen figure triplets are
48 graphics files. All PDF dimensions, fonts, PNG dimensions and text bounds are
checked by the renderer; render_verification.csv retains receipts.

## Measurement Scope

Scope audit: `smartsim_scope_audit.csv` independently checks inclusive semantic
anchors against descendant-exclusive ownership for every selected SmartSim phase.
It retains rank ranges, calls per rank per step, and nested MPI barrier seconds.
`plot_values_before_scope_audit.csv` preserves the pre-audit values;
`scope_audit_numeric_diff.csv` records every old/new value and numeric delta.
The audit found no numeric accounting correction warranted in these captures.
Figure legends now explicitly identify SmartSim host scope and AIx scatter/wait.
The machine-readable phase keys are unchanged to permit exact before/after joins.
See `scope_audit.md` in the profile tables for the source-level investigation.

One selected artifactual profile capture per cell, not three independent repeats.
No CI or repetition dots. Five steady inference calls follow one warmup in the
12-step captures. The extractor requires solver_step_ml_steady ancestry, excludes
warmup ancestry, checks every solver rank's visit count, and validates native DL
region counts against the steady calls before admission. No lifecycle subtraction.

AIx columns are all-solver-rank host gather/scatter and controller-rank host device
(aix_controller_inference, NOT the all-rank inferenceDevice wrapper). Device host includes GPU work and must
NOT be added to H2D/forward/D2H. CUDA-event H2D/forward/D2H measurements are not
admitted from these host Score-P timers; missing CUDA is explicitly marked, not
invented. PhyDLL is optimized packed, blocking/default, native DL receive/copy,
host H2D, host forward, host D2H, output send, and other measured DL staging.
dl_send_output and its dl_send descendant contribute only once. SmartSim columns
are solver-side put, synchronous model request, retrieval/unpack, chunk planning;
none are CUDA forward. These are different provider scopes, NOT a common critical
path decomposition or uninstrumented production latency.

CUDA availability statements and missing_cuda.csv describe this unchanged host
extractor's admission policy, not a fresh finding that raw CUDA stage evidence
is absent. AIx five-stage extraction is under separate investigation; this
rendering-only change neither resolves that investigation nor changes extraction.

Each callpath's EXCLUSIVE seconds belongs once to its deepest semantic ancestor.
For each phase: sum owned seconds / (active scope ranks * steady calls). Rank
indices and visit counts are exported; device-only controller work is not diluted
over worker ranks. There is no GPU-count multiplication, independent phase maximum,
or global makespan claim. SmartSim put may have two visits per inference call;
normalization remains per steady call, NOT per put visit. Missing regions have blank values, never zero bars.
The raw extractor's all-solver-rank normalized metrics remain in raw_callpaths.csv
for audit, distinct from these explicitly scoped displayed phase metrics.
The four new WaterCNN collective trace profiles are parsed and validated separately
in trace_profile_*.csv, not pooled with the preferred untraced captures. The two
new P2P traces are outside this baseline-only comparison. All 24 AIx CUDA stage
values (8 cells x H2D/forward/D2H) remain unavailable, not zero.

## Selection And Provenance

Only baseline default-layout AIx collective, optimized PhyDLL default, SmartSim
shared default and per-GPU are admitted. All 32 identities are investigated. On one GPU,
shared/per-GPU are separate captures of coincident topologies, not independent
deployment effects or replicates. Phase colors encode semantics, not provider
identity; the per-GPU deployment color for other figures is #568775.

Inventory is bounded to gpu_refresh group manifests and the successful pilot22
retry manifests. Successful receipts, matching model/resolution/chunk/batch,
SmartSim TPQ1/intra8/inter1, and default blocking PhyDLL are required. Prefer
untraced profiles; within matching captures use later authorized retry/replacement
job order, never fastest values or mtime. Trace-backed profiles are explicitly
tagged in selected_profiles.csv. Original captures are never rewritten. Manifest,
config, environment and each Cube file carry SHA256 hashes. Environment files are
hashed, not copied wholesale. Model-file names in config can be stale driver
placeholders; runtime model/workload receipts are retained rather than relabelled.

Files: selection_candidates.csv (including rejections), selected_profiles.csv,
profile_sources.csv, raw_callpaths.csv (inclusive/exclusive totals and visits),
rank_metrics.csv, plot_values.csv (all plotted values and normalization), coverage.csv,
missing_cuda.csv (24 explicit unavailable CUDA stage values), render_verification.csv,
metadata.json (source/output hashes). No jobs are submitted or modified.

Reproduce from mini_app with the smartsim_cuda-12 Python and TeX Live 2026 on PATH:
`python3 render_terrain_gpu_phases.py`; tests:
`python3 -m unittest test_render_terrain_gpu_phases`.

## Gallery

'''


if __name__ == '__main__':
    main()
