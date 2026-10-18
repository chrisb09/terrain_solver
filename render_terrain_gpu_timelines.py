#!/usr/bin/env python3
"""Read-only WaterCNN trace gallery; run with the smartsim_cuda-12 Python."""
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

import export_trace_intervals as otf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/thesis_gpu_figures'
MMCP = ROOT.parent.parent / 'MMCP_2026_Artifact_Hybrid_Inference'
RENDERER = ROOT.parent / 'CPP-ML-Interface/scripts/plot_p2p_timeline.py'
CAMPAIGNS = ('gpu_refresh_20261014_thesis_trace_val', 'aix_watercnn_trace_completion_20261016_r1')
HARDWARE = ('24c1g', '24p24c1g', '96c4g', '96p96c4g')
REGIONS = ('gatherInputData', 'inferenceDevice', 'scatterOutputData')
REF_LABELS = ('Collective gather end', 'Collective inference end', 'Collective scatter end')
REF_COLORS = ('#23764d', '#67508e', '#ad592b')
COLORS = ('#4c9ed9', '#8e6bbd', '#f4a261')


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path, ts):
    # MMCP imports select Agg for text-bound layout. Explicitly keep PGF/pdflatex
    # for exports, rather than silently using Agg's native PDF font embedding.
    savefig = fig.savefig
    with patch.object(fig, 'savefig', side_effect=lambda *a, **kw: savefig(*a, backend='pgf', **kw)):
        ts.save(fig, path)
    for suffix in ('.pgf', '.pdf', '.png'):
        if not path.with_suffix(suffix).is_file():
            raise RuntimeError(f'Missing figure export: {path}{suffix}')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unique(rows, label):
    if len(rows) != 1:
        raise ValueError(f'{label}: expected one endpoint, got {len(rows)}')
    return rows[0]


def endpoint(rows, event, peer=None):
    return float(unique([r for r in rows if r['event'] == event and
                        (peer is None or int(r['peer_workgroup_rank']) == peer)], event)['time_s'])


def mapping(text):
    """MPI split order is logged explicitly; no modulo membership fallback."""
    groups, ranks, worlds = defaultdict(list), set(), set()
    for world, size, group, order in re.findall(r'Rank (\d+)/(\d+) will be in group (\d+) order (\d+)', text):
        world, size, group, order = map(int, (world, size, group, order))
        if world in ranks:
            raise ValueError('Duplicate logged rank membership')
        ranks.add(world)
        worlds.add(size)
        groups[group].append((order, world))
    if len(worlds) != 1 or ranks != set(range(next(iter(worlds), 0))):
        raise ValueError('Incomplete runtime world mapping')
    controllers = set(map(int, re.findall(r'Rank (\d+)/\d+ on its local machine is \d+/\d+ is the GPU controller!', text)))
    result = {}
    for group, pairs in sorted(groups.items()):
        if len({p[0] for p in pairs}) != len(pairs):
            raise ValueError('Duplicate split order')
        members = [w for _, w in sorted(pairs)]
        if members[0] not in controllers or set(members) & controllers != {members[0]}:
            raise ValueError('Controller/split mapping mismatch')
        result[group] = members
    if len(controllers) != len(result):
        raise ValueError('Incomplete controllers')
    gpu = {int(w): int(g) for w, _, g in re.findall(r'Rank (\d+) of (\d+) uses GPU index (\d+)', text)}
    if any(m[0] not in gpu or gpu[m[0]] != group for group, m in result.items()):
        raise ValueError('Missing/inconsistent controller GPU index')
    return result


def pipeline(stages, n):
    if len(stages) != 5 or n < 1 or any(v < 0 for v in stages):
        raise ValueError('Expected five measured nonnegative stages and WG size')
    return sum(stages) / n + (n - 1) / n * max(stages)


def pair_ranges(rows, family):
    starts = [r for r in rows if r['event'] == family + '_start']
    ends = [r for r in rows if r['event'] == family + '_end']
    key = lambda r: tuple(int(r[k]) for k in ('range_first_rank', 'range_end_rank', 'sample_start', 'sample_count'))
    if not starts or len(starts) != len(ends) or len({key(r) for r in starts}) != len(starts):
        raise ValueError(f'Incomplete/duplicate {family}')
    result = []
    for start in sorted(starts, key=lambda r: float(r['time_s'])):
        end = unique([r for r in ends if key(r) == key(start)], family)
        a, b = float(start['time_s']), float(end['time_s'])
        if b < a:
            raise ValueError('Negative same-clock interval')
        result.append((a, b, key(start)))
    return result


def controller_metrics(rows, n):
    """All durations use one controller clock; validate every peer and range."""
    for event, peers in (('input_credit_send', range(1, n)), ('input_ready', range(n)),
                         ('result_send_start', range(n)), ('result_send_complete', range(1, n))):
        if sorted(int(r['peer_workgroup_rank']) for r in rows if r['event'] == event) != list(peers):
            raise ValueError(f'Missing/duplicate {event} peers')
    start = endpoint(rows, 'input_listen_start')
    end = max(endpoint(rows, 'result_send_complete', p) for p in range(1, n))
    if not endpoint(rows, 'ml_step_start') <= start <= end <= endpoint(rows, 'ml_step_end'):
        raise ValueError('Invalid exchange span')
    for peer in range(n):
        times = [start]
        if peer:
            times.append(endpoint(rows, 'input_credit_send', peer))
        times += [endpoint(rows, 'input_ready', peer), endpoint(rows, 'result_send_start', peer)]
        if peer:
            times.append(endpoint(rows, 'result_send_complete', peer))
        times.append(end)
        if times != sorted(times):
            raise ValueError('Invalid controller causal order')
    ranges = pair_ranges(rows, 'range_inference')
    covered = [p for a, b, key in ranges for p in range(key[0], key[1])]
    if sorted(covered) != list(range(n)) or any(not start <= a <= b <= end for a, b, _ in ranges):
        raise ValueError('Invalid exactly-once ready-range coverage')
    result = dict(exchange_start_s=start, exchange_end_s=end, exchange_ms=(end-start)*1000,
                  full_ml_ms=(endpoint(rows, 'ml_step_end')-endpoint(rows, 'ml_step_start'))*1000,
                  input_start_s=start, input_end_s=max(endpoint(rows, 'input_ready', p) for p in range(n)),
                  inference_start_s=ranges[0][0], inference_end_s=max(b for _, b, _ in ranges),
                  output_start_s=min(endpoint(rows, 'result_send_start', p) for p in range(n)), output_end_s=end,
                  range_requests=len(ranges))
    for family in ('torch_h2d', 'torch_forward', 'torch_d2h'):
        intervals = pair_ranges(rows, family)
        if sorted(k for _, _, k in intervals) != sorted(k for _, _, k in ranges):
            raise ValueError('CUDA/range identity mismatch')
        result[family + '_ms'] = sum(b-a for a, b, _ in intervals)*1000
    return result


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover():
    runs = []
    for campaign in CAMPAIGNS:
        directory = ROOT / 'results/gpu_refresh' / campaign
        for manifest in sorted(directory.glob('*/manifest.csv')):
            for row in read_csv(manifest):
                if row['rc'] != '0' or row['validation_error'] or row['observed_ml_calls'] != '6':
                    raise ValueError(f'Unvalidated run: {row["run_id"]}')
                if (row['model'], row['warm'], row['steady'], row['cap']) != ('watercnn8k', '1', '5', '20000000'):
                    raise ValueError('Wrong workload/schedule')
                row.update(campaign=campaign, manifest=str(manifest.resolve()),
                           run_dir=str((manifest.parent / row['run_id']).resolve()))
                runs.append(row)
    if len(runs) != 12:
        raise ValueError(f'Expected 12 archives, got {len(runs)}')
    selected = {}
    for hw in HARDWARE:
        for variant in ('default', 'p2p1'):
            campaign = CAMPAIGNS[0] if variant == 'p2p1' and hw.startswith('24') else CAMPAIGNS[1]
            selected[hw, variant] = unique([r for r in runs if
                (r['hardware'], r['provider'], r['variant'], r['campaign']) ==
                (hw, 'AIX', variant, campaign)], 'selected cell')
    return runs, selected


def export_archives(runs, out, sources, reuse=False):
    ledger = []
    for run in runs:
        directory = Path(run['run_dir'])
        receipt = directory / 'traces.txt'
        trace = Path(unique(receipt.read_text().splitlines(), 'trace receipt')).resolve()
        output = out / 'tables/traces' / (run['run_id'] + '.csv')
        # Reuse is explicit and only accepts a complete archive fingerprint match.
        files = [trace, trace.parent / 'traces.def', *sorted((trace.parent / 'traces').rglob('*'))]
        fingerprint = {str(p.resolve()): sha(p) for p in files if p.is_file()}
        sources.update(fingerprint)
        for path in (receipt, Path(run['manifest']), trace.parent / 'scorep.cfg'):
            sources[str(path)] = sha(path)
        metadata = output.with_suffix('.json')
        if reuse and output.exists() and metadata.exists() and json.loads(metadata.read_text())['sources'] == fingerprint:
            counts = json.loads(metadata.read_text())['counts']
            if sha(output) != json.loads(metadata.read_text())['csv_sha256']:
                raise ValueError('Changed interval export')
        else:
            print(f'Exporting {run["hardware"]} {run["provider"]} {run["variant"]}', flush=True)
            counts = otf.export_trace(trace, output, run['run_id'], run['provider'], run['variant'], expected_steady=5)
            clock, regions, locations = otf._definitions(otf._stream(trace, definitions=True))
            metadata.write_text(json.dumps(dict(sources=fingerprint, counts=counts, clock=clock,
                regions=regions, locations=locations, csv_sha256=sha(output)), indent=2)+'\n')
        definitions = json.loads(metadata.read_text())
        types = sorted({value[2] for value in definitions['locations'].values()})
        ledger.append(dict(run_id=run['run_id'], hardware=run['hardware'], provider=run['provider'],
            variant=run['variant'], source=str(trace), csv=str(output), **counts,
            clock_ticks_per_second=definitions['clock'][0], location_types=';'.join(types),
            cuda_locations=sum(v[2] != 'CPU_THREAD' for v in definitions['locations'].values()),
            region_names=';'.join(sorted(set(definitions['regions'].values()))),
            scope='inclusive CPU ENTER/LEAVE, not CUDA events',
            deployment_note='1-GPU shared/pergpu equivalent, not independent provider evidence' if run['provider'] == 'SMARTSIM' else ''))
    return ledger


def collect(out, sources, reuse=False):
    runs, selected = discover()
    ledger = export_archives(runs, out, sources, reuse)
    mappings, events, metrics, refs, audit, selection = {}, {}, [], [], [], []
    for (hw, variant), run in selected.items():
        directory = Path(run['run_dir'])
        log = directory / 'driver.log'
        text = log.read_text()
        group_map = mapping(text)
        world = sum(map(len, group_map.values()))
        if world != {'24c1g': 24, '24p24c1g': 48, '96c4g': 96, '96p96c4g': 192}[hw]:
            raise ValueError('Wrong actual runtime world')
        if 'Resolved MODEL_PATH=' not in text or 'watercnn_cuda.pt' not in text or 'nx=7680, nz=4320, chunk_size=60' not in text:
            raise ValueError('Runtime workload/model not verified')
        model = unique(re.findall(r'^Resolved MODEL_PATH=(.+)$', text, re.M), 'runtime model')
        if Path(model).name != 'watercnn_cuda.pt':
            raise ValueError('Wrong resolved model')
        env = json.loads((directory / 'runtime.env.json').read_text())
        if env['ML_BATCH_SIZE_ENV'] != '20000000' or env['MODEL_NAME_ENV'] != 'watercnn':
            raise ValueError('Wrong runtime workload')
        if list(map(int, re.findall(r'^STEP_TIMING step=(\d+) solver=ML\b', text, re.M))) != [2, 4, 6, 8, 10, 12]:
            raise ValueError('Runtime ML steps do not match OTF2 call labels')
        mappings[hw, variant] = group_map
        selection.append(dict(**run, actual_world=world, mapping_source=str(log), resolved_model=model,
                              intra_threads=env.get('MLCOUPLING_INTRA_OP_THREADS', ''),
                              inter_threads=env.get('MLCOUPLING_INTER_OP_THREADS', ''),
                              communication_mode=env.get('AIX_COMMUNICATION_MODE_ENV', ''),
                              credits=env.get('AIX_P2P_INITIAL_CREDITS_ENV', ''),
                              selection_reason='explicit user campaign selection; not fastest run'))
        for path in (log, directory / 'config.toml', directory / 'runtime.env.json', Path(run['manifest']),
                     directory / 'traces.txt'):
            sources[str(path)] = sha(path)
        ldd = directory / 'solver.ldd.txt'
        selection[-1]['binary_resolution_receipt'] = str(ldd) if ldd.exists() else 'unavailable in October 14 campaign'
        if ldd.exists():
            sources[str(ldd)] = sha(ldd)
        identity = dict(hardware=hw, variant=variant, run_id=run['run_id'], job_id=run['job_id'])
        if variant == 'default':
            path = out / 'tables/traces' / (run['run_id'] + '.csv')
            definitions = json.loads(path.with_suffix('.json').read_text())
            if any(v[2] != 'CPU_THREAD' for v in definitions['locations'].values()):
                raise ValueError('Unexpected device locations: inspect CUDA evidence before declaring missing stages')
            wanted = [r for r in read_csv(path) if r['region'] in (*REGIONS, otf.STEADY, otf.WARMUP)
                      and r['thread_id'] == '0']
            for group, members in group_map.items():
                ctrl = members[0]
                for call in range(1, 6):
                    rows = [r for r in wanted if r['world_rank'] == str(ctrl) and r['ml_call_index'] == str(call+1)]
                    wrapper = unique([r for r in rows if r['region'] == otf.STEADY], 'steady wrapper')
                    origin = float(wrapper['start_s'])
                    record = dict(**identity, workgroup=group, controller=ctrl, n_members=len(members),
                                  call=call, otf_call=call+1, solver_step=(call+1)*2,
                                  full_ml_ms=float(wrapper['duration_s'])*1000,
                                  h2d_cuda_ms='', forward_cuda_ms='', d2h_cuda_ms='',
                                  pipeline_prediction_ms='', fullcredit_measured_ms='',
                                  missing='collective CUDA-event stages unavailable; Fullcredit trace unavailable')
                    for phase, region in zip(('input', 'inference', 'output'), REGIONS):
                        row = unique([r for r in rows if r['region'] == region], region)
                        record[phase + '_host_ms'] = float(row['duration_s'])*1000
                        refs.append(dict(**identity, workgroup=group, controller=ctrl, call=call, phase=phase,
                            start_s=float(row['start_s']), end_s=float(row['end_s']), origin_s=origin,
                            start_ms=(float(row['start_s'])-origin)*1000,
                            end_ms=(float(row['end_s'])-origin)*1000,
                            duration_ms=float(row['duration_s'])*1000,
                            scope='OTF2 CPU region, independently zeroed controller solver wrapper', source=str(path)))
                    metrics.append(record)
        else:
            timeline = ROOT / 'logs' / ('aix_p2p_timeline_' + run['run_id'])
            stale = timeline / 'timeline_metadata.json'
            sources[str(stale)] = sha(stale)
            all_rows = []
            for group, members in group_map.items():
                ctrl = members[0]
                by_rank = {}
                for local, world_rank in enumerate(members):
                    path = timeline / f'aix_p2p_timeline_rank_{world_rank}.csv'
                    sources[str(path)] = sha(path)
                    raw = read_csv(path)
                    rows = [{k: v if k == 'event' else float(v) if k == 'time_s' else int(v)
                             for k, v in row.items()} for row in raw]
                    if {r['step'] for r in rows} != set(range(6)) or any(
                        (r['world_rank'], r['workgroup_rank'], r['is_controller']) !=
                        (world_rank, local, int(world_rank == ctrl)) for r in rows):
                        raise ValueError('Timeline rank/call/log membership mismatch')
                    rows.sort(key=lambda r: (r['step'], r['time_s']))
                    by_rank[world_rank] = rows
                    all_rows.extend(rows)
                for call in range(1, 6):
                    controller = [r for r in by_rank[ctrl] if r['step'] == call]
                    values = controller_metrics(controller, len(members))
                    origin = endpoint(controller, 'ml_step_start')
                    metrics.append(dict(**identity, workgroup=group, controller=ctrl, n_members=len(members),
                                        call=call, otf_call=call+1, solver_step=(call+1)*2, origin_s=origin, **values))
                    for local, worker in enumerate(members[1:], 1):
                        rows = [r for r in by_rank[worker] if r['step'] == call]
                        wait, send, complete, received = (endpoint(rows, e) for e in
                            ('input_credit_wait_start', 'input_send_start', 'input_send_complete', 'result_received'))
                        worker_origin = endpoint(rows, 'ml_step_start')
                        if not worker_origin <= wait <= send <= complete <= received <= endpoint(rows, 'ml_step_end'):
                            raise ValueError('Invalid worker same-clock order')
                        credit, ready, post = (endpoint(controller, e, local) for e in
                            ('input_credit_send', 'input_ready', 'result_send_start'))
                        lower = max(credit-send, ready-complete, post-received)
                        upper = ready-send
                        ack = ready-complete
                        audit.append(dict(**identity, workgroup=group, controller=ctrl, worker=worker, local_rank=local,
                            call=call, worker_origin_s=worker_origin, controller_origin_s=origin,
                            worker_wait_s=wait, worker_send_s=send, worker_complete_s=complete, worker_received_s=received,
                            controller_credit_s=credit, controller_ready_s=ready, controller_post_s=post,
                            causal_lower_s=lower, causal_upper_s=upper, causal_feasible=lower <= upper,
                            hypothetical_ack_offset_s=ack, hypothetical_ack_output_ms=(received+ack-post)*1000,
                            applied_offset_s='', clock_policy='rank-local ml_step_start zero; no alignment applied'))
            events[hw] = all_rows
    for hw in HARDWARE:
        if mappings[hw, 'default'] != mappings[hw, 'p2p1']:
            raise ValueError('Collective/P2P memberships do not match')
    return selection, mappings, events, metrics, refs, audit, ledger


def render_collapsed(hw, groups, metrics, refs, out, ts, plt):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(150/25.4, (29+12*len(groups))/25.4))
    ticks, labels = [], []
    for index, (group, members) in enumerate(groups.items()):
        value = unique([r for r in metrics if r['hardware'] == hw and r['variant'] == 'p2p1'
                        and r['call'] == 5 and r['workgroup'] == group], 'collapsed metrics')
        origin = value['origin_s']
        for lane, (phase, label, color) in enumerate(zip(('input', 'inference', 'output'),
                ('Input (controller)', 'Inference (host)', 'Output (controller)'), COLORS)):
            y = index*3.7+lane
            a, b = value[phase+'_start_s'], value[phase+'_end_s']
            ax.broken_barh([((a-origin)*1000, (b-a)*1000)], (y-.28, .56), facecolors=color)
            ticks.append(y)
            labels.append(f'WG {group}  {label}')
        for r, color in zip([unique([r for r in refs if r['hardware'] == hw and r['call'] == 5
                              and r['workgroup'] == group and r['phase'] == phase], phase)
                              for phase in ('input', 'inference', 'output')], REF_COLORS):
            ax.vlines(r['end_ms'], index*3.7-.45, index*3.7+2.45, color=color, ls='--', lw=.8)
        if index:
            ax.axhline(index*3.7-.85, color='.7', lw=.5)
    ax.set_yticks(ticks, labels, fontsize=6)
    ax.set_ylim(ticks[-1]+.6, -.6)
    ax.set_xlabel('Time (ms), independent controller-local zeros', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', alpha=.18)
    fig.legend(handles=[Line2D([], [], color=c, ls='--', label=l) for c, l in zip(REF_COLORS, REF_LABELS)],
               loc='upper center', ncol=3, fontsize=5.5, frameon=False, handlelength=1.2,
               columnspacing=.8, bbox_to_anchor=(.60, .99))
    fig.subplots_adjust(left=.25, right=.985, top=1-7/(29+12*len(groups)), bottom=11/(29+12*len(groups)))
    name = f'watercnn8k_{hw}_default_p2p1_call5_controller_windows'
    save_figure(fig, out / name, ts)
    plt.close(fig)
    return name


def render_detail(hw, group, members, rows, refs, out, ts, plt, renderer, stagger):
    """Reuse actual render_step; remove every cross-clock primitive at the adapter."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    selected = [dict(r) for r in rows if r['step'] == 5 and r['world_rank'] in members]
    ctrl = members[0]
    origins = {w: endpoint([r for r in selected if r['world_rank'] == w], 'ml_step_start') for w in members}
    for row in selected:
        row['time_s'] -= origins[row['world_rank']]
    selected.sort(key=lambda r: r['time_s'])
    # These primitives compare controller and worker clocks. Preserve raw data in
    # the audit, but do not ask the original renderer to draw fictitious transfers.
    safe = [dict(r) for r in selected if r['event'] not in ('result_send_start', 'input_credit_send', 'ml_step_end')]
    original_save = Figure.savefig
    name = f'watercnn8k_{hw}_default_p2p1_call5_wg{group}_detail'

    def save(fig, *args, **kwargs):
        height = 27 + 2.5*len(members)
        fig.set_size_inches(150/25.4, height/25.4)
        fig._suptitle.remove()
        ax = fig.axes[0]
        ax.set_ylabel('World rank', fontsize=7, weight='normal')
        ax.set_yticks(range(len(members)), [f'r{w}' + (' (Ctrl)' if w == ctrl else '') for w in members], fontsize=5)
        for label in ax.get_yticklabels():
            label.set_fontsize(5)
            label.set_weight('normal')
            label.set_color('.2')
        # Remove original cross-rank extrema lines, not local event durations.
        for line in list(ax.lines):
            if line.get_linestyle() == '--':
                line.remove()
        for y, world in enumerate(members[1:], 1):
            worker = [r for r in selected if r['world_rank'] == world]
            a, b = (endpoint(worker, e)*1000 for e in ('input_send_complete', 'result_received'))
            ax.broken_barh([(a, b-a)], (y+.04, .27), facecolors='#f4a261')
            ax.plot(b, y+.17, '|', color='#ad592b', ms=3)
        controller = [r for r in selected if r['world_rank'] == ctrl]
        for row in controller:
            if row['event'] == 'input_credit_send':
                ax.plot(row['time_s']*1000, -.35, 'v', color='#c2a000', ms=2)
            if row['event'] == 'result_send_start':
                ax.plot(row['time_s']*1000, .43, '|', color='#ad592b', ms=3)
                peer = row['peer_workgroup_rank']
                if peer:
                    end = endpoint(controller, 'result_send_complete', peer)
                    ax.broken_barh([(row['time_s']*1000, (end-row['time_s'])*1000)],
                                   (.39, .08), facecolors='#f4a261', alpha=.4)
        reference_rows = [unique([r for r in refs if r['hardware'] == hw and r['workgroup'] == group
                and r['call'] == 5 and r['phase'] == phase], phase) for phase in ('input', 'inference', 'output')]
        for row, color in zip(reference_rows, REF_COLORS):
            ax.axvline(row['end_ms'], color=color, ls='--', lw=.65)
        ax.set_xlim(right=max(ax.get_xlim()[1], max(r['end_ms'] for r in reference_rows)*1.025))
        ax.set_xlabel('Time (ms), independent rank-local ML-entry zeros', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        for legend in list(fig.legends):
            legend.remove()
        handles = [Patch(facecolor=c, label=l) for c, l in
                   [('#4c9ed9', 'Input + ACK'), ('#f0c040', 'Credit wait'),
                    ('#f4a261', 'ACK to output receipt'), ('#8e6bbd', 'Host inference'),
                    ('#2a9d8f', 'H2D'), ('#151515', 'Forward'), ('#ff0033', 'D2H'),
                    ('#f4a261', 'Output (controller)')]]
        fig.legend(handles=handles, ncol=4, loc='upper center', frameon=False,
                   fontsize=5.5, handlelength=1, columnspacing=.8, bbox_to_anchor=(.57, .995))
        fig.legend(handles=[Line2D([], [], color=c, ls='--', label=l) for c, l in zip(REF_COLORS, REF_LABELS)],
                   ncol=3, loc='upper center', frameon=False, fontsize=5.5, handlelength=1,
                   columnspacing=.8, bbox_to_anchor=(.57, 1-7/height))
        fig.subplots_adjust(left=.14, right=.985, bottom=11/height, top=1-16/height)
        stagger(ax, 0)
        with patch.object(Figure, 'savefig', original_save):
            save_figure(fig, out / name, ts)

    with patch.object(renderer, 'get_workgroup_mapping', return_value={ctrl: members}), \
         patch.object(renderer, 'synchronize_step_clocks', side_effect=lambda rows, _: rows), \
         patch.object(Figure, 'savefig', save):
        renderer.render_step(safe, 5, out, 'WaterCNN')
    return name


NOTES = """# WaterCNN Default AIx Timeline Gallery

One warmup (diagnostic call 0), five steady calls (1..5); OTF2 calls 2..6,
solver steps 4,6,8,10,12. Call 5 is a representative last steady call, not
an independent repetition. No rank/call/GPU confidence intervals are estimated.
The selection ledger gives job IDs, manifests, actual runtime worlds and receipts.
24c/24p P2P uses the October 14 validation campaign; 96c/96p P2P and all collective
references use the October 16 completion campaign. Memberships match exactly.

## Clock And Scope

The actual terrain render_step is reused with narrow mapping/drawing overrides.
Every rank independently subtracts its own ml_step_start. The original ACK clock
correction and global-extrema lines are disabled. No cross-rank concurrency or
worker-to-controller transfer latency is implied. Worker blue bars are local
input_send_start -> input_send_complete (includes ACK); yellow bars are credit
waits; orange bars are local input_send_complete -> result_received, including
inference and waiting, NOT output transfer time. Controller credit/post ticks
stay on the controller lane. Controller ready ranges and host-anchored CUDA-event
intervals retain their logged ordering after timestamp sorting, not CSV row order.
Ready-range labels are staggered using measured text bounds from the MMCP adapter.
Per-WG detail is authored at 150 mm with 5 pt ranks; 192 ranks are never squeezed
onto one purportedly readable page. Use individual WG PDFs for detail.

Collapsed figures use ONE axes for all WGs with dividers and common scale.
To avoid unmeasured alignment, these are explicitly CONTROLLER windows:
input_listen_start -> last input_ready, first -> last range_inference,
first result_send_start (including local rank) -> last result_send_complete.
All include gaps. They are not the requested cross-clock worker-send/output-receipt
envelopes: those endpoints cannot be combined without estimating offsets.
The separate causal audit retains raw peer endpoints, necessary offset bounds
L=max(credit-send, ready-ACK, post-receipt), U=ready-send, feasibility, and signed
hypothetical ACK-equality output spans. No offset is applied or duration clamped.
ACK completion and controller input-ready are causally distinct, not calibration.

Collective dashed boundaries are verified OTF2 CPU region EXIT timestamps relative
to that controller's own steady solver wrapper entry, not concatenated durations.
Thus gaps and scope are preserved. Their labels are Collective gather end,
Collective inference end, Collective scatter end. Different runs have independent
zeros; these are references, not global alignment. OTF2 frequency/global offset,
location->MPI-rank mapping and wrapper membership are verified by the existing
export_trace_intervals module (expected_steady=5); full archives are exported.
Its global-offset coordinates do not calibrate diagnostic MPI_Wtime clocks.

## Missing Data

The collective archives export inclusive CPU ENTER/LEAVE intervals, not CUDA-event
H2D/forward/D2H timings. inferenceDevice is host-inclusive and cannot substitute
for CUDA forward. No verified five-stage CUDA vector exists in these collective
captures, so no paired five-stage/prediction figure or fabricated prediction is
rendered. P2P CUDA stage sums are separately tagged, never borrowed as collective.
Fullcredit is absent from these selected campaigns; fields are blank, not zero.
The pipeline-spans folder reports these gaps and preserves all five call values.
Timeline metadata's stale model_path/ranks/warmup_steps are not authoritative:
runtime Resolved MODEL_PATH, grid world_size, logged split order/controller/GPU
mapping, manifest schedule and actual diagnostic membership are checked instead.
Heterogeneous p placements use GPU-hidden CPU-surrogate nodes as logged; these
are actual 48/192 solver worlds, not the launcher-side nominal 24/96 counts.
All data are profile-derived, not native timing repetitions or capacity samples.
In workgroup_calls, full_ml_ms is the collective solver wrapper duration or the
P2P diagnostic AIx function duration according to variant. These scopes are not
silently compared. call_values.csv labels the collective solver-wrapper maximum
and P2P controller-exchange maximum separately. Neither is a global makespan.
Controller output detail bars run from each remote result post to the completion
marker after output MPI_Waitall, not to a worker receipt. The hashed available
communicationStrategy/collectiveCommunication.cpp supports these event meanings;
source inspection alone does not independently prove the exact historical binary.
October 14 has no solver.ldd receipt; October 16 resolution receipts are retained.

## Audit Files

[Selection](../tables/timeline_selection.csv), [memberships](../tables/timeline_membership.csv),
[all WG calls](../tables/timeline_workgroup_calls.csv),
[collective intervals](../tables/timeline_collective_intervals.csv),
[clock audit](../tables/timeline_clock_audit.csv),
[common coverage](../tables/timeline_common_coverage.csv),
[archive exports](../tables/traces/README.md), and [source hashes](metadata.json).
[Pipeline coverage and call maxima](../aix_pipeline_spans/README.md) explicitly
retain missing CUDA stages, predictions and Fullcredit.

## Reproduce

Use smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 with
~/opt/texlive/2026/bin/x86_64-linux on PATH. Run
`python mini_app/render_terrain_gpu_timelines.py` from the workspace root.
`--reuse-exports` reuses only SHA256-validated complete interval exports.
No jobs or existing inputs are edited. All figures use thesis_style.save for
PGF, self-contained PDF and PDF-derived 300 dpi PNG; no explanatory footers.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-exports', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args(argv)
    for directory in ('aix_timelines', 'aix_pipeline_spans', 'tables/traces'):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OTF2_PRINT', str(ROOT.parent / 'CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/otf2-print'))
    sources = {}
    selection, mappings, events, metrics, refs, audit, ledger = collect(OUT, sources, args.reuse_exports)
    for name, rows in (('timeline_selection', selection), ('timeline_workgroup_calls', metrics),
                       ('timeline_collective_intervals', refs), ('timeline_clock_audit', audit),
                       ('timeline_trace_exports', ledger)):
        write_csv(OUT / 'tables' / (name + '.csv'), rows)
    write_csv(OUT / 'tables/timeline_membership.csv', [dict(hardware=hw, variant=mode, workgroup=group,
              controller=members[0], gpu_index=group, local_rank=local, world_rank=world)
              for (hw, mode), groups in mappings.items() for group, members in groups.items()
              for local, world in enumerate(members)])
    write_csv(OUT / 'tables/timeline_common_coverage.csv', [dict(hardware=hw, provider=provider, variant=variant,
        status='exported' if any((r['hardware'], r['provider'], r['variant']) == (hw, provider, variant)
                                for r in ledger) else 'unavailable in selected campaigns',
        archive_count=sum((r['hardware'], r['provider'], r['variant']) == (hw, provider, variant) for r in ledger))
        for hw in HARDWARE for provider, variant in (('AIX', 'default'), ('AIX', 'p2p1'), ('AIX', 'p2pfull'),
                   ('PHYDLL', 'default'), ('SMARTSIM', 'shared'), ('SMARTSIM', 'pergpu'))])
    call_values = []
    for hw in HARDWARE:
        for call in range(1, 6):
            collective = [r for r in metrics if r['hardware'] == hw and r['variant'] == 'default' and r['call'] == call]
            p2p = [r for r in metrics if r['hardware'] == hw and r['variant'] == 'p2p1' and r['call'] == call]
            critical = max(p2p, key=lambda r: r['exchange_ms'])
            call_values.append(dict(hardware=hw, call=call, solver_step=(call+1)*2,
                p2p_controller_exchange_ms=critical['exchange_ms'], p2p_critical_wg=critical['workgroup'],
                collective_max_local_solver_wrapper_ms=max(r['full_ml_ms'] for r in collective),
                collective_five_stage_sum_ms='', pipeline_prediction_ms='', fullcredit_exchange_ms='',
                independent_repetitions=1, confidence_interval='', scope='max of local durations, not global makespan'))
    write_csv(OUT / 'aix_pipeline_spans/call_values.csv', call_values)
    missing = [dict(hardware=hw, collective_cuda_stages='unavailable', fullcredit_trace='unavailable',
                    paired_figure='not rendered: no verified collective five-stage CUDA vector') for hw in HARDWARE]
    write_csv(OUT / 'aix_pipeline_spans/coverage.csv', missing)
    (OUT / 'aix_pipeline_spans/README.md').write_text('# Pipeline Span Coverage\n\n'
        'No paired prediction figures: collective OTF2 CPU regions are not CUDA stages.\n'
        'Fullcredit has no selected trace. Missing values are blank, never inferred.\n'
        'All 100 workgroup-call records (50 collective + 50 P2P) are in '
        '[timeline_workgroup_calls.csv](../tables/timeline_workgroup_calls.csv). '
        'P2P exchange is same-controller input_listen_start -> last result_send_complete; '
        'CUDA sums are P2P-only. Five steady calls belong to one profile repetition; no CI.\n\n'
        '[Call maxima](call_values.csv) | [Missing-data coverage](coverage.csv) | '
        '[Timeline gallery](../aix_timelines/README.md)\n')
    gallery = []
    if not args.export_only:
        sys.path.insert(0, str(Path.home() / 'Master-Thesis/gitlab/sourcecode'))
        import thesis_style as ts
        ts.apply()
        import matplotlib.pyplot as plt
        renderer = load_module(RENDERER, 'terrain_original')
        sys.path.insert(0, str(MMCP / 'scripts'))
        from plot_cmi_aix_terrain_style import stagger_range_labels
        for hw in HARDWARE:
            groups = mappings[hw, 'p2p1']
            gallery.append(render_collapsed(hw, groups, metrics, refs, OUT / 'aix_timelines', ts, plt))
            for group, members in groups.items():
                print(f'Rendering {hw} WG {group} ({len(members)} ranks)', flush=True)
                gallery.append(render_detail(hw, group, members, events[hw], refs, OUT / 'aix_timelines',
                                             ts, plt, renderer, stagger_range_labels))
    for path in (Path(__file__), Path(__file__).with_name('test_render_terrain_gpu_timelines.py'), Path(otf.__file__),
                 RENDERER, MMCP / 'scripts/plot_cmi_aix_terrain_style.py', MMCP / 'scripts/plot_cmi_pipeline_spans.py',
                 Path.home() / 'Master-Thesis/gitlab/sourcecode/thesis_style.py',
                 RENDERER.parent.parent / 'extern/AIxeleratorService/src/communicationStrategy/collectiveCommunication.cpp',
                 RENDERER.parent.parent / 'extern/AIxeleratorService/src/utils/p2pTimeline.h'):
        sources[str(path)] = sha(path)
    metadata = dict(sources=sources, selected_cells=len(selection), trace_archives=len(ledger),
                    workgroup_calls=len(metrics), clock_audit_rows=len(audit), figures=gallery,
                    interval_rows=sum(r['rows'] for r in ledger),
                    interval_labeled_rows=sum(r['labeled_rows'] for r in ledger),
                    selected_aix_archives=sum(r['provider'] == 'AIX' for r in ledger),
                    causal_infeasible=sum(not r['causal_feasible'] for r in audit),
                    hypothetical_ack_negative_output=sum(r['hypothetical_ack_output_ms'] < 0 for r in audit))
    metadata['numeric_sha256'] = {str(p.relative_to(OUT)): sha(p) for p in
        [*sorted((OUT / 'tables').glob('timeline_*.csv')), OUT / 'aix_pipeline_spans/call_values.csv']}
    for path, digest in sources.items():
        if sha(path) != digest:
            raise RuntimeError(f'Source changed during rendering: {path}')
    metadata['source_integrity_verified'] = True
    (OUT / 'aix_timelines/metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    text = NOTES + '\n## Gallery\n'
    for name in gallery:
        text += f'\n### {name}\n[PDF]({name}.pdf) | [PGF]({name}.pgf)\n\n![{name}]({name}.png)\n'
    (OUT / 'aix_timelines/README.md').write_text(text)
    trace_text = ('# Common OTF2 Interval Exports\n\n'
        f'{len(ledger)} complete archives, {metadata["interval_rows"]:,} inclusive CPU intervals '
        f'({metadata["interval_labeled_rows"]:,} assigned to solver calls). '
        'All archives validate one leading warmup and five steady wrappers per solver location. '
        'Counts of warmup_calls/steady_calls in the ledger are location-call counts, not repetitions. '
        'Each CSV uses the unchanged export_trace_intervals.FIELDS schema; each adjacent JSON '
        'contains exact clock properties, location/rank/thread definitions, region names, full '
        'archive SHA256 fingerprints and CSV SHA256. Unknown call membership remains blank.\n\n'
        'Eight AIx cells cover collective/default and P2P Credit=1 at all four placements. '
        'Four supplemental archives cover PhyDLL default at 24c/24p and SmartSim shared/pergpu '
        'at 24c only. One-GPU SmartSim topologies coincide and are not independent deployment '
        'evidence. Missing configurations are N/A, not zero. No Fullcredit archive is selected.\n\n'
        '[Coverage matrix](../timeline_common_coverage.csv) | [Export ledger](../timeline_trace_exports.csv)\n\n'
        '| Hardware | Provider | Variant | Intervals | CSV | Definitions / hashes |\n'
        '|---|---|---|---:|---|---|\n')
    for row in ledger:
        name = Path(row['csv']).name
        trace_text += f'| {row["hardware"]} | {row["provider"]} | {row["variant"]} | {row["rows"]} | [CSV]({name}) | [JSON]({Path(name).with_suffix(".json")}) |\n'
    (OUT / 'tables/traces/README.md').write_text(trace_text)
    print(json.dumps({k: v for k, v in metadata.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
