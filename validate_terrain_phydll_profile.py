#!/usr/bin/env python3
"""Require real per-rank solver and native DL steady coverage in merged MPMD Cube."""
import json
import sys
from pathlib import Path

from extract_scorep_tables import extract, parse_cubex_file


def validate(trees, steps, cubex):
    steady = len(range(4, steps + 1, 2))
    solver = [ns for ns in trees if any(n.name == 'solver_step_ml_steady' and n.visits for n in ns.values())]
    dl = [ns for ns in trees if any(n.name == 'dl_torch_forward' and n.visits for n in ns.values())]
    if len(solver) != 96 or len(dl) != 96:
        raise ValueError(f'Expected 96 solver + 96 DL rank trees, got {len(solver)} + {len(dl)}')
    for ns in dl:
        for region in ('dl_recv', 'dl_torch_forward', 'dl_send_output'):
            calls = sum(n.visits for n in ns.values() if n.name == region)
            if calls != steady:
                raise ValueError(f'{region}: {calls} != {steady} steady calls on a DL rank')
    raw, _ = extract(trees, dict(cubex=str(cubex), config='PHYDLL_CPP'), steady_steps=steady)
    for row in raw:
        if (row['scope'] == 'dl_steady_count_validated' and row['region'] == 'MPI_Barrier'
                and 'dl_send_output' in row['callpath'] and row['raw_calls']):
            raise ValueError('Unexpected output-send logging barrier')
    return dict(solver_ranks=96, dl_ranks=96, steady_steps=steady, cubex=str(cubex), status='SUCCESS')


if __name__ == '__main__':
    cubex = Path(sys.argv[1])
    print(json.dumps(validate(parse_cubex_file(cubex), int(sys.argv[2]), cubex), indent=2))
