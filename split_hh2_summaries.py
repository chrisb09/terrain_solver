#!/usr/bin/env python3
"""Split HH2 wrapper summary logs into per-scenario solver logs.

Each logs/hh2_summary_<jobid>.txt contains the concatenated stdout of several
engine invocations (one per scenario), delimited by:
    === HH2 s<N> (...) start <date> ===      ... engine output ... 
    === HH2 s<N>_<CONFIG_TAG> finished rc=<rc> <date> ===
This script writes each scenario's block to
    logs/mini_app_output_hethires2_s<N>_<config>.txt
so the standard analysis scripts (which expect logs/mini_app_output_<run>.txt)
can be used unchanged.
"""

import re
from pathlib import Path

LOGS = Path(__file__).parent / "logs"
START_RE = re.compile(r"^=== HH2 s(\d+) \(")
FINISH_RE = re.compile(r"^=== HH2 s(\d+)_(\S+) finished rc=(\d+)")


def main():
    written = 0
    for summary in sorted(LOGS.glob("hh2_summary_*.txt")):
        blocks: dict[int, list[str]] = {}
        current = None
        with summary.open() as stream:
            for line in stream:
                m = START_RE.match(line)
                if m:
                    current = int(m.group(1))
                    blocks.setdefault(current, [])
                    continue
                m = FINISH_RE.match(line)
                if m and current is not None:
                    sid, cfg, rc = int(m.group(1)), m.group(2), int(m.group(3))
                    out = LOGS / f"mini_app_output_hethires2_s{sid}_{cfg}.txt"
                    out.write_text("".join(blocks.get(sid, [])))
                    print(f"{summary.name}: s{sid}_{cfg} rc={rc} -> {out.name} "
                          f"({len(blocks.get(sid, []))} lines)")
                    written += 1
                    current = None
                    continue
                if current is not None:
                    blocks.setdefault(current, []).append(line)
    print(f"Wrote {written} per-scenario logs")


if __name__ == "__main__":
    main()
