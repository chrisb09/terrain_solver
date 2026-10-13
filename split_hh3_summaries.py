#!/usr/bin/env python3
"""Split HH3 wrapper summary logs into per-scenario solver logs.

Each logs/hh3_summary_<jobid>.txt contains the concatenated stdout of several
engine invocations (one per scenario), delimited by:
    === HH3 s<N>_<CONFIG>_<LAYOUT> (...) start <date> ===  ... engine output ...
    === HH3 s<N>_<CONFIG>_<LAYOUT> finished rc=<rc> <date> ===
This script writes each successfully finished scenario block to
    logs/mini_app_output_hethires3_s<N>_<config>_<layout>.txt
so the standard analysis scripts (which expect logs/mini_app_output_<run>.txt)
can be used unchanged.

Job summaries are processed in ascending job-id order and a scenario block is
only written when it finished with rc=0, so later successful reruns supersede
earlier failed/timeout attempts (which are simply skipped).
"""

import re
import sys
from pathlib import Path

LOGS = Path(__file__).parent / "logs"
START_RE = re.compile(r"^=== HH3 (s\d+_\S+) \(")
FINISH_RE = re.compile(r"^=== HH3 (s\d+_\S+) finished rc=(\d+)")


def job_id_of(summary: Path) -> int:
    m = re.search(r"_(\d+)$", summary.stem)
    return int(m.group(1)) if m else 0


def load_job_suite_map(path: Path) -> dict[int, str]:
    """Optional mapping jobid -> suite, from logs/newsuites_jobids.txt.

    The new-suite wrappers write summary tags without the suite prefix (same as
    the legacy suite), so the suite of a summary must be resolved by job id.
    File format (space separated): <tag> <jobid> <config> <layout> <suite> <scen>
    """
    mapping: dict[int, str] = {}
    if not path.exists():
        return mapping
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].isdigit():
            mapping[int(parts[1])] = parts[4]
    return mapping


def main():
    # Only split scenario blocks belonging to these suites (default: the
    # original hethires3 suite). A job->suite map disambiguates suites whose
    # summary tags carry no prefix.
    suites = [a for a in sys.argv[1:] if not a.startswith("--")]
    suites = suites or ["hethires3"]
    job_suite = load_job_suite_map(LOGS / "newsuites_jobids.txt")
    written = 0
    summaries = sorted(LOGS.glob("hh3_summary_*.txt"), key=job_id_of)
    for summary in summaries:
        this_suite = job_suite.get(job_id_of(summary))
        if this_suite is None:
            this_suite = "hethires3" if "hethires3" in suites else None
        if this_suite is None or this_suite not in suites:
            continue
        current = None
        blocks: dict[str, list[str]] = {}
        with summary.open() as stream:
            for line in stream:
                m = START_RE.match(line)
                if m:
                    current = m.group(1)
                    blocks.setdefault(current, [])
                    continue
                m = FINISH_RE.match(line)
                if m and current is not None:
                    tag, rc = m.group(1), int(m.group(2))
                    if rc == 0:
                        out = LOGS / f"mini_app_output_{this_suite}_{tag}.txt"
                        out.write_text("".join(blocks.get(current, [])))
                        print(f"{summary.name}: {tag} rc={rc} -> {out.name} "
                              f"({len(blocks.get(current, []))} lines)")
                        written += 1
                    else:
                        print(f"{summary.name}: {tag} rc={rc} (skipped)")
                    current = None
                    continue
                if current is not None:
                    blocks.setdefault(current, []).append(line)
    print(f"Wrote {written} per-scenario logs")


if __name__ == "__main__":
    main()
