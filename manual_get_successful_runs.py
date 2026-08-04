import re
from pathlib import Path

import pandas as pd


LOGS_DIR = Path("../mini_app/logs")
MIN_JOB_ID = 1_237_937
MAX_JOBS = 20
OUTPUT_CSV = Path("successful_runs.csv")


def build_run_name(exp_path: str, content: str) -> str:
    is_cmi = "_cpp_interface_" in exp_path

    if not is_cmi:
        return "SMARTSIM"

    match = re.search(r"_cpp_interface_([^_]+)_", exp_path)
    provider = match.group(1).upper() if match else "UNKNOWN"
    name = f"CMI_{provider}"

    if provider == "PHYDLL":
        name += "_PY" if "phydll_dl_client.py" in content else "_CPP"

    return name


def get_gib(pattern: str, content: str) -> float:
    match = re.search(pattern, content)
    return float(match.group(1)) if match else 0.0


def parse_log_file(file_path: Path) -> dict | None:
    job_match = re.fullmatch(r"mini_app_output_(\d+)\.txt", file_path.name)
    if not job_match:
        return None

    job_id = int(job_match.group(1))

    try:
        content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        print(f"Could not read {file_path}: {exc}")
        return None

    path_match = re.search(
        r"Timing and parameters saved to:\s*(.+?)/timing_and_parameters\.txt",
        content,
    )
    if not path_match:
        print(f"Skipping job {job_id}: experiment path not found")
        return None

    exp_path = path_match.group(1).strip()

    solving_match = re.search(
        r"Solving time:\s*([0-9.]+)\s*seconds",
        content,
    )
    solving_time = (
        float(solving_match.group(1))
        if solving_match
        else None
    )

    ib0_match = re.search(
        r"ib0:\s*rx\s*([0-9.]+)\s*GiB.*?"
        r"tx\s*([0-9.]+)\s*GiB",
        content,
    )
    ib0_rx, ib0_tx = (
        (float(ib0_match.group(1)), float(ib0_match.group(2)))
        if ib0_match
        else (0.0, 0.0)
    )

    lo_match = re.search(
        r"lo:\s*rx\s*([0-9.]+)\s*GiB.*?"
        r"tx\s*([0-9.]+)\s*GiB",
        content,
    )
    lo_rx, lo_tx = (
        (float(lo_match.group(1)), float(lo_match.group(2)))
        if lo_match
        else (0.0, 0.0)
    )

    return {
        "JobID": job_id,
        "RunName": build_run_name(exp_path, content),
        "SolvingTime_s": solving_time,
        "ML_Input_GiB": get_gib(
            r"ml_input:\s*([0-9.]+)\s*GiB",
            content,
        ),
        "ML_Output_GiB": get_gib(
            r"ml_output:\s*([0-9.]+)\s*GiB",
            content,
        ),
        "ML_Preload_GiB": get_gib(
            r"ml_preload:\s*([0-9.]+)\s*GiB",
            content,
        ),
        "IB0_RX_GiB": ib0_rx,
        "IB0_TX_GiB": ib0_tx,
        "LO_RX_GiB": lo_rx,
        "LO_TX_GiB": lo_tx,
    }


def get_job_id(file_path: Path) -> int:
    match = re.fullmatch(r"mini_app_output_(\d+)\.txt", file_path.name)
    return int(match.group(1)) if match else -1


eligible_files = []

for output_file in LOGS_DIR.glob("mini_app_output_*.txt"):
    job_id = get_job_id(output_file)

    if job_id <= MIN_JOB_ID:
        continue

    # For mini_app_output_123.txt, this produces:
    # mini_app_output_123.txt.analysis.md
    analysis_file = output_file.with_name(
        f"{output_file.name}.analysis.md"
    )

    if analysis_file.exists():
        continue

    eligible_files.append(output_file)


# Path.glob() does not guarantee order, and filename sorting is lexical,
# so sort explicitly using the numeric job ID.
eligible_files.sort(key=get_job_id)

selected_files = eligible_files[:MAX_JOBS]

data = []

for file_path in selected_files:
    row = parse_log_file(file_path)
    if row is not None:
        data.append(row)

columns = [
    "JobID",
    "RunName",
    "SolvingTime_s",
    "ML_Input_GiB",
    "ML_Output_GiB",
    "ML_Preload_GiB",
    "IB0_RX_GiB",
    "IB0_TX_GiB",
    "LO_RX_GiB",
    "LO_TX_GiB",
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Found {len(eligible_files)} eligible jobs.")
print(f"Processed {len(selected_files)} log files.")
print(f"Extracted {len(df)} rows.")
print(f"CSV written to: {OUTPUT_CSV.resolve()}")

df
