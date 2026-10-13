"""Opt-in read-only Redis affinity snapshot, executed on the DB allocation host."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


def redis_pid(endpoint):
    host, port = endpoint.rsplit(":", 1)
    with socket.create_connection((host.strip("[]"), int(port)), timeout=2) as connection:
        # Do not apply a remote server's PID to this host's /proc namespace.
        family = connection.family
        with socket.socket(family, socket.SOCK_STREAM) as local_check:
            local_check.bind((connection.getpeername()[0], 0))
        connection.sendall(b"*2\r\n$4\r\nINFO\r\n$6\r\nserver\r\n")
        with connection.makefile("rb") as stream:
            header = stream.readline(256)
            if not header.startswith(b"$"):
                raise ValueError(f"INFO server did not return a RESP bulk string: {header!r}")
            length = int(header[1:].strip())
            if not 0 < length <= 1024 * 1024:
                raise ValueError("Invalid INFO server response length")
            payload = stream.read(length)
            if len(payload) != length or stream.read(2) != b"\r\n":
                raise ValueError("Truncated INFO server response")
    info = dict(line.split(":", 1) for line in payload.decode().splitlines()
                if ":" in line and not line.startswith("#"))
    pid = int(info["process_id"])
    if pid <= 0:
        raise ValueError("Invalid Redis process_id")
    return pid


def process_snapshot(pid, proc_root=Path("/proc")):
    root = Path(proc_root) / str(pid)
    # Field 22 is starttime; comm can itself contain spaces and parentheses.
    start = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
    threads = []
    for task in sorted(root.joinpath("task").iterdir(), key=lambda p: int(p.name)):
        record = {"tid": int(task.name)}
        try:
            status = task.joinpath("status").read_text()
            fields = dict(line.split(":", 1) for line in status.splitlines()
                          if ":" in line)
            record.update(name=fields["Name"].strip(),
                          cpus_allowed=fields["Cpus_allowed"].strip(),
                          cpus_allowed_list=fields["Cpus_allowed_list"].strip(),
                          status=status)
        except (OSError, KeyError) as exc:
            record["error"] = str(exc)
        threads.append(record)
    end = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
    if start != end:
        raise ValueError("PID reused during snapshot")
    return {"pid": pid, "starttime_ticks": int(start), "threads": threads,
            "clock_ticks_per_second": os.sysconf("SC_CLK_TCK")}


def snapshot_endpoints(endpoints, proc_root=Path("/proc")):
    records = []
    for endpoint in endpoints:
        record = {"endpoint": endpoint, "hostname": socket.gethostname(),
                  "pid_source": "Redis INFO server process_id"}
        try:
            pid = redis_pid(endpoint)
            record["pid"] = pid
            record.update(process_snapshot(pid, proc_root))
        except Exception as exc:
            record["error"] = str(exc)
        records.append(record)
    return records


def capture(endpoints, het_group, environment):
    output = None
    result = {"phase": "db_ready", "captured_at_unix": time.time(),
              "endpoints": list(endpoints), "records": [], "state": "started"}
    try:
        output = Path(environment["SMARTSIM_DB_AFFINITY_OUTPUT"])
        output.write_text(json.dumps(result, indent=2) + "\n")
        child_env = {key: value for key, value in environment.items() if not key.startswith("SCOREP_")}
        child_env["PATH"] = ":".join(p for p in child_env.get("PATH", "").split(":")
                                     if "score-p" not in p.lower() and "scorep" not in p.lower())
        for key in ("LD_PRELOAD", "LD_AUDIT"):
            if key in child_env:
                child_env[key] = " ".join(p for p in child_env[key].replace(":", " ").split()
                                         if "scorep" not in p.lower() and "score-p" not in p.lower())
        group = het_group if (environment.get("SLURM_HET_SIZE") or
                              environment.get("SLURM_JOB_NUM_NODES_HET_GROUP_0")) else None
        deadline = time.monotonic() + 20
        nodelist = None
        if group is not None:
            nodelist = environment.get(f"SLURM_NODELIST_HET_GROUP_{group}") or environment.get(f"SLURM_JOB_NODELIST_HET_GROUP_{group}")
            if not nodelist:
                job = environment["SLURM_JOB_ID"].split("+")[0]
                reply = subprocess.run(["scontrol", "show", "job", f"{job}+{group}", "--oneliner"],
                                       env=child_env, capture_output=True, text=True, check=True, timeout=3)
                nodelist = next((item.split("=", 1)[1] for item in reply.stdout.split()
                                 if item.startswith("NodeList=")), None)
        else:
            nodelist = environment.get("SMARTSIM_DB_NODELIST") or environment.get("SLURM_JOB_NODELIST") or environment.get("SLURM_NODELIST")
        if not nodelist:
            raise ValueError("DB allocation nodelist unavailable")
        # One DB node in each authorized small pilot. Never guess from an endpoint IP.
        hosts = subprocess.run(["scontrol", "show", "hostnames", nodelist], env=child_env,
                               capture_output=True, text=True, check=True, timeout=3).stdout.split()
        if len(hosts) != 1:
            raise ValueError(f"Expected one DB allocation host, got {hosts!r}")
        command = ["srun", "--overlap", "--exact", "--nodes=1", "--ntasks=1",
                   "--cpus-per-task=1", "--export=ALL", f"--nodelist={hosts[0]}"]
        if group is not None:
            command.append(f"--het-group={group}")
        command += [sys.executable, str(Path(__file__).resolve()), "--snapshot", *endpoints]
        result["command"] = command
        reply = subprocess.run(command, env=child_env, capture_output=True, text=True,
                               check=True, timeout=max(0.1, deadline - time.monotonic()))
        result["records"] = json.loads(reply.stdout)
        if len(result["records"]) != len(endpoints):
            raise ValueError("Remote snapshot endpoint count mismatch")
        result["stderr"] = reply.stderr
        result["state"] = "complete"
        if any("error" in record or any("error" in thread for thread in record.get("threads", []))
               for record in result["records"]):
            result["state"] = "partial"
            print("Warning: DB affinity snapshot has observation errors", flush=True)
    except Exception as exc:
        result.update(state="error", error=str(exc))
        if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            result["stderr"] = (exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes)
                                else exc.stderr or "")
        print(f"Warning: DB affinity snapshot failed: {exc}", flush=True)
    try:
        if output is not None:
            output.write_text(json.dumps(result, indent=2) + "\n")
    except Exception as exc:
        print(f"Warning: DB affinity snapshot output unavailable: {exc}", flush=True)


def start_capture(endpoints, het_group, environment):
    if environment.get("SMARTSIM_DB_AFFINITY_CAPTURE") != "1":
        return None
    thread = threading.Thread(target=capture, args=(list(endpoints), het_group, dict(environment)),
                              name="db-affinity-observation", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--snapshot":
        raise SystemExit("Usage: smartsim_db_affinity.py --snapshot HOST:PORT [...]")
    print(json.dumps(snapshot_endpoints(sys.argv[2:])))
