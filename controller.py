#!/usr/bin/env python3
"""Sweep controller: submits wandb-agent jobs to SLURM via sbatch.

Each job runs runner.sh, which executes one run of a wandb sweep. At most
max_concurrent_runs jobs are queued at once. Nodes on which recent jobs failed
for node reasons (NODE_FAIL, BOOT_FAIL, or FAILED within fail_max_elapsed
seconds) are excluded automatically. Edit CONFIG or set its environment
variables; sbatch forwards the environment to runner.sh.

Usage:
    python controller.py --sweep <sweep_id>=25 [--sweep <id2>=50] [--dry-run]
    python controller.py --sweeps-file my_sweeps.txt  # "<sweep_id> <runs>" lines
    python controller.py --check-only                 # report bad nodes only

Bare sweep ids are prefixed with $WANDB_ENTITY/$WANDB_PROJECT.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {
    # Prepended to bare sweep ids.
    "wandb_entity": os.environ.get("WANDB_ENTITY", ""),
    "wandb_project": os.environ.get("WANDB_PROJECT", "kv-compaction"),
    "job_name": os.environ.get("SWEEP_JOB_NAME", "kv-compaction"),
    "max_time": os.environ.get("SLURM_TIME", "1-16:00:00"),  # [D-]HH:MM:SS
    "partitions": os.environ.get("SLURM_PARTITION", "gpu").split(","),
    "node_subset": os.environ.get("SLURM_NODELIST", ""),  # --nodelist if set
    "exclude_nodes": os.environ.get("SLURM_EXCLUDE", ""),  # always excluded
    "n_gpus": 2,  # 2: vLLM on the second GPU, HF model on the first; 1: shared
    "n_cpus": 16,
    "mem": "200000M",
    # SLURM stdout/stderr, relative to this script.
    "slurm_log_dir": os.environ.get("SLURM_LOG_DIR", "slurm_logs"),
    "max_concurrent_runs": 20,
    "submit_all_at_once": False,
    "sleep_between_runs": 0,
    "conda_env": os.environ.get("CONDA_ENV", "compaction"),
    "runner": "runner.sh",
    "sacct_lookback": "now-5days",  # how far back to look for failed jobs
    # A FAILED job that died within this many seconds counts as a node failure.
    "fail_max_elapsed": 180,
    # More flagged nodes than this means a systematic failure: exclude none.
    "max_auto_exclude": 8,
}

NODE_FAIL_STATES = ("NODE_FAIL", "BOOT_FAIL")


def run(cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def expand_hostlist(expr):
    """Expand a SLURM hostlist expression like node[001-004,106] to names."""
    if not expr or expr.startswith("None"):
        return []
    return run(["scontrol", "show", "hostnames", expr]).stdout.split()


def parse_elapsed(s):
    """Elapsed as seconds, from [D-]HH:MM:SS."""
    days, rest = (s.split("-", 1) + [None])[:2] if "-" in s else (0, s)
    parts = [int(p) for p in str(rest).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return int(days) * 86400 + h * 3600 + m * 60 + sec


def node_boot_time(node):
    """The node's BootTime as an ISO timestamp string, or None."""
    res = run(["scontrol", "show", "node", node], check=False)
    m = re.search(r"BootTime=(\d\S+)", res.stdout)
    return m.group(1) if m else None


def bad_nodes(job_name, lookback, fail_max_elapsed):
    """Nodes on which recent jobs with this name failed for node reasons.

    Nodes rebooted after their last failure are dropped. Also returns the set
    flagged before that step.
    """
    res = run([
        "sacct", "--name", job_name, "-X", "--noheader", "--parsable2",
        "--starttime", lookback,
        "--format", "JobID,State,ExitCode,NodeList,Elapsed,End",
    ])
    nodes, last_fail = {}, {}
    for line in res.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6:
            continue
        jobid, state, exitcode, nodelist, elapsed, end = fields[:6]
        node_reason = state.startswith(NODE_FAIL_STATES) or (
            state.startswith("FAILED")
            and parse_elapsed(elapsed) <= fail_max_elapsed
        )
        if node_reason:
            for node in expand_hostlist(nodelist):
                nodes.setdefault(node, []).append(f"{jobid} {state} {exitcode}")
                last_fail[node] = max(last_fail.get(node, ""), end)
    raw_flagged = set(nodes)
    for node in list(nodes):
        boot = node_boot_time(node)
        if boot and boot > last_fail[node]:  # ISO strings; "Unknown" end keeps the node
            print(f"[controller] not excluding {node}: rebooted {boot}, "
                  f"after its last failure {last_fail[node]}", flush=True)
            del nodes[node]
    return nodes, raw_flagged


def load_sweeps_file(path):
    """Parse "<sweep_id> <run count>" lines; # comments and blanks ignored."""
    sweeps = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            sweep_id, count = line.rsplit(maxsplit=1)
            sweeps[sweep_id] = int(count)
        except ValueError:
            sys.exit(f"{path}:{lineno}: expected '<sweep_id> <run count>', "
                     f"got: {raw.strip()!r}")
    return sweeps


def active_job_count(job_name, partition_list):
    res = run([
        "squeue", "--me", "--name", job_name, "-p", partition_list,
        "-t", "RUNNING,PENDING,CONFIGURING", "--noheader", "-o", "%i",
    ])
    return len(res.stdout.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="append", metavar="ID=COUNT",
                    help="sweep id and run count (repeatable); COUNT defaults "
                         "to --num")
    ap.add_argument("--sweeps-file", help="file with '<sweep_id> <run count>' "
                    "lines, used instead of --sweep")
    ap.add_argument("--job-name")
    ap.add_argument("--num", type=int, default=25,
                    help="run count for --sweep without =COUNT (default 25)")
    ap.add_argument("--gpus", type=int, metavar="N",
                    help="GPUs per job (default: n_gpus in CONFIG)")
    ap.add_argument("--since", metavar="TIME",
                    help="only consider failures after this sacct time "
                         "(e.g. now-6hours); overrides sacct_lookback")
    ap.add_argument("--check-only", action="store_true",
                    help="report the node-failure scan and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print sbatch commands instead of submitting")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.gpus is not None:
        cfg["n_gpus"] = args.gpus
    if args.job_name:
        cfg["job_name"] = args.job_name
    if args.since:
        cfg["sacct_lookback"] = args.since

    sweeps = {}
    if args.sweep:
        for spec in args.sweep:
            sweep_id, _, count = spec.partition("=")
            sweeps[sweep_id] = int(count) if count else args.num
    elif args.sweeps_file:
        sweeps = load_sweeps_file(Path(args.sweeps_file))
    elif not args.check_only:
        ap.error("give --sweep ID=COUNT or --sweeps-file")

    def full_sweep_id(s):
        s = s.removeprefix("wandb agent ")
        if "/" in s:
            return s
        if not cfg["wandb_entity"]:
            sys.exit(f"[controller] bare sweep id {s!r}: set WANDB_ENTITY "
                     f"or pass entity/project/{s}")
        return f"{cfg['wandb_entity']}/{cfg['wandb_project']}/{s}"

    sweeps = {full_sweep_id(s): n for s, n in sweeps.items()}
    partition_list = ",".join(cfg["partitions"])
    here = Path(__file__).resolve().parent
    runner = here / cfg["runner"]
    slurm_log_dir = here / cfg["slurm_log_dir"]
    static_exclude = set(expand_hostlist(cfg["exclude_nodes"]))

    def scan():
        found, raw_flagged = bad_nodes(cfg["job_name"], cfg["sacct_lookback"],
                                       cfg["fail_max_elapsed"])
        # Failures on many nodes point to a broken sweep, not bad hardware.
        # Judged before reboot forgiveness so reboots do not hide the pattern.
        if len(raw_flagged - static_exclude) > cfg["max_auto_exclude"]:
            if not scan.wave_warned:
                scan.wave_warned = True
                print(f"[controller] WARNING: fast failures on "
                      f"{len(raw_flagged)} nodes "
                      f"({', '.join(sorted(raw_flagged))}); "
                      f"treating as a systematic failure, excluding none. "
                      f"Check the jobs' logs; --since <time> skips old "
                      f"failures.", flush=True)
            return set()
        for node, jobs in sorted(found.items()):
            if node not in scan.reported:
                scan.reported.add(node)
                print(f"[controller] excluding {node}: {len(jobs)} node-reason "
                      f"failure(s), e.g. {jobs[-1]}", flush=True)
        return set(found)
    scan.reported = set(static_exclude)
    scan.wave_warned = False

    auto_exclude = scan()
    if args.check_only:
        if not auto_exclude:
            print(f"[controller] no nodes to auto-exclude for job name "
                  f"{cfg['job_name']!r} since {cfg['sacct_lookback']}")
        return

    if not args.dry_run:
        slurm_log_dir.mkdir(parents=True, exist_ok=True)
    last_scan = time.monotonic()
    for sweep_id, num_runs in sweeps.items():
        print(f"Starting sweep {sweep_id} ({num_runs} runs).", flush=True)
        for _ in range(num_runs):
            if not cfg["submit_all_at_once"]:
                while active_job_count(cfg["job_name"], partition_list) \
                        >= cfg["max_concurrent_runs"]:
                    time.sleep(5)
            # Refresh the bad-node list at most once a minute.
            if time.monotonic() - last_scan > 60:
                auto_exclude |= scan()
                last_scan = time.monotonic()

            exclude = ",".join(
                x for x in [cfg["exclude_nodes"],
                            *sorted(auto_exclude - static_exclude)] if x)
            cmd = [
                "sbatch", "-p", partition_list,
                f"--gres=gpu:{cfg['n_gpus']}",
                f"--mem={cfg['mem']}",
                f"--cpus-per-task={cfg['n_cpus']}",
                f"--comment={sweep_id}",
                f"--time={cfg['max_time']}",
                f"--job-name={cfg['job_name']}",
                f"--output={slurm_log_dir}/%j.out",
            ]
            if cfg["node_subset"]:
                cmd.append(f"--nodelist={cfg['node_subset']}")
            if exclude:
                cmd.append(f"--exclude={exclude}")
            cmd += [str(runner), str(here), sweep_id, cfg["conda_env"]]

            if args.dry_run:
                print(shlex.join(cmd))
            else:
                res = run(cmd, check=False)
                print((res.stdout + res.stderr).strip(), flush=True)
                if res.returncode != 0:
                    sys.exit("[controller] sbatch failed, stopping")
            time.sleep(cfg["sleep_between_runs"])


if __name__ == "__main__":
    main()
