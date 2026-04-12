#!/usr/bin/env python3
"""Sweep the P2P benchmark over compile-time and runtime parameters.

The script reconfigures the existing CMake build for each constexpr tuple,
rebuilds the `p2p` target, launches the benchmark with two MPI ranks, parses
the emitted CSV, and saves results incrementally so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any


DEFAULT_THREADS = [2 * 32, 3 * 32, 5 * 32, 9 * 32, 17 * 32, 33 * 32]
DEFAULT_PIPE_STAGES = [1, 2, 4, 8, 16]
DEFAULT_UNROLL_FACTORS = [2]
DEFAULT_ELEMENTS_PER_THREAD = [2, 4, 8, 16, 32, 64]
DEFAULT_MAX_NUM_CTAS = [2, 4, 8, 16, 32, 64]

DEFAULT_MIN_BYTES = "8K"
DEFAULT_MAX_BYTES = "512M"

RESULTS_JSON = "results.json"
SUMMARY_TXT = "ranked_summary.txt"
LOGS_DIR = "logs"
CONFIG_KEYS = (
    "maxNumCTAs",
    "threads",
    "unrollFactor",
    "pipeStages",
    "elementsPerThread",
)
SUCCESS_STATUS = "success"
FAILED_STATUSES = {
    "build_failed",
    "runtime_failed",
    "shared_memory_exceeded",
    "parse_failed",
}
SHARED_MEMORY_PATTERNS = (
    re.compile(r"required shared memory .* exceeds hardware limits", re.IGNORECASE),
    re.compile(r"uses too much shared data", re.IGNORECASE),
    re.compile(r"uses too much shared memory", re.IGNORECASE),
    re.compile(r"shared memory", re.IGNORECASE),
)


@dataclass(frozen=True)
class SweepConfig:
    maxNumCTAs: int
    threads: int
    unrollFactor: int
    pipeStages: int
    elementsPerThread: int

    def as_params(self) -> dict[str, int]:
        return {
            "maxNumCTAs": self.maxNumCTAs,
            "threads": self.threads,
            "unrollFactor": self.unrollFactor,
            "pipeStages": self.pipeStages,
            "elementsPerThread": self.elementsPerThread,
        }

    def key(self) -> str:
        return (
            f"ctas-{self.maxNumCTAs}_thr-{self.threads}_uf-{self.unrollFactor}"
            f"_ps-{self.pipeStages}_ept-{self.elementsPerThread}"
        )


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return [int(item) for item in items]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def tail_text(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n")
        handle.write(f"[exit_code] {proc.returncode}\n")
        handle.write("[stdout]\n")
        handle.write(proc.stdout)
        if proc.stdout and not proc.stdout.endswith("\n"):
            handle.write("\n")
        handle.write("[stderr]\n")
        handle.write(proc.stderr)
        if proc.stderr and not proc.stderr.endswith("\n"):
            handle.write("\n")
    return CommandResult(
        args=list(cmd),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def run_benchmark_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    timeout_seconds: int | None,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        handle.write(f"[exit_code] {proc.returncode}\n")
        handle.write(f"[timed_out] {'yes' if timed_out else 'no'}\n")
    merged_output = log_path.read_text(encoding="utf-8")
    if merged_output.startswith("$ "):
        newline = merged_output.find("\n")
        merged_output = merged_output[newline + 1 :] if newline != -1 else ""
    return CommandResult(
        args=list(cmd),
        returncode=124 if timed_out else proc.returncode,
        stdout=merged_output,
        stderr="",
        timed_out=timed_out,
    )


def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "metadata": {
                "created_at": utc_now(),
            },
            "results": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def merge_entry(results: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    for idx, existing in enumerate(results):
        if existing.get("params") == entry.get("params"):
            results[idx] = entry
            return
    results.append(entry)


def detect_shared_memory_issue(text: str) -> bool:
    return any(pattern.search(text) for pattern in SHARED_MEMORY_PATTERNS)


def parse_csv_output(stdout: str, cfg: SweepConfig) -> tuple[list[dict[str, Any]], str | None]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    header_index = None
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "bytes" in lowered and "suture(gb/s)" in lowered:
            header_index = idx
            break
    if header_index is None:
        return [], "missing CSV header"

    header = next(csv.reader([lines[header_index]], skipinitialspace=True))
    required_columns = {"bytes", "suture(GB/s)", "error(%)", "threads", "pipeStages", "stageExtent", "unrollFactor"}
    missing = [column for column in required_columns if column not in header]
    if missing:
        return [], f"missing required CSV columns: {', '.join(missing)}"

    rows: list[dict[str, Any]] = []
    reader = csv.reader(lines[header_index + 1 :], skipinitialspace=True)
    for raw in reader:
        if not raw:
            continue
        if len(raw) < len(header):
            continue
        row = {header[i]: raw[i].strip() for i in range(len(header))}
        try:
            parsed = {
                "bytes": int(row["bytes"]),
                "gbps": float(row["suture(GB/s)"]),
                "error_percent": float(row["error(%)"]),
                "raw_fields": row,
            }
        except (TypeError, ValueError) as exc:
            return [], f"failed to parse CSV row {row!r}: {exc}"

        try:
            row_threads = int(row["threads"])
            row_pipe_stages = int(row["pipeStages"])
            row_stage_extent = int(row["stageExtent"])
            row_unroll = int(row["unrollFactor"])
        except ValueError as exc:
            return [], f"failed to parse configuration columns from CSV row {row!r}: {exc}"

        if row_threads != cfg.threads:
            return [], f"CSV threads={row_threads} does not match requested threads={cfg.threads}"
        if row_pipe_stages != cfg.pipeStages:
            return [], f"CSV pipeStages={row_pipe_stages} does not match requested pipeStages={cfg.pipeStages}"
        if row_stage_extent != cfg.elementsPerThread:
            return [], (
                "CSV stageExtent="
                f"{row_stage_extent} does not match requested elementsPerThread={cfg.elementsPerThread}"
            )
        if row_unroll != cfg.unrollFactor:
            return [], f"CSV unrollFactor={row_unroll} does not match requested unrollFactor={cfg.unrollFactor}"

        rows.append(parsed)

    if not rows:
        return [], "CSV header found but no data rows parsed"

    rows.sort(key=lambda item: item["bytes"])
    return rows, None


def summarize_notes(
    configure_proc: CommandResult,
    build_proc: CommandResult,
    run_proc: CommandResult | None,
) -> str:
    notes: list[str] = []
    if configure_proc.returncode != 0:
        notes.append(f"configure exited with code {configure_proc.returncode}")
    if build_proc.returncode != 0:
        notes.append(f"build exited with code {build_proc.returncode}")
    if run_proc is not None and run_proc.returncode != 0:
        if run_proc.timed_out:
            notes.append(f"run timed out after configured limit (exit code {run_proc.returncode})")
        else:
            notes.append(f"run exited with code {run_proc.returncode}")
    if not notes:
        return ""
    return "; ".join(notes)


def sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(entry: dict[str, Any]) -> tuple[int, float, tuple[int, ...]]:
        params = entry["params"]
        success_rank = 0 if entry["status"] == SUCCESS_STATUS else 1
        peak = entry.get("peak_gbps")
        peak_sort = -(peak if isinstance(peak, (int, float)) else -1.0)
        tuple_key = tuple(params[key] for key in CONFIG_KEYS)
        return (success_rank, peak_sort, tuple_key)

    return sorted(entries, key=sort_key)


def write_summary(path: Path, entries: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sort_entries(entries)
    lines = [
        "P2P sweep summary",
        f"Updated: {utc_now()}",
        f"Build directory: {metadata.get('build_dir', '')}",
        f"Executable: {metadata.get('executable', '')}",
        f"Launcher: {' '.join(metadata.get('launcher', []))}",
        f"NVSHMEM_BOOTSTRAP: {metadata.get('nvshmem_bootstrap', '')}",
        f"Graph launches: {metadata.get('graph_launches', '')}",
        f"Runs: {metadata.get('runs', '')}",
        f"Warmup: {metadata.get('warmup', '')}",
        f"Run timeout seconds: {metadata.get('run_timeout_seconds', '')}",
        f"Completed entries: {len(sorted_entries)}",
        "",
        "Rank | Status | Peak GB/s | maxNumCTAs | threads | unrollFactor | pipeStages | elementsPerThread | Notes",
        "-" * 120,
    ]
    for idx, entry in enumerate(sorted_entries, start=1):
        params = entry["params"]
        peak = entry.get("peak_gbps")
        peak_text = f"{peak:.6f}" if isinstance(peak, (int, float)) else "-"
        lines.append(
            f"{idx:4d} | {entry['status']:<22} | {peak_text:>10} | "
            f"{params['maxNumCTAs']:>10} | {params['threads']:>7} | {params['unrollFactor']:>12} | "
            f"{params['pipeStages']:>10} | {params['elementsPerThread']:>17} | "
            f"{entry.get('notes', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_launcher() -> list[str]:
    launcher = ["mpirun"]
    if os.geteuid() == 0:
        launcher.append("--allow-run-as-root")
    launcher.extend(["-np", "2"])
    return launcher


def default_run_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("NVSHMEM_BOOTSTRAP", "MPI")
    return env


def build_cmake_args(cfg: SweepConfig, build_dir: Path) -> list[str]:
    return [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        f"-DP2P_THREADS={cfg.threads}",
        f"-DP2P_UNROLL_FACTOR={cfg.unrollFactor}",
        f"-DP2P_PIPE_STAGES={cfg.pipeStages}",
        f"-DP2P_ELEMENTS_PER_THREAD={cfg.elementsPerThread}",
    ]


def should_skip(existing: dict[str, Any] | None, force: bool, rerun_failed: bool) -> bool:
    if existing is None:
        return False
    status = existing.get("status")
    if force:
        return False
    if status == SUCCESS_STATUS:
        return True
    if rerun_failed and status in FAILED_STATUSES:
        return False
    return True


def select_configs(args: argparse.Namespace) -> list[SweepConfig]:
    threads = parse_int_list(args.threads, DEFAULT_THREADS)
    pipe_stages = parse_int_list(args.pipe_stages, DEFAULT_PIPE_STAGES)
    unroll_factors = parse_int_list(args.unroll_factors, DEFAULT_UNROLL_FACTORS)
    elements_per_thread = parse_int_list(args.elements_per_thread, DEFAULT_ELEMENTS_PER_THREAD)
    max_num_ctas = parse_int_list(args.max_num_ctas, DEFAULT_MAX_NUM_CTAS)

    configs = [
        SweepConfig(max_num_ctas, threads_value, unroll_factor, pipe_stage, ept)
        for max_num_ctas, threads_value, unroll_factor, pipe_stage, ept in product(
            max_num_ctas,
            threads,
            unroll_factors,
            pipe_stages,
            elements_per_thread,
        )
    ]
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs


def maybe_tqdm(iterable: list[SweepConfig]) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(iterable, unit="cfg")
    except Exception:
        return iterable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="cmake-build-release", help="CMake build directory")
    parser.add_argument("--results-dir", default="results/p2p_sweep", help="Directory for JSON, summary, and logs")
    parser.add_argument("--launcher", default=None, help="Launcher command prefix, e.g. 'mpirun -np 2'")
    parser.add_argument("--exe", default=None, help="Override executable path")
    parser.add_argument("--min-bytes", default=DEFAULT_MIN_BYTES, help="Minimum message size passed to p2p")
    parser.add_argument("--max-bytes", default=DEFAULT_MAX_BYTES, help="Maximum message size passed to p2p")
    parser.add_argument(
        "--nvshmem-bootstrap",
        default="MPI",
        help="Value exported as NVSHMEM_BOOTSTRAP for benchmark runs; use an empty string to leave unset",
    )
    parser.add_argument("--threads", default=None, help="Comma-separated thread counts")
    parser.add_argument("--pipe-stages", default=None, help="Comma-separated pipe stages")
    parser.add_argument("--unroll-factors", default=None, help="Comma-separated unroll factors")
    parser.add_argument("--elements-per-thread", default=None, help="Comma-separated elements per thread")
    parser.add_argument("--max-num-ctas", default=None, help="Comma-separated max CTA values")
    parser.add_argument("--graph-launches", type=int, default=None, help="Override benchmark graph_launches")
    parser.add_argument("--runs", type=int, default=None, help="Override benchmark runs")
    parser.add_argument("--warmup", type=int, default=None, help="Override benchmark warmup")
    parser.add_argument(
        "--run-timeout-seconds",
        type=int,
        default=900,
        help="Kill a benchmark run if it exceeds this many seconds; use 0 to disable",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tuples after filtering")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned sweep without building or running")
    parser.add_argument("--force", action="store_true", help="Rerun all tuples, including successful ones")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun tuples whose prior status was not success")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm even if installed")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = (repo_root / args.build_dir).resolve()
    results_dir = (repo_root / args.results_dir).resolve()
    results_path = results_dir / RESULTS_JSON
    summary_path = results_dir / SUMMARY_TXT
    logs_dir = results_dir / LOGS_DIR

    launcher = shlex.split(args.launcher) if args.launcher else default_launcher()
    exe_path = (repo_root / args.exe).resolve() if args.exe else (build_dir / "p2p").resolve()
    configs = select_configs(args)

    if not configs:
        print("No configurations selected.", file=sys.stderr)
        return 1

    state = load_results(results_path)
    entries = state.setdefault("results", [])
    existing_by_key = {
        SweepConfig(**entry["params"]).key(): entry
        for entry in entries
        if all(key in entry.get("params", {}) for key in CONFIG_KEYS)
    }
    state["metadata"] = {
        **state.get("metadata", {}),
        "updated_at": utc_now(),
        "repo_root": str(repo_root),
        "build_dir": str(build_dir),
        "results_dir": str(results_dir),
        "executable": str(exe_path),
        "launcher": launcher,
        "nvshmem_bootstrap": args.nvshmem_bootstrap,
        "min_bytes": args.min_bytes,
        "max_bytes": args.max_bytes,
        "graph_launches": args.graph_launches,
        "runs": args.runs,
        "warmup": args.warmup,
        "run_timeout_seconds": args.run_timeout_seconds,
        "parameter_space": {
            "threads": sorted({cfg.threads for cfg in configs}),
            "pipeStages": sorted({cfg.pipeStages for cfg in configs}),
            "unrollFactor": sorted({cfg.unrollFactor for cfg in configs}),
            "elementsPerThread": sorted({cfg.elementsPerThread for cfg in configs}),
            "maxNumCTAs": sorted({cfg.maxNumCTAs for cfg in configs}),
        },
    }

    planned = []
    for cfg in configs:
        existing = existing_by_key.get(cfg.key())
        if should_skip(existing, args.force, args.rerun_failed):
            continue
        planned.append(cfg)

    print(f"Selected {len(configs)} total configurations.")
    print(f"Will execute {len(planned)} configurations; {len(configs) - len(planned)} will be skipped from resume state.")
    if args.dry_run:
        for cfg in planned:
            print(json.dumps(cfg.as_params(), sort_keys=True))
        return 0

    start_time = time.monotonic()
    completed_in_this_run = 0
    iterable: Any = planned if args.no_progress else maybe_tqdm(planned)
    for cfg in iterable:
        completed_in_this_run += 1
        elapsed = time.monotonic() - start_time
        avg = elapsed / completed_in_this_run if completed_in_this_run else 0.0
        remaining = avg * (len(planned) - completed_in_this_run)
        progress_line = (
            f"[{completed_in_this_run}/{len(planned)}] {cfg.key()} "
            f"elapsed={format_seconds(elapsed)} eta={format_seconds(remaining)}"
        )
        if hasattr(iterable, "set_description_str"):
            iterable.set_description_str(progress_line)
        else:
            print(progress_line)

        config_log_dir = logs_dir / cfg.key()
        configure_log = config_log_dir / "configure.log"
        build_log = config_log_dir / "build.log"
        run_log = config_log_dir / "run.log"

        configure_cmd = build_cmake_args(cfg, build_dir)
        configure_proc = run_command(configure_cmd, repo_root, configure_log)
        if configure_proc.returncode == 0:
            build_proc = run_command(
                ["cmake", "--build", str(build_dir), "--target", "p2p", "--parallel"],
                repo_root,
                build_log,
            )
        else:
            build_log.parent.mkdir(parents=True, exist_ok=True)
            build_log.write_text(
                "$ cmake --build "
                + shlex.quote(str(build_dir))
                + " --target p2p --parallel\n[skipped] configure step failed\n",
                encoding="utf-8",
            )
            build_proc = CommandResult(
                args=["cmake", "--build", str(build_dir), "--target", "p2p", "--parallel"],
                returncode=1,
                stdout="",
                stderr="build skipped because configure step failed",
            )

        status = SUCCESS_STATUS
        gbps_rows: list[dict[str, Any]] = []
        parse_error: str | None = None
        run_proc: CommandResult | None = None

        combined_build_text = "\n".join(
            part for part in (configure_proc.stdout, configure_proc.stderr, build_proc.stdout, build_proc.stderr) if part
        )
        if configure_proc.returncode != 0 or build_proc.returncode != 0:
            status = "shared_memory_exceeded" if detect_shared_memory_issue(combined_build_text) else "build_failed"
        else:
            env = default_run_env()
            if args.nvshmem_bootstrap == "":
                env.pop("NVSHMEM_BOOTSTRAP", None)
            else:
                env["NVSHMEM_BOOTSTRAP"] = args.nvshmem_bootstrap

            run_cmd = launcher + [str(exe_path), args.min_bytes, args.max_bytes, str(cfg.maxNumCTAs)]
            if args.graph_launches is not None:
                run_cmd.append(str(args.graph_launches))
            if args.runs is not None:
                run_cmd.append(str(args.runs))
            if args.warmup is not None:
                if args.runs is None:
                    run_cmd.append("256")
                run_cmd.append(str(args.warmup))

            timeout_seconds = None if args.run_timeout_seconds == 0 else args.run_timeout_seconds
            run_proc = run_benchmark_command(run_cmd, repo_root, run_log, env=env, timeout_seconds=timeout_seconds)
            combined_run_text = "\n".join(part for part in (run_proc.stdout, run_proc.stderr) if part)
            if run_proc.returncode != 0:
                status = "shared_memory_exceeded" if detect_shared_memory_issue(combined_run_text) else "runtime_failed"
            else:
                gbps_rows, parse_error = parse_csv_output(run_proc.stdout, cfg)
                if parse_error is not None:
                    if "Two processes required" in combined_run_text:
                        status = "runtime_failed"
                    else:
                        status = "parse_failed"

        peak_gbps = max((row["gbps"] for row in gbps_rows), default=None)
        entry = {
            "params": cfg.as_params(),
            "status": status,
            "peak_gbps": peak_gbps,
            "gbps_by_message_size": gbps_rows,
            "notes": parse_error or summarize_notes(configure_proc, build_proc, run_proc),
            "configure_log": str(configure_log),
            "build_log": str(build_log),
            "run_log": str(run_log),
            "configure_log_tail": tail_text("\n".join([configure_proc.stdout, configure_proc.stderr])),
            "build_log_tail": tail_text("\n".join([build_proc.stdout, build_proc.stderr])),
            "run_log_tail": tail_text(run_proc.stdout if run_proc is not None else ""),
            "completed_at": utc_now(),
        }
        merge_entry(entries, entry)
        state["metadata"]["updated_at"] = utc_now()
        atomic_write_json(results_path, state)
        write_summary(summary_path, entries, state["metadata"])

    print(f"Results JSON: {results_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
