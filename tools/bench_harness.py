#!/usr/bin/env python3
"""Benchmark harness for the FLUX CLI.

Runs ./flux multiple times, parses machine-readable timing JSON from stdout
(produced by --timing), and emits aggregate stats as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    w = idx - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "stdev": float("nan")}
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse_timing_json(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and '"event":"timing"' in line:
            return json.loads(line)
    return None


def build_command(args: argparse.Namespace, out_path: str) -> list[str]:
    cmd = [
        args.flux_binary,
        "-d",
        args.model_dir,
        "-p",
        args.prompt,
        "-o",
        out_path,
        "-W",
        str(args.width),
        "-H",
        str(args.height),
        "-s",
        str(args.steps),
        "-S",
        str(args.seed),
        "--timing",
    ]
    if args.mmap:
        cmd.append("--mmap")
    else:
        cmd.append("--no-mmap")
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def run_once(args: argparse.Namespace, run_idx: int) -> dict:
    out_path = str(Path(args.output_dir) / f"bench_{run_idx:04d}.png")
    cmd = build_command(args, out_path)

    wall_start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_s = time.perf_counter() - wall_start

    if proc.returncode != 0:
        raise RuntimeError(
            f"flux failed (run {run_idx}) code={proc.returncode}\n"
            f"CMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )

    timing = parse_timing_json(proc.stdout)
    if timing is None:
        raise RuntimeError(
            "Missing timing JSON from flux stdout. Ensure binary supports --timing.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    timing["wall_s"] = wall_s
    return timing


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark FLUX CLI timings")
    p.add_argument("--flux-binary", default="./flux")
    p.add_argument("--model-dir", default="flux-klein-model")
    p.add_argument("--prompt", default="A photo of a small robot")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output-dir", default="/tmp")
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(args.warmup):
        _ = run_once(args, i)

    runs = [run_once(args, args.warmup + i) for i in range(args.repeats)]

    load_vals = [r["load_s"] for r in runs]
    gen_vals = [r["generate_s"] for r in runs]
    e2e_vals = [r["end_to_end_s"] for r in runs]
    wall_vals = [r["wall_s"] for r in runs]

    summary = {
        "metadata": {
            "flux_binary": args.flux_binary,
            "model_dir": args.model_dir,
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "mmap": args.mmap,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "metrics_s": {
            "load": summarize(load_vals),
            "generate": summarize(gen_vals),
            "end_to_end": summarize(e2e_vals),
            "wall": summarize(wall_vals),
        },
        "runs": runs,
    }

    text = json.dumps(summary, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
