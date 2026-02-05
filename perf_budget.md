# Performance Budget & Baseline Workflow

This repository includes a lightweight benchmark harness (`tools/bench_harness.py`) and stable CLI timing JSON output.

## 1) Capture a local baseline

Build first (pick backend):

```bash
make generic
# or: make blas
```

Run benchmark harness:

```bash
python3 tools/bench_harness.py \
  --flux-binary ./flux \
  --model-dir flux-klein-model \
  --prompt "A photo of a small robot" \
  --width 256 --height 256 --steps 4 --seed 42 \
  --repeats 5 --warmup 1 \
  --output-dir /tmp/flux-bench \
  --json-out /tmp/flux-bench/baseline.json
```

The harness invokes:

```bash
./flux ... --timing --json-out <per-run-timing-file>
```

and aggregates metrics in `<baseline.json>`.

## 2) Collect a candidate run

After your changes, rerun with the same args and save a second file:

```bash
python3 tools/bench_harness.py ... --json-out /tmp/flux-bench/candidate.json
```

## 3) Compare candidate vs baseline

Quick Python comparator:

```bash
python3 - <<'PY'
import json
b = json.load(open('/tmp/flux-bench/baseline.json'))
c = json.load(open('/tmp/flux-bench/candidate.json'))
for k in ['load', 'generate', 'end_to_end', 'wall']:
    bm = b['metrics_s'][k]['mean']
    cm = c['metrics_s'][k]['mean']
    delta = ((cm - bm) / bm * 100.0) if bm else float('nan')
    print(f"{k:>10}: baseline={bm:.3f}s candidate={cm:.3f}s delta={delta:+.2f}%")
PY
```

## 4) Suggested budget guidance (informational)

- `generate.mean`: keep within **+10%** of baseline.
- `end_to_end.mean`: keep within **+10%** of baseline.
- `load.mean`: monitor separately; may vary with mmap/backend changes.

CI currently **does not enforce hard failures** for perf budgets; timing JSON is uploaded as artifacts for inspection.
