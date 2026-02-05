#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic golden CLI tests")
    parser.add_argument("--flux-binary", default="./flux")
    parser.add_argument("--golden", default="tests/fixtures/golden.json")
    args = parser.parse_args()

    flux_binary = Path(args.flux_binary).resolve()
    golden_path = Path(args.golden)

    if not flux_binary.exists():
        print(f"ERROR: Missing binary: {flux_binary}", file=sys.stderr)
        return 1

    with golden_path.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    failures = 0
    for case in golden["cases"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / f"{case['name']}.ppm"
            cmd = [
                str(flux_binary),
                "--smoke",
                "--deterministic",
                "-d", "dummy-model-dir",
                "-p", case["prompt"],
                "-o", str(out_path),
                "-W", str(case["width"]),
                "-H", str(case["height"]),
                "-s", str(case["steps"]),
                "-S", str(case["seed"]),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            digest = sha256_file(out_path) if out_path.exists() else None

            ok = True
            if proc.returncode != case["expected_exit_code"]:
                ok = False
            if not out_path.exists():
                ok = False
            if "[deterministic] mode enabled" not in proc.stderr:
                ok = False
            if digest != case["expected_sha256"]:
                ok = False

            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {case['name']}")

            if not ok:
                failures += 1
                print(f"  cmd: {' '.join(cmd)}")
                print(f"  exit: got={proc.returncode} expected={case['expected_exit_code']}")
                print(f"  output_exists: {out_path.exists()}")
                print(f"  sha256: got={digest} expected={case['expected_sha256']}")
                print(f"  stderr:\n{proc.stderr}")

    if failures:
        print(f"Golden tests failed: {failures} case(s)", file=sys.stderr)
        return 1

    print(f"Golden tests passed: {len(golden['cases'])} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
