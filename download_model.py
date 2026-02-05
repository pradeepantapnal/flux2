#!/usr/bin/env python3
"""
Download FLUX.2-klein-4B model files from HuggingFace using a pinned manifest.

Usage:
    python download_model.py [--output-dir DIR] [--manifest PATH]

Requirements:
    pip install huggingface_hub

This downloader is strict:
- uses exact repo_id + revision from manifest
- downloads only files listed in manifest
- verifies SHA-256 and size for every file
- exits non-zero on any mismatch
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to read manifest {path}: {e}") from e

    if not isinstance(manifest, dict):
        raise RuntimeError("Manifest root must be a JSON object")

    for key in ("repo_id", "revision", "files"):
        if key not in manifest:
            raise RuntimeError(f"Manifest missing required key: {key}")

    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise RuntimeError("Manifest 'files' must be a non-empty array")

    for i, entry in enumerate(manifest["files"]):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Manifest file entry #{i} must be an object")
        for key in ("path", "sha256", "size"):
            if key not in entry:
                raise RuntimeError(f"Manifest file entry #{i} missing key: {key}")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise RuntimeError(f"Manifest file entry #{i} has invalid path")
        if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64:
            raise RuntimeError(
                f"Manifest file entry #{i} has invalid sha256 (must be 64 hex chars)"
            )
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise RuntimeError(f"Manifest file entry #{i} has invalid size")

    return manifest


def verify_file(path: Path, expected_sha256: str, expected_size: int) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing downloaded file: {path}")

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download FLUX.2-klein-4B model files from HuggingFace using pinned manifest"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./flux-klein-model",
        help="Output directory (default: ./flux-klein-model)",
    )
    parser.add_argument(
        "--manifest",
        default="tools/model_manifest.json",
        help="Path to model manifest JSON (default: tools/model_manifest.json)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return 1

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    try:
        manifest = load_manifest(manifest_path)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    repo_id = manifest["repo_id"]
    revision = manifest["revision"]
    files = manifest["files"]

    print("FLUX.2-klein-4B Model Downloader (manifest-pinned)")
    print("===================================================")
    print(f"Manifest:  {manifest_path}")
    print(f"Repo:      {repo_id}")
    print(f"Revision:  {revision}")
    print(f"Output:    {output_dir}")
    print(f"Files:     {len(files)}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for entry in files:
            rel = entry["path"]
            expected_sha256 = entry["sha256"].lower()
            expected_size = entry["size"]

            print(f"Downloading {rel}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=rel,
                revision=revision,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )

            print(f"Verifying   {rel}...")
            verify_file(Path(downloaded_path), expected_sha256, expected_size)

        print()
        print("Download complete and verified.")
        print("All files match pinned sha256 + size in the manifest.")
        print()
        print("Usage:")
        print(f"  ./flux -d {output_dir} -p \"your prompt\" -o output.png")
        return 0

    except Exception as e:
        print()
        print(f"ERROR: {e}")
        print("Aborting due to download/verification failure.")
        print()
        print("If you need to authenticate, run:")
        print("  huggingface-cli login")
        return 1


if __name__ == "__main__":
    sys.exit(main())
