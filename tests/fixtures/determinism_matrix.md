# Determinism Matrix

Golden regression tests intentionally run a tiny matrix on the **generic** backend with `--smoke --deterministic`.

| Case | Width x Height | Steps | Seed | Expected Behavior |
|---|---:|---:|---:|---|
| tiny_seed_1 | 64x64 | 1 | 1 | Stable output SHA256 |
| rect_seed_42 | 96x64 | 2 | 42 | Stable output SHA256 |
| square_seed_777 | 80x80 | 4 | 777 | Stable output SHA256 |
| wide_seed_2025 | 128x64 | 3 | 2025 | Stable output SHA256 |

Each case validates:
1. exit code
2. output artifact written
3. SHA256 exact match

The deterministic mode banner must also be present in stderr.
