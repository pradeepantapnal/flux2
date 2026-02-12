# AGENTS.md (project guidance for Codex)

## Commands
- Build: `make blas` (Linux) or `make mps` (Apple Silicon macOS); use `make generic` when dependency-free fallback is needed.
- Unit tests: `make test-quick`
- Integration tests: `make test` and `make golden-test`
- Format/Lint: No dedicated formatter/linter target is defined in the Makefile; rely on compiler warnings (`-Wall -Wextra`) via `make generic`/`make blas` and keep existing style consistent.
- Benchmarks: No benchmark target is currently defined; use timing guidance in `README.md` and performance notes in `perf_budget.md`.

## Conventions
- Coding style: C-first codebase; follow existing style in touched files and keep changes minimal and local.
- Error handling policy: APIs that can fail return `NULL`/error status and expose details through `flux_get_error()`; always check return values and propagate/report errors.
- Logging policy: Prefer concise stderr diagnostics for failures and keep informational output minimal in library paths.
- Allowed dependencies: Keep core builds dependency-light; optional acceleration dependencies are OpenBLAS (Linux), Accelerate/Metal (macOS), and CUDA toolkit for CUDA backend.

## Refactor rules
- Small commits, one concern per commit
- No API breaks without explicit approval
- Prefer additive changes + deprecations over rewrites
- Always show build/test outputs
