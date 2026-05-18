# Paladin Commenter v2

[![Build](https://img.shields.io/badge/build-cargo%20test-brightgreen)](#build)

A context-aware Rust CLI that chunks code by syntax boundaries, sends chunks to a local Ollama/Gemma model, and can generate analysis reports or safely rewrite a commented copy of the source file.

## Why this exists

Small local models can get stuck when you send huge files or arbitrary character slices. This tool avoids fixed-size splitting and uses Tree-sitter to keep whole semantic blocks together whenever possible:

- functions
- impl blocks
- structs
- enums
- traits
- modules
- constants
- type aliases
- macros
- imports and top-level gaps

If a block is too large, it recursively descends into smaller AST nodes instead of cutting the code randomly.

## Features

- **AST-based chunking** for Rust using tree-sitter
- **Ollama `/api/generate`** integration with configurable model
- Model health check through `/api/tags`
- Dry-run mode to inspect chunks without calling Ollama
- **Three processing modes**: analyze, comment, rewrite
- **Four prompt profiles**: explain, maintainer-comments, security, architecture
- **Chunk response cache** with optional TTL (`--cache-ttl`)
- **Structured logging** via `tracing` with `--log-level` and `--quiet`
- **Progress bar** (`--progress`) for long-running jobs
- Retry handling with configurable max retries
- Request timeout per chunk
- Machine-readable JSON run log (versioned schema)
- Chunk range controls: `--from-chunk`, `--only-chunk`, `--skip-failed`
- Optional `rustfmt` after rewrite
- **Type-safe chunk kinds** via `ChunkKind` enum

## Build

```bash
cargo build --release
```

## Quick start

### Check chunking only (no Ollama required)

```bash
cargo run -- \
  --file src/main.rs \
  --dry-run \
  --max-chars 4000 \
  --target-chars 2500
```

### Analyze a file

```bash
cargo run -- \
  --file src/main.rs \
  --mode analyze \
  --model gemma4:e4b \
  --output paladin-analysis.md
```

### Create a safely commented copy

```bash
cargo run -- \
  --file src/main.rs \
  --mode rewrite \
  --model gemma4:e4b \
  --output src/main.commented.rs
```

The original file is **never** modified.

### Use cache with TTL

```bash
cargo run -- \
  --file src/main.rs \
  --mode rewrite \
  --cache \
  --cache-ttl 86400   # 24 hours; 0 = unlimited
```

Cache files are stored in `.paladin-cache/`.

### Show a progress bar

```bash
cargo run -- \
  --file src/main.rs \
  --mode analyze \
  --progress
```

### Control log verbosity

```bash
# Debug output
cargo run -- --file src/main.rs --dry-run --log-level debug

# Silence everything except errors
cargo run -- --file src/main.rs --mode analyze --quiet
```

### Resume after a failed chunk

```bash
cargo run -- \
  --file src/main.rs \
  --mode rewrite \
  --from-chunk 8
```

### Process one problematic chunk

```bash
cargo run -- \
  --file src/main.rs \
  --mode analyze \
  --only-chunk 7
```

## Recommended settings for small local models

```bash
--max-chars 4000 --target-chars 2500 --chunk-timeout-seconds 180 --max-retries 1
```

## Running tests

```bash
cargo test
```

## Notes

- For local models, prefer smaller chunk sizes. A chunk that is too big can make the model loop, hallucinate, or return malformed code.
- The `--skip-health-check` flag lets you run without verifying Ollama connectivity first.
- The JSON run log now includes a `version` field for forward compatibility.
