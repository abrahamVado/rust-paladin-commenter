# Paladin Commenter v2

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

- AST-based chunking for Rust
- Ollama `/api/generate` integration
- model health check through `/api/tags`
- dry-run mode to inspect chunks
- analysis report mode
- safe rewrite mode that writes a new file instead of overwriting the original
- chunk cache to avoid re-running slow local model calls
- retry handling
- request timeout per chunk
- run log JSON
- chunk range controls such as `--from-chunk`, `--only-chunk`, and `--skip-failed`
- optional `rustfmt` after rewrite

## Build

```bash
cargo build --release
```

## Check chunking only

```bash
cargo run -- \
  --file src/main.rs \
  --dry-run \
  --max-chars 4000 \
  --target-chars 2500
```

## Analyze a file

```bash
cargo run -- \
  --file src/main.rs \
  --mode analyze \
  --model gemma4:e4b \
  --output paladin-analysis.md
```

## Create a safely commented copy

```bash
cargo run -- \
  --file src/main.rs \
  --mode rewrite \
  --model gemma4:e4b \
  --output src/main.commented.rs
```

The original file is not modified.

## Use cache

```bash
cargo run -- \
  --file src/main.rs \
  --mode rewrite \
  --cache
```

Cache files are stored in `.paladin-cache/`.

## Resume after a failed chunk

```bash
cargo run -- \
  --file src/main.rs \
  --mode rewrite \
  --from-chunk 8
```

## Process one problematic chunk

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

## Notes

For local models, prefer smaller chunk sizes. A chunk that is too big can make the model loop, hallucinate, or return malformed code.
