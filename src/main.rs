mod cache;
mod chunker;
mod ollama;
mod prompt;
mod report;
mod rewrite;
mod runlog;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::cache::ChunkCache;
use crate::chunker::{chunk_rust_source, ChunkConfig};
use crate::ollama::OllamaClient;
use crate::prompt::{build_analysis_prompt, build_commenting_prompt, PromptProfile};
use crate::report::{write_markdown_report, AnalysisItem};
use crate::rewrite::{build_rewritten_source, rustfmt_file_if_available, validate_generated_chunk};
use crate::runlog::{ChunkRunStatus, RunLog};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum Mode {
    /// Analyze chunks and write a Markdown report.
    Analyze,
    /// Ask the model to return commented code chunks and save them in a report.
    Comment,
    /// Ask the model to return commented code chunks and write a new commented source file.
    Rewrite,
}

#[derive(Debug, Parser)]
#[command(name = "paladin-commenter")]
#[command(about = "Context-aware Rust file chunker that sends semantic chunks to a local Ollama/Gemma model")]
struct Cli {
    /// Rust source file to process.
    #[arg(short, long)]
    file: PathBuf,

    /// Ollama model name.
    #[arg(short, long, default_value = "gemma4:e4b")]
    model: String,

    /// Ollama base URL.
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    ollama_url: String,

    /// Maximum characters per semantic chunk.
    #[arg(long, default_value_t = 6000)]
    max_chars: usize,

    /// Soft target size. Small chunks are merged until they approach this size.
    #[arg(long, default_value_t = 3500)]
    target_chars: usize,

    /// Output file. Markdown for analyze/comment mode. Rust source for rewrite mode.
    #[arg(short, long, default_value = "paladin-analysis.md")]
    output: PathBuf,

    /// Processing mode.
    #[arg(long, value_enum, default_value_t = Mode::Analyze)]
    mode: Mode,

    /// Prompt profile.
    #[arg(long, value_enum, default_value_t = PromptProfile::MaintainerComments)]
    profile: PromptProfile,

    /// Print chunk boundaries without calling Ollama.
    #[arg(long)]
    dry_run: bool,

    /// Enable chunk response cache.
    #[arg(long)]
    cache: bool,

    /// Cache directory.
    #[arg(long, default_value = ".paladin-cache")]
    cache_dir: PathBuf,

    /// Start processing at this chunk index, 1-based.
    #[arg(long)]
    from_chunk: Option<usize>,

    /// Process only this chunk index, 1-based.
    #[arg(long)]
    only_chunk: Option<usize>,

    /// Continue when a chunk fails.
    #[arg(long)]
    skip_failed: bool,

    /// Maximum retries per chunk after the first attempt.
    #[arg(long, default_value_t = 1)]
    max_retries: usize,

    /// HTTP timeout per chunk request.
    #[arg(long, default_value_t = 240)]
    chunk_timeout_seconds: u64,

    /// Number of context tokens requested from Ollama.
    #[arg(long, default_value_t = 8192)]
    num_ctx: u32,

    /// Run rustfmt after rewrite mode if rustfmt is installed.
    #[arg(long, default_value_t = true)]
    rustfmt: bool,

    /// Write a machine-readable JSON run log.
    #[arg(long, default_value = "paladin-run.json")]
    run_log: PathBuf,

    /// Skip the Ollama model availability check.
    #[arg(long)]
    no_health_check: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let started = Instant::now();

    let source = fs::read_to_string(&cli.file)
        .with_context(|| format!("failed to read file: {}", cli.file.display()))?;

    let config = ChunkConfig {
        max_chars: cli.max_chars,
        target_chars: cli.target_chars,
    };

    let chunks = chunk_rust_source(&source, config).context("failed to chunk Rust source")?;

    println!("File: {}", cli.file.display());
    println!("Chunks: {}", chunks.len());

    for chunk in &chunks {
        println!(
            "- #{:03} {:>20} lines {}-{} chars={} bytes={}..{}",
            chunk.index,
            chunk.kind,
            chunk.start_line,
            chunk.end_line,
            chunk.text.chars().count(),
            chunk.start_byte,
            chunk.end_byte
        );
    }

    if cli.dry_run {
        println!("Dry run enabled. Ollama was not called.");
        return Ok(());
    }

    let client = OllamaClient::new(
        cli.ollama_url.clone(),
        cli.model.clone(),
        cli.chunk_timeout_seconds,
        cli.num_ctx,
    );

    if !cli.no_health_check {
        let models = client.list_models().context("Ollama health check failed")?;
        if !models.iter().any(|m| m == &cli.model) {
            anyhow::bail!(
                "model '{}' was not found in Ollama. Available models: {}",
                cli.model,
                models.join(", ")
            );
        }
        println!("Ollama reachable. Model '{}' found.", cli.model);
    }

    let cache = if cli.cache {
        Some(ChunkCache::new(cli.cache_dir.clone())?)
    } else {
        None
    };

    let mut run_log = RunLog::new(&cli.file, &cli.model, &cli.mode_string(), chunks.len());
    let mut report_items = Vec::new();
    let mut replacements = Vec::new();

    for chunk in chunks.iter() {
        if let Some(only) = cli.only_chunk {
            if chunk.index != only {
                continue;
            }
        }
        if let Some(from) = cli.from_chunk {
            if chunk.index < from {
                continue;
            }
        }

        let prompt = match cli.mode {
            Mode::Analyze => build_analysis_prompt(chunk, cli.profile),
            Mode::Comment | Mode::Rewrite => build_commenting_prompt(chunk, cli.profile),
        };

        let chunk_started = Instant::now();
        println!(
            "[{}/{}] Processing {} lines {}-{}...",
            chunk.index,
            chunks.len(),
            chunk.kind,
            chunk.start_line,
            chunk.end_line
        );

        let response_result = get_or_generate(
            &client,
            cache.as_ref(),
            &cli.model,
            &cli.mode_string(),
            chunk.index,
            &chunk.text,
            &prompt,
            cli.max_retries,
        );

        match response_result {
            Ok(response) => {
                if cli.mode == Mode::Rewrite {
                    let validation = validate_generated_chunk(&response, &chunk.text);
                    if let Err(err) = validation {
                        let msg = format!("generated chunk failed validation: {err}");
                        run_log.push_chunk(ChunkRunStatus::failed(chunk, &msg, chunk_started.elapsed()));
                        if cli.skip_failed {
                            eprintln!("Chunk #{} failed validation. Keeping original chunk.", chunk.index);
                            continue;
                        }
                        anyhow::bail!("chunk #{}: {}", chunk.index, msg);
                    }
                    replacements.push((chunk.clone(), response.clone()));
                }

                report_items.push(AnalysisItem {
                    index: chunk.index,
                    kind: chunk.kind.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    source_preview: chunk.preview(24),
                    model_response: response,
                });

                run_log.push_chunk(ChunkRunStatus::ok(chunk, chunk_started.elapsed()));
                println!("Chunk #{} ok in {:.1}s", chunk.index, chunk_started.elapsed().as_secs_f32());
            }
            Err(err) => {
                let msg = err.to_string();
                run_log.push_chunk(ChunkRunStatus::failed(chunk, &msg, chunk_started.elapsed()));
                eprintln!("Chunk #{} failed: {}", chunk.index, msg);
                if !cli.skip_failed {
                    run_log.finish(started.elapsed());
                    run_log.write(&cli.run_log)?;
                    anyhow::bail!("stopped because chunk #{} failed. Use --skip-failed to continue.", chunk.index);
                }
            }
        }
    }

    match cli.mode {
        Mode::Analyze | Mode::Comment => {
            write_markdown_report(&cli.output, &cli.file, &report_items)
                .with_context(|| format!("failed to write report: {}", cli.output.display()))?;
            println!("Report written to {}", cli.output.display());
        }
        Mode::Rewrite => {
            let rewritten = build_rewritten_source(&source, &replacements);
            fs::write(&cli.output, rewritten)
                .with_context(|| format!("failed to write rewritten file: {}", cli.output.display()))?;
            if cli.rustfmt {
                rustfmt_file_if_available(&cli.output);
            }
            println!("Rewritten file written to {}", cli.output.display());
        }
    }

    run_log.finish(started.elapsed());
    run_log.write(&cli.run_log)?;
    println!("Run log written to {}", cli.run_log.display());

    Ok(())
}

fn get_or_generate(
    client: &OllamaClient,
    cache: Option<&ChunkCache>,
    model: &str,
    mode: &str,
    chunk_index: usize,
    chunk_text: &str,
    prompt: &str,
    max_retries: usize,
) -> Result<String> {
    if let Some(cache) = cache {
        if let Some(hit) = cache.get(model, mode, chunk_index, chunk_text, prompt)? {
            println!("Chunk #{} loaded from cache", chunk_index);
            return Ok(hit);
        }
    }

    let mut last_err = None;
    for attempt in 0..=max_retries {
        if attempt > 0 {
            println!("Retrying chunk #{} attempt {}/{}", chunk_index, attempt + 1, max_retries + 1);
        }

        match client.generate(prompt) {
            Ok(response) => {
                if let Some(cache) = cache {
                    cache.put(model, mode, chunk_index, chunk_text, prompt, &response)?;
                }
                return Ok(response);
            }
            Err(err) => {
                last_err = Some(err);
            }
        }
    }

    Err(last_err.expect("at least one attempt should run"))
}

trait ModeExt {
    fn mode_string(&self) -> String;
}

impl ModeExt for Cli {
    fn mode_string(&self) -> String {
        match self.mode {
            Mode::Analyze => "analyze".to_string(),
            Mode::Comment => "comment".to_string(),
            Mode::Rewrite => "rewrite".to_string(),
        }
    }
}
