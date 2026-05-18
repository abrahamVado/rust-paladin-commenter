mod cache;
mod chunk_kind;
mod chunker;
mod ollama;
mod prompt;
mod report;
mod rewrite;
mod runlog;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::cache::ChunkCache;
use crate::chunker::{chunk_rust_source, ChunkConfig};
use crate::ollama::OllamaClient;
use crate::prompt::{build_analysis_prompt, build_commenting_prompt, PromptProfile};
use crate::report::{write_markdown_report, AnalysisItem};
use crate::rewrite::{build_rewritten_source, rustfmt_file_if_available, validate_generated_chunk};
use crate::runlog::{ChunkRunStatus, RunLog};

// ---------------------------------------------------------------------------
// CLI types
// ---------------------------------------------------------------------------

/// Defines the primary operational mode of the Paladin commenter.
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum Mode {
    /// Analyze chunks and write a Markdown report containing analysis and comments.
    Analyze,
    /// Ask the model to return commented code chunks and save them in a report (Markdown format).
    Comment,
    /// Ask the model to return commented code chunks and write a new commented source file (Rust source format).
    Rewrite,
}

/// Structured log level selectable from the CLI.
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Command-line interface arguments for the Paladin commenter tool.
#[derive(Debug, Parser)]
#[command(name = "paladin-commenter", version)]
#[command(
    about = "Context-aware Rust file chunker that sends semantic chunks to a local Ollama/Gemma model"
)]
struct Cli {
    /// Rust source file to process. This file is read and chunked internally.
    #[arg(short, long)]
    file: PathBuf,

    /// Ollama model name. Specifies which local model to use for inference.
    #[arg(short, long, default_value = "gemma4:e4b")]
    model: String,

    /// Ollama base URL. The endpoint where the local model server is running.
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    ollama_url: String,

    /// Maximum characters per semantic chunk. This is a hard limit to prevent API failures.
    #[arg(long, default_value_t = 6000)]
    max_chars: usize,

    /// Soft target size. The chunker attempts to group chunks until they approach this size for better context.
    #[arg(long, default_value_t = 3500)]
    target_chars: usize,

    /// Output file. Markdown for analyze/comment mode, or Rust source for rewrite mode.
    #[arg(short, long, default_value = "paladin-analysis.md")]
    output: PathBuf,

    /// Processing mode. Determines the output format and the prompt structure used.
    #[arg(long, value_enum, default_value_t = Mode::Analyze)]
    mode: Mode,

    /// Prompt profile. Controls the style and depth of the generated comments (e.g., Maintainer, Beginner).
    #[arg(long, value_enum, default_value_t = PromptProfile::MaintainerComments)]
    profile: PromptProfile,

    /// If set, the tool will only perform chunking and logging without making any actual API calls to Ollama.
    #[arg(long)]
    dry_run: bool,

    /// Enable chunk response cache. If true, the tool will check the cache before calling the model.
    #[arg(long)]
    cache: bool,

    /// Cache directory. Location where chunk results are stored to avoid redundant API calls.
    #[arg(long, default_value = ".paladin-cache")]
    cache_dir: PathBuf,

    /// Cache time-to-live in seconds. Entries older than this are treated as misses, forcing a re-run.
    /// Set to 0 to disable TTL (keep entries forever).
    #[arg(long, default_value_t = 0)]
    cache_ttl: u64,

    /// Start processing at this chunk index, 1-based. Allows resuming interrupted runs.
    #[arg(long)]
    from_chunk: Option<usize>,

    /// Process only this chunk index, 1-based. Useful for debugging or targeted updates.
    #[arg(long)]
    only_chunk: Option<usize>,

    /// If a chunk fails processing (e.g., API error), setting this to true allows the process to continue with the next chunk.
    #[arg(long)]
    skip_failed: bool,

    /// Maximum retries per chunk after the first attempt. Handles transient network or API errors.
    #[arg(long, default_value_t = 1)]
    max_retries: usize,

    /// HTTP timeout per chunk request in seconds. Limits the time spent waiting for a single model response.
    #[arg(long, default_value_t = 240)]
    chunk_timeout_seconds: u64,

    /// Number of context tokens requested from Ollama. Must be large enough to hold the chunk content and prompt.
    #[arg(long, default_value_t = 8192)]
    num_ctx: u32,

    /// If true, runs `rustfmt` on the generated source code after processing, ensuring standard formatting.
    #[arg(long, default_value_t = true)]
    rustfmt: bool,

    /// Write a machine-readable JSON run log. Records status and metadata for reproducibility.
    #[arg(long, default_value = "paladin-run.json")]
    run_log: PathBuf,

    /// Skip the Ollama model availability check. Use with caution, as connectivity issues may not be caught.
    #[arg(long)]
    skip_health_check: bool,

    /// Logging verbosity. Controls the amount of information printed during execution.
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,

    /// Suppress all output except errors (shorthand for --log-level error).
    #[arg(long)]
    quiet: bool,

    /// Show a progress bar during chunk processing, providing visual feedback to the user.
    #[arg(long)]
    progress: bool,
}

impl Cli {
    /// Validate CLI arguments before heavy work begins.
    ///
    /// This function enforces business rules regarding the relationship between
    /// `--max-chars` and `--target-chars`.
    ///
    /// # Errors
    /// Returns an error if:
    /// 1. `max_chars` is less than 1000.
    /// 2. `target_chars` exceeds `max_chars`.
    fn validate(&self) -> Result<()> {
        if self.max_chars < 1000 {
            anyhow::bail!("--max-chars must be at least 1000 (got {})", self.max_chars);
        }
        if self.target_chars > self.max_chars {
            anyhow::bail!(
                "--target-chars ({}) cannot exceed --max-chars ({})",
                self.target_chars,
                self.max_chars
            );
        }
        Ok(())
    }

    /// Converts the internal `Mode` enum into a displayable string.
    fn mode_string(&self) -> String {
        match self.mode {
            Mode::Analyze => "analyze".to_string(),
            Mode::Comment => "comment".to_string(),
            Mode::Rewrite => "rewrite".to_string(),
        }
    }

    /// Determines the effective logging level based on the `quiet` flag and the
    /// configured `log_level`.
    ///
    /// The `quiet` flag overrides all other logging settings, forcing the level to ERROR.
    fn effective_log_level(&self) -> tracing::Level {
        if self.quiet {
            return tracing::Level::ERROR;
        }
        match self.log_level {
            LogLevel::Error => tracing::Level::ERROR,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Trace => tracing::Level::TRACE,
        }
    }

    /// Calculates the cache Time-To-Live (TTL) duration.
    ///
    /// If `cache_ttl` is 0, it implies no caching is desired, and `None` is returned.
    /// Otherwise, it converts the stored seconds value into a `Duration`.
    fn cache_ttl_duration(&self) -> Option<Duration> {
        if self.cache_ttl == 0 {
            None
        } else {
            Some(Duration::from_secs(self.cache_ttl))
        }
    }
}

// ---------------------------------------------------------------------------
// Logging setup
// ---------------------------------------------------------------------------

/// Initializes the global tracing subscriber.
///
/// This function sets up the logging backend for the entire application.
/// It uses `tracing_subscriber::fmt` to format logs and sets the maximum
/// logging level based on the provided `level`.
///
/// # Arguments
/// * `level` - The minimum logging level (e.g., `tracing::Level::INFO`) to process.
fn setup_logging(level: tracing::Level) {
    use tracing_subscriber::fmt;
    let builder = fmt::Subscriber::builder()
        .with_max_level(level)
        .with_target(false) // Suppress module names in logs for cleaner output
        .with_ansi(true);
    // Set the global default subscriber. This must be done early in the program lifecycle.
    tracing::subscriber::set_global_default(builder.finish())
        .expect("failed to set tracing subscriber");
}
// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Main entry point for the paladin-commenter tool.
///
/// This function orchestrates the entire process: reading the source file,
/// chunking it, interacting with the LLM via Ollama (with caching and retries),
/// and finally writing the results (either a report or a rewritten file).
fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up structured logging first so that validation errors are visible.
    setup_logging(cli.effective_log_level());

    // Validate CLI arguments against business rules (e.g., file existence, model name).
    cli.validate()?;

    let started = Instant::now();

    info!(file = %cli.file.display(), model = %cli.model, mode = %cli.mode_string(), "Starting paladin-commenter");

    // Read the entire source file content into memory.
    let source = fs::read_to_string(&cli.file)
        .with_context(|| format!("failed to read file: {}", cli.file.display()))?;

    let config = ChunkConfig {
        max_chars: cli.max_chars,
        target_chars: cli.target_chars,
    };

    // Split the source code into manageable chunks for LLM processing.
    let chunks = chunk_rust_source(&source, config).context("failed to chunk Rust source")?;

    info!(chunks = chunks.len(), "File chunked successfully");

    // Log chunk boundaries for debugging/visibility.
    for chunk in &chunks {
        debug!(
            index = chunk.index,
            kind = %chunk.kind,
            lines = format!("{}-{}", chunk.start_line, chunk.end_line),
            chars = chunk.text.chars().count(),
            "Chunk boundary"
        );
    }

    if cli.dry_run {
        // If dry-run is enabled, we only simulate the process and print the chunk list
        // to stdout, avoiding any network calls or file modifications.
        for chunk in &chunks {
            println!(
                "  #{:03} {:>20} lines {}-{} chars={} bytes={}..{}",
                chunk.index,
                chunk.kind,
                chunk.start_line,
                chunk.end_line,
                chunk.text.chars().count(),
                chunk.start_byte,
                chunk.end_byte
            );
        }
        info!("Dry run enabled — Ollama was not called.");
        return Ok(());
    }

    // ----- Ollama client setup -----
    let client = OllamaClient::new(
        cli.ollama_url.clone(),
        cli.model.clone(),
        cli.chunk_timeout_seconds,
        cli.num_ctx,
    );

    // Perform a health check to ensure the specified model is available locally.
    if !cli.skip_health_check {
        let models = client.list_models().context("Ollama health check failed")?;
        if !models.iter().any(|m| m == &cli.model) {
            anyhow::bail!(
                "model '{}' was not found in Ollama. Available models: {}",
                cli.model,
                models.join(", ")
            );
        }
        info!(model = %cli.model, "Ollama reachable — model found");
    }

    // ----- Cache setup -----
    // Initialize the cache if the user has enabled caching.
    let cache = if cli.cache {
        Some(ChunkCache::new(
            cli.cache_dir.clone(),
            cli.cache_ttl_duration(),
        )?)
    } else {
        None
    };

    // ----- Processing loop initialization -----
    let mut run_log = RunLog::new(&cli.file, &cli.model, &cli.mode_string(), chunks.len());
    let mut report_items = Vec::new();
    let mut replacements = Vec::new();

    // Optional progress bar setup for better UX during long runs.
    let pb = if cli.progress {
        let bar = ProgressBar::new(chunks.len() as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta} remaining)",
                )
                .expect("invalid progress bar template")
                .progress_chars("█▓░"),
        );
        Some(bar)
    } else {
        None
    };

    for chunk in chunks.iter() {
        // Business rule: Filter chunks based on user input (index range).
        if let Some(only) = cli.only_chunk {
            if chunk.index != only {
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                continue;
            }
        }
        if let Some(from) = cli.from_chunk {
            if chunk.index < from {
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                continue;
            }
        }

        // Determine the prompt structure based on the selected mode (Analyze, Comment, Rewrite).
        let prompt = match cli.mode {
            Mode::Analyze => build_analysis_prompt(chunk, cli.profile),
            Mode::Comment | Mode::Rewrite => build_commenting_prompt(chunk, cli.profile),
        };

        let chunk_started = Instant::now();
        info!(
            chunk = chunk.index,
            total = chunks.len(),
            kind = %chunk.kind,
            lines = format!("{}-{}", chunk.start_line, chunk.end_line),
            "Processing chunk"
        );

        // Core logic: Get response, utilizing cache and retries if necessary.
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
                // Side effect: If in Rewrite mode, the generated content must be validated
                // against the original chunk structure to prevent invalid code.
                if cli.mode == Mode::Rewrite {
                    let validation = validate_generated_chunk(&response, &chunk.text);
                    if let Err(err) = validation {
                        let msg = format!("generated chunk failed validation: {err}");
                        run_log.push_chunk(ChunkRunStatus::failed(
                            chunk,
                            &msg,
                            chunk_started.elapsed(),
                        ));

                        // Business rule: If validation fails, check if we should skip the failure.
                        if cli.skip_failed {
                            warn!(chunk = chunk.index, error = %msg, "Keeping original chunk");
                            if let Some(ref pb) = pb {
                                pb.inc(1);
                            }
                            continue;
                        }
                        // Critical failure: Stop processing if validation fails and skipping is disabled.
                        anyhow::bail!("chunk #{}: {}", chunk.index, msg);
                    }
                    replacements.push((chunk.clone(), response.clone()));
                }

                // Collect results for the final report/output.
                report_items.push(AnalysisItem {
                    index: chunk.index,
                    kind: chunk.kind.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    source_preview: chunk.preview(24),
                    model_response: response,
                });

                run_log.push_chunk(ChunkRunStatus::ok(chunk, chunk_started.elapsed()));
                info!(
                    chunk = chunk.index,
                    elapsed_s = format!("{:.1}", chunk_started.elapsed().as_secs_f32()),
                    "Chunk OK"
                );
            }
            Err(err) => {
                let msg = err.to_string();
                run_log.push_chunk(ChunkRunStatus::failed(chunk, &msg, chunk_started.elapsed()));
                error!(chunk = chunk.index, error = %msg, "Chunk failed");

                // Failure handling: If we fail and are not allowed to skip,
                // we must write the log and exit gracefully.
                if !cli.skip_failed {
                    run_log.finish(started.elapsed());
                    run_log.write(&cli.run_log)?;
                    anyhow::bail!(
                        "stopped because chunk #{} failed. Use --skip-failed to continue.",
                        chunk.index
                    );
                }
            }
        }

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(ref pb) = pb {
        pb.finish_with_message("Done");
    }

    // ----- Output writing phase -----
    match cli.mode {
        Mode::Analyze | Mode::Comment => {
            // Outputting a structured markdown report summarizing analysis/comments.
            write_markdown_report(&cli.output, &cli.file, &report_items)
                .with_context(|| format!("failed to write report: {}", cli.output.display()))?;
            info!(path = %cli.output.display(), "Report written");
        }
        Mode::Rewrite => {
            // Outputting the fully rewritten source code.
            let rewritten = build_rewritten_source(&source, &replacements);
            fs::write(&cli.output, rewritten).with_context(|| {
                format!("failed to write rewritten file: {}", cli.output.display())
            })?;

            // Optional formatting step to ensure the output is idiomatic Rust.
            if cli.rustfmt {
                rustfmt_file_if_available(&cli.output);
            }
            info!(path = %cli.output.display(), "Rewritten file written");
        }
    }

    // Final logging: Write the complete run log detailing success/failure/timing.
    run_log.finish(started.elapsed());
    run_log.write(&cli.run_log)?;
    info!(
        path = %cli.run_log.display(),
        elapsed_s = format!("{:.1}", started.elapsed().as_secs_f32()),
        "Run log written"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Cache-aware generation with retries
// ---------------------------------------------------------------------------
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
    // Check cache first to avoid unnecessary API calls.
    if let Some(cache) = cache {
        // The cache key includes all parameters that define the generated content.
        if let Some(hit) = cache.get(model, mode, chunk_index, chunk_text, prompt)? {
            info!(chunk = chunk_index, "Loaded from cache");
            return Ok(hit);
        }
    }

    let mut last_err = None;
    // Attempt generation up to max_retries + 1 times.
    for attempt in 0..=max_retries {
        if attempt > 0 {
            warn!(
                chunk = chunk_index,
                attempt = attempt + 1,
                max_attempts = max_retries + 1,
                "Retrying chunk"
            );
        }

        match client.generate(prompt) {
            Ok(response) => {
                // Cache the successful response if a cache object was provided.
                if let Some(cache) = cache {
                    cache.put(model, mode, chunk_index, chunk_text, prompt, &response)?;
                }
                return Ok(response);
            }
            Err(err) => {
                // Store the error, which will be returned if all attempts fail.
                last_err = Some(err);
            }
        }
    }

    // Return the last encountered error if all attempts failed.
    Err(last_err.expect("at least one attempt should run"))
}
