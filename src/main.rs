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
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::cache::ChunkCache;
use crate::chunker::{chunk_rust_source, ChunkConfig};
use crate::ollama::OllamaClient;
use crate::prompt::{build_analysis_prompt, build_commenting_prompt, PromptProfile};
use crate::report::{format_analysis_report, AnalysisItem};
use crate::rewrite::{
    build_rewritten_source, normalize_generated_chunk, rustfmt_file_if_available,
    validate_generated_chunk,
};
use crate::runlog::{ChunkRunStatus, RunLog};

/// Defines the operational mode of the commenter.
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum Mode {
    /// Analyze chunks and write plain text output. This mode is read-only and generates
    /// a report detailing suggested changes without applying them.
    Analyze,
    /// Ask the model to return commented code chunks and offer to apply them in place.
    /// This mode is suitable for reviewing and applying comments manually.
    Comment,
    /// Ask the model to return commented code chunks and offer to apply them in place.
    /// This mode attempts to rewrite the source file directly with the model's suggestions.
    Rewrite,
}

/// Defines the minimum logging level for the application.
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Command line arguments and configuration for the Paladin Commenter tool.
#[derive(Debug, Parser)]
#[command(name = "paladin-commenter", version)]
#[command(
    about = "Context-aware Rust file chunker that sends semantic chunks to a local Ollama/Gemma model"
)]
struct Cli {
    /// Rust source file to process. This file must exist and be readable.
    #[arg(short, long)]
    file: PathBuf,

    /// Ollama model name. Specifies which local model to use for generation.
    #[arg(short, long, default_value = "gemma4:e4b")]
    model: String,

    /// Ollama base URL. The endpoint where the model service is running.
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    ollama_url: String,

    /// Maximum characters per semantic chunk. This is a hard limit to prevent API failures.
    #[arg(long, default_value_t = 6000)]
    max_chars: usize,

    /// Soft target size. Small chunks are merged until they approach this size, optimizing context usage.
    #[arg(long, default_value_t = 3500)]
    target_chars: usize,

    /// Output file used by analyze mode. The generated report will overwrite this file.
    #[arg(short, long, default_value = "paladin-analysis.txt")]
    output: PathBuf,

    /// Processing mode. Determines if the output is analyzed, commented, or rewritten.
    #[arg(long, value_enum, default_value_t = Mode::Comment)]
    mode: Mode,

    /// Prompt profile. Controls the style and depth of the generated comments (e.g., maintainer vs. doc comments).
    #[arg(long, value_enum, default_value_t = PromptProfile::MaintainerComments)]
    profile: PromptProfile,

    /// Print chunk boundaries without calling Ollama. Useful for debugging chunking logic.
    #[arg(long)]
    dry_run: bool,

    /// Enable chunk response cache. If true, the tool will check the cache before calling the model.
    #[arg(long)]
    cache: bool,

    /// Cache directory. Location where cached chunk results are stored.
    #[arg(long, default_value = ".paladin-cache")]
    cache_dir: PathBuf,

    /// Cache time-to-live in seconds. Entries older than this are treated as misses.
    /// Set to 0 to disable TTL (keep entries forever).
    #[arg(long, default_value_t = 0)]
    cache_ttl: u64,

    /// Start processing at this chunk index, 1-based. Allows resuming interrupted runs.
    #[arg(long)]
    from_chunk: Option<usize>,

    /// Process only this chunk index, 1-based. Useful for testing specific parts of the file.
    #[arg(long)]
    only_chunk: Option<usize>,

    /// Continue processing even if a chunk fails (e.g., due to API error or model refusal).
    #[arg(long)]
    skip_failed: bool,

    /// Maximum retries per chunk after the first attempt. Handles transient network or API errors.
    #[arg(long, default_value_t = 1)]
    max_retries: usize,

    /// HTTP timeout per chunk request in seconds. Limits how long the tool waits for a model response.
    #[arg(long, default_value_t = 240)]
    chunk_timeout_seconds: u64,

    /// Number of context tokens requested from Ollama. Determines the maximum input size for the model.
    #[arg(long, default_value_t = 8192)]
    num_ctx: u32,

    /// Run rustfmt after applying comments if rustfmt is installed. Ensures the output code is formatted correctly.
    #[arg(long, default_value_t = true)]
    rustfmt: bool,

    /// Write a machine-readable JSON run log. Records status and metadata for reproducibility.
    #[arg(long, default_value = "paladin-run.json")]
    run_log: PathBuf,

    /// Skip the Ollama model availability check. Use with caution, as connection failures may occur.
    #[arg(long)]
    skip_health_check: bool,

    /// Logging verbosity. Controls the amount of information printed to stdout/stderr.
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,

    /// Suppress all output except errors (shorthand for --log-level error).
    #[arg(long)]
    quiet: bool,
}

impl Cli {
    /// Validates the business constraints defined by the command-line arguments.
    ///
    /// This function enforces rules regarding character limits and mode consistency.
    ///
    /// # Errors
    /// Returns an `anyhow::Error` if the configured values violate the defined constraints.
    fn validate(&self) -> Result<()> {
        // Business rule: The maximum allowed characters must be at least 1000.
        if self.max_chars < 1000 {
            anyhow::bail!("--max-chars must be at least 1000 (got {})", self.max_chars);
        }
        // Business rule: The target character count cannot exceed the maximum allowed characters.
        if self.target_chars > self.max_chars {
            anyhow::bail!(
                "--target-chars ({}) cannot exceed --max-chars ({})",
                self.target_chars,
                self.max_chars
            );
        }
        Ok(())
    }

    /// Converts the internal `Mode` enum into its corresponding string representation.
    fn mode_string(&self) -> String {
        match self.mode {
            Mode::Analyze => "analyze".to_string(),
            Mode::Comment => "comment".to_string(),
            Mode::Rewrite => "rewrite".to_string(),
        }
    }

    /// Determines the effective logging level based on both the explicit `log_level`
    /// and the `quiet` flag.
    ///
    /// If `quiet` is true, logging is suppressed to only critical errors, regardless
    /// of the configured `log_level`.
    fn effective_log_level(&self) -> tracing::Level {
        if self.quiet {
            // Side effect: If quiet mode is enabled, we only log errors, overriding
            // any potentially higher configured log level.
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

    /// Converts the stored cache TTL (Time To Live) integer into an `Option<Duration>`.
    ///
    /// Returns `None` if the TTL is zero, indicating no caching should occur.
    fn cache_ttl_duration(&self) -> Option<Duration> {
        if self.cache_ttl == 0 {
            None
        } else {
            Some(Duration::from_secs(self.cache_ttl))
        }
    }
}

/// Initializes the global tracing subscriber for structured logging.
///
/// This function sets up the logging backend for the entire application.
///
/// # Arguments
/// * `level` - The minimum logging level (e.g., `tracing::Level::INFO`) to be recorded.
fn setup_logging(level: tracing::Level) {
    use tracing_subscriber::fmt;
    let builder = fmt::Subscriber::builder()
        .with_max_level(level)
        .with_target(false) // Do not include module path in logs
        .with_ansi(true);
    // Side effect: This call sets the global default logging subscriber.
    // Failure here indicates a critical environment setup issue.
    tracing::subscriber::set_global_default(builder.finish())
        .expect("failed to set tracing subscriber");
}
fn main() -> Result<()> {
    // Parse command line arguments and initialize the application state.
    let cli = Cli::parse();
    // Set up logging based on the user-provided effective log level.
    setup_logging(cli.effective_log_level());
    // Validate CLI arguments (e.g., file existence, required flags).
    cli.validate()?;

    let started = Instant::now();
    info!(
        file = %cli.file.display(),
        model = %cli.model,
        mode = %cli.mode_string(),
        "Starting paladin-commenter"
    );

    // Read the entire source file content into a string. This is the primary input source.
    let source = fs::read_to_string(&cli.file)
        .with_context(|| format!("failed to read file: {}", cli.file.display()))?;

    // Configure chunk processing parameters based on CLI input.
    let config = ChunkConfig {
        max_chars: cli.max_chars,
        target_chars: cli.target_chars,
    };

    // Split the source code into manageable chunks for LLM processing.
    let chunks = chunk_rust_source(&source, config).context("failed to chunk Rust source")?;
    info!(chunks = chunks.len(), "File chunked successfully");

    // Log chunk boundaries for debugging purposes.
    for chunk in &chunks {
        debug!(
            index = chunk.index,
            kind = %chunk.kind,
            lines = format!("{}-{}", chunk.start_line, chunk.end_line),
            chars = chunk.text.chars().count(),
            "Chunk boundary"
        );
    }

    // Handle dry run mode: process chunks and report results without calling the LLM or writing changes.
    if cli.dry_run {
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
        info!("Dry run enabled - Ollama was not called.");
        return Ok(());
    }

    // Initialize the Ollama client connection.
    let client = OllamaClient::new(
        cli.ollama_url.clone(),
        cli.model.clone(),
        cli.chunk_timeout_seconds,
        cli.num_ctx,
    );

    // Perform a health check against the Ollama service to ensure the specified model exists.
    if !cli.skip_health_check {
        let models = client.list_models().context("Ollama health check failed")?;
        if !models.iter().any(|m| m == &cli.model) {
            // Critical failure: The required model is not available on the Ollama server.
            anyhow::bail!(
                "model '{}' was not found in Ollama. Available models: {}",
                cli.model,
                models.join(", ")
            );
        }
        info!(model = %cli.model, "Ollama reachable - model found");
    }

    // Initialize the cache if requested. This handles local caching of LLM responses
    // to avoid redundant API calls and speed up subsequent runs.
    let cache = if cli.cache {
        Some(ChunkCache::new(
            cli.cache_dir.clone(),
            cli.cache_ttl_duration(),
        )?)
    } else {
        None
    };

    // Initialize logging structures for the run report.
    let mut run_log = RunLog::new(&cli.file, &cli.model, &cli.mode_string(), chunks.len());
    let mut report_items = Vec::new();
    // Stores (original_chunk, generated_response) pairs for rewriting the source file.
    let mut replacements = Vec::new();

    // Main processing loop: Iterate over all source code chunks.
    for chunk in chunks.iter() {
        // Apply filtering logic based on CLI arguments (only_chunk, from_chunk).
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

        // Determine the prompt structure based on the desired operation (Analyze vs Comment/Rewrite).
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

        // Attempt to get a response from cache or generate a new one via the LLM client.
        match get_or_generate(
            &client,
            cache.as_ref(),
            &cli.model,
            &cli.mode_string(),
            chunk.index,
            &chunk.text,
            &prompt,
            cli.max_retries,
        ) {
            Ok(response) => {
                // If the mode requires modification (Comment or Rewrite), validate the LLM output.
                if matches!(cli.mode, Mode::Comment | Mode::Rewrite) {
                    let normalized_response = normalize_generated_chunk(&response);
                    let validation = validate_generated_chunk(&normalized_response, &chunk.text);
                    if let Err(err) = validation {
                        let msg = format!("generated chunk failed validation: {err}");
                        run_log.push_chunk(ChunkRunStatus::failed(
                            chunk,
                            &msg,
                            chunk_started.elapsed(),
                        ));
                        // If skipping failed chunks, log a warning and continue to the next chunk.
                        if cli.skip_failed {
                            warn!(chunk = chunk.index, error = %msg, "Keeping original chunk");
                            continue;
                        }
                        // Otherwise, fail the entire process immediately.
                        anyhow::bail!("chunk #{}: {}", chunk.index, msg);
                    }
                    // Store the successful replacement pair.
                    replacements.push((chunk.clone(), normalized_response.clone()));

                    // Record the sanitized response for the final analysis report.
                    report_items.push(AnalysisItem {
                        index: chunk.index,
                        kind: chunk.kind.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        source_preview: chunk.preview(24),
                        model_response: normalized_response,
                    });
                } else {
                    // Record the successful response for the final analysis report.
                    report_items.push(AnalysisItem {
                        index: chunk.index,
                        kind: chunk.kind.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        source_preview: chunk.preview(24),
                        model_response: response,
                    });
                }

                // Log successful processing status.
                run_log.push_chunk(ChunkRunStatus::ok(chunk, chunk_started.elapsed()));
                info!(
                    chunk = chunk.index,
                    elapsed_s = format!("{:.1}", chunk_started.elapsed().as_secs_f32()),
                    "Chunk OK"
                );
            }
            Err(err) => {
                let msg = err.to_string();
                // Log failure status.
                run_log.push_chunk(ChunkRunStatus::failed(chunk, &msg, chunk_started.elapsed()));
                error!(chunk = chunk.index, error = %msg, "Chunk failed");
                // Failure handling: If not skipping failed chunks, write the partial log and exit.
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
    }

    // Final output phase: Write results based on the execution mode.
    match cli.mode {
        Mode::Analyze => {
            // Write a comprehensive analysis report to the specified output file.
            let analysis_output = format_analysis_report(&cli.file, &report_items);
            fs::write(&cli.output, analysis_output).with_context(|| {
                format!("failed to write analysis output: {}", cli.output.display())
            })?;
            info!(path = %cli.output.display(), "Analysis written");
        }
        Mode::Comment | Mode::Rewrite => {
            // Reconstruct the entire source file using the original chunks and the generated replacements.
            let rewritten = build_rewritten_source(&source, &replacements);

            // Prompt the user before applying changes to the original file (safety mechanism).
            if prompt_apply_changes(&cli.file)? {
                // Write the rewritten content back to the original file path.
                fs::write(&cli.file, rewritten)
                    .with_context(|| format!("failed to update file: {}", cli.file.display()))?;

                // Run rustfmt if requested, ensuring the modified code adheres to standard formatting.
                if cli.rustfmt {
                    rustfmt_file_if_available(&cli.file);
                }
                info!(path = %cli.file.display(), "Changes applied");
            } else {
                info!(path = %cli.file.display(), "Changes discarded");
            }
        }
    }

    // Final logging: Write the complete run log and report total execution time.
    run_log.finish(started.elapsed());
    run_log.write(&cli.run_log)?;
    info!(
        path = %cli.run_log.display(),
        elapsed_s = format!("{:.1}", started.elapsed().as_secs_f32()),
        "Run log written"
    );

    Ok(())
}
fn prompt_apply_changes(path: &std::path::Path) -> Result<bool> {
    // Prompt the user for confirmation before applying changes to the file.
    print!("Apply generated comments to {}? [y/N]: ", path.display());
    io::stdout().flush().context("failed to flush prompt")?;

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("failed to read confirmation")?;

    // Check if the user input matches 'y' or 'yes' (case-insensitive).
    let normalized = input.trim().to_ascii_lowercase();
    Ok(matches!(normalized.as_str(), "y" | "yes"))
}

/// Retrieves a result for a given chunk, utilizing a cache if available.
///
/// If the result is not found in the cache, it attempts to generate it using the
/// provided Ollama client, implementing a retry mechanism for transient failures.
///
/// # Arguments
/// * `client` - The client used to interact with the LLM (Ollama).
/// * `cache` - Optional cache to store and retrieve previous results.
/// * `model` - The name of the LLM model used.
/// * `mode` - The operational mode (e.g., "commenting", "refactoring").
/// * `chunk_index` - The index of the code chunk being processed.
/// * `chunk_text` - The source code text for the current chunk.
/// * `prompt` - The full prompt sent to the LLM.
/// * `max_retries` - The maximum number of times to retry the API call upon failure.
///
/// # Returns
/// A `Result<String>` containing the generated text on success, or an error if all attempts fail.
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
    // 1. Cache Lookup: Check if the result is already available in the cache.
    if let Some(cache) = cache {
        if let Some(hit) = cache.get(model, mode, chunk_index, chunk_text, prompt)? {
            info!(chunk = chunk_index, "Loaded from cache");
            return Ok(hit);
        }
    }

    let mut last_err = None;
    // 2. Generation Loop: Attempt to generate the result, retrying on failure.
    // The loop runs from attempt 0 up to and including max_retries.
    for attempt in 0..=max_retries {
        if attempt > 0 {
            // Log a warning if this is a retry attempt.
            warn!(
                chunk = chunk_index,
                attempt = attempt + 1,
                max_attempts = max_retries + 1,
                "Retrying chunk"
            );
        }

        match client.generate(prompt) {
            Ok(response) => {
                // Success: Cache the result if a cache object is provided.
                if let Some(cache) = cache {
                    cache.put(model, mode, chunk_index, chunk_text, prompt, &response)?;
                }
                return Ok(response);
            }
            Err(err) => {
                // Failure: Record the error and continue to the next attempt if available.
                last_err = Some(err);
            }
        }
    }

    // If all attempts fail, return the last encountered error.
    Err(last_err.expect("at least one attempt should run"))
}
