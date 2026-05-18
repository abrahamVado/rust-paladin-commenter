use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::process::Command;
use tracing::{info, warn};
use tree_sitter::Parser;

use crate::chunker::CodeChunk;

/// Normalize common LLM formatting mistakes before validation.
///
/// Today this mainly strips a single outer Markdown code fence such as
/// ```rust ... ``` when the model otherwise returned valid Rust code.
pub fn normalize_generated_chunk(generated: &str) -> String {
    let trimmed = generated.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }

    let mut lines = trimmed.lines();
    let Some(first_line) = lines.next() else {
        return trimmed.to_string();
    };

    if !first_line.trim_start().starts_with("```") {
        return trimmed.to_string();
    }

    let remaining: Vec<&str> = lines.collect();
    let Some(last_line) = remaining.last() else {
        return trimmed.to_string();
    };

    if !last_line.trim().starts_with("```") {
        return trimmed.to_string();
    }

    remaining[..remaining.len().saturating_sub(1)]
        .join("\n")
        .trim()
        .to_string()
}

/// Validate that a model-generated chunk is safe to substitute for the original.
///
/// Checks for:
/// - Empty output
/// - Markdown fences (the model was asked to return raw code)
/// - Output that is suspiciously larger than the original (>3×)
/// - Rust parse errors according to tree-sitter
pub fn validate_generated_chunk(generated: &str, original: &str) -> Result<()> {
    if generated.trim().is_empty() {
        return Err(anyhow!("model returned an empty chunk"));
    }

    if generated.contains("```") {
        return Err(anyhow!(
            "model returned Markdown fences instead of raw Rust code"
        ));
    }

    let generated_chars = generated.chars().count();
    let original_chars = original.chars().count().max(1);
    if generated_chars > original_chars * 3 {
        return Err(anyhow!(
            "model output is more than 3x larger than original chunk; refusing unsafe replacement"
        ));
    }

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .context("failed to load tree-sitter Rust grammar")?;

    let tree = parser
        .parse(generated, None)
        .ok_or_else(|| anyhow!("tree-sitter failed to parse generated chunk"))?;

    if tree.root_node().has_error() {
        return Err(anyhow!("generated chunk contains Rust parse errors"));
    }

    Ok(())
}

/// Build the final source by splicing model-generated replacements into the
/// original source at matching byte ranges.
pub fn build_rewritten_source(source: &str, replacements: &[(CodeChunk, String)]) -> String {
    let mut sorted = replacements.to_vec();
    sorted.sort_by_key(|(chunk, _)| chunk.start_byte);

    let mut output = String::new();
    let mut cursor = 0usize;

    for (chunk, replacement) in sorted {
        if cursor < chunk.start_byte {
            output.push_str(&source[cursor..chunk.start_byte]);
        }
        output.push_str(replacement.trim_end());
        output.push('\n');
        cursor = chunk.end_byte;
    }

    if cursor < source.len() {
        output.push_str(&source[cursor..]);
    }

    output
}

/// Run `rustfmt` on the given file if the binary is available on `$PATH`.
pub fn rustfmt_file_if_available(path: &Path) {
    let status = Command::new("rustfmt").arg(path).status();
    match status {
        Ok(s) if s.success() => info!(path = %path.display(), "rustfmt applied"),
        Ok(s) => warn!(path = %path.display(), code = %s, "rustfmt exited with non-zero status"),
        Err(_) => warn!("rustfmt not available — skipping formatting"),
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_generated_chunk;

    #[test]
    fn strips_rust_fences() {
        let input = "```rust\nfn demo() {}\n```";
        assert_eq!(normalize_generated_chunk(input), "fn demo() {}");
    }

    #[test]
    fn strips_plain_fences() {
        let input = "```\nfn demo() {}\n```";
        assert_eq!(normalize_generated_chunk(input), "fn demo() {}");
    }

    #[test]
    fn leaves_unfenced_code_alone() {
        let input = "fn demo() {}";
        assert_eq!(normalize_generated_chunk(input), "fn demo() {}");
    }
}
