use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::process::Command;
use tree_sitter::Parser;

use crate::chunker::CodeChunk;

pub fn validate_generated_chunk(generated: &str, original: &str) -> Result<()> {
    if generated.trim().is_empty() {
        return Err(anyhow!("model returned an empty chunk"));
    }

    if generated.contains("```") {
        return Err(anyhow!("model returned Markdown fences instead of raw Rust code"));
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

pub fn rustfmt_file_if_available(path: &Path) {
    let status = Command::new("rustfmt").arg(path).status();
    match status {
        Ok(status) if status.success() => println!("rustfmt applied to {}", path.display()),
        Ok(status) => eprintln!("rustfmt exited with status {} for {}", status, path.display()),
        Err(_) => eprintln!("rustfmt not available. Skipping formatting."),
    }
}
