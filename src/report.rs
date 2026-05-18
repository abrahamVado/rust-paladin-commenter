use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AnalysisItem {
    pub index: usize,
    pub kind: String,
    pub start_line: usize,
    pub end_line: usize,
    pub source_preview: String,
    pub model_response: String,
}

pub fn write_markdown_report(output: &Path, input_file: &Path, items: &[AnalysisItem]) -> Result<()> {
    let mut markdown = String::new();

    markdown.push_str("# Paladin Code Analysis Report\n\n");
    markdown.push_str(&format!("Input file: `{}`\n\n", input_file.display()));
    markdown.push_str(&format!("Total processed chunks: `{}`\n\n", items.len()));

    for item in items {
        markdown.push_str(&format!(
            "## Chunk #{:03} — `{}` lines {}-{}\n\n",
            item.index, item.kind, item.start_line, item.end_line
        ));

        markdown.push_str("### Source preview\n\n");
        markdown.push_str("```rust\n");
        markdown.push_str(&item.source_preview);
        markdown.push_str("\n```\n\n");

        markdown.push_str("### Model response\n\n");
        markdown.push_str(&item.model_response);
        markdown.push_str("\n\n---\n\n");
    }

    fs::write(output, markdown)
        .with_context(|| format!("failed to write {}", output.display()))?;

    Ok(())
}
