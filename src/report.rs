use std::path::Path;

use crate::chunk_kind::ChunkKind;

#[derive(Debug, Clone)]
pub struct AnalysisItem {
    pub index: usize,
    pub kind: ChunkKind,
    pub start_line: usize,
    pub end_line: usize,
    pub source_preview: String,
    pub model_response: String,
}

/// Format analysis results as plain text.
pub fn format_analysis_report(input_file: &Path, items: &[AnalysisItem]) -> String {
    let mut output = String::new();

    output.push_str(&format!("Input file: {}\n", input_file.display()));
    output.push_str(&format!("Total processed chunks: {}\n\n", items.len()));

    for item in items {
        output.push_str(&format!(
            "Chunk #{:03} [{}] lines {}-{}\n",
            item.index, item.kind, item.start_line, item.end_line
        ));
        output.push_str("Source preview:\n");
        output.push_str(&item.source_preview);
        output.push_str("\n\nModel response:\n");
        output.push_str(&item.model_response);
        output.push_str("\n\n");
    }

    output
}
