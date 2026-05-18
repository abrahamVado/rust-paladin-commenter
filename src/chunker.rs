use anyhow::{anyhow, Context, Result};
use tree_sitter::{Node, Parser};

use crate::chunk_kind::ChunkKind;

/// Configuration controlling how source code is split into semantic chunks.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Hard upper limit on characters per chunk.
    pub max_chars: usize,
    /// Soft target — adjacent chunks smaller than this are merged together.
    pub target_chars: usize,
}

/// A single semantic code chunk extracted from a Rust source file.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// 1-based index within the chunked file.
    pub index: usize,
    /// Semantic kind (function, struct, impl, gap, etc.).
    pub kind: crate::chunk_kind::ChunkKind,
    pub start_byte: usize,
    pub end_byte: usize,
    /// 1-based source line where the chunk starts.
    pub start_line: usize,
    /// 1-based source line where the chunk ends.
    pub end_line: usize,
    /// Full text of the chunk.
    pub text: String,
}

impl CodeChunk {
    pub fn preview(&self, max_lines: usize) -> String {
        self.text
            .lines()
            .take(max_lines)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug, Clone)]
struct CandidateChunk {
    kind: crate::chunk_kind::ChunkKind,
    start_byte: usize,
    end_byte: usize,
    start_line: usize,
    end_line: usize,
}

/// Parse a Rust source string and split it into semantic chunks respecting AST boundaries.
///
/// Uses tree-sitter to identify top-level items (functions, structs, impls, …),
/// inserts gap chunks for inter-item code, and merges small neighbours up to
/// `config.target_chars`.
pub fn chunk_rust_source(source: &str, config: ChunkConfig) -> Result<Vec<CodeChunk>> {
    if config.max_chars < 1000 {
        return Err(anyhow!("max_chars should be at least 1000"));
    }
    if config.target_chars > config.max_chars {
        return Err(anyhow!("target_chars cannot be greater than max_chars"));
    }

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .context("failed to load tree-sitter Rust grammar")?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow!("tree-sitter failed to parse source"))?;

    let root = tree.root_node();
    let mut candidates = Vec::new();
    collect_semantic_candidates(root, source, config.max_chars, &mut candidates);

    if candidates.is_empty() {
        candidates.push(CandidateChunk::from_node(root, "source_file"));
    }

    candidates.sort_by_key(|c| c.start_byte);
    let candidates = remove_nested_duplicates(candidates);
    let candidates = include_gaps_as_chunks(source, candidates);
    let merged = merge_small_neighbors(source, candidates, config);

    Ok(merged
        .into_iter()
        .enumerate()
        .map(|(i, c)| CodeChunk {
            index: i + 1,
            kind: c.kind,
            start_byte: c.start_byte,
            end_byte: c.end_byte,
            start_line: c.start_line,
            end_line: c.end_line,
            text: source[c.start_byte..c.end_byte].to_string(),
        })
        .collect())
}

fn collect_semantic_candidates(
    node: Node,
    source: &str,
    max_chars: usize,
    out: &mut Vec<CandidateChunk>,
) {
    let kind = node.kind();
    let is_semantic = is_semantic_boundary(kind);
    let char_len = node_text(source, node).chars().count();

    if is_semantic && char_len <= max_chars {
        out.push(CandidateChunk::from_node(node, kind));
        return;
    }

    // If an impl/module/function is too large, descend into its named children.
    // This avoids arbitrary character cuts while still escaping huge blocks.
    let mut cursor = node.walk();
    let mut found_child = false;
    for child in node.children(&mut cursor) {
        if child.is_named() {
            found_child = true;
            collect_semantic_candidates(child, source, max_chars, out);
        }
    }

    // Last-resort fallback: if we could not find smaller semantic nodes and this node is too big,
    // keep it as one chunk rather than corrupting code with a random split.
    if is_semantic && !found_child {
        out.push(CandidateChunk::from_node(node, kind));
    }
}

fn is_semantic_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "impl_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "mod_item"
            | "macro_definition"
            | "const_item"
            | "static_item"
            | "type_item"
            | "use_declaration"
    )
}

impl CandidateChunk {
    fn from_node(node: Node, fallback_kind: &str) -> Self {
        Self {
            kind: ChunkKind::from(fallback_kind),
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
        }
    }

    fn char_len(&self, source: &str) -> usize {
        source[self.start_byte..self.end_byte].chars().count()
    }
}

fn node_text<'a>(source: &'a str, node: Node) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

fn remove_nested_duplicates(chunks: Vec<CandidateChunk>) -> Vec<CandidateChunk> {
    let mut result: Vec<CandidateChunk> = Vec::new();

    'outer: for chunk in chunks {
        for existing in &result {
            if chunk.start_byte >= existing.start_byte && chunk.end_byte <= existing.end_byte {
                continue 'outer;
            }
        }
        result.push(chunk);
    }

    result
}

fn include_gaps_as_chunks(source: &str, chunks: Vec<CandidateChunk>) -> Vec<CandidateChunk> {
    let mut result = Vec::new();
    let mut cursor = 0usize;

    for chunk in chunks {
        if cursor < chunk.start_byte {
            let gap = &source[cursor..chunk.start_byte];
            if !gap.trim().is_empty() {
                result.push(gap_chunk(source, cursor, chunk.start_byte));
            }
        }
        cursor = chunk.end_byte;
        result.push(chunk);
    }

    if cursor < source.len() {
        let gap = &source[cursor..];
        if !gap.trim().is_empty() {
            result.push(gap_chunk(source, cursor, source.len()));
        }
    }

    result.sort_by_key(|c| c.start_byte);
    result
}

fn gap_chunk(source: &str, start_byte: usize, end_byte: usize) -> CandidateChunk {
    CandidateChunk {
        kind: ChunkKind::Gap,
        start_byte,
        end_byte,
        start_line: byte_to_line(source, start_byte),
        end_line: byte_to_line(source, end_byte),
    }
}

fn byte_to_line(source: &str, byte: usize) -> usize {
    source[..byte].bytes().filter(|b| *b == b'\n').count() + 1
}

fn merge_small_neighbors(
    source: &str,
    chunks: Vec<CandidateChunk>,
    config: ChunkConfig,
) -> Vec<CandidateChunk> {
    let mut result = Vec::new();
    let mut current: Option<CandidateChunk> = None;

    for next in chunks {
        match current.take() {
            None => current = Some(next),
            Some(mut existing) => {
                let combined_chars = source[existing.start_byte..next.end_byte].chars().count();
                let existing_chars = existing.char_len(source);
                let next_chars = next.char_len(source);

                let should_merge = existing.kind == ChunkKind::Gap
                    || next.kind == ChunkKind::Gap
                    || (existing_chars < config.target_chars
                        && next_chars < config.target_chars
                        && combined_chars <= config.max_chars);

                if should_merge && combined_chars <= config.max_chars {
                    existing.end_byte = next.end_byte;
                    existing.end_line = next.end_line;
                    existing.kind = merge_kind(&existing.kind, &next.kind);
                    current = Some(existing);
                } else {
                    result.push(existing);
                    current = Some(next);
                }
            }
        }
    }

    if let Some(last) = current {
        result.push(last);
    }

    result
}

fn merge_kind(a: &crate::chunk_kind::ChunkKind, b: &crate::chunk_kind::ChunkKind) -> crate::chunk_kind::ChunkKind {
    use crate::chunk_kind::ChunkKind::*;
    if a == b {
        a.clone()
    } else if matches!(a, Gap) {
        b.clone()
    } else if matches!(b, Gap) {
        a.clone()
    } else {
        Mixed
    }
}
