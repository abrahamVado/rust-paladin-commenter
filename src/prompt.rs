use clap::ValueEnum;

use crate::chunker::CodeChunk;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum PromptProfile {
    /// General explanations.
    Explain,
    /// Comments that help maintainers and future LLM analysis.
    MaintainerComments,
    /// Focus on risky logic and security-sensitive areas.
    Security,
    /// Focus on architecture, responsibilities, and boundaries.
    Architecture,
}

pub fn build_analysis_prompt(chunk: &CodeChunk, profile: PromptProfile) -> String {
    let profile_text = profile_instructions(profile);
    format!(
        r#"You are a senior Rust engineer reviewing one complete semantic code chunk.

Profile:
{profile_text}

Task:
1. Explain what this code does.
2. Identify public APIs, important structs, functions, traits, and side effects.
3. Identify places where comments would help future maintainers and future LLM code analysis.
4. Suggest concise Rust doc comments or inline comments.
5. Do not rewrite the whole file.
6. Do not invent context outside this chunk.
7. If the chunk is incomplete, say exactly what context is missing.

Chunk metadata:
- chunk_index: {index}
- kind: {kind}
- lines: {start_line}-{end_line}

Code chunk:
```rust
{code}
```
"#,
        profile_text = profile_text,
        index = chunk.index,
        kind = chunk.kind,
        start_line = chunk.start_line,
        end_line = chunk.end_line,
        code = chunk.text
    )
}

pub fn build_commenting_prompt(chunk: &CodeChunk, profile: PromptProfile) -> String {
    let profile_text = profile_instructions(profile);
    format!(
        r#"You are a senior Rust engineer adding useful comments to code.

Return ONLY valid Rust code for this chunk.

Profile:
{profile_text}

Rules:
- Preserve the original behavior exactly.
- Preserve function names, types, visibility, attributes, derives, and signatures.
- Add concise comments only where they improve understanding.
- Keep existing comments that are already clear and useful.
- Improve existing comments when they are vague, outdated, redundant, or missing important context.
- Prefer Rust doc comments `///` for public functions, structs, enums, traits, and methods.
- Use inline `//` comments only for non-obvious logic.
- Comment architecture, business rules, side effects, I/O, security-sensitive logic, parsing, retries, cache behavior, and failure modes.
- Do not over-comment obvious code.
- Do not wrap the result in Markdown fences.
- Do not explain your changes outside the code.
- Do not remove good existing comments.
- Do not add TODOs unless the original code already implies incomplete behavior.

Chunk metadata:
- chunk_index: {index}
- kind: {kind}
- lines: {start_line}-{end_line}

Original code:
{code}
"#,
        profile_text = profile_text,
        index = chunk.index,
        kind = chunk.kind,
        start_line = chunk.start_line,
        end_line = chunk.end_line,
        code = chunk.text
    )
}

fn profile_instructions(profile: PromptProfile) -> &'static str {
    match profile {
        PromptProfile::Explain => {
            "Focus on clear explanations of purpose, inputs, outputs, and control flow."
        }
        PromptProfile::MaintainerComments => {
            "Focus on comments that help humans and future LLMs understand architecture, business rules, side effects, and non-obvious implementation decisions."
        }
        PromptProfile::Security => {
            "Focus on security-sensitive behavior: parsing, auth, tokens, secrets, external calls, file system writes, unsafe assumptions, and error handling."
        }
        PromptProfile::Architecture => {
            "Focus on module responsibility, boundaries, dependencies, data flow, and how this chunk fits into the larger system."
        }
    }
}
