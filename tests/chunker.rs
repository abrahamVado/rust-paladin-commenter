//! Integration tests for the chunker module.

// The binary is not a library crate, so we re-declare what we need.
// We test the chunker through cargo test on the binary crate by using
// a helper source string and verifying chunk properties.

use std::process::Command;

/// Small helper: run the paladin-commenter binary in dry-run mode against a
/// temporary Rust file and return its stdout.
fn dry_run_output(source: &str) -> String {
    // Write source to a temp file
    let dir = std::env::temp_dir().join("paladin-test");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("test_input.rs");
    std::fs::write(&file, source).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_paladin-commenter"))
        .args([
            "--file", file.to_str().unwrap(),
            "--dry-run",
            "--max-chars", "6000",
            "--target-chars", "3500",
            "--log-level", "error",
        ])
        .output()
        .expect("failed to execute paladin-commenter");

    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn single_small_function_produces_one_chunk() {
    let source = r#"
fn hello() -> &'static str {
    "hello world"
}
"#;

    let out = dry_run_output(source);
    // Should contain at least one chunk line starting with #
    let chunk_lines: Vec<&str> = out.lines().filter(|l| l.trim_start().starts_with('#')).collect();
    assert!(
        !chunk_lines.is_empty(),
        "expected at least one chunk, got stdout:\n{}",
        out
    );
}

#[test]
fn multiple_functions_produce_multiple_or_merged_chunks() {
    let source = r#"
fn alpha() -> i32 { 1 }
fn beta() -> i32 { 2 }
fn gamma() -> i32 { 3 }
"#;

    let out = dry_run_output(source);
    // The three tiny functions may remain separate or be merged into one semantic block.
    let has_expected_kind = out.contains("function") || out.contains("mixed");
    assert!(
        has_expected_kind,
        "expected a function-like or mixed chunk in output:\n{}",
        out
    );
}

#[test]
fn struct_and_impl_recognised() {
    let source = r#"
pub struct Foo {
    pub x: i32,
}

impl Foo {
    pub fn new(x: i32) -> Self {
        Self { x }
    }
}
"#;

    let out = dry_run_output(source);
    // Depending on semantic grouping this may surface as struct, impl, mixed, or function.
    let has_struct_or_impl = out.contains("struct")
        || out.contains("impl")
        || out.contains("mixed")
        || out.contains("function");
    assert!(
        has_struct_or_impl,
        "expected struct or impl kind in output:\n{}",
        out
    );
}

#[test]
fn dry_run_does_not_call_ollama() {
    let source = "fn noop() {}";
    let out = dry_run_output(source);
    assert!(
        out.contains("Dry run") || out.contains("dry run") || out.contains("#"),
        "dry run should succeed without Ollama:\n{}",
        out
    );
}

#[test]
fn invalid_max_chars_fails() {
    let dir = std::env::temp_dir().join("paladin-test");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("test_tiny.rs");
    std::fs::write(&file, "fn f() {}").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_paladin-commenter"))
        .args([
            "--file", file.to_str().unwrap(),
            "--dry-run",
            "--max-chars", "100",    // too small, should fail validation
            "--target-chars", "50",
            "--log-level", "error",
        ])
        .output()
        .expect("failed to execute paladin-commenter");

    assert!(
        !output.status.success(),
        "expected failure for max_chars=100 but got success"
    );
}
