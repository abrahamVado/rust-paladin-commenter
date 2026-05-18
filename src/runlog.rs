use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::chunker::CodeChunk;

/// Top-level JSON run log capturing timing, success/failure counts, and per-chunk status.
#[derive(Debug, Serialize)]
pub struct RunLog {
    version: u8,
    started_at: DateTime<Utc>,
    finished_at: Option<DateTime<Utc>>,
    duration_seconds: Option<f32>,
    file: String,
    model: String,
    mode: String,
    chunks_total: usize,
    chunks_ok: usize,
    chunks_failed: usize,
    chunks: Vec<ChunkRunStatus>,
}

/// Per-chunk outcome persisted in the run log.
#[derive(Debug, Serialize)]
pub struct ChunkRunStatus {
    index: usize,
    kind: String,
    start_line: usize,
    end_line: usize,
    status: String,
    duration_seconds: f32,
    error: Option<String>,
}

impl RunLog {
    pub fn new(file: &Path, model: &str, mode: &str, chunks_total: usize) -> Self {
        Self {
            version: 1,
            started_at: Utc::now(),
            finished_at: None,
            duration_seconds: None,
            file: file.display().to_string(),
            model: model.to_string(),
            mode: mode.to_string(),
            chunks_total,
            chunks_ok: 0,
            chunks_failed: 0,
            chunks: Vec::new(),
        }
    }

    pub fn push_chunk(&mut self, status: ChunkRunStatus) {
        if status.status == "ok" {
            self.chunks_ok += 1;
        } else {
            self.chunks_failed += 1;
        }
        self.chunks.push(status);
    }

    pub fn finish(&mut self, duration: Duration) {
        self.finished_at = Some(Utc::now());
        self.duration_seconds = Some(duration.as_secs_f32());
    }

    pub fn write(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
            .with_context(|| format!("failed to write run log: {}", path.display()))?;
        Ok(())
    }
}

impl ChunkRunStatus {
    pub fn ok(chunk: &CodeChunk, duration: Duration) -> Self {
        Self {
            index: chunk.index,
            kind: chunk.kind.to_string(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            status: "ok".to_string(),
            duration_seconds: duration.as_secs_f32(),
            error: None,
        }
    }

    pub fn failed(chunk: &CodeChunk, error: &str, duration: Duration) -> Self {
        Self {
            index: chunk.index,
            kind: chunk.kind.to_string(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            status: "failed".to_string(),
            duration_seconds: duration.as_secs_f32(),
            error: Some(error.to_string()),
        }
    }
}
