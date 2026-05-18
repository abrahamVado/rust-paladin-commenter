use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ChunkCache {
    dir: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    model: String,
    mode: String,
    chunk_index: usize,
    chunk_hash: String,
    prompt_hash: String,
    response: String,
}

impl ChunkCache {
    pub fn new(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create cache dir: {}", dir.display()))?;
        Ok(Self { dir })
    }

    pub fn get(
        &self,
        model: &str,
        mode: &str,
        chunk_index: usize,
        chunk_text: &str,
        prompt: &str,
    ) -> Result<Option<String>> {
        let path = self.path_for(model, mode, chunk_index, chunk_text, prompt);
        if !path.exists() {
            return Ok(None);
        }

        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read cache file: {}", path.display()))?;
        let entry: CacheEntry = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse cache file: {}", path.display()))?;

        Ok(Some(entry.response))
    }

    pub fn put(
        &self,
        model: &str,
        mode: &str,
        chunk_index: usize,
        chunk_text: &str,
        prompt: &str,
        response: &str,
    ) -> Result<()> {
        let path = self.path_for(model, mode, chunk_index, chunk_text, prompt);
        let entry = CacheEntry {
            model: model.to_string(),
            mode: mode.to_string(),
            chunk_index,
            chunk_hash: hash_text(chunk_text),
            prompt_hash: hash_text(prompt),
            response: response.to_string(),
        };
        let json = serde_json::to_string_pretty(&entry)?;
        fs::write(&path, json)
            .with_context(|| format!("failed to write cache file: {}", path.display()))?;
        Ok(())
    }

    fn path_for(
        &self,
        model: &str,
        mode: &str,
        chunk_index: usize,
        chunk_text: &str,
        prompt: &str,
    ) -> PathBuf {
        let key = hash_text(&format!(
            "{}:{}:{}:{}:{}",
            model,
            mode,
            chunk_index,
            hash_text(chunk_text),
            hash_text(prompt)
        ));
        self.dir.join(format!("chunk-{:03}-{}.json", chunk_index, key))
    }
}

fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}
