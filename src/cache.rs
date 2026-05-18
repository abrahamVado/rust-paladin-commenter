use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// On-disk per-chunk response cache.
///
/// Stores computed responses keyed by a composite SHA-256 hash of
/// (model, mode, chunk_index, chunk_content_hash, prompt_hash).
/// This ensures that the response is deterministic based on all inputs.
/// An optional TTL causes stale entries to be treated as cache misses.
#[derive(Debug, Clone)]
pub struct ChunkCache {
    dir: PathBuf,
    ttl: Option<Duration>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    model: String,
    mode: String,
    chunk_index: usize,
    chunk_hash: String,
    prompt_hash: String,
    response: String,
    /// Seconds since UNIX epoch when the entry was written. Used for TTL checks.
    created_at_epoch: u64,
}

impl ChunkCache {
    /// Create (or open) a chunk cache rooted at `dir`.
    /// If `ttl` is `Some`, entries older than the duration are ignored.
    ///
    /// This function ensures the cache directory exists, failing if permissions
    /// or disk space prevent creation.
    pub fn new(dir: PathBuf, ttl: Option<Duration>) -> Result<Self> {
        fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create cache dir: {}", dir.display()))?;
        Ok(Self { dir, ttl })
    }

    /// Try to load a cached response.  Returns `None` on miss or expired entry.
    ///
    /// The cache key is derived from all input parameters (model, mode, chunk_text, prompt).
    /// If the file exists but the content is stale (due to TTL), it is treated as a miss.
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

        // Check TTL expiry
        if let Some(ttl) = self.ttl {
            let now_epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            // If the time elapsed since creation exceeds the TTL, treat it as expired (cache miss).
            if now_epoch.saturating_sub(entry.created_at_epoch) > ttl.as_secs() {
                return Ok(None);
            }
        }

        Ok(Some(entry.response))
    }

    /// Persist a model response into the cache.
    ///
    /// The cache key is derived from all input parameters. If a key already exists,
    /// it will be overwritten, effectively updating the cache entry.
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
        let created_at_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = CacheEntry {
            model: model.to_string(),
            mode: mode.to_string(),
            chunk_index,
            chunk_hash: hash_text(chunk_text),
            prompt_hash: hash_text(prompt),
            response: response.to_string(),
            created_at_epoch,
        };
        let json = serde_json::to_string_pretty(&entry)?;
        // Write the entry to disk, overwriting any existing file at this path.
        fs::write(&path, json)
            .with_context(|| format!("failed to write cache file: {}", path.display()))?;
        Ok(())
    }

    /// Generates the unique file path for a given set of inputs.
    ///
    /// The path is derived from a composite hash of all inputs to ensure
    /// that the cache key is deterministic and unique to the input combination.
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
        // Truncate hash to first 16 hex chars for shorter filenames
        let short_key = &key[..16.min(key.len())];
        self.dir
            .join(format!("chunk-{:03}-{}.json", chunk_index, short_key))
    }
}

/// Compute a hex-encoded SHA-256 digest.
fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}
