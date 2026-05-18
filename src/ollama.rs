use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    http: Client,
    num_ctx: u32,
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: GenerateOptions,
}

#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f32,
    num_ctx: u32,
    repeat_penalty: f32,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String, timeout_seconds: u64, num_ctx: u32) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .expect("failed to build HTTP client");

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            http,
            num_ctx,
        }
    }

    pub fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self
            .http
            .get(&url)
            .send()
            .with_context(|| format!("failed to call Ollama at {}", url))?
            .error_for_status()
            .context("Ollama returned an error status for /api/tags")?
            .json::<TagsResponse>()
            .context("failed to decode Ollama /api/tags response")?;

        Ok(response.models.into_iter().map(|m| m.name).collect())
    }

    pub fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.base_url);

        let body = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
            options: GenerateOptions {
                temperature: 0.1,
                num_ctx: self.num_ctx,
                repeat_penalty: 1.12,
            },
        };

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .with_context(|| format!("failed to call Ollama at {}", url))?
            .error_for_status()
            .context("Ollama returned an error status")?
            .json::<GenerateResponse>()
            .context("failed to decode Ollama JSON response")?;

        if !response.done {
            anyhow::bail!("Ollama response did not finish cleanly");
        }

        Ok(response.response.trim().to_string())
    }
}
