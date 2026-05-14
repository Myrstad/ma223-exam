use std::fs;
use serde::{Deserialize, Serialize};
use crate::config::model::CnnConfig;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EarlyStoppingConfig {
    pub patience: u32,
    pub min_delta: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub train_split: f32,
    pub valid_split: f32,
    pub test_split: f32,
    pub forms_file: String,
    pub images_dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    pub base_dir: String,
}

fn default_repeats() -> u32 { 1 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub experiment_name: String,
    #[serde(default = "default_repeats")]
    pub num_repeats: u32,
    pub batch_size: u32,
    pub cnn: CnnConfig,
    pub early_stopping: EarlyStoppingConfig,
    pub data: DataConfig,
    pub output: OutputConfig,
}


impl Config {
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path)
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_json::from_str(&contents)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }
}

#[derive(Debug)]
pub enum ConfigError {
    IoError(String),
    ParseError(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::IoError(s) => write!(f, "IO error: {}", s),
            ConfigError::ParseError(s) => write!(f, "Parse error: {}", s),
        }
    }
}

impl std::error::Error for ConfigError {}