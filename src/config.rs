use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum MaxEpochValue {
    Integer(u32),
    String(String),
}

impl MaxEpochValue {
    pub fn is_no_improvement(&self) -> bool {
        matches!(self, MaxEpochValue::String(s) if s == "no_improvement")
    }
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            MaxEpochValue::Integer(n) => Some(*n),
            MaxEpochValue::String(_) => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hyperparameters {
    pub max_epochs: Vec<MaxEpochValue>,
    pub dropout: Vec<f32>,
    pub learning_rate: Vec<f32>,
    pub activation: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CnnConfig {
    pub conv_layers: u32,
    pub filters: Vec<u32>,
    pub kernel_size: u32,
    pub pool_size: u32,
    pub fc_neurons: u32,
    pub num_classes: u32,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub filters: Vec<u32>,
    pub fc_neurons: u32,
    pub dropout: f32,
    pub activation: String,
    pub num_classes: u32,
}

impl ModelConfig {
    pub fn from_cnn_and_hyper(cnn: &CnnConfig, dropout: f32, activation: String) -> Self {
        Self {
            filters: cnn.filters.clone(),
            fc_neurons: cnn.fc_neurons,
            dropout,
            activation,
            num_classes: cnn.num_classes,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EarlyStoppingConfig {
    pub patience: u32,
    pub min_delta: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub train_split: f32,
    pub valid_split: f32,
    pub forms_file: String,
    pub images_dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    pub base_dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub experiment_name: String,
    pub batch_size: u32,
    pub cnn: CnnConfig,
    pub early_stopping: EarlyStoppingConfig,
    pub data: DataConfig,
    pub output: OutputConfig,
}

impl Hyperparameters {
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path)
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_json::from_str(&contents)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    pub fn total_experiments(&self) -> usize {
        self.max_epochs.len() 
            * self.dropout.len() 
            * self.learning_rate.len() 
            * self.activation.len()
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path)
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_json::from_str(&contents)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentParams {
    pub id: String,
    pub max_epochs: MaxEpochValue,
    pub dropout: f32,
    pub learning_rate: f32,
    pub activation: String,
}

impl Hyperparameters {
    pub fn iter_experiments(&self) -> Vec<ExperimentParams> {
        let mut experiments = Vec::new();
        let mut id = 1;
        
        for max_epoch in &self.max_epochs {
            for dropout in &self.dropout {
                for lr in &self.learning_rate {
                    for activation in &self.activation {
                        experiments.push(ExperimentParams {
                            id: format!("exp_{:03}", id),
                            max_epochs: max_epoch.clone(),
                            dropout: *dropout,
                            learning_rate: *lr,
                            activation: activation.clone(),
                        });
                        id += 1;
                    }
                }
            }
        }
        experiments
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

#[cfg(test)]
mod tests_integration {
    use super::*;

    #[test]
    fn test_load_hyperparameters() {
        let hyper = Hyperparameters::load("config/hyperparameters.json").unwrap();
        assert_eq!(hyper.total_experiments(), 81);
    }

    #[test]
    fn test_load_config() {
        let config = Config::load("config/config.json").unwrap();
        assert_eq!(config.experiment_name, "author_classification_grid_v1");
    }

    #[test]
    fn test_generate_all_experiments() {
        let hyper = Hyperparameters::load("config/hyperparameters.json").unwrap();
        let experiments = hyper.iter_experiments();
        assert_eq!(experiments.len(), 81);
        
        assert_eq!(experiments[0].max_epochs.to_u32(), Some(25));
        assert_eq!(experiments[0].dropout, 0.2);
        assert_eq!(experiments[0].learning_rate, 0.001);
        assert_eq!(experiments[0].activation, "gelu");
        
        assert_eq!(experiments.last().unwrap().id, "exp_081");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn temp_json(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    #[test]
    fn test_parse_hyperparameters() {
        let content = r#"{
            "max_epochs": [25, 50, "no_improvement"],
            "dropout": [0.2, 0.3],
            "learning_rate": [0.001, 0.01],
            "activation": ["gelu", "relu"]
        }"#;
        
        let file = temp_json(content);
        let hyper = Hyperparameters::load(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(hyper.max_epochs.len(), 3);
        assert_eq!(hyper.dropout.len(), 2);
        assert_eq!(hyper.learning_rate.len(), 2);
        assert_eq!(hyper.activation.len(), 2);
        assert_eq!(hyper.total_experiments(), 24);
    }

    #[test]
    fn test_max_epochs_value() {
        assert!(!MaxEpochValue::Integer(25).is_no_improvement());
        assert!(!MaxEpochValue::Integer(50).is_no_improvement());
        assert!(MaxEpochValue::String("no_improvement".to_string()).is_no_improvement());
        
        assert_eq!(MaxEpochValue::Integer(25).to_u32(), Some(25));
        assert_eq!(MaxEpochValue::String("no_improvement".to_string()).to_u32(), None);
    }

    #[test]
    fn test_experiment_generation() {
        let hyper = Hyperparameters {
            max_epochs: vec![MaxEpochValue::Integer(25)],
            dropout: vec![0.2, 0.3],
            learning_rate: vec![0.001],
            activation: vec!["gelu".to_string()],
        };
        
        let experiments = hyper.iter_experiments();
        assert_eq!(experiments.len(), 2);
        assert_eq!(experiments[0].id, "exp_001");
        assert_eq!(experiments[1].id, "exp_002");
    }

    #[test]
    fn test_parse_config() {
        let content = r#"{
            "experiment_name": "test_exp",
            "batch_size": 64,
            "cnn": {
                "conv_layers": 2,
                "filters": [32, 64],
                "kernel_size": 3,
                "pool_size": 2,
                "fc_neurons": 512,
                "num_classes": 50
            },
            "early_stopping": {
                "patience": 10,
                "min_delta": 0.001
            },
            "data": {
                "train_split": 0.8,
                "valid_split": 0.2,
                "forms_file": "data/forms.txt",
                "images_dir": "data/images"
            },
            "output": {
                "base_dir": "results"
            }
        }"#;
        
        let file = temp_json(content);
        let config = Config::load(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.experiment_name, "test_exp");
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.cnn.num_classes, 50);
        assert_eq!(config.early_stopping.patience, 10);
    }
}