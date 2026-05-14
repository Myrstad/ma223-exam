use std::fs;
use serde::{Deserialize, Serialize};
use crate::config::app::ConfigError;

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

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentParams {
    pub id: String,
    pub max_epochs: MaxEpochValue,
    pub dropout: f32,
    pub learning_rate: f32,
    pub activation: String,
}


#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hyperparameters {
    pub max_epochs: Vec<MaxEpochValue>,
    pub dropout: Vec<f32>,
    pub learning_rate: Vec<f32>,
    pub activation: Vec<String>,
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
