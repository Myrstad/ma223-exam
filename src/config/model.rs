use serde::{Deserialize, Serialize};

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
    pub kernel_size: u32,
    pub pool_size: u32,
    pub fc_neurons: u32,
    pub dropout: f32,
    pub activation: String,
    pub num_classes: u32,
}

impl ModelConfig {
    pub fn from_cnn_and_hyper(cnn: &CnnConfig, dropout: f32, activation: String) -> Self {
        Self {
            filters: cnn.filters.clone(),
            kernel_size: cnn.kernel_size,
            pool_size: cnn.pool_size,
            fc_neurons: cnn.fc_neurons,
            dropout,
            activation,
            num_classes: cnn.num_classes,
        }
    }
}