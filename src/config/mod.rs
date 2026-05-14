pub mod tests;
pub mod hyper;
pub mod model;
pub mod app;

pub use hyper::{
  Hyperparameters,
  ExperimentParams
};
pub use model::{
    CnnConfig,
    ModelConfig,
};

pub use app::{
    Config,
};