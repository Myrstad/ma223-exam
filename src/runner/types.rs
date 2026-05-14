
pub struct TrainingHistory {
    pub train_loss: Vec<f32>,
    pub train_acc: Vec<f32>,
    pub valid_loss: Vec<f32>,
    pub valid_acc: Vec<f32>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            train_acc: Vec::new(),
            valid_loss: Vec::new(),
            valid_acc: Vec::new(),
        }
    }
}

pub struct ExperimentResults {
    pub experiment_id: String,
    pub total_epochs: u32,
    pub early_stopped: bool,
    pub best_epoch: u32,
    pub best_valid_loss: f32,
    pub best_valid_acc: f32,
    pub final_train_loss: f32,
    pub final_train_acc: f32,
    pub test_acc: f32,
    pub test_loss: f32,
    pub test_num_samples: u32,
    pub test_num_correct: u32,
    pub runtime_seconds: f64,
}