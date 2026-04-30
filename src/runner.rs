use std::fs;
use std::time::Instant;

use burn::data::dataloader::DataLoaderBuilder;
use burn::train::LearnerBuilder;

use crate::config::ExperimentParams;
use crate::config::ModelConfig;
use crate::{AuthorClassifier, IAMDataset, IAMBatcher, MyAutodiffBackend, MyBackend};

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
    pub runtime_seconds: f64,
}

fn read_last_metric(path: &str) -> f32 {
    if let Ok(content) = fs::read_to_string(path) {
        if let Some(line) = content.lines().last() {
            if let Some(val) = line.split(',').next() {
                let v: f32 = val.parse().unwrap_or(0.0);
                // Values stored as percentages (e.g., 96.875 means 96.875%)
                return if v > 1.0 { v / 100.0 } else { v };
            }
        }
    }
    0.0
}

fn parse_metrics(exp_id: &str, total_epochs: u32) -> (f32, f32, f32, f32, u32) {
    let base = format!("results/experiments/{}/model", exp_id);
    
    let mut final_train_loss = 0.0f32;
    let mut final_train_acc = 0.0f32;
    let mut best_valid_loss = f32::MAX;
    let mut best_valid_acc = 0.0f32;
    let mut best_epoch = total_epochs;
    
    for epoch in 1..=total_epochs {
        let tlp = format!("{}/train/epoch-{}/Loss.log", base, epoch);
        let tap = format!("{}/train/epoch-{}/Accuracy.log", base, epoch);
        let vlp = format!("{}/valid/epoch-{}/Loss.log", base, epoch);
        let vap = format!("{}/valid/epoch-{}/Accuracy.log", base, epoch);
        
        if epoch == total_epochs {
            final_train_loss = read_last_metric(&tlp);
            final_train_acc = read_last_metric(&tap);
        }
        
        let vloss = read_last_metric(&vlp);
        let vacc = read_last_metric(&vap);
        
        if vloss < best_valid_loss {
            best_valid_loss = vloss;
            best_valid_acc = vacc;
            best_epoch = epoch;
        }
    }
    
    if best_valid_loss == f32::MAX {
        best_valid_loss = 0.0;
    }
    
    (final_train_loss, final_train_acc, best_valid_loss, best_valid_acc, best_epoch)
}

pub fn run_experiment(
    experiment: &ExperimentParams,
    dataset_train: IAMDataset,
    dataset_valid: IAMDataset,
    cnn_config: &crate::config::CnnConfig,
    batch_size: u32,
    _patience: u32,
    device: &burn::backend::wgpu::WgpuDevice,
    prefix: &str,
) -> (TrainingHistory, ExperimentResults) {
    let start = Instant::now();
    
    let model_config = ModelConfig::from_cnn_and_hyper(
        cnn_config,
        experiment.dropout,
        experiment.activation.clone(),
    );

    println!("  Running {} with lr={}, dropout={}, act={}", 
        experiment.id, experiment.learning_rate, experiment.dropout, experiment.activation);

    let max_epochs = experiment.max_epochs.to_u32().unwrap_or(50);
    
    let batcher_train = IAMBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = IAMBatcher::<MyBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size as usize).shuffle(42).num_workers(4).build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size as usize).num_workers(4).build(dataset_valid);

    let learner = LearnerBuilder::new(format!("results/experiments/{}/{}/model", prefix, experiment.id))
        .metric_train_numeric(burn::train::metric::AccuracyMetric::new())
        .metric_valid_numeric(burn::train::metric::AccuracyMetric::new())
        .metric_train_numeric(burn::train::metric::LossMetric::new())
        .metric_valid_numeric(burn::train::metric::LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(max_epochs as usize)
        .build(
            AuthorClassifier::<MyAutodiffBackend>::new(device, &model_config),
            burn::optim::AdamConfig::new().init(),
            experiment.learning_rate as f64,
        );

    let _trained_model = learner.fit(dataloader_train, dataloader_valid);
    
    let runtime = start.elapsed().as_secs_f64();
    
    let (final_train_loss, final_train_acc, best_valid_loss, best_valid_acc, best_epoch) = 
        parse_metrics(&format!("{}/{}", prefix, experiment.id), max_epochs);
    
    let history = TrainingHistory::new();
    let results = ExperimentResults {
        experiment_id: experiment.id.clone(),
        total_epochs: max_epochs,
        early_stopped: false,
        best_epoch,
        best_valid_loss,
        best_valid_acc,
        final_train_loss,
        final_train_acc,
        runtime_seconds: runtime,
    };

    println!("  {} completed in {:.1}s: train_acc={:.1}%, valid_loss={:.4}, valid_acc={:.1}%", 
        experiment.id, runtime, final_train_acc * 100.0, best_valid_loss, best_valid_acc * 100.0);

    (history, results)
}

pub fn save_results(
    exp_id: &str,
    _history: &TrainingHistory,
    results: &ExperimentResults,
    experiment: &ExperimentParams,
    prefix: &str,
) -> std::io::Result<()> {
    let base_dir = format!("results/experiments/{}/{}", prefix, exp_id);
    fs::create_dir_all(&base_dir)?;

    let config_json = serde_json::to_string_pretty(experiment)?;
    fs::write(format!("{}/config.json", base_dir), config_json)?;

    let summary = serde_json::json!({
        "experiment_id": exp_id,
        "hyperparameters": {
            "max_epochs": experiment.max_epochs,
            "dropout": experiment.dropout,
            "learning_rate": experiment.learning_rate,
            "activation": experiment.activation,
        },
        "training": {
            "total_epochs": results.total_epochs,
            "early_stopped": results.early_stopped,
            "best_epoch": results.best_epoch,
            "runtime_seconds": results.runtime_seconds,
        },
        "metrics": {
            "train": {
                "final_loss": results.final_train_loss,
                "final_accuracy": results.final_train_acc * 100.0,
            },
            "valid": {
                "best_loss": results.best_valid_loss,
                "best_accuracy": results.best_valid_acc * 100.0,
                "epoch": results.best_epoch,
            },
        },
    });
    
    fs::write(
        format!("{}/summary.json", base_dir),
        serde_json::to_string_pretty(&summary)?,
    )?;

    // Write history CSVs
    let model_base = format!("{}/model", base_dir);
    let mut train_csv = String::from("epoch,loss,accuracy\n");
    let mut valid_csv = String::from("epoch,loss,accuracy\n");
    
    for epoch in 1..=results.total_epochs {
        let tlp = format!("{}/train/epoch-{}/Loss.log", model_base, epoch);
        let tap = format!("{}/train/epoch-{}/Accuracy.log", model_base, epoch);
        let vlp = format!("{}/valid/epoch-{}/Loss.log", model_base, epoch);
        let vap = format!("{}/valid/epoch-{}/Accuracy.log", model_base, epoch);
        
        let tl = read_last_metric(&tlp);
        let ta = read_last_metric(&tap);
        let vl = read_last_metric(&vlp);
        let va = read_last_metric(&vap);
        
        train_csv.push_str(&format!("{},{},{}\n", epoch, tl, ta));
        valid_csv.push_str(&format!("{},{},{}\n", epoch, vl, va));
    }
    
    fs::write(format!("{}/history_train.csv", base_dir), train_csv)?;
    fs::write(format!("{}/history_valid.csv", base_dir), valid_csv)?;

    Ok(())
}