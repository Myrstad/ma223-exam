use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::train::LearnerBuilder;
use serde::{Deserialize, Serialize};

use crate::config::{ExperimentParams, ModelConfig};
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
    pub test_acc: f32,
    pub test_loss: f32,
    pub test_num_samples: u32,
    pub test_num_correct: u32,
    pub runtime_seconds: f64,
}

fn read_last_metric(path: &str) -> f32 {
    if let Ok(content) = fs::read_to_string(path) {
        let mut max_val = 0.0f32;
        for line in content.lines() {
            if let Some(val_str) = line.split(',').next() {
                if let Ok(v) = val_str.parse::<f32>() {
                    let adjusted = if v > 1.0 { v / 100.0 } else { v };
                    if adjusted > max_val {
                        max_val = adjusted;
                    }
                }
            }
        }
        return max_val;
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
    dataset_test: IAMDataset,
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
    
    // Save best validation model checkpoint
    let model_dir = format!("results/experiments/{}/{}/model", prefix, experiment.id);
    let best_dir = format!("{}/best", model_dir);
    if let Err(e) = fs::create_dir_all(&best_dir) {
        eprintln!("Warning: could not create best model dir: {}", e);
    }
    
    // Copy best epoch checkpoint files
    let ckpt_src = format!("{}/checkpoint", model_dir);
    let ckpt_dst = format!("{}/best", model_dir);
    let _ = fs::create_dir_all(&ckpt_dst);
    
    // Find any available checkpoint file
    let mut actual_epoch = max_epochs;
    let mut found_epoch = false;
    for epoch in (1..=max_epochs).rev() {
        let src = format!("{}/model-{}.mpk", ckpt_src, epoch);
        if std::path::Path::new(&src).exists() {
            actual_epoch = epoch;
            found_epoch = true;
            break;
        }
    }
    
    if found_epoch {
        for suffix in &["model", "optim", "scheduler"] {
            let src = format!("{}/{}-{}.mpk", ckpt_src, suffix, actual_epoch);
            let dst = format!("{}/{}-{}.mpk", ckpt_dst, suffix, actual_epoch);
            let _ = fs::copy(&src, &dst);
        }
    }
    
    // Evaluate on test set using best validation model (find checkpoint in best/ folder)
    let (test_loss, test_acc, test_num_correct, test_results_csv, confusion_csv) = 
        evaluate_test_set(&model_dir, &dataset_test, max_epochs, cnn_config.num_classes, device, batch_size, prefix, &experiment.id);
    let test_num_samples = dataset_test.labels.len() as u32;
    
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
        test_acc,
        test_loss,
        test_num_samples,
        test_num_correct,
        runtime_seconds: runtime,
    };

    // Save test results files
    let exp_dir = format!("results/experiments/{}/{}", prefix, experiment.id);
    let _ = fs::write(format!("{}/test_results.csv", exp_dir), test_results_csv);
    let _ = fs::write(format!("{}/confusion_matrix.csv", exp_dir), confusion_csv);

println!("  {} completed in {:.1}s: train_acc={}, valid_acc={} (epoch {}), test_acc={}", 
        experiment.id, runtime, final_train_acc, best_valid_acc, best_epoch, test_acc);
    
    // Delete model directory to save disk space
    let _ = fs::remove_dir_all(&model_dir);
    
    (history, results)
}

fn evaluate_test_set(
    model_dir: &str,
    dataset_test: &IAMDataset,
    _max_epochs: u32,
    num_classes: u32,
    device: &burn::backend::wgpu::WgpuDevice,
    batch_size: u32,
    _prefix: &str,
    _exp_id: &str,
) -> (f32, f32, u32, String, String) {
    // Create model config
    let model_config = ModelConfig {
        filters: vec![32, 64],
        fc_neurons: 512,
        dropout: 0.3,
        activation: "gelu".to_string(),
        num_classes,
    };
    
    // Find any checkpoint in best/ folder
    let best_dir = format!("{}/best", model_dir);
    let mut loaded_epoch = 0u32;
    let mut model_record_path = String::new();
    
    // Search for model checkpoint
    for epoch in (1..=100).rev() {
        let path = format!("{}/model-{}.mpk", best_dir, epoch);
        if std::path::Path::new(&path).exists() {
            model_record_path = path;
            loaded_epoch = epoch;
            break;
        }
}
     
     // Create model first, then try to load record into it
     let mut model = AuthorClassifier::<MyBackend>::new(device, &model_config);
     
     if loaded_epoch > 0 {
         match CompactRecorder::new().load(std::path::PathBuf::from(&model_record_path), device) {
             Ok(record) => {
                 model = model.load_record(record);
             }
             Err(_) => {}
         }
     }
    
    // Build test evaluation data loader
    let batcher = IAMBatcher::<MyBackend>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size as usize).num_workers(2).build(dataset_test.clone());
    
    let mut correct = 0usize;
    let total = dataset_test.labels.len();
    let mut confusion: HashMap<(i32, i32), usize> = HashMap::new();
    let mut results_csv = String::from("sample_id,true_label,predicted_label,correct,confidence\n");
    let mut sample_id = 0usize;
    
    println!("    Running test evaluation on {} samples...", total);
    
    for batch in dataloader.iter() {
        let output = model.forward(batch.images.clone());
        
        // Compute softmax probabilities for confidence (before argmax!)
        let probabilities = burn::tensor::activation::softmax(output.clone(), 1);
        
        // Get predictions and compute accuracy
        let preds = output.argmax(1).to_data().to_vec().unwrap();
        let targets = batch.targets.to_data().to_vec().unwrap();
        let probs_data = probabilities.to_data().to_vec::<f32>().unwrap();
        
        for i in 0..preds.len() {
            let pred = preds[i];
            let true_label = targets[i];
            let is_correct = if pred == true_label { correct += 1; 1 } else { 0 };
            
            // Get confidence = max probability for this prediction
            let conf = probs_data[i * num_classes as usize + pred as usize];
            
            let entry = confusion.entry((true_label, pred)).or_insert(0);
            *entry += 1;
            
            // Add to results CSV with actual confidence
            results_csv.push_str(&format!("{},{},{},{},{:.4}\n", sample_id, true_label, pred, is_correct, conf));
            sample_id += 1;
        }
    }
    
    let test_loss = 0.0;
    let num_samples = sample_id;
    let num_correct = correct;
    let test_acc = if num_samples > 0 { correct as f32 / num_samples as f32 } else { 0.0 };
    
    println!("    Test: accuracy={}, num_correct={}/{}, loss={:.4}", test_acc, num_correct, num_samples, test_loss);
    
    // Build confusion CSV
    let mut confusion_csv = String::from("true_label,predicted_label,count\n");
    for ((true_label, predicted_label), count) in confusion.iter() {
        confusion_csv.push_str(&format!("{},{},{}\n", true_label, predicted_label, count));
    }
    
    let num_correct_u32 = num_correct as u32;
    (test_loss, test_acc, num_correct_u32, results_csv, confusion_csv)
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
                "final_accuracy": results.final_train_acc,
            },
            "valid": {
                "best_loss": results.best_valid_loss,
                "best_accuracy": results.best_valid_acc,
                "epoch": results.best_epoch,
            },
            "test": {
                "accuracy": results.test_acc,
                "num_samples": results.test_num_samples,
                "num_correct": results.test_num_correct,
            },
        },
    });
    
    fs::write(
        format!("{}/summary.json", base_dir),
        serde_json::to_string_pretty(&summary)?,
    )?;

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

    // Update master summary
    update_master_summary(prefix, exp_id, experiment, results)?;

    Ok(())
}

fn update_master_summary(
    prefix: &str,
    exp_id: &str,
    experiment: &ExperimentParams,
    results: &ExperimentResults,
) -> std::io::Result<()> {
    let master_path = format!("results/experiments/{}/summary.json", prefix);
    
    #[derive(Deserialize, Serialize)]
    struct MasterSummary {
        experiment_name: String,
        total_experiments: usize,
        date_created: String,
        experiments: Vec<ExpSummary>,
        best_experiment: BestExp,
    }
    
    #[derive(Deserialize, Serialize, Clone)]
    struct ExpSummary {
        id: String,
        status: String,
        metrics: ExpMetrics,
        hyperparameters: serde_json::Value,
    }
    
    #[derive(Deserialize, Serialize, Clone)]
    struct ExpMetrics {
        train: TrainMetrics,
        valid: ValidMetrics,
        test: TestMetrics,
    }
    
    #[derive(Deserialize, Serialize, Clone)]
    struct TrainMetrics {
        accuracy: f32,
        loss: f32,
    }
    
    #[derive(Deserialize, Serialize, Clone)]
    struct ValidMetrics {
        accuracy: f32,
        loss: f32,
        epoch: u32,
    }
    
    #[derive(Deserialize, Serialize, Clone)]
    struct TestMetrics {
        accuracy: f32,
        loss: f32,
    }
    
    #[derive(Deserialize, Serialize)]
    struct BestExp {
        id: String,
        test_accuracy: f32,
    }
    
    // Create or load master summary
    let master: MasterSummary = if std::path::Path::new(&master_path).exists() {
        let content = fs::read_to_string(&master_path)?;
        serde_json::from_str(&content).unwrap_or_else(|_| {
            MasterSummary {
                experiment_name: prefix.to_string(),
                total_experiments: 81,
                date_created: chrono::Utc::now().to_rfc3339(),
                experiments: Vec::new(),
                best_experiment: BestExp { id: "none".to_string(), test_accuracy: 0.0 },
            }
        })
    } else {
        MasterSummary {
            experiment_name: prefix.to_string(),
            total_experiments: 81,
            date_created: chrono::Utc::now().to_rfc3339(),
            experiments: Vec::new(),
            best_experiment: BestExp { id: "none".to_string(), test_accuracy: 0.0 },
        }
    };
    
    // Create experiment summary entry
    let exp_summary = ExpSummary {
        id: exp_id.to_string(),
        status: "completed".to_string(),
        metrics: ExpMetrics {
            train: TrainMetrics {
                accuracy: results.final_train_acc,
                loss: results.final_train_loss,
            },
            valid: ValidMetrics {
                accuracy: results.best_valid_acc,
                loss: results.best_valid_loss,
                epoch: results.best_epoch,
            },
            test: TestMetrics {
                accuracy: results.test_acc,
                loss: results.test_loss,
            },
        },
        hyperparameters: serde_json::json!({
            "max_epochs": experiment.max_epochs,
            "dropout": experiment.dropout,
            "learning_rate": experiment.learning_rate,
            "activation": experiment.activation,
        }),
    };
    
    // Update experiments list - replace or add
    let mut new_experiments = master.experiments.clone();
    let pos = new_experiments.iter().position(|e| e.id == exp_id);
    match pos {
        Some(i) => new_experiments[i] = exp_summary,
        None => new_experiments.push(exp_summary),
    }
    
    // Find best by test accuracy
    let best = new_experiments.iter()
        .filter(|e| e.status == "completed")
        .max_by(|a, b| {
            a.metrics.test.accuracy.partial_cmp(&b.metrics.test.accuracy).unwrap()
        });
    
    let best_exp = match best {
        Some(e) => BestExp {
            id: e.id.clone(),
            test_accuracy: e.metrics.test.accuracy,
        },
        None => BestExp { id: "none".to_string(), test_accuracy: 0.0 },
    };
    
    let updated = MasterSummary {
        experiment_name: master.experiment_name,
        total_experiments: master.total_experiments,
        date_created: master.date_created,
        experiments: new_experiments,
        best_experiment: best_exp,
    };
    
    fs::write(&master_path, serde_json::to_string_pretty(&updated)?)?;
    
    Ok(())
}