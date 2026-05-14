use std::fs;
use std::time::Instant;
use burn::data::dataloader::DataLoaderBuilder;
use burn::record::CompactRecorder;
use burn::train::LearnerBuilder;
use crate::config::{ExperimentParams, ModelConfig};
use crate::iam::{IAMBatcher, IAMDataset};
use crate::types::{MyAutodiffBackend, MyBackend};
use crate::author_classifier::AuthorClassifier;
use crate::runner::evaluation::{evaluate_test_set, read_last_metric};
use crate::runner::parse_metrics;
use crate::runner::types::{ExperimentResults, TrainingHistory};

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
        .with_file_checkpointer(CompactRecorder::new())
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
        evaluate_test_set(&model_dir, &dataset_test, max_epochs, &model_config, device, batch_size, prefix, &experiment.id);
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

    // Build History CSVs BEFORE we delete the model directory
    let mut train_csv = String::from("epoch,loss,accuracy\n");
    let mut valid_csv = String::from("epoch,loss,accuracy\n");

    for epoch in 1..=max_epochs {
        let tlp = format!("{}/train/epoch-{}/Loss.log", model_dir, epoch);
        let tap = format!("{}/train/epoch-{}/Accuracy.log", model_dir, epoch);
        let vlp = format!("{}/valid/epoch-{}/Loss.log", model_dir, epoch);
        let vap = format!("{}/valid/epoch-{}/Accuracy.log", model_dir, epoch);

        let tl = read_last_metric(&tlp);
        let ta = read_last_metric(&tap);
        let vl = read_last_metric(&vlp);
        let va = read_last_metric(&vap);

        train_csv.push_str(&format!("{},{},{}\n", epoch, tl, ta));
        valid_csv.push_str(&format!("{},{},{}\n", epoch, vl, va));
    }

    let _ = fs::write(format!("{}/history_train.csv", exp_dir), train_csv);
    let _ = fs::write(format!("{}/history_valid.csv", exp_dir), valid_csv);

    println!("  {} completed in {:.1}s: train_acc={}, valid_acc={} (epoch {}), test_acc={}",
             experiment.id, runtime, final_train_acc, best_valid_acc, best_epoch, test_acc);

    // We delete model to not run out of disk space D:
    let _ = fs::remove_dir_all(&model_dir);

    (history, results)
}