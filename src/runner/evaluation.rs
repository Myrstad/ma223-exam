use std::collections::HashMap;
use std::fs;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use crate::author_classifier::AuthorClassifier;
use crate::config::ModelConfig;
use crate::iam::{IAMBatcher, IAMDataset};
use crate::types::MyBackend;

pub fn read_last_metric(path: &str) -> f32 {
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

pub fn parse_metrics(exp_id: &str, total_epochs: u32) -> (f32, f32, f32, f32, u32) {
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


pub fn evaluate_test_set(
    model_dir: &str,
    dataset_test: &IAMDataset,
    _max_epochs: u32,
    model_config: &ModelConfig,
    device: &burn::backend::wgpu::WgpuDevice,
    batch_size: u32,
    _prefix: &str,
    _exp_id: &str,
) -> (f32, f32, u32, String, String) {
    let num_classes = model_config.num_classes;

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
    let mut model = AuthorClassifier::<MyBackend>::new(device, model_config);

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
