use std::collections::{HashMap, HashSet};
use crate::config::Config;
use crate::data::{load_metadata, DatasetMetadata};
use crate::iam::IAMDataset;

/// Prepares train/validation/test datasets with stratified splitting
pub fn prepare_datasets(config: &Config) -> (IAMDataset, IAMDataset, IAMDataset) {
    let forms_path = &config.data.forms_file;
    let images_path = &config.data.images_dir;

    let metadata = load_metadata(forms_path, images_path)
        .expect("Failed to parse dataset metadata");

    // Stratified split: group by author
    let mut author_groups: HashMap<i32, Vec<usize>> = HashMap::new();
    for idx in 0..metadata.image_paths.len() {
        let label = metadata.labels[idx];
        author_groups.entry(label).or_insert(Vec::new()).push(idx);
    }

    let mut train_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Split each author's samples
    for (_, indices) in author_groups.iter_mut() {
        let total = indices.len();
        let train_split = (total as f32 * config.data.train_split) as usize;
        let valid_split = (total as f32 * (config.data.train_split + config.data.valid_split)) as usize;
        
        for (i, &idx) in indices.iter().enumerate() {
            if i < train_split {
                train_indices.push(idx);
            } else if i < valid_split {
                valid_indices.push(idx);
            } else {
                test_indices.push(idx);
            }
        }
    }

    let dataset_train = create_dataset(&metadata, &train_indices);
    let dataset_valid = create_dataset(&metadata, &valid_indices);
    let dataset_test = create_dataset(&metadata, &test_indices);

    println!("Train: {} samples, Valid: {} samples, Test: {} samples", 
        train_indices.len(), valid_indices.len(), test_indices.len());

    verify_splits(&metadata, &train_indices, &valid_indices);

    (dataset_train, dataset_valid, dataset_test)
}

fn create_dataset(metadata: &DatasetMetadata, indices: &[usize]) -> IAMDataset {
    let image_paths: Vec<_> = indices.iter().map(|&i| metadata.image_paths[i].clone()).collect();
    let labels: Vec<_> = indices.iter().map(|&i| metadata.labels[i]).collect();
    IAMDataset::new(DatasetMetadata {
        image_paths,
        labels,
        authors: Vec::new(),
        num_classes: metadata.num_classes,
    })
}

fn verify_splits(metadata: &DatasetMetadata, train_indices: &[usize], valid_indices: &[usize]) {
    let train_authors: HashSet<_> = train_indices.iter().map(|&i| metadata.labels[i]).collect();
    let valid_authors: HashSet<_> = valid_indices.iter().map(|&i| metadata.labels[i]).collect();
    println!("Train authors: {}, Valid authors: {}", train_authors.len(), valid_authors.len());

    let missing_in_train: Vec<_> = valid_authors.difference(&train_authors).collect();
    let missing_in_valid: Vec<_> = train_authors.difference(&valid_authors).collect();
    if missing_in_train.is_empty() && missing_in_valid.is_empty() {
        println!("All authors appear in both train and validation sets (closed-set)");
    } else {
        println!("Authors missing - train: {:?}, valid: {:?}", missing_in_train, missing_in_valid);
    }
}
