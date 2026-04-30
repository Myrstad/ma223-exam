mod data;
mod config;
mod runner;

use crate::config::{Config, Hyperparameters, ModelConfig};
use crate::data::init_dataset;
use crate::runner::run_experiment;
// We don't need Autodiff for the seed call itself,
// but we'll use it for your model later.
use burn::backend::Autodiff;
use burn::backend::{wgpu::WgpuDevice, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::nn;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::MaxPool2d;
use burn::nn::{BatchNormConfig, DropoutConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::{Backend, TensorData};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Shape, Tensor};
use burn::train::{
    ClassificationOutput,
    TrainOutput, TrainStep, ValidStep,
};
use image::io::Reader as ImageReader;

// In Burn 0.16, Wgpu defaults to <f32, i32, u8>.
// You rarely need to specify the GraphicsApi manually anymore.
type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

// --- 1. THE MODEL ---
#[derive(Module, Debug)]
pub struct AuthorClassifier<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: nn::Dropout,
    batch_norm: nn::BatchNorm<B, 2>,
    activation: String,
}

impl<B: Backend> AuthorClassifier<B> {
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let initializer = nn::Initializer::KaimingNormal {
            gain: 1.414,
            fan_out_only: false,
        };

        let filter1 = config.filters[0] as usize;
        let filter2 = config.filters.get(1).copied().unwrap_or(config.filters[0]) as usize;

        let conv1 = Conv2dConfig::new([1, filter1], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_initializer(initializer.clone())
            .init(device);
        let conv2 = Conv2dConfig::new([filter1, filter2], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_initializer(initializer.clone())
            .init(device);

        let pool_config = nn::pool::MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        let fc1 = LinearConfig::new(filter2 * 16 * 16, config.fc_neurons as usize)
            .with_initializer(initializer)
            .init(device);

        let fc2 = LinearConfig::new(config.fc_neurons as usize, config.num_classes as usize)
            .init(device);

        Self {
            conv1,
            conv2,
            pool: pool_config,
            fc1,
            fc2,
            dropout: DropoutConfig::new(config.dropout as f64).init(),
            batch_norm: BatchNormConfig::new(filter1).init(device),
            activation: config.activation.clone(),
        }
    }

    fn apply_activation<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self.activation.as_str() {
            "relu" => burn::tensor::activation::relu(x),
            "tanh" => burn::tensor::activation::tanh(x),
            "sigmoid" => burn::tensor::activation::sigmoid(x),
            "gelu" => burn::tensor::activation::gelu(x),
            "silu" => burn::tensor::activation::silu(x),
            _ => burn::tensor::activation::gelu(x),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.batch_norm.forward(x);
        let x = self.apply_activation(x);
        let x = self.pool.forward(x);

        let x = self.conv2.forward(x);
        let x = self.apply_activation(x);
        let x = self.pool.forward(x);

        let x = x.flatten(1, 3);
        let x = self.fc1.forward(x);
        let x = self.apply_activation(x);
        let x = self.dropout.forward(x);

        self.fc2.forward(x)
    }
}

// --- 2. DATASET & BATCHER ---
#[derive(Clone, Debug)]
pub struct IAMItem {
    pub pixels: Vec<f32>,
    pub label: i32,
}

#[derive(Clone, Debug)]
pub struct IAMDataset {
    pub image_paths: Vec<std::path::PathBuf>,
    pub labels: Vec<i32>,
}

impl IAMDataset {
    pub fn new(metadata: data::DatasetMetadata) -> Self {
        Self { image_paths: metadata.image_paths, labels: metadata.labels }
    }
}

impl Dataset<IAMItem> for IAMDataset {
    fn get(&self, index: usize) -> Option<IAMItem> {
        let path = &self.image_paths[index];
        let label = self.labels[index];
        let img = ImageReader::open(path).ok()?.decode().ok()?;
        let img = img.grayscale().resize_exact(64, 64, image::imageops::FilterType::Lanczos3).to_luma8();
        let pixels: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
        if pixels.len() != 4096 {
            return None;
        }
        Some(IAMItem { pixels, label })
    }
    fn len(&self) -> usize { self.image_paths.len() }
}

#[derive(Clone)] // Required for DataLoader
pub struct IAMBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> IAMBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct IAMBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<IAMItem, IAMBatch<B>> for IAMBatcher<B> {
    fn batch(&self, items: Vec<IAMItem>) -> IAMBatch<B> {
        let images = items.iter()
            .map(|item| TensorData::new(item.pixels.clone(), Shape::new([1, 64, 64])))
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device))
            .collect();

        let targets = items.iter()
            .map(|item| TensorData::new(vec![item.label], Shape::new([1])))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, &self.device))
            .collect();

        IAMBatch {
            images: Tensor::stack(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

impl<B: AutodiffBackend> TrainStep<IAMBatch<B>, ClassificationOutput<B>> for AuthorClassifier<B> {
    fn step(&self, batch: IAMBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(batch.images);
        let loss = nn::loss::CrossEntropyLossConfig::new()
            .init(&item.device())
            .forward(item.clone(), batch.targets.clone());

        TrainOutput::new(self, loss.backward(), ClassificationOutput::new(loss, item, batch.targets))
    }
}

impl<B: Backend> ValidStep<IAMBatch<B>, ClassificationOutput<B>> for AuthorClassifier<B> {
    fn step(&self, batch: IAMBatch<B>) -> ClassificationOutput<B> {
        let item = self.forward(batch.images);
        let loss = nn::loss::CrossEntropyLossConfig::new()
            .init(&item.device())
            .forward(item.clone(), batch.targets.clone());

        ClassificationOutput::new(loss, item, batch.targets)
    }
}

fn main() {
    init_dataset().expect("Failed to initialize dataset");
    let device = WgpuDevice::default();

    // Load config early - needed for data split
    let app_config = Config::load("config/config.json")
        .expect("Failed to load config");

    let forms_path = "data/iam_top50/forms_for_parsing.txt";
    let images_path = "data/iam_top50/data_subset";

    let metadata = data::load_metadata(forms_path, images_path)
        .expect("Failed to parse dataset metadata");

    // Stratified split: from config
    let mut author_groups: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();

    for idx in 0..metadata.image_paths.len() {
        let label = metadata.labels[idx];
        author_groups.entry(label).or_insert(Vec::new()).push(idx);
    }

    let mut train_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut test_indices = Vec::new();

    for (_, indices) in author_groups.iter_mut() {
        let total = indices.len();
        let train_split = (total as f32 * app_config.data.train_split) as usize;
        let valid_split = (total as f32 * (app_config.data.train_split + app_config.data.valid_split)) as usize;
        
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

    fn create_dataset(metadata: &data::DatasetMetadata, indices: &[usize]) -> IAMDataset {
        let image_paths: Vec<_> = indices.iter().map(|&i| metadata.image_paths[i].clone()).collect();
        let labels: Vec<_> = indices.iter().map(|&i| metadata.labels[i]).collect();
        IAMDataset::new(data::DatasetMetadata {
            image_paths,
            labels,
            authors: Vec::new(),
            num_classes: metadata.num_classes,
        })
    }

    let dataset_train = create_dataset(&metadata, &train_indices);
    let dataset_valid = create_dataset(&metadata, &valid_indices);
    let dataset_test = create_dataset(&metadata, &test_indices);

    println!("Train: {} samples, Valid: {} samples, Test: {} samples", 
        train_indices.len(), valid_indices.len(), test_indices.len());

    // Verify all authors appear in both sets
    let train_authors: std::collections::HashSet<_> = train_indices.iter().map(|&i| metadata.labels[i]).collect();
    let valid_authors: std::collections::HashSet<_> = valid_indices.iter().map(|&i| metadata.labels[i]).collect();
    println!("Train authors: {}, Valid authors: {}", train_authors.len(), valid_authors.len());

    let missing_in_train: Vec<_> = valid_authors.difference(&train_authors).collect();
    let missing_in_valid: Vec<_> = train_authors.difference(&valid_authors).collect();
    if missing_in_train.is_empty() && missing_in_valid.is_empty() {
        println!("✅ All authors appear in both train and validation sets (closed-set)");
    } else {
        println!("⚠️ Authors missing - train: {:?}, valid: {:?}", missing_in_train, missing_in_valid);
    }

    // Load configs and run experiments
    let hyper = Hyperparameters::load("config/hyperparameters.json")
        .expect("Failed to load hyperparameters");
    let app_config = Config::load("config/config.json")
        .expect("Failed to load config");

    let experiments = hyper.iter_experiments();
    let total = experiments.len();
    let prefix = &app_config.experiment_name;
    
    println!("Loaded {} experiments to run", total);
    println!("Experiment prefix: {}", prefix);
    println!("Config: batch_size={}, num_classes={}", app_config.batch_size, app_config.cnn.num_classes);
    println!();

    // Count completed experiments
    let mut completed = 0;
    let mut pending = Vec::new();
    
    for exp in &experiments {
        let exp_dir = format!("results/experiments/{}/{}", prefix, exp.id);
        if std::path::Path::new(&exp_dir).exists() {
            completed += 1;
        } else {
            pending.push(exp.clone());
        }
    }
    
    if completed == total {
        println!("All {} experiments already completed!", total);
        return;
    }
    
    println!("{} already completed, {} to run\n", completed, pending.len());
    
    // Run pending experiments
    for (i, exp) in pending.iter().enumerate() {
        println!("=== [{}/{}] Running experiment {} ===", i + 1, pending.len(), exp.id);
        
        let (history, results) = run_experiment(
            exp,
            dataset_train.clone(),
            dataset_valid.clone(),
            dataset_test.clone(),
            &app_config.cnn,
            app_config.batch_size,
            app_config.early_stopping.patience,
            &device,
            prefix,
        );
        
        println!("Final: train_acc={}, valid_loss={:.4}, valid_acc={}, test_acc={}", 
            results.final_train_acc, results.best_valid_loss, results.best_valid_acc, 
            results.test_acc);
        
        // Save results
        runner::save_results(&results.experiment_id, &history, &results, exp, prefix).expect("Failed to save results");
        println!("Results saved to results/experiments/{}/{}\n", prefix, exp.id);
    }
    
    println!("=== All {} experiments completed! ===", total);
}