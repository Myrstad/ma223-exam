mod data;

use burn::nn::{pool, BatchNormConfig, DropoutConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::backend::{Wgpu, wgpu::WgpuDevice};
// We don't need Autodiff for the seed call itself,
// but we'll use it for your model later.
use burn::backend::Autodiff;
use burn::prelude::{Backend, TensorData};
use crate::data::init_dataset;
use burn::data::dataset::Dataset;
use image::io::Reader as ImageReader;
use burn::tensor::{Tensor, Int, Data, Shape};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::nn;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::MaxPool2d;
use burn::optim::AdamConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};

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
}

impl<B: Backend> AuthorClassifier<B> {
    pub fn new(device: &B::Device) -> Self {
        let initializer = nn::Initializer::KaimingNormal {
            gain: 1.414, // Standard for ReLU/Gelu
            fan_out_only: false,
        };

        let conv1 = Conv2dConfig::new([1, 32], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_initializer(initializer.clone())
            .init(device);
        let conv2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_initializer(initializer.clone())
            .init(device);

        let pool_config = nn::pool::MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        let fc1 = LinearConfig::new(64 * 16 * 16, 512)
            .with_initializer(initializer)
            .init(device);

        let fc2 = LinearConfig::new(512, 50).init(device);

        Self {
            conv1,
            conv2,
            pool: pool_config,
            fc1,
            fc2,
            dropout: DropoutConfig::new(0.3).init(),
            batch_norm: BatchNormConfig::new(32).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = self.batch_norm.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.pool.forward(x);

        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.pool.forward(x);

        let x = x.flatten(1, 3);
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::gelu(x);
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

pub struct IAMDataset {
    image_paths: Vec<std::path::PathBuf>,
    labels: Vec<i32>,
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
        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&item.device())
            .forward(item.clone(), batch.targets.clone());

        TrainOutput::new(self, loss.backward(), ClassificationOutput::new(loss, item, batch.targets))
    }
}

impl<B: Backend> ValidStep<IAMBatch<B>, ClassificationOutput<B>> for AuthorClassifier<B> {
    fn step(&self, batch: IAMBatch<B>) -> ClassificationOutput<B> {
        let item = self.forward(batch.images);
        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&item.device())
            .forward(item.clone(), batch.targets.clone());

        ClassificationOutput::new(loss, item, batch.targets)
    }
}

fn main() {
    init_dataset().expect("Failed to initialize dataset");
    let device = WgpuDevice::default();

    let forms_path = "data/iam_top50/forms_for_parsing.txt";
    let images_path = "data/iam_top50/data_subset";

    let metadata = data::load_metadata(forms_path, images_path)
        .expect("Failed to parse dataset metadata");

    let num_images = metadata.image_paths.len();
    let split_index = (num_images as f32 * 0.8) as usize;
    let dataset_train = IAMDataset::new(data::DatasetMetadata {
        image_paths: metadata.image_paths[..split_index].to_vec(),
        labels: metadata.labels[..split_index].to_vec(),
        num_classes: 50,
    });
    let dataset_valid = IAMDataset::new(data::DatasetMetadata {
        image_paths: metadata.image_paths[split_index..].to_vec(),
        labels: metadata.labels[split_index..].to_vec(),
        num_classes: 50,
    });

    let batcher_train = IAMBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = IAMBatcher::<MyBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(64).shuffle(42).num_workers(8).build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(64).shuffle(42).num_workers(8).build(dataset_valid);

    let learner = LearnerBuilder::new("./tmp/iam-classification")
        .metric_train_numeric(burn::train::metric::AccuracyMetric::new())
        .metric_valid_numeric(burn::train::metric::AccuracyMetric::new())
        .metric_train_numeric(burn::train::metric::LossMetric::new())
        .metric_valid_numeric(burn::train::metric::LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(50)
        .build(
            AuthorClassifier::<MyAutodiffBackend>::new(&device),
            AdamConfig::new().init(),
            1e-4, // Learning Rate
        );

    let _model_trained = learner.fit(dataloader_train, dataloader_valid);
    println!("Training complete! Checkpoints saved to ./tmp/iam-classification");
}