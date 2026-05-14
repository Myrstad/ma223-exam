use crate::author_classifier::AuthorClassifier;
use crate::data;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::nn;
use burn::prelude::{Backend, Int, Shape, Tensor, TensorData};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use image::io::Reader as ImageReader;


// types for parsing IAM and representing IAM

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