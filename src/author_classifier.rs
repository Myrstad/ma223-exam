use burn::module::Module;
use burn::nn;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNormConfig, DropoutConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::nn::pool::MaxPool2d;
use burn::prelude::{Backend, Tensor};
use crate::config::ModelConfig;

// The ML model itself. Which consists of convolutional layers, pooling layers, batch normalization layers, fully connected 1, and 2 (dense layers). Dropout and activation
#[derive(Module, Debug)]
pub struct AuthorClassifier<B: Backend> {
    conv_layers: Vec<Conv2d<B>>,
    pool_layers: Vec<MaxPool2d>,
    batch_norms: Vec<nn::BatchNorm<B, 2>>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: nn::Dropout,
    activation: String,
}

impl<B: Backend> AuthorClassifier<B> {
    // Creates a new Author Classifier CNN model
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        // Initialize using kaimingNormal (https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/)
        let initializer = nn::Initializer::KaimingNormal {
            gain: 1.414,
            fan_out_only: false,
        };

        let num_conv_layers = config.filters.len();
        let mut conv_layers = Vec::with_capacity(num_conv_layers);
        let mut pool_layers = Vec::with_capacity(num_conv_layers);
        let mut batch_norms = Vec::with_capacity(num_conv_layers);

        let mut in_channels = 1;

        let kernel_size = [config.kernel_size as usize, config.kernel_size as usize];
        let pool_size = [config.pool_size as usize, config.pool_size as usize];

        // for each convolution layer, create it based on the config, initialize it, and create a pooling layer and batch normalization layer
        for i in 0..num_conv_layers {
            let out_channels = config.filters[i] as usize;

            let conv = Conv2dConfig::new([in_channels, out_channels], kernel_size)
                .with_padding(PaddingConfig2d::Same)
                .with_initializer(initializer.clone())
                .init(device);
            conv_layers.push(conv);

            let pool = nn::pool::MaxPool2dConfig::new(pool_size)
                .with_strides(pool_size) // use stride == pool_size typically
                .init();
            pool_layers.push(pool);

            let bn = BatchNormConfig::new(out_channels).init(device);
            batch_norms.push(bn);

            in_channels = out_channels;
        }

        // Calculate flattened size after pooling
        // initial image is 64x64
        let mut spatial_dim = 64;
        for _ in 0..num_conv_layers {
            spatial_dim = spatial_dim / (config.pool_size as usize);
        }
        let flattened_size = in_channels * spatial_dim * spatial_dim;

        let fc1 = LinearConfig::new(flattened_size, config.fc_neurons as usize)
            .with_initializer(initializer)
            .init(device);

        let fc2 = LinearConfig::new(config.fc_neurons as usize, config.num_classes as usize)
            .init(device);

        Self {
            conv_layers,
            pool_layers,
            batch_norms,
            fc1,
            fc2,
            dropout: DropoutConfig::new(config.dropout as f64).init(),
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
            "linear" => x,
            _ => burn::tensor::activation::gelu(x),
        }
    }

    /// Forward propagation, applies all of the layers in the model basically and applies activation
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = x;

        for i in 0..self.conv_layers.len() {
            x = self.conv_layers[i].forward(x);
            x = self.batch_norms[i].forward(x);
            x = self.apply_activation(x);
            x = self.pool_layers[i].forward(x);
        }

        let x = x.flatten(1, 3);
        let x = self.fc1.forward(x);
        let x = self.apply_activation(x);
        let x = self.dropout.forward(x);

        self.fc2.forward(x)
    }
}