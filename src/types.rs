use burn::backend::Autodiff;
use burn::backend::Wgpu;

pub type MyBackend = Wgpu;
pub type MyAutodiffBackend = Autodiff<MyBackend>; // Makes the backend support backpropagation
