mod types;
mod data;
mod config;
mod runner;
mod author_classifier;
mod iam;

use burn::backend::wgpu::WgpuDevice;

use crate::config::Config;
use crate::data::{init_dataset, prepare_datasets};
use crate::runner::{build_experiment_queue, run_pending_experiments};

fn main() {
    init_dataset().expect("Failed to initialize dataset");
    let device = WgpuDevice::default();

    let app_config = Config::load("config/config.json")
        .expect("Failed to load config");

    let (dataset_train, dataset_valid, dataset_test) = prepare_datasets(&app_config);

    // We only want to run "new" expiriments we havent ran before, in case the program was previously terminated during this experiment config
    let (mut pending, total) = build_experiment_queue(&app_config);
    
    if pending.is_empty() {
        println!("All {} experiments already completed!", total);
        return;
    }
    
    let prefix = app_config.experiment_name.clone();
    run_pending_experiments(&device, dataset_train, dataset_valid, dataset_test, &app_config, &prefix, &mut pending);
    
    println!("=== All {} experiments completed! ===", total);
}
