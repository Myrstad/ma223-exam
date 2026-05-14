use crate::config::hyper::Hyperparameters;
use crate::config::Config;
use crate::iam::IAMDataset;
use crate::runner::persistence::save_results;
use crate::runner::run_experiment;
use burn::backend::wgpu::WgpuDevice;

#[derive(Clone, Debug)]
pub struct ExpRun {
    pub exp: crate::config::hyper::ExperimentParams,
    pub repeat_idx: u32,
    pub run_id: String,
}

/// Builds the queue of pending experiments and returns (pending, total)
pub fn build_experiment_queue(config: &Config) -> (Vec<ExpRun>, usize) {
    let hyper = Hyperparameters::load("config/hyperparameters.json")
        .expect("Failed to load hyperparameters");

    let experiments = hyper.iter_experiments();
    let total = experiments.len() * (config.num_repeats as usize);
    let prefix = &config.experiment_name;
    
    println!("Loaded {} experiment configurations to run {} repeats each ({} total)",
             experiments.len(), config.num_repeats, total);
    println!("Experiment prefix: {}", prefix);
    println!("Config: batch_size={}, num_classes={}", config.batch_size, config.cnn.num_classes);
    println!();

    let mut completed = 0;
    let mut pending = Vec::new();

    for exp in &experiments {
        for repeat in 0..config.num_repeats {
            let run_id = if config.num_repeats > 1 {
                format!("{}_rep{}", exp.id, repeat + 1)
            } else {
                exp.id.clone()
            };

            let exp_dir = format!("results/experiments/{}/{}", prefix, run_id);
            if std::path::Path::new(&exp_dir).exists() {
                completed += 1;
            } else {
                pending.push(ExpRun {
                    exp: exp.clone(),
                    repeat_idx: repeat,
                    run_id,
                });
            }
        }
    }

    println!("{} already completed, {} to run\n", completed, pending.len());

    (pending, total)
}

pub fn run_pending_experiments(
    device: &WgpuDevice,
    dataset_train: IAMDataset,
    dataset_valid: IAMDataset,
    dataset_test: IAMDataset,
    app_config: &Config,
    prefix: &String,
    pending: &mut Vec<ExpRun>
) {
    for (i, run) in pending.iter().enumerate() {
        println!("=== [{}/{}] Running experiment {} ===", i + 1, pending.len(), run.run_id);

        // We clone the experiment and override the ID so runner saves to correct path
        let mut run_exp = run.exp.clone();
        run_exp.id = run.run_id.clone();

        let (history, results) = run_experiment(
            &run_exp,
            dataset_train.clone(),
            dataset_valid.clone(),
            dataset_test.clone(),
            &app_config.cnn,
            app_config.batch_size,
            app_config.early_stopping.patience,
            device,
            prefix,
        );

        println!("Final: train_acc={}, valid_loss={:.4}, valid_acc={}, test_acc={}",
                 results.final_train_acc, results.best_valid_loss, results.best_valid_acc,
                 results.test_acc);

        // Save results
        save_results(&results.experiment_id, &history, &results, &run_exp, prefix).expect("Failed to save results");
        println!("Results saved to results/experiments/{}/{}\n", prefix, run_exp.id);
    }
}
