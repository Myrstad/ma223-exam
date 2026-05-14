use std::fs;
use serde::{Deserialize, Serialize};
use crate::config::ExperimentParams;
use crate::runner::types::{ExperimentResults, TrainingHistory};

pub fn save_results(
    exp_id: &str,
    _history: &TrainingHistory,
    results: &ExperimentResults,
    experiment: &ExperimentParams,
    prefix: &str,
) -> std::io::Result<()> {
    let base_dir = format!("results/experiments/{}/{}", prefix, exp_id);
    fs::create_dir_all(&base_dir)?;

    let config_json = serde_json::to_string_pretty(experiment)?;
    fs::write(format!("{}/config.json", base_dir), config_json)?;

    let summary = serde_json::json!({
        "experiment_id": exp_id,
        "hyperparameters": {
            "max_epochs": experiment.max_epochs,
            "dropout": experiment.dropout,
            "learning_rate": experiment.learning_rate,
            "activation": experiment.activation,
        },
        "training": {
            "total_epochs": results.total_epochs,
            "early_stopped": results.early_stopped,
            "best_epoch": results.best_epoch,
            "runtime_seconds": results.runtime_seconds,
        },
        "metrics": {
            "train": {
                "final_loss": results.final_train_loss,
                "final_accuracy": results.final_train_acc,
            },
            "valid": {
                "best_loss": results.best_valid_loss,
                "best_accuracy": results.best_valid_acc,
                "epoch": results.best_epoch,
            },
            "test": {
                "accuracy": results.test_acc,
                "num_samples": results.test_num_samples,
                "num_correct": results.test_num_correct,
            },
        },
    });

    fs::write(
        format!("{}/summary.json", base_dir),
        serde_json::to_string_pretty(&summary)?,
    )?;

    // Update master summary
    update_master_summary(prefix, exp_id, experiment, results)?;

    Ok(())
}

fn update_master_summary(
    prefix: &str,
    exp_id: &str,
    experiment: &ExperimentParams,
    results: &ExperimentResults,
) -> std::io::Result<()> {
    let master_path = format!("results/experiments/{}/summary.json", prefix);

    #[derive(Deserialize, Serialize)]
    struct MasterSummary {
        experiment_name: String,
        total_experiments: usize,
        date_created: String,
        experiments: Vec<ExpSummary>,
        best_experiment: BestExp,
    }

    #[derive(Deserialize, Serialize, Clone)]
    struct ExpSummary {
        id: String,
        status: String,
        metrics: ExpMetrics,
        hyperparameters: serde_json::Value,
    }

    #[derive(Deserialize, Serialize, Clone)]
    struct ExpMetrics {
        train: TrainMetrics,
        valid: ValidMetrics,
        test: TestMetrics,
    }

    #[derive(Deserialize, Serialize, Clone)]
    struct TrainMetrics {
        accuracy: f32,
        loss: f32,
    }

    #[derive(Deserialize, Serialize, Clone)]
    struct ValidMetrics {
        accuracy: f32,
        loss: f32,
        epoch: u32,
    }

    #[derive(Deserialize, Serialize, Clone)]
    struct TestMetrics {
        accuracy: f32,
        loss: f32,
    }

    #[derive(Deserialize, Serialize)]
    struct BestExp {
        id: String,
        test_accuracy: f32,
    }

    // Create or load master summary
    let master: MasterSummary = if std::path::Path::new(&master_path).exists() {
        let content = fs::read_to_string(&master_path)?;
        serde_json::from_str(&content).unwrap_or_else(|_| {
            MasterSummary {
                experiment_name: prefix.to_string(),
                total_experiments: 81,
                date_created: chrono::Utc::now().to_rfc3339(),
                experiments: Vec::new(),
                best_experiment: BestExp { id: "none".to_string(), test_accuracy: 0.0 },
            }
        })
    } else {
        MasterSummary {
            experiment_name: prefix.to_string(),
            total_experiments: 81,
            date_created: chrono::Utc::now().to_rfc3339(),
            experiments: Vec::new(),
            best_experiment: BestExp { id: "none".to_string(), test_accuracy: 0.0 },
        }
    };

    // Create experiment summary entry
    let exp_summary = ExpSummary {
        id: exp_id.to_string(),
        status: "completed".to_string(),
        metrics: ExpMetrics {
            train: TrainMetrics {
                accuracy: results.final_train_acc,
                loss: results.final_train_loss,
            },
            valid: ValidMetrics {
                accuracy: results.best_valid_acc,
                loss: results.best_valid_loss,
                epoch: results.best_epoch,
            },
            test: TestMetrics {
                accuracy: results.test_acc,
                loss: results.test_loss,
            },
        },
        hyperparameters: serde_json::json!({
            "max_epochs": experiment.max_epochs,
            "dropout": experiment.dropout,
            "learning_rate": experiment.learning_rate,
            "activation": experiment.activation,
        }),
    };

    // Update experiments list - replace or add
    let mut new_experiments = master.experiments.clone();
    let pos = new_experiments.iter().position(|e| e.id == exp_id);
    match pos {
        Some(i) => new_experiments[i] = exp_summary,
        None => new_experiments.push(exp_summary),
    }

    // Find best by test accuracy
    let best = new_experiments.iter()
        .filter(|e| e.status == "completed")
        .max_by(|a, b| {
            a.metrics.test.accuracy.partial_cmp(&b.metrics.test.accuracy).unwrap()
        });

    let best_exp = match best {
        Some(e) => BestExp {
            id: e.id.clone(),
            test_accuracy: e.metrics.test.accuracy,
        },
        None => BestExp { id: "none".to_string(), test_accuracy: 0.0 },
    };

    let updated = MasterSummary {
        experiment_name: master.experiment_name,
        total_experiments: master.total_experiments,
        date_created: master.date_created,
        experiments: new_experiments,
        best_experiment: best_exp,
    };

    fs::write(&master_path, serde_json::to_string_pretty(&updated)?)?;

    Ok(())
}