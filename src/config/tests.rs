
#[cfg(test)]
mod tests_integration {
    use crate::config::app::Config;
    use crate::config::hyper::Hyperparameters;

    #[test]
    fn test_load_hyperparameters() {
        let hyper = Hyperparameters::load("config/hyperparameters.json").unwrap();
        assert_eq!(hyper.total_experiments(), 81);
    }

    #[test]
    fn test_load_config() {
        let config = Config::load("config/config.json").unwrap();
        assert_eq!(config.experiment_name, "author_classification_grid_v1");
    }

    #[test]
    fn test_generate_all_experiments() {
        let hyper = Hyperparameters::load("config/hyperparameters.json").unwrap();
        let experiments = hyper.iter_experiments();
        assert_eq!(experiments.len(), 81);
        
        assert_eq!(experiments[0].max_epochs.to_u32(), Some(25));
        assert_eq!(experiments[0].dropout, 0.2);
        assert_eq!(experiments[0].learning_rate, 0.001);
        assert_eq!(experiments[0].activation, "gelu");
        
        assert_eq!(experiments.last().unwrap().id, "exp_081");
    }
}

#[cfg(test)]
mod tests {
    use crate::config::app::Config;
    use crate::config::hyper::{Hyperparameters, MaxEpochValue};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn temp_json(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    #[test]
    fn test_parse_hyperparameters() {
        let content = r#"{
            "max_epochs": [25, 50, "no_improvement"],
            "dropout": [0.2, 0.3],
            "learning_rate": [0.001, 0.01],
            "activation": ["gelu", "relu"]
        }"#;
        
        let file = temp_json(content);
        let hyper = Hyperparameters::load(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(hyper.max_epochs.len(), 3);
        assert_eq!(hyper.dropout.len(), 2);
        assert_eq!(hyper.learning_rate.len(), 2);
        assert_eq!(hyper.activation.len(), 2);
        assert_eq!(hyper.total_experiments(), 24);
    }

    #[test]
    fn test_max_epochs_value() {
        assert!(!MaxEpochValue::Integer(25).is_no_improvement());
        assert!(!MaxEpochValue::Integer(50).is_no_improvement());
        assert!(MaxEpochValue::String("no_improvement".to_string()).is_no_improvement());
        
        assert_eq!(MaxEpochValue::Integer(25).to_u32(), Some(25));
        assert_eq!(MaxEpochValue::String("no_improvement".to_string()).to_u32(), None);
    }

    #[test]
    fn test_experiment_generation() {
        let hyper = Hyperparameters {
            max_epochs: vec![MaxEpochValue::Integer(25)],
            dropout: vec![0.2, 0.3],
            learning_rate: vec![0.001],
            activation: vec!["gelu".to_string()],
        };
        
        let experiments = hyper.iter_experiments();
        assert_eq!(experiments.len(), 2);
        assert_eq!(experiments[0].id, "exp_001");
        assert_eq!(experiments[1].id, "exp_002");
    }

    #[test]
    fn test_parse_config() {
        let content = r#"{
            "experiment_name": "test_exp",
            "num_repeats": 2,
            "batch_size": 64,
            "cnn": {
                "conv_layers": 2,
                "filters": [32, 64],
                "kernel_size": 3,
                "pool_size": 2,
                "fc_neurons": 512,
                "num_classes": 50
            },
            "early_stopping": {
                "patience": 10,
                "min_delta": 0.001
            },
            "data": {
                "train_split": 0.8,
                "valid_split": 0.2,
                "test_split": 0.0,
                "forms_file": "data/forms.txt",
                "images_dir": "data/images"
            },
            "output": {
                "base_dir": "results"
            }
        }"#;
        
        let file = temp_json(content);
        let config = Config::load(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.experiment_name, "test_exp");
        assert_eq!(config.num_repeats, 2);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.cnn.num_classes, 50);
        assert_eq!(config.early_stopping.patience, 10);
    }
}