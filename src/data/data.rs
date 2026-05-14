use std::{fs, io};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, Cursor};
use std::path::{Path, PathBuf};


const DATASET_URL: &str = "https://www.kaggle.com/api/v1/datasets/download/tejasreddy/iam-handwriting-top50";
const DATASET_PATH: &str = "./data/iam_top50";

pub struct DatasetMetadata {
    pub image_paths: Vec<PathBuf>,
    pub labels: Vec<i32>,
    pub authors: Vec<String>,
    pub num_classes: usize,
}

pub fn init_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(DATASET_PATH);

    if path.exists() {
        println!("Dataset found at {:?}", DATASET_PATH);
        return Ok(());
    }

    println!("Dataset not found. Downloading...");

    fs::create_dir_all("./data")?;

    // Download the file
    let response = reqwest::blocking::get(DATASET_URL)?;
    let mut content = Cursor::new(response.bytes()?);

    println!("Extracting dataset...");
    zip_extract::extract(&mut content, path, true)?;

    println!("Dataset ready!");
    Ok(())
}

fn extract_form_id(filename: &str) -> Option<String> {
    filename.split("-s").next().map(|s| s.to_string())
}

pub fn load_metadata(forms_file: &str, data_dir: &str) -> io::Result<DatasetMetadata> {
    let mut form_to_author: HashMap<String, String> = HashMap::new();

    // Parse the text file
    let file = File::open(forms_file)?;
    for line in io::BufReader::new(file).lines() {
        let l = line?;
        if l.is_empty() || l.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = l.split_whitespace().collect();
        if parts.len() >= 2 {
            let form_id = parts[0].to_string();
            let author_id = parts[1].to_string();
            form_to_author.insert(form_id, author_id);
        }
    }

    let mut image_paths = Vec::new();
    let mut raw_labels = Vec::new();
    let mut raw_authors = Vec::new();
    let mut unique_authors = HashMap::new();
    let mut class_count = 0;

    for entry in walkdir::WalkDir::new(data_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
    {
        let path = entry.path().to_path_buf();
        let filename = path.file_name().unwrap().to_str().unwrap();

        if let Some(form_id) = extract_form_id(filename) {
            if let Some(author_id) = form_to_author.get(&form_id) {
                // Encode labels as integers (0 to N-1) for the Neural Network
                let label = *unique_authors.entry(author_id.clone()).or_insert_with(|| {
                    let val = class_count;
                    class_count += 1;
                    val
                });

                image_paths.push(path);
                raw_labels.push(label);
                raw_authors.push(author_id.clone());
            }
        }
    }

    println!("Matched {} images for {} authors", image_paths.len(), class_count);

    Ok(DatasetMetadata {
        image_paths,
        labels: raw_labels,
        authors: raw_authors,
        num_classes: class_count as usize,
    })
}