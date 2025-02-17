use arrow::array::{BinaryArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use rand::rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use walkdir::WalkDir;

// Define a struct to store dataset metadata
#[derive(Serialize, Deserialize)]
struct DatasetInfo {
    dataset_name: String,
    dataset_type: String,
    num_samples: usize,
    format: String,
}

// Define the chunk size constant for processing images
const CHUNK_SIZE: usize = 49152;

// Define the thread count constant for parallel processing
const THREAD_COUNT: usize = 8;

// Function to read an image file as raw bytes
fn read_image_as_bytes(image_path: &Path) -> Option<Vec<u8>> {
    // Open the file at the given path and return None on failure
    let mut file = File::open(image_path).ok()?;

    // Create a new buffer to store file contents
    let mut buffer = Vec::new();

    // Read the entire file into the buffer, returning None on failure
    file.read_to_end(&mut buffer).ok()?;

    // Return the buffer containing the file bytes
    Some(buffer)
}

// Function to collect image paths and labels from a directory
fn collect_image_paths(data_dir: &Path) -> Vec<(PathBuf, String)> {
    // Walk through the directory recursively and filter valid entries
    WalkDir::new(data_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            // Get the path from the entry
            let path = entry.path();
            // Check if the path is a file
            if path.is_file() {
                // Check if the file has an extension
                if let Some(ext) = path.extension() {
                    // Check if the extension is "webp"
                    if ext == "webp" {
                        // Get the parent directory of the file
                        if let Some(parent) = path.parent() {
                            // Get the label from the parent's file name as a string
                            if let Some(label) = parent.file_name().and_then(|s| s.to_str()) {
                                // Return the path and label as a tuple
                                return Some((path.to_path_buf(), label.to_string()));
                            }
                        }
                    }
                }
            }
            None
        })
        .collect()
}

// Function to process images in chunks and save them as Arrow files
fn save_to_chunked_arrow(
    image_paths: Vec<(PathBuf, String)>,
    output_dir: &Path,
    dataset_name: &str,
) {
    // Calculate the total number of samples from the image paths vector
    let total_samples = image_paths.len();

    // Calculate the number of chunks needed by rounding up
    let num_chunks = (total_samples + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Create a shared schema with two fields: image (binary) and label (UTF8), wrapped in an Arc for thread safety
    let schema = Arc::new(Schema::new(vec![
        Field::new("image", DataType::Binary, false),
        Field::new("label", DataType::Utf8, false),
    ]));

    // Print status message with dataset details
    println!(
        "Saving dataset '{}' with {} samples in {} chunks...",
        dataset_name, total_samples, num_chunks
    );

    // Create a channel to signal thread completion
    let (tx, rx) = mpsc::channel();

    // Create an Arc Mutex to manage the active thread count
    let active_threads = Arc::new(Mutex::new(0));

    // Iterate over each chunk (with its index) from the image paths
    for (i, chunk) in image_paths.chunks(CHUNK_SIZE).enumerate() {
        // Create the output file name for the current chunk in the format "data-00000-of-000XX.arrow"
        let file_name = format!("data-{:05}-of-{:05}.arrow", i, num_chunks);

        // Create the full file path in the output directory
        let file_path = output_dir.join(&file_name);

        // Clone the shared schema for use in the thread
        let schema_clone = Arc::clone(&schema);

        // Clone the sender for the thread
        let tx_clone = tx.clone();

        // Clone the active_threads Arc for the thread
        let active_threads_clone = Arc::clone(&active_threads);

        // Convert the current chunk slice to a vector
        let chunk = chunk.to_vec();

        // Loop until the number of active threads is less than THREAD_COUNT
        loop {
            // Lock the mutex to get the current active thread count
            let count = *active_threads_clone.lock().unwrap();

            // Break the loop if fewer than THREAD_COUNT threads are active
            if count < THREAD_COUNT {
                break;
            }

            // Sleep for a short duration to avoid busy-waiting
            thread::sleep(Duration::from_millis(100));
        }
        // Increment the active thread count before spawning a new thread
        {
            // Lock the mutex to modify the count
            let mut count = active_threads_clone.lock().unwrap();

            // Increment the active thread count by one
            *count += 1;
        }

        // Spawn a new thread to process the current chunk
        thread::spawn(move || {
            // Process the chunk by reading images and cloning labels; skip any failed reads
            let chunk_data: Vec<(Vec<u8>, String)> = chunk
                .iter()
                .filter_map(|(path, label)| {
                    read_image_as_bytes(path).map(|img_data| (img_data, label.clone()))
                })
                .collect();

            // Map each image data to a byte slice for Arrow array creation
            let images: Vec<&[u8]> = chunk_data
                .iter()
                .map(|(image, _)| image.as_slice())
                .collect();

            // Map each label to a string slice
            let labels: Vec<&str> = chunk_data.iter().map(|(_, label)| label.as_str()).collect();

            // Create a BinaryArray from the image byte slices
            let image_array = BinaryArray::from(images);

            // Create a StringArray from the labels
            let label_array = StringArray::from(labels);

            // Create a RecordBatch using the cloned schema and the two arrays
            let batch = arrow::record_batch::RecordBatch::try_new(
                schema_clone.clone(),
                vec![
                    std::sync::Arc::new(image_array),
                    std::sync::Arc::new(label_array),
                ],
            )
            .expect("Failed to create Arrow record batch");

            // Create the output file for writing the Arrow data
            let file = File::create(&file_path).expect("Failed to create Arrow file");

            // Create a FileWriter using the schema reference from the cloned Arc
            let mut writer =
                FileWriter::try_new(file, &*schema_clone).expect("Failed to create Arrow writer");

            // Write the RecordBatch data to the file
            writer.write(&batch).expect("Failed to write Arrow data");

            // Finalize the writing process to complete the Arrow file
            writer.finish().expect("Failed to finalize Arrow file");

            // Print a message indicating the chunk has been saved
            println!("Saved chunk {} -> {:?}", i, file_path);

            // Signal completion by sending a unit value through the channel
            tx_clone.send(()).unwrap();

            // Decrement the active thread count after the task is complete
            let mut count = active_threads_clone.lock().unwrap();

            // Decrement the active thread count by one
            *count -= 1;
        });
    }

    // Wait for all spawned threads to finish processing by receiving a signal for each chunk
    for _ in 0..num_chunks {
        rx.recv().unwrap();
    }

    // Save the dataset metadata and state after all chunks are processed
    save_metadata(output_dir, dataset_name, total_samples, num_chunks);
}

// Function to save dataset metadata and state information
fn save_metadata(output_dir: &Path, dataset_name: &str, num_samples: usize, num_chunks: usize) {
    // Create a DatasetInfo struct with the provided metadata
    let metadata = DatasetInfo {
        dataset_name: dataset_name.to_string(),
        dataset_type: "imagefolder".to_string(),
        num_samples,
        format: "arrow".to_string(),
    };

    // Serialize the metadata struct into a pretty JSON string
    let metadata_json =
        serde_json::to_string_pretty(&metadata).expect("Failed to serialize metadata");

    // Create the full path for the metadata file "dataset_info.json"
    let metadata_path = output_dir.join("dataset_info.json");

    // Create the metadata file
    let mut file = File::create(metadata_path).expect("Failed to create metadata file");

    // Write the JSON metadata into the file
    file.write_all(metadata_json.as_bytes())
        .expect("Failed to write metadata file");

    // Create a JSON object for the state information with data file names and type
    let state = serde_json::json!({
        "_data_files": (0..num_chunks).map(|i| {
            serde_json::json!({ "filename": format!("data-{:05}-of-{:05}.arrow", i, num_chunks) })
        }).collect::<Vec<_>>(),
        "_type": "arrow"
    });

    // Serialize the state JSON into a pretty string
    let state_json = serde_json::to_string_pretty(&state).expect("Failed to serialize state.json");

    // Create the full path for the state file "state.json"
    let state_path = output_dir.join("state.json");

    // Create the state file
    let mut file = File::create(state_path).expect("Failed to create state file");

    // Write the JSON state into the file
    file.write_all(state_json.as_bytes())
        .expect("Failed to write state file");

    // Print a message indicating that metadata and state.json have been saved successfully
    println!("Metadata and state.json saved in {:?}", output_dir);
}

// Main function to execute the dataset processing pipeline
fn main() {
    // Define the input dataset path
    let dataset_path = Path::new("D:/datasets/imagenet21k-p");

    // Define the output path for the Arrow dataset
    let output_path = Path::new("D:/datasets/imagenet21k-p-arrow");

    // Create the output directory if it does not exist
    fs::create_dir_all(output_path).expect("Failed to create output directory");

    // Define the path for the training data
    let train_path = dataset_path.join("train");

    // Define the path for the validation data
    let val_path = dataset_path.join("validation");

    // Print a message indicating scanning of the training dataset
    println!("Scanning train dataset...");

    // Collect image paths and labels for the training dataset
    let mut train_image_paths = collect_image_paths(&train_path);

    // Print a message indicating shuffling of the training dataset
    println!("Shuffling train dataset...");

    // Shuffle the training image paths using the thread random number generator
    train_image_paths.shuffle(&mut rng());

    // Create the output directory for training data
    let train_output = output_path.join("train");
    fs::create_dir_all(&train_output).expect("Failed to create train output directory");

    // Print a message indicating saving of the training dataset
    println!("Saving train dataset...");

    // Process and save the training dataset in chunks
    save_to_chunked_arrow(train_image_paths, &train_output, "imagenet21k-train");

    // Print a message indicating scanning of the validation dataset
    println!("Scanning validation dataset...");

    // Collect image paths and labels for the validation dataset
    let mut val_image_paths = collect_image_paths(&val_path);

    // Print a message indicating shuffling of the validation dataset
    println!("Shuffling validation dataset...");

    // Shuffle the validation image paths using the thread random number generator
    val_image_paths.shuffle(&mut rng());

    // Create the output directory for validation data
    let val_output = output_path.join("validation");
    fs::create_dir_all(&val_output).expect("Failed to create validation output directory");

    // Print a message indicating saving of the validation dataset
    println!("Saving validation dataset...");

    // Process and save the validation dataset in chunks
    save_to_chunked_arrow(val_image_paths, &val_output, "imagenet21k-validation");

    // Print a final message indicating that the dataset has been saved successfully
    println!("Dataset saved successfully in {:?}", output_path);
}
