# Arrow Datasets

Arrow Datasets is a Rust-based utility that converts large image datasets into Apache Arrow files. It is designed to efficiently process image data using parallel processing and chunking, making it ideal for machine learning pipelines and data analytics workflows.

## Features

- **Efficient Data Processing:** Converts image datasets into the high-performance Apache Arrow format.
- **Parallel Processing:** Utilizes multi-threading to process images in chunks for faster execution.
- **Chunked Data Storage:** Splits data into manageable chunks, each saved as a separate Arrow file.
- **Metadata Generation:** Automatically creates metadata (`dataset_info.json`) and state (`state.json`) files to describe the dataset.
- **Flexible Dataset Handling:** Supports training and validation datasets stored in a folder structure.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (version 1.XX or later) with Cargo.
- A dataset organized by labels (each label in its own folder) with images in `.webp` format.

## Installation

### Clone the Repository

```bash
git clone https://github.com/anto18671/arrow-datasets.git
cd arrow-datasets
```

### Build the Project

```bash
cargo build --release
```

## Usage

1. **Configure Dataset Paths:**  
   Update the input and output paths in `src/main.rs` if needed. By default, the code expects:

   - Training dataset at `D:/datasets/imagenet21k-p/train`
   - Validation dataset at `D:/datasets/imagenet21k-p/validation`
   - Output dataset will be saved at `D:/datasets/imagenet21k-p-arrow`

2. **Run the Application:**  
   Execute the project with the following command:
   ```bash
   cargo run --release
   ```
3. **Process Overview:**
   - The application scans the input directories to locate `.webp` images.
   - It collects image paths and their corresponding labels.
   - The images are shuffled and processed in parallel, split into chunks defined by a constant chunk size.
   - Each chunk is converted into an Apache Arrow file and saved in the output directory.
   - Metadata and state files are generated to document the dataset.

## Configuration

- **CHUNK_SIZE:**  
  Controls the number of samples per Arrow file. The default value is `49152`.

- **THREAD_COUNT:**  
  Determines the maximum number of threads used for parallel processing. The default value is `8`.

Modify these constants as necessary to fit your dataset size and available hardware resources.

## Directory Structure

- **Input Dataset Structure:**

  ```
  dataset_path/
  ├── train/
  │   ├── label1/
  │   │   └── image1.webp
  │   │   └── image2.webp
  │   └── label2/
  │       └── image1.webp
  └── validation/
      ├── label1/
      │   └── image1.webp
      └── label2/
          └── image1.webp
  ```

- **Output Dataset Structure:**
  ```
  output_path/
  ├── train/
  │   ├── data-00000-of-000XX.arrow
  │   ├── data-00001-of-000XX.arrow
  │   └── ...
  ├── validation/
  │   ├── data-00000-of-000XX.arrow
  │   ├── data-00001-of-000XX.arrow
  │   └── ...
  ├── dataset_info.json
  └── state.json
  ```

## How It Works

1. **Dataset Scanning:**  
   The tool recursively scans the specified directories for image files with a `.webp` extension.
2. **Data Collection:**  
   It collects each image's path along with its label (derived from the parent directory name).
3. **Data Shuffling:**  
   The image paths are shuffled to ensure randomness in the output.
4. **Chunk Processing:**  
   The dataset is split into chunks defined by `CHUNK_SIZE`. Each chunk is processed in a separate thread, ensuring that no more than `THREAD_COUNT` threads run concurrently.
5. **Arrow File Creation:**
   - Image data is read as binary data and stored in an Arrow BinaryArray.
   - Corresponding labels are stored in an Arrow StringArray.
   - These arrays are used to create a RecordBatch, which is then written to an Arrow file.
6. **Metadata Generation:**  
   After processing, the tool generates:
   - A `dataset_info.json` file containing dataset metadata.
   - A `state.json` file listing all generated Arrow files and their configuration.

## Dependencies

- [Apache Arrow](https://arrow.apache.org/) – Columnar in-memory analytics.
- [rand](https://crates.io/crates/rand) – Random number generation for shuffling.
- [serde](https://serde.rs/) – Serialization and deserialization of JSON.
- [walkdir](https://crates.io/crates/walkdir) – Directory traversal.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or suggestions, please open an issue on the [GitHub repository](https://github.com/anto18671/arrow-datasets.git).
