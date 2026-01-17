# Source Code Overview

This directory contains the core source code for the Encrypted Traffic IDS project.

## Key Files:

*   `run_pipeline.py`: This is the main script for the end-to-end machine learning pipeline. It handles data loading, preprocessing, feature engineering, model training (CNN-BiLSTM with Attention), and evaluation. It is designed to be scalable using Dask and configurable via `ids_configuration.yaml`.
*   `ids_configuration.yaml`: The primary configuration file for `run_pipeline.py`. It defines data paths, model hyperparameters, training settings, and other pipeline parameters.
*   `cli/cliInterface.py`: Provides a command-line interface (CLI) for interacting with the IDS system, including running the complete pipeline.
*   `run_all_workflows.py`: A utility script to run the `run_pipeline.py` across multiple datasets and classification types for comprehensive testing and evaluation.

## Running the Pipeline

The recommended way to run the complete IDS pipeline is using the provided CLI.

### Prerequisites:

Ensure you have all necessary dependencies installed (e.g., from `requirements.txt`).

### Usage:

Navigate to the project root directory and use the `main.py` script as the entry point for the CLI.

#### Example: Run the pipeline with default settings

```bash
python 04_Source_Code/main.py run
```

#### Example: Run the pipeline with a specific data file and multiclass classification

```bash
python 04_Source_Code/main.py run --data "01_Data/CSVs/Darknet.csv" --classification_type multiclass --epochs 20
```

#### Example: Run the pipeline with a custom configuration file

```bash
python 04_Source_Code/main.py run --config "path/to/your/custom_config.yaml"
```

### Configuration:

The pipeline's behavior is primarily controlled by `ids_configuration.yaml`. You can modify this file to adjust:

*   **`data.paths`**: Specify the input data files (CSV or ARFF).
*   **`data.label_col`**: The name of the target variable column.
*   **`model`**: Model architecture parameters (e.g., `dropout_rate`, `attention_heads`).
*   **`training`**: Training parameters (e.g., `learning_rate`, `num_epochs`, `batch_size`).

For more advanced configuration, refer to the comments within `ids_configuration.yaml` and `run_pipeline.py`.