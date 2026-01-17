import subprocess
from pathlib import Path
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the base directory for the project
PROJECT_ROOT = Path(__file__).parent.parent

# Define the path to the run_pipeline.py script
WORKFLOW_SCRIPT = PROJECT_ROOT / '04_Source_Code' / 'run_pipeline.py'

# Define the base output directory
OUTPUT_BASE_DIR = PROJECT_ROOT / 'outputs' / 'all_workflows'

# Ensure the base output directory exists
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# List of dataset directories and individual files
DATA_PATHS = [
    PROJECT_ROOT / '01_Data' / 'CSVs',
    PROJECT_ROOT / '01_Data' / 'CSVs' / 'MachineLearningCVE',
    PROJECT_ROOT / '01_Data' / 'CSVs' / 'TrafficLabelling',
    PROJECT_ROOT / '01_Data' / 'Scenario A1-ARFF',
    PROJECT_ROOT / '01_Data' / 'Scenario A2-ARFF',
    PROJECT_ROOT / '01_Data' / 'Scenario B-ARFF',
]

def run_workflow_for_dataset(data_file: Path, classification_type: str, epochs: int = 50):
    """
    Runs the run_pipeline.py for a given dataset and classification type.
    """
    logger.info(f"Running workflow for: {data_file.name} - {classification_type} classification")

    # Construct the command to run the workflow script
    command = [
        'python3',
        str(WORKFLOW_SCRIPT),
        '--data', str(data_file),
        '--classification_type', classification_type,
        '--epochs', str(epochs),
        '--output_base_dir', str(OUTPUT_BASE_DIR)
    ]

    try:
        # Run the command
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"Successfully ran workflow for {data_file.name} ({classification_type}).")
        logger.debug(f"STDOUT:\n{process.stdout}")
        logger.debug(f"STDERR:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running workflow for {data_file.name} ({classification_type}): {e}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for {data_file.name} ({classification_type}): {e}")

def main():
    """
    Main function to iterate through all specified datasets and run workflows.
    """
    all_files_to_process = []

    # Collect all files to process
    for data_path in DATA_PATHS:
        if data_path.is_file():
            all_files_to_process.append(data_path)
        elif data_path.is_dir():
            for ext in ['*.csv', '*.arff']:
                all_files_to_process.extend(data_path.glob(ext))
        else:
            logger.warning(f"Invalid data path specified: {data_path}. Skipping.")
    
    if not all_files_to_process:
        logger.error("No data files found to process. Exiting.")
        return

    logger.info(f"Found {len(all_files_to_process)} data files to process.")

    for data_file in sorted(all_files_to_process):
        logger.info(f"\n{'='*100}\nProcessing dataset: {data_file.name}\n{'='*100}")
        
        # Run for binary classification
        run_workflow_for_dataset(data_file, "binary")
        
        # Run for multiclass classification
        run_workflow_for_dataset(data_file, "multiclass")

    logger.info("\nAll workflows completed.")

if __name__ == '__main__':
    main()
