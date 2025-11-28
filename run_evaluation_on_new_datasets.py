
import sys
import logging
from pathlib import Path
import pandas as pd
import glob
import dask.dataframe as dd

# Add project root to path to allow importing improvedCompleteWorkflow
project_root = Path(__file__).parent
sys.path.append(str(project_root / '04_Source_Code'))

try:
    from improvedCompleteWorkflow import ImprovedCompleteWorkflow
except ImportError as e:
    print("Could not import ImprovedCompleteWorkflow. Make sure the script is in 04_Source_Code.")
    print(e)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Removes leading/trailing spaces from column names."""
    df.columns = [col.strip() for col in df.columns]
    return df

def preprocess_for_workflow(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Prepares a dataframe for the ImprovedCompleteWorkflow."""
    logger.info(f"Preprocessing {dataset_name} dataset...")
    
    df = clean_column_names(df)
    
    # The 'TrafficLabelling_' datasets have extra non-numeric columns that must be dropped.
    # The 'MachineLearningCVE' datasets are mostly numeric, but we'll check anyway.
    
    # Identify columns that are definitely not features
    # Based on manual inspection of the CSV headers
    non_feature_cols = [
        'Source IP', 'Source Port', 'Destination IP', 
        'Destination Port', 'Protocol', 'Timestamp'
    ]
    
    cols_to_keep = [col for col in df.columns if col not in non_feature_cols]
    
    if len(cols_to_keep) < len(df.columns):
        logger.info(f"Selecting feature columns and label. Dropping {len(df.columns) - len(cols_to_keep)} non-feature columns.")
        df = df[cols_to_keep]
        
    # The workflow also handles a 'Label' column by name. Let's make sure it's consistent.
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})

    # The workflow's internal label encoder handles the 'label' column.
    # All other columns should be numeric at this point.
    
    # Let's verify dtypes before returning
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric = [col for col in df.columns if col not in numeric_cols and col != 'label']
    
    if non_numeric:
        # Keep 'Flow ID' if it's in non_numeric and needed for grouping
        if 'Flow ID' in non_numeric:
            logger.info(f"Retaining 'Flow ID' even though it's non-numeric.")
            non_numeric.remove('Flow ID')
        if non_numeric: # If there are still other non-numeric columns to drop
            logger.warning(f"Found unexpected non-numeric columns: {non_numeric}. Dropping them.")
            df = df.drop(columns=non_numeric)

    logger.info(f"Preprocessing for {dataset_name} complete. Shape: {df.shape}")
    return df

def process_directory_by_file(data_dir: Path, dataset_name: str):
    """
    Processes each CSV file in a directory individually to avoid memory issues.
    """
    logger.info(f"Processing files from {data_dir} one by one...")
    csv_files = sorted(glob.glob(str(data_dir / '*.csv')))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return
        
    for i, file_path in enumerate(csv_files):
        file_name = Path(file_path).name
        logger.info("-" * 60)
        logger.info(f"Processing file {i+1}/{len(csv_files)}: {file_name}")
        logger.info("-" * 60)

        try:
            df = pd.read_csv(file_path, encoding="latin1")

            # Downsample if the file is too large to avoid memory issues
            if len(df) > 10000:
                logger.info(f"File has {len(df)} rows, downsampling to 10,000.")
                df = df.sample(n=10000, random_state=42)
            
            processed_df = preprocess_for_workflow(df, file_name)
            
            if processed_df.empty:
                logger.warning(f"Skipping {file_name} as it was empty after preprocessing.")
                continue
            
            if 'label' not in processed_df.columns:
                logger.error(f"Critical: 'label' column not found in {file_name} after preprocessing. Skipping.")
                continue
                
            X = processed_df.drop(columns=['label'])
            y = processed_df['label']

            # Define a unique output directory for this specific file's results
            output_dir = project_root / "outputs" / f"evaluation_{dataset_name}" / file_name.replace('.csv', '')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create a config dictionary for the workflow
            workflow_config = {
                'data': {
                    'output_dir': str(output_dir),
                    'label_col': 'label', # Assuming 'label' is the label column
                    'test_size': 0.2,
                    'validation_size': 0.15,
                    'random_state': 42
                },
                'training': {
                    'device': 'cpu', # Hardcoding to 'cpu' as requested
                    'num_epochs': 2, # Reduced for faster execution
                    'batch_size': 512, # Default from original call
                    'early_stopping_patience': 10, # Default value
                    'learning_rate': 0.001, # Default value
                    'focal_gamma': 2.0 # Default value
                },
                'model': {
                    # Default model config, adjust if needed
                    'dropout_rate': 0.2,
                    'attention_heads': 4
                }
            }
            # Combine X and y into a single pandas DataFrame
            combined_df = pd.concat([X, y], axis=1)
            
            # Convert to Dask DataFrame
            ddf_for_workflow = dd.from_pandas(combined_df, npartitions=1) # Use 1 partition for smaller sampled data
            
            # Instantiate and run the workflow
            workflow = ImprovedCompleteWorkflow(workflow_config)
            results = workflow.run(ddf_for_workflow)
            
            logger.info(f"Workflow for {file_name} completed. Results saved to {output_dir}")

        except Exception as e:
            logger.critical(f"An error occurred while processing {file_name}: {e}", exc_info=True)



def main():
    """
    Main function to run the evaluation on the new datasets.
    """
    datasets_to_test = {
        "MachineLearningCVE": project_root / "01_Data" / "MachineLearningCVE",
        "TrafficLabelling": project_root / "01_Data" / "TrafficLabelling_",
    }
    
    for name, path in datasets_to_test.items():
        logger.info("=" * 80)
        logger.info(f"STARTING EVALUATION FOR DIRECTORY: {name}")
        logger.info("=" * 80)
        
        process_directory_by_file(path, name)

        logger.info("=" * 80)
        logger.info(f"COMPLETED EVALUATION FOR DIRECTORY: {name}")
        logger.info("=" * 80)

if __name__ == '__main__':
    main()
