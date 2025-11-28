import pandas as pd
from pathlib import Path
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def combine_csv_files(input_dirs: list, output_filename: str = 'combined_all_csv_data.csv', output_dir: str = '01_Data/02_Processed', sample_per_file: int = None) -> Path:
    """
    Combines all CSV files from a list of input directories into a single CSV file.
    If sample_per_file is provided, only this many rows will be read from each CSV file.

    Args:
        input_dirs (list): A list of paths to directories containing CSV files.
        output_filename (str): The name of the output combined CSV file.
        output_dir (str): The directory where the combined CSV file will be saved.
        sample_per_file (int, optional): If provided, only this many rows will be read from each CSV file.
                                         Useful for handling large datasets and memory constraints.

    Returns:
        Path: The absolute path to the combined CSV file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_output_path = output_path / output_filename
    
    # 1. First Pass: Collect all unique column names from all CSVs
    all_unique_columns = set()
    logger.info("  [1/2] Collecting all unique column names...")
    for input_dir_str in input_dirs:
        input_path = Path(input_dir_str)
        if not input_path.is_dir():
            continue
        csv_files = list(input_path.glob('*.csv'))
        for csv_file in csv_files:
            try:
                # Read only header to get column names
                df_head = pd.read_csv(csv_file, encoding='utf-8', nrows=0)
                all_unique_columns.update([col.strip() for col in df_head.columns])
            except UnicodeDecodeError:
                try:
                    df_head = pd.read_csv(csv_file, encoding='latin1', nrows=0)
                    all_unique_columns.update([col.strip() for col in df_head.columns])
                except Exception as e:
                    logger.warning(f"    Could not read header of {csv_file.name} to collect columns: {e}")
            except Exception as e:
                logger.warning(f"    Could not read header of {csv_file.name} to collect columns: {e}")
    
    if not all_unique_columns:
        logger.error("No unique columns found across all CSVs. Exiting.")
        return None
    
    sorted_columns = sorted(list(all_unique_columns))
    logger.info(f"    Found {len(sorted_columns)} unique columns.")
    
    # Flag to check if header has been written
    header_written = False
    total_rows_processed = 0 # Renamed for clarity, now counts rows

    logger.info("=" * 80)
    logger.info("COMBINING CSV DATASETS")
    logger.info("=" * 80)

    for input_dir_str in input_dirs:
        input_path = Path(input_dir_str)
        if not input_path.is_dir():
            logger.warning(f"Input directory not found: {input_path}. Skipping.")
            continue

        logger.info(f"\nProcessing directory: {input_path}")
        csv_files = list(input_path.glob('*.csv'))

        if not csv_files:
            logger.info(f"No CSV files found in {input_path}. Skipping.")
            continue

        for csv_file in csv_files:
            try:
                # Read in chunks to handle potentially large individual files
                for i, chunk_df in enumerate(pd.read_csv(csv_file, encoding='utf-8', chunksize=10000, nrows=sample_per_file)):
                    # Clean column names of the chunk
                    chunk_df.columns = [col.strip() for col in chunk_df.columns]
                    # Reindex chunk to have all unique columns
                    chunk_df = chunk_df.reindex(columns=sorted_columns)
                    
                    if not header_written:
                        chunk_df.to_csv(final_output_path, mode='w', header=True, index=False)
                        header_written = True
                        logger.info(f"  Created combined CSV with header from {csv_file.name}")
                    else:
                        chunk_df.to_csv(final_output_path, mode='a', header=False, index=False)
                    logger.info(f"  Appended {len(chunk_df)} rows from {csv_file.name} (chunk {i+1})")
                    total_rows_processed += len(chunk_df) # Increment by rows in chunk
            except UnicodeDecodeError:
                logger.warning(f"  UTF-8 decode error for {csv_file.name}. Trying 'latin1' encoding.")
                try:
                    for i, chunk_df in enumerate(pd.read_csv(csv_file, encoding='latin1', chunksize=10000, nrows=sample_per_file)):
                        # Clean column names of the chunk
                        chunk_df.columns = [col.strip() for col in chunk_df.columns]
                        # Reindex chunk to have all unique columns
                        chunk_df = chunk_df.reindex(columns=sorted_columns)
                        
                        if not header_written:
                            chunk_df.to_csv(final_output_path, mode='w', header=True, index=False)
                            header_written = True
                            logger.info(f"  Created combined CSV with header from {csv_file.name} (latin1)")
                        else:
                            chunk_df.to_csv(final_output_path, mode='a', header=False, index=False)
                        logger.info(f"  Appended {len(chunk_df)} rows from {csv_file.name} (chunk {i+1}, latin1)")
                        total_rows_processed += len(chunk_df)
                except Exception as e:
                    logger.error(f"Error reading {csv_file.name} with 'latin1' encoding: {e}. Skipping file.")
                    continue
            except Exception as e:
                logger.error(f"Error reading {csv_file.name}: {e}. Skipping file.")
                continue
            
    if not header_written: # If no files were processed successfully
        logger.error("No data was processed or written to the combined CSV. Exiting.")
        return None

    logger.info(f"\nCombined CSV saved to: {final_output_path}")
    logger.info(f"Total rows processed: {total_rows_processed}")
    
    return final_output_path

    return final_output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine multiple CSV files from specified directories.")
    parser.add_argument("--input_dirs", nargs='+', required=True, help="List of input directories containing CSV files.")
    parser.add_argument("--output_filename", type=str, default="combined_all_csv_data.csv", help="Name of the output combined CSV file.")
    parser.add_argument("--output_dir", type=str, default="01_Data/02_Processed", help="Directory where the combined CSV file will be saved.")
    parser.add_argument("--sample_per_file", type=int, help="Number of rows to sample from each CSV file before combining.")

    args = parser.parse_args()

    combined_file_path = combine_csv_files(args.input_dirs, args.output_filename, args.output_dir, args.sample_per_file)
    if combined_file_path:
        logger.info(f"Successfully created combined CSV: {combined_file_path}")
    else:
        logger.error("Failed to create combined CSV.")