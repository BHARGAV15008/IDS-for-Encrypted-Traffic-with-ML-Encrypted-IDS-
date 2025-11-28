"""
Time Series Converter
Converts combined dataset into time series format for temporal analysis
Supports sliding windows, sequence generation, and temporal feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import importlib.util

# Dynamically load CSVDataPreprocessor
try:
    spec_data_preprocessor = importlib.util.spec_from_file_location(
        "dataPreprocessor",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/01_Data/02_Processed/dataPreprocessor.py"
    )
    dataPreprocessor_module = importlib.util.module_from_spec(spec_data_preprocessor)
    spec_data_preprocessor.loader.exec_module(dataPreprocessor_module)
    CSVDataPreprocessor = dataPreprocessor_module.CSVDataPreprocessor
except Exception as e:
    print(f"Error loading CSVDataPreprocessor: {e}")
    import sys
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesConverter:
    """Convert tabular data to time series format"""
    
    def __init__(self, outputDir: str = '01_Data/03_TimeSeries'):
        """
        Args:
            outputDir: Directory to save time series datasets
        """
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)
        self.csv_preprocessor = CSVDataPreprocessor() # Initialize the preprocessor
        
    def createTimeSeriesDataset(
        self,
        dataPath: str,
        windowSize: int = 10,
        stride: int = 1,
        sortBy: str = 'duration',
        groupBy: Optional[str] = None,
        includeLabels: bool = True,
        outputFilename: str = 'timeseries_dataset.npz'
    ) -> Dict[str, np.ndarray]:
        """
        Create time series dataset using sliding windows
        
        Args:
            dataPath: Path to combined dataset CSV
            windowSize: Number of time steps in each sequence
            stride: Step size for sliding window
            sortBy: Column to sort by (temporal ordering)
            groupBy: Optional column to group sequences (e.g., 'source_file', 'scenario')
            includeLabels: Whether to include labels in output
            outputFilename: Name of output file
            
        Returns:
            Dictionary with sequences, labels, and metadata
        """
        logger.info("=" * 80)
        logger.info("CREATING TIME SERIES DATASET")
        logger.info("=" * 80)
        
        # Load data
        logger.info(f"\nLoading data from: {dataPath}")
        df = pd.read_csv(dataPath)
        logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Preprocess the CSV data using CSVDataPreprocessor
        logger.info("Preprocessing CSV data with CSVDataPreprocessor...")
        df = self.csv_preprocessor.preprocessCSV(df)
        logger.info("CSV data preprocessing complete.")
        
        # Identify feature columns (exclude non-numeric)
        excludeCols = ['label', 'scenario', 'source_file', 'flowId'] # Added flowId to exclude
        
        # Get only numeric columns
        numericCols = df.select_dtypes(include=[np.number]).columns.tolist()
        featureCols = [col for col in numericCols if col not in excludeCols]
        
        # Encode categorical columns if any remain
        categoricalCols = df.select_dtypes(include=['object']).columns.tolist()
        categoricalCols = [col for col in categoricalCols if col not in excludeCols]
        
        if categoricalCols:
            logger.info(f"Encoding {len(categoricalCols)} categorical columns: {categoricalCols}")
            
            for col in categoricalCols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                featureCols.append(col)
        
        logger.info(f"\nFeature columns: {len(featureCols)}")
        logger.info(f"Window size: {windowSize}")
        logger.info(f"Stride: {stride}")
        
        # Sort data for temporal ordering
        if sortBy in df.columns:
            logger.info(f"Sorting by: {sortBy}")
            df = df.sort_values(by=sortBy).reset_index(drop=True)
        
        # Generate sequences
        if groupBy and groupBy in df.columns:
            logger.info(f"Grouping by: {groupBy}")
            sequences, labels = self._createGroupedSequences(
                df, featureCols, windowSize, stride, groupBy, includeLabels
            )
        else:
            logger.info("Creating sequences without grouping")
            sequences, labels = self._createSlidingWindowSequences(
                df, featureCols, windowSize, stride, includeLabels
            )
            
        # Save to file
        outputPath = self.outputDir / outputFilename
        logger.info(f"\nSaving time series dataset to: {outputPath}")
        
        saveDict = {
            'sequences': sequences,
            'window_size': windowSize,
            'stride': stride,
            'num_features': len(featureCols),
            'feature_names': featureCols
        }
        
        if includeLabels and labels is not None:
            saveDict['labels'] = labels
            
        np.savez_compressed(outputPath, **saveDict)
        
        # Generate summary
        self._generateTimeSeriesSummary(sequences, labels, windowSize, featureCols, outputPath)
        
        logger.info("\n" + "=" * 80)
        logger.info("TIME SERIES DATASET CREATED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return saveDict
        
    def _createSlidingWindowSequences(
        self,
        df: pd.DataFrame,
        featureCols: List[str],
        windowSize: int,
        stride: int,
        includeLabels: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences using sliding window approach
        
        Returns:
            Tuple of (sequences, labels)
            sequences shape: (num_sequences, window_size, num_features)
            labels shape: (num_sequences,)
        """
        logger.info("\nGenerating sliding window sequences...")
        
        # Extract features
        features = df[featureCols].values
        
        # Calculate number of sequences
        numSequences = (len(features) - windowSize) // stride + 1
        numFeatures = len(featureCols)
        
        logger.info(f"  Total samples: {len(features):,}")
        logger.info(f"  Number of sequences: {numSequences:,}")
        
        # Initialize arrays
        sequences = np.zeros((numSequences, windowSize, numFeatures), dtype=np.float32)
        labels = None
        
        if includeLabels and 'label' in df.columns:
            labels = np.zeros(numSequences, dtype=np.int64)
        
        # Generate sequences
        for i in tqdm(range(numSequences), desc="  Creating sequences"):
            startIdx = i * stride
            endIdx = startIdx + windowSize
            
            sequences[i] = features[startIdx:endIdx]
            
            if labels is not None:
                # Use the label of the last time step in the window
                labels[i] = df['label'].iloc[endIdx - 1]
                
        logger.info(f"✓ Created {numSequences:,} sequences")
        
        return sequences, labels
        
    def _createGroupedSequences(
        self,
        df: pd.DataFrame,
        featureCols: List[str],
        windowSize: int,
        stride: int,
        groupBy: str,
        includeLabels: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences grouped by a specific column
        Ensures sequences don't cross group boundaries
        
        Returns:
            Tuple of (sequences, labels)
        """
        logger.info(f"\nGenerating grouped sequences by '{groupBy}'...")
        
        allSequences = []
        allLabels = [] if includeLabels and 'label' in df.columns else None
        
        groups = df.groupby(groupBy)
        logger.info(f"  Number of groups: {len(groups)}")
        
        for groupName, groupDf in tqdm(groups, desc="  Processing groups"):
            # Extract features for this group
            features = groupDf[featureCols].values
            
            if len(features) < windowSize:
                logger.debug(f"  Skipping group '{groupName}' - insufficient samples ({len(features)} < {windowSize})")
                continue
                
            # Calculate sequences for this group
            numSequences = (len(features) - windowSize) // stride + 1
            
            for i in range(numSequences):
                startIdx = i * stride
                endIdx = startIdx + windowSize
                
                sequence = features[startIdx:endIdx]
                allSequences.append(sequence)
                
                if allLabels is not None:
                    label = groupDf['label'].iloc[endIdx - 1]
                    allLabels.append(label)
                    
        # Convert to numpy arrays
        sequences = np.array(allSequences, dtype=np.float32)
        labels = np.array(allLabels, dtype=np.int64) if allLabels is not None else None
        
        logger.info(f"✓ Created {len(sequences):,} sequences from {len(groups)} groups")
        
        return sequences, labels
        
    def createMultiWindowDataset(
        self,
        dataPath: str,
        windowSizes: List[int] = [5, 10, 20, 50],
        stride: int = 1,
        outputPrefix: str = 'timeseries'
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Create multiple time series datasets with different window sizes
        Useful for multi-scale temporal analysis
        
        Args:
            dataPath: Path to combined dataset
            windowSizes: List of window sizes to generate
            stride: Stride for sliding window
            outputPrefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping window size to dataset
        """
        logger.info("=" * 80)
        logger.info("CREATING MULTI-WINDOW TIME SERIES DATASETS")
        logger.info("=" * 80)
        
        results = {}
        
        for windowSize in windowSizes:
            logger.info(f"\n[Window Size: {windowSize}]")
            logger.info("-" * 80)
            
            outputFilename = f'{outputPrefix}_window{windowSize}.npz'
            
            dataset = self.createTimeSeriesDataset(
                dataPath=dataPath,
                windowSize=windowSize,
                stride=stride,
                outputFilename=outputFilename
            )
            
            results[windowSize] = dataset
            
        logger.info("\n" + "=" * 80)
        logger.info(f"CREATED {len(windowSizes)} TIME SERIES DATASETS")
        logger.info("=" * 80)
        
        return results
        
    def createSequenceWithContext(
        self,
        dataPath: str,
        windowSize: int = 10,
        pastContext: int = 5,
        futureContext: int = 5,
        outputFilename: str = 'timeseries_with_context.npz'
    ) -> Dict[str, np.ndarray]:
        """
        Create sequences with past and future context
        Useful for attention mechanisms and bidirectional models
        
        Args:
            dataPath: Path to combined dataset
            windowSize: Main window size
            pastContext: Number of past time steps to include
            futureContext: Number of future time steps to include
            outputFilename: Output filename
            
        Returns:
            Dictionary with sequences and context
        """
        logger.info("=" * 80)
        logger.info("CREATING TIME SERIES WITH CONTEXT")
        logger.info("=" * 80)
        
        # Load data
        df = pd.read_csv(dataPath)
        
        # Identify features
        excludeCols = ['label', 'scenario', 'source_file']
        featureCols = [col for col in df.columns if col not in excludeCols]
        features = df[featureCols].values
        
        totalWindow = pastContext + windowSize + futureContext
        numSequences = len(features) - totalWindow + 1
        numFeatures = len(featureCols)
        
        logger.info(f"Window size: {windowSize}")
        logger.info(f"Past context: {pastContext}")
        logger.info(f"Future context: {futureContext}")
        logger.info(f"Total window: {totalWindow}")
        logger.info(f"Number of sequences: {numSequences:,}")
        
        # Initialize arrays
        sequences = np.zeros((numSequences, windowSize, numFeatures), dtype=np.float32)
        pastContextSeq = np.zeros((numSequences, pastContext, numFeatures), dtype=np.float32)
        futureContextSeq = np.zeros((numSequences, futureContext, numFeatures), dtype=np.float32)
        labels = np.zeros(numSequences, dtype=np.int64) if 'label' in df.columns else None
        
        # Generate sequences
        for i in tqdm(range(numSequences), desc="Creating sequences with context"):
            # Past context
            pastContextSeq[i] = features[i:i + pastContext]
            
            # Main sequence
            sequences[i] = features[i + pastContext:i + pastContext + windowSize]
            
            # Future context
            futureContextSeq[i] = features[i + pastContext + windowSize:i + totalWindow]
            
            # Label (from main sequence end)
            if labels is not None:
                labels[i] = df['label'].iloc[i + pastContext + windowSize - 1]
                
        # Save
        outputPath = self.outputDir / outputFilename
        saveDict = {
            'sequences': sequences,
            'past_context': pastContextSeq,
            'future_context': futureContextSeq,
            'window_size': windowSize,
            'past_context_size': pastContext,
            'future_context_size': futureContext,
            'num_features': numFeatures,
            'feature_names': featureCols
        }
        
        if labels is not None:
            saveDict['labels'] = labels
            
        np.savez_compressed(outputPath, **saveDict)
        
        logger.info(f"\n✓ Saved to: {outputPath}")
        logger.info("=" * 80)
        
        return saveDict
        
    def _generateTimeSeriesSummary(
        self,
        sequences: np.ndarray,
        labels: Optional[np.ndarray],
        windowSize: int,
        featureCols: List[str],
        outputPath: Path
    ):
        """Generate summary statistics for time series dataset"""
        logger.info("\n" + "=" * 80)
        logger.info("TIME SERIES DATASET SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nDataset Shape:")
        logger.info(f"  Sequences: {sequences.shape}")
        logger.info(f"  - Number of sequences: {sequences.shape[0]:,}")
        logger.info(f"  - Window size: {sequences.shape[1]}")
        logger.info(f"  - Number of features: {sequences.shape[2]}")
        
        logger.info(f"\nMemory Usage:")
        memoryMB = sequences.nbytes / (1024 ** 2)
        logger.info(f"  Sequences: {memoryMB:.2f} MB")
        
        if labels is not None:
            labelMemoryMB = labels.nbytes / (1024 ** 2)
            logger.info(f"  Labels: {labelMemoryMB:.2f} MB")
            logger.info(f"  Total: {memoryMB + labelMemoryMB:.2f} MB")
            
            logger.info(f"\nLabel Distribution:")
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                percentage = (count / len(labels)) * 100
                labelName = "VPN" if label == 1 else "non-VPN" if label == 0 else f"Class {label}"
                logger.info(f"  {labelName}: {count:,} ({percentage:.2f}%)")
                
        logger.info(f"\nStatistics:")
        logger.info(f"  Mean: {np.mean(sequences):.6f}")
        logger.info(f"  Std: {np.std(sequences):.6f}")
        logger.info(f"  Min: {np.min(sequences):.6f}")
        logger.info(f"  Max: {np.max(sequences):.6f}")
        
        # Save summary to text file
        summaryPath = outputPath.parent / f'{outputPath.stem}_summary.txt'
        with open(summaryPath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TIME SERIES DATASET SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {outputPath.name}\n\n")
            
            f.write("Shape:\n")
            f.write(f"  Sequences: {sequences.shape}\n")
            f.write(f"  Number of sequences: {sequences.shape[0]:,}\n")
            f.write(f"  Window size: {sequences.shape[1]}\n")
            f.write(f"  Number of features: {sequences.shape[2]}\n\n")
            
            f.write("Features:\n")
            for i, col in enumerate(featureCols, 1):
                f.write(f"  {i:3d}. {col}\n")
                
            if labels is not None:
                f.write("\nLabel Distribution:\n")
                unique, counts = np.unique(labels, return_counts=True)
                for label, count in zip(unique, counts):
                    percentage = (count / len(labels)) * 100
                    labelName = "VPN" if label == 1 else "non-VPN" if label == 0 else f"Class {label}"
                    f.write(f"  {labelName}: {count:,} ({percentage:.2f}%)\n")
                    
        logger.info(f"\n✓ Summary saved to: {summaryPath}")


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert dataset to time series format')
    parser.add_argument('--data', required=True, help='Path to combined dataset CSV')
    parser.add_argument('--window-size', type=int, default=10, help='Window size for sequences')
    parser.add_argument('--stride', type=int, default=1, help='Stride for sliding window')
    parser.add_argument('--group-by', help='Column to group sequences by')
    parser.add_argument('--output', default='timeseries_dataset.npz', help='Output filename')
    parser.add_argument('--output-dir', default='01_Data/03_TimeSeries', help='Output directory')
    parser.add_argument('--multi-window', action='store_true', help='Create multiple window sizes')
    parser.add_argument('--with-context', action='store_true', help='Include past/future context')
    
    args = parser.parse_args()
    
    converter = TimeSeriesConverter(outputDir=args.output_dir)
    
    if args.multi_window:
        # Create multiple window sizes
        converter.createMultiWindowDataset(
            dataPath=args.data,
            windowSizes=[5, 10, 20, 50],
            stride=args.stride
        )
    elif args.with_context:
        # Create with context
        converter.createSequenceWithContext(
            dataPath=args.data,
            windowSize=args.window_size,
            pastContext=5,
            futureContext=5,
            outputFilename=args.output
        )
    else:
        # Standard time series
        converter.createTimeSeriesDataset(
            dataPath=args.data,
            windowSize=args.window_size,
            stride=args.stride,
            groupBy=args.group_by,
            outputFilename=args.output
        )


if __name__ == '__main__':
    main()
