"""
Data Combiner Script
Combines all ARFF files from multiple scenario folders into a single CSV dataset
Handles ARFF to CSV conversion and data merging
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARFFToCSVConverter:
    """Convert ARFF files to CSV format"""
    
    def __init__(self):
        self.attributeNames = []
        self.attributeTypes = []
        
    def parseArff(self, arffPath: str) -> pd.DataFrame:
        """
        Parse ARFF file and convert to DataFrame
        
        Args:
            arffPath: Path to ARFF file
            
        Returns:
            DataFrame with parsed data
        """
        logger.info(f"Parsing ARFF file: {Path(arffPath).name}")
        
        with open(arffPath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Parse header
        dataSection = False
        attributes = []
        data = []
        
        for line in lines:
            # Remove trailing commas and strip
            line = line.rstrip(',').strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                continue
                
            # Check for @data section (case insensitive)
            if line.lower().startswith('@data'):
                dataSection = True
                continue
                
            # Parse attributes
            if line.lower().startswith('@attribute'):
                # Extract attribute name
                match = re.match(r'@attribute\s+(\S+)\s+(.+)', line, re.IGNORECASE)
                if match:
                    attrName = match.group(1).strip("'\"").strip()
                    attrType = match.group(2).strip().rstrip(',')
                    attributes.append(attrName)
                continue
                
            # Parse data
            if dataSection:
                # Simple comma split (ARFF data is simple CSV)
                values = line.split(',')
                # Strip whitespace from each value
                values = [v.strip() for v in values]
                # Only add if we have the right number of columns
                if len(values) == len(attributes):
                    data.append(values)
                    
        # Create DataFrame
        if not data:
            logger.warning(f"No data found in {Path(arffPath).name}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=attributes)
        
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
                
        logger.info(f"  ✓ Parsed {len(df)} rows, {len(df.columns)} columns")
        
        return df
        
    def _splitCsvLine(self, line: str) -> List[str]:
        """Split CSV line handling quoted values"""
        values = []
        current = []
        inQuotes = False
        
        for char in line:
            if char == '"' or char == "'":
                inQuotes = not inQuotes
            elif char == ',' and not inQuotes:
                values.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
                
        if current:
            values.append(''.join(current).strip())
            
        return values


class DataCombiner:
    """Combine multiple datasets from different scenarios"""
    
    def __init__(self, outputDir: str = '01_Data/02_Processed'):
        """
        Args:
            outputDir: Directory to save combined dataset
        """
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)
        self.converter = ARFFToCSVConverter()
        
    def combineScenarios(
        self,
        scenarioDirs: List[str],
        outputFilename: str = 'combined_dataset.csv',
        addScenarioLabel: bool = True,
        sampleSize: int = None,
        createVpnLabel: bool = True
    ) -> pd.DataFrame:
        """
        Combine all ARFF files from multiple scenario directories
        
        Args:
            scenarioDirs: List of scenario directory paths
            outputFilename: Name of output CSV file
            addScenarioLabel: Whether to add scenario identifier column
            sampleSize: Optional sample size per file (for testing)
            createVpnLabel: Whether to create VPN/non-VPN labels (1=VPN, 0=non-VPN)
            
        Returns:
            Combined DataFrame
        """
        logger.info("=" * 80)
        logger.info("COMBINING DATASETS FROM MULTIPLE SCENARIOS")
        logger.info("=" * 80)
        
        allDataframes = []
        fileCount = 0
        totalRows = 0
        
        for scenarioIdx, scenarioDir in enumerate(scenarioDirs):
            scenarioPath = Path(scenarioDir)
            scenarioName = scenarioPath.name
            
            logger.info(f"\n[Scenario {scenarioIdx + 1}/{len(scenarioDirs)}] Processing: {scenarioName}")
            logger.info("-" * 80)
            
            # Find all ARFF files
            arffFiles = list(scenarioPath.glob('*.arff'))
            
            if not arffFiles:
                logger.warning(f"  No ARFF files found in {scenarioName}")
                continue
                
            logger.info(f"  Found {len(arffFiles)} ARFF files")
            
            # Process each ARFF file
            for arffFile in tqdm(arffFiles, desc=f"  Processing {scenarioName}"):
                try:
                    # Convert ARFF to DataFrame
                    df = self.converter.parseArff(str(arffFile))
                    
                    if df.empty:
                        continue
                        
                    # Sample if requested
                    if sampleSize and len(df) > sampleSize:
                        df = df.sample(n=sampleSize, random_state=42)
                        logger.info(f"    Sampled {sampleSize} rows from {arffFile.name}")
                        
                    # Add scenario label
                    if addScenarioLabel:
                        df['scenario'] = scenarioName
                        df['source_file'] = arffFile.name
                    
                    # Create VPN label if requested
                    if createVpnLabel:
                        df = self._createVpnLabel(df, arffFile.name)
                        
                    allDataframes.append(df)
                    fileCount += 1
                    totalRows += len(df)
                    
                except Exception as e:
                    logger.error(f"  ✗ Error processing {arffFile.name}: {str(e)}")
                    continue
                    
        if not allDataframes:
            logger.error("No data was successfully loaded!")
            return pd.DataFrame()
            
        logger.info("\n" + "=" * 80)
        logger.info("COMBINING ALL DATAFRAMES")
        logger.info("=" * 80)
        
        # Combine all dataframes
        logger.info(f"Combining {len(allDataframes)} dataframes...")
        combinedDf = pd.concat(allDataframes, ignore_index=True, sort=False)
        
        # Handle missing values
        logger.info("Handling missing values...")
        missingBefore = combinedDf.isnull().sum().sum()
        
        # Fill numeric columns with median
        numericCols = combinedDf.select_dtypes(include=[np.number]).columns
        for col in numericCols:
            if combinedDf[col].isnull().any():
                combinedDf[col].fillna(combinedDf[col].median(), inplace=True)
                
        # Fill categorical columns with mode
        categoricalCols = combinedDf.select_dtypes(include=['object']).columns
        for col in categoricalCols:
            if combinedDf[col].isnull().any():
                combinedDf[col].fillna(combinedDf[col].mode()[0] if not combinedDf[col].mode().empty else 'Unknown', inplace=True)
                
        missingAfter = combinedDf.isnull().sum().sum()
        logger.info(f"  ✓ Filled {missingBefore - missingAfter} missing values")
        
        # Remove duplicate rows
        logger.info("Removing duplicate rows...")
        beforeDuplicates = len(combinedDf)
        combinedDf.drop_duplicates(inplace=True)
        afterDuplicates = len(combinedDf)
        logger.info(f"  ✓ Removed {beforeDuplicates - afterDuplicates} duplicate rows")
        
        # Save to CSV
        outputPath = self.outputDir / outputFilename
        logger.info(f"\nSaving combined dataset to: {outputPath}")
        combinedDf.to_csv(outputPath, index=False)
        
        # Generate summary statistics
        self._generateSummary(combinedDf, fileCount, scenarioDirs)
        
        # Save summary
        summaryPath = self.outputDir / 'dataset_summary.txt'
        self._saveSummary(combinedDf, summaryPath, fileCount, scenarioDirs)
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA COMBINATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Combined dataset saved to: {outputPath}")
        logger.info(f"Summary saved to: {summaryPath}")
        
        return combinedDf
        
    def combine_csv_scenarios(
        self,
        scenario_dirs: List[str],
        output_filename: str = 'combined_dataset.csv',
        add_scenario_label: bool = True,
        sample_size: int = None
    ) -> pd.DataFrame:
        """
        Combine all CSV files from multiple scenario directories
        
        Args:
            scenario_dirs: List of scenario directory paths
            output_filename: Name of output CSV file
            add_scenario_label: Whether to add scenario identifier column
            sample_size: Optional sample size per file (for testing)
            
        Returns:
            Combined DataFrame
        """
        logger.info("=" * 80)
        logger.info("COMBINING CSV DATASETS FROM MULTIPLE SCENARIOS")
        logger.info("=" * 80)
        
        all_dataframes = []
        file_count = 0
        
        for scenario_dir in scenario_dirs:
            scenario_path = Path(scenario_dir)
            logger.info(f"Processing scenario: {scenario_path.name}")
            
            csv_files = list(scenario_path.glob('*.csv'))
            if not csv_files:
                logger.warning(f"No CSV files found in {scenario_dir}")
                continue
            
            for csv_file in tqdm(csv_files, desc=f"Processing {scenario_path.name}"):
                try:
                    try:
                        df = pd.read_csv(csv_file, low_memory=False)
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_file, low_memory=False, encoding='latin-1')
                    
                    # Clean column names
                    df.columns = [col.strip() for col in df.columns]
                    
                    # Downcast numeric columns
                    df = self._downcast_numeric(df)
                    
                    if sample_size and len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                    
                    if add_scenario_label:
                        df['scenario'] = scenario_path.name
                        
                    all_dataframes.append(df)
                    file_count += 1
                except Exception as e:
                    logger.error(f"Error processing {csv_file.name}: {e}")
                    continue
        
        if not all_dataframes:
            logger.error("No data was successfully loaded!")
            return pd.DataFrame()
            
        logger.info(f"Combining {len(all_dataframes)} dataframes...")
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        if sample_size and len(combined_df) > sample_size:
            combined_df = combined_df.sample(n=sample_size, random_state=42)

        # Handle missing values and duplicates as in combineScenarios
        # (This part can be refactored into a common function)
        logger.info("Handling missing values...")
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if combined_df[col].isnull().any():
                combined_df[col].fillna(combined_df[col].median(), inplace=True)
                
        categorical_cols = combined_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if combined_df[col].isnull().any():
                combined_df[col].fillna(combined_df[col].mode()[0] if not combined_df[col].mode().empty else 'Unknown', inplace=True)

        logger.info("Removing duplicate rows...")
        combined_df.drop_duplicates(inplace=True)
        
        output_path = self.outputDir / output_filename
        logger.info(f"Saving combined dataset to: {output_path}")
        combined_df.to_csv(output_path, index=False)
        
        logger.info("=" * 80)
        logger.info("CSV DATA COMBINATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return combined_df
        
    def _downcast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric columns to save memory"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        return df

    def _createVpnLabel(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Create VPN label based on filename
        VPN = 1, non-VPN = 0
        
        Args:
            df: DataFrame
            filename: Source filename
            
        Returns:
            DataFrame with label column
        """
        # Check if label column already exists
        labelCols = ['label', 'Label', 'class', 'Class']
        hasLabel = any(col in df.columns for col in labelCols)
        
        if not hasLabel:
            # Determine label from filename
            filename_lower = filename.lower()
            
            if 'vpn' in filename_lower and 'no-vpn' not in filename_lower:
                # VPN traffic
                df['label'] = 1
                logger.debug(f"    Created label: VPN (1) for {filename}")
            elif 'no-vpn' in filename_lower or 'novpn' in filename_lower:
                # Non-VPN traffic
                df['label'] = 0
                logger.debug(f"    Created label: non-VPN (0) for {filename}")
            else:
                # Check if it's from AllinOne (mixed traffic)
                if 'allinone' in filename_lower:
                    # For AllinOne files, we'll need to check if there's an existing label
                    # If not, mark as mixed (we can use -1 or create separate logic)
                    df['label'] = -1  # Mixed traffic, needs further classification
                    logger.debug(f"    Created label: Mixed (-1) for {filename}")
                else:
                    # Default to non-VPN if unclear
                    df['label'] = 0
                    logger.debug(f"    Created label: non-VPN (0, default) for {filename}")
        else:
            # Label exists, map it to binary if needed
            existingLabelCol = next(col for col in labelCols if col in df.columns)
            
            # If label is string-based, convert to binary
            if df[existingLabelCol].dtype == 'object':
                df['label'] = df[existingLabelCol].apply(
                    lambda x: 1 if 'vpn' in str(x).lower() else 0
                )
                logger.debug(f"    Converted existing label to binary for {filename}")
            else:
                # Rename to standard 'label' column
                if existingLabelCol != 'label':
                    df['label'] = df[existingLabelCol]
                    
        return df
    
    def _generateSummary(self, df: pd.DataFrame, fileCount: int, scenarioDirs: List[str]):
        """Generate and display summary statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nSource Information:")
        logger.info(f"  - Scenarios processed: {len(scenarioDirs)}")
        logger.info(f"  - Files combined: {fileCount}")
        
        logger.info(f"\nDataset Dimensions:")
        logger.info(f"  - Total rows: {len(df):,}")
        logger.info(f"  - Total columns: {len(df.columns)}")
        
        logger.info(f"\nColumn Types:")
        logger.info(f"  - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        logger.info(f"  - Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
        
        logger.info(f"\nMemory Usage:")
        logger.info(f"  - Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Class distribution if label column exists
        labelCols = ['label', 'Label', 'class', 'Class', 'target', 'Target']
        labelCol = None
        for col in labelCols:
            if col in df.columns:
                labelCol = col
                break
                
        if labelCol:
            logger.info(f"\nClass Distribution ({labelCol}):")
            classCounts = df[labelCol].value_counts()
            for className, count in classCounts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  - {className}: {count:,} ({percentage:.2f}%)")
                
        # Scenario distribution if added
        if 'scenario' in df.columns:
            logger.info(f"\nScenario Distribution:")
            scenarioCounts = df['scenario'].value_counts()
            for scenario, count in scenarioCounts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  - {scenario}: {count:,} ({percentage:.2f}%)")
                
    def _saveSummary(self, df: pd.DataFrame, summaryPath: Path, fileCount: int, scenarioDirs: List[str]):
        """Save summary to text file"""
        with open(summaryPath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMBINED DATASET SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Source Information:\n")
            f.write(f"  - Scenarios: {len(scenarioDirs)}\n")
            for i, scenario in enumerate(scenarioDirs, 1):
                f.write(f"    {i}. {Path(scenario).name}\n")
            f.write(f"  - Files combined: {fileCount}\n\n")
            
            f.write("Dataset Dimensions:\n")
            f.write(f"  - Total rows: {len(df):,}\n")
            f.write(f"  - Total columns: {len(df.columns)}\n\n")
            
            f.write("Columns:\n")
            for i, col in enumerate(df.columns, 1):
                dtype = df[col].dtype
                nullCount = df[col].isnull().sum()
                f.write(f"  {i:3d}. {col:40s} ({dtype}) - {nullCount} nulls\n")
                
            f.write("\n" + "=" * 80 + "\n")


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine ARFF or CSV files from multiple scenarios')
    parser.add_argument('--scenarios', nargs='+', required=True, help='Scenario directories')
    parser.add_argument('--output', default='combined_dataset.csv', help='Output filename')
    parser.add_argument('--output-dir', default='01_Data/02_Processed', help='Output directory')
    parser.add_argument('--sample', type=int, help='Sample size per file (for testing)')
    parser.add_argument('--no-scenario-label', action='store_true', help='Do not add scenario labels')
    parser.add_argument('--file-type', type=str, default='arff', choices=['arff', 'csv'], help='Type of files to combine')
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = DataCombiner(outputDir=args.output_dir)
    
    # Combine datasets based on file type
    if args.file_type == 'arff':
        combinedDf = combiner.combineScenarios(
            scenarioDirs=args.scenarios,
            outputFilename=args.output,
            addScenarioLabel=not args.no_scenario_label,
            sampleSize=args.sample
        )
    else:
        combinedDf = combiner.combine_csv_scenarios(
            scenario_dirs=args.scenarios,
            output_filename=args.output,
            add_scenario_label=not args.no_scenario_label,
            sample_size=args.sample
        )
    
    if not combinedDf.empty:
        print(f"\n✓ Successfully combined {len(combinedDf):,} rows")
        print(f"✓ Output saved to: {Path(args.output_dir) / args.output}")


def testCombinedDataset(datasetPath: str = '01_Data/02_Processed/combined_all_scenarios.csv'):
    """
    Test the combined dataset
    
    Args:
        datasetPath: Path to combined dataset
    """
    logger.info("=" * 80)
    logger.info("TESTING COMBINED DATASET")
    logger.info("=" * 80)
    
    # Load dataset
    logger.info(f"\nLoading dataset from: {datasetPath}")
    df = pd.read_csv(datasetPath)
    
    logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Check for label column
    if 'label' not in df.columns:
        logger.error("✗ No 'label' column found!")
        return False
        
    logger.info("\n" + "-" * 80)
    logger.info("LABEL DISTRIBUTION")
    logger.info("-" * 80)
    
    labelCounts = df['label'].value_counts().sort_index()
    totalSamples = len(df)
    
    for label, count in labelCounts.items():
        percentage = (count / totalSamples) * 100
        labelName = "VPN" if label == 1 else "non-VPN" if label == 0 else "Mixed/Other"
        logger.info(f"  Label {label} ({labelName:10s}): {count:8,} samples ({percentage:5.2f}%)")
        
    # Check for missing values
    logger.info("\n" + "-" * 80)
    logger.info("DATA QUALITY CHECKS")
    logger.info("-" * 80)
    
    missingTotal = df.isnull().sum().sum()
    logger.info(f"  Missing values: {missingTotal}")
    
    if missingTotal > 0:
        logger.warning(f"  ⚠️  Found {missingTotal} missing values")
        topMissing = df.isnull().sum().sort_values(ascending=False).head(5)
        logger.info("  Top columns with missing values:")
        for col, count in topMissing.items():
            if count > 0:
                logger.info(f"    - {col}: {count}")
    else:
        logger.info("  ✓ No missing values")
        
    # Check for duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"  Duplicate rows: {duplicates}")
    if duplicates > 0:
        logger.warning(f"  ⚠️  Found {duplicates} duplicate rows")
    else:
        logger.info("  ✓ No duplicate rows")
        
    # Check data types
    logger.info(f"\n  Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    logger.info(f"  Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Check for infinite values
    numericCols = df.select_dtypes(include=[np.number]).columns
    infCount = 0
    for col in numericCols:
        infCount += np.isinf(df[col]).sum()
        
    logger.info(f"  Infinite values: {infCount}")
    if infCount > 0:
        logger.warning(f"  ⚠️  Found {infCount} infinite values")
    else:
        logger.info("  ✓ No infinite values")
        
    # Sample data
    logger.info("\n" + "-" * 80)
    logger.info("SAMPLE DATA (First 5 rows)")
    logger.info("-" * 80)
    
    # Show first few columns and label
    displayCols = list(df.columns[:5]) + ['label']
    if 'scenario' in df.columns and 'scenario' not in displayCols:
        displayCols.append('scenario')
    if 'source_file' in df.columns and 'source_file' not in displayCols:
        displayCols.append('source_file')
        
    print(df[displayCols].head())
    
    # Check class balance
    logger.info("\n" + "-" * 80)
    logger.info("CLASS BALANCE ANALYSIS")
    logger.info("-" * 80)
    
    # Calculate imbalance ratio
    if len(labelCounts) == 2:
        majorityClass = labelCounts.max()
        minorityClass = labelCounts.min()
        imbalanceRatio = majorityClass / minorityClass
        
        logger.info(f"  Majority class: {majorityClass:,} samples")
        logger.info(f"  Minority class: {minorityClass:,} samples")
        logger.info(f"  Imbalance ratio: {imbalanceRatio:.2f}:1")
        
        if imbalanceRatio > 3:
            logger.warning(f"  ⚠️  Dataset is imbalanced (ratio > 3:1)")
            logger.info(f"  💡 Consider using class weights or resampling")
        else:
            logger.info(f"  ✓ Dataset is reasonably balanced")
    
    logger.info("\n" + "=" * 80)
    logger.info("DATASET TEST COMPLETED")
    logger.info("=" * 80)
    logger.info(f"\n✓ Dataset is ready for training!")
    logger.info(f"  Use: python 04_Source_Code/run_pipeline.py --data \"{datasetPath}\"")
    
    return True


if __name__ == '__main__':
    main()
