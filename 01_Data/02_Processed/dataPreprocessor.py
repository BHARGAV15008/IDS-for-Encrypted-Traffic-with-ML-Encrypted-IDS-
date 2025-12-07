"""
Data Preprocessing Module
Handles preprocessing for both CSV and PCAP files
Supports encrypted and unencrypted traffic data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataPreprocessor:
    """Base class for data preprocessing"""
    
    def __init__(self):
        self.labelEncoder = LabelEncoder()
        self.columnMapping = {}
        self.isFitted = False
        
    def handleMissingValues(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            data: Input dataframe
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop', 'zero')
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if strategy == 'drop':
            return data.dropna()
        elif strategy == 'zero':
            return data.fillna(0)
        elif strategy == 'mean':
            numericCols = data.select_dtypes(include=[np.number]).columns
            data[numericCols] = data[numericCols].fillna(data[numericCols].mean())
            return data
        elif strategy == 'median':
            numericCols = data.select_dtypes(include=[np.number]).columns
            data[numericCols] = data[numericCols].fillna(data[numericCols].median())
            return data
        elif strategy == 'mode':
            for col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 0)
            return data
        else:
            return data
            
    def removeOutliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from dataset
        
        Args:
            data: Input dataframe
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
        """
        logger.info(f"Removing outliers using method: {method}")
        
        numericCols = data.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numericCols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lowerBound = Q1 - threshold * IQR
                upperBound = Q3 + threshold * IQR
                data = data[(data[col] >= lowerBound) & (data[col] <= upperBound)]
                
        elif method == 'zscore':
            from scipy import stats
            zScores = np.abs(stats.zscore(data[numericCols]))
            data = data[(zScores < threshold).all(axis=1)]
            
        logger.info(f"Dataset size after outlier removal: {len(data)}")
        return data
        
    def encodeCategoricalFeatures(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            data: Input dataframe
            columns: List of columns to encode (if None, auto-detect)
        """
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
        logger.info(f"Encoding categorical features: {columns}")
        
        for col in columns:
            if col in data.columns:
                if not self.isFitted:
                    data[col] = self.labelEncoder.fit_transform(data[col].astype(str))
                else:
                    # Handle unseen labels
                    knownLabels = set(self.labelEncoder.classes_)
                    data[col] = data[col].apply(
                        lambda x: self.labelEncoder.transform([x])[0] if x in knownLabels else -1
                    )
                    
        return data
        
    def normalizeData(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            data: Input dataframe
            method: Normalization method ('minmax', 'standard', 'robust')
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        numericCols = data.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return data
            
        data[numericCols] = scaler.fit_transform(data[numericCols])
        logger.info(f"Data normalized using {method} method")
        
        return data


class CSVDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor for CSV-based network traffic data"""
    
    def __init__(self):
        super().__init__()
        self.requiredColumns = []
        
    def loadCSV(self, filePath: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file in chunks to handle large datasets.
        
        Args:
            filePath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
        """
        logger.info(f"Loading CSV file in chunks: {filePath}")
        
        try:
            return pd.read_csv(filePath, **kwargs)
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
            
    def standardizeColumnNames(self, data: pd.DataFrame, labelColumn: str) -> pd.DataFrame:
        """
        Standardize column names across different datasets
        Common datasets: NSL-KDD, UNSW-NB15, CIC-IDS, etc.
        """
        # Common column mappings
        columnMappings = {
            'Source IP': 'srcIP',
            'src_ip': 'srcIP',
            'source_ip': 'srcIP',
            'saddr': 'srcIP',
            'Destination IP': 'dstIP',
            'dst_ip': 'dstIP',
            'destination_ip': 'dstIP',
            'daddr': 'dstIP',
            'Source Port': 'srcPort',
            'src_port': 'srcPort',
            'source_port': 'srcPort',
            'sport': 'srcPort',
            'Destination Port': 'dstPort',
            'dst_port': 'dstPort',
            'destination_port': 'dstPort',
            'dport': 'dstPort',
            'Proto': 'protocol',
            'protocol': 'protocol',
            'Protocol': 'protocol',
            'Time': 'timestamp',
            'stime': 'timestamp',
            'Timestamp': 'timestamp',
            'pkt_len': 'pktLen',
            'packet_length': 'pktLen',
            'tot_len': 'totalLength',
            'flow_duration': 'flowDuration',
            'dur': 'flowDuration',
            'Attack': 'label',
            'attack': 'label',
            'attack_cat': 'label',
            'Class': 'label',
            'class': 'label',
        }
        logger.info(f"Original columns before standardization: {data.columns.tolist()}")
        newCols = {}
        for col in data.columns:
            stripped_col = col.strip()
            if stripped_col in columnMappings:
                newCols[col] = columnMappings[stripped_col]
            else:
                newCols[col] = stripped_col
        data = data.rename(columns=newCols)
        logger.info(f"Columns after stripping and initial standardization: {data.columns.tolist()}")
        
        # Standardize the 'label' column
        if labelColumn in data.columns and labelColumn != 'label':
             data = data.rename(columns={labelColumn: 'label'})

        if 'label' in data.columns:
            data['label'] = data['label'].astype(str).str.strip().str.replace(' ', '_')
        else:
            # Find a potential label column
            for col in data.columns:
                if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower():
                    data = data.rename(columns={col: 'label'})
                    break
        
        logger.info("Column names standardized")
        return data
        
    def _toCamelCase(self, text: str) -> str:
        """Convert string to camelCase"""
        # Replace common separators
        text = text.replace('-', '_').replace(' ', '_')
        
        # Split by underscore
        parts = text.split('_')
        
        # First part lowercase, rest capitalized
        if len(parts) > 1:
            return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])
        return text.lower()
        
    def preprocessCSV(
        self,
        data: pd.DataFrame,
        labelColumn: str = 'label',
        handleMissing: bool = True,
        removeOutliers: bool = False,
        encodeCategorical: bool = True,
        normalize: bool = False
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for CSV data
        
        Args:
            data: Input dataframe
            labelColumn: Name of the label column
            handleMissing: Whether to handle missing values
            removeOutliers: Whether to remove outliers
            encodeCategorical: Whether to encode categorical features
            normalize: Whether to normalize data
        """
        logger.info("Starting CSV preprocessing pipeline")
        
        # Standardize column names
        data = self.standardizeColumnNames(data, labelColumn)
        
        # Create flowId for grouping
        if all(col in data.columns for col in ['srcIP', 'srcPort', 'dstIP', 'dstPort', 'protocol']):
            data['flowId'] = data.apply(
                lambda row: f"{row['srcIP']}:{row['srcPort']}-{row['dstIP']}:{row['dstPort']}-{row['protocol']}",
                axis=1
            )
            logger.info("Created 'flowId' column for grouping")
        else:
            logger.warning("Could not create 'flowId' as one or more required columns are missing.")

        # Handle missing values
        if handleMissing:
            data = self.handleMissingValues(data, strategy='mean')
            
        # Remove outliers
        if removeOutliers:
            data = self.removeOutliers(data, method='iqr')
            
        # Encode categorical features
        if encodeCategorical:
            data = self.encodeCategoricalFeatures(data)
            
        # Normalize
        if normalize:
            data = self.normalizeData(data, method='standard')
            
        logger.info("CSV preprocessing complete")
        return data


class PCAPDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor for PCAP-based network traffic data"""
    
    def __init__(self):
        super().__init__()
        self.flowTimeout = 120  # seconds
        
    def pcapToFlows(self, pcapPath: str, outputPath: Optional[str] = None) -> pd.DataFrame:
        """
        Convert PCAP file to flow-based CSV
        
        Args:
            pcapPath: Path to PCAP file
            outputPath: Optional path to save CSV
        """
        logger.info(f"Converting PCAP to flows: {pcapPath}")
        
        try:
            # Try using scapy
            from scapy.all import rdpcap, IP, TCP, UDP
            
            packets = rdpcap(pcapPath)
            flows = {}
            
            for pkt in packets:
                if IP in pkt:
                    # Create flow key
                    if TCP in pkt:
                        flowKey = (
                            pkt[IP].src, pkt[TCP].sport,
                            pkt[IP].dst, pkt[TCP].dport,
                            'TCP'
                        )
                    elif UDP in pkt:
                        flowKey = (
                            pkt[IP].src, pkt[UDP].sport,
                            pkt[IP].dst, pkt[UDP].dport,
                            'UDP'
                        )
                    else:
                        continue
                        
                    # Initialize flow if new
                    if flowKey not in flows:
                        flows[flowKey] = {
                            'srcIP': flowKey[0],
                            'srcPort': flowKey[1],
                            'dstIP': flowKey[2],
                            'dstPort': flowKey[3],
                            'protocol': flowKey[4],
                            'packets': [],
                            'timestamps': [],
                            'sizes': []
                        }
                        
                    # Add packet info
                    flows[flowKey]['packets'].append(pkt)
                    flows[flowKey]['timestamps'].append(float(pkt.time))
                    flows[flowKey]['sizes'].append(len(pkt))
                    
            # Convert flows to dataframe
            flowData = []
            for flowKey, flowInfo in flows.items():
                flowData.append({
                    'flowId': f"{flowInfo['srcIP']}:{flowInfo['srcPort']}-{flowInfo['dstIP']}:{flowInfo['dstPort']}-{flowInfo['protocol']}",
                    'srcIP': flowInfo['srcIP'],
                    'srcPort': flowInfo['srcPort'],
                    'dstIP': flowInfo['dstIP'],
                    'dstPort': flowInfo['dstPort'],
                    'protocol': flowInfo['protocol'],
                    'pktCount': len(flowInfo['packets']),
                    'totalBytes': sum(flowInfo['sizes']),
                    'flowDuration': max(flowInfo['timestamps']) - min(flowInfo['timestamps']),
                    'pktSizeMean': np.mean(flowInfo['sizes']),
                    'pktSizeStd': np.std(flowInfo['sizes']),
                    'iatMean': np.mean(np.diff(flowInfo['timestamps'])) if len(flowInfo['timestamps']) > 1 else 0,
                    'iatStd': np.std(np.diff(flowInfo['timestamps'])) if len(flowInfo['timestamps']) > 1 else 0
                })
                
            df = pd.DataFrame(flowData)
            
            if outputPath:
                df.to_csv(outputPath, index=False)
                logger.info(f"Flows saved to: {outputPath}")
                
            logger.info(f"Extracted {len(df)} flows from PCAP")
            return df
            
        except ImportError:
            logger.error("Scapy not installed. Install with: pip install scapy")
            raise
        except Exception as e:
            logger.error(f"Error processing PCAP: {str(e)}")
            raise
            
    def extractEncryptedFeatures(self, pcapPath: str) -> pd.DataFrame:
        """
        Extract features specific to encrypted traffic (TLS/SSL)
        
        Args:
            pcapPath: Path to PCAP file
        """
        logger.info(f"Extracting encrypted traffic features from: {pcapPath}")
        
        try:
            from scapy.all import rdpcap, IP, TCP
            from scapy.layers.ssl_tls import TLS
            
            packets = rdpcap(pcapPath)
            tlsFlows = {}
            
            for pkt in packets:
                if IP in pkt and TCP in pkt:
                    # Check for TLS
                    if pkt.haslayer(TLS) or pkt[TCP].dport == 443 or pkt[TCP].sport == 443:
                        flowKey = (
                            pkt[IP].src, pkt[TCP].sport,
                            pkt[IP].dst, pkt[TCP].dport
                        )
                        
                        if flowKey not in tlsFlows:
                            tlsFlows[flowKey] = {
                                'srcIP': flowKey[0],
                                'srcPort': flowKey[1],
                                'dstIP': flowKey[2],
                                'dstPort': flowKey[3],
                                'tlsPackets': 0,
                                'payloadSizes': [],
                                'timestamps': []
                            }
                            
                        tlsFlows[flowKey]['tlsPackets'] += 1
                        tlsFlows[flowKey]['payloadSizes'].append(len(pkt))
                        tlsFlows[flowKey]['timestamps'].append(float(pkt.time))
                        
            # Convert to dataframe
            tlsData = []
            for flowKey, flowInfo in tlsFlows.items():
                tlsData.append({
                    'srcIP': flowInfo['srcIP'],
                    'srcPort': flowInfo['srcPort'],
                    'dstIP': flowInfo['dstIP'],
                    'dstPort': flowInfo['dstPort'],
                    'tlsPacketCount': flowInfo['tlsPackets'],
                    'payloadSizeMean': np.mean(flowInfo['payloadSizes']),
                    'payloadSizeStd': np.std(flowInfo['payloadSizes']),
                    'payloadSizeEntropy': self._calculateEntropy(flowInfo['payloadSizes'])
                })
                
            df = pd.DataFrame(tlsData)
            logger.info(f"Extracted encrypted features from {len(df)} TLS flows")
            return df
            
        except ImportError:
            logger.error("Scapy with TLS support not installed")
            raise
        except Exception as e:
            logger.error(f"Error extracting encrypted features: {str(e)}")
            raise
            
    def _calculateEntropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy"""
        if not values:
            return 0.0
        valueCounts = pd.Series(values).value_counts(normalize=True)
        entropy = -np.sum(valueCounts * np.log2(valueCounts + 1e-10))
        return entropy
        
    def preprocessPCAP(
        self,
        pcapPath: str,
        extractEncrypted: bool = True,
        outputPath: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for PCAP data
        
        Args:
            pcapPath: Path to PCAP file
            extractEncrypted: Whether to extract encrypted traffic features
            outputPath: Optional path to save processed data
        """
        logger.info("Starting PCAP preprocessing pipeline")
        
        # Convert to flows
        flowData = self.pcapToFlows(pcapPath)
        
        # Extract encrypted features if requested
        if extractEncrypted:
            try:
                encryptedFeatures = self.extractEncryptedFeatures(pcapPath)
                # Merge with flow data
                flowData = pd.merge(
                    flowData,
                    encryptedFeatures,
                    on=['srcIP', 'srcPort', 'dstIP', 'dstPort'],
                    how='left'
                )
            except Exception as e:
                logger.warning(f"Could not extract encrypted features: {str(e)}")
                
        # Handle missing values
        flowData = self.handleMissingValues(flowData, strategy='zero')
        
        if outputPath:
            flowData.to_csv(outputPath, index=False)
            logger.info(f"Processed data saved to: {outputPath}")
            
        logger.info("PCAP preprocessing complete")
        return flowData


class DataSplitter:
    """Split data into training, validation, and test sets"""
    
    def __init__(
        self,
        trainSize: float = 0.7,
        valSize: float = 0.15,
        testSize: float = 0.15,
        randomState: int = 42,
        stratify: bool = True
    ):
        """
        Args:
            trainSize: Proportion of training data
            valSize: Proportion of validation data
            testSize: Proportion of test data
            randomState: Random seed
            stratify: Whether to stratify split by labels
        """
        assert abs(trainSize + valSize + testSize - 1.0) < 1e-6, "Sizes must sum to 1.0"
        
        self.trainSize = trainSize
        self.valSize = valSize
        self.testSize = testSize
        self.randomState = randomState
        self.stratify = stratify
        
    def split(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        
        Returns:
            xTrain, xVal, xTest, yTrain, yVal, yTest
        """
        logger.info(f"Splitting data: train={self.trainSize}, val={self.valSize}, test={self.testSize}")
        
        stratifyParam = labels if self.stratify else None
        
        # First split: separate test set
        xTemp, xTest, yTemp, yTest = train_test_split(
            features,
            labels,
            test_size=self.testSize,
            random_state=self.randomState,
            stratify=stratifyParam
        )
        
        # Second split: separate train and validation
        valSizeAdjusted = self.valSize / (self.trainSize + self.valSize)
        stratifyParam = yTemp if self.stratify else None
        
        xTrain, xVal, yTrain, yVal = train_test_split(
            xTemp,
            yTemp,
            test_size=valSizeAdjusted,
            random_state=self.randomState,
            stratify=stratifyParam
        )
        
        logger.info(f"Split complete: train={len(xTrain)}, val={len(xVal)}, test={len(xTest)}")
        
        return xTrain, xVal, xTest, yTrain, yVal, yTest


class DataPreprocessingPipeline:
    """Complete data preprocessing pipeline for both CSV and PCAP"""
    
    def __init__(self, dataType: str = 'csv'):
        """
        Args:
            dataType: Type of data ('csv' or 'pcap')
        """
        self.dataType = dataType
        
        if dataType == 'csv':
            self.preprocessor = CSVDataPreprocessor()
        elif dataType == 'pcap':
            self.preprocessor = PCAPDataPreprocessor()
        else:
            raise ValueError(f"Unknown data type: {dataType}")
            
        self.splitter = DataSplitter()
        
    def process(
        self,
        inputPath: str,
        labelColumn: str = 'label',
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process data from file
        
        Args:
            inputPath: Path to input file
            labelColumn: Name of label column
            **kwargs: Additional preprocessing arguments
            
        Returns:
            features, labels
        """
        logger.info(f"Processing {self.dataType} data from: {inputPath}")
        
        if self.dataType == 'csv':
            processed_chunks = []
            dtype = {labelColumn: 'str'}
            chunk_iterator = pd.read_csv(inputPath, chunksize=100000, dtype=dtype)
            for chunk in chunk_iterator:
                processed_chunk = self.preprocessor.preprocessCSV(chunk, labelColumn=labelColumn, **kwargs)
                processed_chunks.append(processed_chunk)
            data = pd.concat(processed_chunks, ignore_index=True)

        elif self.dataType == 'pcap':
            data = self.preprocessor.preprocessPCAP(inputPath, **kwargs)
            
        # Separate features and labels
        if labelColumn in data.columns:
            labels = data[labelColumn]
            features = data.drop(columns=[labelColumn])
        else:
            logger.warning(f"Label column '{labelColumn}' not found. Returning all data as features.")
            features = data
            labels = pd.Series([0] * len(data))
            
        logger.info(f"Processing complete: {len(features)} samples, {len(features.columns)} features")
        
        return features, labels
        
    def processAndSplit(
        self,
        inputPath: str,
        labelColumn: str = 'label',
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Process data and split into train/val/test
        
        Returns:
            xTrain, xVal, xTest, yTrain, yVal, yTest
        """
        features, labels = self.process(inputPath, labelColumn, **kwargs)
        return self.splitter.split(features, labels)
