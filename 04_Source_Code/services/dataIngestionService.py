"""
Data Ingestion Service

Microservice for processing different data formats:
- CSV files (flow-based features)
- PCAP files (packet captures)

Handles data loading, preprocessing, and feature extraction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
from datetime import datetime
import importlib.util
import sys
import os

try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Dynamically load BaseService
try:
    spec_base_service = importlib.util.spec_from_file_location(
        "baseService",
        "e:/IDS for Encrypted Traffic with ML (Encrypted IDS)/04_Source_Code/services/baseService.py"
    )
    baseService_module = importlib.util.module_from_spec(spec_base_service)
    spec_base_service.loader.exec_module(baseService_module)
    BaseService = baseService_module.BaseService
except Exception as e:
    print(f"Error loading BaseService in dataIngestionService: {e}")
    sys.exit(1)


class CSVDataProcessor:
    """Processor for CSV format data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV processor.
        
        Args:
            config: Processor configuration
        """
        self.config = config
        self.featureColumns = config.get('featureColumns', None)
        self.labelColumn = config.get('labelColumn', 'label')
        self.normalize = config.get('normalize', True)
        
    def loadData(self, filePath: str) -> pd.DataFrame:
        """
        Load CSV data.
        
        Args:
            filePath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(filePath)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {str(e)}")
    
    def extractFeatures(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract features from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing features and labels
        """
        # Select feature columns
        if self.featureColumns:
            featureCols = [col for col in self.featureColumns if col in df.columns]
        else:
            # Use all numeric columns except label
            featureCols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.labelColumn in featureCols:
                featureCols.remove(self.labelColumn)
        
        features = df[featureCols].values
        
        # Extract labels if available
        labels = None
        if self.labelColumn in df.columns:
            labels = df[self.labelColumn].values
        
        # Normalize features
        if self.normalize:
            features = self._normalizeFeatures(features)
        
        return {
            'features': features,
            'labels': labels,
            'featureNames': featureCols,
            'sampleCount': len(df)
        }
    
    def _normalizeFeatures(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range.
        
        Args:
            features: Input features
            
        Returns:
            Normalized features
        """
        # Handle NaN and Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Min-max normalization
        minVals = np.min(features, axis=0, keepdims=True)
        maxVals = np.max(features, axis=0, keepdims=True)
        
        # Avoid division by zero
        denominator = maxVals - minVals
        denominator[denominator == 0] = 1.0
        
        normalized = (features - minVals) / denominator
        
        return normalized


class PCAPDataProcessor:
    """Processor for PCAP format data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PCAP processor.
        
        Args:
            config: Processor configuration
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for PCAP processing. Install with: pip install scapy")
        
        self.config = config
        self.maxPackets = config.get('maxPackets', 100)
        self.maxPacketSize = config.get('maxPacketSize', 1500)
        
    def loadData(self, filePath: str) -> List:
        """
        Load PCAP data.
        
        Args:
            filePath: Path to PCAP file
            
        Returns:
            List of packets
        """
        try:
            packets = scapy.rdpcap(filePath)
            return packets
        except Exception as e:
            raise ValueError(f"Failed to load PCAP file: {str(e)}")
    
    def extractFeatures(self, packets: List) -> Dict[str, np.ndarray]:
        """
        Extract features from PCAP packets.
        
        Args:
            packets: List of Scapy packets
            
        Returns:
            Dictionary containing extracted features
        """
        features = []
        metadata = []
        
        for i, packet in enumerate(packets[:self.maxPackets]):
            # Extract packet features
            packetFeatures = self._extractPacketFeatures(packet)
            features.append(packetFeatures)
            
            # Extract metadata
            packetMeta = self._extractPacketMetadata(packet)
            metadata.append(packetMeta)
        
        # Convert to numpy array
        featuresArray = np.array(features)
        
        return {
            'features': featuresArray,
            'metadata': metadata,
            'packetCount': len(features),
            'labels': None
        }
    
    def _extractPacketFeatures(self, packet) -> np.ndarray:
        """
        Extract numerical features from a single packet.
        
        Args:
            packet: Scapy packet
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.maxPacketSize)
        
        try:
            # Get raw packet bytes
            rawBytes = bytes(packet)
            
            # Convert to feature vector (pad or truncate)
            length = min(len(rawBytes), self.maxPacketSize)
            features[:length] = np.frombuffer(rawBytes[:length], dtype=np.uint8)
            
            # Normalize to [0, 1]
            features = features / 255.0
            
        except Exception:
            pass  # Return zero vector if extraction fails
        
        return features
    
    def _extractPacketMetadata(self, packet) -> Dict[str, Any]:
        """
        Extract metadata from packet.
        
        Args:
            packet: Scapy packet
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'timestamp': float(packet.time) if hasattr(packet, 'time') else None,
            'length': len(packet),
            'protocol': None,
            'srcIP': None,
            'dstIP': None,
            'srcPort': None,
            'dstPort': None
        }
        
        try:
            # Extract IP layer info
            if packet.haslayer(scapy.IP):
                ip = packet[scapy.IP]
                metadata['srcIP'] = ip.src
                metadata['dstIP'] = ip.dst
                metadata['protocol'] = ip.proto
            
            # Extract TCP layer info
            if packet.haslayer(scapy.TCP):
                tcp = packet[scapy.TCP]
                metadata['srcPort'] = tcp.sport
                metadata['dstPort'] = tcp.dport
                metadata['protocol'] = 'TCP'
            
            # Extract UDP layer info
            elif packet.haslayer(scapy.UDP):
                udp = packet[scapy.UDP]
                metadata['srcPort'] = udp.sport
                metadata['dstPort'] = udp.dport
                metadata['protocol'] = 'UDP'
        
        except Exception:
            pass  # Keep default None values if extraction fails
        
        return metadata


class DataIngestionService(BaseService):
    """Data Ingestion Service for multiple data formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Data Ingestion Service.
        
        Args:
            config: Service configuration
        """
        super().__init__("DataIngestionService", config)
        
        self.csvProcessor = None
        self.pcapProcessor = None
        self.outputDir = Path(self.getConfig('outputDir', './outputs/ingestion'))
        self.outputDir.mkdir(parents=True, exist_ok=True)
        
    def _onStart(self) -> None:
        """Initialize data processors on service start."""
        # Initialize CSV processor
        csvConfig = self.getConfig('csvConfig', {})
        self.csvProcessor = CSVDataProcessor(csvConfig)
        
        # Initialize PCAP processor
        pcapConfig = self.getConfig('pcapConfig', {})
        try:
            self.pcapProcessor = PCAPDataProcessor(pcapConfig)
            self.logger.info("PCAP processor initialized")
        except ImportError as e:
            self.logger.warning(f"PCAP processor not available: {str(e)}")
            self.pcapProcessor = None
        
        self.logger.info("Data Ingestion Service initialized")
    
    def _onStop(self) -> None:
        """Cleanup on service stop."""
        self.csvProcessor = None
        self.pcapProcessor = None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data file through appropriate processor.
        
        Args:
            data: Dictionary containing:
                - filePath: Path to data file
                - fileType: 'csv' or 'pcap'
                
        Returns:
            Dictionary containing extracted features
        """
        if not self.isRunning:
            raise RuntimeError("Data Ingestion Service is not running")
        
        filePath = data.get('filePath')
        fileType = data.get('fileType')
        
        if not filePath:
            raise ValueError("No file path provided")
        
        if not Path(filePath).exists():
            raise FileNotFoundError(f"File not found: {filePath}")
        
        # Auto-detect file type if not specified
        if not fileType:
            fileType = self._detectFileType(filePath)
        
        self.logger.info(f"Processing {fileType} file: {filePath}")
        
        try:
            if fileType == 'csv':
                result = self._processCSV(filePath)
            elif fileType == 'pcap':
                result = self._processPCAP(filePath)
            else:
                raise ValueError(f"Unsupported file type: {fileType}")
            
            # Save processed data
            self._saveProcessedData(result, filePath, fileType)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            raise
    
    def _processCSV(self, filePath: str) -> Dict[str, Any]:
        """
        Process CSV file.
        
        Args:
            filePath: Path to CSV file
            
        Returns:
            Processed data dictionary
        """
        # Load data
        df = self.csvProcessor.loadData(filePath)
        
        # Extract features
        extracted = self.csvProcessor.extractFeatures(df)
        
        return {
            'fileType': 'csv',
            'filePath': filePath,
            'features': extracted['features'],
            'labels': extracted['labels'],
            'featureNames': extracted['featureNames'],
            'sampleCount': extracted['sampleCount'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _processPCAP(self, filePath: str) -> Dict[str, Any]:
        """
        Process PCAP file.
        
        Args:
            filePath: Path to PCAP file
            
        Returns:
            Processed data dictionary
        """
        if self.pcapProcessor is None:
            raise RuntimeError("PCAP processor is not available")
        
        # Load packets
        packets = self.pcapProcessor.loadData(filePath)
        
        # Extract features
        extracted = self.pcapProcessor.extractFeatures(packets)
        
        return {
            'fileType': 'pcap',
            'filePath': filePath,
            'features': extracted['features'],
            'metadata': extracted['metadata'],
            'packetCount': extracted['packetCount'],
            'labels': extracted['labels'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _detectFileType(self, filePath: str) -> str:
        """
        Auto-detect file type from extension.
        
        Args:
            filePath: Path to file
            
        Returns:
            Detected file type
        """
        extension = Path(filePath).suffix.lower()
        
        if extension == '.csv':
            return 'csv'
        elif extension in ['.pcap', '.pcapng', '.cap']:
            return 'pcap'
        else:
            raise ValueError(f"Cannot detect file type from extension: {extension}")
    
    def _saveProcessedData(
        self, 
        result: Dict[str, Any], 
        filePath: str,
        fileType: str
    ) -> None:
        """
        Save processed data to output directory.
        
        Args:
            result: Processed data
            filePath: Original file path
            fileType: File type
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fileName = Path(filePath).stem
            
            # Save metadata
            metadataFile = self.outputDir / f"{fileName}_{fileType}_{timestamp}_metadata.json"
            metadata = {
                k: v for k, v in result.items() 
                if k not in ['features', 'labels']
            }
            
            with open(metadataFile, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save features as numpy array
            featuresFile = self.outputDir / f"{fileName}_{fileType}_{timestamp}_features.npy"
            np.save(featuresFile, result['features'])
            
            # Save labels if available
            if result.get('labels') is not None:
                labelsFile = self.outputDir / f"{fileName}_{fileType}_{timestamp}_labels.npy"
                np.save(labelsFile, result['labels'])
            
            self.logger.info(f"Saved processed data to {self.outputDir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save processed data: {str(e)}")
